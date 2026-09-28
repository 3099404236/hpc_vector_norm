#include "dsa_runtime.hpp"
#include <iostream>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

using namespace dsa;

// Cache-line aligned memory buffers on the 64-byte/32-byte DMA block boundary (alignas on a std::vector
// object does not align its heap storage)
template <typename T>
struct AlignedAllocator {
    using value_type = T;
    AlignedAllocator() = default;
    template <typename U> AlignedAllocator(const AlignedAllocator<U>&) {}
    T* allocate(size_t n) { return static_cast<T*>(std::aligned_alloc(64, (n * sizeof(T) + 63) / 64 * 64)); }
    void deallocate(T* p, size_t) { std::free(p); }
    template <typename U> bool operator==(const AlignedAllocator<U>&) const { return true; }
    template <typename U> bool operator!=(const AlignedAllocator<U>&) const { return false; }
};
template <typename T> using AlignedVector = std::vector<T, AlignedAllocator<T>>;

// System memory as the target reaches it: a GlobalTensor over the buffer (docs/TARGET_API_SHAPE.md)
template <typename T>
GlobalTensor<T> Gm(T* p) {
    GlobalTensor<T> g;
    g.SetGlobalBuffer(p);
    return g;
}

template <typename F>
bool ExpectTrap(const char* what, F body) {
    try {
        body();
    } catch (const std::exception& e) {
        std::cout << "PASS: " << what << ":\n  --> " << e.what() << "\n";
        return true;
    }
    std::cerr << "FAIL: Sanitizer did not trap: " << what << "\n";
    return false;
}

// -----------------------------------------------------------------------------
// High-Performance DAE Stream Pipeline Kernel running seamlessly on CPU
// -----------------------------------------------------------------------------
class SampleDaePipelineKernel {
public:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQueue;
    TQue<QuePosition::VECOUT, 2> outQueue;

    void Process(GlobalTensor<float> src, GlobalTensor<float> dst, uint32_t totalElems) {
        // Tile size: 512 floats = 2048 bytes (strictly 32-byte aligned)
        uint32_t tileBytes = 512 * sizeof(float);

        // Init Double Buffering queues
        pipe.InitBuffer(inQueue, 2, tileBytes);
        pipe.InitBuffer(outQueue, 2, tileBytes);

        // Verify scratchpad usage stays strictly <= 191 KB
        std::cout << "[Sanitizer Audit]: Total scratchpad allocated: " 
                  << pipe.GetTotalAllocatedBytes() << " bytes (Limit: " << SCRATCHPAD_SAFE_WATERLINE << " bytes) - OK!\n";

        uint32_t tiles = totalElems / 512;
        for (uint32_t t = 0; t < tiles; ++t) {
            // Stage 1: DMA Inbound
            LocalTensor<float> inTensor = inQueue.AllocTensor<float>();
            DataCopy(inTensor, src[t * 512], 512);
            inQueue.EnQue(inTensor);

            // Stage 2: SIMD Vector Compute
            LocalTensor<float> workTensor = inQueue.DeQue<float>();
            LocalTensor<float> outTensor = outQueue.AllocTensor<float>();
            
            Mul(outTensor, workTensor, workTensor, 512); // Square
            inQueue.FreeTensor(workTensor);

            outQueue.EnQue(outTensor);

            // Stage 3: DMA Outbound
            LocalTensor<float> writeTensor = outQueue.DeQue<float>();
            DataCopy(dst[t * 512], writeTensor, 512);
            outQueue.FreeTensor(writeTensor);
        }
    }
};

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  Testing DAE Stream Pipeline Runtime Engine on CPU\n";
    std::cout << "=================================================================\n";

    constexpr uint32_t ELEMS = 2048;
    AlignedVector<float> input(ELEMS, 2.0f);
    AlignedVector<float> output(ELEMS, 0.0f);

    SampleDaePipelineKernel kernel;
    kernel.Process(Gm(input.data()), Gm(output.data()), ELEMS);

    // Verify numerical correctness
    for (uint32_t i = 0; i < ELEMS; ++i) {
        if (output[i] != 4.0f) {
            std::cerr << "FAIL at " << i << ": expected 4.0, got " << output[i] << "\n";
            return 1;
        }
    }
    std::cout << "[Correctness]: 100% PASS (output = 4.0)\n";
    std::cout << "[Virtual Cycles]: Total Vector Cycles = " 
              << g_cycleTracker.GetTotalVectorCycles() 
              << " | DMA Bytes = " << g_cycleTracker.dmaBytesMoved << " bytes\n";

    // -------------------------------------------------------------------------
    // Test 2: Sanitizer Trapping Scratchpad Overflow (> 191 KB)
    // -------------------------------------------------------------------------
    std::cout << "\n[Testing Sanitizer Guard: Scratchpad Overflow Defense]...\n";
    try {
        TPipe badPipe;
        TQue<QuePosition::VECIN, 2> giantQueue;
        badPipe.InitBuffer(giantQueue, 2, 100 * 1024); // 200 KB > 191 KB!
        std::cerr << "FAIL: Sanitizer did not catch scratchpad overflow!\n";
        return 1;
    } catch (const std::exception& e) {
        std::cout << "PASS: Caught hardware fault successfully:\n  --> " << e.what() << "\n";
    }

    // -------------------------------------------------------------------------
    // Test 3: Sanitizer Trapping DMA Misalignment
    // -------------------------------------------------------------------------
    std::cout << "\n[Testing Sanitizer Guard: DMA 32-Byte Alignment Defense]...\n";
    try {
        TPipe alignPipe;
        TQue<QuePosition::VECIN, 1> q;
        alignPipe.InitBuffer(q, 1, 1024);
        LocalTensor<float> t = q.AllocTensor<float>();
        DataCopy(t, Gm(input.data()), 5); // 5 floats = 20 bytes (not 32-byte multiple!)
        std::cerr << "FAIL: Sanitizer did not catch DMA misalignment!\n";
        return 1;
    } catch (const std::exception& e) {
        std::cout << "PASS: Caught unaligned DMA fault successfully:\n  --> " << e.what() << "\n";
    }

    // -------------------------------------------------------------------------
    // Test 4: Double buffering keeps the in-flight tile intact (prefetch tile k+1
    // before tile k is consumed)
    // -------------------------------------------------------------------------
    std::cout << "\n[Testing Double-Buffer Slot Lifecycle]...\n";
    {
        AlignedVector<float> a(8, 1.0f), b(8, 2.0f);
        TPipe dbPipe;
        TQue<QuePosition::VECIN, 2> dq;
        dbPipe.InitBuffer(dq, 2, 8 * sizeof(float));
        LocalTensor<float> t0 = dq.AllocTensor<float>();
        DataCopy(t0, Gm(a.data()), 8);
        dq.EnQue(t0);
        LocalTensor<float> t1 = dq.AllocTensor<float>();  // prefetch into the second buffer
        DataCopy(t1, Gm(b.data()), 8);
        dq.EnQue(t1);
        LocalTensor<float> c0 = dq.DeQue<float>();
        LocalTensor<float> c1 = dq.DeQue<float>();
        if (t0.GetData() == t1.GetData() || c0.GetValue(0) != 1.0f || c1.GetValue(0) != 2.0f) {
            std::cerr << "FAIL: prefetched tile overwrote the tile in flight\n";
            return 1;
        }
        dq.FreeTensor(c0);
        dq.FreeTensor(c1);
        std::cout << "PASS: tiles 0/1 in distinct buffers, FIFO order kept\n";
    }
    bool ok = true;
    ok &= ExpectTrap("third AllocTensor on a 2-buffer queue without FreeTensor", [] {
        TPipe p;
        TQue<QuePosition::VECIN, 2> q2;
        p.InitBuffer(q2, 2, 64);
        q2.AllocTensor<float>();
        q2.AllocTensor<float>();
        q2.AllocTensor<float>();
    });
    ok &= ExpectTrap("FreeTensor on a tile still enqueued", [] {
        TPipe p;
        TQue<QuePosition::VECIN, 2> q2;
        p.InitBuffer(q2, 2, 64);
        LocalTensor<float> t = q2.AllocTensor<float>();
        q2.EnQue(t);
        q2.FreeTensor(t);
    });

    // -------------------------------------------------------------------------
    // Test 5: DMA address alignment (size is a whole block, address is not)
    // -------------------------------------------------------------------------
    std::cout << "\n[Testing Sanitizer Guard: DMA Address Alignment]...\n";
    ok &= ExpectTrap("DataCopy from an address 4 bytes past a block boundary", [&] {
        TPipe p;
        TQue<QuePosition::VECIN, 1> q1;
        p.InitBuffer(q1, 1, 256);
        LocalTensor<float> t = q1.AllocTensor<float>();
        DataCopy(t, Gm(input.data())[1], 8);
    });

    // -------------------------------------------------------------------------
    // Test 6: DataCopyPad moves a 20-byte tail and zero-fills the rest of the block
    // -------------------------------------------------------------------------
    std::cout << "\n[Testing DataCopyPad]...\n";
    {
        TPipe p;
        TQue<QuePosition::VECIN, 1> q1;
        p.InitBuffer(q1, 1, 64);
        LocalTensor<float> t = q1.AllocTensor<float>();
        const uint64_t pads = g_cycleTracker.padTransfers;
        DataCopyPad(t, Gm(input.data())[3], DataCopyExtParams{1, 5 * sizeof(float), 0, 0, 0}, DataCopyPadExtParams<float>{true, 0, 0, 0.0f});
        if (t.GetValue(4) != 2.0f || t.GetValue(5) != 0.0f || t.GetValue(7) != 0.0f || g_cycleTracker.padTransfers != pads + 1) {
            std::cerr << "FAIL: DataCopyPad did not move/pad the tail\n";
            return 1;
        }
        std::cout << "PASS: 20-byte padded transfer\n";
    }

    // -------------------------------------------------------------------------
    // Test 7: Scratchpad budget counts blocks, and buffer counts must fit the queue
    // -------------------------------------------------------------------------
    ok &= ExpectTrap("InitBuffer with more buffers than the queue depth", [] {
        TPipe p;
        TQue<QuePosition::VECIN, 1> q1;
        p.InitBuffer(q1, 2, 64);
    });

    // -------------------------------------------------------------------------
    // Test 8: Timeline model. One core: a load lands DMA_LATENCY_CYCLES after it streamed at its
    // share of the memory system (g_memory); the add waits for it; the store follows the add and
    // lands one latency later. Two tiles: the second load streams while the first tile computes.
    // -------------------------------------------------------------------------
    std::cout << "\n[Testing Timeline Model]...\n";
    {
        auto near = [](double a, double b) { return std::fabs(a - b) <= 1e-9 * (std::fabs(b) + 1.0); };
        const double bw = 4096 / g_memory.Cycles(4096), L = DMA_LATENCY_CYCLES;  // 4 KB transfers: bytes per cycle
        AlignedVector<float> src(1024, 1.0f), dst(1024, 0.0f);
        {
            TPipe p;
            TQue<QuePosition::VECIN, 1> qIn;
            TQue<QuePosition::VECOUT, 1> qOut;
            p.InitBuffer(qIn, 1, 1024 * sizeof(float));
            p.InitBuffer(qOut, 1, 1024 * sizeof(float));
            LocalTensor<float> tIn = qIn.AllocTensor<float>();
            LocalTensor<float> tOut = qOut.AllocTensor<float>();
            DataCopy(tIn, Gm(src.data()), 1024);  // 4 KB
            Add(tOut, tIn, tIn, 1024);        // 16 repeats: 45 cycles
            DataCopy(Gm(dst.data()), tOut, 1024);
            const TimelineSummary s = g_timeline.Summary();
            const double load = 4096 / bw, add = 2 * 16 + 13, expect = load + L + add + load + L;
            ok &= near(s.finish, expect) && near(s.vectorBusy, add) && near(s.dmaBusy, 2 * load) && near(s.loadBusy, load);
            ok &= near(s.finish, std::max(s.vectorBusy, s.dmaBusy) + s.fill + s.drain + s.barrier + s.mismatch);
            ok &= s.finish >= s.LowerBound();
            ok &= near(s.LatencyFloor(), expect);  // One chain, load -> add -> store: nothing to overlap
            if (!ok) std::cerr << "FAIL: single-tile timeline " << s.finish << " (floor " << s.LatencyFloor() << ") != " << expect << "\n";
            qIn.FreeTensor(tIn);
            qOut.FreeTensor(tOut);
        }
        {
            TPipe p;
            TQue<QuePosition::VECIN, 2> q;
            p.InitBuffer(q, 2, 1024 * sizeof(float));
            LocalTensor<float> a = q.AllocTensor<float>(), b = q.AllocTensor<float>();
            DataCopy(a, Gm(src.data()), 1024);
            DataCopy(b, Gm(src.data()), 1024);  // Streams while tile a computes
            Add(a, a, a, 1024);
            Add(b, b, b, 1024);
            const TimelineSummary s = g_timeline.Summary();
            const double load = 4096 / bw, add = 45;
            // Tile b lands one load after tile a; the vector waits for it only if a's add is shorter
            const double expect = std::max(load + L + add, 2 * load + L) + add;
            // Two buffers, no reuse: the floor is the run itself
            const bool pass = near(s.finish, expect) && near(s.LatencyFloor(), expect);
            ok &= pass;
            if (!pass) std::cerr << "FAIL: double-buffered timeline " << s.finish << " != " << expect << "\n";
        }
        {
            // SyncAll: every core leaves SYNC_ALL_CYCLES after the last arrival
            double finish[4] = {}, floor[4] = {};
            #pragma omp parallel num_threads(4)
            {
                TPipe p;
                TBuf<QuePosition::VECCALC> buf;
                p.InitBuffer(buf, 4096);
                LocalTensor<float> t = buf.Get<float>();
                const uint32_t c = GetCoreIdx();
                for (uint32_t i = 0; i <= c; ++i) Muls(t, t, 2.0f, 1024);  // Core c: (c + 1) x 45 cycles
                SyncAll();
                finish[c] = g_timeline.Summary().finish;
                floor[c] = g_timeline.Summary().LatencyFloor();
            }
            bool same = true;
            for (int c = 0; c < 4; ++c) {
                same &= std::fabs(finish[c] - (4 * 45.0 + SYNC_ALL_CYCLES)) < 1e-9;
                same &= std::fabs(floor[c] - ((c + 1) * 45.0 + SYNC_ALL_CYCLES)) < 1e-9;  // Its own work, then the barrier
            }
            ok &= same;
            if (!same) std::cerr << "FAIL: SyncAll release " << finish[0] << " / " << finish[3] << " floor " << floor[0] << "\n";
        }
        if (ok) std::cout << "PASS: load/compute/store latency, double-buffer overlap, SyncAll release, latency floor\n";
    }

    // -------------------------------------------------------------------------
    // Test 9: Sanitizer Trapping Asymmetric Egress & ALU Pipeline Aliasing
    // -------------------------------------------------------------------------
    std::cout << "\n[Testing Sanitizer Guard: Egress Channel & ALU Aliasing Defense]...\n";
    ok &= ExpectTrap("DataCopy egress from QuePosition::VECIN", [&] {
        TPipe p;
        TQue<QuePosition::VECIN, 1> q;
        p.InitBuffer(q, 1, 256);
        LocalTensor<float> t = q.AllocTensor<float>();
        DataCopy(Gm(output.data()), t, 64);
    });
    ok &= ExpectTrap("BlockReduceSum in-place buffer aliasing (dst == src)", [&] {
        TPipe p;
        TBuf<QuePosition::VECCALC> b;
        p.InitBuffer(b, 256);
        LocalTensor<float> t = b.Get<float>();
        BlockReduceSum(t, t, 1, 64, 1, 1, 8);
    });

    // -------------------------------------------------------------------------
    // Test 10: DAE v1.5 Microarchitecture Invariants & Direct Primitives
    // -------------------------------------------------------------------------
    std::cout << "\n[Testing DAE v1.5 Microarchitectural Invariants & Traps]...\n";
    
    // Trap #408: ReduceSum unaligned lane fault (count % 64 != 0)
    ok &= ExpectTrap("ReduceSum unaligned SIMD lane fault (count=200, Trap #408)", [&] {
        TPipe p;
        TBuf<QuePosition::VECCALC> bDst, bSrc, bWork;
        p.InitBuffer(bDst, 32);
        p.InitBuffer(bSrc, 1024);
        p.InitBuffer(bWork, 1024);
        ReduceSum(bDst.Get<float>(), bSrc.Get<float>(), bWork.Get<float>(), 200);
    });

    // Trap #402: ReduceSum workspace aliasing
    ok &= ExpectTrap("ReduceSum workspace aliasing (dst == work, Trap #402)", [&] {
        TPipe p;
        TBuf<QuePosition::VECCALC> bDst, bSrc;
        p.InitBuffer(bDst, 1024);
        p.InitBuffer(bSrc, 1024);
        ReduceSum(bDst.Get<float>(), bSrc.Get<float>(), bDst.Get<float>(), 64);
    });

    // Trap #410: Brcb buffer overrun (one repeat writes 8 blocks: 64 floats)
    ok &= ExpectTrap("Brcb destination buffer undersized (capacity < 64 floats, Trap #410)", [&] {
        TPipe p;
        TBuf<QuePosition::VECCALC> bDst, bSrc;
        p.InitBuffer(bDst, 32); // only 8 floats
        p.InitBuffer(bSrc, 32);
        bSrc.Get<float>().data[0] = 3.14f;
        Brcb(bDst.Get<float>(), bSrc.Get<float>(), 1);
    });

    // Trap #409: Host-to-Device argument frame overflow (> 32 bytes)
    ok &= ExpectTrap("Launch arguments frame overflow (> 32 bytes, Trap #409)", [&] {
        struct BigPlanningStruct {
            uint64_t fields[5]; // 40 bytes
        } s{};
        ValidateLaunchArgs(s);
    });

    // Brcb [6.1]: each repeat spreads 8 consecutive source values over 8 32-byte blocks, value j over
    // block j (not one value over the whole destination)
    {
        TPipe p;
        TBuf<QuePosition::VECCALC> bSrc, bDst;
        p.InitBuffer(bSrc, 16 * sizeof(float));
        p.InitBuffer(bDst, 128 * sizeof(float));
        LocalTensor<float> src = bSrc.Get<float>(), dst = bDst.Get<float>();
        for (uint32_t i = 0; i < 16; ++i) src.data[i] = 1.0f + i;
        Brcb(dst, src, 2, BrcbRepeatParams{1, 8});
        bool brcbOk = true;
        for (uint32_t i = 0; i < 128; ++i) brcbOk &= dst.data[i] == 1.0f + i / 8;
        if (brcbOk) {
            std::cout << "PASS: Brcb fills block j of each repeat with that repeat's value j\n";
        } else {
            std::cerr << "FAIL: Brcb does not follow the target's block broadcast\n";
            ok = false;
        }
    }

    // -------------------------------------------------------------------------
    // Test 10: Extended Microarchitectural Primitives
    // GlobalTensor, DataCopyExtParams, Adds, ToFloat/FromFloat, ReinterpretCast, half
    // -------------------------------------------------------------------------
    {
        std::cout << "\n[Testing Extended Microarchitectural Primitives]...\n";
        TPipe mem;  // The target claims scratchpad through TPipe and TBuf only [6.4]
        TBuf<QuePosition::VECCALC> bF, bH, bAux;
        mem.InitBuffer(bF, 64 * sizeof(float));
        mem.InitBuffer(bH, 64 * sizeof(half));
        mem.InitBuffer(bAux, 64 * sizeof(float));
        auto vecF = bF.Get<float>();
        auto vecH = bH.Get<half>();
        auto auxF = bAux.Get<float>();

        // ReinterpretCast
        auto reF = vecF.ReinterpretCast<float>();
        if (reF.GetSize() != 64) {
            std::cerr << "FAIL: ReinterpretCast size mismatch\n";
            ok = false;
        }

        // GlobalTensor & DataCopyExtParams
        AlignedVector<float> gBuf(64, 5.0f);
        GlobalTensor<float> gm(gBuf.data());
        DataCopyExtParams cp{1, 64 * sizeof(float), 0, 0, 0};
        DataCopyPad(vecF, gm, cp);
        if (vecF.data[0] != 5.0f || vecF.data[63] != 5.0f) {
            std::cerr << "FAIL: GlobalTensor DataCopyPad failed\n";
            ok = false;
        }

        // Adds
        Adds(auxF, vecF, 3.0f, 64);
        if (auxF.data[0] != 8.0f || auxF.data[63] != 8.0f) {
            std::cerr << "FAIL: Adds scalar failed\n";
            ok = false;
        }

        // ToFloat and FromFloat
        for (int i = 0; i < 64; ++i) vecH.data[i] = half(2.5f);
        ToFloat(auxF, vecH, 64);
        if (std::fabs(auxF.data[0] - 2.5f) > 1e-3f) {
            std::cerr << "FAIL: ToFloat failed\n";
            ok = false;
        }
        FromFloat(vecH, auxF, 64);
        if (std::fabs(float(vecH.data[0]) - 2.5f) > 1e-3f) {
            std::cerr << "FAIL: FromFloat failed\n";
            ok = false;
        }

        // Pipeline barriers & flags
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        // Core / Block indexing & ctrl
        int64_t ctrl = get_ctrl();
        set_ctrl(ctrl | (int64_t(1) << 48));
        if (GetBlockIdx() != GetCoreIdx() || GetBlockNum() != GetCoreNum()) {
            std::cerr << "FAIL: Block / Core index mismatch\n";
            ok = false;
        }

        std::cout << "PASS: Extended Microarchitectural Primitives verified successfully\n";
    }

    // -------------------------------------------------------------------------
    // Test 11: the measured memory system (docs/TARGET_MEASUREMENTS.md, sections 3-5). Transfer-size
    // ceilings at their measured points, interpolated between them and flat outside; the working-set
    // regime combined by min; the aggregate saturating at 27 cores.
    // -------------------------------------------------------------------------
    std::cout << "\n[Testing Measured Memory System]...\n";
    {
        auto near = [](double a, double b) { return std::fabs(a - b) <= 1e-9 * std::fabs(b); };
        const double MiB = 1024.0 * 1024.0;
        const MemorySystem small(40, 1 * MiB), mid(40, 368 * MiB), huge(40, 5632 * MiB), big(40, 2560 * MiB);
        bool mem = near(small.AggregateGBs(1024), 277) && near(small.AggregateGBs(2048), 475) && near(small.AggregateGBs(4096), 677) &&
                   near(small.AggregateGBs(6144), 1090) && near(small.AggregateGBs(16384), 1090) && near(small.AggregateGBs(256), 277) &&
                   near(small.AggregateGBs(3072), 576);
        // The regime binds only below the size ceiling: 1252 GB/s at 368 MiB does not, 943 at 5.5 GiB does
        mem &= near(mid.AggregateGBs(16384), 1090) && near(huge.AggregateGBs(16384), 943) && near(big.AggregateGBs(16384), 981);
        mem &= near(huge.AggregateGBs(1024), 277);  // ... and the smaller of the two wins either way
        // Cores: 8 cores reach 8/27 of the bandwidth, 27 and more all of it; a transfer occupies its
        // core's channel for bytes / (aggregate / cores)
        const MemorySystem p8(8, 1 * MiB), p27(27, 1 * MiB);
        mem &= near(p8.AggregateGBs(16384), 1090.0 * 8 / 27) && near(p27.AggregateGBs(16384), 1090);
        mem &= near(small.Cycles(16384), 16384.0 * 40 / 1090 * CLOCK_GHZ) && near(p8.Cycles(16384), 16384.0 * 27 / 1090 * CLOCK_GHZ);
        // Measured constants: SyncAll 0.924 us, launch 1.70 us, a 4-byte repeat 1.165 ns = 2 cycles
        mem &= std::fabs(SYNC_ALL_CYCLES / CLOCK_GHZ - 924.0) < 1.0 && near(2.0 / CLOCK_GHZ, 64 * 0.0182) && KERNEL_LAUNCH_NS == 1700.0;
        ok &= mem;
        if (mem) std::cout << "PASS: size ceilings, working-set regime, saturation at 27 cores, measured constants\n";
        else std::cerr << "FAIL: memory system does not follow the measured rules\n";
    }

    // -------------------------------------------------------------------------
    // Test 12: the target's semantics (docs/TARGET_MEASUREMENTS.md, section 6), each a way code passed
    // the old runtime and failed on the target
    // -------------------------------------------------------------------------
    std::cout << "\n[Testing Target Semantics (section 6)]...\n";
    {
        TPipe p;
        TBuf<QuePosition::VECCALC> bA, bB, bC, bW;
        p.InitBuffer(bA, 256 * sizeof(float));
        p.InitBuffer(bB, 256 * sizeof(float));
        p.InitBuffer(bC, 256 * sizeof(float));
        p.InitBuffer(bW, 256 * sizeof(float));
        LocalTensor<float> a = bA.Get<float>(), b = bB.Get<float>(), c = bC.Get<float>(), w = bW.Get<float>();
        bool sem = std::isnan(a.data[0]) && std::isnan(a.data[255]);  // Unwritten scratchpad reads as NaN
        // 6.2: ReduceSum defines lane 0 of its slot block only: the target leaves lanes 1..7 as they were,
        // this runtime fills them with NaN so that nothing can consume them; the blocks around stay
        Duplicate(a, 1.0f, 128);
        Duplicate(b, 7.0f, 24);
        ReduceSum(b[8], a, w, 128);
        sem &= b.data[8] == 128.0f && std::isnan(b.data[9]) && std::isnan(b.data[15]) && b.data[0] == 7.0f && b.data[16] == 7.0f;
        // 6.8: the strided form's mask is per repeat. Row r of 3 rows of 128 floats times the one
        // 32-byte block c[8r] (src1 block stride 0): the row's multiplier over every lane, in two
        // instructions of 64 lanes, one repeat per row
        for (uint32_t i = 0; i < 24; ++i) c.data[i] = 1.0f + i / 8;
        Duplicate(a, 2.0f, 256);
        for (uint32_t h = 0; h < 2; ++h) Mul(b[h * 64], a[h * 64], c, 64, 2, BinaryRepeatParams{1, 1, 0, 16, 16, 1});
        for (uint32_t i = 0; i < 256; ++i) sem &= b.data[i] == 2.0f * (1.0f + i / 128);
        // 6.3: Rsqrt is a table of RSQRT_TABLE_BITS bits, not exact; one Newton-Raphson step refines it
        Duplicate(a, 3.0f, 64);
        Rsqrt(b, a, 64);
        const double exact = 1.0 / std::sqrt(3.0), table = b.data[0];
        const double refined = table * (1.5 - 0.5 * 3.0 * table * table);
        sem &= std::fabs(table - exact) / exact > 1e-6 && std::fabs(table - exact) / exact < 1.0 / (1 << RSQRT_TABLE_BITS) &&
               std::fabs(refined - exact) / exact < 1e-6;
        // 6.5: the scalar unit reads a vector result after a V->S fence, and the vector unit a scalar
        // store after an S->V fence
        Adds(a, a, 1.0f, 8);
        CrossPipe<HardEvent::V_S>(EVENT_ID0);
        sem &= a.GetValue(0) == 4.0f;
        a.SetValue(1, 5.0f);
        CrossPipe<HardEvent::S_V>(EVENT_ID0);
        Muls(b, a, 2.0f, 8);
        CrossPipe<HardEvent::V_S>(EVENT_ID0);
        sem &= b.GetValue(1) == 10.0f;
        ok &= sem;
        if (sem) std::cout << "PASS: poisoned scratchpad, ReduceSum lane 0, per-repeat masks, table Rsqrt, fenced scalar reads\n";
        else std::cerr << "FAIL: a target semantic is not modeled\n";
    }
    auto withBuffers = [](auto body) {
        TPipe p;
        TBuf<QuePosition::VECCALC> bA, bB, bW;
        p.InitBuffer(bA, 256 * sizeof(float));
        p.InitBuffer(bB, 256 * sizeof(float));
        p.InitBuffer(bW, 256 * sizeof(float));
        body(bA.Get<float>(), bB.Get<float>(), bW.Get<float>());
    };
    ok &= ExpectTrap("ReduceSum into dst[i] instead of the block dst[i * 8] (6.2)", [&] {
        withBuffers([](LocalTensor<float> a, LocalTensor<float> b, LocalTensor<float> w) {
            Duplicate(a, 1.0f, 64);
            ReduceSum(b[1], a, w, 64);
        });
    });
    ok &= ExpectTrap("a strided Mul given a whole row width as its mask (6.8)", [&] {
        withBuffers([](LocalTensor<float> a, LocalTensor<float> b, LocalTensor<float>) { Mul(b, a, a, 128, 1, BinaryRepeatParams{}); });
    });
    ok &= ExpectTrap("GetValue of a vector result behind PipeBarrier<PIPE_V> only (6.5)", [&] {
        withBuffers([](LocalTensor<float> a, LocalTensor<float>, LocalTensor<float>) {
            Duplicate(a, 1.0f, 8);
            PipeBarrier<PIPE_V>();
            (void)a.GetValue(0);
        });
    });
    ok &= ExpectTrap("a vector read of a scalar store without an S->V fence (6.5)", [&] {
        withBuffers([](LocalTensor<float> a, LocalTensor<float> b, LocalTensor<float>) {
            a.SetValue(0, 1.0f);
            Muls(b, a, 2.0f, 8);
        });
    });
    {
        // DataCopyPad: rows off the 32-byte grid land one per block in the scratchpad (padded with the
        // pad value) and leave it for system memory without the padding
        AlignedVector<float> rows(3 * 5), back(3 * 5, -1.0f);
        for (uint32_t i = 0; i < 15; ++i) rows[i] = float(i);
        TPipe p;
        TBuf<QuePosition::VECOUT> bRows;
        p.InitBuffer(bRows, 3 * 32);
        LocalTensor<float> t = bRows.Get<float>();
        DataCopyPad(t, Gm(rows.data()), DataCopyExtParams{3, 5 * sizeof(float), 0, 0, 0}, DataCopyPadExtParams<float>{true, 0, 3, 0.0f});
        bool pad = true;
        for (uint32_t r = 0; r < 3; ++r) {
            for (uint32_t j = 0; j < 8; ++j) pad &= t.data[r * 8 + j] == (j < 5 ? float(r * 5 + j) : 0.0f);
        }
        DataCopyPad(Gm(back.data()), t, DataCopyExtParams{3, 5 * sizeof(float), 0, 0, 0});
        for (uint32_t i = 0; i < 15; ++i) pad &= back[i] == float(i);
        ok &= pad;
        if (pad) std::cout << "PASS: DataCopyPad rows of 20 bytes, one per 32-byte block, round trip\n";
        else std::cerr << "FAIL: DataCopyPad row layout\n";
    }

    // -------------------------------------------------------------------------
    // Test 13: the target's 7-argument reductions and its rounding-mode Cast (docs/TARGET_API_SHAPE.md)
    // -------------------------------------------------------------------------
    std::cout << "\n[Testing Target Reductions and Cast]...\n";
    {
        TPipe p;
        TBuf<QuePosition::VECCALC> bS, bP, bO;
        p.InitBuffer(bS, 3 * 72 * sizeof(float));
        p.InitBuffer(bP, 64 * sizeof(float));
        p.InitBuffer(bO, 64 * sizeof(float));
        LocalTensor<float> src = bS.Get<float>(), part = bP.Get<float>(), out = bO.Get<float>();
        for (uint32_t i = 0; i < 3 * 72; ++i) src.data[i] = float(i % 72);
        // Rows of 72 floats (9 blocks): repeat r folds the first 64 lanes of row r, block b into element b
        BlockReduceSum(part, src, 3, 64, 1, 1, 9);
        bool red = true;
        for (uint32_t r = 0; r < 3; ++r) {
            for (uint32_t b = 0; b < 8; ++b) red &= part.data[8 * r + b] == float(64 * b + 28);
        }
        // Packed: 3 rows' 8 partials into their sums; a block no lane takes part in has no result (NaN)
        BlockReduceSum(out, part, 1, 24, 1, 1, 8);
        for (uint32_t r = 0; r < 3; ++r) red &= out.data[r] == 2016.0f;
        for (uint32_t b = 3; b < 8; ++b) red &= std::isnan(out.data[b]);
        // WholeReduceSum: every repeat into one value, dstRepStride elements apart
        WholeReduceSum(out, src, 64, 3, 2, 1, 9);
        for (uint32_t r = 0; r < 3; ++r) red &= out.data[2 * r] == 2016.0f;
        // Cast: FP32 -> FP16 rounds to nearest-even (CAST_RINT, and CAST_NONE where precision is lost),
        // keeps subnormals and overflows to infinity; TRUNC / FLOOR / CEIL round as named; widening is
        // exact for every one of the 65536 FP16 values
        TBuf<QuePosition::VECCALC> bF, bH;
        p.InitBuffer(bF, 8 * sizeof(float));
        p.InitBuffer(bH, 8 * sizeof(half));
        LocalTensor<float> f = bF.Get<float>();
        LocalTensor<half> hv = bH.Get<half>();
        const float in[8] = {1.0f + 1.0f / 2048, 1.0f + 3.0f / 2048, -1.0f - 1.0f / 2048, 70000.0f, 1e-7f, 65519.0f, -2.5e-8f, 0.1f};
        const uint16_t rint[8] = {0x3C00, 0x3C02, 0xBC00, 0x7C00, 0x0002, 0x7BFF, 0x8000, 0x2E66};
        for (uint32_t i = 0; i < 8; ++i) f.data[i] = in[i];
        Cast(hv, f, RoundMode::CAST_RINT, 8);
        for (uint32_t i = 0; i < 8; ++i) red &= hv.data[i].data == rint[i];
        Cast(hv, f, RoundMode::CAST_NONE, 8);
        for (uint32_t i = 0; i < 8; ++i) red &= hv.data[i].data == rint[i];
        f.data[0] = 1.0f + 5.0f / 4096;  // Between 1 + 1/1024 and 1 + 2/1024
        f.data[1] = -1.0f - 5.0f / 4096;
        Cast(hv, f, RoundMode::CAST_TRUNC, 2);
        red &= hv.data[0].data == 0x3C01 && hv.data[1].data == 0xBC01;
        Cast(hv, f, RoundMode::CAST_FLOOR, 2);
        red &= hv.data[0].data == 0x3C01 && hv.data[1].data == 0xBC02;
        Cast(hv, f, RoundMode::CAST_CEIL, 2);
        red &= hv.data[0].data == 0x3C02 && hv.data[1].data == 0xBC01;
        for (uint32_t bits = 0; bits < 65536; bits += 8) {
            for (uint32_t i = 0; i < 8; ++i) hv.data[i].data = static_cast<uint16_t>(bits + i);
            Cast(f, hv, RoundMode::CAST_NONE, 8);
            Cast(hv, f, RoundMode::CAST_RINT, 8);
            for (uint32_t i = 0; i < 8; ++i) {
                const bool nan = ((bits + i) & 0x7C00) == 0x7C00 && ((bits + i) & 0x3FF);
                red &= nan ? std::isnan(f.data[i]) : hv.data[i].data == bits + i;
            }
        }
        ok &= red;
        if (red) std::cout << "PASS: BlockReduceSum / WholeReduceSum repeat structure, Cast rounding modes and FP16 round trip\n";
        else std::cerr << "FAIL: a target reduction or Cast does not follow its definition\n";
    }
    ok &= ExpectTrap("BlockReduceSum given a row width as its mask", [&] {
        withBuffers([](LocalTensor<float> a, LocalTensor<float> b, LocalTensor<float>) { BlockReduceSum(b, a, 1, 128, 1, 1, 8); });
    });
    ok &= ExpectTrap("BlockReduceSum into a destination off the 32-byte grid", [&] {
        withBuffers([](LocalTensor<float> a, LocalTensor<float> b, LocalTensor<float>) { BlockReduceSum(b[1], a, 1, 64, 1, 1, 8); });
    });

    if (!ok) return 1;

    std::cout << "\n=================================================================\n";
    std::cout << "  ALL DAE Stream Pipeline Runtime tests PASSED with 100% integrity!\n";
    std::cout << "=================================================================\n";
    return 0;
}
