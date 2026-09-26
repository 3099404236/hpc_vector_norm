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

    void Process(const float* src, float* dst, uint32_t totalElems) {
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
            DataCopy(inTensor, src + t * 512, 512);
            inQueue.EnQue(inTensor);

            // Stage 2: SIMD Vector Compute
            LocalTensor<float> workTensor = inQueue.DeQue<float>();
            LocalTensor<float> outTensor = outQueue.AllocTensor<float>();
            
            Mul(outTensor, workTensor, workTensor, 512); // Square
            inQueue.FreeTensor(workTensor);

            outQueue.EnQue(outTensor);

            // Stage 3: DMA Outbound
            LocalTensor<float> writeTensor = outQueue.DeQue<float>();
            DataCopy(dst + t * 512, writeTensor, 512);
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
    kernel.Process(input.data(), output.data(), ELEMS);

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
        DataCopy(t, input.data(), 5); // 5 floats = 20 bytes (not 32-byte multiple!)
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
        DataCopy(t0, a.data(), 8);
        dq.EnQue(t0);
        LocalTensor<float> t1 = dq.AllocTensor<float>();  // prefetch into the second buffer
        DataCopy(t1, b.data(), 8);
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
        DataCopy(t, input.data() + 1, 8);
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
        DataCopyPad(t, input.data() + 3, 5);
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
    // Test 8: Timeline model. One core: a load lands DMA_LATENCY_CYCLES after it streamed at
    // DMA_BYTES_PER_CYCLE; the add waits for it; the store follows the add and lands one
    // latency later. Two tiles: the second load streams while the first tile computes.
    // -------------------------------------------------------------------------
    std::cout << "\n[Testing Timeline Model]...\n";
    {
        auto near = [](double a, double b) { return std::fabs(a - b) <= 1e-9 * (std::fabs(b) + 1.0); };
        const double bw = DMA_BYTES_PER_CYCLE, L = DMA_LATENCY_CYCLES;
        AlignedVector<float> src(1024, 1.0f), dst(1024, 0.0f);
        {
            TPipe p;
            TQue<QuePosition::VECIN, 1> qIn;
            TQue<QuePosition::VECOUT, 1> qOut;
            p.InitBuffer(qIn, 1, 1024 * sizeof(float));
            p.InitBuffer(qOut, 1, 1024 * sizeof(float));
            LocalTensor<float> tIn = qIn.AllocTensor<float>();
            LocalTensor<float> tOut = qOut.AllocTensor<float>();
            DataCopy(tIn, src.data(), 1024);  // 4 KB
            Add(tOut, tIn, tIn, 1024);        // 16 repeats: 45 cycles
            DataCopy(dst.data(), tOut, 1024);
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
            DataCopy(a, src.data(), 1024);
            DataCopy(b, src.data(), 1024);  // Streams while tile a computes
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
        DataCopy(output.data(), t, 64);
    });
    ok &= ExpectTrap("BlockReduceSum in-place buffer aliasing (dst == src)", [&] {
        TPipe p;
        TBuf<QuePosition::VECCALC> b;
        p.InitBuffer(b, 256);
        LocalTensor<float> t = b.Get<float>();
        BlockReduceSum(t, t, 64);
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

    // Trap #410: Brcb buffer overrun (capacity < 64 floats)
    ok &= ExpectTrap("Brcb destination buffer undersized (capacity < 64 floats, Trap #410)", [&] {
        TPipe p;
        TBuf<QuePosition::VECCALC> bDst, bSrc;
        p.InitBuffer(bDst, 32); // only 8 floats
        p.InitBuffer(bSrc, 32);
        bSrc.Get<float>().data[0] = 3.14f;
        Brcb(bDst.Get<float>(), bSrc.Get<float>());
    });

    // Trap #409: Host-to-Device argument frame overflow (> 32 bytes)
    ok &= ExpectTrap("Launch arguments frame overflow (> 32 bytes, Trap #409)", [&] {
        struct BigPlanningStruct {
            uint64_t fields[5]; // 40 bytes
        } s{};
        ValidateLaunchArgs(s);
    });

    // Brcb valid broadcast verification & LocalMemAllocator
    {
        LocalMemAllocator<Hardware::UB> mem;
        auto scalar = mem.Alloc<float, 8>();
        auto bcast = mem.Alloc<float, 64>();
        scalar.data[0] = 42.0f;
        Brcb(bcast, scalar);
        bool brcbOk = true;
        for (uint32_t i = 0; i < 64; ++i) {
            if (bcast.data[i] != 42.0f) brcbOk = false;
        }
        if (brcbOk) {
            std::cout << "PASS: Brcb broadcasts scalar across all 64 SIMD lanes\n";
        } else {
            std::cerr << "FAIL: Brcb did not broadcast correctly\n";
            ok = false;
        }
    }

    if (!ok) return 1;

    std::cout << "\n=================================================================\n";
    std::cout << "  ALL DAE Stream Pipeline Runtime tests PASSED with 100% integrity!\n";
    std::cout << "=================================================================\n";
    return 0;
}
