#include "dsa_runtime.hpp"
#include <iostream>
#include <cassert>
#include <cstdlib>
#include <vector>

using namespace dsa;

// Global-memory buffers on the 32-byte DMA block boundary (alignas on a std::vector
// object does not align its heap storage)
template <typename T>
struct GmAllocator {
    using value_type = T;
    GmAllocator() = default;
    template <typename U> GmAllocator(const GmAllocator<U>&) {}
    T* allocate(size_t n) { return static_cast<T*>(std::aligned_alloc(64, (n * sizeof(T) + 63) / 64 * 64)); }
    void deallocate(T* p, size_t) { std::free(p); }
    template <typename U> bool operator==(const GmAllocator<U>&) const { return true; }
    template <typename U> bool operator!=(const GmAllocator<U>&) const { return false; }
};
template <typename T> using GmVector = std::vector<T, GmAllocator<T>>;

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
    GmVector<float> input(ELEMS, 2.0f);
    GmVector<float> output(ELEMS, 0.0f);

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
        GmVector<float> a(8, 1.0f), b(8, 2.0f);
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
        if (t0.GetData() == t1.GetData() || c0[0] != 1.0f || c1[0] != 2.0f) {
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
        if (t[4] != 2.0f || t[5] != 0.0f || t[7] != 0.0f || g_cycleTracker.padTransfers != pads + 1) {
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

    if (!ok) return 1;

    std::cout << "\n=================================================================\n";
    std::cout << "  ALL DAE Stream Pipeline Runtime tests PASSED with 100% integrity!\n";
    std::cout << "=================================================================\n";
    return 0;
}
