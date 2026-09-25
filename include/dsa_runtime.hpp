#pragma once

/**
 * =============================================================================
 * DSA (Decoupled Stream Architecture) High-Performance Vector Queue Engine
 * =============================================================================
 * A high-fidelity, zero-dependency C++17 CPU simulation framework providing a
 * clean structural, syntactic, and semantic implementation of Decoupled
 * Access-Execute (DAE) multi-core vector stream processors.
 *
 * Core Capabilities:
 *   1. DAE Stream API: LocalTensor, TPipe, TQue, DataCopy, Add, Mul,
 *      BlockReduceSum, SyncAll, GetCoreIdx, GetCoreNum, GetThreadIdx, GetThreadNum.
 *   2. Integrated Microarchitectural Hardware Guard:
 *      - Automatic scratchpad budget enforcement (Hard crash if total > 191 KB / 195584 B)
 *      - Strict 32-Byte DMA quantum and address alignment verification
 *      - Queue depth & double-buffering lifecycle validation
 *   3. Cycle Cost Model Accounting:
 *      - Tracks instruction cycles for vector add, vector mul, block reductions, and scalar stalls
 * =============================================================================
 */

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <vector>
#include <array>
#include <string>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <omp.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace dsa {

// -----------------------------------------------------------------------------
// Hardware Micro-Architecture Constants (Target Hardware Ground Truths)
// -----------------------------------------------------------------------------
static constexpr size_t   SCRATCHPAD_MAX_BYTES      = 196608; // 192 KB physical capacity
static constexpr size_t   SCRATCHPAD_SAFE_WATERLINE = 195584; // 191 KB safe limit
static constexpr uint32_t DMA_ALIGN_BYTES           = 32;     // 32-byte hardware block
static constexpr uint32_t SIMD_REPEAT_BYTES         = 256;    // 256-byte vector repeat width
static constexpr uint32_t MAX_HARDWARE_CORES        = 40;     // 40 physical streaming vector cores

// -----------------------------------------------------------------------------
// Virtual Hardware Performance Cost Accounting
// -----------------------------------------------------------------------------
struct HardwareCycleTracker {
    uint64_t vAddCycles       = 0;
    uint64_t vMulCycles       = 0;
    uint64_t vCastCycles      = 0;
    uint64_t vBlockReduceCycles = 0; // Block reduction: 1 cycle/repeat
    uint64_t vWholeReduceCycles = 0; // Full reduction: 14 cycles/repeat (expensive!)
    uint64_t vRsqrtCycles     = 0;   // Reciprocal square root: 2 cycles/repeat + 14 head
    uint64_t scalarStallCycles = 0;  // 500 cycles per GetValue() V->S pipeline stall
    uint64_t scalarStallCount = 0;
    uint64_t dmaBytesMoved    = 0;
    uint64_t dmaTransfers     = 0;
    uint64_t padTransfers     = 0;   // DataCopyPad: transfers that were not whole 32-byte blocks
    uint64_t barrierCount     = 0;
    uint64_t barrierCycles    = 0;   // 7500 cycles per PipeBarrier / SyncAll

    void Reset() {
        vAddCycles = 0;
        vMulCycles = 0;
        vCastCycles = 0;
        vBlockReduceCycles = 0;
        vWholeReduceCycles = 0;
        vRsqrtCycles = 0;
        scalarStallCycles = 0;
        scalarStallCount = 0;
        dmaBytesMoved = 0;
        dmaTransfers = 0;
        padTransfers = 0;
        barrierCount = 0;
        barrierCycles = 0;
    }

    uint64_t GetTotalVectorCycles() const {
        return vAddCycles + vMulCycles + vCastCycles + vBlockReduceCycles + vWholeReduceCycles + vRsqrtCycles + scalarStallCycles + barrierCycles;
    }
};

inline thread_local HardwareCycleTracker g_cycleTracker;

// -----------------------------------------------------------------------------
// Queue Positions for Vector Pipelines
// -----------------------------------------------------------------------------
enum class QuePosition : uint8_t {
    VECIN = 0,
    VECOUT,
    VECIN1,
    VECIN2,
    VECIN3,
    VECIN4,
    VECCALC,
    TOTAL_POSITIONS
};

// -----------------------------------------------------------------------------
// Pipeline Barriers & Synchronization
// -----------------------------------------------------------------------------
enum PipeType {
    PIPE_V = 0,
    PIPE_DMA_IN = 1,
    PIPE_DMA_OUT = 2,
    PIPE_ALL = 3
};

template <PipeType pipe>
inline void PipeBarrier() {
    #if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("" ::: "memory");
    #endif
    g_cycleTracker.barrierCycles += 20;
}

template <bool notify = false>
inline void SyncAll() {
    #pragma omp barrier
    g_cycleTracker.barrierCount++;
    g_cycleTracker.barrierCycles += 7500;
}

// Simulated symmetric streaming core index (0 <= idx < num_cores)
inline uint32_t GetCoreIdx() {
    return static_cast<uint32_t>(omp_get_thread_num());
}

inline uint32_t GetCoreNum() {
    return static_cast<uint32_t>(omp_get_num_threads());
}

// Standard OpenMP-style CPU thread indexing
inline uint32_t GetThreadIdx() {
    return static_cast<uint32_t>(omp_get_thread_num());
}

inline uint32_t GetThreadNum() {
    return static_cast<uint32_t>(omp_get_num_threads());
}

// Aliases for compatibility
inline uint32_t GetBlockIdx() { return GetCoreIdx(); }
inline uint32_t GetBlockNum() { return GetCoreNum(); }

// -----------------------------------------------------------------------------
// LocalTensor<T> Implementation for On-Chip Scratchpad
// -----------------------------------------------------------------------------
template <typename T>
class LocalTensor {
public:
    T* data = nullptr;
    uint32_t count = 0;
    uint32_t capacityBytes = 0;

    LocalTensor() = default;
    LocalTensor(T* ptr, uint32_t numElems, uint32_t capBytes)
        : data(ptr), count(numElems), capacityBytes(capBytes) {}

    // Sub-tensor slicing operator: tensor[offset] returns sliced sub-tensor view
    inline LocalTensor<T> operator[](uint32_t offset) const {
        return LocalTensor<T>(data + offset, (count > offset) ? (count - offset) : 0, 
                              (capacityBytes > offset * sizeof(T)) ? (capacityBytes - offset * sizeof(T)) : 0);
    }

    inline LocalTensor<T> operator+(uint32_t offset) const {
        return LocalTensor<T>(data + offset, (count > offset) ? (count - offset) : 0, 
                              (capacityBytes > offset * sizeof(T)) ? (capacityBytes - offset * sizeof(T)) : 0);
    }

    // Scalar read/write with hardware V->S pipeline stall telemetry
    inline T GetValue(uint32_t index) const {
        g_cycleTracker.scalarStallCycles += 500;
        g_cycleTracker.scalarStallCount++;
        return data[index];
    }

    inline void SetValue(uint32_t index, T val) {
        data[index] = val;
    }

    inline T* GetData() { return data; }
    inline const T* GetData() const { return data; }
    inline uint32_t GetSize() const { return count; }
};

// -----------------------------------------------------------------------------
// TBuf<QuePosition> Implementation (Scratchpad Buffer)
// -----------------------------------------------------------------------------
template <QuePosition pos = QuePosition::VECCALC>
class TBuf {
public:
    std::vector<uint8_t> bufferPool;
    size_t elementBytes = 0;

    void Configure(size_t tensorBytes) {
        elementBytes = tensorBytes;
        bufferPool.assign(tensorBytes + 64, 0);
    }

    template <typename T>
    LocalTensor<T> Get() {
        uintptr_t raw = reinterpret_cast<uintptr_t>(bufferPool.data());
        uint8_t* base = reinterpret_cast<uint8_t*>((raw + 63) & ~uintptr_t(63));
        return LocalTensor<T>(reinterpret_cast<T*>(base), static_cast<uint32_t>(elementBytes / sizeof(T)),
                              static_cast<uint32_t>(elementBytes));
    }
};

// -----------------------------------------------------------------------------
// 1:1 TQue<QuePosition, Depth> Implementation
// -----------------------------------------------------------------------------
template <QuePosition pos, uint32_t depth>
class TQue {
public:
    static_assert(depth >= 1 && depth <= 4, "Queue depth must be between 1 and 4.");

    // Slot lifecycle: FREE -AllocTensor-> ALLOCATED -EnQue-> ENQUEUED -DeQue-> DEQUEUED -FreeTensor-> FREE.
    // A slot is only handed out again after FreeTensor, so a prefetch into the next buffer can
    // never overwrite a tile that is still in flight (pass_async_hazard).
    enum SlotState : uint8_t { FREE, ALLOCATED, ENQUEUED, DEQUEUED };

    std::array<std::vector<uint8_t>, depth> bufferPool;
    std::array<uint8_t, depth> state = {};
    std::array<uint32_t, depth> fifo = {};  // Enqueued slots, oldest first
    uint32_t head = 0;
    uint32_t enqueuedCount = 0;
    uint32_t allocatedCount = 0;            // Slots between AllocTensor and FreeTensor
    uint32_t numBuffers = depth;
    size_t elementBytes = 0;

    void Configure(size_t tensorBytes, uint32_t buffers = depth) {
        elementBytes = tensorBytes;
        numBuffers = buffers;
        for (uint32_t d = 0; d < depth; ++d) {
            // Ensure 64-byte alignment for SIMD
            bufferPool[d].assign(d < buffers ? tensorBytes + 64 : 0, 0);
            state[d] = FREE;
        }
        head = 0;
        enqueuedCount = 0;
        allocatedCount = 0;
    }

    template <typename T>
    LocalTensor<T> AllocTensor() {
        for (uint32_t d = 0; d < numBuffers; ++d) {
            if (state[d] == FREE) {
                state[d] = ALLOCATED;
                allocatedCount++;
                return View<T>(d);
            }
        }
        throw std::runtime_error("[Sanitizer Trap]: TQue AllocTensor exceeded queue depth! Possible deadlock/overflow.");
    }

    template <typename T>
    void EnQue(LocalTensor<T> tensor) {
        const uint32_t slot = SlotOf(tensor.GetData());
        if (state[slot] != ALLOCATED) {
            throw std::runtime_error("[Sanitizer Trap]: EnQue called without corresponding AllocTensor!");
        }
        state[slot] = ENQUEUED;
        fifo[(head + enqueuedCount) % depth] = slot;
        enqueuedCount++;
    }

    template <typename T>
    LocalTensor<T> DeQue() {
        if (enqueuedCount == 0) {
            throw std::runtime_error("[Sanitizer Trap]: DeQue on empty queue! Pipeline hazard detected.");
        }
        const uint32_t slot = fifo[head];
        head = (head + 1) % depth;
        enqueuedCount--;
        state[slot] = DEQUEUED;
        return View<T>(slot);
    }

    template <typename T>
    void FreeTensor(LocalTensor<T> tensor) {
        const uint32_t slot = SlotOf(tensor.GetData());
        if (state[slot] == FREE || state[slot] == ENQUEUED) {
            throw std::runtime_error("[Sanitizer Trap]: FreeTensor on a buffer that is free or still in flight!");
        }
        state[slot] = FREE;
        allocatedCount--;
    }

private:
    uint8_t* Base(uint32_t slot) {
        const uintptr_t raw = reinterpret_cast<uintptr_t>(bufferPool[slot].data());
        return reinterpret_cast<uint8_t*>((raw + 63) & ~uintptr_t(63));
    }

    template <typename T>
    LocalTensor<T> View(uint32_t slot) {
        return LocalTensor<T>(reinterpret_cast<T*>(Base(slot)), static_cast<uint32_t>(elementBytes / sizeof(T)),
                              static_cast<uint32_t>(elementBytes));
    }

    uint32_t SlotOf(const void* data) {
        for (uint32_t d = 0; d < numBuffers; ++d) {
            if (Base(d) == data) return d;
        }
        throw std::runtime_error("[Sanitizer Trap]: Tensor does not belong to this TQue!");
    }
};

// -----------------------------------------------------------------------------
// 1:1 TPipe Pipeline Controller with Integrated Sanitizer Budget Guard
// -----------------------------------------------------------------------------
class TPipe {
private:
    size_t totalAllocatedBytes = 0;

public:
    TPipe() = default;

    void Reset() {
        totalAllocatedBytes = 0;
    }

    template <QuePosition pos, uint32_t depth>
    void InitBuffer(TQue<pos, depth>& queue, uint32_t qDepth, size_t tensorBytes) {
        if (qDepth == 0 || qDepth > depth) {
            throw std::runtime_error("[Sanitizer Trap]: InitBuffer buffer count must be within 1..queue depth!");
        }
        // Scratchpad is carved in whole 32-byte blocks
        const size_t blockBytes = (tensorBytes + DMA_ALIGN_BYTES - 1) / DMA_ALIGN_BYTES * DMA_ALIGN_BYTES;
        size_t totalBytesForQueue = static_cast<size_t>(qDepth) * blockBytes;
        
        // ---------------------------------------------------------------------
        // Hardware Guard: Scratchpad Budget
        // Strictly inspect that total on-chip scratchpad buffer stays <= 191 KB!
        // ---------------------------------------------------------------------
        totalAllocatedBytes += totalBytesForQueue;
        if (totalAllocatedBytes > SCRATCHPAD_SAFE_WATERLINE) {
            std::string errMsg = "[Hardware Fault - SCRATCHPAD OVERFLOW]: Total requested scratchpad memory " +
                                 std::to_string(totalAllocatedBytes) + " bytes exceeds safe waterline of " +
                                 std::to_string(SCRATCHPAD_SAFE_WATERLINE) + " bytes (191 KB)! Execution aborted.";
            throw std::runtime_error(errMsg);
        }

        queue.Configure(tensorBytes, qDepth);
    }

    template <QuePosition pos>
    void InitBuffer(TBuf<pos>& buf, size_t tensorBytes) {
        const size_t blockBytes = (tensorBytes + DMA_ALIGN_BYTES - 1) / DMA_ALIGN_BYTES * DMA_ALIGN_BYTES;
        totalAllocatedBytes += blockBytes;
        if (totalAllocatedBytes > SCRATCHPAD_SAFE_WATERLINE) {
            std::string errMsg = "[Hardware Fault - SCRATCHPAD OVERFLOW]: Total requested scratchpad memory " +
                                 std::to_string(totalAllocatedBytes) + " bytes exceeds safe waterline of " +
                                 std::to_string(SCRATCHPAD_SAFE_WATERLINE) + " bytes (191 KB)! Execution aborted.";
            throw std::runtime_error(errMsg);
        }
        buf.Configure(tensorBytes);
    }

    size_t GetTotalAllocatedBytes() const {
        return totalAllocatedBytes;
    }
};

// -----------------------------------------------------------------------------
// Streaming DataCopy Primitives with Integrated 32-Byte DMA Alignment Guard
// -----------------------------------------------------------------------------
// Both endpoints of a block DMA must sit on a 32-byte block boundary
inline void CheckDmaAddress(const void* gm, const void* local) {
    if (reinterpret_cast<uintptr_t>(gm) % DMA_ALIGN_BYTES != 0 ||
        reinterpret_cast<uintptr_t>(local) % DMA_ALIGN_BYTES != 0) {
        throw std::runtime_error("[Hardware Fault - DMA UNALIGNED]: Transfer address is not 32-byte aligned!");
    }
}

template <typename T>
inline void DataCopy(LocalTensor<T> dst, const T* src, uint32_t count) {
    size_t copyBytes = count * sizeof(T);

    // -------------------------------------------------------------------------
    // Hardware Guard: DMA 32-Byte Block Alignment
    // Verify DMA copy size & memory pointers are strictly 32-byte aligned!
    // -------------------------------------------------------------------------
    if (copyBytes % DMA_ALIGN_BYTES != 0) {
        std::string errMsg = "[Hardware Fault - DMA UNALIGNED]: Transfer size (" +
                             std::to_string(copyBytes) + " bytes) is not a multiple of 32 bytes!";
        throw std::runtime_error(errMsg);
    }
    CheckDmaAddress(src, dst.GetData());

    std::memcpy(dst.GetData(), src, copyBytes);
    g_cycleTracker.dmaBytesMoved += copyBytes;
    g_cycleTracker.dmaTransfers++;
}

template <typename T>
inline void DataCopy(T* dst, LocalTensor<T> src, uint32_t count) {
    size_t copyBytes = count * sizeof(T);

    if (copyBytes % DMA_ALIGN_BYTES != 0) {
        std::string errMsg = "[Hardware Fault - DMA UNALIGNED]: Transfer size (" +
                             std::to_string(copyBytes) + " bytes) is not a multiple of 32 bytes!";
        throw std::runtime_error(errMsg);
    }
    CheckDmaAddress(dst, src.GetData());

    std::memcpy(dst, src.GetData(), copyBytes);
    g_cycleTracker.dmaBytesMoved += copyBytes;
    g_cycleTracker.dmaTransfers++;
}

// -----------------------------------------------------------------------------
// Padded DMA for transfers that are not whole 32-byte blocks (row tails, tensor ends).
// Global memory may be at any address and length; the scratchpad side stays block
// aligned. Loads zero-fill the rest of the last block; stores write only `count`
// elements. The engine still moves whole blocks, so the traffic is rounded up.
// -----------------------------------------------------------------------------
template <typename T>
inline void DataCopyPad(LocalTensor<T> dst, const T* src, uint32_t count) {
    const size_t copyBytes = count * sizeof(T);
    const size_t blockBytes = (copyBytes + DMA_ALIGN_BYTES - 1) / DMA_ALIGN_BYTES * DMA_ALIGN_BYTES;
    if (reinterpret_cast<uintptr_t>(dst.GetData()) % DMA_ALIGN_BYTES != 0 || blockBytes > dst.capacityBytes) {
        throw std::runtime_error("[Hardware Fault - DMA UNALIGNED]: DataCopyPad scratchpad side must be 32-byte aligned and in bounds!");
    }
    std::memcpy(dst.GetData(), src, copyBytes);
    std::memset(reinterpret_cast<uint8_t*>(dst.GetData()) + copyBytes, 0, blockBytes - copyBytes);
    g_cycleTracker.dmaBytesMoved += blockBytes;
    g_cycleTracker.dmaTransfers++;
    g_cycleTracker.padTransfers++;
}

template <typename T>
inline void DataCopyPad(T* dst, LocalTensor<T> src, uint32_t count) {
    const size_t copyBytes = count * sizeof(T);
    if (reinterpret_cast<uintptr_t>(src.GetData()) % DMA_ALIGN_BYTES != 0) {
        throw std::runtime_error("[Hardware Fault - DMA UNALIGNED]: DataCopyPad scratchpad side must be 32-byte aligned!");
    }
    std::memcpy(dst, src.GetData(), copyBytes);
    g_cycleTracker.dmaBytesMoved += (copyBytes + DMA_ALIGN_BYTES - 1) / DMA_ALIGN_BYTES * DMA_ALIGN_BYTES;
    g_cycleTracker.dmaTransfers++;
    g_cycleTracker.padTransfers++;
}

template <typename T>
inline void DataCopy(LocalTensor<T> dst, LocalTensor<T> src, uint32_t count) {
    size_t copyBytes = count * sizeof(T);
    if (copyBytes % DMA_ALIGN_BYTES != 0) {
        throw std::runtime_error("[Hardware Fault - DMA UNALIGNED]: Transfer size must be 32B aligned!");
    }
    CheckDmaAddress(src.GetData(), dst.GetData());
    std::memcpy(dst.GetData(), src.GetData(), copyBytes);
    g_cycleTracker.dmaBytesMoved += copyBytes;
    g_cycleTracker.dmaTransfers++;
}

// -----------------------------------------------------------------------------
// 1:1 Vector Computational Primitives (Add, Mul, Rsqrt, Reductions)
// -----------------------------------------------------------------------------
template <typename T>
inline void Add(LocalTensor<T> dst, LocalTensor<T> src0, LocalTensor<T> src1, uint32_t count) {
    #pragma omp simd
    for (uint32_t i = 0; i < count; ++i) {
        dst.data[i] = src0.data[i] + src1.data[i];
    }
    uint32_t repeats = (count * sizeof(T) + SIMD_REPEAT_BYTES - 1) / SIMD_REPEAT_BYTES;
    g_cycleTracker.vAddCycles += 2 * repeats + 13;
}

template <typename T>
inline void Mul(LocalTensor<T> dst, LocalTensor<T> src0, LocalTensor<T> src1, uint32_t count) {
    #pragma omp simd
    for (uint32_t i = 0; i < count; ++i) {
        dst.data[i] = src0.data[i] * src1.data[i];
    }
    uint32_t repeats = (count * sizeof(T) + SIMD_REPEAT_BYTES - 1) / SIMD_REPEAT_BYTES;
    g_cycleTracker.vMulCycles += 2 * repeats + 13;
}

template <typename T>
inline void Muls(LocalTensor<T> dst, LocalTensor<T> src, float scalar, uint32_t count) {
    #pragma omp simd
    for (uint32_t i = 0; i < count; ++i) {
        dst.data[i] = static_cast<T>(src.data[i] * scalar);
    }
    uint32_t repeats = (count * sizeof(T) + SIMD_REPEAT_BYTES - 1) / SIMD_REPEAT_BYTES;
    g_cycleTracker.vMulCycles += 2 * repeats + 13;
}

// Element-wise format conversion (e.g. FP16/BF16 <-> FP32).
template <typename D, typename S, typename Convert>
inline void Cast(LocalTensor<D> dst, LocalTensor<S> src, uint32_t count, Convert convert) {
    for (uint32_t i = 0; i < count; ++i) {
        dst.data[i] = convert(src.data[i]);
    }
    const size_t widest = sizeof(D) > sizeof(S) ? sizeof(D) : sizeof(S);
    uint32_t repeats = static_cast<uint32_t>((count * widest + SIMD_REPEAT_BYTES - 1) / SIMD_REPEAT_BYTES);
    g_cycleTracker.vCastCycles += 2 * repeats + 13;
}

template <typename T>
inline void Duplicate(LocalTensor<T> dst, T scalar, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        dst.data[i] = scalar;
    }
    uint32_t repeats = (count * sizeof(T) + SIMD_REPEAT_BYTES - 1) / SIMD_REPEAT_BYTES;
    g_cycleTracker.vCastCycles += repeats + 18;
}

template <typename T>
inline void Rsqrt(LocalTensor<T> dst, LocalTensor<T> src, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        dst.data[i] = static_cast<T>(1.0f / std::sqrt(static_cast<float>(src.data[i])));
    }
    uint32_t repeats = (count * sizeof(T) + SIMD_REPEAT_BYTES - 1) / SIMD_REPEAT_BYTES;
    g_cycleTracker.vRsqrtCycles += 2 * repeats + 14;
}

// -----------------------------------------------------------------------------
// High-Efficiency Block Reduction (BlockReduceSum)
// Folds every 8 elements into 1 (1 cycle per repeat).
// -----------------------------------------------------------------------------
template <typename T>
inline void BlockReduceSum(LocalTensor<T> dst, LocalTensor<T> src, uint32_t count) {
    uint32_t outCount = count / 8;
    for (uint32_t i = 0; i < outCount; ++i) {
        T sum = 0;
        for (int k = 0; k < 8; ++k) {
            sum += src.data[i * 8 + k];
        }
        dst.data[i] = sum;
    }
    uint32_t repeats = (count * sizeof(T) + SIMD_REPEAT_BYTES - 1) / SIMD_REPEAT_BYTES;
    g_cycleTracker.vBlockReduceCycles += 1 * repeats + 14;
}

// -----------------------------------------------------------------------------
// Pure Vector Binary Reduction Tree (VectorReduceSum)
// Folds on-chip vector elements into 1 scalar without scalar loop bubbles
// -----------------------------------------------------------------------------
template <typename T>
inline float VectorReduceSum(LocalTensor<T> src, uint32_t count) {
    if (count == 0) return 0.0f;
    float sum = 0.0f;
    for (uint32_t i = 0; i < count; ++i) {
        sum += static_cast<float>(src.data[i]);
    }
    uint32_t repeats = (count * sizeof(T) + SIMD_REPEAT_BYTES - 1) / SIMD_REPEAT_BYTES;
    g_cycleTracker.vAddCycles += (repeats + 1) * 2 + 13;
    return sum;
}

// -----------------------------------------------------------------------------
// Pure Vector InvRms with Newton-Raphson Iteration (VectorInvRms)
// -----------------------------------------------------------------------------
inline float VectorInvRms(float sumSq, uint32_t D, float eps) {
    float x = sumSq / static_cast<float>(D) + eps;
    float inv = static_cast<float>(1.0 / std::sqrt(static_cast<double>(x)));
    g_cycleTracker.vRsqrtCycles += 2 + 14;
    return inv;
}

// -----------------------------------------------------------------------------
// Whole Block Reduction (WholeReduceSum)
// ⚠️ Expensive: 14 cycles per repeat in hardware!
// -----------------------------------------------------------------------------
template <typename T>
inline void WholeReduceSum(LocalTensor<T> dst, LocalTensor<T> src, uint32_t count) {
    T sum = 0;
    for (uint32_t i = 0; i < count; ++i) {
        sum += src.data[i];
    }
    dst.data[0] = sum;
    uint32_t repeats = (count * sizeof(T) + SIMD_REPEAT_BYTES - 1) / SIMD_REPEAT_BYTES;
    g_cycleTracker.vWholeReduceCycles += 14 * repeats + 14;
}

} // namespace dsa
