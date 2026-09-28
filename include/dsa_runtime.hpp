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
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <array>
#include <string>
#include <stdexcept>
#include <limits>
#include <type_traits>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <omp.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

// -----------------------------------------------------------------------------
// Freestanding & Low-Latency Assert Guard (Zero Exception Overhead)
// -----------------------------------------------------------------------------
#ifndef DSA_ASSERT
#define DSA_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[DSA Hardware Trap]: " << (msg) << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            std::abort(); \
        } \
    } while (0)
#endif

namespace dsa {

#ifndef DAE_HALF_DEFINED
#define DAE_HALF_DEFINED
struct half {
    uint16_t data = 0;
    half() = default;
    // Rounded to nearest-even, subnormals kept, past 65504 infinity: the target's CAST_RINT
    half(float f) {
        uint32_t x;
        std::memcpy(&x, &f, 4);
        const uint32_t sign = (x >> 16) & 0x8000u, ax = x & 0x7FFFFFFFu;
        if (ax >= 0x7F800000u) {  // Infinity, NaN
            data = static_cast<uint16_t>(sign | (ax > 0x7F800000u ? 0x7E00u : 0x7C00u));
        } else if (ax >= 0x477FF000u) {  // Rounds past the largest finite value
            data = static_cast<uint16_t>(sign | 0x7C00u);
        } else if (ax < 0x38800000u) {  // Subnormal: |f| / 2^-24, rounded to nearest-even
            float a;
            std::memcpy(&a, &ax, 4);
            data = static_cast<uint16_t>(sign | static_cast<uint32_t>(std::nearbyint(a * 16777216.0f)));
        } else {
            const uint32_t mant = ax & 0x7FFFFFu, rem = mant & 0x1FFFu;
            uint32_t h = (((ax >> 23) - 127 + 15) << 10) | (mant >> 13);
            if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;  // A carry rolls into the exponent
            data = static_cast<uint16_t>(sign | h);
        }
    }
    operator float() const {
        uint32_t sign = (data & 0x8000) << 16;
        int32_t exp = (data >> 10) & 0x1f;
        uint32_t mant = (data & 0x3ff) << 13;
        if (exp == 0) {  // Subnormal: normalized, then rebiased like any other value
            if (mant == 0) return (sign ? -0.0f : 0.0f);
            exp = 1;
            while (!(mant & 0x00800000)) { mant <<= 1; exp--; }
            mant &= 0x007fffff;
            exp = exp - 15 + 127;
        } else if (exp == 31) {
            exp = 255;
        } else {
            exp = exp - 15 + 127;
        }
        uint32_t x = sign | (static_cast<uint32_t>(exp) << 23) | mant;
        float f;
        std::memcpy(&f, &x, 4);
        return f;
    }
    half operator+(half other) const { return half(float(*this) + float(other)); }
    half operator*(half other) const { return half(float(*this) * float(other)); }
    half& operator+=(half other) { *this = *this + other; return *this; }
    half& operator*=(half other) { *this = *this * other; return *this; }
};
#endif

#ifndef DAE_BFLOAT16_DEFINED
#define DAE_BFLOAT16_DEFINED
struct bfloat16_t {
    uint16_t data = 0;
    bfloat16_t() = default;
    bfloat16_t(float f) {
        uint32_t x;
        std::memcpy(&x, &f, 4);
        data = static_cast<uint16_t>(x >> 16);
    }
    operator float() const {
        uint32_t x = static_cast<uint32_t>(data) << 16;
        float f;
        std::memcpy(&f, &x, 4);
        return f;
    }
    bfloat16_t operator+(bfloat16_t other) const { return bfloat16_t(float(*this) + float(other)); }
    bfloat16_t operator*(bfloat16_t other) const { return bfloat16_t(float(*this) * float(other)); }
    bfloat16_t& operator+=(bfloat16_t other) { *this = *this + other; return *this; }
    bfloat16_t& operator*=(bfloat16_t other) { *this = *this * other; return *this; }
};
#endif

// Global Memory Descriptor (Many-core stream processor system memory buffer)
template <typename T>
class GlobalTensor {
public:
    T* data = nullptr;

    GlobalTensor() = default;
    explicit GlobalTensor(T* ptr) : data(ptr) {}

    void SetGlobalBuffer(T* ptr, size_t count = 0) {
        (void)count;
        data = ptr;
    }

    T* GetData() const { return data; }
    operator T*() const { return data; }
    operator const T*() const { return data; }
    T* operator+(size_t offset) const { return data + offset; }
    GlobalTensor<T> operator[](size_t offset) const { return GlobalTensor<T>(data + offset); }
};


// -----------------------------------------------------------------------------
// Freestanding Execution Guidelines: Master Coordinator vs. Worker Threads
// -----------------------------------------------------------------------------
// In zero-allocation, ultra-low-latency multi-core execution:
// 1. Master Coordinator (DaePipeline::Execute / Master Thread):
//    - Evaluates TilingConfig on the master CPU thread before the parallel region.
//    - Manages shared pre-allocated reduction workspace buffer (64-byte aligned).
//    - Dispatches the OpenMP worker team passing POD pointers and configurations.
// 2. Worker Threads (Core::Execute / Worker Thread):
//    - Strictly freestanding: NO dynamic memory allocation (no malloc, new, std::vector).
//    - No C++ exception handling inside workers (use DSA_ASSERT / error traps).
//    - Uses fixed-size stack arrays (e.g. float sums[128]) or caller-provided workspace.
//    - LocalTensor passed by value or const-reference (never LocalTensor*).
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Hardware Micro-Architecture Constants (Target Hardware Ground Truths)
// -----------------------------------------------------------------------------
static constexpr size_t   SCRATCHPAD_MAX_BYTES      = 196608; // 192 KB physical capacity
static constexpr size_t   SCRATCHPAD_SAFE_WATERLINE = 195584; // 191 KB safe limit
static constexpr uint32_t DMA_ALIGN_BYTES           = 32;     // 32-byte hardware block
static constexpr uint32_t SIMD_REPEAT_BYTES         = 256;    // 256-byte vector repeat width
static constexpr uint32_t MAX_HARDWARE_CORES        = 40;     // 40 physical streaming vector cores
static constexpr uint8_t  SCRATCHPAD_POISON         = 0xFF;   // What a new scratchpad buffer reads as (NaN)

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
    uint64_t scalarStallCycles = 0;  // GET_VALUE_CYCLES per GetValue() V->S read
    uint64_t scalarStallCount = 0;
    uint64_t dmaBytesMoved    = 0;
    uint64_t dmaTransfers     = 0;
    uint64_t padTransfers     = 0;   // DataCopyPad: transfers that were not whole 32-byte blocks
    uint64_t barrierCount     = 0;
    uint64_t barrierCycles    = 0;   // SYNC_ALL_CYCLES per SyncAll, PIPE_ALL_CYCLES per PipeBarrier<PIPE_ALL>
    uint64_t queueSequencerCycles = 0; // State-machine queue lifecycle penalty (1800-2500 cycles per tile for TQue)

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
        queueSequencerCycles = 0;
    }

    uint64_t GetTotalVectorCycles() const {
        return vAddCycles + vMulCycles + vCastCycles + vBlockReduceCycles + vWholeReduceCycles + vRsqrtCycles + scalarStallCycles + barrierCycles + queueSequencerCycles;
    }
};

inline thread_local HardwareCycleTracker g_cycleTracker;

// -----------------------------------------------------------------------------
// Timeline model: when each operation of a core runs, not only what it costs.
// A core has three in-order units:
//   VECTOR  the vector pipe: every vector instruction, for its cycle cost
//   DMA     the system-memory channel, shared by loads and stores: a transfer occupies it
//           for g_memory.Cycles(bytes) (MemorySystem, below), and its data lands
//           DMA_LATENCY_CYCLES after that (the latency of back-to-back transfers overlaps)
//   LOCAL   scratchpad-to-scratchpad copies (no system-memory traffic)
// Operations issue in program order. Each starts once its unit is free and its operands
// are: a 32-byte scratchpad block can be read once its last write has landed, and written
// once its last read has ended. A core arrives at SyncAll once its vector and local units
// are idle and its stores have landed (loads may stay in flight); all cores leave
// SYNC_ALL_CYCLES after the last arrival. Times are vector cycles (CLOCK_GHZ per ns). The
// kernel launch (KERNEL_LAUNCH_NS) precedes every core's timeline.
//
// Latency floor: the same program replayed alongside on a relaxed core, where
//   - an operation waits only for its unit and its operands' data, never for a buffer's
//     previous reads or writes to finish (unlimited scratchpad: no buffer reuse);
//   - loads and stores stream through two queues of their own, so a store waiting for its data
//     never holds up the loads behind it;
//   - SyncAll releases the core SYNC_ALL_CYCLES after its own arrival.
// Every operation starts no later than on the real core (by induction over the program), so
// no run finishes before the replay, nor before the channel's total occupancy plus one latency.
// Excess = finish - floor is what buffer reuse, the shared channel's order and waiting for
// other cores at SyncAll cost; the floor itself is DMA latency and dependencies.
// -----------------------------------------------------------------------------
// Timing measured on the target (docs/TARGET_MEASUREMENTS.md, sections in brackets).
//
// The time base [2]: a vector instruction costs 2 cycles per 256-byte repeat, and one repeat of a
// 4-byte pass measures 64 x 0.0182 ns, so a cycle is half of that. CLOCK_GHZ is this conversion,
// not a measured clock frequency. The fixed per-instruction terms of the cycle costs (+13, +14,
// +15, +18) are not measured [10]: their magnitude should not decide a plan on its own.
static constexpr double   VECTOR_REPEAT_NS      = 64 * 0.0182;               // [2] One repeat of a 4-byte pass
static constexpr double   CLOCK_GHZ             = 2.0 / VECTOR_REPEAT_NS;    // Cycles per ns: 1.717
static constexpr double   KERNEL_LAUNCH_NS      = 1700.0;  // [1] Launch and per-core init, the same for any core count
static constexpr double   SYNC_ALL_NS           = 924.0;   // [1] SyncAll
static constexpr double   PIPE_ALL_NS           = 12.0;    // [1] PipeBarrier<PIPE_ALL>, once the pipes are drained
static constexpr double   GET_VALUE_NS          = 1.3;     // [1] A scalar read of the scratchpad (upper bound)
// DMA latency, from a transfer's end on the channel to its data landing: NOT measured. The doc's
// launch (1.70 us) is the whole of the fastest 1 x 64 run, of which per-core init keeps the vector
// unit busy 1.34 us [1]: that leaves ~0.36 us for a load, its compute and a store, and the best
// known C1-C5 times leave no more than ~0.1 us per latency. (The 800 ns used before was inferred
// from targets of an earlier benchmark; with the measured launch it would put C1 at 3.4 us and C5
// at 7.9 us or more, against 1.70 and 5.20 us measured.)
static constexpr double   DMA_LATENCY_NS        = 100.0;
static constexpr double   DMA_LATENCY_CYCLES    = DMA_LATENCY_NS * CLOCK_GHZ;
static constexpr double   LOCAL_BYTES_PER_CYCLE = SIMD_REPEAT_BYTES;
static constexpr uint32_t SYNC_ALL_CYCLES       = static_cast<uint32_t>(SYNC_ALL_NS * CLOCK_GHZ + 0.5);
static constexpr uint32_t PIPE_ALL_CYCLES       = static_cast<uint32_t>(PIPE_ALL_NS * CLOCK_GHZ + 0.5);
static constexpr uint32_t GET_VALUE_CYCLES      = static_cast<uint32_t>(GET_VALUE_NS * CLOCK_GHZ + 0.5);

// -----------------------------------------------------------------------------
// System memory [3-5]. The P cores of a launch share an aggregate bandwidth
//     bw(P, b) = min(regime(working set), ceiling(b)) * min(1, P / DMA_SATURATION_CORES)
// for transfers of b bytes each: below 27 cores every core's DMA issue rate is the limit (bw / 27
// per core), from 27 cores on the memory system (bw / P per core) [4]. The transfer-size ceiling [3]
// and the working-set regime [5] combine by min: an efficiency factor multiplied onto one rate
// cannot reproduce shapes that stop improving above 4 KB transfers. Both tables are interpolated
// linearly (the working set on a log scale) and held flat outside the measured range.
// -----------------------------------------------------------------------------
struct BandwidthPoint { double bytes, gbs; };
static constexpr BandwidthPoint DMA_SIZE_CEILING[] = {{1024, 277}, {2048, 475}, {4096, 677}, {6144, 1090}};  // [3]
static constexpr double MIB = 1024.0 * 1024.0;
static constexpr BandwidthPoint DMA_REGIME_CEILING[] = {                                                      // [5]
    {30 * MIB, 1945}, {48 * MIB, 2020}, {80 * MIB, 2046}, {132 * MIB, 1408}, {368 * MIB, 1252}, {2560 * MIB, 981}, {5632 * MIB, 943}};
static constexpr double DMA_SATURATION_CORES = 27.0;                                                          // [4]

template <size_t N>
inline double InterpolateBandwidth(const BandwidthPoint (&t)[N], double x, bool logScale) {
    if (x <= t[0].bytes) return t[0].gbs;
    for (size_t i = 1; i < N; ++i) {
        if (x > t[i].bytes) continue;
        const double a = logScale ? std::log(t[i - 1].bytes) : t[i - 1].bytes, b = logScale ? std::log(t[i].bytes) : t[i].bytes;
        return t[i - 1].gbs + ((logScale ? std::log(x) : x) - a) / (b - a) * (t[i].gbs - t[i - 1].gbs);
    }
    return t[N - 1].gbs;
}

class MemorySystem {
public:
    MemorySystem() : MemorySystem(MAX_HARDWARE_CORES, 0.0) {}
    // A launch of `cores` cores that touches `workingSet` bytes of system memory
    MemorySystem(uint32_t cores, double workingSet)
        : cores_(cores ? cores : 1u), workingSet_(workingSet), regime_(InterpolateBandwidth(DMA_REGIME_CEILING, workingSet, true)) {}

    uint32_t Cores() const { return cores_; }
    double WorkingSet() const { return workingSet_; }
    // Aggregate GB/s of the launch's cores, all streaming transfers of `bytes`
    double AggregateGBs(double bytes) const {
        const double ceiling = InterpolateBandwidth(DMA_SIZE_CEILING, bytes, false);
        const double share = cores_ < DMA_SATURATION_CORES ? cores_ / DMA_SATURATION_CORES : 1.0;
        return (regime_ < ceiling ? regime_ : ceiling) * share;
    }
    // Channel cycles of one transfer of `bytes` on one core: its bytes at the core's share
    double Cycles(double bytes) const { return bytes * cores_ / AggregateGBs(bytes) * CLOCK_GHZ; }

private:
    uint32_t cores_;
    double workingSet_, regime_;
};

// The memory system of the launch in progress: the coordinator sets it before the cores start
inline MemorySystem g_memory;

// One core's timeline, summarized. The critical unit is the busier of VECTOR and DMA; its
// idle time splits into fill (before its first operation), drain (after its last), barrier
// (inside SyncAll) and mismatch (everything else), so
//   finish = max(vectorBusy, dmaBusy) + fill + drain + barrier + mismatch.
struct TimelineSummary {
    double finish = 0;      // Every unit idle and every transfer landed
    double vectorBusy = 0;  // Vector instructions (scalar stalls included)
    double dmaBusy = 0;     // System-memory channel occupancy
    double loadBusy = 0;    // ... of which loads
    double barrier = 0;     // Critical unit idle inside SyncAll (arrival to release)
    double syncCycles = 0;  // SYNC_ALL_CYCLES per SyncAll: the barrier's own cost
    double fill = 0, drain = 0, mismatch = 0;
    double floor = 0;       // Latency floor (CoreTimeline): no run of this program finishes sooner
    bool dmaBound = false;  // The critical unit is DMA

    // The vector unit idles through every SyncAll; loads may keep streaming through one
    double LowerBound() const { return vectorBusy + syncCycles > dmaBusy ? vectorBusy + syncCycles : dmaBusy; }
    double Bubble() const { return finish - LowerBound(); }
    // The DMA latency and dependencies of the program: no run finishes sooner
    double LatencyFloor() const { return floor; }
};

struct BlockSpan { uint32_t b0 = 0, b1 = 0; };  // Scratchpad blocks [b0, b1); empty: not scratchpad

class CoreTimeline {
public:
    enum Unit : uint32_t { VECTOR = 0, DMA = 1, LOCAL = 2, UNITS = 3 };
    using Span = BlockSpan;

    static constexpr uint32_t BLOCKS = SCRATCHPAD_MAX_BYTES / DMA_ALIGN_BYTES;
    static constexpr uint32_t MAX_REGIONS = 64, MAX_BARRIERS = 16;

    void Reset() {
        nRegions = nextBlock = nBarriers = syncs = nRecent = 0;
        loadsDone = storesDone = loadBusy = floorLoads = floorStores = floorStoreFree = 0;
        for (uint32_t u = 0; u < UNITS; ++u) unitFree[u] = busy[u] = first[u] = last[u] = floorFree[u] = 0, used[u] = false;
        vecEpoch = scalarEpoch = vsFence = svFence = 0;
    }

    // A scratchpad buffer the pipe has carved: blocks are numbered in claim order
    void Register(const void* base, size_t bytes) {
        const uint32_t n = static_cast<uint32_t>((bytes + DMA_ALIGN_BYTES - 1) / DMA_ALIGN_BYTES);
        if (nRegions == MAX_REGIONS || nextBlock + n > BLOCKS) return;  // Untracked: no ordering
        const uintptr_t a = reinterpret_cast<uintptr_t>(base);
        regions[nRegions++] = {a, a + bytes, nextBlock};
        for (uint32_t b = nextBlock; b < nextBlock + n; ++b) ready[b] = lastRead[b] = floorReady[b] = 0, vecWrote[b] = scalarWrote[b] = 0;
        nextBlock += n;
    }

    Span SpanOf(const void* p, size_t bytes) const {
        const uintptr_t a = reinterpret_cast<uintptr_t>(p);
        for (uint32_t i = 0; bytes && i < nRegions; ++i) {
            const Region& r = regions[i];
            if (a < r.base || a >= r.end) continue;
            const uintptr_t off = a - r.base, end = off + bytes < r.end - r.base ? off + bytes : r.end - r.base;
            return {r.block0 + static_cast<uint32_t>(off / DMA_ALIGN_BYTES),
                    r.block0 + static_cast<uint32_t>((end + DMA_ALIGN_BYTES - 1) / DMA_ALIGN_BYTES)};
        }
        return {};
    }

    // A vector instruction writing `w` and reading `r0`, `r1`. It sees the scalar unit's stores only
    // after an S->V fence [6.5].
    void Vector(double cycles, Span w, Span r0 = {}, Span r1 = {}) {
        CheckVisible(r0, scalarWrote, svFence, "a vector instruction reads a block the scalar unit wrote after the last S->V fence");
        CheckVisible(r1, scalarWrote, svFence, "a vector instruction reads a block the scalar unit wrote after the last S->V fence");
        ++vecEpoch;
        for (uint32_t b = w.b0; b < w.b1; ++b) vecWrote[b] = vecEpoch;
        InPipe(cycles, w, r0, r1);
    }
    // The scalar unit reads `r` (GetValue): in the vector pipe's order, for `cycles`, and only what a
    // V->S fence made visible [6.5]
    void ScalarRead(double cycles, Span r) {
        CheckVisible(r, vecWrote, vsFence, "GetValue reads a block the vector unit wrote after the last V->S fence "
                                           "(CrossPipe<V_S> or PipeBarrier<PIPE_ALL>; PipeBarrier<PIPE_V> is not one)");
        InPipe(cycles, {}, r, {});
    }
    // The scalar unit writes `w` (SetValue), in the vector pipe's order; the vector unit sees it after
    // an S->V fence
    void ScalarWrite(Span w) {
        ++scalarEpoch;
        for (uint32_t b = w.b0; b < w.b1; ++b) scalarWrote[b] = scalarEpoch;
        InPipe(0, w, {}, {});
    }
    // Fences: everything the vector unit (scalar unit) wrote so far is visible to the other one
    void FenceVS() { vsFence = vecEpoch; }
    void FenceSV() { svFence = scalarEpoch; }
    // System memory -> scratchpad
    void Load(double bytes, Span w) {
        const double s = Start(DMA, w, {}, {}), occupied = g_memory.Cycles(bytes);
        Occupy(DMA, s, occupied);
        loadBusy += occupied;
        const double landed = s + occupied + DMA_LATENCY_CYCLES;
        Write(w, landed);
        if (landed > loadsDone) loadsDone = landed;
        floorFree[DMA] += occupied;  // The floor's load queue
        const double fl = floorFree[DMA] + DMA_LATENCY_CYCLES;
        FloorWrite(w, fl);
        floorLoads = fl > floorLoads ? fl : floorLoads;
    }
    // Scratchpad -> system memory: the source is free once streamed out
    void Store(double bytes, Span r) {
        const double s = Start(DMA, {}, r, {}), occupied = g_memory.Cycles(bytes);
        Occupy(DMA, s, occupied);
        Read(r, s + occupied);
        if (s + occupied + DMA_LATENCY_CYCLES > storesDone) storesDone = s + occupied + DMA_LATENCY_CYCLES;
        double fs = floorStoreFree;  // The floor's store queue
        for (uint32_t b = r.b0; b < r.b1; ++b) fs = floorReady[b] > fs ? floorReady[b] : fs;
        floorStoreFree = fs + occupied;
        fs = floorStoreFree + DMA_LATENCY_CYCLES;
        floorStores = fs > floorStores ? fs : floorStores;
    }
    // Scratchpad -> scratchpad
    void Copy(double bytes, Span w, Span r) {
        const double s = Start(LOCAL, w, r, {}), d = bytes / LOCAL_BYTES_PER_CYCLE;
        Occupy(LOCAL, s, d);
        Read(r, s + d);
        Write(w, s + d);
        floorFree[LOCAL] = FloorStart(LOCAL, r, {}) + d;
        FloorWrite(w, floorFree[LOCAL]);
    }

    // Every unit idle and every transfer landed
    double Drained() const {
        double t = loadsDone > storesDone ? loadsDone : storesDone;
        for (uint32_t u = 0; u < UNITS; ++u) t = unitFree[u] > t ? unitFree[u] : t;
        return t;
    }
    // Ready for SyncAll: vector and local units idle, every store landed
    double Arrival() const {
        double t = storesDone > unitFree[VECTOR] ? storesDone : unitFree[VECTOR];
        return unitFree[LOCAL] > t ? unitFree[LOCAL] : t;
    }
    uint32_t Syncs() const { return syncs; }
    // A SyncAll: arrived at `arrive`, released at `release`; loads in flight keep streaming
    void Barrier(double arrive, double release) {
        if (nBarriers < MAX_BARRIERS) {
            double streaming = 0;  // Channel time inside the barrier (the latest transfers only)
            for (uint32_t i = 0; i < nRecent; ++i) {
                const double s = recentFrom[i] > arrive ? recentFrom[i] : arrive, e = recentTo[i] < release ? recentTo[i] : release;
                streaming += e > s ? e - s : 0;
            }
            from[nBarriers] = arrive, to[nBarriers] = release, dmaInside[nBarriers] = streaming, ++nBarriers;
        }
        for (uint32_t u = 0; u < UNITS; ++u) unitFree[u] = unitFree[u] > release ? unitFree[u] : release;
        ++syncs;
        double fa = floorStores > floorFree[VECTOR] ? floorStores : floorFree[VECTOR];  // Its own arrival, without buffer reuse
        fa = floorFree[LOCAL] > fa ? floorFree[LOCAL] : fa;
        for (uint32_t u = 0; u < UNITS; ++u) floorFree[u] = floorFree[u] > fa + SYNC_ALL_CYCLES ? floorFree[u] : fa + SYNC_ALL_CYCLES;
        floorStoreFree = floorStoreFree > fa + SYNC_ALL_CYCLES ? floorStoreFree : fa + SYNC_ALL_CYCLES;
    }
    // Every unit waits until the core is drained, then `cycles` more (PipeBarrier<PIPE_ALL>)
    void DrainAll(double cycles = 0) {
        const double t = Drained() + cycles, f = FloorDrained() + cycles;
        for (uint32_t u = 0; u < UNITS; ++u) unitFree[u] = t, floorFree[u] = f;
        floorStoreFree = f;
    }

    TimelineSummary Summary() const {
        TimelineSummary s;
        s.finish = Drained();
        s.vectorBusy = busy[VECTOR];
        s.dmaBusy = busy[DMA];
        s.loadBusy = loadBusy;
        s.syncCycles = static_cast<double>(syncs) * SYNC_ALL_CYCLES;
        s.floor = FloorDrained();
        if (busy[DMA] > 0 && busy[DMA] + DMA_LATENCY_CYCLES > s.floor) s.floor = busy[DMA] + DMA_LATENCY_CYCLES;
        s.dmaBound = busy[DMA] > busy[VECTOR];
        const Unit u = s.dmaBound ? DMA : VECTOR;
        const double begin = used[u] ? first[u] : s.finish, end = used[u] ? last[u] : s.finish;
        double before = 0, after = 0, idle = 0;  // The critical unit's idle time inside barriers
        for (uint32_t i = 0; i < nBarriers; ++i) {
            const double window = to[i] - from[i], quiet = window - (u == DMA ? dmaInside[i] : 0);
            s.barrier += window;
            idle += quiet;
            if (to[i] <= begin) before += quiet;
            else if (from[i] >= end) after += quiet;
        }
        s.fill = begin - before;
        s.drain = s.finish - end - after;
        s.mismatch = s.finish - busy[u] - s.fill - s.drain - idle;
        s.barrier = idle;
        return s;
    }

private:
    struct Region { uintptr_t base, end; uint32_t block0; };

    // A vector-pipe operation: timing only
    void InPipe(double cycles, Span w, Span r0, Span r1) {
        const double s = Start(VECTOR, w, r0, r1), e = s + cycles;
        Occupy(VECTOR, s, cycles);
        Read(r0, e);
        Read(r1, e);
        Write(w, e);
        const double fe = FloorStart(VECTOR, r0, r1) + cycles;
        floorFree[VECTOR] = fe;
        FloorWrite(w, fe);
    }
    static void CheckVisible(Span r, const uint32_t* wrote, uint32_t fence, const char* what) {
        for (uint32_t b = r.b0; b < r.b1; ++b) {
            if (wrote[b] > fence) throw std::runtime_error(std::string("[Sanitizer Trap - PIPE VISIBILITY]: ") + what);
        }
    }

    // The floor's replay: the unit and the operands' data only
    double FloorStart(Unit u, Span r0, Span r1) const {
        double t = floorFree[u];
        for (uint32_t b = r0.b0; b < r0.b1; ++b) t = floorReady[b] > t ? floorReady[b] : t;
        for (uint32_t b = r1.b0; b < r1.b1; ++b) t = floorReady[b] > t ? floorReady[b] : t;
        return t;
    }
    void FloorWrite(Span w, double t) {
        for (uint32_t b = w.b0; b < w.b1; ++b) floorReady[b] = t;
    }
    double FloorDrained() const {
        double t = floorLoads > floorStores ? floorLoads : floorStores;
        for (uint32_t u = 0; u < UNITS; ++u) t = floorFree[u] > t ? floorFree[u] : t;
        return floorStoreFree > t ? floorStoreFree : t;
    }

    double Start(Unit u, Span w, Span r0, Span r1) const {
        double t = unitFree[u];
        for (uint32_t b = r0.b0; b < r0.b1; ++b) t = ready[b] > t ? ready[b] : t;
        for (uint32_t b = r1.b0; b < r1.b1; ++b) t = ready[b] > t ? ready[b] : t;
        for (uint32_t b = w.b0; b < w.b1; ++b) {
            t = ready[b] > t ? ready[b] : t;
            t = lastRead[b] > t ? lastRead[b] : t;
        }
        return t;
    }
    void Occupy(Unit u, double start, double cycles) {
        if (!used[u]) first[u] = start, used[u] = true;
        busy[u] += cycles;
        unitFree[u] = start + cycles;
        if (unitFree[u] > last[u]) last[u] = unitFree[u];
        if (u == DMA) {  // The latest channel intervals: what may still stream inside a SyncAll
            const uint32_t i = nRecent < RECENT ? nRecent++ : (DropOldest(), RECENT - 1);
            recentFrom[i] = start, recentTo[i] = start + cycles;
        }
    }
    void DropOldest() {
        for (uint32_t i = 1; i < RECENT; ++i) recentFrom[i - 1] = recentFrom[i], recentTo[i - 1] = recentTo[i];
    }
    void Read(Span r, double t) {
        for (uint32_t b = r.b0; b < r.b1; ++b) lastRead[b] = t > lastRead[b] ? t : lastRead[b];
    }
    void Write(Span w, double t) {
        for (uint32_t b = w.b0; b < w.b1; ++b) ready[b] = t;
    }

    Region regions[MAX_REGIONS] = {};
    uint32_t nRegions = 0, nextBlock = 0, nBarriers = 0, syncs = 0;
    double ready[BLOCKS] = {};     // When each block's last write lands
    double lastRead[BLOCKS] = {};  // When each block's last read ends
    double floorReady[BLOCKS] = {};  // The floor's replay: when each block's last write lands
    double unitFree[UNITS] = {}, busy[UNITS] = {}, first[UNITS] = {}, last[UNITS] = {};
    bool used[UNITS] = {};
    double loadsDone = 0, storesDone = 0, loadBusy = 0;
    double floorFree[UNITS] = {}, floorStoreFree = 0, floorLoads = 0, floorStores = 0;  // floorFree[DMA]: the load queue
    double from[MAX_BARRIERS] = {}, to[MAX_BARRIERS] = {}, dmaInside[MAX_BARRIERS] = {};
    static constexpr uint32_t RECENT = 16;
    double recentFrom[RECENT] = {}, recentTo[RECENT] = {};
    uint32_t nRecent = 0;
    // Scalar <-> vector visibility [6.5]: per block, the vector (scalar) write that last touched it;
    // a fence makes every write up to its epoch visible to the other unit
    uint32_t vecWrote[BLOCKS] = {}, scalarWrote[BLOCKS] = {};
    uint32_t vecEpoch = 0, scalarEpoch = 0, vsFence = 0, svFence = 0;
};

inline thread_local CoreTimeline g_timeline;
inline double g_syncArrival[2][256];  // SyncAll arrival times by core, two phases in turn

template <typename T>
inline CoreTimeline::Span SpanOf(const T* p, uint64_t count) {
    return g_timeline.SpanOf(p, count * sizeof(T));
}

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
    PIPE_ALL = 3,
    PIPE_MTE2 = 1,
    PIPE_MTE3 = 2,
    PIPE_S = 4
};

// PIPE_ALL drains every pipe of the core, DMA included, then costs PIPE_ALL_NS [1], and makes each
// unit's writes visible to the others. A barrier on one pipe costs <= 0.19 ns on the target [6.5],
// nothing on this timeline, and makes nothing visible: PipeBarrier<PIPE_V> is not a V->S fence.
template <PipeType pipe>
inline void PipeBarrier() {
    #if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("" ::: "memory");
    #endif
    if (pipe == PIPE_ALL) {
        g_cycleTracker.barrierCycles += PIPE_ALL_CYCLES;
        g_timeline.DrainAll(PIPE_ALL_CYCLES);
        g_timeline.FenceVS();
        g_timeline.FenceSV();
    }
}

inline void pipe_barrier(PipeType pipe) {
    if (pipe == PIPE_ALL) {
        PipeBarrier<PIPE_ALL>();
    } else if (pipe == PIPE_V) {
        PipeBarrier<PIPE_V>();
    } else if (pipe == PIPE_DMA_IN) {
        PipeBarrier<PIPE_DMA_IN>();
    } else if (pipe == PIPE_DMA_OUT) {
        PipeBarrier<PIPE_DMA_OUT>();
    }
}

inline void set_flag(PipeType src, PipeType dst, uint8_t eventId = 0) {
    (void)src; (void)dst; (void)eventId;
    #if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("" ::: "memory");
    #endif
}

inline void wait_flag(PipeType src, PipeType dst, uint8_t eventId = 0) {
    (void)src; (void)dst; (void)eventId;
    #if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("" ::: "memory");
    #endif
}

// -----------------------------------------------------------------------------
// DAE v1.5 Point-to-Point Pipeline Scoreboard Fences (HardEvent Scoreboard Flags)
// Avoids PIPE_ALL flushes by synchronizing only the dependent execution units.
// -----------------------------------------------------------------------------
enum class HardEvent : uint8_t {
    MTE2_V = 0,   // DMU Ingress -> VPU (Data streamed to scratchpad)
    V_MTE3 = 1,   // VPU -> DMU Egress (Vector compute finished, ready to stream out)
    MTE3_S = 2,   // DMU Egress -> SPU (Egress committed to system memory)
    V_S    = 3,   // VPU -> SPU (Vector result needed by scalar unit)
    S_V    = 4,   // SPU -> VPU (Scalar control ready for vector pipeline)
    MTE2_S = 5,   // DMU Ingress -> SPU
    S_MTE2 = 6,   // SPU -> DMU Ingress
    MTE3_V = 7,   // DMU Egress -> VPU
    V_MTE2 = 8    // VPU -> DMU Ingress
};

static constexpr uint8_t EVENT_ID0 = 0;
static constexpr uint8_t EVENT_ID1 = 1;
static constexpr uint8_t EVENT_ID2 = 2;
static constexpr uint8_t EVENT_ID3 = 3;

template <HardEvent E>
inline void SetFlag(uint8_t eventId = EVENT_ID0) {
    (void)eventId;
    #if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("" ::: "memory");
    #endif
}

// Waiting on V_S (S_V) is the fence that makes the vector (scalar) unit's writes visible to the
// scalar (vector) unit [6.5]
template <HardEvent E>
inline void WaitFlag(uint8_t eventId = EVENT_ID0) {
    (void)eventId;
    #if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("" ::: "memory");
    #endif
    if constexpr (E == HardEvent::V_S) g_timeline.FenceVS();
    if constexpr (E == HardEvent::S_V) g_timeline.FenceSV();
}

template <HardEvent E>
inline void CrossPipe(uint8_t eventId = EVENT_ID0) {
    SetFlag<E>(eventId);
    WaitFlag<E>(eventId);
}

// All-core barrier. Timeline: each core arrives drained; all leave SYNC_ALL_CYCLES after the
// last arrival.
template <bool notify = false>
inline void SyncAll() {
    const uint32_t core = static_cast<uint32_t>(omp_get_thread_num()), cores = static_cast<uint32_t>(omp_get_num_threads());
    const double arrive = g_timeline.Arrival();
    double* arrivals = g_syncArrival[g_timeline.Syncs() & 1];
    if (core < 256) arrivals[core] = arrive;
    #pragma omp barrier
    double last = arrive;
    for (uint32_t c = 0; c < cores && c < 256; ++c) last = arrivals[c] > last ? arrivals[c] : last;
    g_timeline.Barrier(arrive, last + SYNC_ALL_CYCLES);
    g_cycleTracker.barrierCount++;
    g_cycleTracker.barrierCycles += SYNC_ALL_CYCLES;
}

// Simulated symmetric streaming core index (0 <= idx < num_cores)
inline uint32_t GetCoreIdx() {
    return static_cast<uint32_t>(omp_get_thread_num());
}

inline uint32_t GetCoreNum() {
    return static_cast<uint32_t>(omp_get_num_threads());
}

inline uint32_t GetBlockIdx() {
    return GetCoreIdx();
}

inline uint32_t GetBlockNum() {
    return GetCoreNum();
}

inline int64_t get_ctrl() {
    return 0;
}

inline void set_ctrl(int64_t ctrl) {
    (void)ctrl;
}

// Standard OpenMP-style CPU thread indexing
inline uint32_t GetThreadIdx() {
    return static_cast<uint32_t>(omp_get_thread_num());
}

inline uint32_t GetThreadNum() {
    return static_cast<uint32_t>(omp_get_num_threads());
}

// Freestanding mathematical primitives (Zero libc/STL dependencies for device workers)
template <typename T>
constexpr const T& Min(const T& a, const T& b) {
    return (b < a) ? b : a;
}

template <typename T>
constexpr const T& Max(const T& a, const T& b) {
    return (a < b) ? b : a;
}

// -----------------------------------------------------------------------------
// LocalTensor<T> Implementation for Core-Local Scratchpad. Its position is set by the buffer that
// hands it out (TBuf<POS>, TQue<POS>) and read with GetPosition(): the target's LocalTensor has no
// public `pos` member [6.4].
// -----------------------------------------------------------------------------
template <typename T>
class LocalTensor {
    QuePosition pos = QuePosition::TOTAL_POSITIONS;

public:
    T* data = nullptr;
    uint32_t count = 0;
    uint32_t capacityBytes = 0;

    LocalTensor() = default;
    LocalTensor(T* ptr, uint32_t numElems, uint32_t capBytes, QuePosition p = QuePosition::TOTAL_POSITIONS)
        : pos(p), data(ptr), count(numElems), capacityBytes(capBytes) {}

    // Sub-tensor slicing operator: tensor[offset] returns sliced sub-tensor view
    inline LocalTensor<T> operator[](uint32_t offset) const {
        return LocalTensor<T>(data + offset, (count > offset) ? (count - offset) : 0, 
                              (capacityBytes > offset * sizeof(T)) ? (capacityBytes - offset * sizeof(T)) : 0, pos);
    }

    inline LocalTensor<T> operator+(uint32_t offset) const {
        return LocalTensor<T>(data + offset, (count > offset) ? (count - offset) : 0, 
                              (capacityBytes > offset * sizeof(T)) ? (capacityBytes - offset * sizeof(T)) : 0, pos);
    }

    // Scalar read [1]: GET_VALUE_NS once the value's producer has finished (the scalar unit issues
    // in order, so what follows waits too), with V->S telemetry. The value must have been made
    // visible by a V->S fence [6.5].
    inline T GetValue(uint32_t index) const {
        g_cycleTracker.scalarStallCycles += GET_VALUE_CYCLES;
        g_cycleTracker.scalarStallCount++;
        g_timeline.ScalarRead(GET_VALUE_CYCLES, SpanOf(data + index, 1));
        return data[index];
    }

    // The scalar unit writes in the vector pipe's order (timeline: no cost); the vector unit sees it
    // after an S->V fence [6.5]
    inline void SetValue(uint32_t index, T val) {
        g_timeline.ScalarWrite(SpanOf(data + index, 1));
        data[index] = val;
    }

    inline T* GetData() { return data; }
    inline const T* GetData() const { return data; }
    inline uint32_t GetSize() const { return count; }
    inline QuePosition GetPosition() const { return pos; }

    template <typename U>
    inline LocalTensor<U> ReinterpretCast() const {
        return LocalTensor<U>(reinterpret_cast<U*>(data),
                              (sizeof(U) > 0) ? (capacityBytes / sizeof(U)) : 0,
                              capacityBytes, pos);
    }
};

// -----------------------------------------------------------------------------
// TBuf<QuePosition> Implementation (Scratchpad Buffer)
// -----------------------------------------------------------------------------
template <QuePosition pos = QuePosition::VECCALC>
class TBuf {
public:
    std::vector<uint8_t> bufferPool;
    size_t elementBytes = 0;

    // Scratchpad holds whatever it held before: a new buffer reads as NaN (0xFF bytes) until written,
    // so a kernel that consumes what it never wrote (a reduction slot's stale lanes) fails here too
    void Configure(size_t tensorBytes) {
        elementBytes = tensorBytes;
        bufferPool.assign(tensorBytes + 64, SCRATCHPAD_POISON);
        g_timeline.Register(Get<uint8_t>().GetData(), tensorBytes);
    }

    template <typename T>
    LocalTensor<T> Get() {
        uintptr_t raw = reinterpret_cast<uintptr_t>(bufferPool.data());
        uint8_t* base = reinterpret_cast<uint8_t*>((raw + 63) & ~uintptr_t(63));
        return LocalTensor<T>(reinterpret_cast<T*>(base), static_cast<uint32_t>(elementBytes / sizeof(T)),
                              static_cast<uint32_t>(elementBytes), pos);
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
    // never overwrite a tile that is still in flight (async queue hazard guard).
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
            bufferPool[d].assign(d < buffers ? tensorBytes + 64 : 0, SCRATCHPAD_POISON);  // Unwritten: NaN
            state[d] = FREE;
            if (d < buffers) g_timeline.Register(Base(d), tensorBytes);
        }
        head = 0;
        enqueuedCount = 0;
        allocatedCount = 0;
    }

    template <typename T>
    LocalTensor<T> AllocTensor() {
        g_cycleTracker.queueSequencerCycles += 625;
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
        g_cycleTracker.queueSequencerCycles += 625;
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
        g_cycleTracker.queueSequencerCycles += 625;
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
        g_cycleTracker.queueSequencerCycles += 625;
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
                              static_cast<uint32_t>(elementBytes), pos);
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
    TPipe() { g_timeline.Reset(); }  // A new pipe is a new kernel on this core: its timeline starts at 0

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
        // Strictly inspect that total core-local scratchpad buffer stays <= 191 KB!
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
// Scratchpad allocation [6.4]
// -----------------------------------------------------------------------------
namespace Hardware {
    struct Scratchpad {};
    using SPM = Scratchpad;
    using UB = Scratchpad; // Unified Scratchpad Buffer
}

// [NOT ON THE TARGET] LocalMemAllocator does not exist. The scratchpad is
// claimed with TPipe::InitBuffer(TBuf&, bytes) and sliced by the kernel via
// TBuf::Get<T>(). Kept only so existing code still names something; every
// member is a hard error, because an allocator that hands out arbitrary
// alignments cannot model a machine whose DMA end requires 32-byte starts.
template <typename TargetSpace = Hardware::UB>
class LocalMemAllocator {
    static_assert(sizeof(TargetSpace) == 0,
                  "[NOT ON THE TARGET] use TPipe::InitBuffer(TBuf&, bytes) and TBuf::Get<T>(), and round every segment start up to 32 bytes.");
};

// -----------------------------------------------------------------------------
// Streaming DataCopy Primitives with Integrated 32-Byte DMA Alignment Guard
// -----------------------------------------------------------------------------
// Both endpoints of a block DMA must sit on a 32-byte block boundary
inline void CheckDmaAddress(const void* systemMem, const void* local) {
    if (reinterpret_cast<uintptr_t>(systemMem) % DMA_ALIGN_BYTES != 0 ||
        reinterpret_cast<uintptr_t>(local) % DMA_ALIGN_BYTES != 0) {
        throw std::runtime_error("[Hardware Fault - DMA UNALIGNED]: Transfer address is not 32-byte aligned!");
    }
}

// -----------------------------------------------------------------------------
// Strided & Padded DMA Parameter Descriptors
// -----------------------------------------------------------------------------
struct DataCopyExtParams {
    uint16_t blockCount = 1;
    uint32_t blockLen = 0; // Length in bytes
    uint16_t srcStride = 0;
    uint16_t dstStride = 0;
    uint32_t rsv = 0;
};

template <typename T>
struct DataCopyPadExtParams {
    bool isPad = false;
    uint8_t leftPadding = 0;
    uint8_t rightPadding = 0;
    T paddingValue = T(0);
};

struct UnaryRepeatParams {
    uint8_t dstRepStride = 1;
    uint8_t srcRepStride = 1;
    uint8_t dstBlkStride = 8;
    uint8_t srcBlkStride = 8;
};

enum class RoundMode {
    CAST_NONE = 0,
    CAST_RINT = 1,
    CAST_FLOOR = 2,
    CAST_CEIL = 3,
    CAST_TRUNC = 4
};

// -----------------------------------------------------------------------------
// The simulator's copy engine, on the system-memory address a GlobalTensor holds. Kernels reach it
// only through the GlobalTensor forms below: the target has no raw-pointer DMA.
//
// Padded DMA (DataCopyPad): transfers that are not whole 32-byte blocks, such as row tails and rows
// whose width is off the 32-byte grid. System memory may be at any address and length; the
// scratchpad side is block aligned [6.6]: each of a descriptor's `blockCount` rows occupies whole
// 32-byte blocks there, the next row starting `dstStride` (load) / `srcStride` (store) blocks after
// the previous row's last block. In system memory the rows are `blockLen` bytes long and follow each
// other after `srcStride` (load) / `dstStride` (store) bytes. A load fills the rest of each row's last
// block with paddingValue when isPad is set and leaves it as it was otherwise (undefined on the
// target); a store writes the rows' bytes only. The engine moves whole blocks, so the traffic is
// rounded up, and one descriptor is one transfer.
// -----------------------------------------------------------------------------
namespace detail {
template <typename T>
inline void BlockLoad(LocalTensor<T> dst, const T* src, uint32_t count) {
    const size_t copyBytes = size_t(count) * sizeof(T);
    if (copyBytes % DMA_ALIGN_BYTES != 0) {
        throw std::runtime_error("[Hardware Fault - DMA UNALIGNED]: Transfer size (" + std::to_string(copyBytes) +
                                 " bytes) is not a multiple of 32 bytes!");
    }
    CheckDmaAddress(src, dst.GetData());
    if (copyBytes > dst.capacityBytes) throw std::runtime_error("[Hardware Fault - DMA OVERRUN]: DataCopy runs past its scratchpad buffer!");
    std::memcpy(dst.GetData(), src, copyBytes);
    g_cycleTracker.dmaBytesMoved += copyBytes;
    g_cycleTracker.dmaTransfers++;
    g_timeline.Load(static_cast<double>(copyBytes), SpanOf(dst.GetData(), count));
}

template <typename T>
inline void BlockStore(T* dst, LocalTensor<T> src, uint32_t count) {
    const size_t copyBytes = size_t(count) * sizeof(T);
    if (copyBytes % DMA_ALIGN_BYTES != 0) {
        throw std::runtime_error("[Hardware Fault - DMA UNALIGNED]: Transfer size (" + std::to_string(copyBytes) +
                                 " bytes) is not a multiple of 32 bytes!");
    }
    CheckDmaAddress(dst, src.GetData());
    // Egress routes VECOUT -> system memory only (PIPE_DMA_OUT): egress from VECIN deadlocks the crossbar
    if (src.GetPosition() == QuePosition::VECIN) {
        throw std::runtime_error("[Hardware Fault - INVALID DMA EGRESS CHANNEL]: "
                                 "DataCopy egress to system memory attempted from QuePosition::VECIN! "
                                 "The stream processor features asymmetric DMA routing: Ingress streams "
                                 "system memory -> VECIN (PIPE_DMA_IN), while Egress strictly routes "
                                 "VECOUT -> system memory (PIPE_DMA_OUT). Egress via VECIN causes "
                                 "crossbar interconnect deadlock and pipeline timeout (Trap #401).");
    }
    std::memcpy(dst, src.GetData(), copyBytes);
    g_cycleTracker.dmaBytesMoved += copyBytes;
    g_cycleTracker.dmaTransfers++;
    g_timeline.Store(static_cast<double>(copyBytes), SpanOf(src.GetData(), count));
}

template <typename T>
inline void PadLoad(LocalTensor<T> dst, const T* src, DataCopyExtParams cp, DataCopyPadExtParams<T> pad) {
    const size_t row = (size_t(cp.blockLen) + DMA_ALIGN_BYTES - 1) / DMA_ALIGN_BYTES * DMA_ALIGN_BYTES;
    const size_t pitch = row + size_t(cp.dstStride) * DMA_ALIGN_BYTES, extent = cp.blockCount ? (cp.blockCount - 1) * pitch + row : 0;
    if (reinterpret_cast<uintptr_t>(dst.GetData()) % DMA_ALIGN_BYTES != 0 || extent > dst.capacityBytes) {
        throw std::runtime_error("[Hardware Fault - DMA UNALIGNED]: DataCopyPad scratchpad side must be 32-byte aligned and in bounds!");
    }
    uint8_t* d = reinterpret_cast<uint8_t*>(dst.GetData());
    const uint8_t* s = reinterpret_cast<const uint8_t*>(src);
    for (uint32_t b = 0; b < cp.blockCount; ++b) {
        std::memcpy(d + b * pitch, s + size_t(b) * (cp.blockLen + cp.srcStride), cp.blockLen);
        if (!pad.isPad) continue;
        for (size_t o = cp.blockLen; o + sizeof(T) <= row; o += sizeof(T)) std::memcpy(d + b * pitch + o, &pad.paddingValue, sizeof(T));
    }
    g_cycleTracker.dmaBytesMoved += cp.blockCount * row;
    g_cycleTracker.dmaTransfers++;
    g_cycleTracker.padTransfers++;
    g_timeline.Load(static_cast<double>(cp.blockCount * row), g_timeline.SpanOf(dst.GetData(), extent));
}

template <typename T>
inline void PadStore(T* dst, LocalTensor<T> src, DataCopyExtParams cp) {
    const size_t row = (size_t(cp.blockLen) + DMA_ALIGN_BYTES - 1) / DMA_ALIGN_BYTES * DMA_ALIGN_BYTES;
    const size_t pitch = row + size_t(cp.srcStride) * DMA_ALIGN_BYTES, extent = cp.blockCount ? (cp.blockCount - 1) * pitch + row : 0;
    if (reinterpret_cast<uintptr_t>(src.GetData()) % DMA_ALIGN_BYTES != 0 || extent > src.capacityBytes) {
        throw std::runtime_error("[Hardware Fault - DMA UNALIGNED]: DataCopyPad scratchpad side must be 32-byte aligned and in bounds!");
    }
    if (src.GetPosition() == QuePosition::VECIN) {
        throw std::runtime_error("[Hardware Fault - INVALID DMA EGRESS CHANNEL]: "
                                 "DataCopyPad egress to system memory attempted from QuePosition::VECIN! "
                                 "Egress transfers strictly require QuePosition::VECOUT (Trap #401).");
    }
    uint8_t* d = reinterpret_cast<uint8_t*>(dst);
    const uint8_t* s = reinterpret_cast<const uint8_t*>(src.GetData());
    for (uint32_t b = 0; b < cp.blockCount; ++b) std::memcpy(d + size_t(b) * (cp.blockLen + cp.dstStride), s + b * pitch, cp.blockLen);
    g_cycleTracker.dmaBytesMoved += cp.blockCount * row;
    g_cycleTracker.dmaTransfers++;
    g_cycleTracker.padTransfers++;
    g_timeline.Store(static_cast<double>(cp.blockCount * row), g_timeline.SpanOf(src.GetData(), extent));
}
}  // namespace detail

// ---- The target's forms: system memory through a GlobalTensor -----------------------------------
template <typename T>
inline void DataCopy(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t count) {
    detail::BlockLoad(dst, src.GetData(), count);
}

template <typename T>
inline void DataCopy(GlobalTensor<T> dst, LocalTensor<T> src, uint32_t count) {
    detail::BlockStore(dst.GetData(), src, count);
}

template <typename T>
inline void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src, DataCopyExtParams cp, DataCopyPadExtParams<T> pad = {}) {
    detail::PadLoad(dst, src.GetData(), cp, pad);
}

template <typename T>
inline void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src, DataCopyExtParams cp) {
    detail::PadStore(dst.GetData(), src, cp);
}

// One row of `count` elements: a load zero-fills the rest of its last block (hardware zero-fill)
template <typename T>
inline void LoadPad(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t count) {
    detail::PadLoad(dst, src.GetData(), DataCopyExtParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0},
                    DataCopyPadExtParams<T>{true, 0, 0, T(0)});
}

template <typename T>
inline void StorePad(GlobalTensor<T> dst, LocalTensor<T> src, uint32_t count) {
    detail::PadStore(dst.GetData(), src, DataCopyExtParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0});
}

// ---- Refused: the forms the target does not have (docs/TARGET_API_SHAPE.md) ----------------------
// [NOT ON THE TARGET] the target has no raw-pointer form of DataCopy; a device function cannot cast a global-memory pointer to a typed pointer.
// Use instead: SetGlobalBuffer on a GlobalTensor<T>, then DataCopy(LocalTensor, GlobalTensor, count).
template <typename T>
inline void DataCopy(LocalTensor<T>, const T*, uint32_t) {
    static_assert(sizeof(T) == 0,
                  "the target has no raw-pointer form of DataCopy; a device function cannot cast a global-memory pointer to a typed pointer.  SetGlobalBuffer on a GlobalTensor<T>, then DataCopy(LocalTensor, GlobalTensor, count).");
}

// [NOT ON THE TARGET] the target has no raw-pointer form of DataCopy.
// Use instead: DataCopy(GlobalTensor, LocalTensor, count).
template <typename T>
inline void DataCopy(T*, LocalTensor<T>, uint32_t) {
    static_assert(sizeof(T) == 0,
                  "the target has no raw-pointer form of DataCopy.  DataCopy(GlobalTensor, LocalTensor, count).");
}

// [NOT ON THE TARGET] the target has no raw-pointer form of DataCopyPad, and no 3-argument form either.
// Use instead: DataCopyPad(LocalTensor, GlobalTensor, DataCopyExtParams, DataCopyPadExtParams).
template <typename T>
inline void DataCopyPad(LocalTensor<T>, const T*, uint32_t) {
    static_assert(sizeof(T) == 0,
                  "the target has no raw-pointer form of DataCopyPad, and no 3-argument form either.  DataCopyPad(LocalTensor, GlobalTensor, DataCopyExtParams, DataCopyPadExtParams).");
}

// [NOT ON THE TARGET] the target has no raw-pointer form of DataCopyPad, and no 3-argument form either.
// Use instead: DataCopyPad(GlobalTensor, LocalTensor, DataCopyExtParams).
template <typename T>
inline void DataCopyPad(T*, LocalTensor<T>, uint32_t) {
    static_assert(sizeof(T) == 0,
                  "the target has no raw-pointer form of DataCopyPad, and no 3-argument form either.  DataCopyPad(GlobalTensor, LocalTensor, DataCopyExtParams).");
}

// [NOT ON THE TARGET] the target's DataCopyPad always takes a DataCopyExtParams describing blockCount / blockLen / strides; there is no count-only form.
// Use instead: DataCopyPad(LocalTensor, GlobalTensor, DataCopyExtParams, DataCopyPadExtParams).
template <typename T>
inline void DataCopyPad(LocalTensor<T>, GlobalTensor<T>, uint32_t) {
    static_assert(sizeof(T) == 0,
                  "the target's DataCopyPad always takes a DataCopyExtParams describing blockCount / blockLen / strides; there is no count-only form.  DataCopyPad(LocalTensor, GlobalTensor, DataCopyExtParams, DataCopyPadExtParams).");
}

// [NOT ON THE TARGET] the target's DataCopyPad always takes a DataCopyExtParams describing blockCount / blockLen / strides; there is no count-only form.
// Use instead: DataCopyPad(GlobalTensor, LocalTensor, DataCopyExtParams).
template <typename T>
inline void DataCopyPad(GlobalTensor<T>, LocalTensor<T>, uint32_t) {
    static_assert(sizeof(T) == 0,
                  "the target's DataCopyPad always takes a DataCopyExtParams describing blockCount / blockLen / strides; there is no count-only form.  DataCopyPad(GlobalTensor, LocalTensor, DataCopyExtParams).");
}

// [NOT ON THE TARGET] the target has no raw-pointer form of DataCopyPad.
// Use instead: DataCopyPad(LocalTensor, GlobalTensor, DataCopyExtParams, DataCopyPadExtParams).
template <typename T>
inline void DataCopyPad(LocalTensor<T>, const T*, DataCopyExtParams, DataCopyPadExtParams<T> = {}) {
    static_assert(sizeof(T) == 0,
                  "the target has no raw-pointer form of DataCopyPad.  DataCopyPad(LocalTensor, GlobalTensor, DataCopyExtParams, DataCopyPadExtParams).");
}

// [NOT ON THE TARGET] the target has no raw-pointer form of DataCopyPad.
// Use instead: DataCopyPad(GlobalTensor, LocalTensor, DataCopyExtParams).
template <typename T>
inline void DataCopyPad(T*, LocalTensor<T>, DataCopyExtParams) {
    static_assert(sizeof(T) == 0,
                  "the target has no raw-pointer form of DataCopyPad.  DataCopyPad(GlobalTensor, LocalTensor, DataCopyExtParams).");
}

// [NOT ON THE TARGET] the target has no raw-pointer form.
// Use instead: LoadPad(LocalTensor, GlobalTensor, count).
template <typename T>
inline void LoadPad(LocalTensor<T>, const T*, uint32_t) {
    static_assert(sizeof(T) == 0,
                  "the target has no raw-pointer form.  LoadPad(LocalTensor, GlobalTensor, count).");
}

// [NOT ON THE TARGET] the target has no raw-pointer form.
// Use instead: StorePad(GlobalTensor, LocalTensor, count).
template <typename T>
inline void StorePad(T*, LocalTensor<T>, uint32_t) {
    static_assert(sizeof(T) == 0,
                  "the target has no raw-pointer form.  StorePad(GlobalTensor, LocalTensor, count).");
}

// Scratchpad -> scratchpad (the local copy unit)
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
    g_timeline.Copy(static_cast<double>(copyBytes), SpanOf(dst.GetData(), count), SpanOf(src.GetData(), count));
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
    g_timeline.Vector(2 * repeats + 13, SpanOf(dst.data, count), SpanOf(src0.data, count), SpanOf(src1.data, count));
}

template <typename T>
inline void Mul(LocalTensor<T> dst, LocalTensor<T> src0, LocalTensor<T> src1, uint32_t count) {
    #pragma omp simd
    for (uint32_t i = 0; i < count; ++i) {
        dst.data[i] = src0.data[i] * src1.data[i];
    }
    uint32_t repeats = (count * sizeof(T) + SIMD_REPEAT_BYTES - 1) / SIMD_REPEAT_BYTES;
    g_cycleTracker.vMulCycles += 2 * repeats + 13;
    g_timeline.Vector(2 * repeats + 13, SpanOf(dst.data, count), SpanOf(src0.data, count), SpanOf(src1.data, count));
}

template <typename T>
inline void Muls(LocalTensor<T> dst, LocalTensor<T> src, float scalar, uint32_t count) {
    #pragma omp simd
    for (uint32_t i = 0; i < count; ++i) {
        dst.data[i] = static_cast<T>(src.data[i] * scalar);
    }
    uint32_t repeats = (count * sizeof(T) + SIMD_REPEAT_BYTES - 1) / SIMD_REPEAT_BYTES;
    g_cycleTracker.vMulCycles += 2 * repeats + 13;
    g_timeline.Vector(2 * repeats + 13, SpanOf(dst.data, count), SpanOf(src.data, count));
}

template <typename T>
inline void Adds(LocalTensor<T> dst, LocalTensor<T> src, float scalar, uint32_t count) {
    #pragma omp simd
    for (uint32_t i = 0; i < count; ++i) {
        dst.data[i] = static_cast<T>(static_cast<float>(src.data[i]) + scalar);
    }
    uint32_t repeats = (count * sizeof(T) + SIMD_REPEAT_BYTES - 1) / SIMD_REPEAT_BYTES;
    g_cycleTracker.vAddCycles += 2 * repeats + 13;
    g_timeline.Vector(2 * repeats + 13, SpanOf(dst.data, count), SpanOf(src.data, count));
}

// -----------------------------------------------------------------------------
// Cast: element-wise format conversion (FP32 <-> FP16 / BF16) under a rounding mode. Widening is
// exact; narrowing rounds as `mode` says, CAST_NONE rounding to nearest-even as CAST_RINT does
// wherever precision is lost. FP16 keeps subnormals and overflows to infinity.
// -----------------------------------------------------------------------------
namespace detail {
inline float AsFloat(float v) { return v; }
inline float AsFloat(half v) { return static_cast<float>(v); }
inline float AsFloat(bfloat16_t v) { return static_cast<float>(v); }

inline uint32_t FloatBits(float f) {
    uint32_t b;
    std::memcpy(&b, &f, sizeof b);
    return b;
}

// Round x to the grid of `ulp`-spaced values the narrow format can hold near it, in `mode`
inline float RoundToGrid(float x, float ulp, RoundMode mode) {
    const double q = static_cast<double>(x) / ulp;
    double r;
    switch (mode) {
        case RoundMode::CAST_FLOOR: r = std::floor(q); break;
        case RoundMode::CAST_CEIL: r = std::ceil(q); break;
        case RoundMode::CAST_TRUNC: r = std::trunc(q); break;
        default: r = std::nearbyint(q); break;  // Nearest, ties to even (the default FP environment)
    }
    return static_cast<float>(r * ulp);
}

inline half ToHalf(float x, RoundMode mode) {
    half h;
    const uint32_t b = FloatBits(x);
    const uint16_t sign = static_cast<uint16_t>((b >> 16) & 0x8000u);
    if (std::isnan(x)) { h.data = static_cast<uint16_t>(sign | 0x7E00u); return h; }
    const float a = std::fabs(x);
    // The spacing of FP16 values near |x|: 2^-24 below 2^-14 (subnormals), else 2^(e - 10)
    int e = 0;
    std::frexp(a, &e);  // a = m * 2^e, m in [0.5, 1)
    const float ulp = a < 6.103515625e-05f ? 5.9604644775390625e-08f : std::ldexp(1.0f, e - 11);
    const float r = std::fabs(RoundToGrid(x, ulp, mode));
    if (r >= 65520.0f || (r > 65504.0f)) {  // Past the largest finite value: infinity, or 65504 when rounding toward it
        const bool toward = mode == RoundMode::CAST_TRUNC || (mode == RoundMode::CAST_FLOOR && !sign) || (mode == RoundMode::CAST_CEIL && sign);
        h.data = static_cast<uint16_t>(sign | (toward ? 0x7BFFu : 0x7C00u));
        return h;
    }
    if (r < 6.103515625e-05f) {  // Subnormal (or zero): r / 2^-24 counts the ulps
        h.data = static_cast<uint16_t>(sign | static_cast<uint16_t>(r / 5.9604644775390625e-08f));
        return h;
    }
    const uint32_t rb = FloatBits(r);
    const int32_t exp = static_cast<int32_t>((rb >> 23) & 0xFFu) - 127 + 15;
    h.data = static_cast<uint16_t>(sign | (exp << 10) | ((rb >> 13) & 0x3FFu));
    return h;
}

inline bfloat16_t ToBF16(float x, RoundMode mode) {
    bfloat16_t h;
    const uint32_t b = FloatBits(x);
    if (std::isnan(x)) { h.data = static_cast<uint16_t>((b >> 16) | 0x0040u); return h; }
    if (std::isinf(x) || x == 0.0f) { h.data = static_cast<uint16_t>(b >> 16); return h; }
    const float a = std::fabs(x);
    int e = 0;
    std::frexp(a, &e);
    const float ulp = std::ldexp(1.0f, std::max(e - 8, -133));  // 8 significant bits (subnormals below 2^-126)
    const float r = RoundToGrid(x, ulp, mode);
    h.data = static_cast<uint16_t>(FloatBits(r) >> 16);
    return h;
}

template <typename D, typename S>
inline D Convert(S x, RoundMode mode) {
    const float f = AsFloat(x);
    if constexpr (std::is_same<D, float>::value) { (void)mode; return f; }
    else if constexpr (std::is_same<D, half>::value) return ToHalf(f, mode);
    else if constexpr (std::is_same<D, bfloat16_t>::value) return ToBF16(f, mode);
    else { static_assert(sizeof(D) == 0, "Cast converts between float, half and bfloat16_t"); return D(); }
}
}  // namespace detail

template <typename D, typename S>
inline void Cast(LocalTensor<D> dst, LocalTensor<S> src, RoundMode mode, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        dst.data[i] = detail::Convert<D>(src.data[i], mode);
    }
    const size_t widest = sizeof(D) > sizeof(S) ? sizeof(D) : sizeof(S);
    uint32_t repeats = static_cast<uint32_t>((count * widest + SIMD_REPEAT_BYTES - 1) / SIMD_REPEAT_BYTES);
    g_cycleTracker.vCastCycles += 2 * repeats + 13;
    g_timeline.Vector(2 * repeats + 13, SpanOf(dst.data, count), SpanOf(src.data, count));
}

template <typename D, typename S>
inline void Cast(LocalTensor<D> dst, LocalTensor<S> src, RoundMode mode, uint64_t count, uint8_t rep, UnaryRepeatParams params) {
    (void)rep; (void)params;
    Cast(dst, src, mode, static_cast<uint32_t>(count));
}

// [NOT ON THE TARGET] the target's Cast takes a rounding mode, not a converter function, and the mode comes before the count.
// Use instead: Cast(dst, src, RoundMode::CAST_NONE, count).
template <typename D, typename S, typename Convert>
inline void Cast(LocalTensor<D>, LocalTensor<S>, uint32_t, Convert) {
    static_assert(sizeof(S) == 0,
                  "the target's Cast takes a rounding mode, not a converter function, and the mode comes before the count.  Cast(dst, src, RoundMode::CAST_NONE, count).");
}

// [NOT ON THE TARGET] the target's Cast takes a rounding mode, not a converter function, and the mode comes before the count.
// Use instead: Cast(dst, src, RoundMode::CAST_NONE, count).
template <typename D, typename S>
inline void Cast(LocalTensor<D>, LocalTensor<S>, uint32_t) {
    static_assert(sizeof(S) == 0,
                  "the target's Cast takes a rounding mode, not a converter function, and the mode comes before the count.  Cast(dst, src, RoundMode::CAST_NONE, count).");
}

template <typename T>
inline void ToFloat(LocalTensor<float> dst, LocalTensor<T> src, uint32_t count) {
    if constexpr (std::is_same_v<T, float>) {
        Adds(dst, src, 0.0f, count);
    } else {
        Cast(dst, src, RoundMode::CAST_NONE, count);
    }
}

template <typename T>
inline void FromFloat(LocalTensor<T> dst, LocalTensor<float> src, uint32_t count) {
    if constexpr (std::is_same_v<T, float>) {
        Adds(dst, src, 0.0f, count);
    } else {
        Cast(dst, src, RoundMode::CAST_RINT, count);
    }
}

template <typename T>
inline void Duplicate(LocalTensor<T> dst, T scalar, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        dst.data[i] = scalar;
    }
    uint32_t repeats = (count * sizeof(T) + SIMD_REPEAT_BYTES - 1) / SIMD_REPEAT_BYTES;
    g_cycleTracker.vCastCycles += repeats + 18;
    g_timeline.Vector(repeats + 18, SpanOf(dst.data, count));
}

// Rsqrt [6.3]: a low-precision table lookup, not a reciprocal square root to the last bit. Its result
// keeps RSQRT_TABLE_BITS fraction bits (the rest truncated), so a kernel refines it by Newton-Raphson.
// There is no scalar VectorInvRms on the target.
static constexpr uint32_t RSQRT_TABLE_BITS = 11;

template <typename T>
inline void Rsqrt(LocalTensor<T> dst, LocalTensor<T> src, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        float r = 1.0f / std::sqrt(static_cast<float>(src.data[i]));
        if (std::isfinite(r)) {
            uint32_t bits;
            std::memcpy(&bits, &r, sizeof bits);
            bits &= ~((1u << (23 - RSQRT_TABLE_BITS)) - 1u);
            std::memcpy(&r, &bits, sizeof bits);
        }
        dst.data[i] = static_cast<T>(r);
    }
    uint32_t repeats = (count * sizeof(T) + SIMD_REPEAT_BYTES - 1) / SIMD_REPEAT_BYTES;
    g_cycleTracker.vRsqrtCycles += 2 * repeats + 14;
    g_timeline.Vector(2 * repeats + 14, SpanOf(dst.data, count), SpanOf(src.data, count));
}

// -----------------------------------------------------------------------------
// BlockReduceSum: every 32-byte block of a repeat folds into one value (1 cycle per repeat).
// BlockReduceSum(dst, src, repeatTimes, mask, dstRepStride, srcBlkStride, srcRepStride): repeat r
// reads the 8 blocks of src at (r * srcRepStride + b * srcBlkStride) blocks, b = 0..7, of which the
// first `mask` elements take part (1 to one repeat's lanes); block b's sum is element b of the
// repeat's 8 results, which land at dst + r * dstRepStride * 8 elements (dstRepStride counts one
// repeat's results). A block without a participating element yields no defined result: it reads NaN
// here. The destination starts a block and never overlaps the source (Trap #402).
// -----------------------------------------------------------------------------
template <typename T>
inline void BlockReduceSum(LocalTensor<T> dst, LocalTensor<T> src, uint8_t repeatTimes, uint64_t mask, uint8_t dstRepStride,
                           uint8_t srcBlkStride, uint8_t srcRepStride) {
    constexpr uint32_t blockElems = DMA_ALIGN_BYTES / sizeof(T), lanes = SIMD_REPEAT_BYTES / sizeof(T);
    if (mask == 0 || mask > lanes) {
        throw std::runtime_error("[Hardware Fault - REPEAT MASK]: BlockReduceSum's mask is the element count of one repeat (1 to " +
                                 std::to_string(lanes) + "): got " + std::to_string(mask));
    }
    if (reinterpret_cast<uintptr_t>(dst.GetData()) % DMA_ALIGN_BYTES != 0) {
        throw std::runtime_error("[Hardware Fault - OPERAND UNALIGNED]: BlockReduceSum's destination must start a 32-byte block");
    }
    const size_t srcExtent = repeatTimes ? (size_t(repeatTimes - 1) * srcRepStride + size_t((mask - 1) / blockElems) * srcBlkStride) * blockElems +
                                               (mask - 1) % blockElems + 1
                                         : 0;
    const size_t dstExtent = repeatTimes ? size_t(repeatTimes - 1) * dstRepStride * 8 + 8 : 0;
    if (srcExtent * sizeof(T) > src.capacityBytes || dstExtent * sizeof(T) > dst.capacityBytes) {
        throw std::runtime_error("[Hardware Fault - BUFFER OVERRUN]: a BlockReduceSum operand runs past its buffer");
    }
    const T* s = src.data;
    T* d = dst.data;
    const uintptr_t d0 = reinterpret_cast<uintptr_t>(d), d1 = d0 + dstExtent * sizeof(T);
    const uintptr_t s0 = reinterpret_cast<uintptr_t>(s), s1 = s0 + srcExtent * sizeof(T);
    if (repeatTimes && d0 < s1 && s0 < d1) {
        throw std::runtime_error("[Hardware Fault - VECTOR ALU OPERAND ALIASING]: BlockReduceSum destination overlaps its source! "
                                 "Intra-row folding must ping-pong across disjoint buffers (Trap #402).");
    }
    for (uint32_t r = 0; r < repeatTimes; ++r) {
        for (uint32_t b = 0; b < 8; ++b) {
            const uint64_t first = uint64_t(b) * blockElems;
            T sum = T(0);
            if (first >= mask) {
                sum = static_cast<T>(std::numeric_limits<double>::quiet_NaN());
            } else {
                const size_t base = (size_t(r) * srcRepStride + size_t(b) * srcBlkStride) * blockElems;
                for (uint64_t e = first; e < mask && e < first + blockElems; ++e) sum += s[base + (e - first)];
            }
            d[size_t(r) * dstRepStride * 8 + b] = sum;
        }
    }
    g_cycleTracker.vBlockReduceCycles += 1u * repeatTimes + 14;
    g_timeline.Vector(1u * repeatTimes + 14, SpanOf(d, dstExtent), SpanOf(s, srcExtent));
}

// [NOT ON THE TARGET] the target's BlockReduceSum takes seven arguments: (dst, src, repeatTimes, mask, dstRepStride, srcBlkStride, srcRepStride). A count-only form hides the repeat structure, which is exactly what has to be chosen deliberately on the hardware.
// Use instead: BlockReduceSum(dst, src, repeatTimes, mask, 1, 1, 8) for whole 64-lane repeats of 4-byte elements.
template <typename T>
inline void BlockReduceSum(LocalTensor<T>, LocalTensor<T>, uint32_t) {
    static_assert(sizeof(T) == 0,
                  "the target's BlockReduceSum takes seven arguments: (dst, src, repeatTimes, mask, dstRepStride, srcBlkStride, srcRepStride). A count-only form hides the repeat structure, which is exactly what has to be chosen deliberately on the hardware.  BlockReduceSum(dst, src, repeatTimes, mask, 1, 1, 8) for whole 64-lane repeats of 4-byte elements.");
}

// [NOT ON THE TARGET] no reduction on the target returns a value. ReduceSum returns void, writes only lane 0 of a 32-byte slot, and needs a work tensor; reading the scalar back requires a vector->scalar fence, which inside a per-row loop is one pipeline fence per row.
// Use instead: ReduceSum(dst, src, work, count), then a fence, then dst.GetValue(0) - and offset multi-row destinations by dst[i * 8] for 4-byte elements.
template <typename T>
inline float VectorReduceSum(LocalTensor<T>, uint32_t) {
    static_assert(sizeof(T) == 0,
                  "no reduction on the target returns a value. ReduceSum returns void, writes only lane 0 of a 32-byte slot, and needs a work tensor; reading the scalar back requires a vector->scalar fence, which inside a per-row loop is one pipeline fence per row.  ReduceSum(dst, src, work, count), then a fence, then dst.GetValue(0) - and offset multi-row destinations by dst[i * 8] for 4-byte elements.");
    return 0.0f;
}

// [NOT ON THE TARGET] there is no VectorInvRms instruction.
// Use instead: Rsqrt on a tensor (a low-precision table lookup) followed by
// Newton-Raphson refinement. Skipping the refinement loses accuracy.
template <typename F = float>
inline float VectorInvRms(F, float, float) {
    static_assert(sizeof(F) == 0,
                  "[NOT ON THE TARGET] no VectorInvRms instruction. Use Rsqrt on a tensor (low-precision table lookup) plus Newton-Raphson refinement.");
    return 0.0f;
}

// [NOT ON THE TARGET] there is no VectorInvRms instruction (and the worker has no scalar
// integer-to-float unit either: pass 1 / D from the coordinator).
template <typename F = float>
inline float VectorInvRms(F, uint32_t, float) {
    static_assert(sizeof(F) == 0,
                  "[NOT ON THE TARGET] no VectorInvRms instruction. Use Rsqrt on a tensor (low-precision table lookup) plus Newton-Raphson refinement.");
    return 0.0f;
}

// -----------------------------------------------------------------------------
// WholeReduceSum(dst, src, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride): each repeat
// sums its first `mask` elements (its blocks at (r * srcRepStride + b * srcBlkStride) blocks) into one
// value, which lands at dst + r * dstRepStride elements. Expensive: 14 cycles per repeat.
// -----------------------------------------------------------------------------
template <typename T>
inline void WholeReduceSum(LocalTensor<T> dst, LocalTensor<T> src, uint64_t mask, uint8_t repeatTimes, uint8_t dstRepStride,
                           uint8_t srcBlkStride, uint8_t srcRepStride) {
    constexpr uint32_t blockElems = DMA_ALIGN_BYTES / sizeof(T), lanes = SIMD_REPEAT_BYTES / sizeof(T);
    if (mask == 0 || mask > lanes) {
        throw std::runtime_error("[Hardware Fault - REPEAT MASK]: WholeReduceSum's mask is the element count of one repeat (1 to " +
                                 std::to_string(lanes) + "): got " + std::to_string(mask));
    }
    const size_t srcExtent = repeatTimes ? (size_t(repeatTimes - 1) * srcRepStride + size_t((mask - 1) / blockElems) * srcBlkStride) * blockElems +
                                               (mask - 1) % blockElems + 1
                                         : 0;
    const size_t dstExtent = repeatTimes ? size_t(repeatTimes - 1) * dstRepStride + 1 : 0;
    if (srcExtent * sizeof(T) > src.capacityBytes || dstExtent * sizeof(T) > dst.capacityBytes) {
        throw std::runtime_error("[Hardware Fault - BUFFER OVERRUN]: a WholeReduceSum operand runs past its buffer");
    }
    for (uint32_t r = 0; r < repeatTimes; ++r) {
        T sum = T(0);
        for (uint64_t e = 0; e < mask; ++e) {
            sum += src.data[(size_t(r) * srcRepStride + size_t(e / blockElems) * srcBlkStride) * blockElems + e % blockElems];
        }
        dst.data[size_t(r) * dstRepStride] = sum;
    }
    g_cycleTracker.vWholeReduceCycles += 14u * repeatTimes + 14;
    g_timeline.Vector(14u * repeatTimes + 14, SpanOf(dst.data, dstExtent), SpanOf(src.data, srcExtent));
}

// [NOT ON THE TARGET] the target's WholeReduceSum also takes the repeat structure explicitly.
// Use instead: WholeReduceSum(dst, src, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride).
template <typename T>
inline void WholeReduceSum(LocalTensor<T>, LocalTensor<T>, uint32_t) {
    static_assert(sizeof(T) == 0,
                  "the target's WholeReduceSum also takes the repeat structure explicitly.  WholeReduceSum(dst, src, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride).");
}

// -----------------------------------------------------------------------------
// ReduceSum [6.2]: void. The sum of `count` elements (whole 64-lane repeats) lands in lane 0 of the
// 32-byte block at dst. The target leaves lanes 1..7 holding whatever was there; this runtime fills
// them with NaN, so a kernel that consumes the whole slot fails here instead of silently losing
// accuracy on the hardware. dst must start a block: the slot of row i is dst[i * 8] (4-byte
// elements), never dst[i], which would put rows into one block's lane 0.
// -----------------------------------------------------------------------------
template <typename T>
inline void ReduceSum(LocalTensor<T> dst, LocalTensor<T> src, LocalTensor<T> work, uint32_t count) {
    if (dst.GetData() == src.GetData() || dst.GetData() == work.GetData() || src.GetData() == work.GetData()) {
        throw std::runtime_error("[Hardware Fault - VECTOR ALU OPERAND ALIASING (Trap #402)]: "
                                 "ReduceSum destination, source, and scratch workpad buffers must be strictly disjoint! "
                                 "The SIMD reduction datapath forbids workspace aliasing.");
    }
    if (reinterpret_cast<uintptr_t>(dst.GetData()) % DMA_ALIGN_BYTES != 0) {
        throw std::runtime_error("[Hardware Fault - REDUCTION SLOT UNALIGNED]: ReduceSum writes lane 0 of a 32-byte block: "
                                 "its destination must start one (row i's slot is dst[i * 8], not dst[i])");
    }
    if (count % 64 != 0) {
        throw std::runtime_error("[Hardware Fault - UNALIGNED SIMD LANE FAULT (Trap #408)]: "
                                 "ReduceSum element count (" + std::to_string(count) +
                                 ") is not a multiple of 64! The physical SIMD reduction array operates "
                                 "strictly on 64-lane blocks (256 bytes). Unaligned counts trigger hardware "
                                 "pipeline hang and stream timeout. Non-64 tail vectors must be zero-padded in scratchpad.");
    }
    T sum = 0;
    for (uint32_t i = 0; i < count; ++i) {
        sum += src.data[i];
    }
    dst.data[0] = sum;
    const uint32_t lanes = DMA_ALIGN_BYTES / sizeof(T);
    const uint32_t slot = dst.capacityBytes >= DMA_ALIGN_BYTES ? lanes : 1;
    for (uint32_t i = 1; i < slot; ++i) {
        dst.data[i] = static_cast<T>(std::numeric_limits<double>::quiet_NaN());
    }
    uint32_t repeats = (count * sizeof(T) + SIMD_REPEAT_BYTES - 1) / SIMD_REPEAT_BYTES;
    g_cycleTracker.vBlockReduceCycles += 2 * repeats + 15;
    g_timeline.Vector(2 * repeats + 15, SpanOf(dst.data, slot), SpanOf(src.data, count));
}

// -----------------------------------------------------------------------------
// Brcb [6.1]: a block broadcast, not a scalar one. Each repeat reads 8 consecutive values of src and
// fills 8 32-byte blocks of dst, block j with value j: repeat r writes value src[8r + j] over the
// block at (r * dstRepStride + j * dstBlkStride) blocks. So one Brcb repeat spreads the values of 8
// rows, one block each; a row's value across the row is a strided operation with src1BlkStride = 0.
// -----------------------------------------------------------------------------
struct BrcbRepeatParams {
    uint16_t dstBlkStride = 1;  // Blocks between the 8 blocks of a repeat
    uint16_t dstRepStride = 8;  // Blocks between repeats
};

template <typename T>
inline void Brcb(LocalTensor<T> dst, LocalTensor<T> src, uint32_t repeatTimes, BrcbRepeatParams params = {}) {
    constexpr uint32_t blockElems = DMA_ALIGN_BYTES / sizeof(T);
    const size_t lastBlock = repeatTimes ? size_t(repeatTimes - 1) * params.dstRepStride + 7u * params.dstBlkStride : 0;
    const size_t extent = repeatTimes ? (lastBlock + 1) * blockElems : 0;  // Elements up to the end of the last block
    if (repeatTimes > 255 || size_t(dst.capacityBytes) < extent * sizeof(T) || size_t(src.capacityBytes) < 8u * repeatTimes * sizeof(T) ||
        reinterpret_cast<uintptr_t>(dst.GetData()) % DMA_ALIGN_BYTES != 0) {
        throw std::runtime_error("[Hardware Fault - BUFFER OVERRUN (Trap #410)]: Brcb reads 8 values per repeat and writes 8 whole "
                                 "32-byte blocks: its destination must start a block and hold every block it writes, its source 8 values "
                                 "per repeat (at most 255 repeats)");
    }
    for (uint32_t r = 0; r < repeatTimes; ++r) {
        for (uint32_t j = 0; j < 8; ++j) {
            const T v = src.data[8 * r + j];
            T* block = dst.data + (size_t(r) * params.dstRepStride + size_t(j) * params.dstBlkStride) * blockElems;
            for (uint32_t t = 0; t < blockElems; ++t) block[t] = v;
        }
    }
    g_cycleTracker.vCastCycles += 1 * repeatTimes + 8;
    g_timeline.Vector(1 * repeatTimes + 8, SpanOf(dst.data, extent), SpanOf(src.data, 8u * repeatTimes));
}

// -----------------------------------------------------------------------------
// Strided binary form [6.8]: Mul/Add(dst, src0, src1, mask, repeatTimes, params). `mask` is the
// element count of every repeat (at most one repeat's lanes: 64 at 4 bytes), not a total, and
// `repeatTimes` (at most 255) the number of repeats. Repeat r processes its elements in 32-byte
// blocks: element e of operand X lives at (r * XRepStride + (e / blockElems) * XBlkStride) blocks,
// plus e % blockElems. A block stride of 0 makes every block of a repeat read the same block; a
// repeat stride of 0 makes every repeat read the same repeat.
// -----------------------------------------------------------------------------
struct BinaryRepeatParams {
    uint8_t dstBlkStride = 1;
    uint8_t src0BlkStride = 1;
    uint8_t src1BlkStride = 1;
    uint8_t dstRepStride = 8;
    uint8_t src0RepStride = 8;
    uint8_t src1RepStride = 8;
};

namespace detail {
template <typename T, typename Op>
inline void StridedBinary(LocalTensor<T> dst, LocalTensor<T> src0, LocalTensor<T> src1, uint64_t mask, uint8_t repeatTimes,
                          const BinaryRepeatParams& p, Op op, uint64_t& cycles) {
    constexpr uint32_t blockElems = DMA_ALIGN_BYTES / sizeof(T), lanes = SIMD_REPEAT_BYTES / sizeof(T);
    if (mask == 0 || mask > lanes) {
        throw std::runtime_error("[Hardware Fault - REPEAT MASK]: the strided form's mask is the element count of one repeat (1 to " +
                                 std::to_string(lanes) + "), not a row width: got " + std::to_string(mask));
    }
    auto at = [&](uint32_t r, uint64_t e, uint8_t rep, uint8_t blk) {
        return (size_t(r) * rep + size_t(e / blockElems) * blk) * blockElems + e % blockElems;
    };
    auto extent = [&](uint8_t rep, uint8_t blk) { return repeatTimes ? at(repeatTimes - 1, mask - 1, rep, blk) + 1 : 0; };
    const size_t nd = extent(p.dstRepStride, p.dstBlkStride), n0 = extent(p.src0RepStride, p.src0BlkStride),
                 n1 = extent(p.src1RepStride, p.src1BlkStride);
    if (nd * sizeof(T) > dst.capacityBytes || n0 * sizeof(T) > src0.capacityBytes || n1 * sizeof(T) > src1.capacityBytes) {
        throw std::runtime_error("[Hardware Fault - BUFFER OVERRUN]: a strided operand runs past its buffer");
    }
    for (uint32_t r = 0; r < repeatTimes; ++r) {
        for (uint64_t e = 0; e < mask; ++e) {
            dst.data[at(r, e, p.dstRepStride, p.dstBlkStride)] =
                op(src0.data[at(r, e, p.src0RepStride, p.src0BlkStride)], src1.data[at(r, e, p.src1RepStride, p.src1BlkStride)]);
        }
    }
    cycles += 2u * repeatTimes + 13;
    g_timeline.Vector(2u * repeatTimes + 13, SpanOf(dst.data, nd), SpanOf(src0.data, n0), SpanOf(src1.data, n1));
}
}  // namespace detail

template <typename T>
inline void Mul(LocalTensor<T> dst, LocalTensor<T> src0, LocalTensor<T> src1, uint64_t mask, uint8_t repeatTimes, const BinaryRepeatParams& p) {
    detail::StridedBinary(dst, src0, src1, mask, repeatTimes, p, [](T a, T b) { return a * b; }, g_cycleTracker.vMulCycles);
}

template <typename T>
inline void Add(LocalTensor<T> dst, LocalTensor<T> src0, LocalTensor<T> src1, uint64_t mask, uint8_t repeatTimes, const BinaryRepeatParams& p) {
    detail::StridedBinary(dst, src0, src1, mask, repeatTimes, p, [](T a, T b) { return a + b; }, g_cycleTracker.vAddCycles);
}

// -----------------------------------------------------------------------------
// DAE v1.5 High-Precision Newton-Raphson Refinement
// Refines 11-bit LUT hardware Rsqrt to full 24-bit FP32 precision.
// -----------------------------------------------------------------------------
inline float RefineInvRms(float mean, float inv) {
    if (mean > 0.0f && mean <= 3.402823466e38f) {
        inv = inv * (1.5f - (0.5f * (mean * inv)) * inv);
        inv = inv * (1.5f - (0.5f * (mean * inv)) * inv);
    }
    return inv;
}

// -----------------------------------------------------------------------------
// DAE v1.5 Kernel Launch Constant Frame Guard (Trap #409)
// -----------------------------------------------------------------------------
template <typename ArgsStruct>
inline void ValidateLaunchArgs(const ArgsStruct&) {
    if (sizeof(ArgsStruct) > 32) {
        throw std::runtime_error("[Hardware Fault - CONSTANT REGISTER FRAME OVERFLOW (Trap #409)]: "
                                 "Kernel launch argument structure size (" + std::to_string(sizeof(ArgsStruct)) +
                                 " bytes) exceeds the 32-byte physical constant register frame! "
                                 "Host-to-device kernel boundaries must be flattened into primitive scalars and 64-bit pointers.");
    }
}

// ---------------------------------------------------------------------------
// Accessors that exist here but NOT on the target.
//
//   LocalTensor<T>::GetData()   - the target's LocalTensor exposes no raw
//                                 pointer. Used only by this runtime's own
//                                 checks; a kernel that calls it will not
//                                 translate.
//   LocalTensor<T>::pos         - no such field on the target. Here it is
//                                 private: a buffer's position is set by the
//                                 TBuf / TQue that hands it out and read with
//                                 GetPosition(), and t.pos = ... does not
//                                 compile, as on the target.
//   GlobalTensor<T>::GetData(),
//   its conversions to T* and
//   operator+                   - the simulator's view of the address a
//                                 GlobalTensor holds; the DMA forms above use
//                                 it. A kernel indexes with operator[] instead.
//   GetCoreIdx / GetCoreNum /
//   GetThreadIdx / GetThreadNum - not target APIs. Use GetBlockIdx /
//                                 GetBlockNum.
//   CrossPipe                   - not a target API either; it is a two-line
//                                 helper (SetFlag then WaitFlag on the same
//                                 event id). Keep using it, but expect to carry
//                                 the helper across.
//
// These are left in place because the runtime's own checks depend on them.
// ---------------------------------------------------------------------------

} // namespace dsa
