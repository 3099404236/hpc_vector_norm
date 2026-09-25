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
// Timeline model: when each operation of a core runs, not only what it costs.
// A core has three in-order units:
//   VECTOR  the vector pipe: every vector instruction, for its cycle cost
//   DMA     the system-memory channel, shared by loads and stores: a transfer occupies it
//           for bytes / DMA_BYTES_PER_CYCLE, and its data lands DMA_LATENCY_CYCLES after
//           that (the latency of back-to-back transfers overlaps)
//   LOCAL   scratchpad-to-scratchpad copies (no system-memory traffic)
// Operations issue in program order. Each starts once its unit is free and its operands
// are: a 32-byte scratchpad block can be read once its last write has landed, and written
// once its last read has ended. A core arrives at SyncAll once its vector and local units
// are idle and its stores have landed (loads may stay in flight); all cores leave
// SYNC_ALL_CYCLES after the last arrival. Times are vector cycles at CLOCK_GHZ.
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
static constexpr double   CLOCK_GHZ             = 1.5;     // Assumed vector clock: converts cycles to time
static constexpr double   DMA_BYTES_PER_CYCLE   = 850.0 / MAX_HARDWARE_CORES / CLOCK_GHZ;  // 21.25 GB/s per core
static constexpr double   DMA_LATENCY_CYCLES    = 1200.0;  // 800 ns from a transfer's issue to its data landing
static constexpr double   LOCAL_BYTES_PER_CYCLE = SIMD_REPEAT_BYTES;
static constexpr uint32_t SYNC_ALL_CYCLES       = 7500;

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

    double LowerBound() const { return (vectorBusy > dmaBusy ? vectorBusy : dmaBusy) + syncCycles; }
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
    }

    // A scratchpad buffer the pipe has carved: blocks are numbered in claim order
    void Register(const void* base, size_t bytes) {
        const uint32_t n = static_cast<uint32_t>((bytes + DMA_ALIGN_BYTES - 1) / DMA_ALIGN_BYTES);
        if (nRegions == MAX_REGIONS || nextBlock + n > BLOCKS) return;  // Untracked: no ordering
        const uintptr_t a = reinterpret_cast<uintptr_t>(base);
        regions[nRegions++] = {a, a + bytes, nextBlock};
        for (uint32_t b = nextBlock; b < nextBlock + n; ++b) ready[b] = lastRead[b] = floorReady[b] = 0;
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

    // A vector instruction writing `w` and reading `r0`, `r1`
    void Vector(double cycles, Span w, Span r0 = {}, Span r1 = {}) {
        const double s = Start(VECTOR, w, r0, r1), e = s + cycles;
        Occupy(VECTOR, s, cycles);
        Read(r0, e);
        Read(r1, e);
        Write(w, e);
        const double fe = FloorStart(VECTOR, r0, r1) + cycles;
        floorFree[VECTOR] = fe;
        FloorWrite(w, fe);
    }
    // System memory -> scratchpad
    void Load(double bytes, Span w) {
        const double s = Start(DMA, w, {}, {}), occupied = bytes / DMA_BYTES_PER_CYCLE;
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
        const double s = Start(DMA, {}, r, {}), occupied = bytes / DMA_BYTES_PER_CYCLE;
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
    // Every unit waits until the core is drained (PipeBarrier<PIPE_ALL>)
    void DrainAll() {
        const double t = Drained(), f = FloorDrained();
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
    PIPE_ALL = 3
};

template <PipeType pipe>
inline void PipeBarrier() {
    #if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("" ::: "memory");
    #endif
    g_cycleTracker.barrierCycles += 20;
    if (pipe == PIPE_ALL) g_timeline.DrainAll();
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

// Standard OpenMP-style CPU thread indexing
inline uint32_t GetThreadIdx() {
    return static_cast<uint32_t>(omp_get_thread_num());
}

inline uint32_t GetThreadNum() {
    return static_cast<uint32_t>(omp_get_num_threads());
}

// -----------------------------------------------------------------------------
// LocalTensor<T> Implementation for Core-Local Scratchpad
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
        g_timeline.Vector(500, {}, SpanOf(data + index, 1));
        return data[index];
    }

    // The scalar unit writes in the vector pipe's order (timeline: no cost)
    inline void SetValue(uint32_t index, T val) {
        g_timeline.Vector(0, SpanOf(data + index, 1));
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
        g_timeline.Register(Get<uint8_t>().GetData(), tensorBytes);
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
            bufferPool[d].assign(d < buffers ? tensorBytes + 64 : 0, 0);
            state[d] = FREE;
            if (d < buffers) g_timeline.Register(Base(d), tensorBytes);
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
// Streaming DataCopy Primitives with Integrated 32-Byte DMA Alignment Guard
// -----------------------------------------------------------------------------
// Both endpoints of a block DMA must sit on a 32-byte block boundary
inline void CheckDmaAddress(const void* systemMem, const void* local) {
    if (reinterpret_cast<uintptr_t>(systemMem) % DMA_ALIGN_BYTES != 0 ||
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
    g_timeline.Load(static_cast<double>(copyBytes), SpanOf(dst.GetData(), count));
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
    g_timeline.Store(static_cast<double>(copyBytes), SpanOf(src.GetData(), count));
}

// -----------------------------------------------------------------------------
// Padded DMA for transfers that are not whole 32-byte blocks (row tails, tensor ends).
// System memory may be at any address and length; the scratchpad side stays block
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
    g_timeline.Load(static_cast<double>(blockBytes), g_timeline.SpanOf(dst.GetData(), blockBytes));
}

template <typename T>
inline void DataCopyPad(T* dst, LocalTensor<T> src, uint32_t count) {
    const size_t copyBytes = count * sizeof(T);
    if (reinterpret_cast<uintptr_t>(src.GetData()) % DMA_ALIGN_BYTES != 0) {
        throw std::runtime_error("[Hardware Fault - DMA UNALIGNED]: DataCopyPad scratchpad side must be 32-byte aligned!");
    }
    std::memcpy(dst, src.GetData(), copyBytes);
    const size_t blockBytes = (copyBytes + DMA_ALIGN_BYTES - 1) / DMA_ALIGN_BYTES * DMA_ALIGN_BYTES;
    g_cycleTracker.dmaBytesMoved += blockBytes;
    g_cycleTracker.dmaTransfers++;
    g_cycleTracker.padTransfers++;
    g_timeline.Store(static_cast<double>(blockBytes), SpanOf(src.GetData(), count));
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

// Element-wise format conversion (e.g. FP16/BF16 <-> FP32).
template <typename D, typename S, typename Convert>
inline void Cast(LocalTensor<D> dst, LocalTensor<S> src, uint32_t count, Convert convert) {
    for (uint32_t i = 0; i < count; ++i) {
        dst.data[i] = convert(src.data[i]);
    }
    const size_t widest = sizeof(D) > sizeof(S) ? sizeof(D) : sizeof(S);
    uint32_t repeats = static_cast<uint32_t>((count * widest + SIMD_REPEAT_BYTES - 1) / SIMD_REPEAT_BYTES);
    g_cycleTracker.vCastCycles += 2 * repeats + 13;
    g_timeline.Vector(2 * repeats + 13, SpanOf(dst.data, count), SpanOf(src.data, count));
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

template <typename T>
inline void Rsqrt(LocalTensor<T> dst, LocalTensor<T> src, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        dst.data[i] = static_cast<T>(1.0f / std::sqrt(static_cast<float>(src.data[i])));
    }
    uint32_t repeats = (count * sizeof(T) + SIMD_REPEAT_BYTES - 1) / SIMD_REPEAT_BYTES;
    g_cycleTracker.vRsqrtCycles += 2 * repeats + 14;
    g_timeline.Vector(2 * repeats + 14, SpanOf(dst.data, count), SpanOf(src.data, count));
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
    g_timeline.Vector(1 * repeats + 14, SpanOf(dst.data, outCount), SpanOf(src.data, count));
}

// -----------------------------------------------------------------------------
// Pure Vector Binary Reduction Tree (VectorReduceSum)
// Folds local vector elements into 1 scalar without scalar loop bubbles
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
    g_timeline.Vector((repeats + 1) * 2 + 13, {}, SpanOf(src.data, count));
    return sum;
}

// -----------------------------------------------------------------------------
// Pure Vector InvRms with Newton-Raphson Iteration (VectorInvRms)
// -----------------------------------------------------------------------------
inline float VectorInvRms(float sumSq, uint32_t D, float eps) {
    float x = sumSq / static_cast<float>(D) + eps;
    float inv = static_cast<float>(1.0 / std::sqrt(static_cast<double>(x)));
    g_cycleTracker.vRsqrtCycles += 2 + 14;
    g_timeline.Vector(2 + 14, {});
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
    g_timeline.Vector(14 * repeats + 14, SpanOf(dst.data, 1), SpanOf(src.data, count));
}

} // namespace dsa
