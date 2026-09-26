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

#ifndef DAE_HALF_DEFINED
#define DAE_HALF_DEFINED
struct half {
    uint16_t data = 0;
    half() = default;
    half(float f) {
        uint32_t x;
        std::memcpy(&x, &f, 4);
        uint32_t sign = (x >> 16) & 0x8000;
        int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
        uint32_t mant = (x >> 13) & 0x3ff;
        if (exp <= 0) {
            data = static_cast<uint16_t>(sign);
        } else if (exp >= 31) {
            data = static_cast<uint16_t>(sign | 0x7c00);
        } else {
            data = static_cast<uint16_t>(sign | (exp << 10) | mant);
        }
    }
    operator float() const {
        uint32_t sign = (data & 0x8000) << 16;
        int32_t exp = (data >> 10) & 0x1f;
        uint32_t mant = (data & 0x3ff) << 13;
        if (exp == 0) {
            if (mant == 0) return (sign ? -0.0f : 0.0f);
            exp = 1;
            while (!(mant & 0x00800000)) { mant <<= 1; exp--; }
            mant &= 0x007fffff;
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

// Global Memory Descriptor (Co-processor system memory buffer)
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
    PIPE_ALL = 3,
    PIPE_MTE2 = 1,
    PIPE_MTE3 = 2,
    PIPE_S = 4
};

template <PipeType pipe>
inline void PipeBarrier() {
    #if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("" ::: "memory");
    #endif
    g_cycleTracker.barrierCycles += 20;
    if (pipe == PIPE_ALL) g_timeline.DrainAll();
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
// DAE v1.5 Point-to-Point Pipeline Scoreboard Fences (HardEvent Tokens)
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

template <HardEvent E>
inline void WaitFlag(uint8_t eventId = EVENT_ID0) {
    (void)eventId;
    #if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("" ::: "memory");
    #endif
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
// LocalTensor<T> Implementation for Core-Local Scratchpad
// -----------------------------------------------------------------------------
template <typename T>
class LocalTensor {
public:
    T* data = nullptr;
    uint32_t count = 0;
    uint32_t capacityBytes = 0;
    QuePosition pos = QuePosition::TOTAL_POSITIONS;

    LocalTensor() = default;
    LocalTensor(T* ptr, uint32_t numElems, uint32_t capBytes, QuePosition p = QuePosition::TOTAL_POSITIONS)
        : data(ptr), count(numElems), capacityBytes(capBytes), pos(p) {}

    // Sub-tensor slicing operator: tensor[offset] returns sliced sub-tensor view
    inline LocalTensor<T> operator[](uint32_t offset) const {
        return LocalTensor<T>(data + offset, (count > offset) ? (count - offset) : 0, 
                              (capacityBytes > offset * sizeof(T)) ? (capacityBytes - offset * sizeof(T)) : 0, pos);
    }

    inline LocalTensor<T> operator+(uint32_t offset) const {
        return LocalTensor<T>(data + offset, (count > offset) ? (count - offset) : 0, 
                              (capacityBytes > offset * sizeof(T)) ? (capacityBytes - offset * sizeof(T)) : 0, pos);
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
// DAE v1.5 Zero-Queue Direct Scratchpad Memory Allocator (LocalMemAllocator)
// Allows direct static local tensor allocation without FIFO queue sequencer tax.
// -----------------------------------------------------------------------------
namespace Hardware {
    struct Scratchpad {};
    using SPM = Scratchpad;
    using UB = Scratchpad; // Unified Scratchpad Buffer
}

template <typename TargetSpace = Hardware::UB>
class LocalMemAllocator {
public:
    alignas(64) uint8_t pool[SCRATCHPAD_SAFE_WATERLINE];
    size_t offset = 0;

    LocalMemAllocator() = default;

    template <typename T, size_t N>
    LocalTensor<T> Alloc() {
        size_t bytes = (N * sizeof(T) + DMA_ALIGN_BYTES - 1) / DMA_ALIGN_BYTES * DMA_ALIGN_BYTES;
        if (offset + bytes > SCRATCHPAD_SAFE_WATERLINE) {
            throw std::runtime_error("[Hardware Fault - SCRATCHPAD OVERFLOW (Trap #400)]: "
                                     "LocalMemAllocator allocation (" + std::to_string(offset + bytes) +
                                     " bytes) exceeds 191 KB scratchpad waterline!");
        }
        uint8_t* ptr = pool + offset;
        offset += bytes;
        g_timeline.Register(ptr, bytes);
        return LocalTensor<T>(reinterpret_cast<T*>(ptr), static_cast<uint32_t>(N),
                              static_cast<uint32_t>(bytes), QuePosition::VECCALC);
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

    // -------------------------------------------------------------------------
    // Hardware Guard: DMA Egress Routing Channel Verification (PIPE_DMA_OUT)
    // Destination is system memory (egress transfer). The target streaming DMA
    // engine routes egress transfers strictly through Channel DMA_OUT
    // (PIPE_DMA_OUT), which is wired to QuePosition::VECOUT. Egress via VECIN
    // causes crossbar interconnect deadlock and pipeline stall.
    // -------------------------------------------------------------------------
    if (src.pos == QuePosition::VECIN) {
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
    if (src.pos == QuePosition::VECIN) {
        throw std::runtime_error("[Hardware Fault - INVALID DMA EGRESS CHANNEL]: "
                                 "DataCopyPad egress to system memory attempted from QuePosition::VECIN! "
                                 "Egress transfers strictly require QuePosition::VECOUT (Trap #401).");
    }
    std::memcpy(dst, src.GetData(), copyBytes);
    const size_t blockBytes = (copyBytes + DMA_ALIGN_BYTES - 1) / DMA_ALIGN_BYTES * DMA_ALIGN_BYTES;
    g_cycleTracker.dmaBytesMoved += blockBytes;
    g_cycleTracker.dmaTransfers++;
    g_cycleTracker.padTransfers++;
    g_timeline.Store(static_cast<double>(blockBytes), SpanOf(src.GetData(), count));
}

template <typename T>
inline void DataCopy(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t count) {
    DataCopy(dst, src.GetData(), count);
}

template <typename T>
inline void DataCopy(GlobalTensor<T> dst, LocalTensor<T> src, uint32_t count) {
    DataCopy(dst.GetData(), src, count);
}

template <typename T>
inline void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t count) {
    DataCopyPad(dst, src.GetData(), count);
}

template <typename T>
inline void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src, uint32_t count) {
    DataCopyPad(dst.GetData(), src, count);
}

template <typename T>
inline void DataCopyPad(LocalTensor<T> dst, const T* src, DataCopyExtParams cp, DataCopyPadExtParams<T> pad = {}) {
    (void)pad;
    uint32_t count = (sizeof(T) > 0) ? (cp.blockLen / sizeof(T)) : 0;
    DataCopyPad(dst, src, count);
}

template <typename T>
inline void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src, DataCopyExtParams cp, DataCopyPadExtParams<T> pad = {}) {
    DataCopyPad(dst, src.GetData(), cp, pad);
}

template <typename T>
inline void DataCopyPad(T* dst, LocalTensor<T> src, DataCopyExtParams cp) {
    uint32_t count = (sizeof(T) > 0) ? (cp.blockLen / sizeof(T)) : 0;
    DataCopyPad(dst, src, count);
}

template <typename T>
inline void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src, DataCopyExtParams cp) {
    DataCopyPad(dst.GetData(), src, cp);
}

// -----------------------------------------------------------------------------
// Direct Byte-Precise Padded Transfer Shorthands (LoadPad / StorePad)
// Tolerates non-32B aligned element counts with hardware zero-fill.
// -----------------------------------------------------------------------------
template <typename T>
inline void LoadPad(LocalTensor<T> dst, const T* src, uint32_t count) {
    DataCopyPad(dst, src, count);
}

template <typename T>
inline void LoadPad(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t count) {
    DataCopyPad(dst, src.GetData(), count);
}

template <typename T>
inline void StorePad(T* dst, LocalTensor<T> src, uint32_t count) {
    DataCopyPad(dst, src, count);
}

template <typename T>
inline void StorePad(GlobalTensor<T> dst, LocalTensor<T> src, uint32_t count) {
    DataCopyPad(dst.GetData(), src, count);
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

template <typename D, typename S>
inline void Cast(LocalTensor<D> dst, LocalTensor<S> src, uint32_t count) {
    Cast(dst, src, count, [](S x) { return static_cast<D>(x); });
}

template <typename D, typename S>
inline void Cast(LocalTensor<D> dst, LocalTensor<S> src, RoundMode mode, uint32_t count) {
    (void)mode;
    Cast(dst, src, count, [](S x) { return static_cast<D>(x); });
}

template <typename D, typename S>
inline void Cast(LocalTensor<D> dst, LocalTensor<S> src, RoundMode mode, uint64_t count, uint8_t rep, UnaryRepeatParams params) {
    (void)mode; (void)rep; (void)params;
    Cast(dst, src, static_cast<uint32_t>(count), [](S x) { return static_cast<D>(x); });
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
    if (dst.GetData() == src.GetData()) {
        throw std::runtime_error("[Hardware Fault - VECTOR ALU OPERAND ALIASING]: "
                                 "BlockReduceSum destination buffer aliases source buffer (dst == src)! "
                                 "The 256-bit SIMD vector pipeline forbids in-place folding within the "
                                 "same buffer due to lack of sub-vector RAW forwarding across 256-bit "
                                 "burst lanes. Intra-row folding must ping-pong across disjoint buffers (Trap #402).");
    }
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
// Pure Vector InvRms with Coordinator-Precomputed Invariant Scale (invD)
// -----------------------------------------------------------------------------
inline float VectorInvRms(float sumSq, float invD, float eps) {
    float x = sumSq * invD + eps;
    float inv = static_cast<float>(1.0 / std::sqrt(static_cast<double>(x)));
    g_cycleTracker.vRsqrtCycles += 2 + 14;
    g_timeline.Vector(2 + 14, {});
    return inv;
}

// Deprecated overload: Worker core performing scalar integer-to-float conversion
[[deprecated("Worker core lacks scalar integer-to-float conversion unit. Pass precomputed invD from Coordinator.")]]
inline float VectorInvRms(float sumSq, uint32_t D, float eps) {
    g_cycleTracker.scalarStallCycles += 20; // 20-cycle scalar conversion penalty
    g_cycleTracker.scalarStallCount++;
    return VectorInvRms(sumSq, 1.0f / static_cast<float>(D), eps);
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

// Whole block reduction overload matching native hardware strides
template <typename T>
inline void WholeReduceSum(LocalTensor<T> dst, LocalTensor<T> src, uint64_t mask, uint8_t repeatTimes,
                           uint8_t srcRepStride, uint8_t dstRepStride, uint8_t dstBlkStride) {
    (void)mask; (void)repeatTimes; (void)srcRepStride; (void)dstRepStride; (void)dstBlkStride;
    T sum = 0;
    for (uint32_t i = 0; i < 64; ++i) {
        sum += src.data[i];
    }
    dst.data[0] = sum;
    g_cycleTracker.vWholeReduceCycles += 14;
    g_timeline.Vector(14, SpanOf(dst.data, 1), SpanOf(src.data, 64));
}

// -----------------------------------------------------------------------------
// DAE v1.5 Vector Reduction with 64-Lane SIMD Geometry & Aliasing Guard
// -----------------------------------------------------------------------------
template <typename T>
inline void ReduceSum(LocalTensor<T> dst, LocalTensor<T> src, LocalTensor<T> work, uint32_t count) {
    if (dst.GetData() == src.GetData() || dst.GetData() == work.GetData() || src.GetData() == work.GetData()) {
        throw std::runtime_error("[Hardware Fault - VECTOR ALU OPERAND ALIASING (Trap #402)]: "
                                 "ReduceSum destination, source, and scratch workpad buffers must be strictly disjoint! "
                                 "The SIMD reduction datapath forbids workspace aliasing.");
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
    uint32_t repeats = (count * sizeof(T) + SIMD_REPEAT_BYTES - 1) / SIMD_REPEAT_BYTES;
    g_cycleTracker.vBlockReduceCycles += 2 * repeats + 15;
    g_timeline.Vector(2 * repeats + 15, SpanOf(dst.data, 1), SpanOf(src.data, count));
}

// -----------------------------------------------------------------------------
// DAE v1.5 Cross-Lane Hardware Broadcast Primitive (Brcb)
// Single-cycle broadcast from scalar into 64-lane SIMD vector buffer.
// -----------------------------------------------------------------------------
struct BrcbRepeatParams {
    uint8_t dstRepStride = 1;
    uint8_t srcRepStride = 8;
};

template <typename T>
inline void Brcb(LocalTensor<T> dst, LocalTensor<T> src, uint32_t repeatCount = 1, BrcbRepeatParams params = {1, 8}) {
    (void)params;
    const size_t minCapacity = 64 * repeatCount;
    if (dst.capacityBytes < minCapacity * sizeof(T)) {
        throw std::runtime_error("[Hardware Fault - BUFFER OVERRUN (Trap #410)]: "
                                 "Brcb destination buffer capacity (" + std::to_string(dst.capacityBytes) +
                                 " bytes) is smaller than required 256-byte hardware burst (64 elements / " +
                                 std::to_string(minCapacity * sizeof(T)) + " bytes)! "
                                 "Brcb operates via 8 x 32-byte SIMD lane bursts; underallocated buffers cause memory corruption.");
    }
    T val = src.data[0];
    for (uint32_t i = 0; i < minCapacity; ++i) {
        dst.data[i] = val;
    }
    g_cycleTracker.vCastCycles += 1 * repeatCount + 8;
    g_timeline.Vector(1 * repeatCount + 8, SpanOf(dst.data, minCapacity), SpanOf(src.data, 1));
}

// -----------------------------------------------------------------------------
// DAE v1.5 Strided Multi-Row Vector Operators & 8-Bit Stride Guard
// -----------------------------------------------------------------------------
struct BinaryRepeatParams {
    uint8_t dstRepStride = 1;
    uint8_t src0RepStride = 1;
    uint8_t src1RepStride = 1;
    uint8_t dstBlkStride = 8;
    uint8_t src0BlkStride = 8;
    uint8_t src1BlkStride = 0;
};

template <typename T>
inline void Mul(LocalTensor<T> dst, LocalTensor<T> src0, LocalTensor<T> src1, uint64_t count, uint8_t rows, BinaryRepeatParams rep) {
    (void)rep;
    for (uint8_t r = 0; r < rows; ++r) {
        for (uint64_t i = 0; i < count; ++i) {
            dst.data[r * count + i] = src0.data[r * count + i] * src1.data[i];
        }
    }
    uint32_t repeats = static_cast<uint32_t>((count * rows * sizeof(T) + SIMD_REPEAT_BYTES - 1) / SIMD_REPEAT_BYTES);
    g_cycleTracker.vMulCycles += 2 * repeats + 13;
    g_timeline.Vector(2 * repeats + 13, SpanOf(dst.data, count * rows), SpanOf(src0.data, count * rows), SpanOf(src1.data, count));
}

template <typename T>
inline void Add(LocalTensor<T> dst, LocalTensor<T> src0, LocalTensor<T> src1, uint64_t count, uint8_t rows, BinaryRepeatParams rep) {
    (void)rep;
    for (uint8_t r = 0; r < rows; ++r) {
        for (uint64_t i = 0; i < count; ++i) {
            dst.data[r * count + i] = src0.data[r * count + i] + src1.data[i];
        }
    }
    uint32_t repeats = static_cast<uint32_t>((count * rows * sizeof(T) + SIMD_REPEAT_BYTES - 1) / SIMD_REPEAT_BYTES);
    g_cycleTracker.vAddCycles += 2 * repeats + 13;
    g_timeline.Vector(2 * repeats + 13, SpanOf(dst.data, count * rows), SpanOf(src0.data, count * rows), SpanOf(src1.data, count));
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

} // namespace dsa
