#pragma once

#include "dsa_runtime.hpp"
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>
#if __has_include(<unistd.h>)
#include <unistd.h>
#endif

namespace hpc {

enum class TilingMode {
    ROW_PARALLEL,  // Units of whole, DMA-aligned rows (zero cross-core communication)
    SPLIT_D,       // Units of one 32-byte block, row-major: <= 2 shared rows per core + one SyncAll
    SPLIT_COLUMNS  // Units of one 32-byte block, column-block-major: every core owns a column band
                   // of every row and streams only that band of gamma/beta; one SyncAll (DAE only)
};

// -----------------------------------------------------------------------------
// Vector-cycle costs of include/dsa_runtime.hpp (one repeat = 256 bytes = 64 floats).
// The planner's cycle model and the kernel's own instruction choices share them.
// -----------------------------------------------------------------------------
struct DaeIsa {
    static constexpr uint64_t Repeats(uint64_t floats) {
        return (floats * sizeof(float) + dsa::SIMD_REPEAT_BYTES - 1) / dsa::SIMD_REPEAT_BYTES;
    }
    static constexpr uint64_t Op(uint64_t n) { return 2 * Repeats(n) + 13; }      // Add, Adds, Mul, Muls, Cast
    static constexpr uint64_t Strided(uint64_t repeats) { return 2 * repeats + 13; }  // Mul/Add, strided form
    static constexpr uint64_t FoldReps(uint64_t repeats) { return repeats + 14; }  // BlockReduceSum (8 -> 1), per repeat
    static constexpr uint64_t Fold(uint64_t n) { return FoldReps(Repeats(n)); }   // ... over n contiguous elements
    static constexpr uint64_t Reduce(uint64_t n) { return 2 * Repeats(n) + 15; }  // ReduceSum
    static constexpr uint64_t Fill(uint64_t n) { return Repeats(n) + 18; }        // Duplicate
    static constexpr uint64_t Rsqrt(uint64_t n) { return 2 * Repeats(n) + 14; }   // Rsqrt
    static constexpr uint64_t Brcb(uint64_t repeats) { return repeats + 8; }      // Brcb: 8 blocks per repeat
    // Sum `runs` runs of `len` squares (whole 64-lane repeats): one 8 -> 1 fold first when that is
    // cheaper and leaves whole repeats for ReduceSum
    static constexpr bool FoldFirst(uint64_t runs, uint64_t len) {
        return len % 512 == 0 && Fold(runs * len) + runs * Reduce(len / 8) < runs * Reduce(len);
    }
    static constexpr uint64_t ReduceRuns(uint64_t runs, uint64_t len) {
        return FoldFirst(runs, len) ? Fold(runs * len) + runs * Reduce(len / 8) : runs * Reduce(len);
    }
};

// -----------------------------------------------------------------------------
// Machine description. Every number the planning equations use comes from here:
//   Target()  the deployment machine: 40 symmetric DAE cores, 32-byte DMA blocks,
//             191 KB scratchpad per core, and the runtime's timing (include/dsa_runtime.hpp),
//             measured on the target (docs/TARGET_MEASUREMENTS.md): kernel launch, SyncAll, the
//             memory system (dsa::MemorySystem) and the time base of the cycle costs.
//   Host(P)   the same machine laws run by P CI threads with the host's measured costs.
// -----------------------------------------------------------------------------
struct HardwareModel {
    uint32_t cores;         // Symmetric cores
    uint32_t quantumBytes;  // DMA block: split boundaries and block transfers are multiples of it
    size_t   spmBytes;      // Per-core scratchpad budget
    double   launchNs;      // Target: the kernel launch, paid by every plan whatever its core count; host: fork/join
    double   syncNs;        // One all-core barrier (SyncAll)
    double   elemNs;        // Host: vector work per element on one core
    double   latencyNs;     // Target: a DMA transfer's data lands this long after it streamed; 0 on the host
    double   clockGHz;      // Target: converts the runtime's vector cycles (DaeIsa) to ns; 0 on the host
    uint32_t batchElems;    // Host: rows whose pass-1 Z stays L1-resident together

    static HardwareModel Target() {
        HardwareModel hw{};
        hw.cores = dsa::MAX_HARDWARE_CORES;
        hw.quantumBytes = dsa::DMA_ALIGN_BYTES;
        hw.spmBytes = dsa::SCRATCHPAD_SAFE_WATERLINE;
        hw.launchNs = dsa::KERNEL_LAUNCH_NS;                     // 1.70 us, measured: any core count
        hw.clockGHz = dsa::CLOCK_GHZ;                            // The measured repeat time of a vector pass
        hw.syncNs = dsa::SYNC_ALL_CYCLES / hw.clockGHz;          // 0.924 us, measured
        hw.elemNs = 0.0;                                         // Vector work: the DaeIsa cycle model
        hw.latencyNs = dsa::DMA_LATENCY_CYCLES / hw.clockGHz;    // Not measured (dsa_runtime.hpp)
        hw.batchElems = 0;
        return hw;
    }

    // The target's timing in vector cycles, as the timeline models use it
    double LatencyCycles() const { return latencyNs * clockGHz; }
    double SyncCycles() const { return syncNs * clockGHz; }
    // The memory system of a launch of `cores` cores over a working set of `workingSet` bytes: what
    // each of its transfers costs the core's DMA channel
    static dsa::MemorySystem Memory(uint32_t cores, double workingSet) { return dsa::MemorySystem(cores, workingSet); }

    static HardwareModel Host(uint32_t threads) {
        HardwareModel hw = Target();
        hw.cores = std::max(1u, std::min(threads, hw.cores));
        hw.launchNs = 3500.0;  // Measured on the 4-core CI VM (docs)
        hw.syncNs = 1000.0;
        hw.elemNs = 0.3;       // Cache-resident rows are compute-bound: any dtype
        hw.latencyNs = 0.0;    // Hardware prefetchers, no explicit DMA
        hw.clockGHz = 0.0;     // No DAE cycle model: SIMD work is elemNs
        hw.batchElems = 2048;
        return hw;
    }
};

// Per-core scratchpad layout of the DAE pipeline. The kernel claims exactly these
// buffers through TPipe, so the planner's total is what the sanitizer checks. Ingress tiles
// sit in VECIN queues; every byte that leaves for system memory leaves from a VECOUT buffer
// (`out` or `rec`) [ARCH CHALLENGE 7: the egress DMA channel reads VECOUT only].
struct DaeLayout {
    uint32_t tile;          // One X1 / X2 / parameter-chunk buffer (native dtype, VECIN), `depth` of each
    uint32_t z;             // FP32 Z of one tile (FP32 row tiles build Z in their egress buffer)
    uint32_t tmp;           // FP32 scratch: the chunk (casts, squares, staged gamma/beta) and the reduction partitions
    uint32_t params;        // Resident FP32 gamma + beta, one row each at the tile's pitch
    uint32_t resident;      // Resident FP32 Z: column-tiled segments, or a core's whole column band
    uint32_t misc;          // Split-D partial-sum records gathered from every core
    bool paramQueue;        // Column tiles stream gamma and beta chunks, a pair per tile
    uint32_t depth = 2;     // Buffers per input queue: tiles in flight
    uint32_t out = 0;       // One egress buffer (VECOUT, native dtype): Y leaves only from these
    uint32_t outDepth = 1;  // Egress buffers: a result streams out while the next ones are written
    uint32_t rec = 0;       // Split-D: this core's published record (VECOUT)
    uint32_t Total() const {
        return depth * tile * (paramQueue ? 4 : 2) + outDepth * out + z + tmp + params + resident + misc + rec;
    }
};

struct TilingConfig {
    TilingMode mode;
    uint32_t blocks;         // Cores that receive work: min(cores, units)
    uint32_t threads;        // Host team: 1 (inline) or all cores, never resized
    uint64_t unitElems;      // Work unit in the flattened M*D stream
    uint64_t units;          // ceil(M*D / unitElems)
    // Host SIMD executor
    uint32_t batchRows;      // Rows whose reductions are in flight together (hides the sqrt latency)
    uint32_t chunkRows;      // Serpentine granularity
    uint32_t residentElems;  // Per-thread resident Z scratchpad (FP32 elements)
    bool streamStores;       // Non-temporal Y stores: working set overflows the last-level cache
    bool serpentine;         // Reverse the chunk order on every other call
    // Target DAE pipeline (per core)
    uint32_t tileRows;       // Rows per tile: row tiles and column bands (0: column tiles)
    uint32_t headRows;       // Row tiles: rows of a core's first tile (0: tileRows)
    uint32_t tailRows;       // Row tiles: rows of a core's last tile (0: what is left)
    bool paramsFirst;        // Row tiles: gamma/beta load before tile 0 (else right after it)
    bool earlyLoads;         // Row tiles: X1/X2 of a later tile load as soon as the tile's Z is built (else after its store)
    uint32_t tileElems;      // Elements per X1/X2 tile (row tiles: at the pitch)
    uint32_t pitch;          // Row pitch inside a tile: D on the 32-byte grid (RowPitch), or the column band width
    uint32_t zResident;      // FP32 Z elements kept resident (column tiles and bands)
    DaeLayout layout;
    double modelNs;          // Modeled time: what Plan() minimizes (target: the launch, then the slowest core's timeline)
    uint64_t modelCycles;    // Target: modeled vector cycles of the busiest core (barriers excluded)
    double modelFinish;      // Target: the slowest core's modeled timeline finish, in vector cycles (launch excluded)
};

// A core's share of the flattened M*D stream: units [floor(U t/n), floor(U (t+1)/n)), i.e.
// complete rows [rowA, rowZ) plus at most two fragments shared with neighbouring cores.
// SPLIT_COLUMNS numbers the units column-block-major (u = j*M + i for block j of row i), so
// the share is a column band of every row: row i covers 32-byte blocks
// [j0 + (i < a0), j1 + (i < a1)).
struct CoreRange {
    struct Fragment { uint32_t row, cb, ce; };
    Fragment frag[2];
    uint32_t nFrag = 0, rowA = 0, rowZ = 0;
    uint32_t j0 = 0, a0 = 0, j1 = 0, a1 = 0;
};

class AdaptiveTiler {
public:
    // A core's row tiles over [rA, rZ): a head tile, body tiles of B rows, then a tail tile
    // (head/tail 0: none).
    struct RowSchedule {
        uint32_t rA, rZ, B, head, tail;
        uint32_t Rows(uint32_t row) const {
            const uint32_t left = rZ - row;
            if (row == rA && head && head < left) return head;
            if (tail && left > tail) return dsa::Min(B, left - tail);
            return dsa::Min(B, left);
        }
    };

    static constexpr uint32_t MAX_THREADS = dsa::MAX_HARDWARE_CORES;
    static constexpr uint32_t MAX_BATCH = 64;
    static constexpr double CHUNK_BYTES = 64 * 1024;     // Host serpentine chunk (prefetch-friendly run)
    static constexpr uint32_t LANES = dsa::SIMD_REPEAT_BYTES / sizeof(float);  // FP32 lanes of one repeat
    static constexpr uint32_t ROW_GROUP = 128;           // DAE rows whose inverse RMS one lane chain computes
    static constexpr uint32_t RSQRT_BITS = dsa::RSQRT_TABLE_BITS;  // Rsqrt table precision: Newton-Raphson refines it
    // The DAE scratch partitions, in FP32 elements, each a TBuf of its own:
    //   chunk  casts, squares and staged gamma/beta
    //   fold   an 8 -> 1 fold of the chunk [ARCH CHALLENGE 7: no ALU operand aliasing]
    //   slots  a row group's 8 partial sums per row, which Brcb then overwrites with the group's
    //          inverse RMS, a block per row; column tiles: a segment's 32-byte slot, whose lane 0 alone
    //          a ReduceSum defines [6.2], and the block its inverse RMS is spread over; a spare slot
    //   sums, inv, nr  a group's packed row sums, their inverse RMS and Newton-Raphson term
    //   work   the ReduceSum workpad; long rows accumulate their 64-lane column sums in it
    static constexpr uint32_t TMP_BYTES = 8192;
    static constexpr uint32_t TMP_FLOATS = TMP_BYTES / sizeof(float);
    static constexpr uint32_t FOLD_FLOATS = TMP_FLOATS / 8;
    static constexpr uint32_t SLOT_FLOATS = 8 * (ROW_GROUP + 1);
    static constexpr uint32_t LANE_FLOATS = ROW_GROUP;
    static constexpr uint32_t WORK_FLOATS = LANES;
    static constexpr uint32_t SCRATCH_BYTES = (TMP_FLOATS + FOLD_FLOATS + SLOT_FLOATS + 3 * LANE_FLOATS + WORK_FLOATS) * sizeof(float);
    // Column tiles: slots 0 and 1 hold the Split-D fragments' sums, and a segment's inverse RMS is
    // spread over the 8 blocks from COLUMN_BC_SLOT
    static constexpr uint32_t COLUMN_BC_SLOT = 8;

    static inline CoreRange Range(const TilingConfig& cfg, uint32_t M, uint32_t D, uint32_t t, uint32_t n) {
        CoreRange r;
        if (cfg.mode == TilingMode::SPLIT_COLUMNS) {
            const uint64_t u0 = cfg.units * t / n, u1 = cfg.units * (t + 1) / n;
            if (u0 >= u1 || M == 0) return r;
            r.rowZ = M;
            r.j0 = static_cast<uint32_t>(u0 / M);
            r.a0 = static_cast<uint32_t>(u0 % M);
            r.j1 = static_cast<uint32_t>(u1 / M);
            r.a1 = static_cast<uint32_t>(u1 % M);
            return r;
        }
        if (n == 1) {  // One core owns every row: skips four 64-bit divisions (~45 ns of a 1x64 call)
            r.rowZ = D ? M : 0;
            return r;
        }
        const uint64_t total = static_cast<uint64_t>(M) * D;
        const uint64_t e0 = dsa::Min(cfg.units * t / n * cfg.unitElems, total);
        const uint64_t e1 = dsa::Min(cfg.units * (t + 1) / n * cfg.unitElems, total);
        if (e0 >= e1) return r;
        const uint32_t r0 = static_cast<uint32_t>(e0 / D), r1 = static_cast<uint32_t>((e1 - 1) / D);
        const uint32_t c0 = static_cast<uint32_t>(e0 % D), c1 = static_cast<uint32_t>((e1 - 1) % D + 1);
        r.rowA = r0;
        r.rowZ = r1 + 1;
        if (c0 > 0 || (r0 == r1 && c1 < D)) { r.frag[r.nFrag++] = {r0, c0, r0 == r1 ? c1 : D}; ++r.rowA; }
        if (r1 > r0 && c1 < D) { r.frag[r.nFrag++] = {r1, 0, c1}; --r.rowZ; }
        return r;
    }

    // Cores sharing row `row` under an n-way split: owner(u) = floor(((u + 1) n - 1) / U)
    static inline void RowOwners(const TilingConfig& cfg, uint32_t D, uint32_t row, uint32_t n,
                                 uint32_t& first, uint32_t& last) {
        const uint64_t U = cfg.units, u0 = static_cast<uint64_t>(row) * D / cfg.unitElems;
        const uint64_t u1 = (static_cast<uint64_t>(row + 1) * D - 1) / cfg.unitElems;
        first = static_cast<uint32_t>(((u0 + 1) * n - 1) / U);
        last = static_cast<uint32_t>(((u1 + 1) * n - 1) / U);
    }

    // SPLIT_COLUMNS: row i's columns [cb, ce) (empty when cb >= ce) and the core's band
    static inline void BandRow(const CoreRange& r, uint32_t i, uint32_t q, uint32_t& cb, uint32_t& ce) {
        cb = (r.j0 + (i < r.a0 ? 1u : 0u)) * q;
        ce = (r.j1 + (i < r.a1 ? 1u : 0u)) * q;
    }
    static inline uint32_t BandBegin(const CoreRange& r, uint32_t q) { return r.j0 * q; }
    static inline uint32_t BandEnd(const CoreRange& r, uint32_t q) { return (r.j1 + (r.a1 > 0 ? 1u : 0u)) * q; }

    // Split-D partial records per core in system memory (whole 32-byte blocks)
    static inline uint32_t RecordFloats(const TilingConfig& cfg, uint32_t M) {
        if (cfg.mode == TilingMode::SPLIT_D) return 16;                      // {sum0, 0 x7}, {sum1, 0 x7}
        if (cfg.mode == TilingMode::SPLIT_COLUMNS) return (M + 7) / 8 * 8;  // One partial per row
        return 0;
    }

    // -------------------------------------------------------------------------
    // (1) Decomposition. Every candidate is a balanced split of the flattened M*D stream
    // into units, and Plan() keeps the one whose busiest core finishes first:
    //   inline         one core, no launch (host)
    //   ROW_PARALLEL   units of one row: on the target a row off the 32-byte grid starts a block of
    //                  its own in the scratchpad (a padded DMA descriptor per tile), so any row
    //                  count balances to one row
    //   SPLIT_D        units of one 32-byte block [ARCH CHALLENGE 1]: every core within one
    //                  block of the mean for any M and any core count, plus one SyncAll
    //   SPLIT_COLUMNS  the same blocks numbered column-major (target only): same balance, and
    //                  each core reads gamma/beta for its column band once instead of per row
    // Host cost: launch + barrier + elements * elemNs. Target cost: the kernel launch, then the
    // finish time of the slowest core on the timeline models of (3), SyncAll included.
    // -------------------------------------------------------------------------
    static inline TilingConfig Plan(uint32_t M, uint32_t D, uint32_t elemBytes, const HardwareModel& hw,
                                    size_t llcBytes = 0) {
        // Inline on the caller is a candidate where engaging the cores costs something the caller
        // does not pay (the host's fork/join); every target plan is a kernel launch, whatever its
        // core count, and one core cannot beat the same rows on min(cores, units) cores
        const bool inlineCandidate = (hw.clockGHz <= 0 && hw.launchNs > 0) || hw.cores < 2 || static_cast<uint64_t>(M) * D == 0;
        TilingConfig best{};
        bool have = false;
        if (inlineCandidate) {
            best = Build(M, D, elemBytes, hw, TilingMode::ROW_PARALLEL, false, llcBytes);
            have = true;
            if (hw.cores < 2 || static_cast<uint64_t>(M) * D == 0) return best;
        }
        for (const TilingMode mode : {TilingMode::ROW_PARALLEL, TilingMode::SPLIT_D, TilingMode::SPLIT_COLUMNS}) {
            if (mode == TilingMode::SPLIT_COLUMNS && hw.clockGHz <= 0) continue;  // DAE executor only
            // A candidate only has to be modeled until it cannot beat the best so far
            const TilingConfig c = Build(M, D, elemBytes, hw, mode, true, llcBytes, have ? best.modelNs : std::numeric_limits<double>::infinity());
            if (!have || c.modelNs < best.modelNs) best = c, have = true;  // Ties keep the simpler plan
        }
        return best;
    }

    // Completes a plan for a given decomposition (tests force every mode through here).
    static inline TilingConfig Build(uint32_t M, uint32_t D, uint32_t elemBytes, const HardwareModel& hw,
                                     TilingMode mode, bool parallel, size_t llcBytes = 0,
                                     double boundNs = std::numeric_limits<double>::infinity()) {
        TilingConfig cfg{};
        const uint64_t total = static_cast<uint64_t>(M) * D;
        const uint32_t P = std::max(1u, hw.cores);
        cfg.mode = mode;
        cfg.unitElems = mode == TilingMode::ROW_PARALLEL ? RowUnit(D, elemBytes, hw) : QuantumElems(elemBytes, hw);
        cfg.units = (total + cfg.unitElems - 1) / cfg.unitElems;
        cfg.blocks = parallel ? static_cast<uint32_t>(std::min<uint64_t>(P, std::max<uint64_t>(1, cfg.units))) : 1;
        cfg.threads = parallel ? P : 1;

        // (2) Host SIMD executor: L1-resident batches, resident Z, streaming stores past the LLC.
        cfg.batchRows = hw.batchElems ? std::max(1u, std::min(MAX_BATCH, hw.batchElems / std::max(1u, D))) : 1;
        cfg.chunkRows = std::max(1u, static_cast<uint32_t>(CHUNK_BYTES / std::max(1.0, 3.0 * D * elemBytes)));
        cfg.residentElems = static_cast<uint32_t>(hw.spmBytes / sizeof(float));
        cfg.streamStores = llcBytes && 3.0 * total * elemBytes > static_cast<double>(llcBytes);
        cfg.serpentine = true;

        if (hw.clockGHz > 0) {  // Explicit DMA pipeline: tiles, layout, modeled time (the bound less the launch)
            PlanTiles(cfg, M, D, elemBytes, hw, (boundNs - hw.launchNs) * hw.clockGHz);
        } else {
            const double sync = mode == TilingMode::ROW_PARALLEL ? 0.0 : hw.syncNs;
            cfg.modelNs = (cfg.threads > 1 ? hw.launchNs : 0.0) + sync + MaxLoad(total, cfg.unitElems, cfg.blocks) * hw.elemNs;
        }
        return cfg;
    }

    // Bytes of system memory a launch touches: X1, X2 and Y, gamma and beta
    static inline double WorkingSet(uint32_t M, uint32_t D, uint32_t s) { return (3.0 * M * D + 2.0 * D) * s; }

    // The memory system a plan's launch streams through: its cores over the tensor's working set
    static inline dsa::MemorySystem Memory(const TilingConfig& cfg, uint32_t M, uint32_t D, uint32_t s) {
        return HardwareModel::Memory(cfg.blocks, WorkingSet(M, D, s));
    }

    // A target plan's modeled time: the kernel launch, then its slowest core's timeline
    static inline void SetModel(TilingConfig& cfg, double finish, double vector, const HardwareModel& hw) {
        cfg.modelFinish = finish;
        cfg.modelNs = hw.launchNs + finish / hw.clockGHz;
        cfg.modelCycles = static_cast<uint64_t>(vector + 0.5);
    }

    // Busiest core's elements when `total` splits into units of `unit` over P cores
    static inline uint64_t MaxLoad(uint64_t total, uint64_t unit, uint32_t P) {
        const uint64_t U = (total + unit - 1) / unit;
        if (U == 0) return 0;
        const uint64_t T = std::min<uint64_t>(P, U);
        return std::min(total, (U + T - 1) / T * unit);
    }

    static inline uint32_t QuantumElems(uint32_t elemBytes, const HardwareModel& hw) {
        return std::max(1u, hw.quantumBytes / elemBytes);
    }

    // One row per unit: rows balance to one row for any width (a row off the 32-byte grid starts a
    // scratchpad block of its own, RowPitch)
    static inline uint64_t RowUnit(uint32_t D, uint32_t, const HardwareModel&) { return D ? D : 1; }

    static inline uint32_t Align32(uint64_t bytes) {
        return static_cast<uint32_t>((bytes + dsa::DMA_ALIGN_BYTES - 1) / dsa::DMA_ALIGN_BYTES * dsa::DMA_ALIGN_BYTES);
    }

    // A row's pitch in a tile, in elements: D rounded up to the 32-byte grid, so every row starts a
    // block [6.6] (and its FP32 form does too): what lets a strided instruction take every row of a
    // tile. The padded DMA fills the lanes past D with 0.
    static inline uint32_t RowPitch(uint32_t D, uint32_t s) { return Align32(uint64_t(D) * s) / s; }

    // Rows of `w` FP32 lanes on the 32-byte grid fit the strided form's 8-bit repeat stride (in blocks)
    static constexpr bool StridedRows(uint64_t w) { return w % 8 == 0 && w / 8 <= 255; }

    static constexpr uint64_t CeilDiv(uint64_t a, uint64_t b) { return (a + b - 1) / b; }
    static constexpr uint64_t NextPow2(uint64_t n) { return n <= 1 ? 1 : 2 * NextPow2((n + 1) / 2); }

    // gamma and beta: one FP32 row each at `pitch`, every row on the 32-byte grid
    static inline uint32_t ParamBytes(uint64_t pitch) { return 2 * Align32(4 * pitch); }

    // Row tiles: X1/X2 (`depth` each), `outDepth` egress buffers of a tile, and for 16-bit data
    // the FP32 Z tile (FP32 builds Z in the egress buffer), rows at RowPitch
    static inline DaeLayout RowLayout(uint64_t rows, uint32_t D, uint32_t s, uint32_t depth = 2, uint32_t outDepth = 1) {
        const uint64_t e = rows * RowPitch(D, s);
        return {Align32(e * s), s < 4 ? Align32(e * 4) : 0u, SCRATCH_BYTES, ParamBytes(RowPitch(D, s)), 0u, 0u,
                false, depth, Align32(e * s), outDepth, 0u};
    }

    static inline DaeLayout BandLayout(uint64_t rows, uint32_t pitch, uint32_t M, uint32_t s, uint32_t misc, uint32_t rec,
                                       uint32_t depth = 2, uint32_t outDepth = 1) {
        return {Align32(rows * pitch * s), 0u, SCRATCH_BYTES, ParamBytes(pitch), Align32(uint64_t(M) * pitch * 4), misc, false,
                depth, Align32(rows * pitch * s), outDepth, rec};
    }

    // Column tiles: X1, X2 and gamma/beta chunk pairs double-buffered, two egress buffers
    static inline DaeLayout ColumnLayout(uint64_t tile, uint64_t resident, uint32_t s, uint32_t misc, uint32_t rec) {
        return {Align32(tile * s), Align32(tile * 4), SCRATCH_BYTES, 0u, Align32(resident * 4), misc, true, 2, Align32(tile * s), 2, rec};
    }

    // ReduceSum's rows: whole 64-lane repeats (a count off the lanes hangs the reduction: Trap #408)
    static constexpr uint32_t LanePad(uint64_t n) { return static_cast<uint32_t>((n + LANES - 1) / LANES * LANES); }

    // Newton-Raphson steps after Rsqrt: each doubles the bits, until they exceed the output's
    // significand (16-bit outputs: one step, FP32: two)
    static inline uint32_t NewtonSteps(uint32_t s) {
        uint32_t steps = 0;
        for (uint32_t bits = RSQRT_BITS; bits <= (s == 4 ? 24u : 11u); bits *= 2) ++steps;
        return steps;
    }

    // Inverse RMS of n rows in vector lanes (DaePipeline::LaneInvRms): Muls by invD, Adds eps,
    // Rsqrt, and each Newton-Raphson step's four instructions after a setup
    static inline uint64_t LaneRmsCycles(uint64_t n, uint32_t s) {
        const uint32_t steps = NewtonSteps(s);
        return 2 * DaeIsa::Op(n) + DaeIsa::Rsqrt(n) + (steps ? (1 + 4 * steps) * DaeIsa::Op(n) : 0);
    }

    static inline size_t LastLevelCacheBytes() {
        static const size_t bytes = [] {
            long v = -1;
#if defined(_SC_LEVEL3_CACHE_SIZE)
            v = sysconf(_SC_LEVEL3_CACHE_SIZE);
            if (v <= 0) v = sysconf(_SC_LEVEL2_CACHE_SIZE);
#endif
            return v > 0 ? static_cast<size_t>(v) : static_cast<size_t>(32) << 20;
        }();
        return bytes;
    }

    // -------------------------------------------------------------------------
    // Vector-cycle model of DaePipeline, instruction by instruction (DaeIsa costs). The
    // planner uses it to weigh plans; tests hold it to the runtime's own cycle counts.
    // -------------------------------------------------------------------------
    // Z = X1 + X2 over e elements (Core::AddInputs): FP32 one add; 16-bit the widened X1, then X2
    // widened and added a chunk at a time
    static inline uint64_t BuildCycles(uint64_t e, uint32_t s) {
        uint64_t c = DaeIsa::Op(e);
        if (s < 4) {
            for (uint64_t o = 0; o < e; o += TMP_FLOATS) c += 2 * DaeIsa::Op(std::min<uint64_t>(TMP_FLOATS, e - o));
        }
        return c;
    }

    // Lane 0 of a slot (+)= the sum of squares of a run of n elements (Core::RunSum): chunk-sized
    // pieces, each squared after its last repeat was zeroed (so it pads with 0 to whole repeats) and
    // reduced; every piece after its segment's first reduces into the spare slot and adds it. `first`:
    // the run starts its segment.
    static inline uint64_t RunSumCycles(uint64_t n, bool first) {
        uint64_t c = 0;
        for (uint64_t o = 0; o < n; o += TMP_FLOATS) {
            const uint64_t len = std::min<uint64_t>(TMP_FLOATS, n - o), padded = LanePad(len);
            c += (padded != len ? DaeIsa::Fill(LANES) : 0) + DaeIsa::Op(len) + DaeIsa::ReduceRuns(1, padded);
            if (!(first && o == 0)) c += DaeIsa::Op(1);
        }
        return c;
    }

    // Row sums as partials [6.2]. No ReduceSum: its lanes 1..7 are undefined, so a slot cannot be
    // folded. Instead every row's squares are summed column-wise down to one 64-lane repeat and folded
    // once (BlockReduceSum) into 8 partials per row, and a second fold packs 8 rows' partials into
    // their sums (GroupNormCycles).
    //
    // The chunk's pitch for rows of pitch w: rows of whole repeats, at least one
    static constexpr uint64_t ChunkPitch(uint64_t w) { return w < LANES ? LANES : w; }

    // A piece of a row whose squares take the chunk alone (Core::PieceSum): `zeros` lanes of the chunk
    // zeroed, L squares written, then the P2 lanes (a power of two of repeats) halved down to one repeat
    static inline uint64_t PieceCycles(uint64_t L, uint64_t zeros, uint64_t P2) {
        uint64_t c = (zeros ? DaeIsa::Fill(zeros) : 0) + (L ? DaeIsa::Op(L) : 0);
        for (uint64_t h = P2 / 2; h >= LANES; h /= 2) c += DaeIsa::Op(h);
        return c;
    }
    // A row piece of L squares: padded to P2 = a power of two of repeats, the lanes from its last whole
    // repeat on zeroed before the squares are written
    static constexpr uint64_t PiecePad(uint64_t L) { return LANES * NextPow2(CeilDiv(L, LANES)); }
    static constexpr uint64_t PieceZeros(uint64_t L) { return PiecePad(L) != L ? PiecePad(L) - L / LANES * LANES : 0; }

    // Rows of P squares that fold 8 -> 1 before their columns are added (Core::ChunkPartials): whole
    // 64-lane repeats, so every folded row starts a block, and at least 512 lanes, so it fills a repeat
    static constexpr bool FoldFirstRows(uint64_t P) { return P % LANES == 0 && P >= 8 * LANES && P <= TMP_FLOATS; }

    // m rows' squares in the chunk at pitch P down to 8 partials each (Core::ChunkPartials): folded 8 -> 1
    // first where FoldFirstRows allows (one fold of every repeat into the fold partition), then each
    // row's 64-lane columns added onto its first (one strided Add per column for all m rows), then one
    // fold of every row
    static inline uint64_t ChunkPartialCycles(uint64_t m, uint64_t P) {
        if (FoldFirstRows(P)) return DaeIsa::FoldReps(m * P / LANES) + (CeilDiv(P / 8, LANES) - 1) * DaeIsa::Strided(m) + DaeIsa::FoldReps(m);
        return (CeilDiv(P, LANES) - 1) * DaeIsa::Strided(m) + DaeIsa::FoldReps(m);
    }

    // The 64-lane column sums of one piece of L squares of a long row (Core::PieceColumns): folded first
    // where it can be, else halved (PieceCycles)
    static inline uint64_t PieceColumnCycles(uint64_t L) {
        if (FoldFirstRows(L)) return DaeIsa::Op(L) + DaeIsa::FoldReps(L / LANES) + (CeilDiv(L / 8, LANES) - 1) * DaeIsa::Strided(1);
        return PieceCycles(L, PieceZeros(L), PiecePad(L));
    }

    // Rows whose squares the chunk takes a chunk of rows at a time (Core::RowPartials): two or more of
    // them, or one that folds first; wider rows go one at a time, in pieces
    static inline bool ChunkRows(uint64_t w) {
        const uint64_t P = ChunkPitch(w), per = TMP_FLOATS / P;
        return per >= 2 || (per == 1 && FoldFirstRows(P));
    }

    // 8 partials per row for n rows of pitch w into the slots (Core::RowPartials): rows the chunk takes
    // (ChunkRows) go a chunk at a time: the squares (one Mul, or for rows narrower than a repeat the
    // chunk zeroed and one strided Mul), then ChunkPartialCycles; a wider row goes alone, in chunk-sized
    // pieces (PieceColumnCycles) accumulated in the work partition, then its fold
    static inline uint64_t RowPartialCycles(uint64_t n, uint64_t w) {
        const uint64_t P = ChunkPitch(w), per = TMP_FLOATS / P;
        if (ChunkRows(w)) {
            uint64_t c = 0;
            for (uint64_t r0 = 0; r0 < n; r0 += per) {
                const uint64_t m = std::min(per, n - r0);
                c += (w < LANES ? DaeIsa::Fill(m * LANES) + DaeIsa::Strided(m) : DaeIsa::Op(m * w)) + ChunkPartialCycles(m, P);
            }
            return c;
        }
        uint64_t row = 0;
        const uint64_t pieces = CeilDiv(w, TMP_FLOATS);
        for (uint64_t o = 0; o < w; o += TMP_FLOATS) {
            const uint64_t L = std::min<uint64_t>(TMP_FLOATS, w - o);
            row += PieceColumnCycles(L) + (pieces > 1 ? DaeIsa::Op(LANES) : 0);
        }
        return n * (row + DaeIsa::FoldReps(1));
    }

    // n elements *= the value of one 32-byte block (Core::MulByBlock): 64-lane repeats, at most 255 per
    // instruction, then the tail
    static inline uint64_t BlockMulCycles(uint64_t n) {
        const uint64_t full = n / LANES;
        uint64_t c = 0;
        for (uint64_t r = 0; r < full; r += 255) c += DaeIsa::Strided(std::min<uint64_t>(255, full - r));
        return c + (n % LANES ? DaeIsa::Strided(1) : 0);
    }

    // n rows of pitch w (*|+)= a block per row (`perRow`) or one parameter row (Core::RowOp): one strided
    // instruction per 64-lane column, a repeat per row, where the repeat stride reaches the rows and that
    // is no dearer than row by row (a few wide rows are cheaper one at a time)
    static inline bool StridedRowOp(uint64_t n, uint64_t w, bool perRow) {
        return StridedRows(w) && CeilDiv(w, LANES) * DaeIsa::Strided(n) <= n * (perRow ? BlockMulCycles(w) : DaeIsa::Op(w));
    }
    static inline uint64_t RowOpCycles(uint64_t n, uint64_t w, bool perRow) {
        if (StridedRowOp(n, w, perRow)) return CeilDiv(w, LANES) * DaeIsa::Strided(n);
        return n * (perRow ? BlockMulCycles(w) : DaeIsa::Op(w));
    }

    // A group of n rows of pitch w up to its scaling (Core::NormalizeRows): the partials, the fold that
    // packs them into row sums, the lane chain, the Brcb and the scaling. gamma and beta follow,
    // RowOpCycles(n, w, false) each.
    static inline uint64_t GroupNormCycles(uint64_t n, uint64_t w, uint32_t s) {
        return RowPartialCycles(n, w) + DaeIsa::FoldReps(CeilDiv(n, 8)) + LaneRmsCycles(n, s) + DaeIsa::Brcb(CeilDiv(n, 8)) +
               RowOpCycles(n, w, true);
    }

    // A band group's partials (Core::BandPartials): the band rows' own columns squared into a zeroed
    // chunk, one Mul each, then summed and folded as RowPartialCycles does; `lens` the rows' own column
    // counts (0: none of the row is the core's)
    static inline uint64_t BandPartialCycles(const uint32_t* lens, uint64_t n, uint64_t pitch) {
        const uint64_t P = ChunkPitch(pitch), per = TMP_FLOATS / P, k = CeilDiv(P, LANES);
        uint64_t c = 0;
        for (uint64_t r0 = 0; r0 < n; r0 += per) {
            const uint64_t m = std::min(per, n - r0);
            if (ChunkRows(pitch)) {
                c += DaeIsa::Fill(m * P) + ChunkPartialCycles(m, P);
                for (uint64_t i = r0; i < r0 + m; ++i) c += lens[i] ? DaeIsa::Op(lens[i]) : 0;
            } else {
                c += PieceCycles(lens[r0], LANES * NextPow2(k), LANES * NextPow2(k)) + DaeIsa::FoldReps(1);
            }
        }
        return c;
    }

    // One row tile of k rows of pitch w (Core::RowTiles) with gamma and beta: Z, every group, then
    // (16-bit) the narrowing into the egress buffer
    static inline uint64_t RowTileCycles(uint64_t k, uint64_t w, uint32_t s) {
        uint64_t c = BuildCycles(k * w, s) + (s < 4 ? DaeIsa::Op(k * w) : 0);
        for (uint64_t g = 0; g < k; g += ROW_GROUP) {
            const uint64_t n = std::min<uint64_t>(ROW_GROUP, k - g);
            c += GroupNormCycles(n, w, s) + 2 * RowOpCycles(n, w, false);
        }
        return c;
    }

    // Widening gamma and beta of w columns (FP32 parameters arrive by DMA only, Core::WidenParams): one
    // piece each when both fit the staging buffer (`stageBytes`: the Z tile or the scratch chunk), else
    // a chunk at a time; every piece with its padding to the 32-byte grid
    static inline uint64_t ParamCycles(uint64_t w, uint32_t s, uint64_t stageBytes = TMP_BYTES) {
        if (s == 4) return 0;
        if (2ull * Align32(w * s) <= stageBytes) return 2 * DaeIsa::Op(Align32(w * s) / s);
        uint64_t c = 0;
        const uint64_t chunk = TMP_BYTES / s;
        for (uint64_t o = 0; o < w; o += chunk) c += 2 * DaeIsa::Op(Align32(std::min(chunk, w - o) * s) / s);
        return c;
    }

    // -------------------------------------------------------------------------
    // Timeline model of row tiles [pipeline bubbles]. One core's row tiles replayed at tile
    // granularity with dsa::CoreTimeline's rules and DaePipeline's issue order: an in-order
    // vector pipe; one system-memory channel for loads and stores in issue order, each
    // transfer landing DMA_LATENCY_CYCLES after it streamed; scratchpad-to-scratchpad copies on
    // their own unit; a buffer refilled only once its last read ended. Times in vector cycles.
    // -------------------------------------------------------------------------
    struct RowPipe {             // Scheduling choices of a row-tile plan
        uint32_t B, head, tail;  // Body, head and tail tile rows (head/tail 0: none)
        uint32_t depth;          // Tiles in flight per input queue
        uint32_t outDepth;       // Egress buffers (VECOUT), 1 or 2
        bool paramsFirst, earlyLoads;
    };
    struct RowTimeline { double finish, vector, dma; };

    // A row-tile plan with the given scheduling choices. Its modeled time and vector cycles
    // are its slowest core's (cores differ by one row: every distinct row count counts).
    static inline void ApplyRowPipe(TilingConfig& cfg, uint32_t M, uint32_t D, uint32_t s, const HardwareModel& hw,
                                    const RowPipe& pp) {
        double finish = 0, vector = 0;
        uint32_t seen[3] = {0, 0, 0}, nSeen = 0;
        const dsa::MemorySystem mem = Memory(cfg, M, D, s);
        for (uint32_t t = 0; t < cfg.blocks; ++t) {
            const CoreRange r = Range(cfg, M, D, t, cfg.blocks);
            const uint32_t rows = r.rowZ - r.rowA;
            if (!rows || std::find(seen, seen + nSeen, rows) != seen + nSeen) continue;
            if (nSeen < 3) seen[nSeen++] = rows;
            const RowTimeline x = RowTilesTimeline(rows, D, s, pp, hw, mem);
            finish = std::max(finish, x.finish);
            vector = std::max(vector, x.vector);
        }
        cfg.tileRows = pp.B;
        cfg.pitch = RowPitch(D, s);
        cfg.tileElems = pp.B * cfg.pitch;
        cfg.headRows = pp.head;
        cfg.tailRows = pp.tail;
        cfg.paramsFirst = pp.paramsFirst;
        cfg.earlyLoads = pp.earlyLoads;
        cfg.zResident = 0;
        cfg.layout = RowLayout(pp.B, D, s, pp.depth, pp.outDepth);
        SetModel(cfg, finish, vector, hw);
    }

    // `bound`: give up (finish = infinity) once the core cannot finish before it. Uniform body
    // tiles make the replay a max-plus linear recurrence: once the whole state has shifted by the
    // same amount over c tiles it keeps doing so, and the replay jumps ahead by whole periods.
    // `mem`: the launch's memory system, which prices every transfer by its size.
    static inline RowTimeline RowTilesTimeline(uint64_t rows, uint32_t D, uint32_t s, const RowPipe& pp, const HardwareModel& hw,
                                               const dsa::MemorySystem& mem, double bound = std::numeric_limits<double>::infinity()) {
        using I = DaeIsa;
        const double L = hw.LatencyCycles();
        const uint32_t d = pp.depth, od = pp.outDepth, R = static_cast<uint32_t>(rows), B = pp.B, W = RowPitch(D, s);
        RowTimeline out{std::numeric_limits<double>::infinity(), 0.0, 0.0};
        if (od < 1 || od > 4) return out;
        // The core's tiles, as RowSchedule cuts them: [head] [B x body] [partial] [tail]
        const uint32_t head = pp.head && pp.head < R ? pp.head : 0, rest = R - head;
        const uint32_t tail = pp.tail && rest > pp.tail ? pp.tail : 0;
        const uint32_t body = (rest - tail) / B, partial = (rest - tail) % B;
        auto tileVector = [&](uint32_t k) { return k ? RowTileCycles(k, W, s) : 0; };
        auto tileBytes = [&](uint32_t k) { return static_cast<double>(uint64_t(k) * W * s); };  // Rows on the 32-byte grid
        auto tileChannel = [&](uint32_t k) { return mem.Cycles(tileBytes(k)); };  // One tile's transfer on the channel
        const uint64_t zBytes = RowLayout(B, D, s, d, od).z, stage = std::max<uint64_t>(TMP_BYTES, zBytes);
        const bool staged = s == 4 || 2ull * Align32(uint64_t(D) * s) <= stage;
        const uint64_t chunk = TMP_BYTES / s;  // Unstaged 16-bit parameters: scratch-sized pieces
        double paramChannel = 0;
        if (staged) paramChannel = 2.0 * mem.Cycles(Align32(uint64_t(D) * s));
        else for (uint64_t o = 0; o < D; o += chunk) paramChannel += 2.0 * mem.Cycles(Align32(std::min<uint64_t>(chunk, D - o) * s));
        const uint64_t bodyVector = tileVector(B);
        const double bodyChannel = tileChannel(B);
        uint64_t vectorLeft = ParamCycles(D, s, stage) + tileVector(head) + body * bodyVector + tileVector(partial) + tileVector(tail);
        double channelLeft = paramChannel + 3 * (tileChannel(head) + body * bodyChannel + tileChannel(partial) + tileChannel(tail));
        const uint32_t last = tail ? tail : partial ? partial : body ? B : head;
        // Lower bound: the vector unit starts after a load landed and the last store lands after
        // it; the channel streams everything and the last store lands after that
        if (std::max(L + vectorLeft + tileChannel(last) + L, channelLeft + L) >= bound) return out;

        const RowSchedule tiles{0, R, B, pp.head, pp.tail};
        double ch = 0, vec = 0, loads = 0, stores = 0;
        double land1[4] = {}, land2[4] = {}, x1Read[4] = {};  // Per input buffer: landed; X1 read by the tile's first instruction
        double yFree[4] = {};                                  // Per egress buffer: its last store streamed out
        // X2 of the tile a buffer last held is read a scratch chunk at a time: a smaller tile
        // reloading it waits only for the chunks it overwrites
        constexpr uint32_t CHUNKS = 64;
        double x2Read[4][CHUNKS] = {};
        uint32_t x2Chunks[4] = {};
        auto x2Free = [&](uint32_t slot, uint64_t elems) {
            if (!x2Chunks[slot]) return 0.0;
            return x2Read[slot][std::min<uint64_t>(x2Chunks[slot] - 1, (elems - 1) / TMP_FLOATS)];
        };
        uint32_t next1 = 0, next2 = 0;
        uint64_t n1 = 0, n2 = 0, done = 0;
        auto transfer = [&](double ready, double bytes) {  // Landing time of one transfer
            const double start = std::max(ch, ready), occupied = mem.Cycles(bytes);
            ch = start + occupied;
            out.dma += occupied;
            channelLeft -= occupied;
            return ch + L;
        };
        auto load1 = [&] {
            const uint32_t k = tiles.Rows(next1), slot = static_cast<uint32_t>(n1++ % d);
            land1[slot] = transfer(x1Read[slot], tileBytes(k));
            loads = std::max(loads, land1[slot]);
            next1 += k;
        };
        auto load2 = [&] {
            const uint32_t k = tiles.Rows(next2), slot = static_cast<uint32_t>(n2++ % d);
            land2[slot] = transfer(x2Free(slot, uint64_t(k) * W), tileBytes(k));
            loads = std::max(loads, land2[slot]);
            next2 += k;
        };
        auto compute = [&](double ready, uint64_t cycles) {
            vec = std::max(vec, ready) + static_cast<double>(cycles);
            out.vector += static_cast<double>(cycles);
            vectorLeft -= cycles;
            return vec;
        };
        // Prologue: the first `depth` tiles and gamma/beta
        double landG = 0, landB = 0;
        auto stageParams = [&] {
            if (!staged) return;
            landG = transfer(0, Align32(uint64_t(D) * s));
            landB = transfer(0, Align32(uint64_t(D) * s));
            loads = std::max(loads, landB);
        };
        if (pp.paramsFirst) stageParams();
        load1();
        load2();
        if (!pp.paramsFirst) stageParams();
        for (uint32_t i = 1; i < d && next1 < R; ++i) {
            load1();
            load2();
        }
        double readyG = landG, readyB = landB;  // gamma / beta widened in par
        if (s < 4 && staged) {
            readyG = compute(landG, I::Op(W));
            readyB = compute(landB, I::Op(W));
        } else if (s < 4) {  // Scratch-sized pieces, each loaded once the previous one is widened
            double tmpFree = 0;
            for (int k = 0; k < 2; ++k) {
                for (uint64_t o = 0; o < D; o += chunk) {
                    const uint64_t m = std::min<uint64_t>(chunk, D - o);
                    const double landed = transfer(tmpFree, Align32(m * s));
                    loads = std::max(loads, landed);
                    tmpFree = compute(landed, I::Op(Align32(m * s) / s));
                }
                (k == 0 ? readyG : readyB) = vec;
            }
        }
        // The groups of a tile once its Z is built: each group's inverse RMS, then gamma and beta,
        // each once its parameter row is in place
        auto groups = [&](double t, uint32_t k) {
            for (uint32_t g = 0; g < k; g += ROW_GROUP) {
                const uint32_t n = std::min<uint32_t>(ROW_GROUP, k - g);
                const double op = static_cast<double>(RowOpCycles(n, W, false));
                t = std::max(t + static_cast<double>(GroupNormCycles(n, W, s)), readyG) + op;
                t = std::max(t, readyB) + op;
            }
            return t;
        };
        // Steady-state detection: snapshots of the state, rotated to the next tile's buffers
        constexpr uint32_t HIST = 8;
        double hist[HIST][4 + 16 + 4];
        const uint32_t width = 4 + 4 * d + od, firstBody = head ? 1u : 0u, bodyEnd = firstBody + body;  // Tile indices
        auto snapshot = [&](double* v) {
            v[0] = vec, v[1] = ch, v[2] = loads, v[3] = stores;
            for (uint32_t i = 0; i < d; ++i) {
                const uint32_t slot = static_cast<uint32_t>((done + i) % d);
                v[4 + i] = land1[slot], v[4 + d + i] = land2[slot], v[4 + 2 * d + i] = x1Read[slot];
                v[4 + 3 * d + i] = x2Chunks[slot] ? x2Read[slot][x2Chunks[slot] - 1] : 0.0;
            }
            for (uint32_t j = 0; j < od; ++j) v[4 + 4 * d + j] = yFree[(done + j) % od];
        };
        auto shifted = [&](const double* a, const double* b, double& by) {  // a = b + by in every component
            by = a[0] - b[0];
            for (uint32_t i = 1; i < width; ++i) {
                if (std::fabs((a[i] - b[i]) - by) > 1e-9 * std::max(1.0, std::fabs(a[i]))) return false;
            }
            return true;
        };
        for (uint32_t row = 0; row < R;) {
            const uint32_t k = tiles.Rows(row), slot = static_cast<uint32_t>(done % d), ys = static_cast<uint32_t>(done % od);
            const uint64_t e = uint64_t(k) * W, C = RowTileCycles(k, W, s);
            if (s == 4) {  // Z = X1 + X2 straight into the egress buffer, once its last store streamed out
                const double start = std::max({vec, land1[slot], land2[slot], yFree[ys]});
                x1Read[slot] = x2Read[slot][0] = start + I::Op(e);  // X1 and X2 are read by the add, all at once
                x2Chunks[slot] = 1;
                vec = groups(start + static_cast<double>(I::Op(e)), k);
            } else {
                const double start = std::max(vec, land1[slot]), x2 = std::max(start + I::Op(e), land2[slot]);
                x1Read[slot] = start + I::Op(e);  // X1 is read by its widening into Z
                double phase = 0;
                x2Chunks[slot] = 0;
                for (uint64_t o = 0; o < e; o += TMP_FLOATS) {  // Widen X2 and add, a scratch chunk at a time
                    const uint64_t m = std::min<uint64_t>(TMP_FLOATS, e - o);
                    if (x2Chunks[slot] < CHUNKS) x2Read[slot][x2Chunks[slot]++] = x2 + phase + I::Op(m);
                    else x2Read[slot][CHUNKS - 1] = x2 + phase + I::Op(m);
                    phase += 2.0 * I::Op(m);
                }
                // The groups, then the narrowing into the egress buffer once its last store streamed out
                vec = std::max(groups(x2 + phase, k), yFree[ys]) + static_cast<double>(I::Op(e));
            }
            out.vector += static_cast<double>(C);
            vectorLeft -= C;
            if (pp.earlyLoads) {  // Both input buffers are free once Z is built
                if (next1 < R) load1();
                if (next2 < R) load2();
            }
            // Y leaves from the egress buffer: the store follows the tile's last instruction
            const double start = std::max(ch, vec), occupied = tileChannel(k);
            ch = start + occupied;
            out.dma += occupied;
            channelLeft -= occupied;
            yFree[ys] = ch;
            stores = std::max(stores, ch + L);
            if (!pp.earlyLoads) {
                if (next1 < R) load1();
                if (next2 < R) load2();
            }
            row += k;
            ++done;
            if (std::max(vec + vectorLeft + tileChannel(last) + L, ch + channelLeft + L) >= bound) {
                out.finish = std::numeric_limits<double>::infinity();
                return out;
            }
            // In the body, once periodic, jump whole periods (whole buffer rotations keep the buffers)
            if (done < firstBody + 2 || done >= bodyEnd) continue;
            snapshot(hist[done % HIST]);
            for (uint32_t c = 1; c <= 4 && done >= firstBody + 2 * c + 1; ++c) {
                double by = 0, by2 = 0;
                if (!shifted(hist[done % HIST], hist[(done - c) % HIST], by) ||
                    !shifted(hist[(done - 1) % HIST], hist[(done - 1 - c) % HIST], by2) || std::fabs(by - by2) > 1e-9 * std::max(1.0, by)) {
                    continue;
                }
                uint32_t period = c;
                while (period % d || period % od) period += c;  // Whole rotations of the input and egress buffers
                // Every load issued inside the jump must still be a body tile
                const uint64_t ahead = std::max(n1, n2) - done;
                const uint64_t room = bodyEnd > done + ahead + 1 ? bodyEnd - done - ahead - 1 : 0;
                const uint64_t m = room / period * period;
                if (!m) break;
                const double shift = by * static_cast<double>(m / c);
                vec += shift, ch += shift, loads += shift, stores += shift;
                for (uint32_t i = 0; i < d; ++i) {
                    land1[i] += shift, land2[i] += shift, x1Read[i] += shift;
                    for (uint32_t c2 = 0; c2 < x2Chunks[i]; ++c2) x2Read[i][c2] += shift;
                }
                for (uint32_t j = 0; j < od; ++j) yFree[j] += shift;
                row += static_cast<uint32_t>(m) * B, next1 += static_cast<uint32_t>(m) * B, next2 += static_cast<uint32_t>(m) * B;
                done += m, n1 += m, n2 += m;
                out.vector += static_cast<double>(m * bodyVector);
                out.dma += static_cast<double>(m) * 3 * bodyChannel;
                vectorLeft -= m * bodyVector;
                channelLeft -= static_cast<double>(m) * 3 * bodyChannel;
                break;
            }
        }
        out.finish = std::max({vec, ch, loads, stores});
        return out;
    }

    // Timeline of a column band (SPLIT_COLUMNS), mirroring BandPhase1 / SyncAll / BandPhase2 for
    // every core: phase 1 loads k band rows per tile (one DMA per row and input) and publishes a
    // record of partials; all cores leave SyncAll SYNC_ALL_CYCLES after the last one stored its
    // record; phase 2 gathers the records, scales the resident Z, applies gamma and beta, narrows
    // (16-bit) or copies (FP32) each tile into the next egress buffer and stores it row by row.
    struct BandPipe {
        uint32_t k, depth;  // Tile rows (a multiple of 8, or every row), tiles in flight
        uint32_t outDepth;  // Egress buffers (VECOUT), 1 or 2
    };
    static inline RowTimeline BandTimeline(const TilingConfig& cfg, uint32_t M, uint32_t D, uint32_t s, uint32_t pitch,
                                           const BandPipe& bp, const HardwareModel& hw,
                                           double bound = std::numeric_limits<double>::infinity()) {
        using I = DaeIsa;
        const double L = hw.LatencyCycles();
        const dsa::MemorySystem mem = Memory(cfg, M, D, s);
        const uint32_t q = QuantumElems(s, hw), R = RecordFloats(cfg, M), nb = cfg.blocks;
        RowTimeline out{0.0, 0.0, 0.0};
        double arrive[MAX_THREADS] = {}, phase1[MAX_THREADS] = {};  // Per core: SyncAll arrival, phase-1 vector cycles
        uint32_t lens[4096], offs[4096];
        const uint32_t od = bp.outDepth;
        if (M > 4096 || nb > MAX_THREADS || od < 1 || od > 2) return out.finish = std::numeric_limits<double>::infinity(), out;
        // Phase 1 of core t: returns its SyncAll arrival
        for (uint32_t t = 0; t < nb; ++t) {
            const CoreRange r = Range(cfg, M, D, t, nb);
            if (r.rowZ <= r.rowA) continue;
            const uint32_t c0 = BandBegin(r, q), w = BandEnd(r, q) - c0;
            for (uint32_t i = 0; i < M; ++i) {
                uint32_t cb, ce;
                BandRow(r, i, q, cb, ce);
                lens[i] = cb < ce ? ce - cb : 0;
                offs[i] = cb < ce ? cb - c0 : 0;
            }
            double ch = 0, vec = 0, cycles = 0, dma = 0;
            double free1[4] = {};                       // Per buffer: when X1's last read ended
            double x2Read[4][64] = {};                  // Per buffer: X2 read ends, a scratch chunk at a time
            uint32_t x2Chunks[4] = {};
            double rowLand1[4096], rowLand2[4096];      // Per row: when its X1 / X2 landed
            auto transfer = [&](double ready, double bytes) {
                const double start = std::max(ch, ready), occupied = mem.Cycles(bytes);
                ch = start + occupied;
                dma += occupied;
                return ch + L;
            };
            auto compute = [&](double ready, uint64_t c) {
                vec = std::max(vec, ready) + static_cast<double>(c);
                cycles += static_cast<double>(c);
                return vec;
            };
            uint32_t loaded = 0, nLoads = 0;  // Rows loaded so far, tiles loaded so far
            auto load = [&] {
                const uint32_t row = loaded, k = std::min(bp.k, M - row), slot = nLoads++ % bp.depth;
                for (uint32_t i = row; i < row + k; ++i) {
                    if (!lens[i]) { rowLand1[i] = rowLand2[i] = 0; continue; }
                    const uint64_t lastElem = uint64_t(i - row) * pitch + offs[i] + lens[i] - 1;
                    const double x2Free = x2Chunks[slot] ? x2Read[slot][std::min<uint64_t>(x2Chunks[slot] - 1, lastElem / TMP_FLOATS)] : 0.0;
                    rowLand1[i] = transfer(free1[slot], Align32(uint64_t(lens[i]) * s));
                    rowLand2[i] = transfer(x2Free, Align32(uint64_t(lens[i]) * s));
                }
                loaded += k;
            };
            load();
            double landG, landB;  // Parameters of the band: staged in the scratch buffer (16-bit) or par
            landG = transfer(0, Align32(uint64_t(w) * s));
            landB = transfer(0, Align32(uint64_t(w) * s));
            for (uint32_t i = 1; i < bp.depth && loaded < M; ++i) load();
            // gamma and beta are widened now, used after SyncAll
            if (s < 4) {
                compute(landG, I::Op(w));
                compute(landB, I::Op(w));
            }
            uint32_t tile = 0;
            for (uint32_t row = 0; row < M; row += bp.k, ++tile) {
                const uint32_t k = std::min(bp.k, M - row), slot = tile % bp.depth;
                const uint64_t e = uint64_t(k) * pitch;
                auto landed = [&](const double* rl, uint64_t o, uint64_t m) {  // Rows under elements [o, o + m)
                    double t0 = 0;
                    for (uint64_t i = o / pitch; i < std::min<uint64_t>(k, (o + m + pitch - 1) / pitch); ++i) t0 = std::max(t0, rl[row + i]);
                    return t0;
                };
                if (s == 4) {
                    compute(std::max(landed(rowLand1, 0, e), landed(rowLand2, 0, e)), I::Op(e));
                    free1[slot] = x2Read[slot][0] = vec;
                    x2Chunks[slot] = 1;
                } else {
                    compute(landed(rowLand1, 0, e), I::Op(e));
                    free1[slot] = vec;
                    x2Chunks[slot] = 0;
                    for (uint64_t o = 0; o < e; o += TMP_FLOATS) {
                        const uint64_t m = std::min<uint64_t>(TMP_FLOATS, e - o);
                        const double read = compute(landed(rowLand2, o, m), I::Op(m));
                        x2Read[slot][std::min(x2Chunks[slot]++, 63u)] = read;
                        compute(0, I::Op(m));
                    }
                }
                // Per group: the partials of the band rows' own columns, and the fold that packs their sums
                // into the record
                for (uint32_t g = 0; g < k; g += ROW_GROUP) {
                    const uint32_t n = std::min(k - g, ROW_GROUP);
                    compute(0, BandPartialCycles(lens + row + g, n, pitch) + I::FoldReps(CeilDiv(n, 8)));
                }
                if (loaded < M) load();
            }
            // The record goes out; SyncAll waits for it to land
            const double start = std::max(ch, vec);
            ch = start + mem.Cycles(R * 4.0);
            dma += mem.Cycles(R * 4.0);
            arrive[t] = std::max(vec, ch + L);
            phase1[t] = cycles;
            out.dma = std::max(out.dma, dma);
            if (arrive[t] >= bound) return out.finish = std::numeric_limits<double>::infinity(), out;
        }
        double release = 0;
        for (uint32_t t = 0; t < nb; ++t) release = std::max(release, arrive[t]);
        release += hw.SyncCycles();
        // Phase 2 of every core, from the release
        double worst = 0;
        double outRead[2][4096];  // Per egress buffer: when each row of the tile it last held streamed out
        for (uint32_t t = 0; t < nb; ++t) {
            const CoreRange r = Range(cfg, M, D, t, nb);
            if (r.rowZ <= r.rowA) continue;
            const uint32_t w = BandEnd(r, q) - BandBegin(r, q);
            for (uint32_t i = 0; i < M; ++i) {
                uint32_t cb, ce;
                BandRow(r, i, q, cb, ce);
                lens[i] = cb < ce ? ce - cb : 0;
            }
            double ch = release, vec = release, stores = 0, cycles = 0;
            uint32_t outRows[2] = {0, 0};
            auto compute = [&](double ready, uint64_t c) {
                vec = std::max(vec, ready) + static_cast<double>(c);
                cycles += static_cast<double>(c);
                return vec;
            };
            ch += mem.Cycles(nb * R * 4.0);  // Every core's record in one transfer
            const double gathered = ch + L;
            compute(gathered, 0);
            for (uint64_t n = nb; n > 1; n -= n / 2) compute(0, I::Op(n / 2 * R));
            uint32_t tile = 0;
            for (uint32_t row = 0; row < M; row += bp.k, ++tile) {
                const uint32_t k = std::min(bp.k, M - row);
                const uint64_t e = uint64_t(k) * pitch;
                // Per group: the lane chain on the totals, the Brcb, the scaling, gamma and beta
                for (uint32_t g = 0; g < k; g += ROW_GROUP) {
                    const uint64_t n = std::min(k - g, ROW_GROUP);
                    compute(0, LaneRmsCycles(n, s) + I::Brcb(CeilDiv(n, 8)) + RowOpCycles(n, pitch, true) + 2 * RowOpCycles(n, pitch, false));
                }
                // Tiles rotate through the egress buffers; the narrowing (16-bit) or copy (FP32)
                // waits until the rows it overwrites have streamed out
                const uint32_t buf = tile % od;
                double freeAt = release;
                for (uint32_t i = 0; i < std::min(k, outRows[buf]); ++i) freeAt = std::max(freeAt, outRead[buf][i]);
                compute(freeAt, I::Op(e));
                for (uint32_t i = 0; i < k; ++i) {
                    outRead[buf][i] = 0;
                    if (!lens[row + i]) continue;
                    const double start = std::max(ch, vec);
                    ch = start + mem.Cycles(Align32(uint64_t(lens[row + i]) * s));
                    outRead[buf][i] = ch;
                    stores = std::max(stores, ch + L);
                }
                outRows[buf] = k;
            }
            (void)w;
            worst = std::max({worst, vec, ch, stores});
            out.vector = std::max(out.vector, phase1[t] + cycles);
        }
        out.finish = worst;
        return out;
    }

    // Resident Z of a segment starting at element `start`: its first tile at the segment's base, every
    // later tile on the 32-byte grid, like its system-memory address (a vector operand starts a block),
    // so the segment takes up to q - 1 more elements than it has
    static constexpr uint64_t ResidentNeed(uint64_t start, uint64_t len, uint32_t q) { return len + (q - start % q) % q; }

    // Timeline of column tiles (row-major Split-D fragments, and rows too long for a row tile),
    // mirroring ColumnPhase1 / SyncAll / ColumnPhase2 with their Sweep1 / Sweep2 for every core.
    // X1, X2 and the gamma and beta chunks each alternate between two buffers; a buffer is refilled
    // once the reads of the elements it overwrites ended (16-bit reads go a scratch chunk at a time).
    // Every tile's result goes out through the next of two egress buffers.
    struct ColumnSim {
        struct Buffer {
            double land = 0;          // When its current content landed
            double read[64] = {};     // When the reads of each scratch chunk ended
            uint32_t chunks = 0;
            double FreeFor(uint64_t elems) const {
                return chunks ? read[std::min<uint64_t>(chunks - 1, (elems - 1) / TMP_FLOATS)] : 0.0;
            }
            void ReadAll(double t) { read[0] = t, chunks = 1; }
        };
        uint32_t D, s, q;
        uint64_t tile;
        dsa::MemorySystem mem;
        double L;
        double ch = 0, vec = 0, loads = 0, stores = 0, cycles = 0, dma = 0;
        Buffer x1[2], x2[2], g[2], b[2], y[2];  // g, b: the two ways of the gamma/beta ring
        uint32_t yNext = 0;  // Egress buffers rotate across every sweep of the core

        double Transfer(double ready, double bytes) {
            const double start = std::max(ch, ready), occupied = mem.Cycles(bytes);
            ch = start + occupied;
            dma += occupied;
            return ch + L;
        }
        double Compute(double ready, uint64_t c) {
            vec = std::max(vec, ready) + static_cast<double>(c);
            cycles += static_cast<double>(c);
            return vec;
        }
        double Bytes(uint64_t n) const { return static_cast<double>(Align32(n * s)); }
        uint32_t TileLength(uint64_t e, uint64_t end) const { return static_cast<uint32_t>(std::min(end, e / q * q + tile) - e); }
        void Load(Buffer& b, uint64_t n) {
            b.land = Transfer(b.FreeFor(n), Bytes(n));
            loads = std::max(loads, b.land);
        }
        void Store(Buffer& b, uint64_t n) {
            const double start = std::max(ch, vec), occupied = mem.Cycles(Bytes(n));
            ch = start + occupied;
            dma += occupied;
            b.ReadAll(ch);
            stores = std::max(stores, ch + L);
        }
        // zt (+|*)= the parameter chunk in `b` (16-bit: widened through the scratch chunk)
        void Combine(Buffer& b, uint64_t n) {
            if (s == 4) {
                b.ReadAll(Compute(b.land, DaeIsa::Op(n)));
                return;
            }
            b.chunks = 0;
            for (uint64_t o = 0; o < n; o += TMP_FLOATS) {
                const uint64_t m = std::min<uint64_t>(TMP_FLOATS, n - o);
                b.read[std::min(b.chunks++, 63u)] = Compute(b.land, DaeIsa::Op(m));
                Compute(0, DaeIsa::Op(m));
            }
        }
        // Z = X1 + X2 into a Z buffer
        void AddInputs(Buffer& a, Buffer& bt, uint64_t n) {
            if (s == 4) {
                const double t = Compute(std::max(a.land, bt.land), DaeIsa::Op(n));
                a.ReadAll(t), bt.ReadAll(t);
                return;
            }
            a.ReadAll(Compute(a.land, DaeIsa::Op(n)));
            Combine(bt, n);
        }
        // Sweep 1 over [start, end): X1 and X2 of tile j+1 load before tile j computes; each tile's
        // sum of squares goes into the segment's slot (RunSumCycles)
        void Sweep1(uint64_t start, uint64_t end) {
            uint32_t j = 0;
            uint64_t e = start;
            uint32_t n = start < end ? TileLength(start, end) : 0;
            auto load = [&](uint32_t slot, uint64_t k) { Load(x1[slot], k), Load(x2[slot], k); };
            if (n) load(0, n);
            while (e < end) {
                const uint64_t next = e + n;
                const uint32_t nn = next < end ? TileLength(next, end) : 0, slot = j % 2;
                if (nn) load((j + 1) % 2, nn);
                AddInputs(x1[slot], x2[slot], n);
                Compute(0, RunSumCycles(n, j == 0));
                e = next, n = nn, ++j;
            }
        }
        // A segment's inverse RMS from its slot, spread by one Brcb (Core::SegmentInvRms)
        void SegmentInvRms() { Compute(0, LaneRmsCycles(1, s) + DaeIsa::Brcb(1)); }
        // The gamma and beta chunks of a tile: loaded as a pair, applied in turn (* gamma + beta)
        void LoadParams(uint32_t slot, uint64_t n) { Load(g[slot], n), Load(b[slot], n); }
        void UseParams(uint32_t slot, uint64_t n) { Combine(g[slot], n), Combine(b[slot], n); }
        // Sweep 2 from resident Z: gamma/beta pairs stream (the first may be in flight: `primed`)
        void Sweep2Resident(uint64_t start, uint64_t end, bool primed) {
            uint32_t j = 0;
            uint64_t e = start;
            uint32_t n = start < end ? TileLength(start, end) : 0;
            if (n && !primed) LoadParams(0, n);
            while (e < end) {
                const uint64_t next = e + n;
                const uint32_t nn = next < end ? TileLength(next, end) : 0, slot = j % 2;
                if (nn) LoadParams((j + 1) % 2, nn);
                Compute(0, BlockMulCycles(n));  // * invRms
                UseParams(slot, n);
                Buffer& out = y[yNext++ % 2];
                Compute(out.FreeFor(n), DaeIsa::Op(n));  // Narrow into the egress buffer once its last store streamed out
                Store(out, n);
                e = next, n = nn, ++j;
            }
        }
        // Sweep 2 recomputing Z: X1, X2 and the gamma/beta pair of tile j+1 load before tile j computes
        void Sweep2Recompute(uint64_t start, uint64_t end) {
            uint32_t j = 0;
            uint64_t e = start;
            uint32_t n = start < end ? TileLength(start, end) : 0;
            auto load = [&](uint32_t slot, uint64_t k) { Load(x1[slot], k), Load(x2[slot], k), LoadParams(slot, k); };
            if (n) load(0, n);
            while (e < end) {
                const uint64_t next = e + n;
                const uint32_t nn = next < end ? TileLength(next, end) : 0, slot = j % 2;
                if (nn) load((j + 1) % 2, nn);
                AddInputs(x1[slot], x2[slot], n);
                Compute(0, BlockMulCycles(n));  // * invRms
                UseParams(slot, n);
                Buffer& out = y[yNext++ % 2];
                Compute(out.FreeFor(n), DaeIsa::Op(n));  // Narrow into the egress buffer once its last store streamed out
                Store(out, n);
                e = next, n = nn, ++j;
            }
        }
    };

    static inline RowTimeline ColumnTimeline(const TilingConfig& cfg, uint32_t M, uint32_t D, uint32_t s, const HardwareModel& hw,
                                             double bound = std::numeric_limits<double>::infinity()) {
        const bool split = cfg.mode == TilingMode::SPLIT_D;
        const uint32_t nb = cfg.blocks, q = QuantumElems(s, hw);
        const dsa::MemorySystem mem = Memory(cfg, M, D, s);
        RowTimeline out{0.0, 0.0, 0.0};
        if (nb > MAX_THREADS) return out.finish = std::numeric_limits<double>::infinity(), out;
        ColumnSim sims[MAX_THREADS];
        bool fragResident[MAX_THREADS][2] = {};
        double release = 0;
        // Phase 1 of every core. Whole-row cores whose rows start on the 32-byte grid behave alike
        // when they hold as many rows: one of them stands for the others.
        uint32_t seenRows[4] = {}, nSeen = 0;
        bool skip[MAX_THREADS] = {};
        for (uint32_t t = 0; t < nb; ++t) {
            const CoreRange r = Range(cfg, M, D, t, nb);
            if (!split && D % q == 0) {
                const uint32_t rows = r.rowZ - r.rowA;
                if (std::find(seenRows, seenRows + nSeen, rows) != seenRows + nSeen) { skip[t] = true; continue; }
                if (nSeen < 4) seenRows[nSeen++] = rows;
            }
            ColumnSim& c = sims[t];
            c.D = D, c.s = s, c.q = q, c.tile = cfg.tileElems, c.mem = mem, c.L = hw.LatencyCycles();
            uint64_t used = 0;
            for (uint32_t k = 0; k < r.nFrag; ++k) {
                const CoreRange::Fragment& f = r.frag[k];
                const uint64_t need = ResidentNeed(uint64_t(f.row) * D + f.cb, f.ce - f.cb, q);
                fragResident[t][k] = cfg.zResident && used + need <= cfg.zResident;
                if (fragResident[t][k]) used = (used + need + q - 1) / q * q;
                c.Sweep1(uint64_t(f.row) * D + f.cb, uint64_t(f.row) * D + f.ce);
            }
            if (split) {  // The core's two records {sum, 0 x7}, built in VECOUT from the slots' lanes 0, go out
                c.Compute(0, DaeIsa::Fill(16) + r.nFrag * DaeIsa::Op(1));
                const double start = std::max(c.ch, c.vec);
                c.ch = start + mem.Cycles(64.0);
                c.dma += mem.Cycles(64.0);
                c.stores = std::max(c.stores, c.ch + c.L);
            }
            for (uint32_t row = r.rowA; row < r.rowZ; ++row) {
                const bool fits = cfg.zResident && used + ResidentNeed(uint64_t(row) * D, D, q) <= cfg.zResident;
                c.Sweep1(uint64_t(row) * D, uint64_t(row + 1) * D);
                c.SegmentInvRms();
                if (fits) c.Sweep2Resident(uint64_t(row) * D, uint64_t(row + 1) * D, false);
                else c.Sweep2Recompute(uint64_t(row) * D, uint64_t(row + 1) * D);
                if (std::max(c.vec, c.ch) >= bound) return out.finish = std::numeric_limits<double>::infinity(), out;
            }
            if (split && r.nFrag > 0 && fragResident[t][0]) {  // The first gamma/beta pair streams in during SyncAll
                const uint64_t start = uint64_t(r.frag[0].row) * D + r.frag[0].cb;
                c.LoadParams(0, c.TileLength(start, start + (r.frag[0].ce - r.frag[0].cb)));
            }
            if (split) release = std::max(release, std::max(c.vec, c.stores));
            if (std::max({c.vec, c.ch, c.stores}) >= bound) return out.finish = std::numeric_limits<double>::infinity(), out;
        }
        release += hw.SyncCycles();
        for (uint32_t t = 0; t < nb; ++t) {
            if (skip[t]) continue;
            ColumnSim& c = sims[t];
            if (split) {  // Phase 2: each fragment's records, then its sweep 2
                const CoreRange r = Range(cfg, M, D, t, nb);
                c.vec = std::max(c.vec, release), c.ch = std::max(c.ch, release);
                double recsRead = 0;
                for (uint32_t k = 0; k < r.nFrag; ++k) {
                    const CoreRange::Fragment& f = r.frag[k];
                    uint32_t first, last;
                    RowOwners(cfg, D, f.row, nb, first, last);
                    const uint32_t lo = 2 * first + std::max(1u, Range(cfg, M, D, first, nb).nFrag) - 1, count = 2 * last - lo + 1;
                    const double landed = c.Transfer(recsRead, count * 32.0);
                    c.loads = std::max(c.loads, landed);
                    const uint64_t n = count * 8ull, padded = LanePad(n);  // The records, zero-padded to whole repeats
                    if (padded != n) c.Compute(0, DaeIsa::Fill(padded - n));
                    recsRead = c.Compute(landed, DaeIsa::Reduce(padded));
                    c.SegmentInvRms();
                    const uint64_t a = uint64_t(f.row) * D + f.cb, z = uint64_t(f.row) * D + f.ce;
                    if (fragResident[t][k]) c.Sweep2Resident(a, z, k == 0);
                    else c.Sweep2Recompute(a, z);
                }
            }
            out.finish = std::max({out.finish, c.vec, c.ch, c.loads, c.stores});
            out.vector = std::max(out.vector, c.cycles);
            out.dma = std::max(out.dma, c.dma);
        }
        return out;
    }

    // A column-band plan with the given scheduling choices (cfg holds the band pitch and the
    // record layout from PlanBand); modeled time and vector cycles are its slowest core's
    static inline void ApplyBandPipe(TilingConfig& cfg, uint32_t M, uint32_t D, uint32_t s, const HardwareModel& hw,
                                     const BandPipe& bp) {
        const RowTimeline t = BandTimeline(cfg, M, D, s, cfg.pitch, bp, hw);
        cfg.tileRows = bp.k;
        cfg.tileElems = bp.k * cfg.pitch;
        cfg.layout = BandLayout(bp.k, cfg.pitch, M, s, cfg.layout.misc, cfg.layout.rec, bp.depth, bp.outDepth);
        SetModel(cfg, t.finish, t.vector, hw);
    }

private:
    // -------------------------------------------------------------------------
    // (3) DAE tiles [ARCH CHALLENGE 2/3, pipeline bubbles]. Every candidate schedule is replayed
    // on the timeline models above (RowTilesTimeline, BandTimeline, ColumnTimeline), which equal
    // the runtime's timeline cycle for cycle, and the one whose slowest core finishes first wins.
    // The 191 KB layout bounds each candidate (the Challenge 2 knapsack):
    //   row tiles     depth x 2 x B W s bytes of X1/X2 tiles (W = RowPitch), outDepth x B W s of
    //                 egress buffers, 4 B W more for the FP32 Z tile at 16 bits, gamma/beta (one
    //                 FP32 row each) and the scratch buffer (the chunk, its fold and the reduction
    //                 partitions)
    // No candidate takes a queue step: X1, X2, gamma/beta chunks and egress buffers are static rings.
    //   column band   k rows at the band pitch, the band's FP32 Z resident across the barrier
    //   column tiles  (10s + 4) bytes per element (X1, X2, gamma and beta chunks, egress, each
    //                 twice; FP32 Z), plus the resident FP32 Z
    // modelNs is the kernel launch and the slowest core's finish time; modelCycles its vector cycles.
    // -------------------------------------------------------------------------
    // `bound`: the modeled cycles a plan has to beat (a plan that cannot keeps modelNs = infinity)
    static inline void PlanTiles(TilingConfig& cfg, uint32_t M, uint32_t D, uint32_t s, const HardwareModel& hw, double bound) {
        cfg.modelNs = std::numeric_limits<double>::infinity();
        if (static_cast<uint64_t>(M) * D == 0) { cfg.modelNs = 0.0; return; }
        if (cfg.mode == TilingMode::SPLIT_COLUMNS) return PlanBand(cfg, M, D, s, hw, bound);
        if (cfg.mode == TilingMode::ROW_PARALLEL && PlanRowTiles(cfg, M, D, s, hw, bound)) return;
        PlanColumnTiles(cfg, M, D, s, hw, bound);
    }

    // Row tiles: whole rows per tile at RowPitch, gamma/beta resident. The plan is the one whose
    // modeled timeline (RowTilesTimeline) finishes first, over
    //   depth         2-4 tiles in flight per input queue: deeper queues hide DMA latency, at
    //                 the price of smaller tiles in the same 191 KB
    //   outDepth      1-2 egress buffers: a second lets a result stream out while the next is
    //                 written
    //   B             body tile rows, up to the largest the layout admits (Challenge 2 knapsack)
    //   head, tail    a small first tile starts the vector unit sooner (prologue fill), a small
    //                 last tile finishes the final store sooner (epilogue drain)
    //   issue order   gamma/beta before tile 0; X1/X2 of a later tile as soon as the tile's Z
    //                 is built (both buffers are free then) rather than after its store
    // Every busy core holds the same rows up to one, so the busiest one is ceil(MaxLoad / D) rows.
    static inline bool PlanRowTiles(TilingConfig& cfg, uint32_t M, uint32_t D, uint32_t s, const HardwareModel& hw, double bound) {
        const uint64_t total = static_cast<uint64_t>(M) * D;
        const uint64_t rows = (MaxLoad(total, cfg.unitElems, cfg.blocks) + D - 1) / D;
        const dsa::MemorySystem mem = Memory(cfg, M, D, s);
        const uint32_t most = static_cast<uint32_t>(rows);  // Rows worth one tile
        // Stage 1: depth and body size, plain issue order; the best few go on
        constexpr uint32_t KEEP = 8;
        struct Found { double finish; RowPipe pp; } top[KEEP];
        for (Found& f : top) f.finish = std::numeric_limits<double>::infinity();
        bool fits = false;
        for (uint32_t depth = 2; depth <= 4; ++depth) {
            for (const uint32_t outDepth : {1u, 2u}) {
                uint32_t cap = 0;  // The largest body tile the layout admits (binary search over rows)
                for (uint32_t lo = 1, hi = most; lo <= hi;) {
                    const uint32_t mid = lo + (hi - lo) / 2;
                    if (RowLayout(mid, D, s, depth, outDepth).Total() <= hw.spmBytes) cap = mid, lo = mid + 1;
                    else hi = mid - 1;
                }
                if (cap) fits = true;
                for (uint32_t B = 1; cap && B <= cap; B = B < 48 ? B + 1 : std::max(B + 1, std::min(cap, B * 9 / 8))) {
                    const RowPipe pp{B, 0, 0, depth, outDepth, false, false};
                    const double f = RowTilesTimeline(rows, D, s, pp, hw, mem, std::min(bound, top[KEEP - 1].finish) * 1.05).finish;
                    for (uint32_t i = 0; i < KEEP; ++i) {
                        if (f < top[i].finish) {
                            for (uint32_t j = KEEP - 1; j > i; --j) top[j] = top[j - 1];
                            top[i] = {f, pp};
                            break;
                        }
                    }
                    if (B == cap) break;
                }
            }
        }
        if (!fits) return false;  // Not even one row fits: column tiles
        // Stage 2: head and tail tiles and issue order around the best few
        RowPipe best{};
        double bestTime = bound;
        for (const Found& f : top) {
            if (!std::isfinite(f.finish)) continue;
            for (const uint32_t head : {0u, 1u, 2u, 4u, 8u}) {
                if (head && (head >= f.pp.B || head >= rows)) continue;
                for (const uint32_t tail : {0u, 1u, 2u, 4u, 8u}) {
                    if (tail && (tail >= f.pp.B || tail >= rows)) continue;
                    for (const bool first : {false, true}) {
                        for (const bool early : {false, true}) {
                            RowPipe pp = f.pp;
                            pp.head = head, pp.tail = tail, pp.paramsFirst = first, pp.earlyLoads = early;
                            const double t = RowTilesTimeline(rows, D, s, pp, hw, mem, bestTime).finish;
                            if (t < bestTime) bestTime = t, best = pp;
                        }
                    }
                }
            }
        }
        // The body tile next to the best one may pair better with its head and tail
        for (int step : {-1, 1}) {
            if (!best.B) break;
            RowPipe pp = best;
            pp.B = static_cast<uint32_t>(static_cast<int>(best.B) + step);
            if (!pp.B || pp.head >= pp.B || pp.tail >= pp.B || RowLayout(pp.B, D, s, pp.depth, pp.outDepth).Total() > hw.spmBytes) continue;
            const double t = RowTilesTimeline(rows, D, s, pp, hw, mem, bestTime).finish;
            if (t < bestTime) bestTime = t, best = pp;
        }
        if (!best.B) {  // No row tiles beat the bound
            cfg.modelNs = std::numeric_limits<double>::infinity();
            return true;
        }
        ApplyRowPipe(cfg, M, D, s, hw, best);
        return true;
    }

    // Column band (SPLIT_COLUMNS): each core keeps the FP32 Z of its band of all M rows across
    // the barrier; tiles of k band rows stream X1/X2 by one DMA per row, gamma/beta come once.
    // Needs rows on the 32-byte grid (every row DMA is whole blocks), band rows that fit the
    // scratch chunk and the band's Z in the scratchpad; otherwise the plan is infeasible and
    // Plan() keeps another decomposition.
    static inline void PlanBand(TilingConfig& cfg, uint32_t M, uint32_t D, uint32_t s, const HardwareModel& hw, double bound) {
        const uint32_t q = QuantumElems(s, hw);
        if (D % q != 0 || M > 4096) return;
        uint32_t widest = 0;  // Band pitch: the widest band of any core, in elements
        for (uint32_t t = 0; t < cfg.blocks; ++t) {
            const CoreRange r = Range(cfg, M, D, t, cfg.blocks);
            if (r.rowZ > r.rowA) widest = std::max(widest, BandEnd(r, q) - BandBegin(r, q));
        }
        if (widest == 0 || !StridedRows(widest)) return;  // Band rows take the strided form's column passes
        const uint32_t R = RecordFloats(cfg, M);
        const uint32_t misc = Align32(uint64_t(cfg.blocks) * R * 4), rec = Align32(uint64_t(R) * 4);  // Gathered; published
        const uint32_t k0 = std::min(M, 8u);
        if (BandLayout(k0, widest, M, s, misc, rec).Total() > hw.spmBytes) return;
        cfg.pitch = widest;
        cfg.zResident = M * widest;
        cfg.layout = BandLayout(k0, widest, M, s, misc, rec);
        // The band tile whose modeled timeline (BandTimeline) finishes first: k rows per tile, a
        // multiple of 8 (a tile's packed row sums fill whole blocks of the record) or every row,
        // tiles in flight, egress buffers
        BandPipe best{k0, 2, 1};
        double bestTime = bound;
        for (uint32_t depth = 2; depth <= 4; ++depth) {
            for (const uint32_t outDepth : {1u, 2u}) {
                for (uint32_t k = k0;;) {
                    if (BandLayout(k, widest, M, s, misc, rec, depth, outDepth).Total() <= hw.spmBytes) {
                        const double t = BandTimeline(cfg, M, D, s, widest, {k, depth, outDepth}, hw, bestTime).finish;
                        if (t < bestTime) bestTime = t, best = {k, depth, outDepth};
                    }
                    if (k == M) break;
                    const uint32_t next = k < 128 ? k + 8 : (k * 9 / 8 + 7) / 8 * 8;
                    k = next >= M ? M : next;  // Always try one tile of every row
                }
            }
        }
        if (bestTime >= bound) return;  // Infeasible, or no better than the bound: modelNs stays infinite
        ApplyBandPipe(cfg, M, D, s, hw, best);
    }

    // Column tiles: Split-D fragments, or rows too long for a row tile. Keep the segment's
    // FP32 Z resident when a useful tile still fits (sweep 2 then re-reads nothing but gamma and beta).
    static inline void PlanColumnTiles(TilingConfig& cfg, uint32_t M, uint32_t D, uint32_t s, const HardwareModel& hw, double bound) {
        const uint64_t total = static_cast<uint64_t>(M) * D;
        const bool split = cfg.mode == TilingMode::SPLIT_D;
        const uint64_t L = MaxLoad(total, cfg.unitElems, cfg.blocks);
        const uint32_t q = QuantumElems(s, hw);
        // Split-D records: every core's two 32-byte records gathered (VECCALC), with room to pad a row's
        // run of them to whole 64-lane repeats; this core's published (VECOUT)
        const uint32_t misc = split ? Align32(uint64_t(LanePad(16ull * cfg.blocks)) * 4) : 0u, rec = split ? 64u : 0u;
        cfg.tileRows = 0;
        cfg.pitch = D;
        // The tile (and resident Z) whose modeled timeline (ColumnTimeline) finishes first;
        // largest tiles first, so the best time so far cuts the others short
        TilingConfig best = cfg;
        RowTimeline bestTime{bound, 0.0, 0.0};
        const uint64_t longest = (std::min<uint64_t>(D, split ? L : D) + q - 1) / q * q;
        // Resident Z: the segments of a core (split: two fragments and a row), with their alignment slack
        const uint64_t most = split ? std::min<uint64_t>(L, 3ull * D) + uint64_t(3) * q : uint64_t(D) + q;
        for (const uint64_t resident : {most, uint64_t(0)}) {
            uint64_t cap = 0;
            for (uint64_t lo = 1, hi = longest / q; lo <= hi;) {
                const uint64_t mid = lo + (hi - lo) / 2;
                if (ColumnLayout(mid * q, resident, s, misc, rec).Total() <= hw.spmBytes) cap = mid * q, lo = mid + 1;
                else hi = mid - 1;
            }
            if (!cap) continue;
            for (uint64_t tile = cap;; tile = std::max<uint64_t>(q, tile * 4 / 5 / q * q)) {
                TilingConfig c = cfg;
                c.tileElems = static_cast<uint32_t>(tile);
                c.zResident = static_cast<uint32_t>(resident);
                c.layout = ColumnLayout(tile, resident, s, misc, rec);
                const RowTimeline t = ColumnTimeline(c, M, D, s, hw, bestTime.finish);
                if (t.finish < bestTime.finish) best = c, bestTime = t;
                if (tile == q) break;
            }
            if (!resident) break;
        }
        if (bestTime.finish >= bound) {  // No better than the bound
            cfg.tileElems = best.tileElems ? best.tileElems : q;
            cfg.layout = ColumnLayout(cfg.tileElems, 0, s, misc, rec);
            cfg.modelNs = std::numeric_limits<double>::infinity();
            return;
        }
        cfg = best;
        SetModel(cfg, bestTime.finish, bestTime.vector, hw);
    }
};

} // namespace hpc
