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
    static constexpr uint64_t Op(uint64_t n) { return 2 * Repeats(n) + 13; }      // Add, Mul, Muls, Cast
    static constexpr uint64_t Fold(uint64_t n) { return Repeats(n) + 14; }        // BlockReduceSum (8 -> 1)
    static constexpr uint64_t Reduce(uint64_t n) { return 2 * Repeats(n) + 15; }  // VectorReduceSum
    static constexpr uint64_t Fill(uint64_t n) { return Repeats(n) + 18; }        // Duplicate
    static constexpr uint64_t kInvRms = 16;                                       // VectorInvRms
    // Sum `runs` runs of `len` squares: one 8 -> 1 fold first when that is cheaper
    static constexpr bool FoldFirst(uint64_t runs, uint64_t len) {
        return len % 8 == 0 && Fold(runs * len) + runs * Reduce(len / 8) < runs * Reduce(len);
    }
    static constexpr uint64_t ReduceRuns(uint64_t runs, uint64_t len) {
        return FoldFirst(runs, len) ? Fold(runs * len) + runs * Reduce(len / 8) : runs * Reduce(len);
    }
};

// -----------------------------------------------------------------------------
// Machine description. Every number the planning equations use comes from here:
//   Target()  the deployment machine: 40 symmetric DAE cores, 32-byte DMA blocks,
//             191 KB scratchpad per core, ~0 fork/join, and the runtime's timing
//             (include/dsa_runtime.hpp): cycle costs, clock, DMA bandwidth and latency.
//   Host(P)   the same machine laws run by P CI threads with the host's measured costs.
// -----------------------------------------------------------------------------
struct HardwareModel {
    uint32_t cores;         // Symmetric cores
    uint32_t quantumBytes;  // DMA block: split boundaries and block transfers are multiples of it
    size_t   spmBytes;      // Per-core scratchpad budget
    double   launchNs;      // Engaging all cores (fork/join)
    double   syncNs;        // One all-core barrier (SyncAll)
    double   elemNs;        // Host: vector work per element on one core
    double   byteNs;        // Streaming cost per byte on one core (1 / per-core bandwidth)
    double   latencyNs;     // Target: a DMA transfer's data lands this long after it streamed; 0 on the host
    double   clockGHz;      // Target: converts the runtime's vector cycles (DaeIsa) to ns; 0 on the host
    uint32_t batchElems;    // Host: rows whose pass-1 Z stays L1-resident together

    static HardwareModel Target() {
        HardwareModel hw{};
        hw.cores = dsa::MAX_HARDWARE_CORES;
        hw.quantumBytes = dsa::DMA_ALIGN_BYTES;
        hw.spmBytes = dsa::SCRATCHPAD_SAFE_WATERLINE;
        hw.launchNs = 0.0;                                                  // Zero fork/join cost
        hw.clockGHz = dsa::CLOCK_GHZ;                                       // 1.5, assumed (docs)
        hw.syncNs = dsa::SYNC_ALL_CYCLES / hw.clockGHz;                     // SyncAll: 7500 cycles
        hw.elemNs = 0.0;                                                    // Vector work: the DaeIsa cycle model
        hw.byteNs = 1.0 / (dsa::DMA_BYTES_PER_CYCLE * dsa::CLOCK_GHZ);      // 850 GB/s shared by 40 cores
        hw.latencyNs = dsa::DMA_LATENCY_CYCLES / dsa::CLOCK_GHZ;            // 800 ns, inferred from the targets (docs)
        hw.batchElems = 0;
        return hw;
    }

    // The target's timing in vector cycles, as the timeline models use it
    double BytesPerCycle() const { return 1.0 / (byteNs * clockGHz); }
    double LatencyCycles() const { return latencyNs * clockGHz; }
    double SyncCycles() const { return syncNs * clockGHz; }

    static HardwareModel Host(uint32_t threads) {
        HardwareModel hw = Target();
        hw.cores = std::max(1u, std::min(threads, hw.cores));
        hw.launchNs = 3500.0;  // Measured on the 4-core CI VM (docs)
        hw.syncNs = 1000.0;
        hw.elemNs = 0.3;       // Cache-resident rows are compute-bound: any dtype
        hw.byteNs = 0.0;
        hw.latencyNs = 0.0;    // Hardware prefetchers, no explicit DMA
        hw.clockGHz = 0.0;     // No DAE cycle model: SIMD work is elemNs
        hw.batchElems = 2048;
        return hw;
    }
};

// Per-core scratchpad layout of the DAE pipeline. The kernel claims exactly these
// buffers through TPipe, so the planner's total is what the sanitizer checks.
struct DaeLayout {
    uint32_t tile;       // One X1 / X2 / parameter-chunk buffer (native dtype), `depth` of each
    uint32_t z;          // FP32 Z of one tile (row tiles in FP32 work in place)
    uint32_t tmp;        // FP32 scratch: casts, squares, folds; staging of gamma/beta
    uint32_t params;     // Resident FP32 gamma + beta, each replicated repRows times
    uint32_t resident;   // Resident FP32 Z: column-tiled segments, or a core's whole column band
    uint32_t misc;       // Split-D partial-sum records (published + gathered)
    bool paramQueue;     // Column tiles stream bias/gamma chunks
    uint32_t depth = 2;  // Buffers per queue: tiles in flight
    uint32_t Total() const { return depth * tile * (paramQueue ? 3 : 2) + z + tmp + params + resident + misc; }
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
    bool earlyX2;            // Row tiles: X2 of a later tile loads as soon as its buffer frees
    uint32_t tileElems;      // Elements per X1/X2 tile
    uint32_t pitch;          // Row pitch inside a tile: D, or the column band width (SPLIT_COLUMNS)
    uint32_t repRows;        // gamma/beta rows replicated in the scratchpad: one op covers repRows rows
    uint32_t zResident;      // FP32 Z elements kept resident (column tiles and bands)
    DaeLayout layout;
    double modelNs;          // Modeled time of the busiest core: what Plan() minimizes
    uint64_t modelCycles;    // Target: modeled vector cycles of the busiest core (barriers excluded)
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
    // (head/tail 0: none). Head, body and tail are whole row units, so every tile starts on the
    // 32-byte grid.
    struct RowSchedule {
        uint32_t rA, rZ, B, head, tail;
        uint32_t Rows(uint32_t row) const {
            const uint32_t left = rZ - row;
            if (row == rA && head && head < left) return head;
            if (tail && left > tail) return std::min(B, left - tail);
            return std::min(B, left);
        }
    };

    static constexpr uint32_t MAX_THREADS = dsa::MAX_HARDWARE_CORES;
    static constexpr uint32_t MAX_BATCH = 64;
    static constexpr uint32_t TMP_BYTES = 8192;          // DAE FP32 scratch chunk
    static constexpr uint32_t TMP_FLOATS = TMP_BYTES / sizeof(float);
    static constexpr uint32_t REP_FLOATS = 2048;         // DAE replicated gamma (and beta): <= 8 KB each
    static constexpr uint32_t ROW_GROUP = 128;           // DAE row sums in flight at once: a worker's fixed per-row arrays
    static constexpr double CHUNK_BYTES = 64 * 1024;     // Host serpentine chunk (prefetch-friendly run)

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
        const uint64_t e0 = std::min(cfg.units * t / n * cfg.unitElems, total);
        const uint64_t e1 = std::min(cfg.units * (t + 1) / n * cfg.unitElems, total);
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
    //   inline         one core, no launch
    //   ROW_PARALLEL   units of p rows, p = 32 / gcd(32, D*s): whole rows AND whole DMA blocks
    //   SPLIT_D        units of one 32-byte block [ARCH CHALLENGE 1]: every core within one
    //                  block of the mean for any M and any core count, plus one SyncAll
    //   SPLIT_COLUMNS  the same blocks numbered column-major (target only): same balance, and
    //                  each core reads gamma/beta for its column band once instead of per row
    // Host cost: launch + barrier + elements * elemNs. Target cost: the finish time of the
    // slowest core on the timeline models of (3), SyncAll included.
    // -------------------------------------------------------------------------
    static inline TilingConfig Plan(uint32_t M, uint32_t D, uint32_t elemBytes, const HardwareModel& hw,
                                    size_t llcBytes = 0) {
        // Inline on the caller is a candidate where engaging the cores costs something (the host);
        // on the target it cannot beat the same rows on min(cores, units) cores
        const bool inlineCandidate = hw.launchNs > 0 || hw.cores < 2 || static_cast<uint64_t>(M) * D == 0;
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

        if (hw.clockGHz > 0) {
            PlanTiles(cfg, M, D, elemBytes, hw, boundNs * hw.clockGHz);  // Explicit DMA pipeline: tiles, layout, modeled time
        } else {
            const double sync = mode == TilingMode::ROW_PARALLEL ? 0.0 : hw.syncNs;
            cfg.modelNs = (cfg.threads > 1 ? hw.launchNs : 0.0) + sync + MaxLoad(total, cfg.unitElems, cfg.blocks) * hw.elemNs;
        }
        return cfg;
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

    // p whole rows per unit, with p = q / gcd(q, D*s): the smallest row group that ends on a DMA block
    static inline uint64_t RowUnit(uint32_t D, uint32_t elemBytes, const HardwareModel& hw) {
        const uint64_t rowBytes = static_cast<uint64_t>(D) * elemBytes;
        if (rowBytes == 0) return 1;
        return hw.quantumBytes / std::gcd<uint64_t>(hw.quantumBytes, rowBytes) * D;
    }

    static inline uint32_t Align32(uint64_t bytes) {
        return static_cast<uint32_t>((bytes + dsa::DMA_ALIGN_BYTES - 1) / dsa::DMA_ALIGN_BYTES * dsa::DMA_ALIGN_BYTES);
    }

    // gamma and beta blocks of `rep` FP32 rows each, every block on the 32-byte grid
    static inline uint32_t ParamBytes(uint64_t rep, uint64_t pitch) { return 2 * Align32(4 * rep * pitch); }

    static inline DaeLayout RowLayout(uint64_t rows, uint32_t D, uint32_t s, uint64_t rep, uint32_t depth = 2) {
        const uint64_t e = rows * D;
        return {Align32(e * s), s < 4 ? Align32(e * 4) : 0u, TMP_BYTES, ParamBytes(rep, D), 0u, 0u, false, depth};
    }

    static inline DaeLayout BandLayout(uint64_t rows, uint32_t pitch, uint32_t M, uint32_t s, uint64_t rep, uint32_t misc,
                                       uint32_t depth = 2) {
        return {Align32(rows * pitch * s), 0u, TMP_BYTES, ParamBytes(rep, pitch), Align32(uint64_t(M) * pitch * 4), misc, false, depth};
    }

    static inline DaeLayout ColumnLayout(uint64_t tile, uint64_t resident, uint32_t s, uint32_t misc) {
        return {Align32(tile * s), Align32(tile * 4), TMP_BYTES, 0u, Align32(resident * 4), misc, true};
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
    // One tile of k rows at pitch w: Z = X1 + X2 + beta, squares and row sums, inverse RMS,
    // scale, gamma, narrow. `lens` gives band rows (SPLIT_COLUMNS) their own column counts
    // (0: empty row); null means k full rows of w. Row sums go in groups of ROW_GROUP rows, the
    // squares of a group through the scratch chunk as many rows at a time as fit.
    static inline uint64_t TileCycles(uint64_t k, uint64_t w, uint32_t s, uint64_t rep, const uint32_t* lens) {
        using I = DaeIsa;
        const uint64_t e = k * w;
        uint64_t c = I::Op(e);  // FP32: X1 + X2; 16-bit: widen X1
        if (s < 4) {
            for (uint64_t o = 0; o < e; o += TMP_FLOATS) c += 2 * I::Op(std::min<uint64_t>(TMP_FLOATS, e - o));
        }
        for (uint64_t g = 0; g < k; g += rep) c += 2 * I::Op(std::min(rep, k - g) * w);  // beta, gamma
        if (w <= TMP_FLOATS) {
            const uint64_t per = TMP_FLOATS / w;
            for (uint64_t g = 0; g < k; g += ROW_GROUP) {
                const uint64_t end = std::min<uint64_t>(k, g + ROW_GROUP);
                for (uint64_t r0 = g; r0 < end; r0 += per) {
                    const uint64_t m = std::min(per, end - r0);
                    c += I::Op(m * w);
                    if (!lens) c += I::ReduceRuns(m, w);
                    else for (uint64_t i = r0; i < r0 + m; ++i) c += lens[i] ? I::Reduce(lens[i]) : 0;
                }
            }
        } else {
            c += k * SquareSumCycles(w);
        }
        for (uint64_t i = 0; i < k; ++i) {
            const uint64_t len = lens ? lens[i] : w;
            c += len ? I::kInvRms + I::Op(len) : 0;  // Inverse RMS and scale
        }
        if (s < 4) c += I::Op(e);  // Narrow into the X1 slot
        return c;
    }

    // Sum of squares of one contiguous run, in scratch-sized pieces
    static inline uint64_t SquareSumCycles(uint64_t n) {
        uint64_t c = 0;
        for (uint64_t o = 0; o < n; o += TMP_FLOATS) {
            const uint64_t m = std::min<uint64_t>(TMP_FLOATS, n - o);
            c += DaeIsa::Op(m) + DaeIsa::ReduceRuns(1, m);
        }
        return c;
    }

    // Widening gamma and beta of w columns (FP32 parameters arrive by DMA only): in one piece
    // when both fit the staging buffer (`stageBytes`: the Z tile or the scratch buffer)
    static inline uint64_t ParamCycles(uint64_t w, uint32_t s, uint64_t stageBytes = TMP_BYTES) {
        if (s == 4) return 0;
        if (2ull * Align32(w * s) <= stageBytes) return 2 * DaeIsa::Op(w);
        uint64_t c = 0;
        const uint64_t chunk = TMP_BYTES / s;
        for (uint64_t o = 0; o < w; o += chunk) c += 2 * DaeIsa::Op(std::min(chunk, w - o));
        return c;
    }

    // Column tiles over [start, end) of the flattened tensor: at most `tile` elements each,
    // interior boundaries on the 32-byte grid (q elements), as DaePipeline cuts them
    static inline uint64_t NextTile(uint64_t e, uint64_t end, uint64_t tile, uint32_t q) {
        return std::min(end, e / q * q + tile);
    }

    // One column-tiled row segment: sweep 1 (+ beta) and sweep 2 (+ gamma)
    static inline uint64_t SegmentCycles(uint64_t start, uint64_t end, uint64_t tile, uint32_t q, uint32_t s, bool resident) {
        using I = DaeIsa;
        auto chunks = [&](uint64_t n) {  // A widen + op pair per scratch-sized piece (16-bit data)
            uint64_t c = 0;
            for (uint64_t o = 0; o < n; o += TMP_FLOATS) c += 2 * I::Op(std::min<uint64_t>(TMP_FLOATS, n - o));
            return c;
        };
        uint64_t c = I::kInvRms;
        for (uint64_t e = start; e < end;) {
            const uint64_t next = NextTile(e, end, tile, q), n = next - e;
            const uint64_t z = s < 4 ? I::Op(n) + chunks(n) : I::Op(n);  // Z = X1 + X2
            const uint64_t param = s < 4 ? chunks(n) : I::Op(n);         // (+|*) a parameter chunk
            c += z + param + SquareSumCycles(n);                          // Sweep 1
            if (!resident) c += z + param;                                // Sweep 2 recomputes Z
            c += I::Op(n) + param + I::Op(n);                             // Muls, gamma, narrow
            e = next;
        }
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
        uint32_t rep, depth;     // Replicated parameter rows; tiles in flight per queue
        bool paramsFirst, earlyX2;
    };
    struct RowTimeline { double finish, vector, dma; };

    // A row-tile plan with the given scheduling choices. Its modeled time and vector cycles
    // are its slowest core's (cores differ by one row unit: every distinct row count counts).
    static inline void ApplyRowPipe(TilingConfig& cfg, uint32_t M, uint32_t D, uint32_t s, const HardwareModel& hw,
                                    const RowPipe& pp) {
        double finish = 0, vector = 0;
        uint32_t seen[3] = {0, 0, 0}, nSeen = 0;
        for (uint32_t t = 0; t < cfg.blocks; ++t) {
            const CoreRange r = Range(cfg, M, D, t, cfg.blocks);
            const uint32_t rows = r.rowZ - r.rowA;
            if (!rows || std::find(seen, seen + nSeen, rows) != seen + nSeen) continue;
            if (nSeen < 3) seen[nSeen++] = rows;
            const RowTimeline x = RowTilesTimeline(rows, D, s, pp, hw);
            finish = std::max(finish, x.finish);
            vector = std::max(vector, x.vector);
        }
        cfg.tileRows = pp.B;
        cfg.tileElems = pp.B * D;
        cfg.headRows = pp.head;
        cfg.tailRows = pp.tail;
        cfg.pitch = D;
        cfg.repRows = pp.rep;
        cfg.paramsFirst = pp.paramsFirst;
        cfg.earlyX2 = pp.earlyX2;
        cfg.zResident = 0;
        cfg.layout = RowLayout(pp.B, D, s, pp.rep, pp.depth);
        cfg.modelCycles = static_cast<uint64_t>(vector + 0.5);
        cfg.modelNs = finish / hw.clockGHz;
    }

    // `bound`: give up (finish = infinity) once the core cannot finish before it. Uniform body
    // tiles make the replay a max-plus linear recurrence: once the whole state has shifted by the
    // same amount over c tiles it keeps doing so, and the replay jumps ahead by whole periods.
    static inline RowTimeline RowTilesTimeline(uint64_t rows, uint32_t D, uint32_t s, const RowPipe& pp, const HardwareModel& hw,
                                               double bound = std::numeric_limits<double>::infinity()) {
        using I = DaeIsa;
        const double bw = hw.BytesPerCycle(), L = hw.LatencyCycles(), lbw = dsa::LOCAL_BYTES_PER_CYCLE;
        const uint32_t d = pp.depth, R = static_cast<uint32_t>(rows), B = pp.B;
        RowTimeline out{std::numeric_limits<double>::infinity(), 0.0, 0.0};
        // The core's tiles, as RowSchedule cuts them: [head] [B x body] [partial] [tail]
        const uint32_t head = pp.head && pp.head < R ? pp.head : 0, rest = R - head;
        const uint32_t tail = pp.tail && rest > pp.tail ? pp.tail : 0;
        const uint32_t body = (rest - tail) / B, partial = (rest - tail) % B;
        auto tileVector = [&](uint32_t k) { return k ? TileCycles(k, D, s, pp.rep, nullptr) : 0; };
        auto tileBytes = [&](uint32_t k) { return static_cast<double>(Align32(uint64_t(k) * D * s)); };
        const uint64_t zBytes = RowLayout(B, D, s, pp.rep, d).z, stage = std::max<uint64_t>(TMP_BYTES, zBytes);
        const bool staged = s == 4 || 2ull * Align32(uint64_t(D) * s) <= stage;
        const uint64_t chunk = TMP_BYTES / s;  // Unstaged 16-bit parameters: scratch-sized pieces
        double paramBytes = 0;
        if (staged) paramBytes = 2.0 * Align32(uint64_t(D) * s);
        else for (uint64_t o = 0; o < D; o += chunk) paramBytes += 2.0 * Align32(std::min<uint64_t>(chunk, D - o) * s);
        const uint64_t bodyVector = tileVector(B);
        uint64_t vectorLeft = ParamCycles(D, s, stage) + tileVector(head) + body * bodyVector + tileVector(partial) + tileVector(tail);
        double channelLeft = paramBytes + 3 * (tileBytes(head) + body * tileBytes(B) + tileBytes(partial) + tileBytes(tail));
        const uint32_t last = tail ? tail : partial ? partial : body ? B : head;
        // Lower bound: the vector unit starts after a load landed and the last store lands after
        // it; the channel streams everything and the last store lands after that
        if (std::max(L + vectorLeft + tileBytes(last) / bw + L, channelLeft / bw + L) >= bound) return out;

        const RowSchedule tiles{0, R, B, pp.head, pp.tail};
        double ch = 0, vec = 0, local = 0, loads = 0, stores = 0;
        double land1[4] = {}, land2[4] = {}, free1[4] = {};
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
            const double start = std::max(ch, ready), occupied = bytes / bw;
            ch = start + occupied;
            out.dma += occupied;
            channelLeft -= bytes;
            return ch + L;
        };
        auto load1 = [&] {
            const uint32_t k = tiles.Rows(next1), slot = static_cast<uint32_t>(n1++ % d);
            land1[slot] = transfer(free1[slot], tileBytes(k));
            loads = std::max(loads, land1[slot]);
            next1 += k;
        };
        auto load2 = [&] {
            const uint32_t k = tiles.Rows(next2), slot = static_cast<uint32_t>(n2++ % d);
            land2[slot] = transfer(x2Free(slot, uint64_t(k) * D), tileBytes(k));
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
        double readyG = landG, readyB = landB;  // gamma / beta row 0 widened in par
        if (s < 4 && staged) {
            readyG = compute(landG, I::Op(D));
            readyB = compute(landB, I::Op(D));
        } else if (s < 4) {  // Scratch-sized pieces, each loaded once the previous one is widened
            double tmpFree = 0;
            for (int k = 0; k < 2; ++k) {
                for (uint64_t o = 0; o < D; o += chunk) {
                    const uint64_t m = std::min<uint64_t>(chunk, D - o);
                    const double landed = transfer(tmpFree, Align32(m * s));
                    loads = std::max(loads, landed);
                    tmpFree = compute(landed, I::Op(m));
                }
                (k == 0 ? readyG : readyB) = vec;
            }
        }
        // Rows [1, rep) by doubling local copies; copied[j]: rows [0, 2^(j+1)) are in place
        double copiedG[32], copiedB[32];
        auto replicate = [&](double ready, double* copied) {
            uint32_t j = 0;
            for (uint32_t h = 1; h < pp.rep; h *= 2, ++j) {
                local = std::max(local, ready) + static_cast<double>(std::min(h, pp.rep - h)) * D * 4 / lbw;
                copied[j] = ready = local;
            }
        };
        replicate(readyG, copiedG);
        replicate(readyB, copiedB);
        auto betaRows = [&](uint32_t n) {  // Beta rows [0, n) in place: a tile adds min(rep, k) at once
            double t = readyB;
            uint32_t j = 0;
            for (uint32_t h = 1; h < pp.rep && h < n; h *= 2, ++j) t = copiedB[j];
            return t;
        };
        // Steady-state detection: snapshots of the state, rotated to the next tile's buffers
        constexpr uint32_t HIST = 8;
        double hist[HIST][4 + 16];
        const uint32_t width = 4 + 4 * d, firstBody = head ? 1u : 0u, bodyEnd = firstBody + body;  // Tile indices
        auto snapshot = [&](double* v) {
            v[0] = vec, v[1] = ch, v[2] = loads, v[3] = stores;
            for (uint32_t i = 0; i < d; ++i) {
                const uint32_t slot = static_cast<uint32_t>((done + i) % d);
                v[4 + i] = land1[slot], v[4 + d + i] = land2[slot], v[4 + 2 * d + i] = free1[slot];
                v[4 + 3 * d + i] = x2Chunks[slot] ? x2Read[slot][x2Chunks[slot] - 1] : 0.0;
            }
        };
        auto shifted = [&](const double* a, const double* b, double& by) {  // a = b + by in every component
            by = a[0] - b[0];
            for (uint32_t i = 1; i < width; ++i) {
                if (std::fabs((a[i] - b[i]) - by) > 1e-9 * std::max(1.0, std::fabs(a[i]))) return false;
            }
            return true;
        };
        for (uint32_t row = 0; row < R;) {
            const uint32_t k = tiles.Rows(row), slot = static_cast<uint32_t>(done % d);
            const uint64_t e = uint64_t(k) * D, C = TileCycles(k, D, s, pp.rep, nullptr);
            const double parB = betaRows(std::min(pp.rep, k));
            if (s == 4) {
                const double start = std::max({vec, land1[slot], land2[slot]});
                x2Read[slot][0] = start + I::Op(e);  // X2 is read by the add, all of it at once
                x2Chunks[slot] = 1;
                vec = std::max(start + I::Op(e), parB) + static_cast<double>(C - I::Op(e));
            } else {
                const double start = std::max(vec, land1[slot]), x2 = std::max(start + I::Op(e), land2[slot]);
                double phase = 0;
                x2Chunks[slot] = 0;
                for (uint64_t o = 0; o < e; o += TMP_FLOATS) {  // Widen X2 and add, a scratch chunk at a time
                    const uint64_t m = std::min<uint64_t>(TMP_FLOATS, e - o);
                    if (x2Chunks[slot] < CHUNKS) x2Read[slot][x2Chunks[slot]++] = x2 + phase + I::Op(m);
                    else x2Read[slot][CHUNKS - 1] = x2 + phase + I::Op(m);
                    phase += 2.0 * I::Op(m);
                }
                vec = std::max(x2 + phase, parB) + static_cast<double>(C - I::Op(e)) - phase;
            }
            out.vector += static_cast<double>(C);
            vectorLeft -= C;
            if (pp.earlyX2 && next2 < R) load2();
            // Y leaves through the X1 buffer: the store follows the tile's last instruction
            const double start = std::max(ch, vec), occupied = tileBytes(k) / bw;
            ch = start + occupied;
            out.dma += occupied;
            channelLeft -= tileBytes(k);
            free1[slot] = ch;
            stores = std::max(stores, ch + L);
            if (next1 < R) load1();
            if (!pp.earlyX2 && next2 < R) load2();
            row += k;
            ++done;
            if (std::max(vec + vectorLeft + tileBytes(last) / bw + L, ch + channelLeft / bw + L) >= bound) {
                out.finish = std::numeric_limits<double>::infinity();
                return out;
            }
            // In the body, once periodic, jump whole periods (a multiple of d keeps the buffers)
            if (done < firstBody + 2 || done >= bodyEnd) continue;
            snapshot(hist[done % HIST]);
            for (uint32_t c = 1; c <= 4 && done >= firstBody + 2 * c + 1; ++c) {
                double by = 0, by2 = 0;
                if (!shifted(hist[done % HIST], hist[(done - c) % HIST], by) ||
                    !shifted(hist[(done - 1) % HIST], hist[(done - 1 - c) % HIST], by2) || std::fabs(by - by2) > 1e-9 * std::max(1.0, by)) {
                    continue;
                }
                uint32_t period = c;
                while (period % d) period += c;
                // Every load issued inside the jump must still be a body tile
                const uint64_t ahead = std::max(n1, n2) - done;
                const uint64_t room = bodyEnd > done + ahead + 1 ? bodyEnd - done - ahead - 1 : 0;
                const uint64_t m = room / period * period;
                if (!m) break;
                const double shift = by * static_cast<double>(m / c);
                vec += shift, ch += shift, loads += shift, stores += shift;
                for (uint32_t i = 0; i < d; ++i) {
                    land1[i] += shift, land2[i] += shift, free1[i] += shift;
                    for (uint32_t c2 = 0; c2 < x2Chunks[i]; ++c2) x2Read[i][c2] += shift;
                }
                row += static_cast<uint32_t>(m) * B, next1 += static_cast<uint32_t>(m) * B, next2 += static_cast<uint32_t>(m) * B;
                done += m, n1 += m, n2 += m;
                out.vector += static_cast<double>(m * bodyVector);
                out.dma += static_cast<double>(m) * 3 * tileBytes(B) / bw;
                vectorLeft -= m * bodyVector;
                channelLeft -= static_cast<double>(m) * 3 * tileBytes(B);
                break;
            }
        }
        out.finish = std::max({vec, ch, local, loads, stores});
        return out;
    }

    // Timeline of a column band (SPLIT_COLUMNS), mirroring BandPhase1 / SyncAll / BandPhase2 for
    // every core: phase 1 loads k band rows per tile (one DMA per row and input) and publishes a
    // record of partials; all cores leave SyncAll SYNC_ALL_CYCLES after the last one stored its
    // record; phase 2 gathers the records, scales the resident Z and stores it row by row.
    struct BandPipe {
        uint32_t k, rep, depth;  // Tile rows, replicated parameter rows, tiles in flight
    };
    static inline RowTimeline BandTimeline(const TilingConfig& cfg, uint32_t M, uint32_t D, uint32_t s, uint32_t pitch,
                                           const BandPipe& bp, const HardwareModel& hw,
                                           double bound = std::numeric_limits<double>::infinity()) {
        using I = DaeIsa;
        const double bw = hw.BytesPerCycle(), L = hw.LatencyCycles(), lbw = dsa::LOCAL_BYTES_PER_CYCLE;
        const uint32_t q = QuantumElems(s, hw), R = RecordFloats(cfg, M), nb = cfg.blocks;
        const uint32_t per = TMP_FLOATS / pitch;  // Band rows whose squares fit the scratch chunk
        RowTimeline out{0.0, 0.0, 0.0};
        double arrive[MAX_THREADS] = {}, phase1[MAX_THREADS] = {};  // Per core: SyncAll arrival, phase-1 vector cycles
        uint32_t lens[4096], offs[4096];
        if (M > 4096 || nb > MAX_THREADS) return out.finish = std::numeric_limits<double>::infinity(), out;
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
            double ch = 0, vec = 0, local = 0, cycles = 0, dma = 0;
            double free1[4] = {};                       // Per buffer: when X1's last read ended
            double x2Read[4][64] = {};                  // Per buffer: X2 read ends, a scratch chunk at a time
            uint32_t x2Chunks[4] = {};
            double rowLand1[4096], rowLand2[4096];      // Per row: when its X1 / X2 landed
            auto transfer = [&](double ready, double bytes) {
                const double start = std::max(ch, ready), occupied = bytes / bw;
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
            double readyG = landG, readyB = landB;
            if (s < 4) {
                readyG = compute(landG, I::Op(w));
                readyB = compute(landB, I::Op(w));
            }
            double copiedB[32];
            auto replicate = [&](double ready, double* copied) {
                uint32_t j = 0;
                for (uint32_t h = 1; h < bp.rep; h *= 2, ++j) {
                    local = std::max(local, ready) + static_cast<double>(std::min(h, bp.rep - h)) * pitch * 4 / lbw;
                    if (copied) copied[j] = local;
                    ready = local;
                }
            };
            replicate(readyG, nullptr);
            replicate(readyB, copiedB);
            auto betaRows = [&](uint32_t n) {
                double t0 = readyB;
                uint32_t j = 0;
                for (uint32_t h = 1; h < bp.rep && h < n; h *= 2, ++j) t0 = copiedB[j];
                return t0;
            };
            compute(0, I::Fill(R));  // The record, zeroed
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
                const double parB = betaRows(std::min(bp.rep, k));
                for (uint32_t g = 0; g < k; g += bp.rep) compute(g == 0 ? parB : 0, I::Op(uint64_t(std::min(bp.rep, k - g)) * pitch));
                for (uint32_t g = 0; g < k; g += ROW_GROUP) {
                    const uint32_t end = std::min(k, g + ROW_GROUP);
                    for (uint32_t r0 = g; r0 < end; r0 += per) {
                        const uint32_t m = std::min(per, end - r0);
                        compute(0, I::Op(uint64_t(m) * pitch));
                        for (uint32_t i = r0; i < r0 + m; ++i) if (lens[row + i]) compute(0, I::Reduce(lens[row + i]));
                    }
                }
                if (loaded < M) load();
            }
            // The record goes out; SyncAll waits for it to land
            const double start = std::max(ch, vec);
            ch = start + R * 4.0 / bw;
            dma += R * 4.0 / bw;
            arrive[t] = std::max({vec, local, ch + L});
            phase1[t] = cycles;
            out.dma = std::max(out.dma, dma);
            if (arrive[t] >= bound) return out.finish = std::numeric_limits<double>::infinity(), out;
        }
        double release = 0;
        for (uint32_t t = 0; t < nb; ++t) release = std::max(release, arrive[t]);
        release += hw.SyncCycles();
        // Phase 2 of every core, from the release
        double worst = 0;
        double gammaEnd[4096], outRead[2][4096];
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
            // 16-bit narrow buffers: when each row of the tile they last held streamed out
            uint32_t outRows[2] = {0, 0};
            auto compute = [&](double ready, uint64_t c) {
                vec = std::max(vec, ready) + static_cast<double>(c);
                cycles += static_cast<double>(c);
                return vec;
            };
            ch += nb * R * 4.0 / bw;
            const double gathered = ch + L;
            compute(gathered, 0);
            for (uint64_t n = nb; n > 1; n -= n / 2) compute(0, I::Op(n / 2 * R));
            uint32_t tile = 0;
            for (uint32_t row = 0; row < M; row += bp.k, ++tile) {
                const uint32_t k = std::min(bp.k, M - row);
                const uint64_t e = uint64_t(k) * pitch;
                for (uint32_t i = 0; i < k; ++i) compute(0, I::Reduce(1));
                for (uint32_t i = 0; i < k; ++i) if (lens[row + i]) compute(0, I::kInvRms + I::Op(lens[row + i]));
                uint32_t groups = 0;
                for (uint32_t g = 0; g < k; g += bp.rep) gammaEnd[groups++] = compute(0, I::Op(uint64_t(std::min(bp.rep, k - g)) * pitch));
                const uint32_t buf = tile % 2;  // Tiles alternate between two X1 buffers
                if (s < 4) {  // Narrow into the X1 buffer, once the rows it overwrites streamed out
                    double freeAt = release;
                    for (uint32_t i = 0; i < std::min(k, outRows[buf]); ++i) freeAt = std::max(freeAt, outRead[buf][i]);
                    compute(freeAt, I::Op(e));
                }
                for (uint32_t i = 0; i < k; ++i) {
                    if (s < 4) outRead[buf][i] = 0;
                    if (!lens[row + i]) continue;
                    const double ready = s < 4 ? vec : gammaEnd[i / bp.rep];
                    const double start = std::max(ch, ready);
                    ch = start + Align32(uint64_t(lens[row + i]) * s) / bw;
                    if (s < 4) outRead[buf][i] = ch;
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

    // Timeline of column tiles (row-major Split-D fragments, and rows too long for a row tile),
    // mirroring ColumnPhase1 / SyncAll / ColumnPhase2 with their Sweep1 / Sweep2 for every core.
    // X1, X2 and the parameter chunks each alternate between two buffers; a buffer is refilled
    // once the reads of the elements it overwrites ended (16-bit reads go a scratch chunk at a time).
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
        double bw, L;
        double ch = 0, vec = 0, loads = 0, stores = 0, cycles = 0, dma = 0;
        Buffer x1[2], x2[2], p[2];
        uint32_t outNext = 0;

        double Transfer(double ready, double bytes) {
            const double start = std::max(ch, ready), occupied = bytes / bw;
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
            const double start = std::max(ch, vec);
            ch = start + Bytes(n) / bw;
            dma += Bytes(n) / bw;
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
        void SquareSums(uint64_t n) {
            for (uint64_t o = 0; o < n; o += TMP_FLOATS) {
                const uint64_t m = std::min<uint64_t>(TMP_FLOATS, n - o);
                Compute(0, DaeIsa::Op(m) + DaeIsa::ReduceRuns(1, m));
            }
        }
        // Sweep 1 over [start, end): X1, X2 and beta of tile j+1 load before tile j computes
        void Sweep1(uint64_t start, uint64_t end) {
            uint32_t j = 0;
            uint64_t e = start;
            uint32_t n = start < end ? TileLength(start, end) : 0;
            auto load = [&](uint32_t slot, uint64_t k) { Load(x1[slot], k), Load(x2[slot], k), Load(p[slot], k); };
            if (n) load(0, n);
            while (e < end) {
                const uint64_t next = e + n;
                const uint32_t nn = next < end ? TileLength(next, end) : 0, slot = j % 2;
                if (nn) load((j + 1) % 2, nn);
                AddInputs(x1[slot], x2[slot], n);
                Combine(p[slot], n);  // beta
                SquareSums(n);
                e = next, n = nn, ++j;
            }
        }
        // Sweep 2 from resident Z: gamma chunks stream (the first may be in flight: `primed`)
        void Sweep2Resident(uint64_t start, uint64_t end, bool primed) {
            Compute(0, DaeIsa::kInvRms);
            uint32_t j = 0;
            uint64_t e = start;
            uint32_t n = start < end ? TileLength(start, end) : 0;
            if (n && !primed) Load(p[0], n);
            while (e < end) {
                const uint64_t next = e + n;
                const uint32_t nn = next < end ? TileLength(next, end) : 0, slot = j % 2;
                if (nn) Load(p[(j + 1) % 2], nn);
                Compute(0, DaeIsa::Op(n));  // Muls
                Combine(p[slot], n);        // gamma
                Buffer& out = x1[0];  // The lowest free X1 buffer: the same one every tile
                Compute(out.FreeFor(n), DaeIsa::Op(n));  // Narrow into an X1 buffer
                Store(out, n);
                e = next, n = nn, ++j;
            }
        }
        // Sweep 2 recomputing Z: X1, X2 and beta re-stream; each gamma chunk loads on the spot
        void Sweep2Recompute(uint64_t start, uint64_t end) {
            Compute(0, DaeIsa::kInvRms);
            uint32_t j = 0;
            uint64_t e = start;
            uint32_t n = start < end ? TileLength(start, end) : 0;
            auto load = [&](uint32_t slot, uint64_t k) { Load(x1[slot], k), Load(x2[slot], k), Load(p[slot], k); };
            if (n) load(0, n);
            while (e < end) {
                const uint64_t next = e + n;
                const uint32_t nn = next < end ? TileLength(next, end) : 0, slot = j % 2;
                if (nn) load((j + 1) % 2, nn);
                AddInputs(x1[slot], x2[slot], n);
                Combine(p[slot], n);        // beta
                Compute(0, DaeIsa::Op(n));  // Muls
                Buffer& g = nn ? p[slot] : p[0];  // gamma: the lowest free buffer (the next beta holds the other)
                Load(g, n);
                Combine(g, n);
                Compute(0, DaeIsa::Op(n));  // Narrow into the X1 buffer
                Store(x1[slot], n);
                e = next, n = nn, ++j;
            }
        }
    };

    static inline RowTimeline ColumnTimeline(const TilingConfig& cfg, uint32_t M, uint32_t D, uint32_t s, const HardwareModel& hw,
                                             double bound = std::numeric_limits<double>::infinity()) {
        const bool split = cfg.mode == TilingMode::SPLIT_D;
        const uint32_t nb = cfg.blocks, q = QuantumElems(s, hw);
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
            c.D = D, c.s = s, c.q = q, c.tile = cfg.tileElems, c.bw = hw.BytesPerCycle(), c.L = hw.LatencyCycles();
            uint64_t used = 0;
            for (uint32_t k = 0; k < r.nFrag; ++k) {
                const CoreRange::Fragment& f = r.frag[k];
                const uint64_t len = f.ce - f.cb;
                fragResident[t][k] = cfg.zResident && used + len <= cfg.zResident;
                if (fragResident[t][k]) used += len;
                c.Sweep1(uint64_t(f.row) * D + f.cb, uint64_t(f.row) * D + f.ce);
            }
            if (split) {  // The core's two records, zeroed and filled, go out
                c.Compute(0, DaeIsa::Fill(16));
                const double start = std::max(c.ch, c.vec);
                c.ch = start + 64.0 / c.bw;
                c.dma += 64.0 / c.bw;
                c.stores = std::max(c.stores, c.ch + c.L);
            }
            for (uint32_t row = r.rowA; row < r.rowZ; ++row) {
                const bool fits = cfg.zResident && used + D <= cfg.zResident;
                c.Sweep1(uint64_t(row) * D, uint64_t(row + 1) * D);
                if (fits) c.Sweep2Resident(uint64_t(row) * D, uint64_t(row + 1) * D, false);
                else c.Sweep2Recompute(uint64_t(row) * D, uint64_t(row + 1) * D);
                if (std::max(c.vec, c.ch) >= bound) return out.finish = std::numeric_limits<double>::infinity(), out;
            }
            if (split && r.nFrag > 0 && fragResident[t][0]) {  // The first gamma chunk streams in during SyncAll
                const uint64_t start = uint64_t(r.frag[0].row) * D + r.frag[0].cb;
                c.Load(c.p[0], c.TileLength(start, start + (r.frag[0].ce - r.frag[0].cb)));
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
                    recsRead = c.Compute(landed, DaeIsa::Reduce(count * 8ull));
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
        cfg.repRows = bp.rep;
        cfg.layout = BandLayout(bp.k, cfg.pitch, M, s, bp.rep, cfg.layout.misc, bp.depth);
        cfg.modelCycles = static_cast<uint64_t>(t.vector + 0.5);
        cfg.modelNs = t.finish / hw.clockGHz;
    }

private:
    // -------------------------------------------------------------------------
    // (3) DAE tiles [ARCH CHALLENGE 2/3, pipeline bubbles]. Every candidate schedule is replayed
    // on the timeline models above (RowTilesTimeline, BandTimeline, ColumnTimeline), which equal
    // the runtime's timeline cycle for cycle, and the one whose slowest core finishes first wins.
    // The 191 KB layout bounds each candidate (the Challenge 2 knapsack):
    //   row tiles     depth x 2 x B D s bytes of X1/X2 tiles, 4 B D more for the FP32 Z tile at
    //                 16 bits, the replicated gamma/beta and the 8 KB scratch chunk
    //   column band   k rows at the band pitch, the band's FP32 Z resident across the barrier
    //   column tiles  (6s + 4) bytes per element, plus the resident FP32 Z
    // modelNs is the slowest core's finish time; modelCycles its vector cycles.
    // -------------------------------------------------------------------------
    // `bound`: the modeled cycles a plan has to beat (a plan that cannot keeps modelNs = infinity)
    static inline void PlanTiles(TilingConfig& cfg, uint32_t M, uint32_t D, uint32_t s, const HardwareModel& hw, double bound) {
        cfg.modelNs = std::numeric_limits<double>::infinity();
        if (static_cast<uint64_t>(M) * D == 0) { cfg.modelNs = 0.0; return; }
        if (cfg.mode == TilingMode::SPLIT_COLUMNS) return PlanBand(cfg, M, D, s, hw, bound);
        if (cfg.mode == TilingMode::ROW_PARALLEL && PlanRowTiles(cfg, M, D, s, hw, bound)) return;
        PlanColumnTiles(cfg, M, D, s, hw, bound);
    }

    // Row tiles: whole rows per tile, gamma/beta resident and replicated `rep` rows. The plan is
    // the one whose modeled timeline (RowTilesTimeline) finishes first, over
    //   depth         2-4 tiles in flight per input queue: deeper queues hide DMA latency, at
    //                 the price of smaller tiles in the same 191 KB
    //   rep           replicated gamma/beta rows (fewer vector instructions, more scratchpad)
    //   B             body tile rows, up to the largest the layout admits (Challenge 2 knapsack)
    //   head, tail    a small first tile starts the vector unit sooner (prologue fill), a small
    //                 last tile finishes the final store sooner (epilogue drain)
    //   issue order   gamma/beta before tile 0; X2 of a later tile as soon as its buffer frees
    // Every busy core holds the same rows up to one unit, so the busiest one is
    // ceil(MaxLoad / D) rows.
    static inline bool PlanRowTiles(TilingConfig& cfg, uint32_t M, uint32_t D, uint32_t s, const HardwareModel& hw, double bound) {
        const uint64_t total = static_cast<uint64_t>(M) * D;
        const uint64_t rows = (MaxLoad(total, cfg.unitElems, cfg.blocks) + D - 1) / D;
        const uint32_t p = static_cast<uint32_t>(RowUnit(D, s, hw) / D);
        const uint32_t repMax = D % 8 == 0 ? std::max<uint32_t>(1, REP_FLOATS / D) : 1;  // 32-byte FP32 rows
        const uint32_t most = static_cast<uint32_t>((rows + p - 1) / p * p);             // Rows worth one tile
        // Stage 1: depth, replication and body size, plain issue order; the best few go on
        constexpr uint32_t KEEP = 8;
        struct Found { double finish; RowPipe pp; } top[KEEP];
        for (Found& f : top) f.finish = std::numeric_limits<double>::infinity();
        bool fits = false;
        for (uint32_t depth = 2; depth <= 4; ++depth) {
            for (const uint32_t repChoice : {repMax, 1u}) {
                uint32_t cap = 0;  // The largest body tile the layout admits (binary search over row units)
                for (uint32_t lo = 1, hi = most / p; lo <= hi;) {
                    const uint32_t mid = lo + (hi - lo) / 2, B = mid * p;
                    if (RowLayout(B, D, s, std::min(repChoice, B), depth).Total() <= hw.spmBytes) cap = B, lo = mid + 1;
                    else hi = mid - 1;
                }
                if (cap) fits = true;
                for (uint32_t B = p; cap && B <= cap; B = B < 48 * p ? B + p : std::max(B + p, std::min(cap, B * 9 / 8 / p * p))) {
                    const RowPipe pp{B, 0, 0, std::min(repChoice, B), depth, false, false};
                    const double f = RowTilesTimeline(rows, D, s, pp, hw, std::min(bound, top[KEEP - 1].finish) * 1.05).finish;
                    for (uint32_t i = 0; i < KEEP; ++i) {
                        if (f < top[i].finish) {
                            for (uint32_t j = KEEP - 1; j > i; --j) top[j] = top[j - 1];
                            top[i] = {f, pp};
                            break;
                        }
                    }
                    if (B == cap) break;
                }
                if (repMax == 1) break;  // One replication choice only
            }
        }
        if (!fits) return false;  // Not even one row unit fits: column tiles
        // Stage 2: head and tail tiles and issue order around the best few
        RowPipe best{};
        double bestTime = bound;
        for (const Found& f : top) {
            if (!std::isfinite(f.finish)) continue;
            for (const uint32_t head : {0u, p, 2 * p, 4 * p, 8 * p}) {
                if (head && (head >= f.pp.B || head >= rows)) continue;
                for (const uint32_t tail : {0u, p, 2 * p, 4 * p, 8 * p}) {
                    if (tail && (tail >= f.pp.B || tail >= rows)) continue;
                    for (const bool first : {false, true}) {
                        for (const bool early : {false, true}) {
                            RowPipe pp = f.pp;
                            pp.head = head, pp.tail = tail, pp.paramsFirst = first, pp.earlyX2 = early;
                            const double t = RowTilesTimeline(rows, D, s, pp, hw, bestTime).finish;
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
            pp.B = static_cast<uint32_t>(static_cast<int>(best.B) + step * static_cast<int>(p));
            pp.rep = std::min(best.rep, pp.B);
            if (!pp.B || pp.head >= pp.B || pp.tail >= pp.B || RowLayout(pp.B, D, s, pp.rep, pp.depth).Total() > hw.spmBytes) continue;
            const double t = RowTilesTimeline(rows, D, s, pp, hw, bestTime).finish;
            if (t < bestTime) bestTime = t, best = pp;
        }
        if (!best.B) {  // Row tiles fit but cannot beat the bound
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
        if (widest == 0 || widest > TMP_FLOATS) return;
        const uint32_t R = RecordFloats(cfg, M);
        const uint32_t misc = Align32((cfg.blocks + 1ull) * R * 4);
        if (BandLayout(1, widest, M, s, 1, misc).Total() > hw.spmBytes) return;
        cfg.pitch = widest;
        cfg.zResident = M * widest;
        cfg.layout = BandLayout(1, widest, M, s, 1, misc);
        // The band tile whose modeled timeline (BandTimeline) finishes first: k rows per tile
        // (every count up to 16, then steps of ~12%), replicated parameter rows, tiles in flight
        BandPipe best{1, 1, 2};
        double bestTime = bound;
        for (uint32_t depth = 2; depth <= 4; ++depth) {
            for (uint32_t k = 1; k <= M; k = k < 16 ? k + 1 : std::max(k + 1, k * 9 / 8)) {
                const uint32_t repMax = std::max(1u, std::min(k, REP_FLOATS / widest));
                for (uint32_t rep = 1;; rep = std::min(repMax, rep * 2)) {
                    if (BandLayout(k, widest, M, s, rep, misc, depth).Total() <= hw.spmBytes) {
                        const double t = BandTimeline(cfg, M, D, s, widest, {k, rep, depth}, hw, bestTime).finish;
                        if (t < bestTime) bestTime = t, best = {k, rep, depth};
                    }
                    if (rep == repMax) break;
                }
                if (k == M) break;
                if (k >= 16 && k * 9 / 8 > M && k < M) k = M - 1;  // Always try one tile of every row
            }
        }
        if (bestTime >= bound) return;  // Infeasible, or no better than the bound: modelNs stays infinite
        ApplyBandPipe(cfg, M, D, s, hw, best);
    }

    // Column tiles: Split-D fragments, or rows too long for a row tile. Keep the segment's
    // FP32 Z resident when a useful tile still fits (sweep 2 then re-reads nothing but gamma).
    static inline void PlanColumnTiles(TilingConfig& cfg, uint32_t M, uint32_t D, uint32_t s, const HardwareModel& hw, double bound) {
        const uint64_t total = static_cast<uint64_t>(M) * D;
        const bool split = cfg.mode == TilingMode::SPLIT_D;
        const uint64_t L = MaxLoad(total, cfg.unitElems, cfg.blocks);
        const uint32_t q = QuantumElems(s, hw);
        const uint32_t misc = split ? Align32((16ull + 16ull * cfg.blocks) * 4) : 0u;
        cfg.tileRows = 0;
        cfg.pitch = D;
        cfg.repRows = 0;
        // The tile (and resident Z) whose modeled timeline (ColumnTimeline) finishes first;
        // largest tiles first, so the best time so far cuts the others short
        TilingConfig best = cfg;
        RowTimeline bestTime{bound, 0.0, 0.0};
        const uint64_t longest = (std::min<uint64_t>(D, split ? L : D) + q - 1) / q * q;
        for (const uint64_t resident : {split ? std::min<uint64_t>(L, 3ull * D) : uint64_t(D), uint64_t(0)}) {
            uint64_t cap = 0;
            for (uint64_t lo = 1, hi = longest / q; lo <= hi;) {
                const uint64_t mid = lo + (hi - lo) / 2;
                if (ColumnLayout(mid * q, resident, s, misc).Total() <= hw.spmBytes) cap = mid * q, lo = mid + 1;
                else hi = mid - 1;
            }
            if (!cap) continue;
            for (uint64_t tile = cap;; tile = std::max<uint64_t>(q, tile * 4 / 5 / q * q)) {
                TilingConfig c = cfg;
                c.tileElems = static_cast<uint32_t>(tile);
                c.zResident = static_cast<uint32_t>(resident);
                c.layout = ColumnLayout(tile, resident, s, misc);
                const RowTimeline t = ColumnTimeline(c, M, D, s, hw, bestTime.finish);
                if (t.finish < bestTime.finish) best = c, bestTime = t;
                if (tile == q) break;
            }
            if (!resident) break;
        }
        if (bestTime.finish >= bound) {  // No better than the bound
            cfg.tileElems = best.tileElems ? best.tileElems : q;
            cfg.layout = ColumnLayout(cfg.tileElems, 0, s, misc);
            cfg.modelNs = std::numeric_limits<double>::infinity();
            return;
        }
        cfg = best;
        cfg.modelCycles = static_cast<uint64_t>(bestTime.vector + 0.5);
        cfg.modelNs = bestTime.finish / hw.clockGHz;
    }
};

} // namespace hpc
