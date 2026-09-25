#pragma once

#include "dsa_runtime.hpp"
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
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
    static constexpr uint64_t kSyncAll = 7500;                                    // SyncAll
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
//             191 KB scratchpad per core, ~0 fork/join, the runtime's cycle costs.
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
    double   tileNs;        // Fixed latency of one DMA tile: the pipeline fill/drain unit
    double   clockGHz;      // Target: converts the runtime's vector cycles (DaeIsa) to ns; 0 on the host
    uint32_t batchElems;    // Host: rows whose pass-1 Z stays L1-resident together

    static HardwareModel Target() {
        HardwareModel hw{};
        hw.cores = dsa::MAX_HARDWARE_CORES;
        hw.quantumBytes = dsa::DMA_ALIGN_BYTES;
        hw.spmBytes = dsa::SCRATCHPAD_SAFE_WATERLINE;
        hw.launchNs = 0.0;                           // Zero fork/join cost
        hw.clockGHz = 1.5;                           // Assumed: the runtime counts cycles, not time (docs)
        hw.syncNs = DaeIsa::kSyncAll / hw.clockGHz;  // SyncAll: 7500 cycles
        hw.elemNs = 0.0;                             // Vector work comes from the DaeIsa cycle model
        hw.byteNs = 40.0 / 850.0;                    // ~850 GB/s aggregate shared by 40 cores
        hw.tileNs = 800.0;                           // DMA tile latency, inferred from the targets (docs)
        hw.batchElems = 0;
        return hw;
    }

    static HardwareModel Host(uint32_t threads) {
        HardwareModel hw = Target();
        hw.cores = std::max(1u, std::min(threads, hw.cores));
        hw.launchNs = 3500.0;  // Measured on the 4-core CI VM (docs)
        hw.syncNs = 1000.0;
        hw.elemNs = 0.3;       // Cache-resident rows are compute-bound: any dtype
        hw.byteNs = 0.0;
        hw.tileNs = 0.0;       // Hardware prefetchers, no explicit DMA tiles
        hw.clockGHz = 0.0;     // No DAE cycle model: SIMD work is elemNs
        hw.batchElems = 2048;
        return hw;
    }
};

// Per-core scratchpad layout of the DAE pipeline. The kernel claims exactly these
// buffers through TPipe, so the planner's total is what the sanitizer checks.
struct DaeLayout {
    uint32_t tile;       // One X1 / X2 / parameter-chunk buffer (native dtype)
    uint32_t z;          // FP32 Z of one tile (row tiles in FP32 work in place)
    uint32_t tmp;        // FP32 scratch: casts, squares, folds; staging of gamma/beta
    uint32_t params;     // Resident FP32 gamma + beta, each replicated repRows times
    uint32_t resident;   // Resident FP32 Z: column-tiled segments, or a core's whole column band
    uint32_t misc;       // Split-D partial-sum records (published + gathered)
    bool paramQueue;     // Column tiles stream bias/gamma chunks
    uint32_t Total() const { return 2 * tile * (paramQueue ? 3 : 2) + z + tmp + params + resident + misc; }
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
    static constexpr uint32_t MAX_THREADS = dsa::MAX_HARDWARE_CORES;
    static constexpr uint32_t MAX_BATCH = 64;
    static constexpr uint32_t TMP_BYTES = 8192;          // DAE FP32 scratch chunk
    static constexpr uint32_t TMP_FLOATS = TMP_BYTES / sizeof(float);
    static constexpr uint32_t REP_FLOATS = 2048;         // DAE replicated gamma (and beta): <= 8 KB each
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
    // Host cost: launch + barrier + elements * elemNs. Target cost: barrier + a 3-stage DMA
    // pipeline over max(vector cycles / clock, DMA bytes * byteNs) of the busiest core.
    // -------------------------------------------------------------------------
    static inline TilingConfig Plan(uint32_t M, uint32_t D, uint32_t elemBytes, const HardwareModel& hw,
                                    size_t llcBytes = 0) {
        TilingConfig best = Build(M, D, elemBytes, hw, TilingMode::ROW_PARALLEL, false, llcBytes);
        if (hw.cores < 2 || static_cast<uint64_t>(M) * D == 0) return best;
        for (const TilingMode mode : {TilingMode::ROW_PARALLEL, TilingMode::SPLIT_D, TilingMode::SPLIT_COLUMNS}) {
            if (mode == TilingMode::SPLIT_COLUMNS && hw.clockGHz <= 0) continue;  // DAE executor only
            const TilingConfig c = Build(M, D, elemBytes, hw, mode, true, llcBytes);
            if (c.modelNs < best.modelNs) best = c;  // Ties keep the simpler plan
        }
        return best;
    }

    // Completes a plan for a given decomposition (tests force every mode through here).
    static inline TilingConfig Build(uint32_t M, uint32_t D, uint32_t elemBytes, const HardwareModel& hw,
                                     TilingMode mode, bool parallel, size_t llcBytes = 0) {
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

        if (hw.tileNs > 0) {
            PlanTiles(cfg, M, D, elemBytes, hw);  // Explicit DMA pipeline: tiles, layout, modeled time
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

    static inline DaeLayout RowLayout(uint64_t rows, uint32_t D, uint32_t s, uint64_t rep) {
        const uint64_t e = rows * D;
        return {Align32(e * s), s < 4 ? Align32(e * 4) : 0u, TMP_BYTES, ParamBytes(rep, D), 0u, 0u, false};
    }

    static inline DaeLayout BandLayout(uint64_t rows, uint32_t pitch, uint32_t M, uint32_t s, uint64_t rep, uint32_t misc) {
        return {Align32(rows * pitch * s), 0u, TMP_BYTES, ParamBytes(rep, pitch), Align32(uint64_t(M) * pitch * 4), misc, false};
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
    // (0: empty row); null means k full rows of w.
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
            for (uint64_t r0 = 0; r0 < k; r0 += per) {
                const uint64_t m = std::min(per, k - r0);
                c += I::Op(m * w);
                if (!lens) c += I::ReduceRuns(m, w);
                else for (uint64_t i = r0; i < r0 + m; ++i) c += lens[i] ? I::Reduce(lens[i]) : 0;
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

    // Widening gamma and beta of w columns (FP32 parameters arrive by DMA only)
    static inline uint64_t ParamCycles(uint64_t w, uint32_t s) {
        if (s == 4) return 0;
        if (2ull * Align32(w * s) <= TMP_BYTES) return 2 * DaeIsa::Op(w);
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

private:
    // -------------------------------------------------------------------------
    // (3) DAE tiles [ARCH CHALLENGE 2/3]. n tiles through the 3-stage pipeline (DMA in ->
    // vector -> DMA out, double-buffered), each stage paying tileNs per tile:
    //   T(n) = (n + 2)(W/n + tileNs)  =>  n* = sqrt(2 W / tileNs)
    // with W = max(vector, DMA) of the busiest core. The largest tile the 191 KB layout admits
    // caps it (the Challenge 2 knapsack):
    //   row tiles     B*(D) = floor((SPM - fixed - 8 rep D) / (b D)) rows, b = 4s + 4 (12 B/elem
    //                 at 16 bits; FP32 computes in place: 16)
    //   column band   k rows at the band pitch, the band's FP32 Z resident across the barrier
    //   column tiles  W* = floor((SPM - fixed - 4 Z_resident) / (6s + 4)) elements
    // The modeled time is the slowest core's; modelCycles its vector cycles.
    // -------------------------------------------------------------------------
    static inline void PlanTiles(TilingConfig& cfg, uint32_t M, uint32_t D, uint32_t s, const HardwareModel& hw) {
        cfg.modelNs = std::numeric_limits<double>::infinity();
        if (static_cast<uint64_t>(M) * D == 0) { cfg.modelNs = 0.0; return; }
        if (cfg.mode == TilingMode::SPLIT_COLUMNS) return PlanBand(cfg, M, D, s, hw);
        if (cfg.mode == TilingMode::ROW_PARALLEL && PlanRowTiles(cfg, M, D, s, hw)) return;
        PlanColumnTiles(cfg, M, D, s, hw);
    }

    // One core: barrier + pipeline over n tiles of W = max(vector, DMA)
    static inline double PipelineNs(const HardwareModel& hw, uint64_t cycles, double dmaBytes, uint64_t tiles, bool sync) {
        const double W = std::max(cycles / hw.clockGHz, dmaBytes * hw.byteNs);
        const double n = static_cast<double>(std::max<uint64_t>(1, tiles));
        return (sync ? hw.syncNs : 0.0) + (n + 2.0) * (W / n + hw.tileNs);
    }

    static inline uint64_t TilesFor(const HardwareModel& hw, uint64_t cycles, double dmaBytes) {
        const double W = std::max(cycles / hw.clockGHz, dmaBytes * hw.byteNs);
        return std::max<uint64_t>(1, static_cast<uint64_t>(std::llround(std::sqrt(2.0 * W / hw.tileNs))));
    }

    // Row tiles: B rows per double-buffered tile, gamma/beta resident and replicated `rep` rows.
    // Replication trades scratchpad for fewer vector instructions (bias and gamma once per rep
    // rows); both options are planned and the modeled time picks. Every busy core holds the
    // same rows up to one unit, so the busiest one is ceil(MaxLoad / D) rows.
    static inline bool PlanRowTiles(TilingConfig& cfg, uint32_t M, uint32_t D, uint32_t s, const HardwareModel& hw) {
        const uint64_t total = static_cast<uint64_t>(M) * D;
        const uint64_t rows = (MaxLoad(total, cfg.unitElems, cfg.blocks) + D - 1) / D;
        const uint64_t p = RowUnit(D, s, hw) / D;
        const uint64_t repMax = D % 8 == 0 ? std::max<uint64_t>(1, REP_FLOATS / D) : 1;  // 32-byte FP32 rows
        const uint64_t perRow = static_cast<uint64_t>(D) * (4ull * s + (s < 4 ? 4 : 0));
        bool found = false;
        TilingConfig best = cfg;
        for (const uint64_t rep : {repMax, uint64_t(1)}) {
            if (found && rep == repMax) continue;
            const uint64_t fixed = TMP_BYTES + ParamBytes(rep, D);
            uint64_t cap = hw.spmBytes > fixed ? (hw.spmBytes - fixed) / perRow / p * p : 0;
            while (cap >= p && RowLayout(cap, D, s, std::min(rep, cap)).Total() > hw.spmBytes) cap -= p;  // Block rounding
            if (cap < p) continue;
            auto cycles = [&](uint64_t B) {
                const uint64_t r = std::min(rep, B);
                return ParamCycles(D, s) + rows / B * TileCycles(B, D, s, r, nullptr) +
                       (rows % B ? TileCycles(rows % B, D, s, r, nullptr) : 0);
            };
            const double dma = 3.0 * s * rows * D + 2.0 * s * D;  // X1 + X2 + Y, gamma/beta once
            const uint64_t n = TilesFor(hw, cycles(std::min(cap, rows)), dma);
            const uint64_t B = std::min(cap, std::max(p, ((rows + n - 1) / n + p - 1) / p * p));
            TilingConfig c = cfg;
            c.tileRows = static_cast<uint32_t>(B);
            c.tileElems = static_cast<uint32_t>(B * D);
            c.pitch = D;
            c.repRows = static_cast<uint32_t>(std::min(rep, B));
            c.zResident = 0;
            c.layout = RowLayout(B, D, s, c.repRows);
            c.modelCycles = cycles(B);
            c.modelNs = PipelineNs(hw, c.modelCycles, dma, (rows + B - 1) / B, false);
            if (!found || c.modelNs < best.modelNs) best = c;
            found = true;
        }
        if (found) cfg = best;
        return found;
    }

    // Column band (SPLIT_COLUMNS): each core keeps the FP32 Z of its band of all M rows across
    // the barrier; tiles of k band rows stream X1/X2 by one DMA per row, gamma/beta come once.
    // Needs rows on the 32-byte grid (every row DMA is whole blocks), band rows that fit the
    // scratch chunk and the band's Z in the scratchpad; otherwise the plan is infeasible and
    // Plan() keeps another decomposition.
    static inline void PlanBand(TilingConfig& cfg, uint32_t M, uint32_t D, uint32_t s, const HardwareModel& hw) {
        const uint32_t q = QuantumElems(s, hw);
        if (D % q != 0) return;
        uint32_t widest = 0;  // Band pitch: the widest band of any core, in elements
        for (uint32_t t = 0; t < cfg.blocks; ++t) {
            const CoreRange r = Range(cfg, M, D, t, cfg.blocks);
            if (r.rowZ > r.rowA) widest = std::max(widest, BandEnd(r, q) - BandBegin(r, q));
        }
        if (widest == 0 || widest > TMP_FLOATS) return;
        const uint32_t R = RecordFloats(cfg, M);
        const uint32_t misc = Align32((cfg.blocks + 1ull) * R * 4);
        if (BandLayout(1, widest, M, s, 1, misc).Total() > hw.spmBytes) return;
        // Slowest core for tiles of k rows with rep-row parameters (bands differ by a block)
        std::vector<uint32_t> lens(M);
        auto cost = [&](uint64_t k, uint64_t rep, uint64_t& cycles, double& dmaMax) {
            double worst = 0.0;
            cycles = 0;
            dmaMax = 0.0;
            for (uint32_t t = 0; t < cfg.blocks; ++t) {
                const CoreRange r = Range(cfg, M, D, t, cfg.blocks);
                if (r.rowZ <= r.rowA) continue;
                const uint32_t w = BandEnd(r, q) - BandBegin(r, q);
                uint64_t elems = 0;
                for (uint32_t i = 0; i < M; ++i) {
                    uint32_t cb, ce;
                    BandRow(r, i, q, cb, ce);
                    lens[i] = cb < ce ? ce - cb : 0;
                    elems += lens[i];
                }
                uint64_t c = ParamCycles(w, s) + DaeIsa::Fill(R) + M * DaeIsa::Reduce(1);
                for (uint64_t n = cfg.blocks; n > 1; n -= n / 2) c += DaeIsa::Op(n / 2 * R);  // Gather tree
                for (uint64_t row = 0; row < M; row += k) c += TileCycles(std::min<uint64_t>(k, M - row), widest, s, rep, lens.data() + row);
                const double dma = 3.0 * s * elems + 2.0 * s * w + 4.0 * R * (cfg.blocks + 1);
                worst = std::max(worst, PipelineNs(hw, c, dma, (M + k - 1) / k, true));
                cycles = std::max(cycles, c);
                dmaMax = std::max(dmaMax, dma);
            }
            return worst;
        };
        uint64_t cycles = 0;
        double dma = 0.0;
        cost(M, 1, cycles, dma);
        uint64_t k = (M + TilesFor(hw, cycles, dma) - 1) / TilesFor(hw, cycles, dma);
        uint64_t rep = std::max<uint64_t>(1, std::min<uint64_t>(k, REP_FLOATS / widest));
        while (BandLayout(k, widest, M, s, rep, misc).Total() > hw.spmBytes) {
            if (rep > 1) --rep;
            else --k;  // k = 1, rep = 1 fits (checked above)
        }
        cfg.tileRows = static_cast<uint32_t>(k);
        cfg.tileElems = static_cast<uint32_t>(k * widest);
        cfg.pitch = widest;
        cfg.repRows = static_cast<uint32_t>(rep);
        cfg.zResident = M * widest;
        cfg.layout = BandLayout(k, widest, M, s, rep, misc);
        cfg.modelNs = cost(k, rep, cfg.modelCycles, dma);
    }

    // Column tiles: Split-D fragments, or rows too long for a row tile. Keep the segment's
    // FP32 Z resident when a useful tile still fits (sweep 2 then re-reads nothing but gamma).
    static inline void PlanColumnTiles(TilingConfig& cfg, uint32_t M, uint32_t D, uint32_t s, const HardwareModel& hw) {
        const uint64_t total = static_cast<uint64_t>(M) * D;
        const bool split = cfg.mode == TilingMode::SPLIT_D;
        const uint64_t L = MaxLoad(total, cfg.unitElems, cfg.blocks);
        const uint32_t q = QuantumElems(s, hw);
        const uint64_t n = std::max<uint64_t>(1, std::llround(std::sqrt(2.0 * L * 5.0 * s * hw.byteNs / hw.tileNs)));
        const uint64_t spm = hw.spmBytes;
        const uint32_t misc = split ? Align32((16ull + 16ull * cfg.blocks) * 4) : 0u;
        const uint64_t fixed = TMP_BYTES + misc;
        const uint64_t perElem = 6ull * s + 4;
        const uint64_t want = std::max<uint64_t>(q, ((L + n - 1) / n + q - 1) / q * q);
        uint64_t resident = split ? std::min<uint64_t>(L, 3ull * D) : D;
        if (fixed + Align32(resident * 4) + std::min<uint64_t>(want, 16ull * q) * perElem > spm) resident = 0;
        uint64_t tile = std::max<uint64_t>(q, std::min(want, (spm - fixed - Align32(resident * 4)) / perElem / q * q));
        while (tile > q && ColumnLayout(tile, resident, s, misc).Total() > spm) tile -= q;
        cfg.tileRows = 0;
        cfg.tileElems = static_cast<uint32_t>(tile);
        cfg.pitch = D;
        cfg.repRows = 0;
        cfg.zResident = static_cast<uint32_t>(resident);
        cfg.layout = ColumnLayout(tile, resident, s, misc);

        // Every core: its fragments and complete rows (complete rows cost the same up to where
        // they start on the 32-byte grid, so their cost is cached by that phase)
        struct Seg { uint64_t cycles = 0, tiles = 0; bool done = false; } rowCost[2][32];
        double worst = 0.0;
        cfg.modelCycles = 0;
        for (uint32_t t = 0; t < cfg.blocks; ++t) {
            const CoreRange r = Range(cfg, M, D, t, cfg.blocks);
            uint64_t cycles = split ? DaeIsa::Fill(16) : 0, tiles = 0, used = 0;
            double dma = split ? 64.0 : 0.0;
            auto segment = [&](uint64_t start, uint64_t end, bool fits, Seg* memo) {
                if (!memo || !memo->done) {
                    Seg cost;
                    cost.cycles = SegmentCycles(start, end, tile, q, s, fits);
                    for (uint64_t e = start; e < end; e = NextTile(e, end, tile, q)) ++cost.tiles;
                    cost.done = true;
                    if (!memo) { cycles += cost.cycles; tiles += cost.tiles; dma += (fits ? 5.0 : 8.0) * s * (end - start); return; }
                    *memo = cost;
                }
                cycles += memo->cycles;
                tiles += memo->tiles;
                dma += (fits ? 5.0 : 8.0) * s * (end - start);  // X1, X2, Y, beta, gamma (+ X1, X2, beta again)
            };
            for (uint32_t k = 0; k < r.nFrag; ++k) {
                const CoreRange::Fragment& f = r.frag[k];
                const bool fits = resident && used + (f.ce - f.cb) <= resident;
                used += fits ? f.ce - f.cb : 0;
                segment(static_cast<uint64_t>(f.row) * D + f.cb, static_cast<uint64_t>(f.row) * D + f.ce, fits, nullptr);
                if (split) {  // Gather of the row's records
                    uint32_t first, last;
                    RowOwners(cfg, D, f.row, cfg.blocks, first, last);
                    const uint64_t count = 2ull * last - (2ull * first + std::max(1u, Range(cfg, M, D, first, cfg.blocks).nFrag) - 1) + 1;
                    cycles += DaeIsa::Reduce(8 * count);
                    dma += 32.0 * count;
                }
            }
            // Row starts repeat on the grid with period q / gcd(D, q): one period, then multiples
            const bool fits = resident && used + D <= resident;
            const uint32_t rows = r.rowZ - r.rowA, period = q / std::gcd(D, q);
            const uint64_t before = cycles, tilesBefore = tiles;
            const double dmaBefore = dma;
            for (uint32_t i = 0; i < std::min(rows, period); ++i) {
                const uint64_t start = static_cast<uint64_t>(r.rowA + i) * D;
                segment(start, start + D, fits, q <= 32 ? &rowCost[fits][start % q] : nullptr);
            }
            if (rows > period) {
                const uint64_t times = rows / period;
                cycles = before + (cycles - before) * times;
                tiles = tilesBefore + (tiles - tilesBefore) * times;
                dma = dmaBefore + (dma - dmaBefore) * times;
                for (uint32_t i = 0; i < rows % period; ++i) {
                    const uint64_t start = static_cast<uint64_t>(r.rowA + times * period + i) * D;
                    segment(start, start + D, fits, q <= 32 ? &rowCost[fits][start % q] : nullptr);
                }
            }
            worst = std::max(worst, PipelineNs(hw, cycles, dma, tiles, split));
            cfg.modelCycles = std::max(cfg.modelCycles, cycles);
        }
        cfg.modelNs = worst;
    }
};

} // namespace hpc
