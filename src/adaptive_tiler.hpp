#pragma once

#include "dsa_runtime.hpp"
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <numeric>
#if __has_include(<unistd.h>)
#include <unistd.h>
#endif

namespace hpc {

enum class TilingMode {
    ROW_PARALLEL, // Units of whole, DMA-aligned rows (zero cross-core communication)
    SPLIT_D       // Units of one 32-byte DMA block: balance to one block + one SyncAll
};

// -----------------------------------------------------------------------------
// Machine description. Every number the planning equations use comes from here:
//   Target()  the deployment machine: 40 symmetric DAE cores, 32-byte DMA blocks,
//             191 KB scratchpad per core (include/dsa_runtime.hpp), ~0 fork/join.
//   Host(P)   the same machine laws run by P CI threads with the host's measured costs.
// -----------------------------------------------------------------------------
struct HardwareModel {
    uint32_t cores;         // Symmetric cores
    uint32_t quantumBytes;  // DMA block: split boundaries and block transfers are multiples of it
    size_t   spmBytes;      // Per-core scratchpad budget
    double   launchNs;      // Engaging all cores (fork/join)
    double   syncNs;        // One all-core barrier (SyncAll)
    double   elemNs;        // Vector work per element on one core
    double   byteNs;        // Streaming cost per byte on one core (1 / per-core bandwidth)
    double   tileNs;        // Fixed latency of one DMA tile: the pipeline fill/drain unit
    uint32_t batchElems;    // Host: rows whose pass-1 Z stays L1-resident together

    static HardwareModel Target() {
        return {dsa::MAX_HARDWARE_CORES, dsa::DMA_ALIGN_BYTES, dsa::SCRATCHPAD_SAFE_WATERLINE,
                0.0,           // Zero fork/join cost
                2000.0,        // SyncAll, inferred from the P05 target (docs)
                0.0,           // 256-byte vector repeats hide under DMA
                40.0 / 850.0,  // ~850 GB/s aggregate shared by 40 cores
                800.0,         // DMA tile latency, inferred from the P01-P03 targets (docs)
                0};
    }

    static HardwareModel Host(uint32_t threads) {
        HardwareModel hw = Target();
        hw.cores = std::max(1u, std::min(threads, hw.cores));
        hw.launchNs = 3500.0;  // Measured on the 4-core CI VM (docs)
        hw.syncNs = 1000.0;
        hw.elemNs = 0.3;       // Cache-resident rows are compute-bound: any dtype
        hw.byteNs = 0.0;
        hw.tileNs = 0.0;       // Hardware prefetchers, no explicit DMA tiles
        hw.batchElems = 2048;
        return hw;
    }
};

// Per-core scratchpad layout of the DAE pipeline. The kernel claims exactly these
// buffers through TPipe, so the planner's total is what the sanitizer checks.
struct DaeLayout {
    uint32_t tile;       // One X1 / X2 / parameter-chunk buffer (native dtype)
    uint32_t z;          // FP32 Z of one tile (row tiles in FP32 work in place)
    uint32_t tmp;        // FP32 scratch: casts, squares, folds
    uint32_t params;     // Resident FP32 gamma + bias (row tiles)
    uint32_t resident;   // Resident FP32 Z of column-tiled segments
    uint32_t misc;       // Split-D partial-sum records
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
    uint32_t tileRows;       // Rows per tile when whole rows fit a tile (0: column tiles)
    uint32_t tileElems;      // Elements per X1/X2 tile
    uint32_t zResident;      // FP32 Z elements kept resident for column-tiled segments
    DaeLayout layout;
};

// A core's share of the flattened M*D stream: units [floor(U t/n), floor(U (t+1)/n)), i.e.
// complete rows [rowA, rowZ) plus at most two fragments shared with neighbouring cores.
struct CoreRange {
    struct Fragment { uint32_t row, cb, ce; };
    Fragment frag[2];
    uint32_t nFrag = 0, rowA = 0, rowZ = 0;
};

class AdaptiveTiler {
public:
    static constexpr uint32_t MAX_THREADS = dsa::MAX_HARDWARE_CORES;

    static inline CoreRange Range(const TilingConfig& cfg, uint32_t M, uint32_t D, uint32_t t, uint32_t n) {
        CoreRange r;
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
    static constexpr uint32_t MAX_BATCH = 64;
    static constexpr uint32_t TMP_BYTES = 8192;          // DAE FP32 scratch chunk
    static constexpr uint32_t MISC_BYTES = 2048;         // DAE partial records (40 x 32 B + publish)
    static constexpr double CHUNK_BYTES = 64 * 1024;     // Host serpentine chunk (prefetch-friendly run)

    static inline TilingConfig Plan(uint32_t M, uint32_t D, uint32_t elemBytes, const HardwareModel& hw,
                                    size_t llcBytes = 0) {
        const uint64_t total = static_cast<uint64_t>(M) * D;
        const uint32_t P = std::max(1u, hw.cores);
        const double c = hw.elemNs + 3.0 * elemBytes * hw.byteNs;  // X1 + X2 in, Y out
        // ---------------------------------------------------------------------
        // (1) Decomposition. Every candidate is a balanced split of the flattened M*D stream
        // into units; its cost is the busiest core's work (closed form):
        //   inline  one core, no launch
        //   rows    units of p rows, p = 32 / gcd(32, D*s): whole rows AND whole DMA blocks
        //   split   units of one 32-byte block [ARCH CHALLENGE 1]: every core within one block
        //           of the mean for any M and any core count, plus one SyncAll
        // ---------------------------------------------------------------------
        const double tInline = total * c;
        const double tRows = hw.launchNs + MaxLoad(total, RowUnit(D, elemBytes, hw), P) * c;
        const double tSplit = hw.launchNs + hw.syncNs + MaxLoad(total, QuantumElems(elemBytes, hw), P) * c;
        const bool parallel = P > 1 && std::min(tRows, tSplit) < tInline;
        const TilingMode mode = parallel && tSplit < tRows ? TilingMode::SPLIT_D : TilingMode::ROW_PARALLEL;
        return Build(M, D, elemBytes, hw, mode, parallel, llcBytes);
    }

    // Completes a plan for a given decomposition (tests force every mode through here).
    static inline TilingConfig Build(uint32_t M, uint32_t D, uint32_t elemBytes, const HardwareModel& hw,
                                     TilingMode mode, bool parallel, size_t llcBytes = 0) {
        TilingConfig cfg{};
        const uint64_t total = static_cast<uint64_t>(M) * D;
        const uint32_t P = std::max(1u, hw.cores);
        cfg.mode = mode;
        cfg.unitElems = mode == TilingMode::SPLIT_D ? QuantumElems(elemBytes, hw) : RowUnit(D, elemBytes, hw);
        cfg.units = (total + cfg.unitElems - 1) / cfg.unitElems;
        cfg.blocks = parallel ? static_cast<uint32_t>(std::min<uint64_t>(P, std::max<uint64_t>(1, cfg.units))) : 1;
        cfg.threads = parallel ? P : 1;

        // (2) Host SIMD executor: L1-resident batches, resident Z, streaming stores past the LLC.
        cfg.batchRows = hw.batchElems ? std::max(1u, std::min(MAX_BATCH, hw.batchElems / std::max(1u, D))) : 1;
        cfg.chunkRows = std::max(1u, static_cast<uint32_t>(CHUNK_BYTES / std::max(1.0, 3.0 * D * elemBytes)));
        cfg.residentElems = static_cast<uint32_t>(hw.spmBytes / sizeof(float));
        cfg.streamStores = llcBytes && 3.0 * total * elemBytes > static_cast<double>(llcBytes);
        cfg.serpentine = true;

        if (hw.tileNs > 0) PlanTiles(cfg, M, D, elemBytes, hw);  // Explicit DMA pipeline only
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

    static inline DaeLayout RowLayout(uint32_t rows, uint32_t D, uint32_t s) {
        const uint64_t e = static_cast<uint64_t>(rows) * D;
        return {Align32(e * s), s < 4 ? Align32(e * 4) : 0u, TMP_BYTES, Align32(8ull * D), 0u, MISC_BYTES, false};
    }

    static inline DaeLayout ColumnLayout(uint32_t tile, uint64_t resident, uint32_t s) {
        return {Align32(static_cast<uint64_t>(tile) * s), Align32(tile * 4ull), TMP_BYTES, 0u, Align32(resident * 4), MISC_BYTES, true};
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

private:
    // -------------------------------------------------------------------------
    // (3) DAE pipeline tiles [ARCH CHALLENGE 2/3]. n tiles through the 3-stage pipeline
    // (DMA in -> vector -> DMA out, double-buffered), each stage paying tileNs per tile:
    //   T(n) = (n + 2)(W/n + tileNs)  =>  n* = sqrt(2 W / tileNs)
    // with W the busiest core's streaming time. The largest tile the 191 KB layout admits
    // caps it (the Challenge 2 knapsack):
    //   row tiles     B*(D) = floor((SPM - fixed - 8D) / (b D)) rows, b = 4s + 4 (12 B/elem at 16 bits;
    //                 FP32 computes in place: 16)
    //   column tiles  W*    = floor((SPM - fixed - 4 Z_resident) / (6s + 4)) elements
    // -------------------------------------------------------------------------
    static inline void PlanTiles(TilingConfig& cfg, uint32_t M, uint32_t D, uint32_t s, const HardwareModel& hw) {
        const uint64_t total = static_cast<uint64_t>(M) * D;
        if (total == 0) return;
        const double c = hw.elemNs + 3.0 * s * hw.byteNs;
        const uint64_t blockElems = MaxLoad(total, cfg.unitElems, cfg.blocks);
        const uint64_t n = hw.tileNs > 0 ? std::max<uint64_t>(1, std::llround(std::sqrt(2.0 * blockElems * c / hw.tileNs))) : 1;
        const uint64_t spm = hw.spmBytes, fixed = TMP_BYTES + MISC_BYTES;
        const uint32_t q = QuantumElems(s, hw);

        if (cfg.mode == TilingMode::ROW_PARALLEL) {
            const uint64_t p = RowUnit(D, s, hw) / D;
            const uint64_t perRow = static_cast<uint64_t>(D) * (4ull * s + (s < 4 ? 4 : 0));
            const uint64_t params = Align32(8ull * D);
            const uint64_t fit = spm > fixed + params ? (spm - fixed - params) / perRow / p * p : 0;
            const uint64_t rows = (blockElems + D - 1) / D;
            uint64_t B = std::min(fit, std::max(p, ((rows + n - 1) / n + p - 1) / p * p));
            while (B >= p && RowLayout(static_cast<uint32_t>(B), D, s).Total() > spm) B -= p;  // Block rounding
            if (B >= p) {
                cfg.tileRows = static_cast<uint32_t>(B);
                cfg.tileElems = static_cast<uint32_t>(B * D);
                cfg.layout = RowLayout(cfg.tileRows, D, s);
                return;
            }
        }
        // Column tiles: split fragments, or rows too long for a row tile. Keep the segment's
        // FP32 Z resident when a useful tile still fits (sweep 2 then re-reads nothing).
        const uint64_t perElem = 6ull * s + 4;
        const uint64_t want = std::max<uint64_t>(q, ((blockElems + n - 1) / n + q - 1) / q * q);
        const uint64_t segment = cfg.mode == TilingMode::SPLIT_D ? std::min<uint64_t>(blockElems, 3ull * D) : D;
        uint64_t resident = segment;
        if (fixed + Align32(resident * 4) + std::min<uint64_t>(want, 16ull * q) * perElem > spm) resident = 0;
        uint64_t tile = std::max<uint64_t>(q, std::min(want, (spm - fixed - Align32(resident * 4)) / perElem / q * q));
        while (tile > q && ColumnLayout(static_cast<uint32_t>(tile), resident, s).Total() > spm) tile -= q;
        cfg.tileRows = 0;
        cfg.tileElems = static_cast<uint32_t>(tile);
        cfg.zResident = static_cast<uint32_t>(resident);
        cfg.layout = ColumnLayout(cfg.tileElems, resident, s);
    }
};

} // namespace hpc
