#pragma once

#include <cstdint>
#include <cstddef>
#include <algorithm>
#if __has_include(<unistd.h>)
#include <unistd.h>
#endif

namespace hpc {

enum class TilingMode {
    ROW_PARALLEL, // Whole rows per thread (zero cross-thread communication)
    SPLIT_D       // Rows cut into 64-byte lines: exact balance + one barrier for the row reduction
};

struct TilingConfig {
    TilingMode mode;
    uint32_t threads;        // OpenMP team: 1 (inline, no fork/join) or the full pool P
    uint32_t unitElems;      // Work quantum: one row (ROW_PARALLEL) or one 64-byte line (SPLIT_D)
    uint32_t unitsPerRow;    // ceil(D / unitElems)
    uint32_t batchRows;      // Rows whose reductions are in flight together (hides the sqrt latency)
    uint32_t chunkRows;      // Serpentine granularity: rows per traversal chunk
    uint32_t residentElems;  // Per-thread resident Z scratchpad (FP32 elements)
    bool streamStores;       // Non-temporal Y stores: working set overflows the last-level cache
    bool serpentine;         // Reverse the traversal on every other call
};

class AdaptiveTiler {
public:
    static constexpr uint32_t MAX_THREADS = 256;
    static constexpr uint32_t MAX_BATCH = 64;
    static constexpr size_t SCRATCHPAD_LIMIT = 191 * 1024; // 191 KB safe line (resident Z budget)
    static constexpr uint32_t LINE_BYTES = 64;             // Split quantum: one cache line = 2 x 32-byte bursts
    static constexpr uint32_t BATCH_ELEMS = 2048;          // Pass-1 work that covers one reduction latency
    static constexpr double CHUNK_BYTES = 64 * 1024;       // Serpentine chunk (prefetch-friendly run)

    // Cost model, measured on the reference 4-core Cascade Lake VM (docs/ARCHITECTURE_CHALLENGES.md).
    // Cache-resident rows are compute-bound, so work is counted in elements, not bytes.
    static constexpr double FORK_JOIN_NS = 3500.0;         // Wake + join of the OpenMP team
    static constexpr double BARRIER_NS = 1000.0;           // One extra team barrier
    static constexpr double CORE_NS_PER_ELEM = 0.3;        // One core, cache-resident, any dtype

    static inline TilingConfig Plan(uint32_t M, uint32_t D, uint32_t elemBytes, uint32_t P, size_t llcBytes) {
        TilingConfig cfg;
        const double rowBytes = 3.0 * D * elemBytes;       // X1 + X2 in, Y out
        const double totalBytes = rowBytes * M;

        // ---------------------------------------------------------------------
        // (1) Team size. t(1) = W*c, t(P) = tau + W*c/P  =>  fork iff W*c > tau*P/(P-1).
        // Never an intermediate team: resizing the OpenMP pool costs far more than it saves.
        // ---------------------------------------------------------------------
        const bool fork = P > 1 && static_cast<double>(M) * D * CORE_NS_PER_ELEM > FORK_JOIN_NS * P / (P - 1);
        cfg.threads = fork ? P : 1;

        // ---------------------------------------------------------------------
        // (2) Work quantum [ARCH CHALLENGE 1]. Whole rows leave the slowest thread
        // (ceil(M/T) - M/T) rows of extra work; 64-byte lines balance every thread to within
        // one line (any M, any T, no power-of-two restriction) at the price of one barrier.
        // ---------------------------------------------------------------------
        const uint32_t T = cfg.threads;
        const double excessRows = static_cast<double>((M + T - 1) / T) - static_cast<double>(M) / T;
        cfg.mode = excessRows * D * CORE_NS_PER_ELEM > BARRIER_NS ? TilingMode::SPLIT_D : TilingMode::ROW_PARALLEL;
        cfg.unitElems = std::max(1u, cfg.mode == TilingMode::SPLIT_D ? LINE_BYTES / elemBytes : D);
        cfg.unitsPerRow = (D + cfg.unitElems - 1) / cfg.unitElems;

        // (3) [ARCH CHALLENGE 2/3] Z stays resident in FP32 (4 B/elem) up to the scratchpad budget;
        // longer row segments recompute Z from X1 + X2 + bias instead of spilling. Short rows run
        // in batches of B* = BATCH_ELEMS / D so B reductions overlap instead of stalling pass 2.
        cfg.residentElems = static_cast<uint32_t>(SCRATCHPAD_LIMIT / sizeof(float));
        cfg.batchRows = std::max(1u, std::min(MAX_BATCH, BATCH_ELEMS / std::max(1u, D)));

        // (4) Stream Y past the caches once X1 + X2 + Y no longer fit in the LLC.
        cfg.streamStores = totalBytes > static_cast<double>(llcBytes);

        // (5) [ARCH CHALLENGE 4] Serpentine: the chunks touched last are still cache-resident,
        // so the next call consumes them first.
        cfg.serpentine = true;
        cfg.chunkRows = std::max(1u, static_cast<uint32_t>(CHUNK_BYTES / std::max(1.0, rowBytes)));
        return cfg;
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
};

} // namespace hpc
