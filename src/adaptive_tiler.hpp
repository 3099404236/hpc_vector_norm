#pragma once

#include <cstdint>
#include <algorithm>

namespace hpc {

enum class TilingMode {
    ROW_PARALLEL, // M >= 32: Core assigned to row blocks (Zero cross-core communication)
    SPLIT_D       // M < 32: Core assigned to (Row, ColumnSlice) with inter-thread reduction
};

struct TilingConfig {
    TilingMode mode;
    uint32_t activeThreads;
    uint32_t rowsPerThread;
    uint32_t slicesPerRow;
    uint32_t sliceD;
    uint32_t batchRows;
    uint32_t tileD;
    size_t localBufferBytes;
};

class AdaptiveTiler {
public:
    static constexpr uint32_t MAX_THREADS = 40;
    static constexpr size_t SCRATCHPAD_LIMIT = 191 * 1024; // 191 KB safe line
    static constexpr uint32_t ALIGN_BYTES = 32;            // 32-byte memory quantum

    static inline TilingConfig Plan(uint32_t M, uint32_t D, uint32_t elemBytes) {
        TilingConfig cfg;
        
        // ---------------------------------------------------------------------
        // Decision Criterion: Row-Parallel vs Split-D
        // ---------------------------------------------------------------------
        if (M >= 32) {
            // Mode 1: Row Parallel
            cfg.mode = TilingMode::ROW_PARALLEL;
            cfg.activeThreads = std::min(MAX_THREADS, M);
            cfg.rowsPerThread = (M + cfg.activeThreads - 1) / cfg.activeThreads;
            cfg.slicesPerRow = 1;
            cfg.sliceD = D;

            // Determine Batch Size & TileD constrained by SCRATCHPAD_LIMIT
            // -----------------------------------------------------------------
            // [ARCH CHALLENGE 2]: Knapsack Batch-Size Formulation (docs/ARCHITECTURE_CHALLENGES.md)
            // State per element ~ 16 bytes (using 64-element streaming reduction).
            // Maximize B such that (2*B*D*elemBytes_in + B*D*elemBytes_out + aux) <= 195584.
            // -----------------------------------------------------------------
            uint32_t maxElementsInLocal = static_cast<uint32_t>(SCRATCHPAD_LIMIT / (elemBytes * 8));
            if (D <= maxElementsInLocal) {
                // Entire row fits in scratchpad
                cfg.tileD = D;
                cfg.batchRows = std::max(1u, std::min(cfg.rowsPerThread, maxElementsInLocal / D));
            } else {
                // Large D: tile across columns
                cfg.tileD = 2048;
                cfg.batchRows = 1;
            }
            cfg.localBufferBytes = cfg.batchRows * cfg.tileD * elemBytes * 4;
        } else {
            // Mode 2: Split-D (Few rows, e.g., M=8, D=32768 or M=1, D=64)
            cfg.mode = TilingMode::SPLIT_D;
            
            if (M == 1) {
                // Single row launch bypass
                cfg.activeThreads = 1;
                cfg.slicesPerRow = 1;
                cfg.sliceD = D;
                cfg.rowsPerThread = 1;
                cfg.batchRows = 1;
                cfg.tileD = D;
            } else {
                // M > 1 and M < 32: divide columns among cores
                // -------------------------------------------------------------
                // [ARCH CHALLENGE 1]: Generalized 40-Thread Split-D (docs/ARCHITECTURE_CHALLENGES.md)
                // Default: Pure mathematical power-of-two slice derivation (max 2^k <= MAX_THREADS / M)
                // Open Extension: Partition into non-power-of-two (e.g. 5 slices per row for M=8)
                // while maintaining strict 32-byte hardware boundary alignment.
                // -------------------------------------------------------------
                uint32_t maxSlices = MAX_THREADS / M;
                uint32_t slices = 1;
                while ((slices << 1) <= maxSlices) {
                    slices <<= 1;
                }

                cfg.slicesPerRow = slices;
                cfg.activeThreads = M * slices;
                cfg.sliceD = (D + slices - 1) / slices;
                
                // Align sliceD to 32-byte hardware boundary
                uint32_t alignElem = ALIGN_BYTES / elemBytes;
                cfg.sliceD = ((cfg.sliceD + alignElem - 1) / alignElem) * alignElem;

                cfg.rowsPerThread = 1;
                cfg.batchRows = 1;
                cfg.tileD = cfg.sliceD;
            }
            // In Split-D, the slice must stay resident in local buffer (Zero reload)
            cfg.localBufferBytes = cfg.sliceD * elemBytes * 4;
        }

        return cfg;
    }
};

} // namespace hpc
