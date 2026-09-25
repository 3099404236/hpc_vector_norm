#pragma once

#include "hpc_vector_norm.hpp"
#include "adaptive_tiler.hpp"
#include <cmath>
#include <vector>
#include <cstring>
#include <omp.h>

namespace hpc {

// -----------------------------------------------------------------------------
// 64-Element SIMD Sliding-Window Reduction (Zero Stack Overflow, Tree Fold)
// -----------------------------------------------------------------------------
static inline float ComputeSumOfSquares(const float* x, uint32_t len) {
    float acc[64] = {0.0f};
    uint32_t num64 = len / 64;

    for (uint32_t c = 0; c < num64; ++c) {
        const float* chunk = x + c * 64;
        #pragma omp simd
        for (int j = 0; j < 64; ++j) {
            acc[j] += chunk[j] * chunk[j];
        }
    }

    // Bisection Tree Reduction: 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
    for (int j = 0; j < 32; ++j) acc[j] += acc[j + 32];
    for (int j = 0; j < 16; ++j) acc[j] += acc[j + 16];
    for (int j = 0; j <  8; ++j) acc[j] += acc[j +  8];
    for (int j = 0; j <  4; ++j) acc[j] += acc[j +  4];
    for (int j = 0; j <  2; ++j) acc[j] += acc[j +  2];
    acc[0] += acc[1];

    // Remainder handling (for non-multiple of 64)
    for (uint32_t j = num64 * 64; j < len; ++j) {
        acc[0] += x[j] * x[j];
    }
    return acc[0];
}

// -----------------------------------------------------------------------------
// Unified Kernel Pipeline Engine (< 250 Lines)
// -----------------------------------------------------------------------------
template <typename T>
class KernelUnifiedPipeline {
public:
    static void Execute(
        const T* x1,
        const T* x2,
        const T* gamma,
        const T* bias,
        T* y,
        uint32_t M,
        uint32_t D,
        float eps
    ) {
        TilingConfig cfg = AdaptiveTiler::Plan(M, D, sizeof(T));

        if (cfg.mode == TilingMode::ROW_PARALLEL) {
            ExecuteRowParallel(x1, x2, gamma, bias, y, M, D, eps, cfg);
        } else {
            ExecuteSplitD(x1, x2, gamma, bias, y, M, D, eps, cfg);
        }
    }

private:
    static void ExecuteRowParallel(
        const T* x1,
        const T* x2,
        const T* gamma,
        const T* bias,
        T* y,
        uint32_t M,
        uint32_t D,
        float eps,
        const TilingConfig& cfg
    ) {
        #pragma omp parallel num_threads(cfg.activeThreads)
        {
            uint32_t tid = omp_get_thread_num();
            uint32_t startRow = tid * cfg.rowsPerThread;
            uint32_t endRow = std::min(M, startRow + cfg.rowsPerThread);

            // Thread-local scratchpad buffer (Max 191 KB, zero stack spill)
            std::vector<float> localZ(D);

            // -----------------------------------------------------------------
            // [ARCH CHALLENGE 3]: Zero-Bubble True Double-Buffering Pipeline (Ping-Pong)
            // (See docs/ARCHITECTURE_CHALLENGES.md)
            // Current: Synchronous sequential iteration.
            // Extension: Overlap SIMD compute of row i with prefetching row i+1.
            // -----------------------------------------------------------------
            for (uint32_t r = startRow; r < endRow; ++r) {
                const T* rowX1 = x1 + r * D;
                const T* rowX2 = x2 + r * D;
                T* rowY = y + r * D;

                // Step 1: Linear accumulation Z = X1 + X2 + bias
                #pragma omp simd
                for (uint32_t c = 0; c < D; ++c) {
                    float b = bias ? static_cast<float>(bias[c]) : 0.0f;
                    localZ[c] = static_cast<float>(rowX1[c]) + static_cast<float>(rowX2[c]) + b;
                }

                // Step 2: Sum of squares using sliding-window 64-element tree reduction
                float sumSq = ComputeSumOfSquares(localZ.data(), D);
                float meanSq = sumSq / static_cast<float>(D);
                float invRms = 1.0f / std::sqrt(meanSq + eps);

                // Step 3: Normalization & affine scaling (Zero reload, in-place localZ)
                #pragma omp simd
                for (uint32_t c = 0; c < D; ++c) {
                    float g = gamma ? static_cast<float>(gamma[c]) : 1.0f;
                    rowY[c] = static_cast<T>(localZ[c] * invRms * g);
                }
            }
        }
    }

    static void ExecuteSplitD(
        const T* x1,
        const T* x2,
        const T* gamma,
        const T* bias,
        T* y,
        uint32_t M,
        uint32_t D,
        float eps,
        const TilingConfig& cfg
    ) {
        // Shared scratchboard for inter-thread reduction across slices (e.g. 32 floats = 128 bytes)
        float scratchShared[AdaptiveTiler::MAX_THREADS] = {0.0f};

        #pragma omp parallel num_threads(cfg.activeThreads)
        {
            uint32_t tid = omp_get_thread_num();
            uint32_t row = tid / cfg.slicesPerRow;
            uint32_t slice = tid % cfg.slicesPerRow;
            uint32_t sliceStart = slice * cfg.sliceD;
            uint32_t sliceLen = std::min(cfg.sliceD, (sliceStart < D) ? (D - sliceStart) : 0);

            if (row < M && sliceLen > 0) {
                const T* rowX1 = x1 + row * D + sliceStart;
                const T* rowX2 = x2 + row * D + sliceStart;
                const T* rowBias = bias ? (bias + sliceStart) : nullptr;
                const T* rowGamma = gamma ? (gamma + sliceStart) : nullptr;
                T* rowY = y + row * D + sliceStart;

                // Resident local buffer for this slice (e.g. 8192 floats = 32 KB << 191 KB)
                std::vector<float> residentZ(sliceLen);

                // Phase 1: Local Z accumulation and local sum of squares
                #pragma omp simd
                for (uint32_t c = 0; c < sliceLen; ++c) {
                    float b = rowBias ? static_cast<float>(rowBias[c]) : 0.0f;
                    residentZ[c] = static_cast<float>(rowX1[c]) + static_cast<float>(rowX2[c]) + b;
                }

                float localSumSq = ComputeSumOfSquares(residentZ.data(), sliceLen);
                scratchShared[tid] = localSumSq;

                // Barrier: Synchronize across threads participating in row reduction
                #pragma omp barrier

                // Phase 2: Lightweight global reduction across slices of the same row
                uint32_t baseTid = row * cfg.slicesPerRow;
                float globalSumSq = 0.0f;
                for (uint32_t s = 0; s < cfg.slicesPerRow; ++s) {
                    globalSumSq += scratchShared[baseTid + s];
                }
                float meanSq = globalSumSq / static_cast<float>(D);
                float invRms = 1.0f / std::sqrt(meanSq + eps);

                // Phase 3: In-place normalization reusing residentZ (Zero GM reload!)
                #pragma omp simd
                for (uint32_t c = 0; c < sliceLen; ++c) {
                    float g = rowGamma ? static_cast<float>(rowGamma[c]) : 1.0f;
                    rowY[c] = static_cast<T>(residentZ[c] * invRms * g);
                }
            }
        }
    }
};

} // namespace hpc
