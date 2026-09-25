#pragma once

#include <cstdint>
#include <cstddef>

namespace hpc {

enum class DataType {
    FP16,
    FP32,
    BF16
};

/**
 * FusedAddRmsNormBias
 * 
 * Mathematical definition:
 *   Z[i, j] = X1[i, j] + X2[i, j] + bias[j]
 *   sigma[i] = sqrt( (1 / D) * sum_{j=0}^{D-1} (Z[i, j]^2) + eps )
 *   Y[i, j] = (Z[i, j] / sigma[i]) * gamma[j]
 * 
 * Target hardware: Up to 40 symmetric cores with L1 scratchpad limits.
 */
void FusedAddRmsNormBias(
    const void* x1,
    const void* x2,
    const void* gamma,
    const void* bias,
    void* y,
    uint32_t rows,
    uint32_t cols,
    DataType dtype,
    float eps = 1e-6f
);

} // namespace hpc
