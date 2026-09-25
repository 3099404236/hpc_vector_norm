#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>

namespace hpc {

enum class DataType {
    FP16,  // IEEE-754 binary16, passed as raw uint16_t storage
    FP32,
    BF16   // bfloat16, passed as raw uint16_t storage
};

/**
 * FusedResidualNormalize
 *
 * Mathematical definition:
 *   Z[i, j] = X1[i, j] + X2[i, j] + bias[j]
 *   sigma[i] = sqrt( (1 / D) * sum_{j=0}^{D-1} (Z[i, j]^2) + eps )
 *   Y[i, j] = (Z[i, j] / sigma[i]) * gamma[j]
 *
 * x1, x2, y are dense row-major [rows, cols]; gamma and bias are [cols] and may be null
 * (gamma = 1, bias = 0). All tensors use `dtype`; arithmetic is FP32 with FP64 row sums.
 * Target architecture: Up to 40 symmetric CPU threads with L1 scratchpad limits.
 */
void FusedResidualNormalize(
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

// Backward-compatible alias
inline void FusedAddRmsNormBias(
    const void* x1, const void* x2, const void* gamma, const void* bias,
    void* y, uint32_t rows, uint32_t cols, DataType dtype, float eps = 1e-6f) {
    FusedResidualNormalize(x1, x2, gamma, bias, y, rows, cols, dtype, eps);
}

// -----------------------------------------------------------------------------
// Scalar storage conversions (round-to-nearest-even, NaN stays NaN) for preparing
// and inspecting FP16 / BF16 tensors.
// -----------------------------------------------------------------------------
inline float BF16ToFloat(uint16_t v) {
    const uint32_t bits = static_cast<uint32_t>(v) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

inline uint16_t FloatToBF16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof x);
    if ((x & 0x7FFFFFFFu) > 0x7F800000u) return static_cast<uint16_t>((x >> 16) | 0x40u);
    return static_cast<uint16_t>((x + 0x7FFFu + ((x >> 16) & 1u)) >> 16);
}

inline float HalfToFloat(uint16_t v) {
    const uint32_t sign = static_cast<uint32_t>(v & 0x8000u) << 16, em = v & 0x7FFFu;
    uint32_t bits;
    if (em >= 0x7C00u) {                 // Inf / NaN
        bits = sign | 0x7F800000u | ((em & 0x3FFu) << 13);
    } else if (em >= 0x0400u) {          // normal: rebias exponent 15 -> 127
        bits = sign | ((em << 13) + 0x38000000u);
    } else {                             // zero / subnormal: m * 2^-24 is exact in FP32
        const float f = static_cast<float>(em) * 5.9604644775390625e-8f;
        std::memcpy(&bits, &f, sizeof bits);
        bits |= sign;
    }
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

inline uint16_t FloatToHalf(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof x);
    const uint32_t sign = (x >> 16) & 0x8000u;
    x &= 0x7FFFFFFFu;
    uint32_t h;
    if (x >= 0x47800000u) {              // overflow -> Inf, NaN -> quiet NaN
        h = x > 0x7F800000u ? 0x7E00u : 0x7C00u;
    } else if (x < 0x38800000u) {        // subnormal / zero: FP32 addition rounds at 2^-24
        float a;
        std::memcpy(&a, &x, sizeof a);
        a += 0.5f;
        std::memcpy(&h, &a, sizeof h);
        h -= 0x3F000000u;
    } else {                             // normal: rebias exponent, round 13 dropped bits
        h = (x + 0xC8000FFFu + ((x >> 13) & 1u)) >> 13;
    }
    return static_cast<uint16_t>(sign | h);
}

} // namespace hpc
