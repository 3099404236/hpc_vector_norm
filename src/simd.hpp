#pragma once

#include "hpc_vector_norm.hpp"
#include <cstdint>
#include <cstring>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace hpc {

// -----------------------------------------------------------------------------
// Storage codecs: arithmetic always runs in FP32 lanes; 16-bit formats are widened
// on load and rounded to nearest-even on store.
// -----------------------------------------------------------------------------
struct F32  { using S = float;    static constexpr S kOne = 1.0f;   };
struct F16  { using S = uint16_t; static constexpr S kOne = 0x3C00; };
struct BF16 { using S = uint16_t; static constexpr S kOne = 0x3F80; };

// -----------------------------------------------------------------------------
// SIMD layer: AVX-512 (16 lanes) > AVX2 + FMA + F16C (8 lanes) > portable scalar
// -----------------------------------------------------------------------------
namespace simd {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
using V = __m512;
using H = __m256i; // 16 packed 16-bit values
constexpr uint32_t W = 16;
inline V Zero() { return _mm512_setzero_ps(); }
inline V Set1(float x) { return _mm512_set1_ps(x); }
inline V Add(V a, V b) { return _mm512_add_ps(a, b); }
inline V Mul(V a, V b) { return _mm512_mul_ps(a, b); }
inline V Fma(V a, V b, V c) { return _mm512_fmadd_ps(a, b, c); }
inline float Sum(V a) { return _mm512_reduce_add_ps(a); }
inline __mmask16 Lanes(uint32_t n) { return static_cast<__mmask16>((1u << n) - 1u); }
inline V Widen(F16, H h) { return _mm512_cvtph_ps(h); }
inline V Widen(BF16, H h) { return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16)); }
inline H Narrow(F16, V v) { return _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }
// BF16 round-to-nearest-even without a NaN fix-up: every NaN this kernel can produce comes
// from BF16 operands or is the default NaN, so its low 16 bits are zero and the rounding
// increment can never carry out of the mantissa.
inline H Narrow(BF16, V v) {
    const __m512i x = _mm512_castps_si512(v);
    const __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(x, 16), _mm512_set1_epi32(1));
    const __m512i r = _mm512_add_epi32(x, _mm512_add_epi32(lsb, _mm512_set1_epi32(0x7FFF)));
    return _mm512_cvtepi32_epi16(_mm512_srli_epi32(r, 16));
}
inline V Load(F32, const float* p) { return _mm512_loadu_ps(p); }
inline V LoadN(F32, const float* p, uint32_t n) { return _mm512_maskz_loadu_ps(Lanes(n), p); }
inline void Store(F32, float* p, V v) { _mm512_storeu_ps(p, v); }
inline void StoreN(F32, float* p, V v, uint32_t n) { _mm512_mask_storeu_ps(p, Lanes(n), v); }
inline void Stream(F32, float* p, V v) { _mm512_stream_ps(p, v); }
template <class C> V Load(C c, const uint16_t* p) { return Widen(c, _mm256_loadu_si256(reinterpret_cast<const H*>(p))); }
template <class C> V LoadN(C c, const uint16_t* p, uint32_t n) { return Widen(c, _mm256_maskz_loadu_epi16(Lanes(n), p)); }
template <class C> void Store(C c, uint16_t* p, V v) { _mm256_storeu_si256(reinterpret_cast<H*>(p), Narrow(c, v)); }
template <class C> void StoreN(C c, uint16_t* p, V v, uint32_t n) { _mm256_mask_storeu_epi16(p, Lanes(n), Narrow(c, v)); }
template <class C> void Stream(C c, uint16_t* p, V v) { _mm256_stream_si256(reinterpret_cast<H*>(p), Narrow(c, v)); }
inline void Fence() { _mm_sfence(); }
#elif defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
using V = __m256;
using H = __m128i; // 8 packed 16-bit values
constexpr uint32_t W = 8;
inline V Zero() { return _mm256_setzero_ps(); }
inline V Set1(float x) { return _mm256_set1_ps(x); }
inline V Add(V a, V b) { return _mm256_add_ps(a, b); }
inline V Mul(V a, V b) { return _mm256_mul_ps(a, b); }
inline V Fma(V a, V b, V c) { return _mm256_fmadd_ps(a, b, c); }
inline float Sum(V a) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)));
}
inline V Widen(F16, H h) { return _mm256_cvtph_ps(h); }
inline V Widen(BF16, H h) { return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16)); }
inline H Narrow(F16, V v) { return _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }
inline H Narrow(BF16, V v) { // RNE; NaN-safe for the same reason as the AVX-512 path
    const __m256i x = _mm256_castps_si256(v);
    const __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(x, 16), _mm256_set1_epi32(1));
    const __m256i r = _mm256_srli_epi32(_mm256_add_epi32(x, _mm256_add_epi32(lsb, _mm256_set1_epi32(0x7FFF))), 16);
    return _mm_packus_epi32(_mm256_castsi256_si128(r), _mm256_extracti128_si256(r, 1));
}
inline V Load(F32, const float* p) { return _mm256_loadu_ps(p); }
inline void Store(F32, float* p, V v) { _mm256_storeu_ps(p, v); }
inline void Stream(F32, float* p, V v) { _mm256_stream_ps(p, v); }
template <class C> V Load(C c, const uint16_t* p) { return Widen(c, _mm_loadu_si128(reinterpret_cast<const H*>(p))); }
template <class C> void Store(C c, uint16_t* p, V v) { _mm_storeu_si128(reinterpret_cast<H*>(p), Narrow(c, v)); }
template <class C> void Stream(C c, uint16_t* p, V v) { _mm_stream_si128(reinterpret_cast<H*>(p), Narrow(c, v)); }
inline void Fence() { _mm_sfence(); }
#else
using V = float; // Portable fallback: one FP32 lane
constexpr uint32_t W = 1;
inline V Zero() { return 0.0f; }
inline V Set1(float x) { return x; }
inline V Add(V a, V b) { return a + b; }
inline V Mul(V a, V b) { return a * b; }
inline V Fma(V a, V b, V c) { return a * b + c; }
inline float Sum(V a) { return a; }
inline V Load(F32, const float* p) { return *p; }
inline V Load(F16, const uint16_t* p) { return HalfToFloat(*p); }
inline V Load(BF16, const uint16_t* p) { return BF16ToFloat(*p); }
inline void Store(F32, float* p, V v) { *p = v; }
inline void Store(F16, uint16_t* p, V v) { *p = FloatToHalf(v); }
inline void Store(BF16, uint16_t* p, V v) { *p = FloatToBF16(v); }
template <class C, class S> void Stream(C c, S* p, V v) { Store(c, p, v); }
inline void Fence() {}
#endif
#if !(defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__))
// Partial vectors (row tails only) through a zero-padded bounce buffer
template <class C, class S> V LoadN(C c, const S* p, uint32_t n) {
    alignas(64) S t[W] = {};
    std::memcpy(t, p, n * sizeof(S));
    return Load(c, t);
}
template <class C, class S> void StoreN(C c, S* p, V v, uint32_t n) {
    alignas(64) S t[W];
    Store(c, t, v);
    std::memcpy(p, t, n * sizeof(S));
}
#endif
} // namespace simd

} // namespace hpc
