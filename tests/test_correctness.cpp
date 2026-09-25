// Correctness sweep: every dtype x tiling path x team size x shape against an FP64 reference.
// Forced plans (Split-D, streaming stores, recomputed Z) reach paths the cost model only
// picks on large machines or large tensors; canaries catch any write outside Y.
#include "hpc_vector_norm.hpp"
#include "kernel_unified.hpp"
#include <omp.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace hpc;

namespace {

struct Variant {
    const char* name;
    bool viaApi;      // public API with the cost-model plan
    bool split;       // force SPLIT_D (64-byte quanta)
    bool stream;      // force non-temporal stores
    uint32_t cap;     // resident Z capacity override (0 = keep plan)
};

const Variant kVariants[] = {
    {"api", true, false, false, 0},
    {"split", false, true, false, 0},
    {"split+stream", false, true, true, 0},
    {"split+stream+recompute", false, true, true, 24},
    {"rows+stream+recompute", false, false, true, 24},
};

template <class C> DataType TypeOf();
template <> DataType TypeOf<F32>() { return DataType::FP32; }
template <> DataType TypeOf<F16>() { return DataType::FP16; }
template <> DataType TypeOf<BF16>() { return DataType::BF16; }

float Decode(float v) { return v; }
float Decode(uint16_t v, F16) { return HalfToFloat(v); }
float Decode(uint16_t v, BF16) { return BF16ToFloat(v); }
template <class C> float Dec(typename C::S v) { if constexpr (sizeof(v) == 4) return Decode(v); else return Decode(v, C{}); }
template <class C> typename C::S Enc(float f) {
    if constexpr (std::is_same<C, F32>::value) return f;
    else if constexpr (std::is_same<C, F16>::value) return FloatToHalf(f);
    else return FloatToBF16(f);
}
template <class C> double RelTol() {
    if constexpr (std::is_same<C, F32>::value) return 1e-5;
    else if constexpr (std::is_same<C, F16>::value) return 1.0 / 1024;  // 2 half-ulps
    else return 1.0 / 128;                                               // 2 bf16 half-ulps
}

// 64-byte aligned view into a padded vector, shifted by `offset` elements
template <class S> struct Buf {
    std::vector<S> mem;
    S* p;
    Buf(size_t n, uint32_t offset) : mem(n + offset + 64) {
        const uintptr_t a = (reinterpret_cast<uintptr_t>(mem.data()) + 63) & ~uintptr_t(63);
        p = reinterpret_cast<S*>(a) + offset;
    }
};

int g_checks = 0, g_failures = 0;

template <class C>
void RunShape(uint32_t M, uint32_t D, std::mt19937& rng) {
    using S = typename C::S;
    const size_t N = static_cast<size_t>(M) * D;
    const bool big = N > 50000;
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<S> x1(N), x2(N), gamma(D), bias(D);
    for (auto& v : x1) v = Enc<C>(dist(rng));
    for (auto& v : x2) v = Enc<C>(dist(rng));
    for (auto& v : gamma) v = Enc<C>(1.0f + 0.5f * dist(rng));
    for (auto& v : bias) v = Enc<C>(0.1f * dist(rng));

    for (int pc = 0; pc < (big ? 1 : 4); ++pc) {
        const bool hasGamma = !(pc & 1), hasBias = !(pc & 2);
        // FP64 reference on the decoded (already quantized) inputs
        std::vector<double> ref(N);
        for (uint32_t i = 0; i < M; ++i) {
            double ss = 0.0;
            for (uint32_t j = 0; j < D; ++j) {
                const size_t k = static_cast<size_t>(i) * D + j;
                ref[k] = double(Dec<C>(x1[k])) + Dec<C>(x2[k]) + (hasBias ? Dec<C>(bias[j]) : 0.0);
                ss += ref[k] * ref[k];
            }
            const double inv = 1.0 / std::sqrt(ss / D + 1e-6);
            for (uint32_t j = 0; j < D; ++j) {
                const size_t k = static_cast<size_t>(i) * D + j;
                ref[k] *= inv * (hasGamma ? Dec<C>(gamma[j]) : 1.0);
            }
        }
        for (uint32_t offset : {0u, 1u}) {
            Buf<S> bx1(N, offset), bx2(N, offset), bg(D, offset), bb(D, offset), by(N, offset);
            std::copy(x1.begin(), x1.end(), bx1.p);
            std::copy(x2.begin(), x2.end(), bx2.p);
            std::copy(gamma.begin(), gamma.end(), bg.p);
            std::copy(bias.begin(), bias.end(), bb.p);
            const S* g = hasGamma ? bg.p : nullptr;
            const S* b = hasBias ? bb.p : nullptr;
            for (int threads : {1, 2, 3, 4, 7, 40}) {
                if (big && threads == 40 && offset) continue;
                for (const Variant& v : kVariants) {
                    std::vector<S> first;
                    for (int call = 0; call < 2; ++call) {  // 2nd call runs the reversed traversal
                        std::memset(by.mem.data(), 0x5A, by.mem.size() * sizeof(S));
                        if (v.viaApi) {
                            omp_set_num_threads(threads);
                            FusedResidualNormalize(bx1.p, bx2.p, g, b, by.p, M, D, TypeOf<C>(), 1e-6f);
                        } else {
                            TilingConfig cfg = AdaptiveTiler::Plan(M, D, sizeof(S), threads, AdaptiveTiler::LastLevelCacheBytes());
                            cfg.threads = threads;
                            if (v.split) {
                                cfg.mode = TilingMode::SPLIT_D;
                                cfg.unitElems = std::max<uint32_t>(1, AdaptiveTiler::LINE_BYTES / sizeof(S));
                                cfg.unitsPerRow = (D + cfg.unitElems - 1) / cfg.unitElems;
                            }
                            cfg.streamStores = cfg.streamStores || v.stream;
                            if (v.cap) cfg.residentElems = v.cap;
                            KernelUnifiedPipeline<C>::ExecutePlan(bx1.p, bx2.p, g, b, by.p, M, D, 1e-6f, cfg);
                        }
                        ++g_checks;
                        double maxErr = 0.0;
                        bool ok = true;
                        for (size_t k = 0; k < N; ++k) {
                            const double err = std::fabs(double(Dec<C>(by.p[k])) - ref[k]);
                            maxErr = std::max(maxErr, err);
                            ok &= err <= 1e-5 + RelTol<C>() * std::fabs(ref[k]);
                        }
                        // Canaries: nothing outside [p, p + N) may be touched
                        const unsigned char* raw = reinterpret_cast<const unsigned char*>(by.mem.data());
                        const size_t lo = (by.p - by.mem.data()) * sizeof(S), hi = lo + N * sizeof(S);
                        for (size_t k = 0; k < by.mem.size() * sizeof(S); ++k) ok &= (k >= lo && k < hi) || raw[k] == 0x5A;
                        if (call == 0) first.assign(by.p, by.p + N);
                        else ok &= N == 0 || std::memcmp(first.data(), by.p, N * sizeof(S)) == 0;
                        if (!ok) {
                            ++g_failures;
                            std::printf("FAIL dtype=%d M=%u D=%u threads=%d gamma=%d bias=%d offset=%u variant=%s call=%d maxErr=%.3g\n",
                                        int(TypeOf<C>()), M, D, threads, hasGamma, hasBias, offset, v.name, call, maxErr);
                        }
                    }
                }
            }
        }
    }
}

template <class C>
void RunNested(std::mt19937& rng) {  // Called from inside a user parallel region
    using S = typename C::S;
    const uint32_t M = 64, D = 3000;
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<S> x1(M * D), x2(M * D), y0(M * D), y1(M * D);
    for (auto& v : x1) v = Enc<C>(dist(rng));
    for (auto& v : x2) v = Enc<C>(dist(rng));
    omp_set_num_threads(4);
    FusedResidualNormalize(x1.data(), x2.data(), nullptr, nullptr, y0.data(), M, D, TypeOf<C>());
    #pragma omp parallel num_threads(2)
    {
        if (omp_get_thread_num() == 0) FusedResidualNormalize(x1.data(), x2.data(), nullptr, nullptr, y1.data(), M, D, TypeOf<C>());
    }
    ++g_checks;
    if (std::memcmp(y0.data(), y1.data(), y0.size() * sizeof(S)) != 0) {
        ++g_failures;
        std::printf("FAIL nested dtype=%d\n", int(TypeOf<C>()));
    }
}

template <class C>
void RunNaN() {  // A NaN input poisons exactly its own row, through every narrowing path
    using S = typename C::S;
    const uint32_t M = 16, D = 300;
    std::vector<S> x1(M * D, Enc<C>(0.5f)), x2(M * D, Enc<C>(-0.25f)), g(D, Enc<C>(1.5f)), y(M * D);
    x1[3 * D + 7] = Enc<C>(std::nanf(""));
    for (int threads : {1, 4}) {
        for (bool split : {false, true}) {
            TilingConfig cfg = AdaptiveTiler::Plan(M, D, sizeof(S), threads, AdaptiveTiler::LastLevelCacheBytes());
            cfg.threads = threads;
            if (split) {
                cfg.mode = TilingMode::SPLIT_D;
                cfg.unitElems = std::max<uint32_t>(1, AdaptiveTiler::LINE_BYTES / sizeof(S));
                cfg.unitsPerRow = (D + cfg.unitElems - 1) / cfg.unitElems;
            }
            KernelUnifiedPipeline<C>::ExecutePlan(x1.data(), x2.data(), g.data(), nullptr, y.data(), M, D, 1e-6f, cfg);
            ++g_checks;
            bool ok = true;
            for (uint32_t i = 0; i < M * D; ++i) ok &= (i / D == 3) == std::isnan(Dec<C>(y[i]));
            if (!ok) {
                ++g_failures;
                std::printf("FAIL nan propagation dtype=%d threads=%d split=%d\n", int(TypeOf<C>()), threads, split);
            }
        }
    }
}

template <class C>
void RunAll(std::mt19937& rng) {
    RunNaN<C>();
    const uint32_t shapes[][2] = {
        {1, 1}, {1, 7}, {1, 64}, {2, 3}, {3, 5}, {7, 200}, {5, 33}, {4, 1000}, {17, 4097},
        {64, 256}, {41, 500}, {0, 16}, {16, 0}, {1, 70001}, {3, 100003}, {8, 32768},
    };
    for (const auto& s : shapes) RunShape<C>(s[0], s[1], rng);
    RunNested<C>(rng);
}

// Conversion helpers against exhaustive / known values
void TestConversions() {
    for (uint32_t h = 0; h < 0x10000; ++h) {  // FP16 -> FP32 -> FP16 round-trips every pattern
        const uint16_t v = static_cast<uint16_t>(h);
        const bool nan = (v & 0x7C00u) == 0x7C00u && (v & 0x3FFu);
        ++g_checks;
        if (!nan && FloatToHalf(HalfToFloat(v)) != v) { ++g_failures; std::printf("FAIL fp16 roundtrip %04x\n", h); }
        if (nan && !std::isnan(HalfToFloat(v))) { ++g_failures; std::printf("FAIL fp16 nan %04x\n", h); }
    }
    const struct { float f; uint16_t h; } cases[] = {
        {65504.0f, 0x7BFF}, {65519.0f, 0x7BFF}, {65520.0f, 0x7C00}, {1e9f, 0x7C00}, {5.9604645e-8f, 0x0001},
        {2.9802322e-8f, 0x0000}, {2.9802326e-8f, 0x0001}, {1.0009765625f, 0x3C01}, {1.00048828125f, 0x3C00},
        {1.00146484375f, 0x3C02}, {-2.0f, 0xC000},
    };
    for (const auto& c : cases) {
        ++g_checks;
        if (FloatToHalf(c.f) != c.h) { ++g_failures; std::printf("FAIL fp16 %.9g -> %04x\n", c.f, FloatToHalf(c.f)); }
    }
    ++g_checks;
    if (FloatToBF16(1.00390625f) != 0x3F80 || FloatToBF16(1.01171875f) != 0x3F82 || FloatToBF16(-3.0f) != 0xC040) {
        ++g_failures;
        std::printf("FAIL bf16 rounding\n");
    }
}

} // namespace

int main() {
    std::mt19937 rng(1234);
    TestConversions();
    RunAll<F32>(rng);
    RunAll<F16>(rng);
    RunAll<BF16>(rng);
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
