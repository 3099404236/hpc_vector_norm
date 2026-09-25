// Target-side checks: the 40-core plans obey the hardware laws, and the DAE pipeline that
// executes them (dsa_runtime, one OpenMP thread per simulated core) is exact, sanitizer-clean
// and only pads DMA transfers where a row does not end on a 32-byte block.
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

int g_checks = 0, g_failures = 0;

void Check(bool ok, const char* what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("FAIL %s\n", what);
    }
}

template <class C> const char* Name();
template <> const char* Name<F32>() { return "FP32"; }
template <> const char* Name<F16>() { return "FP16"; }
template <> const char* Name<BF16>() { return "BF16"; }
template <class C> typename C::S Enc(float f) {
    if constexpr (std::is_same<C, F32>::value) return f;
    else if constexpr (std::is_same<C, F16>::value) return FloatToHalf(f);
    else return FloatToBF16(f);
}
template <class C> double Dec(typename C::S v) {
    if constexpr (std::is_same<C, F32>::value) return v;
    else if constexpr (std::is_same<C, F16>::value) return HalfToFloat(v);
    else return BF16ToFloat(v);
}
template <class C> double RelTol() {
    if constexpr (std::is_same<C, F32>::value) return 1e-5;
    else if constexpr (std::is_same<C, F16>::value) return 1.0 / 1024;
    else return 1.0 / 128;
}

// System memory on the 32-byte DMA grid, optionally shifted by `offset` elements
template <class S> struct HostBuffer {
    std::vector<S> mem;
    S* p;
    HostBuffer(size_t n, uint32_t offset) : mem(n + offset + 64) {
        p = reinterpret_cast<S*>((reinterpret_cast<uintptr_t>(mem.data()) + 63) & ~uintptr_t(63)) + offset;
    }
};

// -----------------------------------------------------------------------------
// 1. Plan invariants for the 15 profiles at full size on the 40-core target
// -----------------------------------------------------------------------------
void CheckProfilePlans() {
    struct Profile { const char* name; uint32_t M, D, s; } profiles[] = {
        {"P01", 1, 64, 2}, {"P02", 7, 200, 4}, {"P03", 128, 256, 4}, {"P04", 768, 192, 2}, {"P05", 8, 32768, 2},
        {"P06", 1536, 576, 2}, {"P07", 10240, 400, 2}, {"P08", 10240, 512, 2}, {"P09", 4096, 1536, 4},
        {"P10", 8192, 1024, 2}, {"P11", 4096, 3072, 4}, {"P12", 4096, 4096, 2}, {"P13", 10240, 3072, 2},
        {"P14", 2097152, 128, 2}, {"P15", 115000, 8192, 2}};
    const HardwareModel hw = HardwareModel::Target();
    char msg[256];
    for (const auto& p : profiles) {
        const TilingConfig t = AdaptiveTiler::Plan(p.M, p.D, p.s, hw);
        const uint64_t total = uint64_t(p.M) * p.D;
        uint64_t lo = ~0ull, hi = 0;
        for (uint32_t b = 0; b < t.blocks; ++b) {
            const uint64_t e0 = std::min(t.units * b / t.blocks * t.unitElems, total);
            const uint64_t e1 = std::min(t.units * (b + 1) / t.blocks * t.unitElems, total);
            lo = std::min(lo, e1 - e0);
            hi = std::max(hi, e1 - e0);
        }
        std::snprintf(msg, sizeof msg, "%s plan: blocks=%u spm=%u units of %llu elems, load %llu..%llu", p.name, t.blocks,
                      t.layout.Total(), (unsigned long long)t.unitElems, (unsigned long long)lo, (unsigned long long)hi);
        Check(t.blocks >= 1 && t.blocks <= dsa::MAX_HARDWARE_CORES, msg);
        Check(t.layout.Total() <= dsa::SCRATCHPAD_SAFE_WATERLINE, msg);
        Check(hi - lo <= t.unitElems, msg);                                   // Balanced to one unit
        Check(t.unitElems * p.s % dsa::DMA_ALIGN_BYTES == 0, msg);            // Units are whole DMA blocks
        Check(t.tileRows ? true : t.tileElems * p.s % dsa::DMA_ALIGN_BYTES == 0, msg);
        Check(t.blocks == std::min<uint64_t>(dsa::MAX_HARDWARE_CORES, t.units), msg);  // Saturates the cores
    }
    // P05: 16384 blocks of 32 bytes over 40 cores -> 409 or 410 blocks each (409.6 mean)
    const TilingConfig p5 = AdaptiveTiler::Plan(8, 32768, 2, hw);
    Check(p5.mode == TilingMode::SPLIT_D && p5.blocks == 40 && p5.unitElems == 16 && p5.units == 16384, "P05 split plan");
    // P04: two pipelined batches of 10 rows per core; P08: >= 24-row batches in 191 KB
    Check(AdaptiveTiler::Plan(768, 192, 2, hw).tileRows == 10, "P04 tile rows");
    Check(AdaptiveTiler::Plan(10240, 512, 2, hw).tileRows >= 24, "P08 tile rows");
}

// -----------------------------------------------------------------------------
// 2. DAE execution vs FP64 reference
// -----------------------------------------------------------------------------
template <class C>
void RunShape(uint32_t M, uint32_t D, int variant, bool hasGamma, bool hasBias, uint32_t offset, std::mt19937& rng) {
    using S = typename C::S;
    const size_t N = size_t(M) * D;
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    HostBuffer<S> x1(N, offset), x2(N, offset), g(D, offset), b(D, offset), y(N, offset);
    for (size_t i = 0; i < N; ++i) { x1.p[i] = Enc<C>(dist(rng)); x2.p[i] = Enc<C>(dist(rng)); }
    for (uint32_t j = 0; j < D; ++j) { g.p[j] = Enc<C>(1.0f + 0.5f * dist(rng)); b.p[j] = Enc<C>(0.1f * dist(rng)); }
    std::memset(y.mem.data(), 0x5A, y.mem.size() * sizeof(S));

    const HardwareModel hw = HardwareModel::Target();
    TilingConfig plan = AdaptiveTiler::Plan(M, D, sizeof(S), hw);           // variant 0: the model's plan
    if (variant == 1) plan = AdaptiveTiler::Build(M, D, sizeof(S), hw, TilingMode::SPLIT_D, true);
    if (variant == 2) plan = AdaptiveTiler::Build(M, D, sizeof(S), hw, TilingMode::ROW_PARALLEL, true);
    DaeStats st;
    char msg[256];
    std::snprintf(msg, sizeof msg, "%s M=%u D=%u variant=%d gamma=%d bias=%d offset=%u blocks=%u mode=%s tileRows=%u tile=%u zRes=%u",
                  Name<C>(), M, D, variant, hasGamma, hasBias, offset, plan.blocks,
                  plan.mode == TilingMode::SPLIT_D ? "split" : "rows", plan.tileRows, plan.tileElems, plan.zResident);
    try {
        st = DaePipeline<C>::Execute(x1.p, x2.p, hasGamma ? g.p : nullptr, hasBias ? b.p : nullptr, y.p, M, D, 1e-6f, plan);
    } catch (const std::exception& e) {
        std::printf("FAIL sanitizer trap: %s\n  %s\n", msg, e.what());
        ++g_checks;
        ++g_failures;
        return;
    }
    bool ok = true;
    double maxErr = 0.0;
    for (uint32_t i = 0; i < M; ++i) {
        double ss = 0.0;
        for (uint32_t j = 0; j < D; ++j) {
            const double z = Dec<C>(x1.p[size_t(i) * D + j]) + Dec<C>(x2.p[size_t(i) * D + j]) + (hasBias ? Dec<C>(b.p[j]) : 0.0);
            ss += z * z;
        }
        const double inv = 1.0 / std::sqrt(ss / D + 1e-6);
        for (uint32_t j = 0; j < D; ++j) {
            const double z = Dec<C>(x1.p[size_t(i) * D + j]) + Dec<C>(x2.p[size_t(i) * D + j]) + (hasBias ? Dec<C>(b.p[j]) : 0.0);
            const double ref = z * inv * (hasGamma ? Dec<C>(g.p[j]) : 1.0);
            const double err = std::fabs(Dec<C>(y.p[size_t(i) * D + j]) - ref);
            maxErr = std::max(maxErr, err);
            ok &= err <= 1e-4 + RelTol<C>() * std::fabs(ref);
        }
    }
    const unsigned char* raw = reinterpret_cast<const unsigned char*>(y.mem.data());
    const size_t lo = (y.p - y.mem.data()) * sizeof(S), hi = lo + N * sizeof(S);
    for (size_t k = 0; k < y.mem.size() * sizeof(S); ++k) ok &= (k >= lo && k < hi) || raw[k] == 0x5A;
    ok &= st.spmBytes <= dsa::SCRATCHPAD_SAFE_WATERLINE;
    // Rows that end on DMA blocks, on block-aligned bases, never need a padded transfer
    if (offset == 0 && uint64_t(D) * sizeof(S) % dsa::DMA_ALIGN_BYTES == 0) ok &= st.padTransfers == 0;
    if (!ok) std::printf("  maxErr=%.3g pads=%llu spm=%zu\n", maxErr, (unsigned long long)st.padTransfers, st.spmBytes);
    Check(ok, msg);
}

template <class C>
void RunAll(std::mt19937& rng) {
    const uint32_t shapes[][2] = {
        {1, 64}, {7, 200}, {128, 256}, {768, 192}, {8, 32768}, {1536, 576}, {3, 100}, {5, 7}, {2, 1000},
        {17, 4097}, {41, 5000}, {1, 70001}, {3, 100003}, {40, 3072}, {130, 128}, {1, 1u << 20},
    };
    for (const auto& s : shapes) {
        const bool big = uint64_t(s[0]) * s[1] > 300000;
        for (int variant = 0; variant < 3; ++variant) {
            for (int pc = 0; pc < (big ? 1 : 4); ++pc) {
                for (uint32_t offset : {0u, 1u}) {
                    if (big && offset) continue;
                    RunShape<C>(s[0], s[1], variant, !(pc & 1), !(pc & 2), offset, rng);
                }
            }
        }
    }
}

} // namespace

int main() {
    std::mt19937 rng(7);
    CheckProfilePlans();
    RunAll<F32>(rng);
    RunAll<F16>(rng);
    RunAll<BF16>(rng);
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
