// Target-side checks: the 40-core plans obey the hardware laws, the planner's vector-cycle
// model equals the runtime's cycle count, and the DAE pipeline that executes the plans
// (dsa_runtime, one OpenMP thread per simulated core) is exact, free of scalar stalls,
// sanitizer-clean, claims exactly the planned scratchpad, and only pads DMA transfers where a
// row does not end on a 32-byte block.
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

const char* ModeName(TilingMode m) {
    return m == TilingMode::SPLIT_D ? "split" : m == TilingMode::SPLIT_COLUMNS ? "band" : "rows";
}

// Vector work of one core in the runtime's cycle model (scalar stalls and barriers excluded)
uint64_t VectorCycles(const dsa::HardwareCycleTracker& t) {
    return t.vAddCycles + t.vMulCycles + t.vCastCycles + t.vBlockReduceCycles + t.vWholeReduceCycles + t.vRsqrtCycles;
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
// 1. Plan invariants: 191 KB, 32-byte units and tiles, one-unit balance, feasible layouts
// -----------------------------------------------------------------------------
void CheckPlan(const TilingConfig& t, uint32_t M, uint32_t D, uint32_t s, const char* what) {
    const HardwareModel hw = HardwareModel::Target();
    const uint32_t q = dsa::DMA_ALIGN_BYTES / s;
    const uint64_t total = static_cast<uint64_t>(M) * D;
    char msg[320];
    std::snprintf(msg, sizeof msg, "%s: M=%u D=%u s=%u mode=%s blocks=%u tileRows=%u tile=%u pitch=%u rep=%u spm=%u", what, M, D, s,
                  ModeName(t.mode), t.blocks, t.tileRows, t.tileElems, t.pitch, t.repRows, t.layout.Total());
    Check(std::isfinite(t.modelNs) && t.tileElems > 0, msg);
    Check(t.blocks >= 1 && t.blocks <= dsa::MAX_HARDWARE_CORES, msg);
    Check(t.layout.Total() <= dsa::SCRATCHPAD_SAFE_WATERLINE, msg);
    Check(t.unitElems * s % dsa::DMA_ALIGN_BYTES == 0 || t.mode == TilingMode::ROW_PARALLEL, msg);  // Split units: DMA blocks
    uint64_t lo = ~0ull, hi = 0;  // Balanced to one unit
    for (uint32_t b = 0; b < t.blocks; ++b) {
        const uint64_t e0 = std::min(t.units * b / t.blocks * t.unitElems, total);
        const uint64_t e1 = std::min(t.units * (b + 1) / t.blocks * t.unitElems, total);
        lo = std::min(lo, e1 - e0);
        hi = std::max(hi, e1 - e0);
    }
    Check(hi - lo <= t.unitElems, msg);
    if (t.mode == TilingMode::ROW_PARALLEL && t.tileRows) {  // Row tiles: whole DMA-aligned row groups
        Check(t.pitch == D && t.tileRows % (AdaptiveTiler::RowUnit(D, s, hw) / D) == 0, msg);
        Check(t.layout.Total() == AdaptiveTiler::RowLayout(t.tileRows, D, s, t.repRows).Total(), msg);
    } else if (t.mode == TilingMode::SPLIT_COLUMNS) {  // Band: rows on the grid, band rows fit the scratch chunk
        Check(D % q == 0 && t.pitch % q == 0 && t.pitch <= AdaptiveTiler::TMP_FLOATS && t.zResident == M * t.pitch, msg);
    } else {
        Check(t.tileRows == 0 && t.tileElems % q == 0, msg);  // Column tiles: interior boundaries on the grid
    }
    if (t.tileRows) Check(t.repRows >= 1 && t.repRows <= t.tileRows && (t.repRows == 1 || t.pitch % 8 == 0), msg);
}

void CheckProfilePlans() {
    struct Profile { const char* name; uint32_t M, D, s; } profiles[] = {
        {"P01", 1, 64, 2}, {"P02", 7, 200, 4}, {"P03", 128, 256, 4}, {"P04", 768, 192, 2}, {"P05", 8, 32768, 2},
        {"P06", 1536, 576, 2}, {"P07", 10240, 400, 2}, {"P08", 10240, 512, 2}, {"P09", 4096, 1536, 4},
        {"P10", 8192, 1024, 2}, {"P11", 4096, 3072, 4}, {"P12", 4096, 4096, 2}, {"P13", 10240, 3072, 2},
        {"P14", 2097152, 128, 2}, {"P15", 115000, 8192, 2}};
    const HardwareModel hw = HardwareModel::Target();
    for (const auto& p : profiles) {
        const TilingConfig t = AdaptiveTiler::Plan(p.M, p.D, p.s, hw);
        CheckPlan(t, p.M, p.D, p.s, p.name);
        Check(t.blocks == std::min<uint64_t>(dsa::MAX_HARDWARE_CORES, t.units), p.name);  // Saturates the cores
    }
    // P05: 16384 blocks of 32 bytes over 40 cores -> 409 or 410 each, column-block-major: every
    // row lands on 5 cores and each core's band of gamma/beta is at most 52 blocks
    const TilingConfig p5 = AdaptiveTiler::Plan(8, 32768, 2, hw);
    Check(p5.mode == TilingMode::SPLIT_COLUMNS && p5.blocks == 40 && p5.unitElems == 16 && p5.units == 16384 &&
          p5.pitch == 52 * 16, "P05 column-band plan");
    // P04: two pipelined batches of 10 rows per core; P08: >= 24-row batches in 191 KB
    Check(AdaptiveTiler::Plan(768, 192, 2, hw).tileRows == 10, "P04 tile rows");
    Check(AdaptiveTiler::Plan(10240, 512, 2, hw).tileRows >= 24, "P08 tile rows");
}

// Every shape and every forced decomposition: feasible plans obey the laws, infeasible bands
// say so (tileElems == 0) instead of overflowing
void CheckPlannerScan() {
    const HardwareModel hw = HardwareModel::Target();
    for (uint32_t s : {2u, 4u}) {
        for (uint32_t M : {1u, 2u, 3u, 5u, 8u, 13u, 39u, 40u, 41u, 100u, 777u, 4096u}) {
            for (uint32_t D = 1; D <= 120000; D = D < 40 ? D + 1 : D * 9 / 8 + 5) {
                CheckPlan(AdaptiveTiler::Plan(M, D, s, hw), M, D, s, "plan");
                for (TilingMode mode : {TilingMode::ROW_PARALLEL, TilingMode::SPLIT_D, TilingMode::SPLIT_COLUMNS}) {
                    const TilingConfig t = AdaptiveTiler::Build(M, D, s, hw, mode, true);
                    if (t.tileElems) CheckPlan(t, M, D, s, "forced");
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 2. DAE execution vs FP64 reference
// -----------------------------------------------------------------------------
template <class C>
void RunPlan(uint32_t M, uint32_t D, const TilingConfig& plan, bool hasGamma, bool hasBias, uint32_t offset,
             std::mt19937& rng, const char* label) {
    using S = typename C::S;
    const size_t N = size_t(M) * D;
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    HostBuffer<S> x1(N, offset), x2(N, offset), g(D, offset), b(D, offset), y(N, offset);
    for (size_t i = 0; i < N; ++i) { x1.p[i] = Enc<C>(dist(rng)); x2.p[i] = Enc<C>(dist(rng)); }
    for (uint32_t j = 0; j < D; ++j) { g.p[j] = Enc<C>(1.0f + 0.5f * dist(rng)); b.p[j] = Enc<C>(0.1f * dist(rng)); }
    std::memset(y.mem.data(), 0x5A, y.mem.size() * sizeof(S));

    DaeStats st;
    char msg[320];
    std::snprintf(msg, sizeof msg, "%s %s M=%u D=%u gamma=%d bias=%d offset=%u blocks=%u mode=%s tileRows=%u tile=%u pitch=%u rep=%u zRes=%u",
                  Name<C>(), label, M, D, hasGamma, hasBias, offset, plan.blocks, ModeName(plan.mode), plan.tileRows,
                  plan.tileElems, plan.pitch, plan.repRows, plan.zResident);
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
    ok &= st.spmBytes == plan.layout.Total();  // Claims exactly the planned (<= 191 KB) scratchpad
    ok &= st.scalarStalls == 0;                // No GetValue V->S stall anywhere
    ok &= st.barriers == (plan.mode == TilingMode::ROW_PARALLEL ? 0u : 1u);
    // Rows that end on DMA blocks, on block-aligned bases, never need a padded transfer
    if (offset == 0 && uint64_t(D) * sizeof(S) % dsa::DMA_ALIGN_BYTES == 0) ok &= st.padTransfers == 0;
    // The planner's cycle model is the runtime's count (it assumes both gamma and beta)
    if (hasGamma && hasBias) ok &= VectorCycles(st.busiest) == plan.modelCycles;
    if (!ok) {
        std::printf("  maxErr=%.3g pads=%llu spm=%zu stalls=%llu barriers=%llu cycles=%llu model=%llu\n", maxErr,
                    (unsigned long long)st.padTransfers, st.spmBytes, (unsigned long long)st.scalarStalls,
                    (unsigned long long)st.barriers, (unsigned long long)VectorCycles(st.busiest),
                    (unsigned long long)plan.modelCycles);
    }
    Check(ok, msg);
}

template <class C>
void RunAll(std::mt19937& rng) {
    using S = typename C::S;
    const uint32_t shapes[][2] = {
        {1, 64}, {7, 200}, {128, 256}, {768, 192}, {8, 32768}, {1536, 576}, {3, 100}, {5, 7}, {2, 1000}, {17, 4097},
        {41, 5000}, {1, 70001}, {3, 100003}, {40, 3072}, {130, 128}, {1, 1u << 20}, {8, 4096}, {13, 12288}, {39, 2048},
    };
    const HardwareModel hw = HardwareModel::Target();
    const char* labels[] = {"model", "split", "rows", "band"};
    for (const auto& s : shapes) {
        const bool big = uint64_t(s[0]) * s[1] > 300000;
        for (int variant = 0; variant < 4; ++variant) {
            TilingConfig plan = AdaptiveTiler::Plan(s[0], s[1], sizeof(S), hw);
            if (variant == 1) plan = AdaptiveTiler::Build(s[0], s[1], sizeof(S), hw, TilingMode::SPLIT_D, true);
            if (variant == 2) plan = AdaptiveTiler::Build(s[0], s[1], sizeof(S), hw, TilingMode::ROW_PARALLEL, true);
            if (variant == 3) plan = AdaptiveTiler::Build(s[0], s[1], sizeof(S), hw, TilingMode::SPLIT_COLUMNS, true);
            if (plan.tileElems == 0) continue;  // No feasible band for this shape
            for (int pc = 0; pc < (big ? 1 : 4); ++pc) {
                for (uint32_t offset : {0u, 1u}) {
                    if (big && offset) continue;
                    RunPlan<C>(s[0], s[1], plan, !(pc & 1), !(pc & 2), offset, rng, labels[variant]);
                }
            }
        }
    }
}

// A plan that fills the scratchpad to the byte must run: the kernel may claim nothing the
// layout does not list (an unplanned 64-byte placeholder buffer once overflowed such plans)
void RunWaterlinePlan(std::mt19937& rng) {
    const uint32_t D = 128, B = 91, M = 2 * B * dsa::MAX_HARDWARE_CORES;
    TilingConfig plan = AdaptiveTiler::Build(M, D, 4, HardwareModel::Target(), TilingMode::ROW_PARALLEL, true);
    plan.tileRows = B;
    plan.tileElems = B * D;
    plan.repRows = 1;
    plan.layout = AdaptiveTiler::RowLayout(B, D, 4, 1);
    plan.modelCycles = AdaptiveTiler::ParamCycles(D, 4) + 2 * AdaptiveTiler::TileCycles(B, D, 4, 1, nullptr);  // 2 tiles per core
    Check(plan.layout.Total() == dsa::SCRATCHPAD_SAFE_WATERLINE, "waterline plan is exactly 195584 bytes");
    RunPlan<F32>(M, D, plan, true, true, 0, rng, "waterline");
}

} // namespace

int main() {
    std::mt19937 rng(7);
    CheckProfilePlans();
    CheckPlannerScan();
    RunWaterlinePlan(rng);
    RunAll<F32>(rng);
    RunAll<F16>(rng);
    RunAll<BF16>(rng);
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
