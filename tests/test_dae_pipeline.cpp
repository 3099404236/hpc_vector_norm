// Target-side checks: the 40-core plans obey the hardware laws, the planner's vector-cycle
// model equals the runtime's cycle count, and the DAE pipeline that executes the plans
// (dsa_runtime, one OpenMP thread per simulated core) is exact, free of scalar stalls,
// sanitizer-clean (DAE v1.4 errata included: no egress from VECIN, no aliased folds), claims
// exactly the planned scratchpad, only pads DMA transfers where a row does not end on a
// 32-byte block, and allocates nothing but the simulator's scratchpad.
// Its workers are freestanding: a trap aborts the process (DSA_ASSERT), which the death tests
// check, and the abort report names the run in progress.
#include "hpc_vector_norm.hpp"
#include "kernel_unified.hpp"
#include <omp.h>
#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <random>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

using namespace hpc;

// -----------------------------------------------------------------------------
// Heap probe: every operator new of the process is counted while armed
// -----------------------------------------------------------------------------
namespace {
std::atomic<bool> g_countNew{false};
std::atomic<uint64_t> g_newCalls{0};

void* CountedNew(std::size_t n, std::size_t align) {
    if (g_countNew.load(std::memory_order_relaxed)) g_newCalls.fetch_add(1, std::memory_order_relaxed);
    n = n ? n : 1;
    void* p = align > alignof(std::max_align_t) ? std::aligned_alloc(align, (n + align - 1) / align * align) : std::malloc(n);
    if (!p) throw std::bad_alloc();
    return p;
}
} // namespace

void* operator new(std::size_t n) { return CountedNew(n, 0); }
void* operator new[](std::size_t n) { return CountedNew(n, 0); }
void* operator new(std::size_t n, std::align_val_t a) { return CountedNew(n, static_cast<std::size_t>(a)); }
void* operator new[](std::size_t n, std::align_val_t a) { return CountedNew(n, static_cast<std::size_t>(a)); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

namespace {

int g_checks = 0, g_failures = 0;
char g_running[400] = "";  // The run in progress, reported if a trap aborts the process

void Check(bool ok, const char* what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("FAIL %s\n", what);
    }
}

void ReportAbort(int) {
    const char head[] = "\nAborted during: ";
    ssize_t r = write(STDERR_FILENO, head, sizeof head - 1);
    r = write(STDERR_FILENO, g_running, std::strlen(g_running));
    r = write(STDERR_FILENO, "\n", 1);
    (void)r;
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
    // Egress buffers for every result, the fold partition, and a VECOUT record for split plans
    Check(t.layout.out == t.layout.tile && t.layout.outDepth >= 1 && t.layout.outDepth <= 2 &&
              t.layout.tmp == AdaptiveTiler::SCRATCH_BYTES, msg);
    Check((t.layout.rec != 0) == (t.mode != TilingMode::ROW_PARALLEL), msg);
    uint64_t lo = ~0ull, hi = 0;  // Balanced to one unit
    for (uint32_t b = 0; b < t.blocks; ++b) {
        const uint64_t e0 = std::min(t.units * b / t.blocks * t.unitElems, total);
        const uint64_t e1 = std::min(t.units * (b + 1) / t.blocks * t.unitElems, total);
        lo = std::min(lo, e1 - e0);
        hi = std::max(hi, e1 - e0);
    }
    Check(hi - lo <= t.unitElems, msg);
    if (t.mode == TilingMode::ROW_PARALLEL && t.tileRows) {  // Row tiles: whole DMA-aligned row groups
        const uint32_t p = static_cast<uint32_t>(AdaptiveTiler::RowUnit(D, s, hw) / D);
        Check(t.pitch == D && t.tileRows % p == 0 && t.headRows % p == 0 && t.tailRows % p == 0, msg);
        Check(t.headRows < t.tileRows && t.tailRows < t.tileRows && t.layout.depth >= 2 && t.layout.depth <= 4, msg);
        Check(t.layout.Total() == AdaptiveTiler::RowLayout(t.tileRows, D, s, t.repRows, t.layout.depth, t.layout.outDepth).Total(), msg);
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
    // P04: its 20 rows stream through several tiles, one in flight while another computes;
    // P08: >= 24 rows in flight within 191 KB (Challenges 2 and 3)
    const TilingConfig p4 = AdaptiveTiler::Plan(768, 192, 2, hw), p8 = AdaptiveTiler::Plan(10240, 512, 2, hw);
    Check(p4.tileRows < 20 && p4.layout.depth >= 2, "P04 pipelined tiles");
    Check(p8.tileRows * p8.layout.depth >= 24 && p8.layout.Total() <= dsa::SCRATCHPAD_SAFE_WATERLINE, "P08 rows in flight");
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

    char msg[320];
    std::snprintf(msg, sizeof msg, "%s %s M=%u D=%u gamma=%d bias=%d offset=%u blocks=%u mode=%s tileRows=%u tile=%u pitch=%u rep=%u zRes=%u",
                  Name<C>(), label, M, D, hasGamma, hasBias, offset, plan.blocks, ModeName(plan.mode), plan.tileRows,
                  plan.tileElems, plan.pitch, plan.repRows, plan.zResident);
    std::snprintf(g_running, sizeof g_running, "%s", msg);
    const DaeStats st = DaePipeline<C>::Execute(x1.p, x2.p, hasGamma ? g.p : nullptr, hasBias ? b.p : nullptr, y.p, M, D, 1e-6f, plan);
    g_running[0] = '\0';
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
    // The planner's cycle model is the runtime's count (it assumes both gamma and beta), and on
    // aligned tensors its timeline is the runtime's: the same finish time, in every mode
    if (hasGamma && hasBias) ok &= VectorCycles(st.busiest) == plan.modelCycles;
    const double modeled = plan.modelNs * dsa::CLOCK_GHZ;
    if (hasGamma && hasBias && offset == 0) ok &= std::fabs(st.timeline.finish - modeled) <= 1e-9 * st.timeline.finish;
    // The timeline's own accounting: finish = busy + fill + drain + barrier + mismatch, above the bound
    const dsa::TimelineSummary& tl = st.timeline;
    ok &= std::fabs(tl.finish - std::max(tl.vectorBusy, tl.dmaBusy) - tl.fill - tl.drain - tl.barrier - tl.mismatch) <= 1e-6 * tl.finish;
    ok &= tl.fill >= -1e-9 && tl.drain >= -1e-9 && tl.mismatch >= -1e-6 * tl.finish && tl.finish >= tl.LowerBound() - 1e-6;
    ok &= tl.finish >= tl.LatencyFloor() - 1e-6 * tl.finish;  // The floor is a bound: no run beats it
    if (!ok) {
        std::printf("  maxErr=%.3g pads=%llu spm=%zu stalls=%llu barriers=%llu cycles=%llu model=%llu finish=%.2f modeled=%.2f floor=%.2f\n",
                    maxErr, (unsigned long long)st.padTransfers, st.spmBytes, (unsigned long long)st.scalarStalls,
                    (unsigned long long)st.barriers, (unsigned long long)VectorCycles(st.busiest),
                    (unsigned long long)plan.modelCycles, tl.finish, modeled, tl.LatencyFloor());
    }
    Check(ok, msg);
}

template <class C>
void RunAll(std::mt19937& rng) {
    using S = typename C::S;
    const uint32_t shapes[][2] = {
        {1, 64}, {7, 200}, {128, 256}, {768, 192}, {8, 32768}, {1536, 576}, {3, 100}, {5, 7}, {2, 1000}, {17, 4097},
        {41, 5000}, {1, 70001}, {3, 100003}, {40, 3072}, {130, 128}, {1, 1u << 20}, {8, 4096}, {13, 12288}, {39, 2048},
        {65536, 24},  // Row tiles of 137-138 rows: more than one row group per tile
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
// layout does not list (an unplanned 64-byte placeholder buffer once overflowed such plans).
// FP32, 72-row tiles: X1/X2 double-buffered and one egress buffer (5 x 36,864 B), 9,216 B of
// scratch and 2,048 B of gamma/beta replicated twice
void RunWaterlinePlan(std::mt19937& rng) {
    const uint32_t D = 128, B = 72, M = 2 * B * dsa::MAX_HARDWARE_CORES;
    const HardwareModel hw = HardwareModel::Target();
    TilingConfig plan = AdaptiveTiler::Build(M, D, 4, hw, TilingMode::ROW_PARALLEL, true);
    AdaptiveTiler::ApplyRowPipe(plan, M, D, 4, hw, {B, 0, 0, 2, 2, 1, false, false});  // 2 tiles of 72 rows per core
    Check(plan.layout.Total() == dsa::SCRATCHPAD_SAFE_WATERLINE, "waterline plan is exactly 195584 bytes");
    RunPlan<F32>(M, D, plan, true, true, 0, rng, "waterline");
}

// Every scheduling choice of row tiles, forced one combination at a time: queue depth, egress
// buffers, head and tail tiles, gamma/beta first or second, early input loads. The kernel must
// follow each schedule exactly: right output, and the model's cycle count and finish time (RunPlan)
template <class C>
void RunScheduleSweep(std::mt19937& rng) {
    const HardwareModel hw = HardwareModel::Target();
    const uint32_t M = 22 * dsa::MAX_HARDWARE_CORES, D = 192, s = sizeof(typename C::S);  // 22 rows per core
    TilingConfig plan = AdaptiveTiler::Build(M, D, s, hw, TilingMode::ROW_PARALLEL, true);
    for (uint32_t depth = 2; depth <= 4; ++depth) {
        for (const uint32_t outDepth : {1u, 2u}) {
            for (const uint32_t head : {0u, 1u, 4u}) {
                for (const uint32_t tail : {0u, 2u}) {
                    for (int order = 0; order < 4; ++order) {
                        const bool first = order & 1, early = order & 2;
                        AdaptiveTiler::ApplyRowPipe(plan, M, D, s, hw, {6, head, tail, 3, depth, outDepth, first, early});
                        char label[128];
                        std::snprintf(label, sizeof label, "schedule depth=%u outDepth=%u head=%u tail=%u paramsFirst=%d earlyLoads=%d",
                                      depth, outDepth, head, tail, first, early);
                        RunPlan<C>(M, D, plan, true, true, 0, rng, label);
                    }
                }
            }
        }
    }
}

// Tiles of many row groups (a worker keeps the sums of ROW_GROUP rows at a time): row tiles as
// large as the scratchpad admits, and a column band run as one tile per core
template <class C>
void RunWidePlans(std::mt19937& rng) {
    using S = typename C::S;
    const HardwareModel hw = HardwareModel::Target();
    struct Case { uint32_t M, D; TilingMode mode; } cases[] = {
        {12000, 8, TilingMode::ROW_PARALLEL},     // 300-row tiles: groups of 128, 128, 44
        {65536, 24, TilingMode::ROW_PARALLEL},    // Squares chunks of 85 rows straddle the groups
        {300, 640, TilingMode::SPLIT_COLUMNS}};   // One 300-row band tile per core
    for (const Case& c : cases) {
        TilingConfig plan = AdaptiveTiler::Build(c.M, c.D, sizeof(S), hw, c.mode, true);
        if (c.mode == TilingMode::ROW_PARALLEL) {  // The largest double-buffered tile, up to the core's rows
            const uint32_t rows = (c.M + plan.blocks - 1) / plan.blocks;
            uint32_t B = 1;
            while (B < rows && AdaptiveTiler::RowLayout(B + 1, c.D, sizeof(S), 1, 2, 1).Total() <= hw.spmBytes) ++B;
            AdaptiveTiler::ApplyRowPipe(plan, c.M, c.D, sizeof(S), hw, {B, 0, 0, 1, 2, 1, false, false});
        } else {
            AdaptiveTiler::ApplyBandPipe(plan, c.M, c.D, sizeof(S), hw, {c.M, 1, 2, 2});
        }
        char msg[160];
        std::snprintf(msg, sizeof msg, "%s wide plan M=%u D=%u mode=%s tileRows=%u", Name<C>(), c.M, c.D, ModeName(plan.mode), plan.tileRows);
        Check(plan.tileElems > 0 && plan.tileRows > 2 * AdaptiveTiler::ROW_GROUP, msg);
        RunPlan<C>(c.M, c.D, plan, true, true, 0, rng, "wide");
        RunPlan<C>(c.M, c.D, plan, false, true, 1, rng, "wide");
    }
}

// -----------------------------------------------------------------------------
// 3. Zero-allocation execution [ARCH CHALLENGE 6]. With a caller-owned workspace, the only heap
// allocations during DaePipeline::Execute are the simulator's scratchpad buffers (dsa_runtime
// backs each TPipe buffer with a std::vector); the coordinator and the workers allocate nothing.
// The coordinator's own workspace gives bit-identical results.
// -----------------------------------------------------------------------------
uint64_t SimulatorBuffers(const DaeLayout& L) {  // TPipe::InitBuffer blocks of one core
    return L.depth * (L.paramQueue ? 3 : 2) + L.outDepth + 1 + (L.z ? 1 : 0) + (L.params ? 1 : 0) + (L.resident ? 1 : 0) +
           (L.misc ? 1 : 0) + (L.rec ? 1 : 0);
}

template <class C>
void CheckAllocations(std::mt19937& rng) {
    using S = typename C::S;
    const HardwareModel hw = HardwareModel::Target();
    struct Case { uint32_t M, D; TilingMode mode; } cases[] = {
        {768, 192, TilingMode::ROW_PARALLEL},  {65536, 24, TilingMode::ROW_PARALLEL}, {1, 70001, TilingMode::ROW_PARALLEL},
        {8, 32768, TilingMode::SPLIT_COLUMNS}, {17, 4097, TilingMode::SPLIT_D},       {3, 100003, TilingMode::SPLIT_D}};
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (const Case& c : cases) {
        const TilingConfig plan = AdaptiveTiler::Build(c.M, c.D, sizeof(S), hw, c.mode, true);
        const size_t N = size_t(c.M) * c.D;
        HostBuffer<S> x1(N, 0), x2(N, 0), g(c.D, 0), b(c.D, 0), y(N, 0), y2(N, 0);
        for (size_t i = 0; i < N; ++i) { x1.p[i] = Enc<C>(dist(rng)); x2.p[i] = Enc<C>(dist(rng)); }
        for (uint32_t j = 0; j < c.D; ++j) { g.p[j] = Enc<C>(1.0f + 0.5f * dist(rng)); b.p[j] = Enc<C>(0.1f * dist(rng)); }
        const size_t bytes = DaePipeline<C>::WorkspaceBytes(plan, c.M);
        float* workspace = bytes ? static_cast<float*>(std::aligned_alloc(64, bytes)) : nullptr;
        char msg[200];
        std::snprintf(msg, sizeof msg, "%s zero-allocation M=%u D=%u mode=%s workspace=%zu B", Name<C>(), c.M, c.D, ModeName(plan.mode), bytes);
        std::snprintf(g_running, sizeof g_running, "%s", msg);
        g_newCalls = 0;
        g_countNew = true;
        const DaeStats st = DaePipeline<C>::Execute(x1.p, x2.p, g.p, b.p, y.p, c.M, c.D, 1e-6f, plan, workspace, bytes);
        g_countNew = false;
        const uint64_t news = g_newCalls;
        const DaeStats own = DaePipeline<C>::Execute(x1.p, x2.p, g.p, b.p, y2.p, c.M, c.D, 1e-6f, plan);
        g_running[0] = '\0';
        std::free(workspace);
        Check(plan.tileElems > 0 && (bytes == 0) == (plan.mode == TilingMode::ROW_PARALLEL), msg);
        Check(news == plan.blocks * SimulatorBuffers(plan.layout), msg);
        Check(std::memcmp(y.p, y2.p, N * sizeof(S)) == 0 && st.vectorCycles == own.vectorCycles && st.dmaBytes == own.dmaBytes &&
                  st.spmBytes == own.spmBytes && VectorCycles(st.busiest) == plan.modelCycles, msg);
        if (news != plan.blocks * SimulatorBuffers(plan.layout)) {
            std::printf("  operator new calls: %llu, simulator buffers: %llu\n", (unsigned long long)news,
                        (unsigned long long)(plan.blocks * SimulatorBuffers(plan.layout)));
        }
    }
}

// -----------------------------------------------------------------------------
// 4. Freestanding traps: a broken coordinator or worker invariant is a DSA_ASSERT, which
// reports "[DSA Hardware Trap]" and aborts instead of throwing. Each case runs in a child
// process: this binary, re-executed with --trap <case>.
// -----------------------------------------------------------------------------
struct TrapCase { const char* name; const char* report; };
const TrapCase kTraps[] = {
    {"plan", "needs a feasible plan"},              // Coordinator: a host plan has no DAE tiles
    {"workspace", "reduction workspace"},           // Coordinator: workspace off the 64-byte grid
    {"team", "core count"},                         // Worker: nested region, one thread for 40 cores
    {"band", "column band wider than the plan"}};   // Worker: band pitch below the band width

int RunTrapCase(const char* name) {
    const uint32_t M = 8, D = 32768;  // P05: the column band on 40 cores
    std::vector<uint16_t> x(size_t(M) * D, 0x3C00), y(size_t(M) * D);
    TilingConfig plan = AdaptiveTiler::Plan(M, D, 2, HardwareModel::Target());
    const size_t bytes = DaePipeline<F16>::WorkspaceBytes(plan, M);
    std::vector<float> workspace(bytes / sizeof(float) + 32);
    auto run = [&] { DaePipeline<F16>::Execute(x.data(), x.data(), nullptr, nullptr, y.data(), M, D, 1e-6f, plan); };
    if (!std::strcmp(name, "plan")) {
        plan = AdaptiveTiler::Plan(M, D, 2, HardwareModel::Host(4));
        run();
    } else if (!std::strcmp(name, "workspace")) {
        float* misaligned = reinterpret_cast<float*>((reinterpret_cast<uintptr_t>(workspace.data()) + 63) / 64 * 64) + 1;
        DaePipeline<F16>::Execute(x.data(), x.data(), nullptr, nullptr, y.data(), M, D, 1e-6f, plan, misaligned, bytes);
    } else if (!std::strcmp(name, "team")) {
        omp_set_max_active_levels(1);
        #pragma omp parallel num_threads(2)
        {
            if (omp_get_thread_num() == 0) run();
        }
    } else if (!std::strcmp(name, "band")) {
        plan.pitch = 16;
        run();
    }
    return 0;  // Nothing trapped
}

void CheckTraps(const char* self) {
    for (const TrapCase& t : kTraps) {
        const std::string cmd = std::string("'") + self + "' --trap " + t.name + " 2>&1";
        FILE* pipe = popen(cmd.c_str(), "r");
        std::string out;
        char buf[256];
        while (pipe && std::fgets(buf, sizeof buf, pipe)) out += buf;
        const int status = pipe ? pclose(pipe) : -1;
        const bool aborted = status != -1 && ((WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT) ||
                                              (WIFEXITED(status) && WEXITSTATUS(status) == 128 + SIGABRT));
        const bool reported = out.find("[DSA Hardware Trap]: [DaePipeline]") != std::string::npos && out.find(t.report) != std::string::npos;
        char msg[160];
        std::snprintf(msg, sizeof msg, "DSA_ASSERT trap '%s' (status %d)", t.name, status);
        Check(aborted && reported, msg);
        if (!(aborted && reported)) std::printf("  child output: %s\n", out.c_str());
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--trap") == 0) return RunTrapCase(argv[2]);
    std::signal(SIGABRT, ReportAbort);
    std::mt19937 rng(7);
    CheckProfilePlans();
    CheckPlannerScan();
    CheckTraps(argv[0]);
    RunWaterlinePlan(rng);
    CheckAllocations<F32>(rng);
    CheckAllocations<F16>(rng);
    CheckAllocations<BF16>(rng);
    RunWidePlans<F32>(rng);
    RunWidePlans<F16>(rng);
    RunWidePlans<BF16>(rng);
    RunScheduleSweep<F32>(rng);
    RunScheduleSweep<F16>(rng);
    RunAll<F32>(rng);
    RunAll<F16>(rng);
    RunAll<BF16>(rng);
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
