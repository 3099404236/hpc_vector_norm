// Target-side checks: the 40-core plans obey the hardware laws, the planner's vector-cycle
// model equals the runtime's cycle count, and the DAE pipeline that executes the plans
// (dsa_runtime, one OpenMP thread per simulated core) is exact, free of scalar stalls,
// sanitizer-clean (DAE v1.4 errata included: no egress from VECIN, no aliased folds; v1.5: no
// unpadded or aliased ReduceSum, no undersized Brcb; the target's semantics: ReduceSum into
// block-aligned slots, Brcb as a block broadcast, strided masks of one repeat, no scalar read of
// the scratchpad, nothing consumed that the kernel did not write), claims exactly the planned
// scratchpad, only pads DMA transfers where a row does not end on a 32-byte block, and allocates
// nothing but the simulator's scratchpad. No kernel takes a queue step: its buffers are static
// rings. Workers are freestanding: a trap aborts the process (DSA_ASSERT), which the death tests
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
// A shape's rows on the row decomposition, each core's rows in a single tile: the schedule the
// removed direct kernel ran [Challenge 8], now just the one-tile row plan. gamma/beta load after
// X1/X2, as the direct kernel issued them.
TilingConfig OneTile(uint32_t M, uint32_t D, uint32_t s) {
    const HardwareModel hw = HardwareModel::Target();
    TilingConfig plan = AdaptiveTiler::Build(M, D, s, hw, TilingMode::ROW_PARALLEL, true);
    const uint32_t rows = static_cast<uint32_t>((AdaptiveTiler::MaxLoad(uint64_t(M) * D, plan.unitElems, plan.blocks) + D - 1) / D);
    AdaptiveTiler::ApplyRowPipe(plan, M, D, s, hw, {rows, 0, 0, 2, 1, false, false});
    return plan;
}

void CheckPlan(const TilingConfig& t, uint32_t M, uint32_t D, uint32_t s, const char* what) {
    const uint32_t q = dsa::DMA_ALIGN_BYTES / s;
    const uint64_t total = static_cast<uint64_t>(M) * D;
    char msg[320];
    std::snprintf(msg, sizeof msg, "%s: M=%u D=%u s=%u mode=%s blocks=%u tileRows=%u tile=%u pitch=%u spm=%u", what, M, D, s,
                  ModeName(t.mode), t.blocks, t.tileRows, t.tileElems, t.pitch, t.layout.Total());
    Check(std::isfinite(t.modelNs) && t.tileElems > 0, msg);
    Check(t.blocks >= 1 && t.blocks <= dsa::MAX_HARDWARE_CORES, msg);
    Check(t.layout.Total() <= dsa::SCRATCHPAD_SAFE_WATERLINE, msg);
    Check(t.unitElems * s % dsa::DMA_ALIGN_BYTES == 0 || t.mode == TilingMode::ROW_PARALLEL, msg);  // Split units: DMA blocks
    // Egress buffers for every result, the scratch buffer with its reduction partitions, and a VECOUT
    // record for split plans
    Check(t.layout.out == t.layout.tile && t.layout.outDepth >= 1 && t.layout.outDepth <= 2 &&
              t.layout.tmp == AdaptiveTiler::SCRATCH_BYTES, msg);
    Check((t.layout.rec != 0) == (t.mode != TilingMode::ROW_PARALLEL), msg);
    // Every core's rows in one tile is a row plan too (the schedule of the former direct kernel): the
    // planner keeps a plan no slower than it
    const uint64_t rows = (AdaptiveTiler::MaxLoad(total, t.unitElems, t.blocks) + D - 1) / D;
    if (t.mode == TilingMode::ROW_PARALLEL && AdaptiveTiler::RowLayout(rows, D, s).Total() <= dsa::SCRATCHPAD_SAFE_WATERLINE) {
        Check(t.modelNs <= OneTile(M, D, s).modelNs * (1 + 1e-12), msg);
    }
    uint64_t lo = ~0ull, hi = 0;  // Balanced to one unit
    for (uint32_t b = 0; b < t.blocks; ++b) {
        const uint64_t e0 = std::min(t.units * b / t.blocks * t.unitElems, total);
        const uint64_t e1 = std::min(t.units * (b + 1) / t.blocks * t.unitElems, total);
        lo = std::min(lo, e1 - e0);
        hi = std::max(hi, e1 - e0);
    }
    Check(hi - lo <= t.unitElems, msg);
    if (t.mode == TilingMode::ROW_PARALLEL && t.tileRows) {  // Row tiles: whole rows at the 32-byte row pitch
        Check(t.unitElems == D && t.pitch == AdaptiveTiler::RowPitch(D, s) && t.pitch * s % dsa::DMA_ALIGN_BYTES == 0 &&
                  t.tileElems == t.tileRows * t.pitch, msg);
        Check(t.headRows < t.tileRows && t.tailRows < t.tileRows && t.layout.depth >= 2 && t.layout.depth <= 4, msg);
        Check(t.layout.Total() == AdaptiveTiler::RowLayout(t.tileRows, D, s, t.layout.depth, t.layout.outDepth).Total(), msg);
    } else if (t.mode == TilingMode::SPLIT_COLUMNS) {
        // Band: rows on the grid, band rows within the strided form's reach, tiles of whole record blocks
        Check(D % q == 0 && t.pitch % q == 0 && AdaptiveTiler::StridedRows(t.pitch) && t.zResident == M * t.pitch &&
                  (t.tileRows % 8 == 0 || t.tileRows == M), msg);
    } else {
        Check(t.tileRows == 0 && t.tileElems % q == 0, msg);  // Column tiles: interior boundaries on the grid
    }
}

// The target benchmark's 15 shapes (docs/TARGET_MEASUREMENTS.md, section 9), with the best known
// time on the target in us
struct BenchCase { const char* name; uint32_t M, D, s; double bestUs; };
const BenchCase kCases[] = {
    {"C1", 1, 64, 2, 1.70},         {"C2", 7, 197, 4, 2.21},        {"C3", 128, 256, 4, 2.47},       {"C4", 766, 193, 2, 6.66},
    {"C5", 8, 32768, 2, 5.20},      {"C6", 1508, 577, 2, 9.62},     {"C7", 3104, 397, 2, 13.90},     {"C8", 10240, 512, 2, 30.16},
    {"C9", 4080, 1536, 4, 68.20},   {"C10", 8192, 1024, 2, 46.63},  {"C11", 3752, 3083, 4, 131.16},  {"C12", 3392, 4096, 2, 76.35},
    {"C13", 10432, 3079, 4, 392.72}, {"C14", 3440640, 128, 2, 3750.12}, {"C15", 117504, 8192, 2, 8321.94}};

// The benchmark's plans obey the laws and use min(40, units) cores. Which schedule each gets is the
// measured model's to decide; the one decision the measurements settle on their own: C5's 8 rows
// split over all 40 cores, since 8 cores reach only 8/27 of the bandwidth (section 4)
void CheckProfilePlans() {
    const HardwareModel hw = HardwareModel::Target();
    for (const BenchCase& c : kCases) {
        const TilingConfig t = AdaptiveTiler::Plan(c.M, c.D, c.s, hw);
        CheckPlan(t, c.M, c.D, c.s, c.name);
        Check(t.blocks == std::min<uint64_t>(dsa::MAX_HARDWARE_CORES, t.units), c.name);  // Saturates the cores
    }
    const TilingConfig c5 = AdaptiveTiler::Plan(8, 32768, 2, hw);
    Check(c5.mode != TilingMode::ROW_PARALLEL && c5.blocks == 40 && c5.unitElems == 16 && c5.units == 16384, "C5 splits over 40 cores");
}

// Every shape and every forced decomposition: feasible plans obey the laws, infeasible bands
// say so (tileElems == 0) instead of overflowing
void CheckPlannerScan() {
    const HardwareModel hw = HardwareModel::Target();
    // Every tensor of at most 1,024 bytes (the shares the direct kernel took [Challenge 8]): a
    // feasible plan, no slower than every core's rows in one tile (CheckPlan)
    for (uint32_t s : {2u, 4u}) {
        for (uint32_t D = 1; D * s <= 1024; ++D) {
            for (uint32_t M = 1; M * D * s <= 1024; ++M) CheckPlan(AdaptiveTiler::Plan(M, D, s, hw), M, D, s, "tiny");
        }
    }
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
DaeStats RunPlan(uint32_t M, uint32_t D, const TilingConfig& plan, bool hasGamma, bool hasBias, uint32_t offset,
                 std::mt19937& rng, const char* label) {
    using S = typename C::S;
    const size_t N = size_t(M) * D;
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    HostBuffer<S> x1(N, offset), x2(N, offset), g(D, offset), b(D, offset), y(N, offset);
    for (size_t i = 0; i < N; ++i) { x1.p[i] = Enc<C>(dist(rng)); x2.p[i] = Enc<C>(dist(rng)); }
    for (uint32_t j = 0; j < D; ++j) { g.p[j] = Enc<C>(1.0f + 0.5f * dist(rng)); b.p[j] = Enc<C>(0.1f * dist(rng)); }
    std::memset(y.mem.data(), 0x5A, y.mem.size() * sizeof(S));

    char msg[320];
    std::snprintf(msg, sizeof msg, "%s %s M=%u D=%u gamma=%d bias=%d offset=%u blocks=%u mode=%s tileRows=%u tile=%u pitch=%u zRes=%u",
                  Name<C>(), label, M, D, hasGamma, hasBias, offset, plan.blocks, ModeName(plan.mode), plan.tileRows,
                  plan.tileElems, plan.pitch, plan.zResident);
    std::snprintf(g_running, sizeof g_running, "%s", msg);
    const DaeStats st = DaePipeline<C>::Execute(x1.p, x2.p, hasGamma ? g.p : nullptr, hasBias ? b.p : nullptr, y.p, M, D, 1e-6f, plan);
    g_running[0] = '\0';
    bool ok = true;
    double maxErr = 0.0;
    // FP64 reference in the target's convention: Z = X1 + X2 (bias never enters the sum of
    // squares), Y = Z / sigma * gamma + bias
    for (uint32_t i = 0; i < M; ++i) {
        double ss = 0.0;
        for (uint32_t j = 0; j < D; ++j) {
            const double z = Dec<C>(x1.p[size_t(i) * D + j]) + Dec<C>(x2.p[size_t(i) * D + j]);
            ss += z * z;
        }
        const double inv = 1.0 / std::sqrt(ss / D + 1e-6);
        for (uint32_t j = 0; j < D; ++j) {
            const double z = Dec<C>(x1.p[size_t(i) * D + j]) + Dec<C>(x2.p[size_t(i) * D + j]);
            const double ref = z * inv * (hasGamma ? Dec<C>(g.p[j]) : 1.0) + (hasBias ? Dec<C>(b.p[j]) : 0.0);
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
    ok &= st.queueCycles == 0;                 // No kernel takes a TQue step: static rings and buffers
    ok &= st.barriers == (plan.mode == TilingMode::ROW_PARALLEL ? 0u : 1u);
    // Rows that end on DMA blocks, on block-aligned bases, never need a padded transfer
    if (offset == 0 && uint64_t(D) * sizeof(S) % dsa::DMA_ALIGN_BYTES == 0) ok &= st.padTransfers == 0;
    // The planner's cycle model is the runtime's count (it assumes both gamma and beta), and on
    // aligned tensors its timeline is the runtime's: the same finish time, in every mode. Its time is
    // that timeline after the kernel launch.
    if (hasGamma && hasBias) ok &= VectorCycles(st.busiest) == plan.modelCycles;
    const double modeled = plan.modelFinish;
    if (hasGamma && hasBias && offset == 0) ok &= std::fabs(st.timeline.finish - modeled) <= 1e-9 * st.timeline.finish;
    ok &= std::fabs(plan.modelNs - (dsa::KERNEL_LAUNCH_NS + plan.modelFinish / dsa::CLOCK_GHZ)) <= 1e-9 * plan.modelNs;
    // The timeline's own accounting: finish = busy + fill + drain + barrier + mismatch, above the bound
    const dsa::TimelineSummary& tl = st.timeline;
    ok &= std::fabs(tl.finish - std::max(tl.vectorBusy, tl.dmaBusy) - tl.fill - tl.drain - tl.barrier - tl.mismatch) <= 1e-6 * tl.finish;
    ok &= tl.fill >= -1e-9 && tl.drain >= -1e-9 && tl.mismatch >= -1e-6 * tl.finish && tl.finish >= tl.LowerBound() - 1e-6;
    ok &= tl.finish >= tl.LatencyFloor() - 1e-6 * tl.finish;  // The floor is a bound: no run beats it
    if (!ok) {
        std::printf("  maxErr=%.3g pads=%llu spm=%zu stalls=%llu barriers=%llu queue=%llu cycles=%llu model=%llu finish=%.2f modeled=%.2f floor=%.2f\n",
                    maxErr, (unsigned long long)st.padTransfers, st.spmBytes, (unsigned long long)st.scalarStalls,
                    (unsigned long long)st.barriers, (unsigned long long)st.queueCycles, (unsigned long long)VectorCycles(st.busiest),
                    (unsigned long long)plan.modelCycles, tl.finish, modeled, tl.LatencyFloor());
    }
    Check(ok, msg);
    return st;
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
// FP16 rows of 12 columns at a 16-column pitch, 805-row tiles: X1/X2 double-buffered and one egress
// buffer (5 x 25,760 B), the FP32 Z tile (51,520 B), 15,136 B of scratch and 128 B of gamma/beta
void RunWaterlinePlan(std::mt19937& rng) {
    const uint32_t D = 12, B = 805, M = 2 * B * dsa::MAX_HARDWARE_CORES;
    const HardwareModel hw = HardwareModel::Target();
    TilingConfig plan = AdaptiveTiler::Build(M, D, 2, hw, TilingMode::ROW_PARALLEL, true);
    AdaptiveTiler::ApplyRowPipe(plan, M, D, 2, hw, {B, 0, 0, 2, 1, false, false});  // 2 tiles of 805 rows per core
    Check(plan.layout.Total() == dsa::SCRATCHPAD_SAFE_WATERLINE, "waterline plan is exactly 195584 bytes");
    RunPlan<F16>(M, D, plan, true, true, 0, rng, "waterline");
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
                        AdaptiveTiler::ApplyRowPipe(plan, M, D, s, hw, {6, head, tail, depth, outDepth, first, early});
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

// Tiles of many row groups (a worker normalizes ROW_GROUP rows at a time): row tiles as large as the
// scratchpad admits, and a column band run as one tile per core
template <class C>
void RunWidePlans(std::mt19937& rng) {
    using S = typename C::S;
    const HardwareModel hw = HardwareModel::Target();
    struct Case { uint32_t M, D; TilingMode mode; } cases[] = {
        {12000, 8, TilingMode::ROW_PARALLEL},     // 300-row tiles: groups of 128, 128, 44
        {65536, 24, TilingMode::ROW_PARALLEL},    // Squares chunks of 85 (FP32) or 64 rows straddle the groups
        {300, 640, TilingMode::SPLIT_COLUMNS}};   // One 300-row band tile per core
    for (const Case& c : cases) {
        TilingConfig plan = AdaptiveTiler::Build(c.M, c.D, sizeof(S), hw, c.mode, true);
        if (c.mode == TilingMode::ROW_PARALLEL) {  // The largest double-buffered tile, up to the core's rows
            const uint32_t rows = (c.M + plan.blocks - 1) / plan.blocks;
            uint32_t B = 1;
            while (B < rows && AdaptiveTiler::RowLayout(B + 1, c.D, sizeof(S), 2, 1).Total() <= hw.spmBytes) ++B;
            AdaptiveTiler::ApplyRowPipe(plan, c.M, c.D, sizeof(S), hw, {B, 0, 0, 2, 1, false, false});
        } else {
            AdaptiveTiler::ApplyBandPipe(plan, c.M, c.D, sizeof(S), hw, {c.M, 2, 1});
        }
        char msg[160];
        std::snprintf(msg, sizeof msg, "%s wide plan M=%u D=%u mode=%s tileRows=%u", Name<C>(), c.M, c.D, ModeName(plan.mode),
                      plan.tileRows);
        Check(plan.tileElems > 0 && plan.tileRows > 2 * AdaptiveTiler::ROW_GROUP && plan.layout.Total() <= hw.spmBytes, msg);
        RunPlan<C>(c.M, c.D, plan, true, true, 0, rng, "wide");
        RunPlan<C>(c.M, c.D, plan, false, true, 1, rng, "wide");
    }
}

// Row tiles across the row pitch's cases, forced: rows narrower than a 32-byte block, off the grid,
// of whole 64-lane repeats or ending mid-repeat (the squares take the strided form, or one Mul per
// row once a padded row is 2,048 lanes), rows past the strided form's 8-bit repeat stride (scaled row
// by row) and rows longer than the scratch chunk (summed in pieces); one tile per core and several.
// Each shape runs with and without gamma/beta, on aligned and offset bases.
template <class C>
void RunPitchSweep(std::mt19937& rng) {
    const HardwareModel hw = HardwareModel::Target();
    const uint32_t s = sizeof(typename C::S);
    for (const uint32_t D : {1u, 7u, 24u, 64u, 100u, 200u, 1000u, 2000u, 2047u, 2100u, 4100u}) {
        for (const uint32_t rows : {1u, 3u, 30u}) {  // Per core
            const uint32_t M = rows * dsa::MAX_HARDWARE_CORES;
            TilingConfig plan = AdaptiveTiler::Build(M, D, s, hw, TilingMode::ROW_PARALLEL, true);
            uint32_t B = 1;  // The core's rows in one tile, or the largest that fits
            while (B < rows && AdaptiveTiler::RowLayout(B + 1, D, s, 2, 1).Total() <= hw.spmBytes) ++B;
            AdaptiveTiler::ApplyRowPipe(plan, M, D, s, hw, {B, 0, 0, 2, 1, false, false});
            char msg[128];
            std::snprintf(msg, sizeof msg, "%s pitch plan M=%u D=%u tileRows=%u pitch=%u", Name<C>(), M, D, plan.tileRows, plan.pitch);
            Check(plan.tileElems > 0 && plan.pitch == AdaptiveTiler::RowPitch(D, s), msg);
            for (int pc = 0; pc < 4; ++pc) {
                for (const uint32_t offset : {0u, 1u}) RunPlan<C>(M, D, plan, !(pc & 1), !(pc & 2), offset, rng, "pitch");
            }
        }
    }
}

// Shares of at most 1,024 bytes, the ones the removed direct kernel took [ARCH CHALLENGE 8] (1 to 16
// rows, rows of up to 1,024 bytes, padded to 64 lanes or not, on and off the 32-byte grid, one core or
// many): the planner's plan and every core's rows in one tile, the direct kernel's schedule, run with
// and without gamma/beta, on aligned and offset bases (RunPlan holds each run to the reference and the
// model). The planner's plan finishes no later on the runtime's own timeline.
template <class C>
void RunTinySweep(std::mt19937& rng) {
    const uint32_t s = sizeof(typename C::S);
    const HardwareModel hw = HardwareModel::Target();
    uint32_t shapes = 0;
    for (const uint32_t D : {1u, 7u, 8u, 24u, 60u, 64u, 65u, 100u, 128u, 200u, 256u, 400u, 512u}) {
        for (const uint32_t M : {1u, 2u, 3u, 5u, 16u, 40u, 100u}) {
            const TilingConfig plan = AdaptiveTiler::Plan(M, D, s, hw), one = OneTile(M, D, s);
            const uint64_t rows = (AdaptiveTiler::MaxLoad(uint64_t(M) * D, one.unitElems, one.blocks) + D - 1) / D;
            if (rows * D * s > 1024) continue;
            ++shapes;
            DaeStats single{};
            for (int pc = 0; pc < 4; ++pc) {
                for (const uint32_t offset : {0u, 1u}) {
                    const DaeStats st = RunPlan<C>(M, D, one, !(pc & 1), !(pc & 2), offset, rng, "one tile");
                    if (pc == 0 && offset == 0) single = st;
                    RunPlan<C>(M, D, plan, !(pc & 1), !(pc & 2), offset, rng, "planned");
                }
            }
            const DaeStats planned = RunPlan<C>(M, D, plan, true, true, 0, rng, "planned");
            char msg[160];
            std::snprintf(msg, sizeof msg, "%s M=%u D=%u: planned %.0f cycles, one tile per core %.0f", Name<C>(), M, D,
                          planned.timeline.finish, single.timeline.finish);
            Check(planned.timeline.finish <= single.timeline.finish * (1 + 1e-12), msg);
        }
    }
    Check(shapes >= 40, "tiny sweep covers its shapes");
}

// C1 (1 x 64) and C2 (7 x 197 FP32): the planner's plan, executed, end to end (the launch, then the
// timeline), printed next to the best known time on the target, and next to every core's rows in one
// tile (the former direct kernel's schedule), which finishes no sooner. No run takes a queue step.
template <class C>
void CheckTinyLatency(const char* name, uint32_t M, uint32_t D, double bestUs, std::mt19937& rng) {
    const HardwareModel hw = HardwareModel::Target();
    const uint32_t s = sizeof(typename C::S);
    const TilingConfig plan = AdaptiveTiler::Plan(M, D, s, hw), one = OneTile(M, D, s);
    auto us = [](double cycles) { return (dsa::KERNEL_LAUNCH_NS + cycles / dsa::CLOCK_GHZ) / 1e3; };
    char label[96];
    std::snprintf(label, sizeof label, "%s planned", name);
    const DaeStats p = RunPlan<C>(M, D, plan, true, true, 0, rng, label);
    std::snprintf(label, sizeof label, "%s one tile", name);
    const DaeStats d = RunPlan<C>(M, D, one, true, true, 0, rng, label);
    std::printf("%s: %s on %u cores, %u-row tiles: %.2f us, %llu cycles, queue cycles %llu; one tile per core %.2f us; best known %.2f us\n",
                name, plan.mode == TilingMode::ROW_PARALLEL ? "rows" : "split", plan.blocks, plan.tileRows, us(p.timeline.finish),
                (unsigned long long)p.vectorCycles, (unsigned long long)p.queueCycles, us(d.timeline.finish), bestUs);
    std::snprintf(label, sizeof label, "%s planned no later than one tile per core, no queue step", name);
    Check(p.queueCycles == 0 && d.queueCycles == 0 && p.timeline.finish <= d.timeline.finish * (1 + 1e-12), label);
}

// -----------------------------------------------------------------------------
// 3. Zero-allocation execution [ARCH CHALLENGE 6]. With a caller-owned workspace, the only heap
// allocations during DaePipeline::Execute are the simulator's scratchpad buffers (dsa_runtime
// backs each TPipe buffer with a std::vector); the coordinator and the workers allocate nothing.
// The coordinator's own workspace gives bit-identical results.
// -----------------------------------------------------------------------------
uint64_t SimulatorBuffers(const TilingConfig& plan) {  // TPipe::InitBuffer blocks of one core
    const DaeLayout& L = plan.layout;
    // One TBuf per ring slot and way (X1 and X2, the gamma and beta chunks, Y), the seven scratch
    // partitions, then the others
    return 2 * L.depth + (L.paramQueue ? 2 * L.depth : 0) + L.outDepth + 7 + (L.z ? 1 : 0) + (L.params ? 1 : 0) + (L.resident ? 1 : 0) +
           (L.misc ? 1 : 0) + (L.rec ? 1 : 0);
}

template <class C>
void CheckAllocations(std::mt19937& rng) {
    using S = typename C::S;
    const HardwareModel hw = HardwareModel::Target();
    struct Case { uint32_t M, D; TilingMode mode; } cases[] = {
        {768, 192, TilingMode::ROW_PARALLEL},  {65536, 24, TilingMode::ROW_PARALLEL},
        {1, 70001, TilingMode::ROW_PARALLEL},  {8, 32768, TilingMode::SPLIT_COLUMNS},
        {17, 4097, TilingMode::SPLIT_D},       {3, 100003, TilingMode::SPLIT_D},
        {1, 64, TilingMode::ROW_PARALLEL},     {40, 64, TilingMode::ROW_PARALLEL},  // One core, 40 cores
        {7, 197, TilingMode::ROW_PARALLEL}};   // Rows off the 32-byte grid: multi-row padded descriptors
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
        std::snprintf(msg, sizeof msg, "%s zero-allocation M=%u D=%u mode=%s workspace=%zu B", Name<C>(), c.M, c.D, ModeName(plan.mode),
                      bytes);
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
        Check(news == plan.blocks * SimulatorBuffers(plan), msg);
        Check(std::memcmp(y.p, y2.p, N * sizeof(S)) == 0 && st.vectorCycles == own.vectorCycles && st.dmaBytes == own.dmaBytes &&
                  st.spmBytes == own.spmBytes && VectorCycles(st.busiest) == plan.modelCycles, msg);
        if (news != plan.blocks * SimulatorBuffers(plan)) {
            std::printf("  operator new calls: %llu, simulator buffers: %llu\n", (unsigned long long)news,
                        (unsigned long long)(plan.blocks * SimulatorBuffers(plan)));
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
    {"plan", "[DaePipeline]: needs a feasible plan"},             // Coordinator: a host plan has no DAE tiles
    {"workspace", "[DaePipeline]: the reduction workspace"},      // Coordinator: workspace off the 64-byte grid
    {"team", "[DaePipeline]: the OpenMP team does not have the plan's core count"},  // Worker: nested region, one thread
    {"band", "[DaePipeline]: column band wider than the plan"},   // Worker: band pitch below the band width
    // Static rings [Challenge 8]: every misuse a TQue's sanitizer trapped, and a ring's own limits
    {"ring-overflow", "[BufferRing]: every slot is in use"},
    {"ring-enqueue", "[BufferRing]: EnQue of a slot that is not allocated"},
    {"ring-empty", "[BufferRing]: DeQue on an empty ring"},
    {"ring-double-free", "[BufferRing]: freeing a slot that is free or still in flight"},
    {"ring-free-in-flight", "[BufferRing]: freeing a slot that is free or still in flight"},
    {"ring-unconsumed", "[BufferRing]: a slot was enqueued and never taken"},
    {"ring-in-use", "[BufferRing]: a slot is still in use at the end of the kernel"},
    {"ring-events", "[BufferRing]: a slot needs a literal event ID of its own"}};

int RunRingTrap(const char* name) {
    dsa::TPipe pipe;
    BufferRing<dsa::QuePosition::VECIN, 2, dsa::HardEvent::MTE2_V, dsa::HardEvent::V_MTE2> ring;
    ring.Init(pipe, !std::strcmp(name, "ring-events") ? 5 : 2, 64, 0);  // Five slots would need EVENT_ID4
    if (!std::strcmp(name, "ring-overflow")) {
        for (int i = 0; i < 3; ++i) ring.Alloc();
    } else if (!std::strcmp(name, "ring-enqueue")) {
        const uint32_t s = ring.Alloc();
        ring.EnQue(s);
        ring.EnQue(s);
    } else if (!std::strcmp(name, "ring-empty")) {
        ring.DeQue();
    } else if (!std::strcmp(name, "ring-double-free")) {
        const uint32_t s = ring.Alloc();
        ring.Free(s);
        ring.Free(s);
    } else if (!std::strcmp(name, "ring-free-in-flight")) {
        const uint32_t s = ring.Alloc();
        ring.EnQue(s);
        ring.Free(s);
    } else if (!std::strcmp(name, "ring-unconsumed")) {
        ring.EnQue(ring.Alloc());
        ring.Drain();
    } else if (!std::strcmp(name, "ring-in-use")) {
        ring.Alloc();
        ring.Drain();
    }
    return 0;  // Nothing trapped
}

int RunTrapCase(const char* name) {
    if (!std::strncmp(name, "ring-", 5)) return RunRingTrap(name);
    const uint32_t M = 8, D = 32768;  // C5: split over 40 cores
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
    } else if (!std::strcmp(name, "band")) {  // A column band whose pitch is narrower than its bands
        plan = AdaptiveTiler::Build(M, D, 2, HardwareModel::Target(), TilingMode::SPLIT_COLUMNS, true);
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
        const bool reported = out.find(std::string("[DSA Hardware Trap]: ") + t.report) != std::string::npos;
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
    RunPitchSweep<F32>(rng);
    RunPitchSweep<F16>(rng);
    RunPitchSweep<BF16>(rng);
    CheckTinyLatency<F16>("C1", 1, 64, 1.70, rng);
    CheckTinyLatency<F32>("C2", 7, 197, 2.21, rng);
    RunTinySweep<F32>(rng);
    RunTinySweep<F16>(rng);
    RunTinySweep<BF16>(rng);
    RunScheduleSweep<F32>(rng);
    RunScheduleSweep<F16>(rng);
    RunAll<F32>(rng);
    RunAll<F16>(rng);
    RunAll<BF16>(rng);
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
