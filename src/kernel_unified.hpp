#pragma once

#include "hpc_vector_norm.hpp"
#include "adaptive_tiler.hpp"
#include "simd.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>
#include <omp.h>

namespace hpc {

// Stride-0 stand-ins for a null bias (0) / gamma (1): the loops stay branch-free
template <class C> struct ParamDefaults {
    using S = typename C::S;
    static constexpr std::array<S, simd::W> Fill(S v) {
        std::array<S, simd::W> a{};
        for (auto& e : a) e = v;
        return a;
    }
    alignas(64) static constexpr std::array<S, simd::W> kZero = Fill(S(0));
    alignas(64) static constexpr std::array<S, simd::W> kOne = Fill(C::kOne);
};

// Thread-local resident Z scratchpad, allocated once per thread (nullptr => recompute Z)
inline float* ThreadScratch(size_t floats) {
    struct Buffer { float* p = nullptr; size_t n = 0; ~Buffer() { std::free(p); } };
    thread_local Buffer buf;
    if (buf.n < floats) {
        std::free(buf.p);
        buf.p = static_cast<float*>(std::aligned_alloc(64, (floats * sizeof(float) + 63) / 64 * 64));
        buf.n = buf.p ? floats : 0;
    }
    return buf.p;
}

// Bias / gamma operands in codec PC (the tensor codec, or FP32 after the per-call widening).
// A null tensor becomes a stride-0 constant (mask 0) so the hot loops stay branch-free.
template <class PC> struct Params {
    const typename PC::S *b, *g;
    size_t bMask, gMask;
    Params At(uint32_t cb) const { return {b + (cb & bMask), g + (cb & gMask), bMask, gMask}; }
};

// -----------------------------------------------------------------------------
// Pass 1: Z = X1 + X2 + bias (optionally kept resident) -> sum(Z^2).
// 4 x W independent FP32 accumulators (the 64-lane sliding window on AVX-512) within
// 4096-element blocks, FP64 across blocks: no FMA latency chain, no drift on long rows.
// -----------------------------------------------------------------------------
constexpr uint32_t kPrefetchBytes = 512;

template <class C, class PC, bool kKeep, bool kPrefetch>
double SumSquares(const typename C::S* x1, const typename C::S* x2, const Params<PC>& p, float* z, uint32_t n) {
    using namespace simd;
    const auto* b = p.b; // Locals: the Z stores cannot alias them, so they stay in registers
    const size_t bMask = p.bMask;
    auto zAt = [&](uint32_t c) { return Add(Add(Load(C{}, x1 + c), Load(C{}, x2 + c)), Load(PC{}, b + (c & bMask))); };
    double total = 0.0;
    for (uint32_t c = 0; c < n;) {
        const uint32_t e = std::min(n, c + 4096u);
        V a0 = Zero(), a1 = Zero(), a2 = Zero(), a3 = Zero();
        for (; c + 4 * W <= e; c += 4 * W) {
            if (kPrefetch) { // DRAM-streaming plans: keep more line fills in flight than the core would
                for (uint32_t q = 0; q < 4 * W * sizeof(typename C::S); q += 64) {
                    __builtin_prefetch(reinterpret_cast<const char*>(x1 + c) + kPrefetchBytes + q);
                    __builtin_prefetch(reinterpret_cast<const char*>(x2 + c) + kPrefetchBytes + q);
                }
            }
            const V z0 = zAt(c), z1 = zAt(c + W), z2 = zAt(c + 2 * W), z3 = zAt(c + 3 * W);
            if (kKeep) {
                Store(F32{}, z + c, z0);
                Store(F32{}, z + c + W, z1);
                Store(F32{}, z + c + 2 * W, z2);
                Store(F32{}, z + c + 3 * W, z3);
            }
            a0 = Fma(z0, z0, a0);
            a1 = Fma(z1, z1, a1);
            a2 = Fma(z2, z2, a2);
            a3 = Fma(z3, z3, a3);
        }
        for (; c + W <= e; c += W) {
            const V v = zAt(c);
            if (kKeep) Store(F32{}, z + c, v);
            a0 = Fma(v, v, a0);
        }
        if (c < e) { // Masked tail: padding lanes load as 0 and add nothing
            const uint32_t k = e - c;
            const V v = Add(Add(LoadN(C{}, x1 + c, k), LoadN(C{}, x2 + c, k)), LoadN(PC{}, b + (c & bMask), k));
            if (kKeep) StoreN(F32{}, z + c, v, k);
            a1 = Fma(v, v, a1);
            c = e;
        }
        total += static_cast<double>(Sum(Add(Add(a0, a1), Add(a2, a3))));
    }
    return total;
}

// -----------------------------------------------------------------------------
// Pass 2: Y = Z * invRms * gamma, with Z read from the resident scratchpad (kKeep) or
// recomputed from cache-hot inputs. kStream writes Y with non-temporal stores.
// -----------------------------------------------------------------------------
template <class C, class PC, bool kKeep, bool kStream>
void Normalize(const typename C::S* x1, const typename C::S* x2, const Params<PC>& p, const float* z,
               float invRms, typename C::S* y, uint32_t n) {
    using namespace simd;
    using S = typename C::S;
    const auto *b = p.b, *g = p.g;
    const size_t bMask = p.bMask, gMask = p.gMask;
    const V s = Set1(invRms);
    auto out = [&](uint32_t c) {
        const V zv = kKeep ? Load(F32{}, z + c) : Add(Add(Load(C{}, x1 + c), Load(C{}, x2 + c)), Load(PC{}, b + (c & bMask)));
        return Mul(Mul(zv, s), Load(PC{}, g + (c & gMask)));
    };
    auto outN = [&](uint32_t c, uint32_t k) {
        const V zv = kKeep ? LoadN(F32{}, z + c, k)
                           : Add(Add(LoadN(C{}, x1 + c, k), LoadN(C{}, x2 + c, k)), LoadN(PC{}, b + (c & bMask), k));
        return Mul(Mul(zv, s), LoadN(PC{}, g + (c & gMask), k));
    };
    uint32_t c = 0;
    if (kStream) { // Peel up to the vector alignment that streaming stores require
        const uintptr_t misalign = (uintptr_t(0) - reinterpret_cast<uintptr_t>(y)) % (W * sizeof(S));
        c = std::min(n, static_cast<uint32_t>(misalign / sizeof(S)));
        if (c) StoreN(C{}, y, outN(0, c), c);
    }
    for (; c + W <= n; c += W) {
        if (kStream) Stream(C{}, y + c, out(c));
        else Store(C{}, y + c, out(c));
    }
    if (c < n) StoreN(C{}, y + c, outN(c, n - c), n - c);
}

// Widen a 16-bit parameter vector to FP32 (once per call, reused by every row of the thread)
template <class C>
void WidenTo(const typename C::S* src, float* dst, uint32_t n) {
    using namespace simd;
    uint32_t c = 0;
    for (; c + W <= n; c += W) Store(F32{}, dst + c, Load(C{}, src + c));
    if (c < n) StoreN(F32{}, dst + c, LoadN(C{}, src + c, n - c), n - c);
}

// Split-D partial sums, one cache line per thread (no false sharing)
struct alignas(64) RowPartial {
    uint32_t row[2];
    double sum[2];
};

// -----------------------------------------------------------------------------
// Unified Kernel Pipeline Engine
// -----------------------------------------------------------------------------
template <class C>
class KernelUnifiedPipeline {
public:
    using S = typename C::S;

    static void Execute(const S* x1, const S* x2, const S* gamma, const S* bias, S* y,
                        uint32_t M, uint32_t D, float eps) {
        // The target laws with the CI host's core count (inside an enclosing parallel
        // region the caller already owns the cores) and measured costs
        const uint32_t P = omp_in_parallel() ? 1u : static_cast<uint32_t>(std::max(1, omp_get_max_threads()));
        // Tiling data is computed once per shape: repeated calls skip the ~70 ns of planning
        struct Memo { uint32_t M = ~0u, D = 0, P = 0; TilingConfig cfg{}; };
        thread_local Memo memo;
        if (memo.M != M || memo.D != D || memo.P != P) {
            memo = {M, D, P, AdaptiveTiler::Plan(M, D, sizeof(S), HardwareModel::Host(P), AdaptiveTiler::LastLevelCacheBytes())};
        }
        ExecutePlan(x1, x2, gamma, bias, y, M, D, eps, memo.cfg);
    }

    // Runs an explicit plan (exposed so tests can force every tiling path)
    static void ExecutePlan(const S* x1, const S* x2, const S* gamma, const S* bias, S* y,
                            uint32_t M, uint32_t D, float eps, const TilingConfig& cfg) {
        if (M == 0 || D == 0) return;
        static std::atomic<uint32_t> epoch{0};
        const bool reverse = cfg.serpentine && (epoch.fetch_add(1, std::memory_order_relaxed) & 1u);
        const Job job{x1, x2, gamma, bias, y, M, D, eps, cfg, reverse};
        const bool stream = cfg.streamStores && reinterpret_cast<uintptr_t>(y) % sizeof(S) == 0;
        RowPartial parts[AdaptiveTiler::MAX_THREADS];
        if (cfg.threads <= 1) {
            stream ? Worker<true>(job, 0, 1, parts) : Worker<false>(job, 0, 1, parts);
            return;
        }
        #pragma omp parallel num_threads(std::min(cfg.threads, AdaptiveTiler::MAX_THREADS))
        {
            const uint32_t tid = omp_get_thread_num(), nt = omp_get_num_threads();
            stream ? Worker<true>(job, tid, nt, parts) : Worker<false>(job, tid, nt, parts);
        }
    }

private:
    static constexpr uint32_t kNoRow = ~0u;
    static constexpr uint32_t kWidenMinRows = 4;     // Rows that amortize widening bias/gamma
    static constexpr uint32_t kWidenMaxElems = 4096; // Widened bias + gamma stay L1/L2-resident

    struct Job {
        const S *x1, *x2, *gamma, *bias;
        S* y;
        uint32_t M, D;
        float eps;
        TilingConfig cfg;
        bool reverse;
    };
    struct Fragment { uint32_t row, cb, ce; float* z; double sum; };

    template <bool kStream, class PC>
    static double Pass1(const Job& j, const Params<PC>& p, uint32_t r, uint32_t cb, uint32_t n, float* z) {
        const size_t o = static_cast<size_t>(r) * j.D + cb;
        return z ? SumSquares<C, PC, true, kStream>(j.x1 + o, j.x2 + o, p.At(cb), z, n)
                 : SumSquares<C, PC, false, kStream>(j.x1 + o, j.x2 + o, p.At(cb), nullptr, n);
    }

    template <bool kStream, class PC>
    static void Pass2(const Job& j, const Params<PC>& p, uint32_t r, uint32_t cb, uint32_t n, const float* z, double sumSq) {
        const size_t o = static_cast<size_t>(r) * j.D + cb;
        const float invRms = static_cast<float>(1.0 / std::sqrt(sumSq / j.D + static_cast<double>(j.eps)));
        if (z) Normalize<C, PC, true, kStream>(j.x1 + o, j.x2 + o, p.At(cb), z, invRms, j.y + o, n);
        else Normalize<C, PC, false, kStream>(j.x1 + o, j.x2 + o, p.At(cb), nullptr, invRms, j.y + o, n);
    }

    template <bool kStream>
    static void Worker(const Job& j, uint32_t tid, uint32_t nt, RowPartial* parts) {
        const uint32_t D = j.D;
        const CoreRange range = AdaptiveTiler::Range(j.cfg, j.M, D, tid, nt);
        Fragment frag[2];
        for (uint32_t k = 0; k < range.nFrag; ++k) frag[k] = {range.frag[k].row, range.frag[k].cb, range.frag[k].ce, nullptr, 0.0};
        const uint32_t nFrag = range.nFrag, rA = range.rowA, rZ = range.rowZ;

        float* scratch = ThreadScratch(j.cfg.residentElems);
        const size_t cap = scratch ? j.cfg.residentElems : 0;
        const Params<C> p16{j.bias ? j.bias : ParamDefaults<C>::kZero.data(), j.gamma ? j.gamma : ParamDefaults<C>::kOne.data(),
                            j.bias ? ~size_t(0) : 0, j.gamma ? ~size_t(0) : 0};
        if constexpr (sizeof(S) == 2) {
            // 16-bit tensors: widen bias/gamma to FP32 once when enough rows reuse them
            // (two conversions fewer per vector in the hot loops).
            if (rZ - rA >= kWidenMinRows && D <= kWidenMaxElems && 2 * static_cast<size_t>(D) <= cap) {
                if (j.bias) WidenTo<C>(j.bias, scratch, D);
                if (j.gamma) WidenTo<C>(j.gamma, scratch + D, D);
                const Params<F32> p32{j.bias ? scratch : ParamDefaults<F32>::kZero.data(),
                                      j.gamma ? scratch + D : ParamDefaults<F32>::kOne.data(), p16.bMask, p16.gMask};
                return Run<kStream>(j, p32, tid, nt, parts, frag, nFrag, rA, rZ, scratch + 2 * D, cap - 2 * D);
            }
        }
        Run<kStream>(j, p16, tid, nt, parts, frag, nFrag, rA, rZ, scratch, cap);
    }

    template <bool kStream, class PC>
    static void Run(const Job& j, const Params<PC>& p, uint32_t tid, uint32_t nt, RowPartial* parts,
                    Fragment* frag, uint32_t nFrag, uint32_t rA, uint32_t rZ, float* scratch, size_t cap) {
        const uint32_t D = j.D;

        // Fragments first so their partial sums are published before the full rows run:
        // the barrier wait then overlaps local work instead of idling.
        size_t used = 0;
        for (uint32_t k = 0; k < nFrag; ++k) {
            const uint32_t len = frag[k].ce - frag[k].cb;
            if (used + len <= cap) { frag[k].z = scratch + used; used += len; }
            frag[k].sum = Pass1<kStream>(j, p, frag[k].row, frag[k].cb, len, frag[k].z);
        }
        const bool split = j.cfg.mode == TilingMode::SPLIT_D;
        if (split) {
            parts[tid] = {{nFrag > 0 ? frag[0].row : kNoRow, nFrag > 1 ? frag[1].row : kNoRow},
                          {nFrag > 0 ? frag[0].sum : 0.0, nFrag > 1 ? frag[1].sum : 0.0}};
        }

        // Complete rows: fully thread-local, Z resident between the two passes. Rows go in
        // batches: all pass-1 reductions of a batch are issued before its first pass 2, so the
        // reduce -> sqrt -> reciprocal chains overlap instead of stalling every short row.
        const uint32_t B = std::max(1u, std::min(j.cfg.batchRows, AdaptiveTiler::MAX_BATCH));
        float* zb = used + static_cast<size_t>(B) * D <= cap ? scratch + used : nullptr;
        double sums[AdaptiveTiler::MAX_BATCH];
        // Serpentine at chunk granularity: chunk order flips per call, rows inside a chunk stay
        // ascending so the hardware prefetchers keep their forward streams.
        const uint32_t CR = std::max(1u, j.cfg.chunkRows / B) * B, K = (rZ - rA + CR - 1) / CR;
        for (uint32_t kk = 0; kk < K; ++kk) {
            const uint32_t s = rA + (j.reverse ? K - 1 - kk : kk) * CR, e = std::min(rZ, s + CR);
            for (uint32_t i = s; i < e; i += B) {
                const uint32_t nb = std::min(B, e - i);
                for (uint32_t k = 0; k < nb; ++k) sums[k] = Pass1<kStream>(j, p, i + k, 0, D, zb ? zb + static_cast<size_t>(k) * D : nullptr);
                for (uint32_t k = 0; k < nb; ++k) Pass2<kStream>(j, p, i + k, 0, D, zb ? zb + static_cast<size_t>(k) * D : nullptr, sums[k]);
            }
        }

        // Split-D reduction: every team member reaches the barrier (empty ranges included);
        // all owners of a row sum the partials in the same order => identical invRms.
        if (split) {
            #pragma omp barrier
            for (uint32_t k = 0; k < nFrag; ++k) {
                uint32_t first, last;
                AdaptiveTiler::RowOwners(j.cfg, D, frag[k].row, nt, first, last);
                double total = 0.0;
                for (uint32_t t = first; t <= last; ++t) {
                    for (uint32_t q = 0; q < 2; ++q) total += parts[t].row[q] == frag[k].row ? parts[t].sum[q] : 0.0;
                }
                Pass2<kStream>(j, p, frag[k].row, frag[k].cb, frag[k].ce - frag[k].cb, frag[k].z, total);
            }
        }
        if (kStream) simd::Fence();
    }
};

// =============================================================================
// Target DAE pipeline: a HardwareModel::Target() plan executed the way the 40-core
// accelerator runs it, through include/dsa_runtime.hpp. One OpenMP thread is one
// simulated core (GetCoreIdx). Every scratchpad buffer is claimed through TPipe
// (191 KB trap); every global-memory transfer is a 32-byte DataCopy, or a counted
// DataCopyPad where a row does not end on a DMA block.
// =============================================================================
struct DaeStats {
    uint64_t vectorCycles = 0;  // Busiest core
    uint64_t dmaBytes = 0;      // All cores
    uint64_t dmaTransfers = 0;
    uint64_t padTransfers = 0;  // Transfers that were not whole 32-byte blocks
    uint64_t barriers = 0;      // Busiest core
    size_t spmBytes = 0;        // Largest per-core scratchpad claim
};

template <class C>
class DaePipeline {
public:
    using S = typename C::S;

    static DaeStats Execute(const S* x1, const S* x2, const S* gamma, const S* bias, S* y,
                            uint32_t M, uint32_t D, float eps, const TilingConfig& plan) {
        DaeStats stats;
        if (M == 0 || D == 0) return stats;
        if (plan.tileElems == 0) throw std::invalid_argument("DaePipeline needs a plan with DAE tiles (HardwareModel::Target())");
        const uint32_t blocks = std::max(1u, std::min(plan.blocks, AdaptiveTiler::MAX_THREADS));
        // Split-D partial records in system memory: one 32-byte block per core
        float* workspace = static_cast<float*>(std::aligned_alloc(64, (blocks * 32u + 63) / 64 * 64));
        std::vector<dsa::HardwareCycleTracker> trackers(blocks);
        std::vector<size_t> spm(blocks, 0);
        std::string error;
        #pragma omp parallel num_threads(blocks)
        {
            const uint32_t coreId = dsa::GetCoreIdx(), numCores = dsa::GetCoreNum();
            dsa::g_cycleTracker.Reset();
            Core core(x1, x2, gamma, bias, y, M, D, eps, plan, workspace, coreId);
            const CoreRange range = AdaptiveTiler::Range(plan, M, D, coreId, numCores);
            const bool split = plan.mode == TilingMode::SPLIT_D;
            std::string err;
            try {
                core.Init();
                core.Phase1(range, split);
            } catch (const std::exception& e) {
                err = e.what();
            }
            if (split) dsa::SyncAll();  // Reached by every core, failed or not: no deadlock
            if (split && err.empty()) {
                try {
                    core.Phase2(range, numCores);
                } catch (const std::exception& e) {
                    err = e.what();
                }
            }
            trackers[coreId] = dsa::g_cycleTracker;
            spm[coreId] = core.pipe.GetTotalAllocatedBytes();
            if (!err.empty()) {
                #pragma omp critical(hpc_dae_error)
                if (error.empty()) error = err;
            }
        }
        std::free(workspace);
        if (!error.empty()) throw std::runtime_error(error);
        for (uint32_t b = 0; b < blocks; ++b) {
            stats.vectorCycles = std::max(stats.vectorCycles, trackers[b].GetTotalVectorCycles());
            stats.dmaBytes += trackers[b].dmaBytesMoved;
            stats.dmaTransfers += trackers[b].dmaTransfers;
            stats.padTransfers += trackers[b].padTransfers;
            stats.barriers = std::max(stats.barriers, trackers[b].barrierCount);
            stats.spmBytes = std::max(stats.spmBytes, spm[b]);
        }
        return stats;
    }

private:
    static constexpr uint32_t kNoRow = ~0u;
    static constexpr uint32_t kSumSlots = 256;  // Row sums kept in misc[256, 512)

    static float ToF32(S v) {
        if constexpr (std::is_same<C, F32>::value) return v;
        else if constexpr (std::is_same<C, F16>::value) return HalfToFloat(v);
        else return BF16ToFloat(v);
    }
    static S FromF32(float v) {
        if constexpr (std::is_same<C, F32>::value) return v;
        else if constexpr (std::is_same<C, F16>::value) return FloatToHalf(v);
        else return FloatToBF16(v);
    }

    template <class T>
    static bool BlockAligned(const T* p, uint64_t n) {
        return reinterpret_cast<uintptr_t>(p) % dsa::DMA_ALIGN_BYTES == 0 && n * sizeof(T) % dsa::DMA_ALIGN_BYTES == 0;
    }
    template <class T>
    static void DmaIn(dsa::LocalTensor<T> dst, const T* src, uint32_t n) {
        if (BlockAligned(src, n)) dsa::DataCopy(dst, src, n);
        else dsa::DataCopyPad(dst, src, n);
    }
    template <class T>
    static void DmaOut(T* dst, dsa::LocalTensor<T> src, uint32_t n) {
        if (BlockAligned(dst, n)) dsa::DataCopy(dst, src, n);
        else dsa::DataCopyPad(dst, src, n);
    }

    struct Core {
        Core(const S* x1_, const S* x2_, const S* gamma_, const S* bias_, S* y_, uint32_t M_, uint32_t D_, float eps_,
             const TilingConfig& plan_, float* workspace_, uint32_t b_)
            : x1(x1_), x2(x2_), gamma(gamma_), bias(bias_), y(y_), M(M_), D(D_), eps(eps_), plan(plan_),
              workspace(workspace_), b(b_) {}

        const S *x1, *x2, *gamma, *bias;
        S* y;
        uint32_t M, D;
        float eps;
        const TilingConfig& plan;
        float* workspace;
        uint32_t b;

        dsa::TPipe pipe;
        dsa::TQue<dsa::QuePosition::VECIN, 2> qX1, qX2, qP;          // Double-buffered tiles
        dsa::TBuf<dsa::QuePosition::VECCALC> qZ, qTmp, qPar, qRes, qMisc;
        dsa::LocalTensor<float> z, tmp, par, res, misc;
        uint32_t tmpCap = 0, quantum = 1;

        void Init() {
            const DaeLayout& L = plan.layout;
            pipe.InitBuffer(qX1, 2, L.tile);
            pipe.InitBuffer(qX2, 2, L.tile);
            if (L.paramQueue) pipe.InitBuffer(qP, 2, L.tile);
            if (L.z) { pipe.InitBuffer(qZ, L.z); z = qZ.Get<float>(); }
            else { pipe.InitBuffer(qZ, 64); z = qZ.Get<float>(); } // Always initialized!
            pipe.InitBuffer(qTmp, L.tmp);
            tmp = qTmp.Get<float>();
            if (L.params) { pipe.InitBuffer(qPar, L.params); par = qPar.Get<float>(); }
            if (L.resident) { pipe.InitBuffer(qRes, L.resident); res = qRes.Get<float>(); }
            pipe.InitBuffer(qMisc, L.misc);
            misc = qMisc.Get<float>();
            tmpCap = L.tmp / sizeof(float) / 8 * 8;
            quantum = std::max<uint32_t>(1, dsa::DMA_ALIGN_BYTES / sizeof(S));
        }

        // ---- Phase 1: fragments' sweep 1 + publish, then every complete row -----------
        void Phase1(const CoreRange& r, bool split) {
            if (plan.tileRows) return RowTiles(r.rowA, r.rowZ);
            used = 0;
            for (uint32_t k = 0; k < r.nFrag; ++k) {
                const CoreRange::Fragment& f = r.frag[k];
                fragZ[k] = Claim(f.ce - f.cb, k);
                fragSum[k] = Sweep1(f.row, f.cb, f.ce, fragZ[k]);
            }
            if (split) {  // One 32-byte record per core: {sum0, sum1, row0, row1, 0...}
                const uint32_t rows[2] = {r.nFrag > 0 ? r.frag[0].row : kNoRow, r.nFrag > 1 ? r.frag[1].row : kNoRow};
                dsa::Duplicate(misc, 0.0f, 8);
                misc.SetValue(0, r.nFrag > 0 ? fragSum[0] : 0.0f);
                misc.SetValue(1, r.nFrag > 1 ? fragSum[1] : 0.0f);
                std::memcpy(misc.data + 2, rows, sizeof rows);
                dsa::DataCopy(workspace + static_cast<size_t>(b) * 8, misc, 8);
            }
            for (uint32_t row = r.rowA; row < r.rowZ; ++row) {
                const size_t mark = used;
                dsa::LocalTensor<float>* zr = Claim(D, 2);
                Sweep2(row, 0, D, zr, Sweep1(row, 0, D, zr));
                used = mark;
            }
        }

        // ---- Phase 2 (after SyncAll): gather the owners' records, finish the fragments ---
        void Phase2(const CoreRange& r, uint32_t nb) {
            for (uint32_t k = 0; k < r.nFrag; ++k) {
                const CoreRange::Fragment& f = r.frag[k];
                uint32_t first, last;
                AdaptiveTiler::RowOwners(plan, D, f.row, nb, first, last);
                dsa::LocalTensor<float> recs = misc[8];
                dsa::DataCopy(recs, workspace + static_cast<size_t>(first) * 8, (last - first + 1) * 8);
                float total = 0.0f;  // Same order on every owner => identical sigma
                for (uint32_t t = 0; t <= last - first; ++t) {
                    uint32_t rows[2];
                    std::memcpy(rows, recs.data + t * 8 + 2, sizeof rows);
                    total += (rows[0] == f.row ? recs.GetValue(t * 8) : 0.0f) + (rows[1] == f.row ? recs.GetValue(t * 8 + 1) : 0.0f);
                }
                Sweep2(f.row, f.cb, f.ce, fragZ[k], total);
            }
        }

        // ---- Row tiles: B whole rows per double-buffered tile, parameters resident -------
        void RowTiles(uint32_t rA, uint32_t rZ) {
            if (rA >= rZ) return;
            LoadVector(par, gamma, 1.0f);
            LoadVector(par[D], bias, 0.0f);
            const uint32_t B = plan.tileRows;
            LoadTile(static_cast<uint64_t>(rA) * D, std::min(B, rZ - rA) * D);
            for (uint32_t row = rA; row < rZ; row += B) {
                const uint32_t n = std::min(B, rZ - row);
                if (row + B < rZ) LoadTile(static_cast<uint64_t>(row + B) * D, std::min(B, rZ - row - B) * D);  // Prefetch
                dsa::LocalTensor<S> a = qX1.template DeQue<S>(), bt = qX2.template DeQue<S>();
                dsa::LocalTensor<float> zt = ZOf(a);
                AddInputs(zt, a, bt, n * D);
                qX2.FreeTensor(bt);
                if (bias) {
                    for (uint32_t i = 0; i < n; ++i) dsa::Add(zt[i * D], zt[i * D], par[D], D);
                }
                for (uint32_t i = 0; i < n; i += kSumSlots) {
                    const uint32_t k = std::min(kSumSlots, n - i);
                    RowSums(zt[i * D], k);  // misc[256 + r]
                    for (uint32_t r = 0; r < k; ++r) {
                        dsa::LocalTensor<float> zr = zt[(i + r) * D];
                        dsa::Muls(zr, zr, InvRms(misc.GetValue(kSumSlots + r)), D);
                        dsa::Mul(zr, zr, par, D);
                    }
                }
                Narrow(a, zt, n * D);
                DmaOut(y + static_cast<uint64_t>(row) * D, a, n * D);
                qX1.FreeTensor(a);
            }
        }

        // ---- Column tiles over one row segment [cb, ce) --------------------------------
        // Sweep 1 accumulates sum(Z^2), writing Z to `zr` (resident) when given.
        float Sweep1(uint32_t row, uint32_t cb, uint32_t ce, dsa::LocalTensor<float>* zr) {
            const uint64_t start = static_cast<uint64_t>(row) * D + cb, end = start + (ce - cb);
            float sum = 0.0f;
            Tiles(start, end, true, [&](uint64_t e, uint32_t n) {
                dsa::LocalTensor<S> a = qX1.template DeQue<S>(), bt = qX2.template DeQue<S>();
                dsa::LocalTensor<float> zt = zr ? (*zr)[static_cast<uint32_t>(e - start)] : z;
                AddInputs(zt, a, bt, n);
                qX1.FreeTensor(a);
                qX2.FreeTensor(bt);
                if (bias) ApplyParam(zt, bias + (e - uint64_t(row) * D), n, false);
                sum += SumSquares(zt, n);
            });
            return sum;
        }

        // Sweep 2 normalizes: from resident Z, or re-streaming X1/X2 when Z did not fit.
        void Sweep2(uint32_t row, uint32_t cb, uint32_t ce, dsa::LocalTensor<float>* zr, float sumSq) {
            const uint64_t start = static_cast<uint64_t>(row) * D + cb, end = start + (ce - cb);
            const float inv = InvRms(sumSq);
            Tiles(start, end, zr == nullptr, [&](uint64_t e, uint32_t n) {
                const uint64_t col = e - uint64_t(row) * D;
                dsa::LocalTensor<S> out;
                dsa::LocalTensor<float> zt;
                if (zr) {
                    zt = (*zr)[static_cast<uint32_t>(e - start)];
                    out = qX1.template AllocTensor<S>();
                } else {
                    out = qX1.template DeQue<S>();
                    dsa::LocalTensor<S> bt = qX2.template DeQue<S>();
                    zt = z;
                    AddInputs(zt, out, bt, n);
                    qX2.FreeTensor(bt);
                    if (bias) ApplyParam(zt, bias + col, n, false);
                }
                dsa::Muls(zt, zt, inv, n);
                if (gamma) ApplyParam(zt, gamma + col, n, true);
                Narrow(out, zt, n);
                DmaOut(y + e, out, n);
                qX1.FreeTensor(out);
            });
        }

        // Tiles of at most tileElems whose interior boundaries sit on the 32-byte grid; with
        // `stream`, X1/X2 of tile k+1 are loaded before tile k is consumed (double buffering).
        template <class Body>
        void Tiles(uint64_t start, uint64_t end, bool stream, Body body) {
            auto length = [&](uint64_t e) { return static_cast<uint32_t>(std::min(end, e / quantum * quantum + plan.tileElems) - e); };
            uint64_t e = start;
            uint32_t n = start < end ? length(start) : 0;
            if (stream && n) LoadTile(e, n);
            while (e < end) {
                const uint64_t next = e + n;
                const uint32_t nn = next < end ? length(next) : 0;
                if (stream && nn) LoadTile(next, nn);  // Prefetch
                body(e, n);
                e = next;
                n = nn;
            }
        }

        void LoadTile(uint64_t e, uint32_t n) {
            dsa::LocalTensor<S> a = qX1.template AllocTensor<S>();
            DmaIn(a, x1 + e, n);
            qX1.EnQue(a);
            dsa::LocalTensor<S> bt = qX2.template AllocTensor<S>();
            DmaIn(bt, x2 + e, n);
            qX2.EnQue(bt);
        }

        // Z = X1 + X2 in FP32 (FP32 tiles in place; 16-bit tiles widened into zt)
        void AddInputs(dsa::LocalTensor<float> zt, dsa::LocalTensor<S> a, dsa::LocalTensor<S> bt, uint32_t n) {
            if constexpr (std::is_same<C, F32>::value) {
                dsa::Add(zt, a, bt, n);
            } else {
                dsa::Cast(zt, a, n, ToF32);
                for (uint32_t o = 0; o < n; o += tmpCap) {
                    const uint32_t k = std::min(tmpCap, n - o);
                    dsa::Cast(tmp, bt[o], k, ToF32);
                    dsa::Add(zt[o], zt[o], tmp, k);
                }
            }
        }

        // zt (+|*)= parameter chunk streamed from system memory
        void ApplyParam(dsa::LocalTensor<float> zt, const S* src, uint32_t n, bool multiply) {
            dsa::LocalTensor<S> pc = qP.template AllocTensor<S>();
            DmaIn(pc, src, n);
            for (uint32_t o = 0; o < n; o += tmpCap) {
                const uint32_t k = std::min(tmpCap, n - o);
                dsa::Cast(tmp, pc[o], k, ToF32);
                if (multiply) dsa::Mul(zt[o], zt[o], tmp, k);
                else dsa::Add(zt[o], zt[o], tmp, k);
            }
            qP.FreeTensor(pc);
        }

        void LoadVector(dsa::LocalTensor<float> dst, const S* src, float fill) {
            if (!src) return dsa::Duplicate(dst, fill, D);
            const uint32_t tileLimit = plan.layout.tile / static_cast<uint32_t>(sizeof(S));
            for (uint32_t o = 0; o < D; o += tileLimit) {
                const uint32_t k = std::min(tileLimit, D - o);
                dsa::LocalTensor<S> st = qX1.template AllocTensor<S>();
                DmaIn(st, src + o, k);
                dsa::Cast(dst[o], st, k, ToF32);
                qX1.FreeTensor(st);
            }
        }

        void Narrow(dsa::LocalTensor<S> out, dsa::LocalTensor<float> zt, uint32_t n) {
            if constexpr (!std::is_same<C, F32>::value) dsa::Cast(out, zt, n, FromF32);
            else if (out.GetData() != zt.GetData()) dsa::Muls(out, zt, 1.0f, n);
        }

        dsa::LocalTensor<float> ZOf(dsa::LocalTensor<S> a) {
            if constexpr (std::is_same<C, F32>::value) return a;
            else return z;
        }

        // sum(Z^2) of one segment: Mul + BlockReduceSum + VectorReduceSum (pure vector binary tree)
        float SumSquares(dsa::LocalTensor<float> zt, uint32_t n) {
            float total = 0.0f;
            for (uint32_t o = 0; o < n; o += tmpCap) {
                uint32_t k = std::min(tmpCap, n - o);
                dsa::Mul(tmp, zt[o], zt[o], k);
                while (k % 8 == 0 && k >= 8) { dsa::BlockReduceSum(tmp, tmp, k); k /= 8; }
                total += dsa::VectorReduceSum(tmp, k);
            }
            return total;
        }

        // sum(Z^2) of `rows` whole rows into misc[256 + r]; pure vector fold keeps rows apart
        void RowSums(dsa::LocalTensor<float> zt, uint32_t rows) {
            if (D > tmpCap) {
                for (uint32_t r = 0; r < rows; ++r) misc.SetValue(kSumSlots + r, SumSquares(zt[r * D], D));
                return;
            }
            const uint32_t per = tmpCap / D;
            for (uint32_t r0 = 0; r0 < rows; r0 += per) {
                const uint32_t k = std::min(per, rows - r0);
                dsa::Mul(tmp, zt[r0 * D], zt[r0 * D], k * D);
                uint32_t len = D;
                while (len % 8 == 0 && len >= 8) { dsa::BlockReduceSum(tmp, tmp, k * len); len /= 8; }
                for (uint32_t r = 0; r < k; ++r) {
                    misc.SetValue(kSumSlots + r0 + r, dsa::VectorReduceSum(tmp[r * len], len));
                }
            }
        }

        float InvRms(float sumSq) const {
            return dsa::VectorInvRms(sumSq, D, eps);
        }

        // Resident Z for a column-tiled segment (slots 0-1: fragments, 2: the current row)
        dsa::LocalTensor<float>* Claim(uint32_t n, uint32_t slot) {
            if (!plan.zResident || used + n > plan.zResident) return nullptr;
            slots[slot] = res[static_cast<uint32_t>(used)];
            used += n;
            return &slots[slot];
        }

        size_t used = 0;
        dsa::LocalTensor<float> slots[3];
        dsa::LocalTensor<float>* fragZ[2] = {nullptr, nullptr};
        float fragSum[2] = {0.0f, 0.0f};
    };
};

} // namespace hpc
