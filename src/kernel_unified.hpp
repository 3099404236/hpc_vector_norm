#pragma once

#include "hpc_vector_norm.hpp"
#include "adaptive_tiler.hpp"
#include "simd.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <type_traits>
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
        if (cfg.mode == TilingMode::SPLIT_COLUMNS) throw std::invalid_argument("SPLIT_COLUMNS plans run on DaePipeline only");
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
// target processor runs it, through include/dsa_runtime.hpp. One OpenMP thread is one
// simulated core (GetCoreIdx). Every scratchpad buffer is claimed through TPipe
// (191 KB trap); every system-memory transfer is a 32-byte DataCopy, or a counted
// DataCopyPad where a transfer does not end on a DMA block. The scalar unit never
// reads the scratchpad (GetValue: a 500-cycle V->S stall): row sums reach it through
// VectorReduceSum, and Split-D partials are combined by vector adds.
//
// Coordinator and workers [ARCH CHALLENGE 6]: DaePipeline::Execute runs on the calling
// thread. It takes a finished plan, owns the 64-byte-aligned reduction workspace and
// dispatches one Core per simulated core with POD arguments. A Core is freestanding: it
// allocates nothing (per-row state lives in fixed arrays on its stack), throws nothing (a
// broken invariant is a DSA_ASSERT trap) and passes LocalTensor views by value.
// =============================================================================
struct DaeStats {
    uint64_t vectorCycles = 0;  // Busiest core: vector + scalar stalls + barriers
    uint64_t dmaBytes = 0;      // All cores
    uint64_t dmaTransfers = 0;
    uint64_t padTransfers = 0;  // Transfers that were not whole 32-byte blocks
    uint64_t barriers = 0;      // Busiest core
    uint64_t scalarStalls = 0;  // GetValue V->S stalls, all cores
    size_t spmBytes = 0;        // Largest per-core scratchpad claim
    dsa::HardwareCycleTracker busiest;  // Full cycle breakdown of the busiest core
};

template <class C>
class DaePipeline {
public:
    using S = typename C::S;

    // Reduction workspace a plan needs: one record of partial sums per core in whole 32-byte
    // blocks, rounded up to 64 bytes (0: the plan never reduces across cores)
    static size_t WorkspaceBytes(const TilingConfig& plan, uint32_t M) {
        const size_t bytes = size_t(Cores(plan)) * AdaptiveTiler::RecordFloats(plan, M) * sizeof(float);
        return (bytes + 63) / 64 * 64;
    }

    // Coordinator with a caller-owned workspace (64-byte aligned, WorkspaceBytes() long): it
    // allocates nothing. The plan is evaluated once, by the caller, never by the workers.
    static DaeStats Execute(const S* x1, const S* x2, const S* gamma, const S* bias, S* y, uint32_t M, uint32_t D,
                            float eps, const TilingConfig& plan, float* workspace, size_t workspaceBytes) {
        DaeStats stats;
        if (M == 0 || D == 0) return stats;
        DSA_ASSERT(plan.tileElems > 0, "[DaePipeline]: needs a feasible plan with DAE tiles (HardwareModel::Target())");
        const size_t need = WorkspaceBytes(plan, M);
        DSA_ASSERT(need == 0 || (workspace && reinterpret_cast<uintptr_t>(workspace) % 64 == 0 && workspaceBytes >= need),
                   "[DaePipeline]: the reduction workspace must be 64-byte aligned and WorkspaceBytes() long");
        if (need) std::memset(workspace, 0, need);  // A core without units publishes nothing: its records read 0
        const Args args{x1, x2, gamma, bias, y, M, D, eps, &plan, workspace, Cores(plan)};
        CoreResult results[AdaptiveTiler::MAX_THREADS];
        #pragma omp parallel num_threads(args.cores)
        {
            Core core(args, dsa::GetCoreIdx());
            core.Execute(dsa::GetCoreNum(), results[core.b]);
        }
        for (uint32_t b = 0; b < args.cores; ++b) {
            const dsa::HardwareCycleTracker& t = results[b].cycles;
            if (t.GetTotalVectorCycles() >= stats.vectorCycles) {
                stats.vectorCycles = t.GetTotalVectorCycles();
                stats.busiest = t;
            }
            stats.scalarStalls += t.scalarStallCount;
            stats.dmaBytes += t.dmaBytesMoved;
            stats.dmaTransfers += t.dmaTransfers;
            stats.padTransfers += t.padTransfers;
            stats.barriers = std::max(stats.barriers, t.barrierCount);
            stats.spmBytes = std::max(stats.spmBytes, results[b].spmBytes);
        }
        return stats;
    }

    // Coordinator with its own workspace: allocated on the calling thread the first time a plan
    // needs more, then reused, so repeated calls allocate nothing
    static DaeStats Execute(const S* x1, const S* x2, const S* gamma, const S* bias, S* y, uint32_t M, uint32_t D,
                            float eps, const TilingConfig& plan) {
        const size_t bytes = M && D ? WorkspaceBytes(plan, M) : 0;
        return Execute(x1, x2, gamma, bias, y, M, D, eps, plan, bytes ? CoordinatorWorkspace(bytes) : nullptr, bytes);
    }

private:
    static uint32_t Cores(const TilingConfig& plan) { return std::max(1u, std::min(plan.blocks, AdaptiveTiler::MAX_THREADS)); }

    static float* CoordinatorWorkspace(size_t bytes) {
        struct Buffer { float* p = nullptr; size_t n = 0; ~Buffer() { std::free(p); } };
        thread_local Buffer buf;
        if (buf.n < bytes) {
            std::free(buf.p);
            buf.p = static_cast<float*>(std::aligned_alloc(64, bytes));
            buf.n = buf.p ? bytes : 0;
        }
        return buf.p;
    }

    // Everything a worker receives: plain pointers and sizes, shared read-only
    struct Args {
        const S *x1, *x2, *gamma, *bias;
        S* y;
        uint32_t M, D;
        float eps;
        const TilingConfig* plan;
        float* workspace;  // Partial-sum records: the only memory the cores share besides X1/X2/Y
        uint32_t cores;    // Simulated cores the plan was made for
    };

    // One core's outcome, written once by its worker
    struct alignas(64) CoreResult {
        dsa::HardwareCycleTracker cycles;
        size_t spmBytes;
    };

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

    struct Cols { uint32_t off, len; };  // A band row's own columns inside its tile row

    // ---- Worker: one simulated core ------------------------------------------------------------
    struct Core {
        Core(const Args& a, uint32_t core)
            : x1(a.x1), x2(a.x2), gamma(a.gamma), bias(a.bias), y(a.y), M(a.M), D(a.D), eps(a.eps), plan(*a.plan),
              workspace(a.workspace), cores(a.cores), b(core) {}

        const S *x1, *x2, *gamma, *bias;
        S* y;
        uint32_t M, D;
        float eps;
        const TilingConfig& plan;
        float* workspace;
        uint32_t cores, b;

        dsa::TPipe pipe;
        dsa::TQue<dsa::QuePosition::VECIN, 2> qX1, qX2, qP;  // Double-buffered tiles; qP: gamma/beta chunks
        dsa::TBuf<dsa::QuePosition::VECCALC> bZ, bTmp, bPar, bRes, bMisc;
        dsa::LocalTensor<float> z, tmp, par, res, misc;
        uint32_t tmpCap = 0, quantum = 1, rep = 1, pitch = 0;

        void Execute(uint32_t numCores, CoreResult& out) {
            // A split plan's SyncAll waits for every core the plan was made for
            DSA_ASSERT(numCores == cores, "[DaePipeline]: the OpenMP team does not have the plan's core count");
            dsa::g_cycleTracker.Reset();
            Init();
            const CoreRange range = AdaptiveTiler::Range(plan, M, D, b, numCores);
            Phase1(range);
            if (plan.mode != TilingMode::ROW_PARALLEL) {  // Split-D: one barrier, passed by every core
                dsa::SyncAll();
                Phase2(range, numCores);
            }
            out.cycles = dsa::g_cycleTracker;
            out.spmBytes = pipe.GetTotalAllocatedBytes();
        }

        void Init() {
            const DaeLayout& L = plan.layout;
            pipe.InitBuffer(qX1, 2, L.tile);
            pipe.InitBuffer(qX2, 2, L.tile);
            if (L.paramQueue) pipe.InitBuffer(qP, 2, L.tile);
            // Exactly the plan's buffers: the claim is DaeLayout::Total(), the number the planner
            // fitted under 191 KB (an unplanned buffer overflows plans sized to the byte)
            if (L.z) { pipe.InitBuffer(bZ, L.z); z = bZ.Get<float>(); }
            pipe.InitBuffer(bTmp, L.tmp);
            tmp = bTmp.Get<float>();
            if (L.params) { pipe.InitBuffer(bPar, L.params); par = bPar.Get<float>(); }
            if (L.resident) { pipe.InitBuffer(bRes, L.resident); res = bRes.Get<float>(); }
            if (L.misc) { pipe.InitBuffer(bMisc, L.misc); misc = bMisc.Get<float>(); }
            tmpCap = L.tmp / sizeof(float) / 8 * 8;
            quantum = std::max<uint32_t>(1, dsa::DMA_ALIGN_BYTES / sizeof(S));
            rep = std::max(1u, plan.repRows);
            pitch = plan.pitch ? plan.pitch : D;
        }

        // ---- Before SyncAll: everything, or sweep 1 of shared rows ------------------------------
        void Phase1(const CoreRange& r) {
            if (plan.mode == TilingMode::SPLIT_COLUMNS) return BandPhase1(r);
            if (plan.tileRows) return RowTiles(r.rowA, r.rowZ);
            ColumnPhase1(r);
        }

        // ---- After SyncAll (Split-D): combine the partial sums, then sweep 2 --------------------
        void Phase2(const CoreRange& r, uint32_t nb) {
            if (plan.mode == TilingMode::SPLIT_COLUMNS) return BandPhase2(r, nb);
            ColumnPhase2(r, nb);
        }

        // ---- Row tiles: B whole rows per double-buffered tile, gamma/beta resident ---------------
        void RowTiles(uint32_t rA, uint32_t rZ) {
            if (rA >= rZ) return;
            const uint32_t B = plan.tileRows;
            float sums[AdaptiveTiler::ROW_GROUP];  // Row sums of squares, on the scalar side
            auto load = [&](uint32_t row) { LoadTile(uint64_t(row) * D, std::min(B, rZ - row) * D); };
            // Prologue [ARCH CHALLENGE 3]: tile 0, gamma/beta and tile 1 are all in flight before
            // the vector unit starts, and widening gamma/beta overlaps tile 1's DMA.
            load(rA);
            const bool staged = StageParams(0, D);
            if (rA + B < rZ) load(rA + B);
            WidenParams(0, D, staged);
            for (uint32_t row = rA; row < rZ; row += B) {
                const uint32_t k = std::min(B, rZ - row);
                dsa::LocalTensor<S> a = qX1.template DeQue<S>(), bt = qX2.template DeQue<S>();
                const dsa::LocalTensor<float> zt = ZOf(a);
                TileZ(zt, a, bt, k);
                qX2.FreeTensor(bt);
                // Every row sum of a group is issued before the group's first scaling
                for (uint32_t g = 0; g < k; g += AdaptiveTiler::ROW_GROUP) {
                    const uint32_t n = std::min(AdaptiveTiler::ROW_GROUP, k - g);
                    RowSums(zt[g * pitch], n, nullptr, sums);
                    ScaleRows(zt[g * pitch], n, nullptr, sums);
                }
                if (gamma) ApplyRows(zt, k, par, true);
                Narrow(a, zt, k * D);
                DmaOut(y + uint64_t(row) * D, a, k * D);
                qX1.FreeTensor(a);
                if (row + 2 * B < rZ) load(row + 2 * B);  // Tile k+2 streams in under tile k+1
            }
        }

        // Z = X1 + X2 (+ beta) for the k rows of a tile at pitch `pitch`
        void TileZ(dsa::LocalTensor<float> zt, dsa::LocalTensor<S> a, dsa::LocalTensor<S> bt, uint32_t k) {
            AddInputs(zt, a, bt, k * pitch);
            if (bias) ApplyRows(zt, k, par[BetaAt()], false);
        }

        // Sums of squares of n <= ROW_GROUP rows into out[0, n). `cols` limits band rows to their
        // own columns (null: full rows).
        void RowSums(dsa::LocalTensor<float> zt, uint32_t n, const Cols* cols, float* out) {
            if (pitch > tmpCap) {  // Long rows (full rows only): one row at a time
                for (uint32_t i = 0; i < n; ++i) out[i] = SumSquares(zt[i * pitch], pitch);
                return;
            }
            const uint32_t per = tmpCap / pitch;  // Rows whose squares fit the scratch chunk
            for (uint32_t r0 = 0; r0 < n; r0 += per) {
                const uint32_t m = std::min(per, n - r0);
                dsa::Mul(tmp, zt[r0 * pitch], zt[r0 * pitch], m * pitch);
                if (cols) {
                    for (uint32_t i = 0; i < m; ++i) {
                        const Cols c = cols[r0 + i];
                        out[r0 + i] = c.len ? dsa::VectorReduceSum(tmp[i * pitch + c.off], c.len) : 0.0f;
                    }
                    continue;
                }
                uint32_t len = pitch;
                if (DaeIsa::FoldFirst(m, len)) {  // One 8 -> 1 fold for all m rows, when cheaper
                    dsa::BlockReduceSum(tmp, tmp, m * len);
                    len /= 8;
                }
                for (uint32_t i = 0; i < m; ++i) out[r0 + i] = dsa::VectorReduceSum(tmp[i * len], len);
            }
        }

        // Z *= invRms(row) for n rows (band rows: their own columns): one VectorInvRms and one
        // Muls per row
        void ScaleRows(dsa::LocalTensor<float> zt, uint32_t n, const Cols* cols, const float* sumSq) {
            for (uint32_t i = 0; i < n; ++i) {
                const Cols c = cols ? cols[i] : Cols{0, pitch};
                if (c.len) dsa::Muls(zt[i * pitch + c.off], zt[i * pitch + c.off], InvRms(sumSq[i]), c.len);
            }
        }

        // zt (+|*)= a resident parameter block of `rep` rows, rep rows per instruction
        void ApplyRows(dsa::LocalTensor<float> zt, uint32_t k, dsa::LocalTensor<float> p, bool multiply) {
            for (uint32_t g = 0; g < k; g += rep) {
                const uint32_t n = std::min(rep, k - g) * pitch;
                if (multiply) dsa::Mul(zt[g * pitch], zt[g * pitch], p, n);
                else dsa::Add(zt[g * pitch], zt[g * pitch], p, n);
            }
        }

        // gamma/beta for columns [c0, c0 + w) as FP32 rows of `pitch` in `par` (gamma rows, then
        // beta rows), each replicated `rep` times. StageParams issues the DMAs (16-bit: into the
        // scratch buffer; FP32: straight into par) and WidenParams widens and replicates them, so
        // the caller can issue a tile's DMA in between.
        bool StageParams(uint32_t c0, uint32_t w) {
            if constexpr (std::is_same<C, F32>::value) {
                if (gamma) DmaIn(par, gamma + c0, w);
                if (bias) DmaIn(par[BetaAt()], bias + c0, w);
                return true;
            } else {
                const uint32_t off = StageOffset(w);
                if (2ull * off * sizeof(S) > plan.layout.tmp) return false;  // WidenParams streams them
                dsa::LocalTensor<S> st = bTmp.Get<S>();
                if (gamma) DmaIn(st, gamma + c0, w);
                if (bias) DmaIn(st[off], bias + c0, w);
                return true;
            }
        }

        void WidenParams(uint32_t c0, uint32_t w, bool staged) {
            if constexpr (!std::is_same<C, F32>::value) {
                dsa::LocalTensor<S> st = bTmp.Get<S>();
                if (staged) {
                    if (gamma) dsa::Cast(par, st, w, ToF32);
                    if (bias) dsa::Cast(par[BetaAt()], st[StageOffset(w)], w, ToF32);
                } else {  // Too long to stage both at once: one scratch-sized piece at a time
                    const uint32_t cap = plan.layout.tmp / sizeof(S) / quantum * quantum;
                    const S* src[2] = {gamma, bias};
                    for (uint32_t k = 0; k < 2; ++k) {
                        if (!src[k]) continue;
                        for (uint32_t o = 0; o < w; o += cap) {
                            const uint32_t m = std::min(cap, w - o);
                            DmaIn(st, src[k] + c0 + o, m);
                            dsa::Cast(par[k * BetaAt() + o], st, m, ToF32);
                        }
                    }
                }
            } else {
                (void)c0;
                (void)w;
                (void)staged;
            }
            if (gamma) Replicate(par);
            if (bias) Replicate(par[BetaAt()]);
        }

        uint32_t StageOffset(uint32_t w) const { return AdaptiveTiler::Align32(uint64_t(w) * sizeof(S)) / sizeof(S); }
        uint32_t BetaAt() const { return AdaptiveTiler::ParamBytes(rep, pitch) / 2 / sizeof(float); }  // Beta block

        // Rows [1, rep) := row 0 by doubling scratchpad-to-scratchpad DMA, no vector work (the
        // planner replicates only rows of whole 32-byte blocks)
        void Replicate(dsa::LocalTensor<float> v) {
            for (uint32_t h = 1; h < rep; h *= 2) dsa::DataCopy(v[h * pitch], v, std::min(h, rep - h) * pitch);
        }

        // ---- SPLIT_COLUMNS: a column band of every row, its FP32 Z resident across SyncAll ------
        void BandPhase1(const CoreRange& r) {
            if (r.rowZ <= r.rowA) return;  // No units: the zeroed records stand for this core
            const uint32_t c0 = AdaptiveTiler::BandBegin(r, quantum), w = AdaptiveTiler::BandEnd(r, quantum) - c0;
            DSA_ASSERT(w <= pitch, "[DaePipeline]: column band wider than the plan's pitch");
            const uint32_t B = plan.tileRows, R = AdaptiveTiler::RecordFloats(plan, M);
            float sums[AdaptiveTiler::ROW_GROUP];
            Cols cols[AdaptiveTiler::ROW_GROUP];
            auto load = [&](uint32_t row) { LoadBand(r, row, std::min(B, r.rowZ - row), c0); };
            load(r.rowA);
            const bool staged = StageParams(c0, w);
            if (r.rowA + B < r.rowZ) load(r.rowA + B);
            WidenParams(c0, w, staged);
            // One record of M partials in whole 32-byte blocks, for every core to gather; the
            // scalar unit writes each row's partial into it as the sum arrives
            dsa::Duplicate(misc, 0.0f, R);
            for (uint32_t row = r.rowA; row < r.rowZ; row += B) {
                const uint32_t k = std::min(B, r.rowZ - row);
                dsa::LocalTensor<S> a = qX1.template DeQue<S>(), bt = qX2.template DeQue<S>();
                const dsa::LocalTensor<float> zt = res[(row - r.rowA) * pitch];
                TileZ(zt, a, bt, k);
                qX1.FreeTensor(a);
                qX2.FreeTensor(bt);
                for (uint32_t g = 0; g < k; g += AdaptiveTiler::ROW_GROUP) {
                    const uint32_t n = std::min(AdaptiveTiler::ROW_GROUP, k - g);
                    BandCols(r, row + g, n, c0, cols);
                    RowSums(zt[g * pitch], n, cols, sums);
                    for (uint32_t i = 0; i < n; ++i) misc.SetValue(row + g + i, sums[i]);
                }
                if (row + 2 * B < r.rowZ) load(row + 2 * B);
            }
            dsa::DataCopy(workspace + size_t(b) * R, misc, R);
        }

        void BandPhase2(const CoreRange& r, uint32_t nb) {
            if (r.rowZ <= r.rowA) return;
            const uint32_t c0 = AdaptiveTiler::BandBegin(r, quantum), R = AdaptiveTiler::RecordFloats(plan, M);
            // Every core's partials in one DMA, summed by a fixed tree of vector adds (the same on
            // every core, so all owners of a row compute the same sigma); one VectorReduceSum per
            // row then hands each total to the scalar unit
            const dsa::LocalTensor<float> recs = misc[R];
            dsa::DataCopy(recs, workspace, nb * R);
            for (uint32_t n = nb; n > 1; n -= n / 2) dsa::Add(recs, recs, recs[(n - n / 2) * R], n / 2 * R);
            const uint32_t B = plan.tileRows;
            float sums[AdaptiveTiler::ROW_GROUP];
            Cols cols[AdaptiveTiler::ROW_GROUP];
            for (uint32_t row = r.rowA; row < r.rowZ; row += B) {
                const uint32_t k = std::min(B, r.rowZ - row);
                const dsa::LocalTensor<float> zt = res[(row - r.rowA) * pitch];
                for (uint32_t g = 0; g < k; g += AdaptiveTiler::ROW_GROUP) {
                    const uint32_t n = std::min(AdaptiveTiler::ROW_GROUP, k - g);
                    for (uint32_t i = 0; i < n; ++i) sums[i] = dsa::VectorReduceSum(recs[row + g + i], 1);
                    BandCols(r, row + g, n, c0, cols);
                    ScaleRows(zt[g * pitch], n, cols, sums);
                }
                if (gamma) ApplyRows(zt, k, par, true);
                StoreBand(r, row, k, c0, zt);
            }
        }

        // k band rows into one tile slot at pitch `pitch`: one whole-block DMA per row and input
        void LoadBand(const CoreRange& r, uint32_t row, uint32_t k, uint32_t c0) {
            dsa::LocalTensor<S> a = qX1.template AllocTensor<S>(), bt = qX2.template AllocTensor<S>();
            for (uint32_t i = 0; i < k; ++i) {
                const Cols c = BandCol(r, row + i, c0);
                if (!c.len) continue;
                const uint64_t e = uint64_t(row + i) * D + c0 + c.off;
                DmaIn(a[i * pitch + c.off], x1 + e, c.len);
                DmaIn(bt[i * pitch + c.off], x2 + e, c.len);
            }
            qX1.EnQue(a);
            qX2.EnQue(bt);
        }

        // Band rows back to Y, one DMA per row (FP32 straight from the resident Z)
        void StoreBand(const CoreRange& r, uint32_t row, uint32_t k, uint32_t c0, dsa::LocalTensor<float> zt) {
            dsa::LocalTensor<S> out;
            if constexpr (std::is_same<C, F32>::value) {
                out = zt;
            } else {
                out = qX1.template AllocTensor<S>();
                Narrow(out, zt, k * pitch);
            }
            for (uint32_t i = 0; i < k; ++i) {
                const Cols c = BandCol(r, row + i, c0);
                if (c.len) DmaOut(y + uint64_t(row + i) * D + c0 + c.off, out[i * pitch + c.off], c.len);
            }
            if constexpr (!std::is_same<C, F32>::value) qX1.FreeTensor(out);
        }

        // A band row's own columns inside its tile row (len 0: the core has none of that row)
        Cols BandCol(const CoreRange& r, uint32_t row, uint32_t c0) const {
            uint32_t cb, ce;
            AdaptiveTiler::BandRow(r, row, quantum, cb, ce);
            return cb < ce ? Cols{cb - c0, ce - cb} : Cols{0, 0};
        }

        void BandCols(const CoreRange& r, uint32_t row, uint32_t n, uint32_t c0, Cols* cols) const {
            for (uint32_t i = 0; i < n; ++i) cols[i] = BandCol(r, row + i, c0);
        }

        // ---- Column tiles: Split-D fragments and rows too long for a row tile --------------------
        void ColumnPhase1(const CoreRange& r) {
            const bool split = plan.mode == TilingMode::SPLIT_D;
            used = 0;
            for (uint32_t k = 0; k < r.nFrag; ++k) {
                const CoreRange::Fragment& f = r.frag[k];
                fragZ[k] = Claim(f.ce - f.cb);
                fragSum[k] = Sweep1(f.row, f.cb, f.ce, fragZ[k]);
            }
            if (split) {  // Two 32-byte records per core: {sum0, 0 x7}, {sum1, 0 x7}
                dsa::Duplicate(misc, 0.0f, 16);
                if (r.nFrag > 0) misc.SetValue(0, fragSum[0]);
                if (r.nFrag > 1) misc.SetValue(8, fragSum[1]);
                dsa::DataCopy(workspace + size_t(b) * 16, misc, 16);
            }
            for (uint32_t row = r.rowA; row < r.rowZ; ++row) {
                const size_t mark = used;
                const dsa::LocalTensor<float> zr = Claim(D);
                Sweep2(row, 0, D, zr, Sweep1(row, 0, D, zr), false);
                used = mark;
            }
            // The first gamma chunk of fragment 0 streams in while this core waits at SyncAll
            primed = split && r.nFrag > 0 && Resident(fragZ[0]) && gamma != nullptr;
            if (primed) {
                const uint64_t start = uint64_t(r.frag[0].row) * D + r.frag[0].cb;
                LoadParam(gamma + r.frag[0].cb, TileLength(start, start + (r.frag[0].ce - r.frag[0].cb)));
            }
        }

        void ColumnPhase2(const CoreRange& r, uint32_t nb) {
            for (uint32_t k = 0; k < r.nFrag; ++k) {
                const CoreRange::Fragment& f = r.frag[k];
                uint32_t first, last;
                AdaptiveTiler::RowOwners(plan, D, f.row, nb, first, last);
                // The row's partials are records [lo, hi]: `first` holds the row as its last
                // fragment, every later owner as its first, and the records in between are zero
                const uint32_t lo = 2 * first + std::max(1u, AdaptiveTiler::Range(plan, M, D, first, nb).nFrag) - 1;
                const uint32_t count = 2 * last - lo + 1;
                const dsa::LocalTensor<float> recs = misc[16];
                dsa::DataCopy(recs, workspace + size_t(lo) * 8, count * 8);
                const float total = dsa::VectorReduceSum(recs, count * 8);  // Same records, same order on every owner
                Sweep2(f.row, f.cb, f.ce, fragZ[k], total, k == 0 && primed);
            }
        }

        // Sweep 1 over row segment [cb, ce): Z = X1 + X2 + beta and sum(Z^2), Z kept in `zr` when
        // it is resident. X1, X2 and the beta chunk of tile k+1 are in flight while tile k is
        // processed.
        float Sweep1(uint32_t row, uint32_t cb, uint32_t ce, dsa::LocalTensor<float> zr) {
            const uint64_t base = uint64_t(row) * D, start = base + cb;
            const bool keep = Resident(zr);
            float sum = 0.0f;
            Tiles(start, base + ce, false,
                  [&](uint64_t e, uint32_t n) {
                      LoadTile(e, n);
                      if (bias) LoadParam(bias + (e - base), n);
                  },
                  [&](uint64_t e, uint32_t n) {
                      dsa::LocalTensor<S> a = qX1.template DeQue<S>(), bt = qX2.template DeQue<S>();
                      dsa::LocalTensor<float> zt = keep ? zr[static_cast<uint32_t>(e - start)] : z;
                      AddInputs(zt, a, bt, n);
                      qX1.FreeTensor(a);
                      qX2.FreeTensor(bt);
                      if (bias) UseParam(zt, n, false);
                      sum += SumSquares(zt, n);
                  });
            return sum;
        }

        // Sweep 2: Y = Z * invRms * gamma. From resident Z only gamma streams (its first chunk may
        // already be in flight: `primed`); otherwise X1/X2 and beta re-stream to recompute Z.
        void Sweep2(uint32_t row, uint32_t cb, uint32_t ce, dsa::LocalTensor<float> zr, float sumSq, bool primed) {
            const uint64_t base = uint64_t(row) * D, start = base + cb, end = base + ce;
            const float inv = InvRms(sumSq);
            if (Resident(zr)) {
                Tiles(start, end, primed,
                      [&](uint64_t e, uint32_t n) { if (gamma) LoadParam(gamma + (e - base), n); },
                      [&](uint64_t e, uint32_t n) {
                          const dsa::LocalTensor<float> zt = zr[static_cast<uint32_t>(e - start)];
                          dsa::Muls(zt, zt, inv, n);
                          if (gamma) UseParam(zt, n, true);
                          dsa::LocalTensor<S> out = qX1.template AllocTensor<S>();
                          Narrow(out, zt, n);
                          DmaOut(y + e, out, n);
                          qX1.FreeTensor(out);
                      });
                return;
            }
            Tiles(start, end, false,
                  [&](uint64_t e, uint32_t n) {
                      LoadTile(e, n);
                      if (bias) LoadParam(bias + (e - base), n);
                  },
                  [&](uint64_t e, uint32_t n) {
                      dsa::LocalTensor<S> a = qX1.template DeQue<S>(), bt = qX2.template DeQue<S>();
                      AddInputs(z, a, bt, n);
                      qX2.FreeTensor(bt);
                      if (bias) UseParam(z, n, false);
                      dsa::Muls(z, z, inv, n);
                      if (gamma) ApplyParamNow(z, gamma + (e - base), n, true);  // qP holds the next beta
                      Narrow(a, z, n);
                      DmaOut(y + e, a, n);
                      qX1.FreeTensor(a);
                  });
        }

        // Tiles over [start, end): `load` for tile k+1 is issued before tile k's `body`
        template <class Load, class Body>
        void Tiles(uint64_t start, uint64_t end, bool primed, Load load, Body body) {
            uint64_t e = start;
            uint32_t n = start < end ? TileLength(start, end) : 0;
            if (n && !primed) load(e, n);
            while (e < end) {
                const uint64_t next = e + n;
                const uint32_t nn = next < end ? TileLength(next, end) : 0;
                if (nn) load(next, nn);  // Prefetch
                body(e, n);
                e = next;
                n = nn;
            }
        }

        // At most tileElems elements, interior tile boundaries on the 32-byte grid
        uint32_t TileLength(uint64_t e, uint64_t end) const {
            return static_cast<uint32_t>(std::min(end, e / quantum * quantum + plan.tileElems) - e);
        }

        void LoadTile(uint64_t e, uint32_t n) {
            dsa::LocalTensor<S> a = qX1.template AllocTensor<S>();
            DmaIn(a, x1 + e, n);
            qX1.EnQue(a);
            dsa::LocalTensor<S> bt = qX2.template AllocTensor<S>();
            DmaIn(bt, x2 + e, n);
            qX2.EnQue(bt);
        }

        void LoadParam(const S* src, uint32_t n) {
            dsa::LocalTensor<S> pc = qP.template AllocTensor<S>();
            DmaIn(pc, src, n);
            qP.EnQue(pc);
        }

        // zt (+|*)= the parameter chunk at the head of qP
        void UseParam(dsa::LocalTensor<float> zt, uint32_t n, bool multiply) {
            dsa::LocalTensor<S> pc = qP.template DeQue<S>();
            Combine(zt, pc, n, multiply);
            qP.FreeTensor(pc);
        }

        // The same with a chunk loaded on the spot, outside the queue order
        void ApplyParamNow(dsa::LocalTensor<float> zt, const S* src, uint32_t n, bool multiply) {
            dsa::LocalTensor<S> pc = qP.template AllocTensor<S>();
            DmaIn(pc, src, n);
            Combine(zt, pc, n, multiply);
            qP.FreeTensor(pc);
        }

        // FP32 chunks combine directly; 16-bit chunks are widened through the scratch buffer
        void Combine(dsa::LocalTensor<float> zt, dsa::LocalTensor<S> pc, uint32_t n, bool multiply) {
            if constexpr (std::is_same<C, F32>::value) {
                if (multiply) dsa::Mul(zt, zt, pc, n);
                else dsa::Add(zt, zt, pc, n);
            } else {
                for (uint32_t o = 0; o < n; o += tmpCap) {
                    const uint32_t k = std::min(tmpCap, n - o);
                    dsa::Cast(tmp, pc[o], k, ToF32);
                    if (multiply) dsa::Mul(zt[o], zt[o], tmp, k);
                    else dsa::Add(zt[o], zt[o], tmp, k);
                }
            }
        }

        // Z = X1 + X2 in FP32 (FP32 row tiles in place; 16-bit widened into zt)
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

        // sum(Z^2) of one run in scratch-sized pieces, each folded 8 -> 1 first when cheaper
        float SumSquares(dsa::LocalTensor<float> zt, uint32_t n) {
            float total = 0.0f;
            for (uint32_t o = 0; o < n; o += tmpCap) {
                uint32_t k = std::min(tmpCap, n - o);
                dsa::Mul(tmp, zt[o], zt[o], k);
                if (DaeIsa::FoldFirst(1, k)) {
                    dsa::BlockReduceSum(tmp, tmp, k);
                    k /= 8;
                }
                total += dsa::VectorReduceSum(tmp, k);
            }
            return total;
        }

        void Narrow(dsa::LocalTensor<S> out, dsa::LocalTensor<float> zt, uint32_t n) {
            if constexpr (!std::is_same<C, F32>::value) dsa::Cast(out, zt, n, FromF32);
            else if (out.GetData() != zt.GetData()) dsa::Muls(out, zt, 1.0f, n);
        }

        dsa::LocalTensor<float> ZOf(dsa::LocalTensor<S> a) {
            if constexpr (std::is_same<C, F32>::value) return a;
            else return z;
        }

        float InvRms(float sumSq) const { return dsa::VectorInvRms(sumSq, D, eps); }

        // Resident Z for a column-tiled segment: a view into `res`, or an empty view when it does
        // not fit (sweep 2 then recomputes Z)
        dsa::LocalTensor<float> Claim(uint32_t n) {
            if (!plan.zResident || used + n > plan.zResident) return {};
            const dsa::LocalTensor<float> v = res[static_cast<uint32_t>(used)];
            used += n;
            return v;
        }

        static bool Resident(dsa::LocalTensor<float> v) { return v.GetData() != nullptr; }

        // Column tiles: state carried across SyncAll
        size_t used = 0;
        bool primed = false;
        dsa::LocalTensor<float> fragZ[2];  // The fragments' resident Z (empty: recomputed)
        float fragSum[2] = {0.0f, 0.0f};
    };
};

} // namespace hpc
