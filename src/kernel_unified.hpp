#pragma once

#include "hpc_vector_norm.hpp"
#include "adaptive_tiler.hpp"
#include "simd.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
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
    size_t bm, gm;
    Params At(uint32_t cb) const { return {b + (cb & bm), g + (cb & gm), bm, gm}; }
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
    const size_t bm = p.bm;
    auto zAt = [&](uint32_t c) { return Add(Add(Load(C{}, x1 + c), Load(C{}, x2 + c)), Load(PC{}, b + (c & bm))); };
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
            const V v = Add(Add(LoadN(C{}, x1 + c, k), LoadN(C{}, x2 + c, k)), LoadN(PC{}, b + (c & bm), k));
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
    const size_t bm = p.bm, gm = p.gm;
    const V s = Set1(invRms);
    auto out = [&](uint32_t c) {
        const V zv = kKeep ? Load(F32{}, z + c) : Add(Add(Load(C{}, x1 + c), Load(C{}, x2 + c)), Load(PC{}, b + (c & bm)));
        return Mul(Mul(zv, s), Load(PC{}, g + (c & gm)));
    };
    auto outN = [&](uint32_t c, uint32_t k) {
        const V zv = kKeep ? LoadN(F32{}, z + c, k)
                           : Add(Add(LoadN(C{}, x1 + c, k), LoadN(C{}, x2 + c, k)), LoadN(PC{}, b + (c & bm), k));
        return Mul(Mul(zv, s), LoadN(PC{}, g + (c & gm), k));
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
        // Inside an enclosing parallel region the caller already owns the cores
        const uint32_t P = omp_in_parallel() ? 1u
            : std::min<uint32_t>(AdaptiveTiler::MAX_THREADS, static_cast<uint32_t>(std::max(1, omp_get_max_threads())));
        ExecutePlan(x1, x2, gamma, bias, y, M, D, eps,
                    AdaptiveTiler::Plan(M, D, sizeof(S), P, AdaptiveTiler::LastLevelCacheBytes()));
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

    // Owner of work unit u under the balanced split [floor(U*t/nt), floor(U*(t+1)/nt))
    static uint32_t Owner(uint64_t u, uint64_t U, uint32_t nt) {
        return static_cast<uint32_t>(((u + 1) * nt - 1) / U);
    }

    template <bool kStream>
    static void Worker(const Job& j, uint32_t tid, uint32_t nt, RowPartial* parts) {
        const uint32_t upr = j.cfg.unitsPerRow, ue = j.cfg.unitElems, D = j.D;
        const uint64_t U = static_cast<uint64_t>(j.M) * upr, u0 = U * tid / nt, u1 = U * (tid + 1) / nt;

        // Thread range: complete rows [rA, rZ) plus up to two Split-D fragments shared with
        // neighbouring threads (a row head and/or a row tail).
        Fragment frag[2];
        uint32_t nFrag = 0, rA = 0, rZ = 0;
        if (u0 < u1) {
            const uint32_t r0 = static_cast<uint32_t>(u0 / upr), r1 = static_cast<uint32_t>((u1 - 1) / upr);
            const uint32_t c0 = static_cast<uint32_t>(std::min<uint64_t>(u0 % upr * ue, D));
            const uint32_t c1 = static_cast<uint32_t>(std::min<uint64_t>(((u1 - 1) % upr + 1) * ue, D));
            rA = r0;
            rZ = r1 + 1;
            if (c0 > 0 || (r0 == r1 && c1 < D)) { frag[nFrag++] = {r0, c0, r0 == r1 ? c1 : D, nullptr, 0.0}; ++rA; }
            if (r1 > r0 && c1 < D) { frag[nFrag++] = {r1, 0, c1, nullptr, 0.0}; --rZ; }
        }

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
                                      j.gamma ? scratch + D : ParamDefaults<F32>::kOne.data(), p16.bm, p16.gm};
                return Run<kStream>(j, p32, tid, nt, parts, frag, nFrag, rA, rZ, scratch + 2 * D, cap - 2 * D);
            }
        }
        Run<kStream>(j, p16, tid, nt, parts, frag, nFrag, rA, rZ, scratch, cap);
    }

    template <bool kStream, class PC>
    static void Run(const Job& j, const Params<PC>& p, uint32_t tid, uint32_t nt, RowPartial* parts,
                    Fragment* frag, uint32_t nFrag, uint32_t rA, uint32_t rZ, float* scratch, size_t cap) {
        const uint32_t upr = j.cfg.unitsPerRow, D = j.D;

        // Fragments first so their partial sums are published before the full rows run:
        // the barrier wait then overlaps local work instead of idling.
        size_t used = 0;
        for (uint32_t k = 0; k < nFrag; ++k) {
            const uint32_t len = frag[k].ce - frag[k].cb;
            if (used + len <= cap) { frag[k].z = scratch + used; used += len; }
            frag[k].sum = Pass1<kStream>(j, p, frag[k].row, frag[k].cb, len, frag[k].z);
        }
        const bool split = upr > 1;
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
            const uint64_t U = static_cast<uint64_t>(j.M) * upr;
            #pragma omp barrier
            for (uint32_t k = 0; k < nFrag; ++k) {
                const uint64_t first = static_cast<uint64_t>(frag[k].row) * upr;
                double total = 0.0;
                for (uint32_t t = Owner(first, U, nt); t <= Owner(first + upr - 1, U, nt); ++t) {
                    for (uint32_t q = 0; q < 2; ++q) total += parts[t].row[q] == frag[k].row ? parts[t].sum[q] : 0.0;
                }
                Pass2<kStream>(j, p, frag[k].row, frag[k].cb, frag[k].ce - frag[k].cb, frag[k].z, total);
            }
        }
        if (kStream) simd::Fence();
    }
};

} // namespace hpc
