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

// Bias / gamma operands of pass 2 in codec PC (the tensor codec, or FP32 after the per-call
// widening). A null tensor becomes a stride-0 constant (mask 0) so the hot loops stay branch-free.
template <class PC> struct Params {
    const typename PC::S *b, *g;
    size_t bMask, gMask;
    Params At(uint32_t cb) const { return {b + (cb & bMask), g + (cb & gMask), bMask, gMask}; }
};

// -----------------------------------------------------------------------------
// Pass 1: Z = X1 + X2 (optionally kept resident) -> sum(Z^2). Bias is not part of Z: it
// never enters the sum of squares (it is added after the normalization, in pass 2).
// 4 x W independent FP32 accumulators (the 64-lane sliding window on AVX-512) within
// 4096-element blocks, FP64 across blocks: no FMA latency chain, no drift on long rows.
// -----------------------------------------------------------------------------
constexpr uint32_t kPrefetchBytes = 512;

template <class C, bool kKeep, bool kPrefetch>
double SumSquares(const typename C::S* x1, const typename C::S* x2, float* z, uint32_t n) {
    using namespace simd;
    auto zAt = [&](uint32_t c) { return Add(Load(C{}, x1 + c), Load(C{}, x2 + c)); };
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
            const V v = Add(LoadN(C{}, x1 + c, k), LoadN(C{}, x2 + c, k));
            if (kKeep) StoreN(F32{}, z + c, v, k);
            a1 = Fma(v, v, a1);
            c = e;
        }
        total += static_cast<double>(Sum(Add(Add(a0, a1), Add(a2, a3))));
    }
    return total;
}

// -----------------------------------------------------------------------------
// Pass 2: Y = Z * invRms * gamma + bias, with Z read from the resident scratchpad (kKeep) or
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
        const V zv = kKeep ? Load(F32{}, z + c) : Add(Load(C{}, x1 + c), Load(C{}, x2 + c));
        return Fma(Mul(zv, s), Load(PC{}, g + (c & gMask)), Load(PC{}, b + (c & bMask)));
    };
    auto outN = [&](uint32_t c, uint32_t k) {
        const V zv = kKeep ? LoadN(F32{}, z + c, k) : Add(LoadN(C{}, x1 + c, k), LoadN(C{}, x2 + c, k));
        return Fma(Mul(zv, s), LoadN(PC{}, g + (c & gMask), k), LoadN(PC{}, b + (c & bMask), k));
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

    template <bool kStream>
    static double Pass1(const Job& j, uint32_t r, uint32_t cb, uint32_t n, float* z) {
        const size_t o = static_cast<size_t>(r) * j.D + cb;
        return z ? SumSquares<C, true, kStream>(j.x1 + o, j.x2 + o, z, n) : SumSquares<C, false, kStream>(j.x1 + o, j.x2 + o, nullptr, n);
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
            frag[k].sum = Pass1<kStream>(j, frag[k].row, frag[k].cb, len, frag[k].z);
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
                for (uint32_t k = 0; k < nb; ++k) sums[k] = Pass1<kStream>(j, i + k, 0, D, zb ? zb + static_cast<size_t>(k) * D : nullptr);
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
// target processor runs it, through include/dsa_runtime.hpp, in the target's API shapes
// (docs/TARGET_API_SHAPE.md). One OpenMP thread is one simulated core (GetBlockIdx).
//
// Global memory. The kernel's arguments are global-memory addresses (GM_ADDR) and 32-bit scalars;
// each address becomes a GlobalTensor through SetGlobalBuffer, the only place a typed pointer is
// formed, and every transfer is a DataCopy / DataCopyPad between a LocalTensor and a GlobalTensor
// indexed by element. Which buffers start on the 32-byte grid, the coordinator says in a launch flag:
// a transfer of whole blocks at both ends is a DataCopy, anything else a DataCopyPad descriptor.
// The tiling data travels as one more address and is copied in first, as GET_TILING_DATA does.
// Scratchpad is claimed as TPipe::InitBuffer(TBuf, bytes) only (no LocalMemAllocator [6.4]); every
// buffer and partition is a TBuf of its own, sliced with operator[] only, never through a raw pointer.
//
// Target semantics (docs/TARGET_MEASUREMENTS.md, section 6). No reduction returns a value, there is
// no scalar inverse RMS, and the scalar unit never reads the scratchpad: a row's sum of squares and
// its inverse RMS stay in the vector unit, instruction by instruction what AdaptiveTiler's cycle
// model counts:
//   sums of squares  every row's squares added column-wise down to one 64-lane repeat (a strided Add
//                    per column for all rows of a chunk), folded by BlockReduceSum into 8 partials
//                    per row, and a second BlockReduceSum packs 8 rows' partials into their sums.
//                    No ReduceSum on this path: it defines lane 0 of its 32-byte slot only [6.2]
//                    (this runtime poisons lanes 1..7), so a slot cannot be folded.
//   Rsqrt            an 11-bit table value [6.3], refined by Newton-Raphson in lanes (LaneInvRms)
//   Brcb             a block broadcast [6.1]: a repeat spreads 8 values over 8 blocks, so a group's
//                    packed inverse RMS fill the slots, one row's value per block
//   strided Mul/Add  one instruction per 64-lane column scales every row of a group by its own
//                    slot (src1BlkStride 0, one block per repeat), or applies the gamma / beta row
//                    to each (src1RepStride 0); its mask is one repeat's element count [6.8]
//   Cast             a rounding mode: CAST_NONE widens, CAST_RINT narrows to nearest-even
// Rows too long for a row tile (column tiles) reduce each segment with ReduceSum into lane 0 of a
// slot and read lane 0 alone. A tile's rows lie at RowPitch, D on the 32-byte grid, which is how a
// multi-row DataCopyPad descriptor lays them out [6.6]: every row starts a block, the padded load
// fills the rest of each row's last block with zeros, which add nothing to its sum, and every vector
// operand starts a block.
//
// Coordinator and workers [ARCH CHALLENGE 6]: DaePipeline::Execute runs on the calling
// thread. It takes a finished plan, owns the 64-byte-aligned reduction workspace and
// launches one worker per simulated core with flat arguments. A worker is freestanding: it
// allocates nothing, throws nothing (a broken invariant is a DSA_ASSERT trap), passes
// LocalTensor views by value and uses dsa::Min and dsa::Max, not <algorithm>.
//
// DAE v1.4 errata [ARCH CHALLENGE 7]: the egress DMA channel reads VECOUT buffers only, so every
// result and every published record leaves from a VECOUT buffer (qY, bRec), never from the VECIN
// input queues; an 8 -> 1 fold never writes what it reads (it lands in a partition of its own);
// and the worker has no scalar integer-to-float unit, so the coordinator passes invD = 1 / D at
// launch.
//
// DAE v1.5 [ARCH CHALLENGE 8]: every TQue lifecycle step costs the queue sequencer 625 cycles, so
// no kernel takes one. X1/X2, gamma/beta chunks and egress buffers are static rings (BufferRing)
// ordered by scoreboard flags, one event ID per slot, and tiles are software pipelined across
// them: `depth` tiles in flight, each slot's flags set and waited once per tile, no PIPE_ALL
// between tiles. A core whose rows fit one tile runs the same kernel with a single tile. The
// kernel launch frame holds only 64-bit addresses and 32-bit scalars (Trap #409), and every
// result reaches the egress channel through a V -> MTE3 flag.
// =============================================================================
// The target's kernel-argument type for a global-memory address (__gm__ uint8_t* there)
using GM_ADDR = uint8_t*;

// A global-memory address as the typed buffer SetGlobalBuffer takes: (__gm__ T*)addr on the target.
// Kernels form typed pointers here only, and only to hand them to SetGlobalBuffer.
template <class T>
inline T* GmBuffer(GM_ADDR addr) { return reinterpret_cast<T*>(addr); }

// The device element of a tensor codec: the target's half / bfloat16_t, so a Cast knows its formats
template <class C> struct DevElem;
template <> struct DevElem<F32> { using type = float; };
template <> struct DevElem<F16> { using type = dsa::half; };
template <> struct DevElem<BF16> { using type = dsa::bfloat16_t; };

// Static buffer ring [ARCH CHALLENGE 8]: `n` slots, each a static TBuf per way, handed out the way a
// TQue hands out its buffers (Alloc: the lowest free slot; DeQue: the oldest enqueued one), so a
// kernel keeps every schedule and every timeline of its TQue version. None of it runs on the queue
// sequencer (a TQue lifecycle step costs 625 cycles): the slot bookkeeping is the kernel's own, and
// scoreboard event flags order the pipes, one literal event ID per slot. READY passes a slot's data
// from its producer to its consumer, RELEASE hands the slot back to be overwritten; every flag set
// is waited exactly once. The TQue sanitizer's lifecycle traps are DSA_ASSERTs here. WAYS buffers
// share each slot's lifecycle: X1 and X2 of a tile travel together.
template <dsa::QuePosition POS, uint32_t WAYS, dsa::HardEvent READY, dsa::HardEvent RELEASE>
class BufferRing {
public:
    static constexpr uint32_t MAX_SLOTS = 4;  // EVENT_ID0..EVENT_ID3

    // `slots` buffers of `slotBytes` per way, on event IDs [firstEvent, firstEvent + slots)
    void Init(dsa::TPipe& pipe, uint32_t slots, uint32_t slotBytes, uint32_t firstEvent) {
        DSA_ASSERT(slots >= 1 && firstEvent + slots <= MAX_SLOTS, "[BufferRing]: a slot needs a literal event ID of its own (EVENT_ID0..EVENT_ID3)");
        n = slots, event0 = firstEvent, head = queued = 0;
        const uint32_t bytes = (slotBytes + dsa::DMA_ALIGN_BYTES - 1) / dsa::DMA_ALIGN_BYTES * dsa::DMA_ALIGN_BYTES;
        for (uint32_t w = 0; w < WAYS; ++w) {
            for (uint32_t s = 0; s < n; ++s) pipe.InitBuffer(buf[w][s], bytes);
        }
        for (uint32_t s = 0; s < MAX_SLOTS; ++s) state[s] = FREE, released[s] = false;
    }

    template <class T>
    dsa::LocalTensor<T> View(uint32_t slot, uint32_t way = 0) { return buf[way][slot].template Get<T>(); }

    // The lowest free slot, once its previous contents are released
    uint32_t Alloc() {
        for (uint32_t s = 0; s < n; ++s) {
            if (state[s] != FREE) continue;
            if (released[s]) {
                dsa::WaitFlag<RELEASE>(Event(s));
                released[s] = false;
            }
            state[s] = ALLOCATED;
            return s;
        }
        DSA_ASSERT(false, "[BufferRing]: every slot is in use (allocation beyond the ring's depth)");
        return 0;
    }
    // The slot's data is issued: its consumer may take it in FIFO order
    void EnQue(uint32_t slot) {
        DSA_ASSERT(slot < n && state[slot] == ALLOCATED, "[BufferRing]: EnQue of a slot that is not allocated");
        state[slot] = ENQUEUED;
        fifo[(head + queued++) % MAX_SLOTS] = slot;
        dsa::SetFlag<READY>(Event(slot));
    }
    uint32_t DeQue() {
        DSA_ASSERT(queued > 0, "[BufferRing]: DeQue on an empty ring");
        const uint32_t slot = fifo[head];
        head = (head + 1) % MAX_SLOTS, --queued;
        state[slot] = DEQUEUED;
        dsa::WaitFlag<READY>(Event(slot));
        return slot;
    }
    void Free(uint32_t slot) {
        DSA_ASSERT(slot < n && (state[slot] == ALLOCATED || state[slot] == DEQUEUED),
                   "[BufferRing]: freeing a slot that is free or still in flight");
        state[slot] = FREE, released[slot] = true;
        dsa::SetFlag<RELEASE>(Event(slot));
    }
    // End of the kernel: every slot consumed and freed, every release flag waited
    void Drain() {
        DSA_ASSERT(queued == 0, "[BufferRing]: a slot was enqueued and never taken");
        for (uint32_t s = 0; s < n; ++s) {
            DSA_ASSERT(state[s] == FREE, "[BufferRing]: a slot is still in use at the end of the kernel");
            if (released[s]) {
                dsa::WaitFlag<RELEASE>(Event(s));
                released[s] = false;
            }
        }
    }

private:
    enum : uint8_t { FREE, ALLOCATED, ENQUEUED, DEQUEUED };
    uint8_t Event(uint32_t slot) const { return static_cast<uint8_t>(dsa::EVENT_ID0 + event0 + slot); }

    dsa::TBuf<POS> buf[WAYS][MAX_SLOTS];
    uint32_t n = 0, event0 = 0, head = 0, queued = 0;
    uint8_t state[MAX_SLOTS] = {};
    uint32_t fifo[MAX_SLOTS] = {};
    bool released[MAX_SLOTS] = {};  // A RELEASE flag is set and not yet waited
};

struct DaeStats {
    uint64_t vectorCycles = 0;  // Busiest core: vector + scalar stalls + barriers + queue sequencer
    uint64_t queueCycles = 0;   // Most TQue sequencer cycles of any core (no kernel takes a TQue step)
    uint64_t dmaBytes = 0;      // All cores
    uint64_t dmaTransfers = 0;
    uint64_t padTransfers = 0;  // Transfers that were not whole 32-byte blocks
    uint64_t barriers = 0;      // Busiest core
    uint64_t scalarStalls = 0;  // GetValue V->S stalls, all cores
    size_t spmBytes = 0;        // Largest per-core scratchpad claim
    dsa::HardwareCycleTracker busiest;  // Full cycle breakdown of the busiest core
    dsa::TimelineSummary timeline;      // Timeline of the core that finishes last
};

template <class C>
class DaePipeline {
public:
    using S = typename C::S;                   // Host element
    using T = typename DevElem<C>::type;       // Device element: the same bits
    static_assert(sizeof(T) == sizeof(S), "the device element carries the host element's bits");

    // Launch flags: which parameters exist, and which system-memory buffers start a 32-byte block
    enum : uint32_t {
        HAS_GAMMA = 1u, HAS_BETA = 2u, X1_ALIGNED = 4u, X2_ALIGNED = 8u, GAMMA_ALIGNED = 16u, BETA_ALIGNED = 32u, Y_ALIGNED = 64u
    };

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
        const float invD = 1.0f / static_cast<float>(D);  // Converted here: the worker has no scalar int-to-float unit
        const uint32_t cores = Cores(plan);
        const uint32_t flags = (gamma ? HAS_GAMMA : 0u) | (bias ? HAS_BETA : 0u) | (OnGrid(x1) ? X1_ALIGNED : 0u) |
                               (OnGrid(x2) ? X2_ALIGNED : 0u) | (OnGrid(gamma) ? GAMMA_ALIGNED : 0u) | (OnGrid(bias) ? BETA_ALIGNED : 0u) |
                               (OnGrid(y) ? Y_ALIGNED : 0u);
        // The launch's memory system: its cores share the bandwidth of its working set (the one the
        // planner priced every transfer with)
        dsa::g_memory = AdaptiveTiler::Memory(plan, M, D, sizeof(S));
        CoreResult results[AdaptiveTiler::MAX_THREADS];
        Launch(cores, Gm(x1), Gm(x2), Gm(gamma), Gm(bias), Gm(y), Gm(workspace), Gm(&plan), M, D, eps, invD, flags, cores, results);
        for (uint32_t b = 0; b < cores; ++b) {
            const dsa::HardwareCycleTracker& t = results[b].cycles;
            if (t.GetTotalVectorCycles() >= stats.vectorCycles) {
                stats.vectorCycles = t.GetTotalVectorCycles();
                stats.busiest = t;
            }
            stats.queueCycles = std::max(stats.queueCycles, t.queueSequencerCycles);
            stats.scalarStalls += t.scalarStallCount;
            stats.dmaBytes += t.dmaBytesMoved;
            stats.dmaTransfers += t.dmaTransfers;
            stats.padTransfers += t.padTransfers;
            stats.barriers = std::max(stats.barriers, t.barrierCount);
            stats.spmBytes = std::max(stats.spmBytes, results[b].spmBytes);
            if (results[b].timeline.finish >= stats.timeline.finish) stats.timeline = results[b].timeline;
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

    // Host side: a buffer's address as a kernel argument, and whether it starts a 32-byte block
    static GM_ADDR Gm(const void* p) { return static_cast<GM_ADDR>(const_cast<void*>(p)); }
    static bool OnGrid(const void* p) { return p && reinterpret_cast<uintptr_t>(p) % dsa::DMA_ALIGN_BYTES == 0; }

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

    // One core's outcome, written once by its worker: the simulator's telemetry, not kernel data
    struct alignas(64) CoreResult {
        dsa::HardwareCycleTracker cycles;
        dsa::TimelineSummary timeline;
        size_t spmBytes;
    };

    // Kernel launch [ARCH CHALLENGE 8]: the constant frame holds 64-bit addresses and 32-bit
    // scalars only, since an aggregate over 32 bytes overflows it (Trap #409). The plan travels as
    // the address of its tiling data, like any other buffer.
    template <class... A>
    static void Launch(uint32_t blockDim, A... args) {
        static_assert(((std::is_pointer<A>::value || std::is_arithmetic<A>::value) && ...),
                      "kernel launch arguments must be flat scalars and addresses");
        (dsa::ValidateLaunchArgs(args), ...);
        #pragma omp parallel num_threads(blockDim)
        Kernel(args...);
    }

    // GET_TILING_DATA: the plan, copied from the global memory its address points to
    static TilingConfig GetTilingData(GM_ADDR tiling) {
        TilingConfig plan;
        std::memcpy(&plan, tiling, sizeof plan);
        return plan;
    }

    // The kernel on one simulated core. workspace: the partial-sum records, the only memory the cores
    // share besides X1/X2/Y; invD = 1 / D, precomputed by the coordinator; flags: HAS_* and *_ALIGNED;
    // cores: the simulated cores the plan was made for. `telemetry` is the simulator's out-parameter
    // (each core's cycle counts and timeline), not part of the kernel.
    static void Kernel(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling,
                       uint32_t M, uint32_t D, float eps, float invD, uint32_t flags, uint32_t cores, CoreResult* telemetry) {
        const uint32_t b = dsa::GetBlockIdx();
        Core core(GetTilingData(tiling), M, D, eps, invD, flags, cores, b);
        core.Bind(x1, x2, gamma, beta, y, workspace);
        core.Execute(dsa::GetBlockNum(), telemetry[b]);
    }

    // inv = 1 / sqrt(sums * invD + eps) for k rows in vector lanes, never through the scalar unit:
    // the Rsqrt table value, refined by Newton-Raphson, inv *= 1.5 - (mean / 2) inv^2, as many steps
    // as the output's precision needs (AdaptiveTiler::NewtonSteps). `sums` ends up holding -mean / 2
    // (`mean` may be `sums` itself); `nr` is scratch. AdaptiveTiler::LaneRmsCycles counts it.
    static void LaneInvRms(dsa::LocalTensor<float> sums, dsa::LocalTensor<float> mean, dsa::LocalTensor<float> inv,
                           dsa::LocalTensor<float> nr, uint32_t k, float invD, float eps) {
        dsa::Muls(mean, sums, invD, k);
        dsa::Adds(mean, mean, eps, k);
        dsa::Rsqrt(inv, mean, k);
        const uint32_t steps = AdaptiveTiler::NewtonSteps(sizeof(S));
        if (!steps) return;
        dsa::Muls(sums, mean, -0.5f, k);  // -mean / 2, in the lanes of the spent row sums
        for (uint32_t i = 0; i < steps; ++i) {
            dsa::Mul(nr, inv, inv, k);
            dsa::Mul(nr, nr, sums, k);
            dsa::Adds(nr, nr, 1.5f, k);
            dsa::Mul(inv, inv, nr, k);
        }
    }

    struct Cols { uint32_t off, len; };  // A band row's own columns inside its tile row

    // ---- Worker: one simulated core ------------------------------------------------------------
    struct Core {
        static constexpr bool kF32 = std::is_same<C, F32>::value;
        static constexpr uint32_t G = AdaptiveTiler::ROW_GROUP, LANES = AdaptiveTiler::LANES, CHUNK = AdaptiveTiler::TMP_FLOATS;

        Core(const TilingConfig& plan, uint32_t M, uint32_t D, float eps, float invD, uint32_t flags, uint32_t cores, uint32_t core)
            : plan(plan), M(M), D(D), eps(eps), invD(invD), flags(flags), cores(cores), b(core),
              hasGamma((flags & HAS_GAMMA) != 0), hasBeta((flags & HAS_BETA) != 0) {}

        const TilingConfig plan;
        uint32_t M, D;
        float eps, invD;
        uint32_t flags, cores, b;
        bool hasGamma, hasBeta;
        dsa::GlobalTensor<T> x1Gm, x2Gm, gammaGm, betaGm, yGm;
        dsa::GlobalTensor<float> wsGm;

        dsa::TPipe pipe;
        // Static rings, no queue sequencer (BufferRing). Ingress: X1 and X2 of `depth` tiles in
        // flight, and for column tiles the gamma and beta chunks of a tile, which travel together
        // (event IDs after the tiles'). Egress: every result leaves from a slot of qY.
        BufferRing<dsa::QuePosition::VECIN, 2, dsa::HardEvent::MTE2_V, dsa::HardEvent::V_MTE2> qX;
        BufferRing<dsa::QuePosition::VECIN, 2, dsa::HardEvent::MTE2_V, dsa::HardEvent::V_MTE2> qP;
        BufferRing<dsa::QuePosition::VECOUT, 1, dsa::HardEvent::V_MTE3, dsa::HardEvent::MTE3_V> qY;
        dsa::TBuf<dsa::QuePosition::VECCALC> bZ, bPar, bRes, bMisc;
        dsa::TBuf<dsa::QuePosition::VECOUT> bRec;  // Split-D: the record this core publishes
        // The scratch partitions (AdaptiveTiler::SCRATCH_BYTES in all), a TBuf each, so an operation
        // that runs past one traps
        dsa::TBuf<dsa::QuePosition::VECCALC> bChunk, bFold, bSlots, bSums, bInv, bNr, bWork;
        dsa::LocalTensor<float> z, par, res, misc, rec;
        dsa::LocalTensor<float> chunk, fold, slots, spare, sums, inv, nr, work;
        uint32_t quantum = 1, pitch = 0;
        // Egress slots stay taken until outDepth - 1 later results took theirs, so results rotate
        // through all of them and a store never holds up the next result
        uint32_t held[4] = {};
        uint32_t nHeld = 0;

        // The global buffers, from the kernel's address arguments
        void Bind(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR workspace) {
            const uint64_t n = uint64_t(M) * D;
            x1Gm.SetGlobalBuffer(GmBuffer<T>(x1), n);
            x2Gm.SetGlobalBuffer(GmBuffer<T>(x2), n);
            yGm.SetGlobalBuffer(GmBuffer<T>(y), n);
            if (hasGamma) gammaGm.SetGlobalBuffer(GmBuffer<T>(gamma), D);
            if (hasBeta) betaGm.SetGlobalBuffer(GmBuffer<T>(beta), D);
            if (workspace) wsGm.SetGlobalBuffer(GmBuffer<float>(workspace), size_t(cores) * AdaptiveTiler::RecordFloats(plan, M));
        }

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
            while (nHeld) qY.Free(held[--nHeld]);
            qX.Drain();
            qP.Drain();
            qY.Drain();
            dsa::CrossPipe<dsa::HardEvent::MTE3_S>(dsa::EVENT_ID0);  // The kernel ends once Y has landed
            out.cycles = dsa::g_cycleTracker;
            out.timeline = dsa::g_timeline.Summary();
            out.spmBytes = pipe.GetTotalAllocatedBytes();
        }

        template <class B>
        dsa::LocalTensor<float> Carve(B& buf, uint32_t floats) {
            pipe.InitBuffer(buf, size_t(floats) * sizeof(float));
            return buf.template Get<float>();
        }

        void Init() {
            const DaeLayout& L = plan.layout;
            DSA_ASSERT(L.out && L.outDepth >= 1 && L.outDepth <= 4 && L.tmp == AdaptiveTiler::SCRATCH_BYTES,
                       "[DaePipeline]: the plan's layout has no egress buffer or no reduction partitions");
            qX.Init(pipe, L.depth, L.tile, 0);
            if (L.paramQueue) qP.Init(pipe, L.depth, L.tile, L.depth);
            qY.Init(pipe, L.outDepth, L.out, 0);
            // Exactly the plan's buffers: the claim is DaeLayout::Total(), the number the planner
            // fitted under 191 KB (an unplanned buffer overflows plans sized to the byte)
            if (L.z) { pipe.InitBuffer(bZ, L.z); z = bZ.Get<float>(); }
            chunk = Carve(bChunk, CHUNK);
            fold = Carve(bFold, AdaptiveTiler::FOLD_FLOATS);
            slots = Carve(bSlots, AdaptiveTiler::SLOT_FLOATS);
            spare = slots[8 * G];  // The block after a group's slots
            sums = Carve(bSums, AdaptiveTiler::LANE_FLOATS);
            inv = Carve(bInv, AdaptiveTiler::LANE_FLOATS);
            nr = Carve(bNr, AdaptiveTiler::LANE_FLOATS);
            work = Carve(bWork, AdaptiveTiler::WORK_FLOATS);
            if (L.params) { pipe.InitBuffer(bPar, L.params); par = bPar.Get<float>(); }
            if (L.resident) { pipe.InitBuffer(bRes, L.resident); res = bRes.Get<float>(); }
            if (L.misc) { pipe.InitBuffer(bMisc, L.misc); misc = bMisc.Get<float>(); }
            if (L.rec) { pipe.InitBuffer(bRec, L.rec); rec = bRec.Get<float>(); }
            quantum = dsa::DMA_ALIGN_BYTES / sizeof(T);
            pitch = plan.pitch ? plan.pitch : D;
        }

        // ---- System memory ------------------------------------------------------------------------
        // A transfer of n elements at element `at` of a buffer whose base is on the grid (`aligned`):
        // whole 32-byte blocks at both ends, so a DataCopy
        bool Blocks(uint32_t aligned, uint64_t at, uint64_t n) const {
            return (flags & aligned) && at * sizeof(T) % dsa::DMA_ALIGN_BYTES == 0 && n * sizeof(T) % dsa::DMA_ALIGN_BYTES == 0;
        }
        // n elements from element `at` of src; a padded transfer fills the rest of its last block with zeros
        void DmaIn(dsa::LocalTensor<T> dst, const dsa::GlobalTensor<T>& src, uint64_t at, uint32_t n, uint32_t aligned) {
            if (Blocks(aligned, at, n)) return dsa::DataCopy(dst, src[at], n);
            dsa::DataCopyPad(dst, src[at], dsa::DataCopyExtParams{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0},
                             dsa::DataCopyPadExtParams<T>{true, 0, 0, T(0)});
        }
        // Egress: the vector unit hands the finished buffer to the egress channel (V -> MTE3 flag);
        // the channel reads VECOUT buffers only
        void DmaOut(uint64_t at, dsa::LocalTensor<T> src, uint32_t n) {
            dsa::CrossPipe<dsa::HardEvent::V_MTE3>(dsa::EVENT_ID0);
            if (Blocks(Y_ALIGNED, at, n)) return dsa::DataCopy(yGm[at], src, n);
            dsa::DataCopyPad(yGm[at], src, dsa::DataCopyExtParams{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0});
        }
        // A record of `floats` (whole blocks) into the workspace at `at`
        void RecordOut(uint64_t at, dsa::LocalTensor<float> src, uint32_t floats) {
            dsa::CrossPipe<dsa::HardEvent::V_MTE3>(dsa::EVENT_ID0);
            dsa::DataCopy(wsGm[at], src, floats);
        }

        // The next egress slot for a result; Release it once its store is issued
        uint32_t TakeOut() { return qY.Alloc(); }
        dsa::LocalTensor<T> Out(uint32_t slot) { return qY.template View<T>(slot); }
        void ReleaseOut(uint32_t slot) {
            held[nHeld++] = slot;
            if (nHeld < plan.layout.outDepth) return;
            qY.Free(held[0]);  // The oldest result's buffer: the next one to be taken
            for (uint32_t i = 1; i < nHeld; ++i) held[i - 1] = held[i];
            --nHeld;
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

        // ---- Row tiles: whole rows per tile at the pitch, `depth` tiles in flight, gamma/beta resident
        void RowTiles(uint32_t rA, uint32_t rZ) {
            if (rA >= rZ) return;
            const AdaptiveTiler::RowSchedule tiles{rA, rZ, plan.tileRows, plan.headRows, plan.tailRows};
            uint32_t next = rA;  // The next tile to load
            auto load = [&] { const uint32_t k = tiles.Rows(next); LoadRows(next, k); next += k; };
            auto refill = [&] { if (next < rZ) load(); };
            // Prologue [ARCH CHALLENGE 3]: the first `depth` tiles and gamma/beta are all in
            // flight before the vector unit starts; widening gamma/beta overlaps their DMA.
            bool staged = plan.paramsFirst && StageParams(0, D);
            load();
            if (!plan.paramsFirst) staged = StageParams(0, D);
            for (uint32_t i = 1; i < plan.layout.depth && next < rZ; ++i) load();
            WidenParams(0, D, staged);
            for (uint32_t row = rA; row < rZ;) {
                const uint32_t k = tiles.Rows(row), t = qX.DeQue();
                // 16-bit: Z in the FP32 Z tile, narrowed into an egress buffer at the end. FP32: Z is
                // built in the egress buffer itself, so the result needs no copy.
                uint32_t o = 0;
                dsa::LocalTensor<float> zt = z;
                if constexpr (kF32) zt = Out(o = TakeOut());
                AddInputs(zt, qX.template View<T>(t, 0), qX.template View<T>(t, 1), k * pitch);  // Z = X1 + X2
                qX.Free(t);  // Both inputs are consumed: Y never lives in an input buffer
                if (plan.earlyLoads) refill();  // X1/X2 of a later tile stream in now
                for (uint32_t g = 0; g < k; g += G) {  // Each row group normalized, then * gamma + beta
                    const uint32_t n = dsa::Min(G, k - g);
                    NormalizeRows(zt[g * pitch], n);
                    Affine(zt[g * pitch], n);
                }
                if constexpr (!kF32) {
                    o = TakeOut();
                    Narrow(Out(o), zt, k * pitch);
                }
                StoreRows(uint64_t(row) * D, Out(o), k);  // Egress from VECOUT
                ReleaseOut(o);
                if (!plan.earlyLoads) refill();
                row += k;
            }
        }

        // X1 and X2 of k rows into the next tile slot
        void LoadRows(uint32_t row, uint32_t k) {
            const uint32_t t = qX.Alloc();
            DmaRowsIn(qX.template View<T>(t, 0), x1Gm, uint64_t(row) * D, k, X1_ALIGNED);
            DmaRowsIn(qX.template View<T>(t, 1), x2Gm, uint64_t(row) * D, k, X2_ALIGNED);
            qX.EnQue(t);
        }

        // k rows of D elements, one transfer each way. Rows at the pitch: where D is off the 32-byte grid
        // (or the base is), one multi-row DataCopyPad descriptor, which starts every row on a block and
        // fills the rest of its last block with zeros [6.6]
        void DmaRowsIn(dsa::LocalTensor<T> dst, const dsa::GlobalTensor<T>& src, uint64_t at, uint32_t k, uint32_t aligned) {
            if (pitch == D && Blocks(aligned, at, uint64_t(k) * D)) return dsa::DataCopy(dst, src[at], k * D);
            dsa::DataCopyPad(dst, src[at], RowsDescriptor(k), dsa::DataCopyPadExtParams<T>{true, 0, 0, T(0)});
        }
        void StoreRows(uint64_t at, dsa::LocalTensor<T> src, uint32_t k) {
            dsa::CrossPipe<dsa::HardEvent::V_MTE3>(dsa::EVENT_ID0);
            if (pitch == D && Blocks(Y_ALIGNED, at, uint64_t(k) * D)) return dsa::DataCopy(yGm[at], src, k * D);
            dsa::DataCopyPad(yGm[at], src, RowsDescriptor(k));
        }
        dsa::DataCopyExtParams RowsDescriptor(uint32_t k) const {
            DSA_ASSERT(k <= 0xFFFFu, "[DaePipeline]: a DataCopyPad descriptor moves at most 65,535 rows");
            return {static_cast<uint16_t>(k), static_cast<uint32_t>(D * sizeof(T)), 0, 0, 0};
        }

        // Z /= sigma for n <= ROW_GROUP rows at the pitch (AdaptiveTiler::GroupNormCycles): 8 partials
        // per row into the slots, one fold that packs them into the rows' sums, the inverse RMS in
        // lanes, spread over the slots by Brcb (a block per row), then one strided Mul per 64-lane column
        void NormalizeRows(dsa::LocalTensor<float> zt, uint32_t n) {
            const uint32_t reps = static_cast<uint32_t>(AdaptiveTiler::CeilDiv(n, 8));
            RowPartials(zt, n);
            dsa::BlockReduceSum(sums, slots, static_cast<uint8_t>(reps), LANES, 1, 1, 8);  // Row i's 8 partials -> sums[i]
            LaneInvRms(sums, sums, inv, nr, n, invD, eps);
            dsa::Brcb(slots, inv, reps);
            RowOp(zt, n, slots, true, true);
        }

        // Row i's 8 partial sums of squares into slot i, for n rows at the pitch (AdaptiveTiler::
        // RowPartialCycles): rows whose squares fit the chunk two or more at a time go together: their
        // squares (one Mul; rows narrower than a repeat zero-padded to one, one strided Mul), each row's
        // 64-lane columns added onto its first (ColumnAdds), one fold of every row; a wider row goes
        // alone, in chunk-sized pieces (PieceSum) accumulated in the work partition, then its fold
        void RowPartials(dsa::LocalTensor<float> zt, uint32_t n) {
            const uint32_t P = static_cast<uint32_t>(AdaptiveTiler::ChunkPitch(pitch)), per = CHUNK / P;
            if (AdaptiveTiler::ChunkRows(pitch)) {
                for (uint32_t r0 = 0; r0 < n; r0 += per) {
                    const uint32_t m = dsa::Min(per, n - r0);
                    const dsa::LocalTensor<float> z0 = zt[r0 * pitch];
                    if (pitch < LANES) {
                        dsa::Duplicate(chunk, 0.0f, m * LANES);
                        const uint8_t w = static_cast<uint8_t>(pitch / 8);
                        dsa::Mul(chunk, z0, z0, pitch, static_cast<uint8_t>(m), dsa::BinaryRepeatParams{1, 1, 1, 8, w, w});
                    } else {
                        dsa::Mul(chunk, z0, z0, m * pitch);
                    }
                    ChunkPartials(m, P, slots[8 * r0]);
                }
                return;
            }
            const uint32_t pieces = static_cast<uint32_t>(AdaptiveTiler::CeilDiv(pitch, CHUNK));
            for (uint32_t i = 0; i < n; ++i) {
                const dsa::LocalTensor<float> zr = zt[i * pitch];
                dsa::LocalTensor<float> cols;
                for (uint32_t o = 0; o < pitch; o += CHUNK) {
                    cols = PieceColumns(zr[o], dsa::Min(CHUNK, pitch - o));
                    if (pieces == 1) continue;
                    if (o == 0) dsa::Adds(work, cols, 0.0f, LANES);
                    else dsa::Add(work, work, cols, LANES);
                }
                dsa::BlockReduceSum(slots[8 * i], pieces > 1 ? work : cols, 1, LANES, 1, 1, 8);
            }
        }

        // m rows' squares in the chunk at pitch P down to 8 partials each at dst (AdaptiveTiler::
        // ChunkPartialCycles): rows of whole repeats and at least 512 lanes fold 8 -> 1 first (every
        // folded row starts a block and fills a repeat), then every row's 64-lane columns are added onto
        // its first, then each row folds into its 8 partials
        void ChunkPartials(uint32_t m, uint32_t P, dsa::LocalTensor<float> dst) {
            if (AdaptiveTiler::FoldFirstRows(P)) {
                dsa::BlockReduceSum(fold, chunk, static_cast<uint8_t>(m * P / LANES), LANES, 1, 1, 8);
                ColumnAdds(fold, m, P / 8);
                dsa::BlockReduceSum(dst, fold, static_cast<uint8_t>(m), LANES, 1, 1, static_cast<uint8_t>(P / 64));
                return;
            }
            ColumnAdds(chunk, m, P);
            dsa::BlockReduceSum(dst, chunk, static_cast<uint8_t>(m), LANES, 1, 1, static_cast<uint8_t>(P / 8));
        }

        // Each of m rows at pitch P in `buf` += its later 64-lane columns: one strided Add per column, a
        // repeat per row
        void ColumnAdds(dsa::LocalTensor<float> buf, uint32_t m, uint32_t P) {
            const uint8_t rs = static_cast<uint8_t>(P / 8);
            const dsa::BinaryRepeatParams rp{1, 1, 1, rs, rs, rs};
            for (uint32_t c = LANES; c < P; c += LANES) dsa::Add(buf, buf, buf[c], dsa::Min(LANES, P - c), static_cast<uint8_t>(m), rp);
        }

        // The 64-lane column sums of one piece of a long row (AdaptiveTiler::PieceColumnCycles), in the
        // partition it returns: folded 8 -> 1 first where the piece allows, else halved (PieceSum)
        dsa::LocalTensor<float> PieceColumns(dsa::LocalTensor<float> zr, uint32_t L) {
            if (AdaptiveTiler::FoldFirstRows(L)) {
                dsa::Mul(chunk, zr, zr, L);
                dsa::BlockReduceSum(fold, chunk, static_cast<uint8_t>(L / LANES), LANES, 1, 1, 8);
                ColumnAdds(fold, 1, L / 8);
                return fold;
            }
            PieceSum(zr, L, static_cast<uint32_t>(AdaptiveTiler::PieceZeros(L)), static_cast<uint32_t>(AdaptiveTiler::PiecePad(L)), 0);
            return chunk;
        }

        // chunk[0, 64) = the 64-lane column sums of a piece (AdaptiveTiler::PieceCycles): `zeros` lanes of
        // the chunk zeroed up to P2, L squares of zr written from chunk offset `off`, then the P2 lanes
        // halved down to one repeat
        void PieceSum(dsa::LocalTensor<float> zr, uint32_t L, uint32_t zeros, uint32_t P2, uint32_t off) {
            if (zeros) dsa::Duplicate(chunk[P2 - zeros], 0.0f, zeros);
            if (L) dsa::Mul(chunk[off], zr, zr, L);
            for (uint32_t h = P2 / 2; h >= LANES; h /= 2) dsa::Add(chunk, chunk, chunk[h], h);
        }

        // n rows at the pitch (*|+)= a block per row (`perRow`: row i's slot, as Brcb spread it) or one
        // parameter row (AdaptiveTiler::RowOpCycles): one strided instruction per 64-lane column, a
        // repeat per row, where the 8-bit repeat stride reaches the rows and that is no dearer; else
        // row by row
        void RowOp(dsa::LocalTensor<float> zt, uint32_t n, dsa::LocalTensor<float> src, bool perRow, bool multiply) {
            if (AdaptiveTiler::StridedRowOp(n, pitch, perRow)) {
                const uint8_t w = static_cast<uint8_t>(pitch / 8);
                const dsa::BinaryRepeatParams rp = perRow ? dsa::BinaryRepeatParams{1, 1, 0, w, w, 1}   // One block per row
                                                          : dsa::BinaryRepeatParams{1, 1, 1, w, w, 0};  // The same row for all
                for (uint32_t c = 0; c < pitch; c += LANES) {
                    const uint32_t mask = dsa::Min(LANES, pitch - c);
                    const dsa::LocalTensor<float> s1 = perRow ? src : src[c];
                    if (multiply) dsa::Mul(zt[c], zt[c], s1, mask, static_cast<uint8_t>(n), rp);
                    else dsa::Add(zt[c], zt[c], s1, mask, static_cast<uint8_t>(n), rp);
                }
                return;
            }
            for (uint32_t i = 0; i < n; ++i) {
                if (perRow) MulByBlock(zt[i * pitch], src[8 * i], pitch);
                else if (multiply) dsa::Mul(zt[i * pitch], zt[i * pitch], src, pitch);
                else dsa::Add(zt[i * pitch], zt[i * pitch], src, pitch);
            }
        }

        // n elements *= the value that fills one 32-byte block (AdaptiveTiler::BlockMulCycles): 64-lane
        // repeats that all read that block (src1 block and repeat strides 0), at most 255 per
        // instruction, then the tail
        void MulByBlock(dsa::LocalTensor<float> zt, dsa::LocalTensor<float> blk, uint32_t n) {
            const dsa::BinaryRepeatParams rp{1, 1, 0, 8, 8, 0};
            const uint32_t full = n / LANES;
            for (uint32_t r = 0; r < full; r += 255) {
                dsa::Mul(zt[r * LANES], zt[r * LANES], blk, LANES, static_cast<uint8_t>(dsa::Min(255u, full - r)), rp);
            }
            if (n % LANES) dsa::Mul(zt[full * LANES], zt[full * LANES], blk, n % LANES, 1, rp);
        }

        // * gamma + beta for n normalized rows at the pitch, from the resident parameter rows: beta is
        // added after the normalization and gamma, never before the sum of squares
        void Affine(dsa::LocalTensor<float> zt, uint32_t n) {
            if (hasGamma) RowOp(zt, n, par, false, true);
            if (hasBeta) RowOp(zt, n, par[BetaAt()], false, false);
        }

        // gamma/beta for columns [c0, c0 + w) as FP32 rows in `par` (gamma, then beta at BetaAt()), their
        // pads zeros. StageParams issues the DMAs (16-bit: into a staging buffer, the Z tile before its
        // first use or else the scratch chunk; FP32: straight into par) and WidenParams widens them, so
        // the caller can issue a tile's DMA in between.
        bool StageParams(uint32_t c0, uint32_t w) {
            if constexpr (kF32) {
                if (hasGamma) DmaIn(par, gammaGm, c0, w, GAMMA_ALIGNED);
                if (hasBeta) DmaIn(par[BetaAt()], betaGm, c0, w, BETA_ALIGNED);
                return true;
            } else {
                const uint32_t off = StageOffset(w);
                if (2ull * off * sizeof(T) > StagingBytes()) return false;  // WidenParams streams them
                const dsa::LocalTensor<T> st = Staging();
                if (hasGamma) DmaIn(st, gammaGm, c0, w, GAMMA_ALIGNED);
                if (hasBeta) DmaIn(st[off], betaGm, c0, w, BETA_ALIGNED);
                return true;
            }
        }

        // 16-bit gamma/beta land in the Z tile when it holds both (it is free until the first tile),
        // else in the scratch chunk
        uint32_t StagingBytes() const { return dsa::Max(plan.layout.z, AdaptiveTiler::TMP_BYTES); }
        dsa::LocalTensor<T> Staging() const {
            return plan.layout.z >= AdaptiveTiler::TMP_BYTES ? z.template ReinterpretCast<T>() : chunk.template ReinterpretCast<T>();
        }

        // Every lane of a padded row is widened, its zeros included (AdaptiveTiler::ParamCycles)
        void WidenParams(uint32_t c0, uint32_t w, bool staged) {
            if constexpr (!kF32) {
                const uint32_t off = StageOffset(w);
                if (staged) {
                    const dsa::LocalTensor<T> st = Staging();
                    if (hasGamma) dsa::Cast(par, st, dsa::RoundMode::CAST_NONE, off);
                    if (hasBeta) dsa::Cast(par[BetaAt()], st[off], dsa::RoundMode::CAST_NONE, off);
                    return;
                }
                // Too long to stage both at once: one chunk-sized piece at a time
                const dsa::LocalTensor<T> st = chunk.template ReinterpretCast<T>();
                const uint32_t cap = AdaptiveTiler::TMP_BYTES / sizeof(T);
                for (uint32_t k = 0; k < 2; ++k) {
                    if (!(k ? hasBeta : hasGamma)) continue;
                    for (uint32_t o = 0; o < w; o += cap) {
                        const uint32_t m = dsa::Min(cap, w - o);
                        DmaIn(st, k ? betaGm : gammaGm, c0 + o, m, k ? BETA_ALIGNED : GAMMA_ALIGNED);
                        dsa::Cast(par[k * BetaAt() + o], st, dsa::RoundMode::CAST_NONE, StageOffset(m));
                    }
                }
            } else {
                (void)c0, (void)w, (void)staged;
            }
        }

        uint32_t StageOffset(uint32_t w) const { return AdaptiveTiler::Align32(uint64_t(w) * sizeof(T)) / sizeof(T); }
        uint32_t BetaAt() const { return AdaptiveTiler::ParamBytes(pitch) / 2 / sizeof(float); }  // The beta row

        // ---- SPLIT_COLUMNS: a column band of every row, its FP32 Z resident across SyncAll ------
        void BandPhase1(const CoreRange& r) {
            if (r.rowZ <= r.rowA) return;  // No units: the zeroed workspace stands for this core's record
            const uint32_t c0 = AdaptiveTiler::BandBegin(r, quantum), w = AdaptiveTiler::BandEnd(r, quantum) - c0;
            DSA_ASSERT(w <= pitch, "[DaePipeline]: column band wider than the plan's pitch");
            const uint32_t B = plan.tileRows, R = AdaptiveTiler::RecordFloats(plan, M);
            auto load = [&](uint32_t row) { LoadBand(r, row, dsa::Min(B, r.rowZ - row), c0); };
            load(r.rowA);
            const bool staged = StageParams(c0, w);
            for (uint32_t i = 1; i < plan.layout.depth && r.rowA + i * B < r.rowZ; ++i) load(r.rowA + i * B);
            WidenParams(c0, w, staged);
            for (uint32_t row = r.rowA; row < r.rowZ; row += B) {
                const uint32_t k = dsa::Min(B, r.rowZ - row), t = qX.DeQue();
                const dsa::LocalTensor<float> zt = res[(row - r.rowA) * pitch];
                AddInputs(zt, qX.template View<T>(t, 0), qX.template View<T>(t, 1), k * pitch);  // Z = X1 + X2
                qX.Free(t);
                // One record of M partials in whole 32-byte blocks, for every core to gather, built in
                // the VECOUT record buffer, where the egress channel can read it: each row group's
                // partials are packed into it by a fold (tiles are multiples of 8 rows, so every fold
                // starts a block; lanes past M are not defined, and nothing reads them)
                for (uint32_t g = 0; g < k; g += G) {
                    const uint32_t n = dsa::Min(G, k - g);
                    BandPartials(r, row + g, n, c0, zt[g * pitch]);
                    dsa::BlockReduceSum(rec[row + g], slots, static_cast<uint8_t>(AdaptiveTiler::CeilDiv(n, 8)), LANES, 1, 1, 8);
                }
                if (row + plan.layout.depth * B < r.rowZ) load(row + plan.layout.depth * B);
            }
            RecordOut(uint64_t(b) * R, rec, R);  // Egress from VECOUT
        }

        // Band row row0 + i's 8 partial sums of squares of its own columns into slot i, for n rows from zg
        // (AdaptiveTiler::BandPartialCycles): the rows' own columns squared into a zeroed chunk (a row
        // the core has none of stays 0), then summed and folded as RowPartials does
        void BandPartials(const CoreRange& r, uint32_t row0, uint32_t n, uint32_t c0, dsa::LocalTensor<float> zg) {
            const uint32_t P = static_cast<uint32_t>(AdaptiveTiler::ChunkPitch(pitch)), per = CHUNK / P;
            for (uint32_t r0 = 0; r0 < n; r0 += dsa::Max(per, 1u)) {
                if (!AdaptiveTiler::ChunkRows(pitch)) {  // One band row at a time
                    const Cols c = BandCol(r, row0 + r0, c0);
                    const uint32_t P2 = static_cast<uint32_t>(LANES * AdaptiveTiler::NextPow2(AdaptiveTiler::CeilDiv(P, LANES)));
                    PieceSum(zg[r0 * pitch + c.off], c.len, P2, P2, c.off);
                    dsa::BlockReduceSum(slots[8 * r0], chunk, 1, LANES, 1, 1, 8);
                    continue;
                }
                const uint32_t m = dsa::Min(per, n - r0);
                dsa::Duplicate(chunk, 0.0f, m * P);
                for (uint32_t i = r0; i < r0 + m; ++i) {
                    const Cols c = BandCol(r, row0 + i, c0);
                    if (c.len) dsa::Mul(chunk[(i - r0) * P + c.off], zg[i * pitch + c.off], zg[i * pitch + c.off], c.len);
                }
                ChunkPartials(m, P, slots[8 * r0]);
            }
        }

        void BandPhase2(const CoreRange& r, uint32_t nb) {
            if (r.rowZ <= r.rowA) return;
            const uint32_t c0 = AdaptiveTiler::BandBegin(r, quantum), R = AdaptiveTiler::RecordFloats(plan, M);
            // Every core's record in one DMA, summed by a fixed tree of vector adds (the same on every
            // core, so all owners of a row compute the same sigma)
            dsa::DataCopy(misc, wsGm[0], nb * R);
            for (uint32_t n = nb; n > 1; n -= n / 2) dsa::Add(misc, misc, misc[(n - n / 2) * R], n / 2 * R);
            const uint32_t B = plan.tileRows;
            for (uint32_t row = r.rowA; row < r.rowZ; row += B) {
                const uint32_t k = dsa::Min(B, r.rowZ - row);
                const dsa::LocalTensor<float> zt = res[(row - r.rowA) * pitch];
                for (uint32_t g = 0; g < k; g += G) {  // Each row group: its inverse RMS in lanes, spread, scaled
                    const uint32_t n = dsa::Min(G, k - g);
                    LaneInvRms(misc[row + g], misc[row + g], inv, nr, n, invD, eps);
                    dsa::Brcb(slots, inv, static_cast<uint32_t>(AdaptiveTiler::CeilDiv(n, 8)));
                    RowOp(zt[g * pitch], n, slots, true, true);
                    Affine(zt[g * pitch], n);
                }
                StoreBand(r, row, k, c0, zt);
            }
        }

        // k band rows into one tile slot at the pitch: one whole-block DMA per row and input
        void LoadBand(const CoreRange& r, uint32_t row, uint32_t k, uint32_t c0) {
            const uint32_t t = qX.Alloc();
            const dsa::LocalTensor<T> a = qX.template View<T>(t, 0), bt = qX.template View<T>(t, 1);
            for (uint32_t i = 0; i < k; ++i) {
                const Cols c = BandCol(r, row + i, c0);
                if (!c.len) continue;
                const uint64_t e = uint64_t(row + i) * D + c0 + c.off;
                DmaIn(a[i * pitch + c.off], x1Gm, e, c.len, X1_ALIGNED);
                DmaIn(bt[i * pitch + c.off], x2Gm, e, c.len, X2_ALIGNED);
            }
            qX.EnQue(t);
        }

        // Band rows back to Y, one DMA per row, from the next egress buffer: the resident Z is
        // narrowed (16-bit) or copied (FP32) into it, so its stores never delay the next tile
        void StoreBand(const CoreRange& r, uint32_t row, uint32_t k, uint32_t c0, dsa::LocalTensor<float> zt) {
            const uint32_t o = TakeOut();
            const dsa::LocalTensor<T> out = Out(o);
            Narrow(out, zt, k * pitch);
            for (uint32_t i = 0; i < k; ++i) {
                const Cols c = BandCol(r, row + i, c0);
                if (c.len) DmaOut(uint64_t(row + i) * D + c0 + c.off, out[i * pitch + c.off], c.len);
            }
            ReleaseOut(o);
        }

        // A band row's own columns inside its tile row (len 0: the core has none of that row)
        Cols BandCol(const CoreRange& r, uint32_t row, uint32_t c0) const {
            uint32_t cb, ce;
            AdaptiveTiler::BandRow(r, row, quantum, cb, ce);
            return cb < ce ? Cols{cb - c0, ce - cb} : Cols{0, 0};
        }

        // ---- Column tiles: Split-D fragments and rows too long for a row tile --------------------
        // A segment's sum of squares accumulates in lane 0 of a slot, the only lane a ReduceSum
        // defines [6.2]: slots 0 and 1 for the fragments, slot 0 for a whole row once the records are
        // out and for a fragment's gathered total. Its inverse RMS is spread over the block at
        // COLUMN_BC_SLOT.
        void ColumnPhase1(const CoreRange& r) {
            const bool split = plan.mode == TilingMode::SPLIT_D;
            used = 0;
            for (uint32_t k = 0; k < r.nFrag; ++k) {
                const CoreRange::Fragment& f = r.frag[k];
                fragZ[k] = Claim(uint64_t(f.row) * D + f.cb, f.ce - f.cb);
                Sweep1(f.row, f.cb, f.ce, fragZ[k], slots[8 * k]);
            }
            if (split) {  // Two 32-byte records {sum, 0 x7} per core, built in VECOUT from the slots' lanes 0
                dsa::Duplicate(rec, 0.0f, 16);
                for (uint32_t k = 0; k < r.nFrag; ++k) dsa::Adds(rec[8 * k], slots[8 * k], 0.0f, 1);
                RecordOut(uint64_t(b) * 16, rec, 16);
            }
            for (uint32_t row = r.rowA; row < r.rowZ; ++row) {
                const uint32_t mark = used;
                const Segment zr = Claim(uint64_t(row) * D, D);
                Sweep1(row, 0, D, zr, slots);
                SegmentInvRms(slots);
                Sweep2(row, 0, D, zr, false);
                used = mark;
            }
            // The first gamma/beta chunks of fragment 0 stream in while this core waits at SyncAll
            primed = split && r.nFrag > 0 && fragZ[0].resident && (hasGamma || hasBeta);
            if (primed) {
                const uint64_t start = uint64_t(r.frag[0].row) * D + r.frag[0].cb;
                LoadParams(r.frag[0].cb, TileLength(start, start + (r.frag[0].ce - r.frag[0].cb)));
            }
        }

        void ColumnPhase2(const CoreRange& r, uint32_t nb) {
            for (uint32_t k = 0; k < r.nFrag; ++k) {
                const CoreRange::Fragment& f = r.frag[k];
                uint32_t first, last;
                AdaptiveTiler::RowOwners(plan, D, f.row, nb, first, last);
                // The row's partials are records [lo, hi]: `first` holds the row as its last
                // fragment, every later owner as its first, and the records in between are zero
                const uint32_t lo = 2 * first + dsa::Max(1u, AdaptiveTiler::Range(plan, M, D, first, nb).nFrag) - 1;
                const uint32_t count = 2 * last - lo + 1, n = 8 * count, padded = AdaptiveTiler::LanePad(n);
                // Zero-padded to whole repeats and reduced into slot 0, the same records in the same order
                // on every owner
                dsa::DataCopy(misc, wsGm[uint64_t(lo) * 8], n);
                if (padded != n) dsa::Duplicate(misc[n], 0.0f, padded - n);
                dsa::ReduceSum(slots, misc, work, padded);
                SegmentInvRms(slots);
                Sweep2(f.row, f.cb, f.ce, fragZ[k], k == 0 && primed);
            }
        }

        // A segment's resident Z: its first tile at the base, every later tile on the 32-byte grid, like
        // its system-memory address (AdaptiveTiler::ResidentNeed); not resident when it does not fit
        struct Segment {
            dsa::LocalTensor<float> z;
            bool resident = false;
            uint32_t lead = 0;  // Elements between the first tile and the second
        };
        Segment Claim(uint64_t start, uint32_t len) {
            const uint32_t lead = static_cast<uint32_t>((quantum - start % quantum) % quantum);
            if (!plan.zResident || used + len + lead > plan.zResident) return {};
            const Segment sg{res[used], true, lead};
            used = (used + len + lead + quantum - 1) / quantum * quantum;
            return sg;
        }
        // The resident Z of the segment's tile at element e
        static dsa::LocalTensor<float> At(const Segment& sg, uint64_t start, uint64_t e) {
            return sg.z[e == start ? 0u : static_cast<uint32_t>(e - start) + sg.lead];
        }

        // Sweep 1 over row segment [cb, ce): Z = X1 + X2 and lane 0 of `slot` = sum(Z^2), Z kept when the
        // segment is resident. X1 and X2 of tile j+1 are in flight while tile j is processed.
        void Sweep1(uint32_t row, uint32_t cb, uint32_t ce, const Segment& sg, dsa::LocalTensor<float> slot) {
            const uint64_t base = uint64_t(row) * D, start = base + cb;
            bool first = true;
            Tiles(start, base + ce, false, [&](uint64_t e, uint32_t n) { LoadTile(e, n); },
                  [&](uint64_t e, uint32_t n) {
                      const uint32_t t = qX.DeQue();
                      const dsa::LocalTensor<float> zt = sg.resident ? At(sg, start, e) : z;
                      AddInputs(zt, qX.template View<T>(t, 0), qX.template View<T>(t, 1), n);
                      qX.Free(t);
                      RunSum(zt, n, slot, first);
                      first = false;
                  });
        }

        // Lane 0 of `slot` (+)= the sum of squares of a run of n elements (AdaptiveTiler::RunSumCycles):
        // chunk-sized pieces, each padded with zeros to whole repeats, squared and reduced. The first
        // piece of a segment (`first`) reduces into the slot, every other one into the spare slot, whose
        // lane 0 it then adds.
        void RunSum(dsa::LocalTensor<float> zt, uint32_t n, dsa::LocalTensor<float> slot, bool first) {
            for (uint32_t o = 0; o < n; o += CHUNK) {
                const uint32_t len = dsa::Min(CHUNK, n - o), padded = AdaptiveTiler::LanePad(len);
                if (padded != len) dsa::Duplicate(chunk[padded - LANES], 0.0f, LANES);  // The last repeat, before its squares
                dsa::Mul(chunk, zt[o], zt[o], len);
                const bool into = first && o == 0;
                ReduceRun(into ? slot : spare, padded);
                if (!into) dsa::Add(slot, slot, spare, 1);
            }
        }

        // Lane 0 of dst = the sum of the chunk's first n squares (whole repeats), folded 8 -> 1 first when
        // cheaper (into the fold partition, never onto the chunk)
        void ReduceRun(dsa::LocalTensor<float> dst, uint32_t n) {
            if (DaeIsa::FoldFirst(1, n)) {
                dsa::BlockReduceSum(fold, chunk, static_cast<uint8_t>(n / LANES), LANES, 1, 1, 8);
                return dsa::ReduceSum(dst, fold, work, n / 8);
            }
            dsa::ReduceSum(dst, chunk, work, n);
        }

        // A segment's inverse RMS from lane 0 of its slot, spread over the block at COLUMN_BC_SLOT by
        // one Brcb (AdaptiveTiler::ColumnSim::SegmentInvRms)
        void SegmentInvRms(dsa::LocalTensor<float> slot) {
            LaneInvRms(slot, slot, inv, nr, 1, invD, eps);
            dsa::Brcb(Spread(), inv, 1);
        }
        dsa::LocalTensor<float> Spread() const { return slots[8 * AdaptiveTiler::COLUMN_BC_SLOT]; }

        // Sweep 2: Y = Z * invRms * gamma + beta. The row's Z is spent by then, so the gamma and beta
        // chunks of every tile stream in as a pair (qP's two ways), the next pair while a tile is
        // processed. From resident Z only the pairs stream (the first may already be in flight:
        // `primed`); otherwise X1/X2 re-stream alongside them to recompute Z.
        void Sweep2(uint32_t row, uint32_t cb, uint32_t ce, const Segment& sg, bool primed) {
            const uint64_t base = uint64_t(row) * D, start = base + cb, end = base + ce;
            const dsa::LocalTensor<float> bc = Spread();
            if (sg.resident) {
                Tiles(start, end, primed, [&](uint64_t e, uint32_t n) { LoadParams(static_cast<uint32_t>(e - base), n); },
                      [&](uint64_t e, uint32_t n) {
                          const dsa::LocalTensor<float> zt = At(sg, start, e);
                          MulByBlock(zt, bc, n);
                          UseParams(zt, n);
                          const uint32_t o = TakeOut();
                          Narrow(Out(o), zt, n);
                          DmaOut(e, Out(o), n);
                          ReleaseOut(o);
                      });
                return;
            }
            Tiles(start, end, false,
                  [&](uint64_t e, uint32_t n) {
                      LoadTile(e, n);
                      LoadParams(static_cast<uint32_t>(e - base), n);
                  },
                  [&](uint64_t e, uint32_t n) {
                      const uint32_t t = qX.DeQue();
                      AddInputs(z, qX.template View<T>(t, 0), qX.template View<T>(t, 1), n);
                      qX.Free(t);
                      MulByBlock(z, bc, n);
                      UseParams(z, n);
                      const uint32_t o = TakeOut();
                      Narrow(Out(o), z, n);
                      DmaOut(e, Out(o), n);
                      ReleaseOut(o);
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
            return static_cast<uint32_t>(dsa::Min<uint64_t>(end, e / quantum * quantum + plan.tileElems) - e);
        }

        void LoadTile(uint64_t e, uint32_t n) {
            const uint32_t t = qX.Alloc();
            DmaIn(qX.template View<T>(t, 0), x1Gm, e, n, X1_ALIGNED);
            DmaIn(qX.template View<T>(t, 1), x2Gm, e, n, X2_ALIGNED);
            qX.EnQue(t);
        }

        // The gamma and beta chunks of columns [c0, c0 + n) into the next parameter slot (way 0:
        // gamma, way 1: beta; an absent parameter leaves its way untouched)
        void LoadParams(uint32_t c0, uint32_t n) {
            if (!hasGamma && !hasBeta) return;
            const uint32_t t = qP.Alloc();
            if (hasGamma) DmaIn(qP.template View<T>(t, 0), gammaGm, c0, n, GAMMA_ALIGNED);
            if (hasBeta) DmaIn(qP.template View<T>(t, 1), betaGm, c0, n, BETA_ALIGNED);
            qP.EnQue(t);
        }

        // zt = zt * gamma + beta with the chunks at the head of qP
        void UseParams(dsa::LocalTensor<float> zt, uint32_t n) {
            if (!hasGamma && !hasBeta) return;
            const uint32_t t = qP.DeQue();
            if (hasGamma) Combine(zt, qP.template View<T>(t, 0), n, true);
            if (hasBeta) Combine(zt, qP.template View<T>(t, 1), n, false);
            qP.Free(t);
        }

        // FP32 chunks combine directly; 16-bit chunks are widened through the scratch chunk
        void Combine(dsa::LocalTensor<float> zt, dsa::LocalTensor<T> pc, uint32_t n, bool multiply) {
            if constexpr (kF32) {
                if (multiply) dsa::Mul(zt, zt, pc, n);
                else dsa::Add(zt, zt, pc, n);
            } else {
                for (uint32_t o = 0; o < n; o += CHUNK) {
                    const uint32_t k = dsa::Min(CHUNK, n - o);
                    dsa::Cast(chunk, pc[o], dsa::RoundMode::CAST_NONE, k);
                    if (multiply) dsa::Mul(zt[o], zt[o], chunk, k);
                    else dsa::Add(zt[o], zt[o], chunk, k);
                }
            }
        }

        // Z = X1 + X2 in FP32 (FP32 row tiles in place; 16-bit widened into zt, X2 a chunk at a time)
        void AddInputs(dsa::LocalTensor<float> zt, dsa::LocalTensor<T> a, dsa::LocalTensor<T> bt, uint32_t n) {
            if constexpr (kF32) {
                dsa::Add(zt, a, bt, n);
            } else {
                dsa::Cast(zt, a, dsa::RoundMode::CAST_NONE, n);
                for (uint32_t o = 0; o < n; o += CHUNK) {
                    const uint32_t k = dsa::Min(CHUNK, n - o);
                    dsa::Cast(chunk, bt[o], dsa::RoundMode::CAST_NONE, k);
                    dsa::Add(zt[o], zt[o], chunk, k);
                }
            }
        }

        // Z into an egress buffer: narrowed to nearest-even 16 bits, or copied (FP32)
        void Narrow(dsa::LocalTensor<T> out, dsa::LocalTensor<float> zt, uint32_t n) {
            if constexpr (kF32) dsa::Muls(out, zt, 1.0f, n);
            else dsa::Cast(out, zt, dsa::RoundMode::CAST_RINT, n);
        }

        // Column tiles: state carried across SyncAll
        uint32_t used = 0;
        bool primed = false;
        Segment fragZ[2];  // The fragments' resident Z (not resident: recomputed)
    };
};

} // namespace hpc
