# Target Hardware: Measured Constants, Semantic Corrections, and Benchmark Ground Truth

> Everything below was **measured on the target machine**, not on the Host CI testbed.
> Each row states the number, how it was obtained, and a confidence level.
> Where a value contradicts one currently hard-coded in `include/dsa_runtime.hpp`, both are shown.
>
> Rows marked **low confidence** should not be used for load-bearing decisions.
> Measure them, or design around them.

---

## Status on this branch

Already applied here, so do not redo them:

| Item | Where |
| :--- | :--- |
| The kernel spec (bias after normalize and gamma, never in the sum of squares) | `4b79e79` |
| Launch 1.70 us, the same for any core count | `KERNEL_LAUNCH_NS` |
| SyncAll 0.924 us | `SYNC_ALL_NS` |
| `bw(P, b) = min(regime, size ceiling) * min(1, P / 27)`, both tables verbatim | `DMA_SIZE_CEILING`, `DMA_REGIME_CEILING`, `DMA_SATURATION_CORES` |

Sections 1, 3, 4 and 5 below are therefore history: they record how those numbers were
obtained, not work that is outstanding. `DMA_BYTES_PER_CYCLE` and `byteNs` appear there as
the values that used to be in their place; both are gone now.

Applied since (the README's implementation section has the details):

| Item | Where |
| :--- | :--- |
| The API forms the target does not have: the DAE kernel compiles against the refusing header (global memory through `GlobalTensor` only, `Cast` with a rounding mode, the 7-argument reductions, `TBuf` only), and `ctest -R target_api_shape` keeps every refused form a compile error | `src/kernel_unified.hpp`, [`TARGET_API_SHAPE.md`](TARGET_API_SHAPE.md) |
| Section 6 in the runtime and the kernel: block `Brcb`, `ReduceSum` lane 0 only, no `VectorInvRms` (table `Rsqrt` + Newton-Raphson), no `LocalMemAllocator` / `pos`, fenced scalar reads, 32-byte segment starts, `invD` from the host, per-repeat masks | `include/dsa_runtime.hpp`, `tests/test_dsa_runtime.cpp` |
| Software pipelining across row chunks: 2-4 tiles in flight on static buffer rings, each slot's flags set and waited once per tile, no `PIPE_ALL` between tiles, for any number of tiles per core | `DaePipeline::Core::RowTiles` |
| The two shapes the direct path could not serve: C5 splits over all 40 cores (row-major Split-D), C15 runs one-row tiles pipelined | `AdaptiveTiler::Plan` |

On 6.2: this runtime fills lanes 1..7 of a `ReduceSum` slot with NaN, stricter than "zero the slots before use". The kernel
reads lane 0 of a slot and nothing else, so it is correct either way; its row sums do not go through `ReduceSum` at all.

Still open:

| Item | Where |
| :--- | :--- |
| The per-instruction issue constants (13 / 14 / 15) are **not measured**; the time base is the measured repeat (1.717 cycles per ns), not a 1.5 GHz clock | section 10 |
| The DMA latency (100 ns in the model) is **not measured** | `DMA_LATENCY_NS` |
| C4, C7, C14 and C15: the model is optimistic against the best known times (0.58-0.79x), so the memory system does not yet explain them | section 9 |

---

## 0. The kernel spec (fixed on this branch by `4b79e79`)

`include/hpc_vector_norm.hpp:19` and the README both state:

```
Z[i, j] = X1[i, j] + X2[i, j] + bias[j]          <-- bias folded into Z
Y[i, j] = Z[i, j] / sigma_i * gamma[j]           <-- no bias at the end
```

The target computes something different:

```
Z[i, j]     = X1[i, j] + X2[i, j]                       // bias is NOT part of Z
sigma_i     = sqrt( (1/D) * sum_j Z[i,j]^2 + eps )      // so bias never enters the sum of squares
Y[i, j]     = Z[i, j] / sigma_i * gamma[j] + bias[j]    // bias is applied AFTER normalize and gamma
```

**Consequence:** the library's own golden reference encodes the same convention as the
library, so `ctest` agrees with itself to 1e-6 and can never surface this. Measured against
the target's convention the deviation is **3.400** — i.e. a different function, not a
precision issue. Every optimization tuned against the current reference is optimizing the
wrong objective.

Fixing this requires touching five sites where `bias` is added (the resident-Z path in
`Core`/`Sweep2` is not a mechanical move: it needs a second parameter stream, because the
row's Z is consumed before bias would be applied).

---

## 1. Fixed overheads

| Quantity | `dsa_runtime.hpp` today | **Measured on target** | How measured | Confidence |
| :--- | :--- | :--- | :--- | :--- |
| Kernel launch / per-core init | `launchNs = 0.0` | **1.70 us, independent of core count** | The fastest known time for a 1x64 case (which does almost no work) is 1.70 us. Hardware counters show the vector unit *busy* for 1.34 us on that 1x64 case: that is per-core init, paid in parallel by all cores, so the total does not scale with P | Med-High |
| `SyncAll()` | `SYNC_ALL_CYCLES = 7500` -> **5.00 us** | **0.924 us** (5.4x lower) | Amplification experiment, 5 runs: 0.795 / 0.902 / 0.924 / 0.953 / 0.975 | **High** |
| `PipeBarrier<PIPE_ALL>()` | not priced | **~12 ns** | Amplification experiment | Med-High |
| `GetValue()` scalar read | not priced | **~1.3 ns** (upper bound) | 16x repetition added only +0.02 us total, so this is a ceiling, not a value | Med |

**Why the `SyncAll` number matters:** at 5.00 us a barrier looks worth eliminating. At
0.924 us it is not. One experiment was spent replacing a cross-core barrier with L2 atomics
to close a 5.6 us gap; the theoretical maximum win was 0.92 us, so the ceiling was below
the gap before any code was written. The measured result was 48% *worse*.

---

## 2. Vector throughput

One repeat = 256 bytes (64 elements at 4 bytes, 128 at 2 bytes).

| Element width | **Measured ns per element per pass** | How measured |
| :--- | :--- | :--- |
| 2 bytes | **0.0111** | Turn one `Mul(aux,y,y,n)` pass count into a macro, sweep N = 0/4/8/16, take the least-squares slope of dT/dN. Output is bit-identical (the last pass overwrites the earlier ones) and the transfer term is independent of N, so differencing removes it entirely |
| 4 bytes | **0.0182** | Same |

The ratio 1.64 matches "4-byte elements cover 64 lanes per repeat instead of 128" — an
independent self-consistency check, which is why these two numbers are trusted.

For this kernel: ~9 passes at 2 bytes (2 widen / add / square / reduce / *inv / *gamma /
+bias / narrow), ~6 at 4 bytes (no conversions) -> **~0.0999 ns/element (2B)**,
**~0.1092 (4B)**.

⚠️ Do not use 0.1800. That value was back-solved by differencing core counts, and its
premise ("transfer cost is independent of core count") is refuted by section 4.

---

## 3. DMA rate ceiling by transfer size (absolute GB/s, not an efficiency factor)

| Bytes in one transfer | **Absolute achievable ceiling** |
| :--- | :--- |
| 1024 B | **277 GB/s** |
| 2048 B | **475 GB/s** |
| 4096 B | **677 GB/s** |
| >= 6144 B | **1090 GB/s** |

Combine with `min`, not multiplication: `effective = min(regime_ceiling, size_ceiling)`.
This is why some shapes stop changing above 4 KB — their regime ceiling is already the
lower of the two. A multiplicative efficiency model cannot reproduce that behaviour.

⚠️ An earlier curve keyed on **row width** and expressed as a **relative efficiency**
(1154 B -> 0.97) was wrong in both respects. It priced a 20-slice redesign at ~0 cost; the
real cost was 0.95 us against a 0.72 us saving, so that design was negative before it was
written.

---

## 4. Aggregate bandwidth vs core count: saturates near P = 27

Same code, core count varied, back-solving achieved bandwidth relative to P = 40:

| P | 8 | 16 | 24 | 40 |
| :--- | :--- | :--- | :--- | :--- |
| Relative bandwidth | **0.46** | **0.84** | **1.02** | 1.00 |

**Cores 28..40 contribute nothing to bandwidth.** At P = 24 the full bandwidth is already
available, and the remaining P = 24 -> 40 time difference is fully explained by the vector
term (self-consistent).

Implication: below ~28 cores the limit is **per-core DMA issue capability**, not the bus.
At P = 40 the machine is already at saturation, so "get more bandwidth" means
**raise per-core DMA efficiency** (larger transfers, fewer instructions), not "approach the
bus ceiling".

This term is absent from `HardwareModel` (`byteNs` is a constant). A model without it
predicted the *direction* of core-count changes correctly on only **25%** of a held-out set
of 20 measured points — worse than a coin flip. Adding
`bw(P) = bw_max * min(1, P/27)` raised the same held-out set to **95%**.

---

## 5. `DMA_BYTES_PER_CYCLE` (850 GB/s) is wrong in kind, not just in value

Achieved aggregate bandwidth, back-solved from measured times
(`traffic / (T - launch - vector_time)`):

| Working set | **Best achieved** |
| :--- | :--- |
| 30 MB | 1945 GB/s |
| 48 MB | 2020 GB/s |
| 80 MB | 2046 GB/s |
| 132 MB | 1408 GB/s |
| 368 MB | 1252 GB/s |
| 2.5 GB | 981 GB/s |
| 5.5 GB | 943 GB/s |

1945-2046 GB/s is far above any plausible main-memory figure: those shapes run largely out
of last-level cache. A single flat `byteNs` cannot express this, and neither can
"bandwidth as a step function of working set" — a lower-bound model built that way was
violated by 43 of 300 measured points, all on the cache-resident shapes.

---

## 6. Semantics where `dsa_runtime.hpp` diverges from the target

These are the ones that cost real submissions. Each is a case where code that passes
`ctest` fails on the target.

### 6.1 `Brcb` — wrong broadcast semantics (produces wrong results)

`dsa_runtime.hpp:1383` reads `src.data[0]` and fills the entire destination with that one
value. The target instead **reads 8 consecutive values from `src` and fills the j-th
32-byte block with the j-th value**.

So `Brcb(bc, inv[i], padded/64, {1,8})` — intended as "broadcast row i's invRms across its
repeats" — on the target splices the invRms of rows i..i+7 into a single row. Accuracy
collapses.

A correct pure-vector broadcast needs either a strided binary op with
`src1BlkStride = src1RepStride = 0` (so every lane reads the same 32-byte block), or one
`Brcb` per 8 rows with the row multiplier already sitting at `src[j]`.

### 6.2 `ReduceSum` returning a value — a CPU convenience that does not exist

On the target the signature is `void ReduceSum(dst, src, work, count)`:

- It returns **void**. Obtaining the scalar requires `GetValue`, and every `GetValue`
  needs a vector->scalar visibility fence first. Inside a per-row loop that is **one
  pipeline fence per row**.
- It writes **only lane 0** of `dst`, but occupies a whole 32-byte block. Multi-row
  reductions must offset by `dst[i * 8]` (4-byte elements). Using `dst[i]` makes adjacent
  rows overwrite each other.
- Lanes 1..7 of the destination are **left untouched**, so they hold whatever was there
  before. Consuming the block wholesale folds stale values into the sum. Measured
  consequence: accuracy **0.2526 ≈ 1/4** (4 rows per group, only row 0 correct).
  **Zero the reduction slots before use.**

There are 5 call sites in `Core` that use the value form, 2 of them inside per-row loops.
`DirectCore` uses the correct 4-argument form and is unaffected.

### 6.3 `VectorInvRms` does not exist on the target

There is no such instruction. The scalar path is `Rsqrt` on a tensor (a low-precision
table lookup) followed by Newton-Raphson refinement, then `GetValue`. `RefineInvRms` in
this repo already has the right shape for the refinement.

`DirectCore`'s `LaneInvRms` is the better design: it stays entirely in the vector unit and
never touches the scalar unit.

### 6.4 `LocalMemAllocator<Hardware::Scratchpad>` and `LocalTensor::pos` do not exist

The target allocates scratchpad as `TPipe::InitBuffer(TBuf&, bytes)` then `TBuf::Get<T>()`
for the whole block, which the kernel slices itself. `LocalTensor` has no public `pos`
member, so `t.pos = pos` does not compile.

### 6.5 `PipeBarrier<PIPE_V>` does not establish vector->scalar visibility

Only `PipeBarrier<PIPE_ALL>()` or `CrossPipe<HardEvent::V_S>()` do. Downgrading a
`CrossPipe<V_S>` to `PipeBarrier<PIPE_V>` produced a maximum relative error of
**5.0e7** with 100% of elements wrong.

Note that on the target a barrier costs <= 0.19 ns, so **removing barriers is not an
optimization** there (on the Host CI machine the same barrier is ~86 ns, which is
misleading).

### 6.6 Scratchpad segment alignment

Every segment used as a DMA source or destination must start on a **32-byte boundary**.
Row widths are not always even (197 / 397 / 577 / 3079 / 3083 appear in the benchmark), so
slicing sequentially by `k * D * sizeof(T)` necessarily produces misaligned starts. Round
each segment start up to 32 bytes.

### 6.7 Integer-to-float conversion is rejected inside device functions

`invD = 1.0f / static_cast<float>(d)` with `uint32_t d` fails to compile on the target
("cast between floating and unsigned integer variable is not allowed"). Pass `1/D` in from
the host instead.

### 6.8 The strided binary form takes a per-repeat mask, not a total count

`Mul(dst, a, b, mask, repeatTimes, BinaryRepeatParams)` — `mask` is the element count *per
repeat* (<= 64 for 4-byte elements), and `repeatTimes` is the number of repeats. Passing a
whole row width `D > 64` as `mask` silently processes only part of the data.

---

## 7. Measurement noise is per-shape, and large for the small shapes

Same binary submitted 20 times (two independent groups, n = 11 and n = 9, in agreement):

| Case | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | C11 | C12 | C13 | C14 | C15 |
| :--- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 1 sigma | 3.0% | 2.4% | 1.3% | 1.6% | 2.3% | 2.1% | 1.8% | 1.7% | 1.0% | 1.4% | 1.1% | 1.0% | 0.4% | 0.2% | 0.1% |
| **3 sigma** | **9.0%** | 7.2% | 3.9% | 4.8% | 6.9% | 6.3% | 5.4% | 5.1% | 3.0% | 4.2% | 3.3% | 3.0% | **1.2%** | **0.6%** | **0.3%** |

Use **3 sigma** as the decision threshold. Consequences:

- Changes below 6-9% on C1 / C2 / C5 / C6 are **not resolvable**. Do not run fine-grained
  A/B experiments there.
- C13 / C14 / C15 resolve below 1% and are the only suitable venue for fine measurements.
- There is also **common-mode day-to-day drift**: the median of an unchanged binary moved
  1.3%-3.3% between days. When comparing two runs, subtract the median change across all
  15 cases first.

---

## 8. Scoring

```
per_case = 100 / (1 + ln(T / t_best) / ln(1.5))        // 100 when T <= t_best
total    = unweighted mean of the 15 per-case scores
```

| Rule | Consequence |
| :--- | :--- |
| Only the **most recent** submission is ranked | Any failed experiment replaces the standing result; always follow one with a known-good submission |
| One case failing accuracy | **Entire submission scores 0** |
| One case timing out | **All subsequent cases are skipped** (scored 0) |
| Unweighted mean | A 1x64 case and a 117504x8192 case are worth the same. Share of runtime != share of score |
| Logarithmic curve | The closer to the record, the more a given speedup is worth. 10% faster is ~+2 points at r = 1.05, under +1 at r = 2.4 |

---

## 9. The 15 benchmark shapes and current standings (us)

| Case | M x D | dtype | Row bytes | 32B-aligned | Reference impl | Median of top 20 | Best known | Achieved BW / best |
| :--- | :--- | :--- | --: | :-- | --: | --: | --: | --: |
| C1 | 1 x 64 | 2B | 128 | yes | 2.13 | 2.16 | 1.70 | - |
| C2 | 7 x 197 | 4B | 788 | **no** | 2.74 | 2.50 | 2.21 | - |
| C3 | 128 x 256 | 4B | 1024 | yes | 3.11 | 2.67 | 2.47 | - |
| C4 | 766 x 193 | 2B | 386 | **no** | 8.83 | 7.79 | 6.66 | - |
| C5 | 8 x 32768 | 2B | 65536 | yes | 10.87 | 5.75 | 5.20 | - |
| C6 | 1508 x 577 | 2B | 1154 | **no** | 13.42 | 12.14 | 9.62 | 0.72x |
| C7 | 3104 x 397 | 2B | 794 | **no** | 17.85 | 16.88 | 13.90 | 0.73x |
| C8 | 10240 x 512 | 2B | 1024 | yes | 34.46 | 33.06 | 30.16 | 0.82x |
| C9 | 4080 x 1536 | 4B | 6144 | yes | 71.56 | 70.40 | 68.20 | 0.94x |
| C10 | 8192 x 1024 | 2B | 2048 | yes | 52.62 | 48.99 | 46.63 | 0.83x |
| C11 | 3752 x 3083 | 4B | 12332 | **no** | 161.39 | 135.12 | 131.16 | 0.77x |
| C12 | 3392 x 4096 | 2B | 8192 | yes | 80.70 | 79.60 | 76.35 | 0.92x |
| C13 | 10432 x 3079 | 4B | 12316 | **no** | 565.99 | 420.86 | 392.72 | **0.65x** |
| C14 | 3440640 x 128 | 2B | 256 | yes | 3943.65 | 3835.96 | 3750.12 | 0.95x |
| C15 | 117504 x 8192 | 2B | 16384 | yes | 8903.53 | 9013.46 | 8321.94 | 0.94x |

Reference implementation total: **68.51**. Best known total: 86.01.

**The last column is the most useful model-free grade**: achieved aggregate bandwidth
divided by the best achieved on the same shape. Worst: C13 0.65, C6 0.72, C7 0.73,
C11 0.77. Already close: C9 0.94, C12 0.92, C14 0.95, C15 0.94.

---

## 10. Per-instruction issue overhead: confirmed in kind, constants unverified

The cost model in `adaptive_tiler.hpp` charges a fixed per-instruction term:

```
Op(n)     = 2 * Repeats(n) + 13
Fold(n)   =     Repeats(n) + 14
Reduce(n) = 2 * Repeats(n) + 15
```

**The amortization rule was measured, and it is the important part**: the fixed term can
only be amortized when the row width is a multiple of 32 bytes *and* rows are actually
batched into one instruction. When the row width is not a multiple of 32, every row needs
its own instruction and pays the fixed cost per row per pass.

Grouping the 15 fixed shapes by row alignment:

| Group | Predicted fixed-term inflation | Measured distance of the best implementation from the lower bound | Correlation |
| :--- | :--- | :--- | :--- |
| Row bytes % 32 == 0 (6 cases) | 4.25x .. 1.05x | **0.98 .. 1.14, uncorrelated with inflation** | 0.286 |
| Row bytes % 32 != 0 (6 cases) | 2.62x .. 1.13x | 0.92 .. 2.41, **tracks inflation** | **0.713** |

This explains why *every* implementation, including the best known ones, sits 1.6-2.4x
above the bandwidth-only lower bound on the three short-row shapes (386 / 794 / 1154 B).

⚠️ The constants 13 / 14 / 15 and `CLOCK_GHZ = 1.5` have **never been measured**. They are
plausible in magnitude but unverified. Before using them for a load-bearing decision,
measure them with a ruler that holds the total element count fixed and sweeps the
instruction count.

---

## 11. Measured result of `DaePipeline::DirectCore` on the target

`DirectCore` was ported and run on the target (with the spec correction from section 0
applied, `Brcb` replaced per 6.1, `LocalMemAllocator` replaced per 6.4, and an outer
row-chunk loop added because one core's rows do not fit the scratchpad for most shapes).

| | Result |
| :--- | :--- |
| **Accuracy** | **15/15, ratio 1.0000 on every case** — the algorithm and the corrected spec are right |
| Total score | 63.33 (reference implementation: 68.51) |
| **Genuinely faster** | **C4 -5.0%** (resolvable; 3 sigma = 4.8%). Short, non-32B-aligned rows — exactly where section 10 predicts the fixed-issue term dominates |
| **Genuinely slower** | **C8 +93.9%, C10 +57.8%** (both resolvable) |
| Within noise | C2 -4.0%, C3 -2.6%, C6 -0.4%, C7 -0.2% |

**Why it is slower.** Each row chunk runs `load -> compute -> store` with a
`PipeBarrier<PIPE_ALL>` between chunks, so **nothing overlaps across chunks**. That turns
`T = max(transfer, compute)` into `T = transfer + compute`. C8 and C10 are precisely the
shapes where the two terms are comparable, so they lose the most.

**What the next version needs: software pipelining across row chunks** — while chunk i is
computing, chunk i+1's DMA should already be in flight. That means multiple buffers and
per-chunk events (`SetFlag`/`WaitFlag`), not a single `PipeBarrier<PIPE_ALL>`.

Also observed: with only 3 rows per chunk (32 chunks per core) the run was **killed by the
stream watchdog**, not merely slowed. Too many chunks is a correctness risk, not just a
performance one.

Two shapes `DirectCore` cannot serve at all: **C5 (8 x 32768)** and **C15 (117504 x 8192)**.
For C15 the resident parameters alone (gamma and bias, each in native and FP32 form at full
width) consume 98 KB, leaving too little for a single row. Both need column-direction
splitting.

---

## 12. Directions already falsified (do not re-run these)

| Direction | Why it cannot work, from the numbers above |
| :--- | :--- |
| Replace the cross-core barrier with L2 atomics | The barrier costs 0.924 us; maximum possible win 0.92 us against a 5.6 us gap. **Ceiling below the gap** |
| Drop to 8 cores for more scratchpad / no barrier | 8 cores reach only 46% of the bandwidth (section 4). One shape needs 5.9 us of pure transfer at that rate, against a 4.44 us target for the whole case |
| More column slices to cut parameter traffic | Saves 0.72 us; costs 0.95 us because the transfer size drops from 8192 B to 3264 B (section 3). **Negative before it is written** |
| Remove barriers | <= 0.19 ns each on the target |
| Fine-grained tuning on C1 / C2 / C6 | 3 sigma is 6-9% there (section 7) |
| Reduce P from 40 to 32 to ease bus contention | Bandwidth saturates at P = 24; fewer cores only adds per-core vector work |
