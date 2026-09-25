# 🏛️ Architecture Challenges & Open Research Vectors

Welcome to the `hpc_vector_norm` performance optimization project!

To achieve theoretical roofline performance without hardcoding specific case branches, we have left **5 major architectural open vectors** for contributors and autonomous AI agents. You are invited to design, mathematically formulate, and implement these solutions.

---

## 🎯 Challenge 1: Asymmetric 40-Thread Split-D Solver (Generalized Beyond Powers of 2)

### The Dilemma
When matrix row count $M < 32$ (e.g., $M = 8$), a naive power-of-two division yields:
$$\text{slices} = \max 2^k \le \lfloor 40 / 8 \rfloor = 4 \implies 8 \times 4 = 32 \text{ threads}$$
This leaves **8 cores idle (20% compute wasted)**.

### The Objective
Design an asymmetric or non-power-of-two partitioning algorithm in `AdaptiveTiler`:
1. Distribute columns of each row across **5 threads** ($8 \times 5 = 40$ threads full saturation!).
2. Since $32768 / 5 = 6553.6$, determine the integer boundary points $\{x_0, x_1, x_2, x_3, x_4, x_5\}$ such that:
   - Each partition slice $(x_{s+1} - x_s)$ is strictly aligned to **32 bytes** ($\lambda = 32\text{B}$).
   - Thread workloads are strictly load-balanced (difference $\le 32\text{ bytes}$).
3. Implement the corresponding multi-slice reduction tree across the 5 threads.

### ✅ CPU Implementation
- **Flattened line decomposition.** Each row is cut into $Q = \lceil D / q \rceil$ units of one 64-byte cache line ($q = 64 / \text{elemBytes}$). The $U = M \cdot Q$ units are split across $T$ threads as $[\lfloor U t / T \rfloor, \lfloor U (t+1) / T \rfloor)$, so any two threads differ by at most one line, for any $M$ and $T$. For $8 \times 32768$ FP16 on 40 threads, $U = 8192$ and every thread gets 204 or 205 lines (all 40 threads busy). The CPU quantum is 64 bytes rather than 32: neighbouring threads never share a Y cache line, and AVX-512 accesses stay line-aligned.
- **Reduction.** Each thread publishes the partial $\sum Z^2$ of its (at most two) shared row fragments into a cache-line-padded slot, then passes one barrier. All owners of row $r$ are the threads $\text{owner}(rQ) \dots \text{owner}(rQ + Q - 1)$ with $\text{owner}(u) = \lfloor ((u+1)T - 1) / U \rfloor$. Every owner sums the partials in the same order, so all of them compute the same $\sigma_r$. Fragments are reduced before the thread's complete rows run, so the barrier wait overlaps useful work. Every team member reaches the barrier, including threads with an empty range. The original code skipped the barrier on empty slices, which can deadlock.
- **When to split.** Whole rows leave the slowest thread $(\lceil M/T \rceil - M/T)$ rows of extra work; split iff that work costs more than a barrier (`AdaptiveTiler::Plan`).
- **Measured (4 cores):** $1 \times 2^{20}$ FP16 784 → 197 µs, $2 \times 2^{20}$ 738 → 372 µs, $1 \times 65536$ 25.7 → 10.3 µs, $6 \times 65536$ 51.6 → 42.9 µs.

---

## 🎯 Challenge 2: Closed-Form Scratchpad Knapsack Formulation for Batch Size

### The Dilemma
Currently, `batchRows` in `AdaptiveTiler` uses conservative estimates.
For Profile 8 ($10240 \times 512$), $D=512$:
- With `batch = 16`, local scratchpad footprint is $\sim 163\text{ KB} \le 191\text{ KB}$.
- To saturate bus bandwidth, we want `batch = 24` or `batch = 32`.
- A naive full auxiliary buffer requires $20\text{ bytes/element}$, which overflows at $B=24$ ($245\text{ KB} > 191\text{ KB}$).

### The Objective
1. Formulate a closed-form analytical equation that maximizes $B$:
   $$B^*(D, \omega) = \max \left\{ B \in \mathbb{N} \;\middle|\; \text{MemoryModel}(B, D, \omega) \le 195{,}584 \right\}$$
2. Integrate the **64-element streaming reduction** so that state footprint per element drops from $20\text{B} \to 12\text{B}$, proving feasibility for $B=24 \sim 32$ for all medium row sizes.

### ✅ CPU Implementation
- On a CPU, X1/X2/Y stream through the cache hierarchy, and the only per-element state is the FP32 Z row (4 B/element) in a per-thread 191 KB scratchpad: $\text{MemoryModel}(B, D) = 4 B D \le 195{,}584$.
- The batch is sized for latency, not capacity: $B^*(D) = \text{clamp}(\lfloor 2048 / D \rfloor, 1, 64)$ rows issue pass 1 before their pass 2, so the $B$ reduce → sqrt → reciprocal chains overlap. The batch's Z (8 KB) stays in L1. Row segments longer than the scratchpad recompute Z from the cache-hot inputs in pass 2 instead of spilling.
- The 64-element sliding window is 4 independent 16-lane FMA accumulators kept in registers. They accumulate in FP32 within 4096-element blocks and in FP64 across blocks, so there is no FMA latency chain and long rows don't drift.
- **Measured:** batching lowers the cost of short rows by 20–30% (e.g. $D = 64$ FP16: 148 → 105 cycles/row).

---

## 🎯 Challenge 3: Zero-Bubble True Double-Buffering Pipeline (Ping-Pong Prefetch)

### The Dilemma
In `KernelUnifiedPipeline::ExecuteRowParallel`, execution currently proceeds synchronously:
$$\text{Load}(r) \to \text{Compute}(r) \to \text{Store}(r)$$
The memory bus and SIMD compute units are never overlapped, leaving a 30%~50% pipeline bubble.

### The Objective
In `kernel_unified.hpp`, implement a classic **Ping-Pong double-buffered asynchronous loop**:
1. Allocate two disjoint scratchpad partitions: `Buffer[0]` and `Buffer[1]`.
2. **Prologue**: Prefetch chunk 0 into `Buffer[0]`.
3. **Steady State Loop** ($i = 0 \dots K-2$):
   - Asynchronously prefetch chunk $i+1$ into `Buffer[(i+1)%2]`.
   - Concurrently execute SIMD compute on chunk $i$ in `Buffer[i%2]`.
   - Write back previous chunk results.
4. **Epilogue**: Drain the final chunk.

### ✅ CPU Implementation
- On a CPU the hardware prefetchers already overlap sequential loads with compute, and an explicit ping-pong copy would only add traffic. The measured bubble was the dependency chain between the passes: pass 1 → horizontal reduction → sqrt → reciprocal → pass 2, about 80–100 cycles per row. Row batching (Challenge 2) removes it.
- Plans that stream from DRAM (working set > LLC) also issue a software prefetch 512 B ahead in pass 1, keeping more line fills in flight than one core would on its own. Measured: P14 −15–30%, P10 and P11 about −10%, neutral on the rest.
- For 16-bit tensors, bias/gamma are widened to FP32 once per call when a thread owns ≥ 4 rows (two conversions fewer per vector; BF16 about −10%). BF16 stores round to nearest-even with plain integer ops (shift, and, add, add, shift, pack) and no NaN fix-up. Every NaN the kernel can produce has zero low bits, so rounding cannot carry out of the mantissa; `tests/` checks NaN propagation.

---

## 🎯 Challenge 4: Cache-Conscious Space-Filling Traversal (Morton / Serpentine Order)

### The Dilemma
For large matrices evaluated over repeated iterations (e.g., Profile 12: $4096 \times 4096$), sequential row access suffers from cold cache eviction when returning to row 0.
The measured bandwidth of $1.06\text{ TB/s}$ is bounded by external memory, while the theoretical record achieves **$2.07\text{ TB/s}$** via 32MB L2/L3 cache residency.

### The Objective
Implement a cache-oblivious or cache-conscious index permutation:
$$\pi: [0, M-1] \to [0, M-1]$$
Using serpentine (snake-like bi-directional) traversal or Morton (Z-order) tiling, maximizing cache hit rate between successive evaluation epochs.

### ✅ CPU Implementation
- **Serpentine over 64 KB chunks.** Each thread splits its rows into chunks of ~64 KB of traffic. The chunk order flips on every call (a global epoch counter), and rows inside a chunk stay ascending. Reversing individual rows measured 10–15% slower, because every row boundary restarted the hardware prefetcher's stream. The chunked version is neutral to positive: P07 improves 15–20% by median, and P10, P14 and tensors far beyond the LLC are flat within noise.
- **Non-temporal stores.** Once $X_1 + X_2 + Y$ exceeds the LLC, Y is written with streaming stores (peeled to vector alignment). This drops the read-for-ownership stream and leaves the LLC to the inputs. A raw read-2/write-1 test gains 27% (31.6 → 40.3 GB/s); the kernel gains 2–6% because 4 cores are also partly compute-bound.

---

## 🎯 Challenge 5: Extreme Vectorization & Register Reuse (Zero-Spill Core)

### The Objective
Refactor the reduction tree and normalization pass to ensure:
1. Zero stack spills verified by `-Wframe-larger-than` or compiler assembly audit.
2. 100% vector instruction saturation (FMA instructions without branch jumps).

### ✅ CPU Implementation
- Explicit intrinsics in `src/simd.hpp`: AVX-512 (16 lanes, masked tails, no scalar remainder loops), AVX2 + FMA + F16C (8 lanes), and a portable scalar path. On this host AVX-512 is 1.3–1.6× faster per element than AVX2.
- The inner loops have no branches besides the loop back-edge. A null bias/gamma becomes a stride-0 constant vector (address mask 0) instead of a per-element test. An `objdump` audit of the pass-1 and pass-2 loop bodies shows only loads, conversions, add/mul/FMA and stores, with no stack traffic.

---

## 📏 Cost Model & Methodology

The planner's constants were measured on the reference host, a 4-core Cascade Lake KVM guest (AVX-512, 1 MB L2/core, 33 MB L3, ~40 GB/s DRAM):

| Constant | Value | Measurement |
| :--- | ---: | :--- |
| `FORK_JOIN_NS` | 3500 | 4-thread floor of the kernel on tiny inputs (an empty `omp parallel` costs 2.3–2.9 µs; libgomp issues a futex wake per barrier) |
| `BARRIER_NS` | 1000 | Extra cost of one team barrier (0.5–1 µs) |
| `CORE_NS_PER_ELEM` | 0.3 | One core, cache-resident rows: FP32 0.27–0.31, FP16 0.26–0.39 ns/element |
| `BATCH_ELEMS` | 2048 | Batch sweep $B \in \{1, 4, \dots, 64\}$ for $D \in \{64, 192, 256, 400\}$ |

The measured T=1/T=4 crossover is 12–16K elements, which matches $\tau \cdot P / ((P-1) c) \approx 15.6\text{K}$. Resizing an OpenMP team between calls cost 70–330 µs per call. For that reason the team is always either the caller alone or all `P` threads. Hosts with a different fork/join cost or core speed can re-tune these constants; the formulas stay the same. `tests/test_correctness.cpp` forces every plan variant (Split-D, streaming, recomputed Z) with 1–40 threads against an FP64 reference, and also checks for writes outside Y and for bitwise-identical results across serpentine directions.
