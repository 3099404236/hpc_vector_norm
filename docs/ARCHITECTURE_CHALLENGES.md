# 🏛️ Architecture Challenges & Open Research Vectors

Welcome to the `hpc_vector_norm` performance optimization project!

To achieve theoretical roofline performance without hardcoding specific case branches, we have left **5 major architectural open vectors** for contributors and autonomous AI agents. You are invited to design, mathematically formulate, and implement these solutions.

Each challenge below has two solutions:

- **40-core target**: a `HardwareModel::Target()` plan executed by `DaePipeline` on the DAE runtime (`include/dsa_runtime.hpp`).
- **CPU**: the host executor on the CI testbed.

Both come from the same planning equations in `src/adaptive_tiler.hpp`; only the machine constants differ (see [Cost Model & Methodology](#-cost-model--methodology)).

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

### ✅ 40-Core Target Implementation (DAE)
- **Flattened 32-byte decomposition.** The whole $M \times D$ tensor is cut into $U = \lceil MD / q \rceil$ units of one 32-byte DMA block ($q = 32 / s$ elements). Core $t$ of $P$ gets units $[\lfloor Ut/P \rfloor, \lfloor U(t+1)/P \rfloor)$ (`AdaptiveTiler::Range`), so any two cores differ by at most one block for every $M$, $D$ and $P$. The busiest core holds $\lceil U/P \rceil$ blocks, the minimum any split into 32-byte blocks can reach.
- **Two numberings of the same blocks.**
  - **Row-major (`SPLIT_D`).** Boundaries are offsets into the flattened tensor, so with a 32-byte-aligned base every boundary is a 32-byte-aligned address, even when $D \cdot s$ is not a multiple of 32. A core shares at most two rows with its neighbours, and reads γ/β once per element it owns.
  - **Column-block-major (`SPLIT_COLUMNS`).** Block $j$ of row $i$ is unit $jM + i$. The balance is identical, but a core's share is a column band of every row: row $i$ covers blocks $[j_0 + [i < a_0],\ j_1 + [i < a_1])$, where $u_0 = j_0 M + a_0$ and $u_1 = j_1 M + a_1$. The core therefore reads γ/β for its band once and reuses them across all $M$ rows. This numbering needs rows on the 32-byte grid, so every row DMA is whole blocks, and the band's FP32 Z must fit in the scratchpad.
- **P05 solved exactly.** $U = 16384$ blocks over $P = 40$ cores gives every core 409 or 410 blocks (6544 or 6560 elements, within 32 bytes of each other). An exactly equal split does not exist, since $16384 / 40 = 409.6$.
  - **Row-major.** Every row lands on exactly 5 cores, with boundaries $\{x_0, \dots, x_5\} = \{0, 6544, 13104, 19648, 26208, 32768\}$.
  - **Column band (what the planner picks).** Each core owns 51 or 52 blocks (816–832 columns) of all 8 rows. Per core, γ/β traffic drops from 26.2 KB to 3.3 KB and system traffic from 66 KB to 44 KB.
  - **Other shapes.** The same code balances any shape to one block, for example 3 × 100003 FP16 → 7488–7504 elements per core, 7 × 5000 → 864–880, 13 × 12288 → 3984–4000.
- **Reduction tree without scalar stalls.** Scalar reads of the scratchpad (`GetValue`) cost 500 cycles each, so partial sums only ever meet in vector registers:
  - **Column band.** Each core publishes one record of $M$ partials in 32-byte blocks, then passes one `SyncAll`. Every core then fetches all $P$ records with one DMA and sums them with a fixed tree of vector adds (40 → 20 → 10 → 5 → 3 → 2 → 1, six `Add`s). One `VectorReduceSum` per row then hands each row total to the scalar unit.
  - **Row-major.** Each core publishes two 32-byte records, $\{\Sigma_0, 0^{\times 7}\}$ and $\{\Sigma_1, 0^{\times 7}\}$. The owners of row $r$ are cores $\text{owner}(u_0) \dots \text{owner}(u_1)$ with $\text{owner}(u) = \lfloor ((u+1)P - 1) / U \rfloor$ (`RowOwners`). Owner `first` holds row $r$ as its last fragment and every later owner holds it as its first, so the row's partials are the contiguous record range $[2\,\text{first} + n_{\text{first}} - 1,\ 2\,\text{last}]$, with zero records in between. Each owner fetches that range with one DMA and reduces it with one `VectorReduceSum`.
  - **Determinism and safety.** Every owner sums the same records in the same order, so all of them compute an identical $\sigma$. Every core reaches the `SyncAll`, including one whose sanitizer trapped, so a fault cannot deadlock the barrier.
- **Resident Z.** Sweep 1 keeps FP32 Z in the scratchpad: the whole band for the column band (8 × 832 floats, 26 KB for P05), or up to $\min(\text{share}, 3D)$ elements for row-major fragments. Sweep 2 normalizes from there and reads only γ, so X1/X2 are never read from main memory a second time. In row-major plans, the first γ chunk of sweep 2 is issued before the `SyncAll` and arrives while the core waits.
- **When to split.** The planner compares the slowest core's modeled time. At 1.5 GHz, P05 costs 28.1 µs with whole rows (only 8 of the 40 cores busy), 14.2 µs with the row-major split and 12.3 µs with the column band. The band wins at every clock from 1.0 to 4.0 GHz. `SyncAll` costs 7500 cycles, which is 72% of the busiest core's 10,365. P05 is the only one of the 15 profiles that splits.
- **Verified** (`tests/test_dae_pipeline.cpp`):
  - 19 shapes, including the non-power-of-2 widths 5 × 7, 3 × 100, 7 × 200, 17 × 4097, 41 × 5000, 1 × 70001 and 3 × 100003;
  - each run in the model-chosen mode and forced into rows, row-major Split-D and the column band (where feasible);
  - with and without γ/β, on 32-byte-aligned and offset bases, in FP32/FP16/BF16;
  - through the sanitizer-enabled runtime, checked against an FP64 reference;
  - with zero scalar stalls, exactly one `SyncAll` for split plans, and the scratchpad claim equal to the plan.

### ✅ CPU Implementation
- **Same partition.** The host executor calls the same `Range`/`RowOwners` code, with $P$ equal to the host's thread count. It also uses 32-byte blocks. At a boundary, two threads write the same 64-byte line of Y once, and that is lost in the noise. Interleaved A/B runs against the previous 64-byte-line version (4 cores, best of 2 × 21 runs of 20 ms):

  | Shape (FP16) | 32-byte blocks | 64-byte lines |
  | :--- | ---: | ---: |
  | 1 × 2^20 | 184 µs | 188 µs |
  | 2 × 2^20 | 301 µs | 321 µs |
  | 1 × 65536 | 9.7 µs | 9.6 µs |
  | 6 × 65536 | 43.5 µs | 42.9 µs |

  The original code took 784, 738, 25.7 and 51.6 µs on these shapes.
- **Reduction.** Each thread publishes the partial $\sum Z^2$ of its (at most two) shared row fragments into a cache-line-padded slot, then passes one barrier. Every owner sums the partials in the same order, so all of them compute the same $\sigma_r$. Fragments are reduced before the thread's complete rows run, so the barrier wait overlaps useful work. Every team member reaches the barrier, including threads with an empty range. The original code skipped the barrier on empty slices, which can deadlock.
- **When to split.** The same busiest-core comparison as the target, with the host's measured constants.

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

### ✅ 40-Core Target Implementation (DAE)
- **An exact memory model.** `DaeLayout` lists the buffers the kernel claims through `TPipe`, and the runtime traps above 195,584 B. The tests require every executed plan to claim exactly its layout, so the planner's arithmetic is checked on every run. For a row tile of $B$ rows, each buffer rounded up to 32 bytes:
  $$\text{MemoryModel}(B, D, s) = \underbrace{2 \cdot 2 \cdot BDs}_{\text{X1, X2 double-buffered}} + \underbrace{4BD\,[s < 4]}_{\text{FP32 Z tile}} + \underbrace{2 \cdot 4\,\text{rep}\,D}_{\gamma,\ \beta\ \text{(FP32, rep rows)}} + \underbrace{8192}_{\text{scratch}}$$
- **20 B → 12 B per element.**
  - Y has no buffer of its own: the result is cast back into the X1 slot and sent from there.
  - $Z^2$ has no buffer either: the squares of a chunk of rows pass through the 8 KB scratch buffer. When the cost model says it is cheaper, one `BlockReduceSum` fold (8 → 1, 1 cycle per repeat) runs over the whole chunk first, and then one `VectorReduceSum` per row hands the row sum to the scalar unit without a stall. For $D = 512$ the fold leaves 64 partials per row, which is exactly the challenge's 64-element streaming accumulator. The kernel never uses the 14-cycle `WholeReduceSum`.
  - What remains is the X1/X2 queues (4 B per element) plus the FP32 Z tile (4 B): 12 B for FP16/BF16. FP32 computes Z in place in the X1 slot, at 16 B.
- **Closed form.** $B^*(D) = \lfloor (195{,}584 - 8{,}192 - 2 \cdot \text{Align}_{32}(4\,\text{rep}\,D)) / (bD) \rfloor$, with $b = 12$ for 16-bit dtypes and 16 for FP32, less the 32-byte rounding of each buffer:

  | $D$ (FP16) | 128 | 192 | 400 | 512 | 576 | 1024 |
  | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
  | $B^*$, γ/β not replicated | 121 | 80 | 38 | 29 | 26 | 14 |
  | $B^*$, γ/β replicated to 8 KB each | 111 | 74 | 35 | 27 | 25 | 13 |

- **Spending spare scratchpad on replicated γ/β.** With γ/β replicated into `rep` rows, bias and γ take one instruction per `rep` rows instead of one per row. The planner plans both variants and keeps the faster one.
  - **P08** runs 10 tiles of 26 rows ($n^* = 10$ sets the tile below the cap) with `rep = 4`, in 184,320 B. $B = 24 \dots 27$ rows all fit.
  - **P14** gives up 10 of its 121 possible rows to replication, which cuts 22.5% of its vector cycles.
- **Scalar stalls were the real batch-size problem.** The earlier kernel read every row sum back through `GetValue`, 500 cycles per row. At P08 that cost 128,000 stall cycles against 59,480 cycles of vector work. The busiest core is now at 52,341 cycles with no stalls, below its DMA time: 786 KB at 21.25 GB/s is 37 µs, against 52,341 cycles, which is 35 µs at 1.5 GHz.
- **Column tiles** (rows longer than a row tile, and row-major Split-D fragments) cost $(6s + 4)$ B per element, plus the resident FP32 Z. The **column band** (P05) keeps the band's FP32 Z for all rows resident across the barrier, next to $k$-row tiles, the replicated γ/β band and the partial-sum records.

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

### ✅ 40-Core Target Implementation (DAE)
- **Ping-pong through the runtime's queues.** X1 and X2 use `TQue<VECIN, 2>`.
  - **Row tiles.** The prologue issues tile 0's DMA, then γ/β, then tile 1's, and only then widens γ/β. So all three transfers are in flight before the first vector instruction, and parameter loading is interleaved with input streaming, as the challenge suggests. Replicating γ/β is scratchpad-to-scratchpad DMA, with no vector cycles. In the steady state, tile $k$'s output DMA frees its slots, the load of tile $k+2$ is issued into them, and tile $k+1$ is computed while it streams in.
  - **Column tiles.** X1, X2 and the β chunk of tile $k+1$ travel together before tile $k$ is processed. A resident sweep 2 streams γ chunks double-buffered, and its first chunk is issued before the `SyncAll`. Interior tile boundaries sit on the 32-byte grid.
- **Pipeline depth from the model.** With $n$ tiles through three stages that each cost $W/n + \text{tileNs}$ per tile, $T(n) = (n+2)(W/n + \text{tileNs})$ and $n^* = \sqrt{2W/\text{tileNs}}$, where $W = \max(\text{vector cycles}/\text{clock},\ \text{DMA bytes} \cdot \text{byteNs})$. The Challenge 2 knapsack caps it.
  - **P04.** Each core's 20 rows cost 2180 vector cycles, which is 1.45 µs at 1.5 GHz and above their 1.12 µs of DMA. That gives $n^* = 2$, so the core runs 2 × 10-row tiles and the second tile's load overlaps the first tile's compute.
  - **Other profiles.** $n^*$ sizes the tiles for P01–P08; from P09 on, the scratchpad bound sets the tile (full plan table in the README).
- **The startup bubble was scalar stalls.** P04's busiest core spent 10,000 of its 12,692 cycles in 20 `GetValue` stalls, one per row. Without them, and with bias and γ applied once per 10-row tile, it takes 2,180 cycles:
  - per row: one `VectorReduceSum`, one `VectorInvRms` and one `Muls`;
  - per tile: seven tile-wide instructions (widen X1, widen X2, add, bias, squares, γ, narrow).
- **A runtime defect had silently disabled double buffering.** `TQue::AllocTensor` returned slot `(tail + allocatedCount) % depth` and ignored slots that were still enqueued. After `EnQue(tile k)`, the prefetch of tile $k+1$ therefore got tile $k$'s buffer and overwrote it before `DeQue`. The queue now tracks each slot (FREE → ALLOCATED → ENQUEUED → DEQUEUED) in FIFO order and traps misuse (`tests/test_dsa_runtime.cpp`). With the original queue, 701 of the DAE suite's 1056 checks failed.

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

### 40-Core Target
Not modeled. The DAE simulation has no L2 shared between cores, so it cannot evaluate a traversal order; each core streams its tiles in ascending order. The host implementation below is where serpentine order was measured.

### ✅ CPU Implementation
- **Serpentine over 64 KB chunks.** Each thread splits its rows into chunks of ~64 KB of traffic. The chunk order flips on every call (a global epoch counter), and rows inside a chunk stay ascending. Reversing individual rows measured 10–15% slower, because every row boundary restarted the hardware prefetcher's stream. The chunked version is neutral to positive: P07 improves 15–20% by median, and P10, P14 and tensors far beyond the LLC are flat within noise.
- **Non-temporal stores.** Once $X_1 + X_2 + Y$ exceeds the LLC, Y is written with streaming stores (peeled to vector alignment). This drops the read-for-ownership stream and leaves the LLC to the inputs. A raw read-2/write-1 test gains 27% (31.6 → 40.3 GB/s); the kernel gains 2–6% because 4 cores are also partly compute-bound.

---

## 🎯 Challenge 5: Extreme Vectorization & Register Reuse (Zero-Spill Core)

### The Objective
Refactor the reduction tree and normalization pass to ensure:
1. Zero stack spills verified by `-Wframe-larger-than` or compiler assembly audit.
2. 100% vector instruction saturation (FMA instructions without branch jumps).

### ✅ 40-Core Target Implementation (DAE)
- **No V→S stalls.** The scalar unit never reads the scratchpad (`GetValue`, 500 cycles). Row sums reach it through `VectorReduceSum`, and Split-D partials are combined by vector adds. Every executed plan in the tests has zero stalls.
- **Few, long instructions.**
  - Every row costs three instructions: `VectorReduceSum`, `VectorInvRms` and `Muls`.
  - Everything else covers a whole tile, an 8 KB scratch chunk, or `rep` replicated rows (bias and γ).
  - One `BlockReduceSum` fold precedes the row reductions only when the runtime's costs make it cheaper. The kernel and the planner decide this with the same formula (`DaeIsa::FoldFirst`).
  - There are no per-element branches.
- **Cycles per element of the busiest core**, vector work only. The 7500-cycle barrier of P05 is not included:

  | Profile | P09 | P13 | P08 | P05 | P04 |
  | :--- | ---: | ---: | ---: | ---: | ---: |
  | Cycles per element | 0.24 | 0.33 | 0.40 | 0.44 | 0.57 |

  Short rows pay more per element for their three per-row instructions. The floor for 16-bit data is eight tile-wide instructions at 2 cycles per 64 elements, 0.25 cycles per element. P13 is within 32% of it.

### ✅ CPU Implementation
- Explicit intrinsics in `src/simd.hpp`: AVX-512 (16 lanes, masked tails, no scalar remainder loops), AVX2 + FMA + F16C (8 lanes), and a portable scalar path. On this host AVX-512 is 1.3–1.6× faster per element than AVX2.
- The inner loops have no branches besides the loop back-edge. A null bias/gamma becomes a stride-0 constant vector (address mask 0) instead of a per-element test. An `objdump` audit of the pass-1 and pass-2 loop bodies shows only loads, conversions, add/mul/FMA and stores, with no stack traffic.

---

## 🎯 Challenge 6: Zero-Allocation Freestanding Worker & Coordinator Decoupling

### The Dilemma
In high-throughput, latency-critical multi-core execution (40 cores), worker threads running the computational pipeline must operate in a pure freestanding execution model:
1. **Zero Dynamic Allocation**: Dynamic heap operations (`std::vector::resize`, `malloc`, `new`) in worker loops invoke global allocator locks (`ptmalloc`), inducing thread lock contention and OS scheduler jitter across 40 cores.
2. **Zero Exception Overhead (`-fno-exceptions`)**: C++ exception unwinding tables pollute instruction cache and increase binary footprint. Worker threads must be freestanding with zero `throw` statements (using `DSA_ASSERT` or status flags).
3. **Coordinator-Worker Orchestration**: Planning (`AdaptiveTiler::Plan`) must NOT be repeated by all 40 worker threads. A single Coordinator (`DaePipeline::Execute`) pre-evaluates the plan outside the OpenMP parallel region and provides a pre-allocated 64-byte aligned reduction workspace buffer (`float* workspace`), dispatching stateless worker routines across OpenMP threads.
4. **Lightweight Value-Semantics**: `LocalTensor<T>` is a 16-byte view (similar to `std::span`). It must be passed by value or const-reference without double-pointer indirection (`LocalTensor*`).

### The Objective
Refactor the inner `Core` execution into a freestanding worker routine:
1. Replace heap containers (`std::vector<float> sums`, `std::vector<Cols> cols`) with fixed-size stack arrays (e.g. `float sums[128]`, `Cols cols[32]`).
2. Remove any `throw` statements from `Core`, replacing them with `DSA_ASSERT`.
3. Accept the pre-allocated reduction workspace buffer passed down from the Coordinator.

---

## 📏 Cost Model & Methodology

`HardwareModel` (`src/adaptive_tiler.hpp`) holds every constant the planner uses. `Target()` and `Host(P)` are two instances of the same equations:

- **Decomposition.** The planner keeps the candidate whose slowest core finishes first. On the host the candidates are inline, rows and row-major Split-D, costed as launch + barrier + elements · `elemNs`. On the target, the column band is a fourth candidate, and each core costs one `SyncAll` (split plans only) plus $(n+2)(W/n + \text{tileNs})$, with $W = \max(\text{vector cycles}/\text{clock},\ \text{DMA bytes} \cdot \text{byteNs})$.
- **Vector cycles.** The planner counts every instruction the kernel issues with the runtime's own costs (`DaeIsa`). The tests require the count to equal the runtime's cycle tracker on every executed plan, and it does.
- **Pipeline depth.** $n^* = \sqrt{2W/\text{tileNs}}$.
- **Scratchpad.** The `DaeLayout` knapsack, including the choice to replicate γ/β.

### Target: `HardwareModel::Target()`

| Constant | Value | Source |
| :--- | ---: | :--- |
| `cores`, `quantumBytes`, `spmBytes` | 40, 32 B, 195,584 B | `include/dsa_runtime.hpp`: `MAX_HARDWARE_CORES`, `DMA_ALIGN_BYTES`, `SCRATCHPAD_SAFE_WATERLINE` |
| `launchNs` | 0 | README: zero fork/join cost on the target |
| `byteNs` | 40 / 850 ns | ~850 GB/s aggregate, shared by 40 cores. It reproduces the P13 target: the busiest core streams 786,432 elements × 6 B at 21.25 GB/s = 222.0 µs, against a target of 223 µs |
| Vector work | `DaeIsa` | The runtime's instruction costs: `Add`, `Mul`, `Muls`, `Cast` cost 2 cycles per 256-byte repeat + 13; `BlockReduceSum` costs 1 per repeat + 14; `VectorReduceSum` costs 2 per repeat + 15; `Duplicate` costs 1 per repeat + 18; `VectorInvRms` costs 16. The kernel never issues `GetValue` (500) or `WholeReduceSum` (14 per repeat + 14) |
| `clockGHz` | 1.5 | Assumed. The runtime counts cycles and defines no clock, so this only converts cycles to time. For the 15 profiles, every clock from 1.0 to 4.0 GHz picks the same decomposition, and every clock from 1.5 to 2.5 GHz the identical plan. At 1.2 GHz and below, some profiles become vector-bound and take one or two more tiles (P10 also starts replicating γ/β); from 3 GHz up, P14 stops replicating γ/β, because its vector work then hides under DMA |
| `syncNs` | 7500 cycles / clock = 5000 ns | `SyncAll`. P05 still splits: whole rows model at 28.1 µs, the column band at 12.3 µs |
| `tileNs` | 800 ns | P01–P03 run one tile per core and exceed their streaming time $W$ by 1.45–1.96 µs, which is one fill plus one drain of 0.73–0.98 µs each. The tilings that Challenges 2 and 3 ask for bracket it independently. P04 in two tiles ($n^* = 2$, with $W$ = 1.45 µs of vector work) requires $465 < \text{tileNs} \le 1292$ ns. P08 in tiles of ≥ 24 rows ($n^* \le 11$) requires $\text{tileNs} > 561$ ns |

The model ranks decompositions and tile sizes; it does not predict latency. The world-record targets imply 850–2070 GB/s effective, because L2 reuse across iterations helps, and a single `byteNs` cannot capture that. The clock and `tileNs` should be recalibrated on physical target hardware.

The runtime simulates the DAE machine's function and capacity: scratchpad budget, DMA alignment, queue lifecycle and instruction cycle counts. It does not simulate the timing of DMA/vector overlap or vector operand alignment. Timing comes only from the model above.

### Host: `HardwareModel::Host(P)`

The host constants were measured on the reference CI host, a 4-core Cascade Lake KVM guest (AVX-512, 1 MB L2/core, 33 MB L3, ~40 GB/s DRAM):

| Constant | Value | Measurement |
| :--- | ---: | :--- |
| `cores` | `min(P, 40)` | `P = omp_get_max_threads()`, or 1 inside an enclosing parallel region |
| `launchNs` | 3500 | 4-thread floor of the kernel on tiny inputs (an empty `omp parallel` costs 2.3–2.9 µs; libgomp issues a futex wake per barrier) |
| `syncNs` | 1000 | Extra cost of one team barrier (0.5–1 µs) |
| `elemNs` | 0.3 | One core, cache-resident rows: FP32 0.27–0.31, FP16 0.26–0.39 ns/element |
| `batchElems` | 2048 | Batch sweep $B \in \{1, 4, \dots, 64\}$ for $D \in \{64, 192, 256, 400\}$ |

- **Crossover.** The measured T=1/T=4 crossover is 12–16K elements, which matches $\text{launch} \cdot P / ((P-1) c) \approx 15.6\text{K}$.
- **Team size.** Resizing an OpenMP team between calls cost 70–330 µs per call. For that reason the team is always either the caller alone or all `P` threads.
- **Planning cost.** Planning takes 60–75 ns on the host, so each thread memoizes the plan of its last shape. A target plan, which models every candidate per core, takes up to 40 µs.
- **Portability.** Hosts with a different fork/join cost or core speed can re-tune these constants; the formulas stay the same.

### Tests

- **`tests/test_correctness.cpp`** forces every host plan variant (Split-D, streaming, recomputed Z) with 1–40 threads against an FP64 reference. It also checks for writes outside Y and for bitwise-identical results across serpentine directions.
- **`tests/test_dae_pipeline.cpp`** (52,630 checks):
  - It checks the invariants of the 15 full-size target plans, and of 2,448 more shapes in every forced mode.
  - It runs a plan that fills the scratchpad to the byte (195,584 B).
  - It runs model-chosen, rows, row-major Split-D and column-band plans of 19 shapes through the DAE runtime: with and without γ/β, on aligned and offset bases, in FP32/FP16/BF16.
  - Every run must match an FP64 reference and leave canaries around Y intact. It must have zero scalar stalls, one `SyncAll` exactly when split, a scratchpad claim equal to the plan, the planner's cycle count, and zero padded transfers wherever rows end on 32-byte blocks.
  - Mutations it catches (failed checks): an unplanned 64-byte buffer (467), one `GetValue` per row (839), a wrong gather tree (288), an off-by-one record range (439), missing γ/β replication (354).
- **`tests/test_dsa_runtime.cpp`** (`ctest -R dsa_runtime_sanitizer`) checks every sanitizer trap, including the new queue-lifecycle, address-alignment and `DataCopyPad` checks.
