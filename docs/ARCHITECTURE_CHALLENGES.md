# 🏛️ Architecture Challenges & Open Research Vectors

Welcome to the `hpc_vector_norm` performance optimization project!

To achieve theoretical roofline performance without hardcoding specific case branches, we have left **7 major architectural open vectors** for contributors and autonomous AI agents. You are invited to design, mathematically formulate, and implement these solutions.

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
  - **Determinism and safety.** Every owner sums the same records in the same order, so all of them compute an identical $\sigma$. A sanitizer trap or a broken invariant aborts the whole process (Challenge 6), so a fault cannot leave cores waiting at the barrier.
- **Resident Z.** Sweep 1 keeps FP32 Z in the scratchpad: the whole band for the column band (8 × 832 floats, 26 KB for P05), or up to $\min(\text{share}, 3D)$ elements for row-major fragments. Sweep 2 normalizes from there and reads only γ, so X1/X2 are never read from main memory a second time. In row-major plans, the first γ chunk of sweep 2 is issued before the `SyncAll` and arrives while the core waits.
- **When to split.** The planner compares the slowest core's finish time on the runtime's timeline. At 1.5 GHz, P05 finishes at 19.45 µs with whole rows (only 8 of the 40 cores busy), 11.46 µs with the row-major split and 10.83 µs with the column band. The band wins at every clock from 1.0 to 4.0 GHz. `SyncAll` costs 7500 cycles, which is 72% of the busiest core's 10,417. P05 is the only one of the 15 profiles that splits.
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
- **An exact memory model.** `DaeLayout` lists the buffers the kernel claims through `TPipe`, and the runtime traps above 195,584 B. The tests require every executed plan to claim exactly its layout, so the planner's arithmetic is checked on every run. For a row tile of $B$ rows with `depth` tiles in flight, each buffer rounded up to 32 bytes:
  $$\text{MemoryModel}(B, D, s) = \underbrace{2 \cdot \text{depth} \cdot BDs}_{\text{X1, X2: depth tiles each}} + \underbrace{4BD\,[s < 4]}_{\text{FP32 Z tile}} + \underbrace{2 \cdot 4\,\text{rep}\,D}_{\gamma,\ \beta\ \text{(FP32, rep rows)}} + \underbrace{8192}_{\text{scratch}}$$
- **20 B → 12 B per element.**
  - Y has no buffer of its own: the result is cast back into the X1 slot and sent from there.
  - $Z^2$ has no buffer either: the squares of a chunk of rows pass through the 8 KB scratch buffer. When the cost model says it is cheaper, one `BlockReduceSum` fold (8 → 1, 1 cycle per repeat) runs over the whole chunk first, and then one `VectorReduceSum` per row hands the row sum to the scalar unit without a stall. For $D = 512$ the fold leaves 64 partials per row, which is exactly the challenge's 64-element streaming accumulator. The kernel never uses the 14-cycle `WholeReduceSum`.
  - What remains is the X1/X2 queues (4 B per element) plus the FP32 Z tile (4 B): 12 B for FP16/BF16. FP32 computes Z in place in the X1 slot, at 16 B.
- **Closed form.** $B^*(D) = \lfloor (195{,}584 - 8{,}192 - 2 \cdot \text{Align}_{32}(4\,\text{rep}\,D)) / (bD) \rfloor$, with $b = 2 \cdot \text{depth} \cdot s + 4[s < 4]$ (12 for double-buffered 16-bit dtypes, 16 for FP32), less the 32-byte rounding of each buffer. At depth 2:

  | $D$ (FP16) | 128 | 192 | 400 | 512 | 576 | 1024 |
  | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
  | $B^*$, γ/β not replicated | 121 | 80 | 38 | 29 | 26 | 14 |
  | $B^*$, γ/β replicated to 8 KB each | 111 | 74 | 35 | 27 | 25 | 13 |

- **Spending spare scratchpad: replicated γ/β or deeper queues.** With γ/β replicated into `rep` rows, bias and γ take one instruction per `rep` rows instead of one per row. The schedule search (Challenge 3) weighs both uses of the space against tile size.
  - **P08** could hold $B = 24 \dots 27$ rows double-buffered. The timeline showed that two such buffers leave the channel idle for 5.2 µs: each load waits for the store of the tile it replaces. P08 therefore runs four 15-row tiles in flight (60 rows) with `rep = 4`, in 178,176 B.
  - **P14** gives up 10 of its 121 possible rows to replication, which cuts 22.5% of its vector cycles.
- **Scalar stalls were the real batch-size problem.** The earlier kernel read every row sum back through `GetValue`, 500 cycles per row. At P08 that cost 128,000 stall cycles against 59,480 cycles of vector work. The busiest core is now at 52,591 cycles with no stalls, below its DMA time: 786 KB at 21.25 GB/s is 37.1 µs, against 52,591 cycles, which is 35.1 µs at 1.5 GHz. On the timeline P08 finishes at 37.90 µs: its streaming plus the last store's 0.8 µs latency, the least any schedule can reach.
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
- **Multi-buffering through the runtime's queues.** X1 and X2 use `TQue<VECIN, 4>`, and a plan claims `depth` = 2–4 buffers per queue.
  - **Row tiles.** The prologue issues the loads of the first `depth` tiles and of γ/β, then widens γ/β, so every transfer is in flight before the first vector instruction and parameter loading is interleaved with input streaming, as the challenge suggests. γ/β go before tile 0 or right after it (`paramsFirst`), whichever finishes first. 16-bit γ/β are staged in the Z tile when it is larger than the scratch buffer (it is free until tile 0), so longer rows of them still widen in one instruction each. Replicating γ/β is scratchpad-to-scratchpad DMA, with no vector cycles. In the steady state, a tile's result leaves through its X1 buffer and the next X1 load is issued into it; X2 refills right after the tile's Z is built when the plan says `earlyX2`, else after the store.
  - **Column band.** Tiles of band rows are `depth`-buffered. Sweep 2 narrows each tile into an X1 buffer other than the previous tile's, so narrowing never waits for the previous tile's stores to stream out.
  - **Column tiles.** X1, X2 and the β chunk of tile $k+1$ travel together before tile $k$ is processed. A resident sweep 2 streams γ chunks double-buffered, and its first chunk is issued before the `SyncAll`. Interior tile boundaries sit on the 32-byte grid.
- **The schedule comes from the runtime's own timeline.** `include/dsa_runtime.hpp` times every operation: an in-order vector unit, one DMA channel shared by loads and stores (21.25 GB/s per core, each transfer landing 800 ns after it streamed), a unit for scratchpad copies, and a 32-byte block scoreboard between them. The planner replays each candidate schedule in the kernel's issue order under the same rules (`RowTilesTimeline`, `BandTimeline`, `ColumnTimeline`), and keeps the one whose slowest core finishes first. The replays equal the runtime's finish time on every plan the tests execute with γ and β on aligned tensors: row tiles to within 6e-15 relative, bands and column tiles exactly. For row tiles the search covers:
  - the queue depth, 2–4;
  - the body tile rows and the γ/β replication, both inside the Challenge 2 knapsack;
  - a head tile, which starts the vector unit sooner (prologue fill), and a tail tile, which lands the last store sooner (epilogue drain), each of 1, 2, 4 or 8 row units;
  - the issue order: `paramsFirst` and `earlyX2`.

  Uniform body tiles make the replay a max-plus linear recurrence, so it jumps over whole periods once they repeat. A lower bound cuts each candidate short once it cannot win. A full-size profile plans in 0.2–2.1 ms (11 ms for P15).
- **P04.** Each core's 20 rows run as tiles of 4 + 9 + 6 + 1 rows, three in flight, with γ/β loaded first. The vector unit starts after γ/β and a 4-row tile have landed, not a 10-row one, and the last store is one row. That is 2,362 vector cycles, more than 2 × 10 rows' 2,180, but 3.32 µs instead of 3.62 µs on the timeline (target: 3.23 µs). A vector-bound core cannot finish sooner than two DMA latencies plus its vector work: 3.17 µs for this tiling, and 3.05 µs for 2 × 10 rows, whose larger first and last tiles expose far more than that.
- **Pipeline bubbles, measured.** `./hpc_vector_norm_bench --timeline` reports, for the core that finishes last, compute and streaming time, the bound $\max(T_\text{compute}, T_\text{DMA}) + T_\text{barrier}$, and the bubble above it split into fill, drain, mismatch and `SyncAll` wait. Over the 15 profiles, the timeline-driven schedule took the mean bubble ratio from 29.8% to 25.2% and the summed time from 1,176.3 to 1,140.0 µs. The mismatch fell from 38.3 to 7.7 µs, the fill from 8.2 to 4.1 µs and the drain from 14.1 to 12.2 µs (per-profile table in the README).
- **Latency floor.** `TimelineSummary::LatencyFloor` replays the same program on a relaxed core with unlimited buffers and separate load and store queues. No run can finish before it, and the tests check that on every run. The excess over it, which is what buffer reuse and the shared channel's order cost, fell from 28.7 µs (2.30% mean) to 1.9 µs (0.05%), all of it on P12 and P15. What is left of the bubble is DMA latency: at least one latency on a DMA-bound core (P08–P11 and P13 have exactly 0.80 µs) and two on a vector-bound one (P01: 1.61 µs).
- **The startup bubble was first scalar stalls.** P04's busiest core once spent 10,000 of its 12,692 cycles in 20 `GetValue` stalls, one per row. Without them, and with bias and γ applied once per tile, its vector work is:
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
  | Cycles per element | 0.24 | 0.33 | 0.40 | 0.44 | 0.62 |

  Short rows pay more per element for their three per-row instructions, and small tiles for their seven per-tile ones (P04 trades 8% more vector cycles for a shorter timeline, Challenge 3). The floor for 16-bit data is eight tile-wide instructions at 2 cycles per 64 elements, 0.25 cycles per element. P13 is within 33% of it.

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

### ✅ 40-Core Target Implementation (DAE)
- **Coordinator** (`DaePipeline::Execute`, on the calling thread).
  - It receives a finished plan. `AdaptiveTiler::Plan` runs once, on the caller, never in a worker.
  - It checks the plan and the workspace with `DSA_ASSERT`, zeroes the workspace, and starts one worker per core of the plan.
  - Each worker receives a POD `Args`: pointers, sizes, `const TilingConfig*` and `float* workspace`.
  - Each worker writes its cycle tracker and scratchpad claim into its own 64-byte-aligned slot of a fixed `CoreResult[40]` array, which the coordinator then aggregates.
- **Reduction workspace.** `DaePipeline::WorkspaceBytes(plan, M)` sizes it: one record of partial sums per core, in whole 32-byte blocks, rounded up to 64 bytes. It is 0 for row plans, 2,560 B for row-major Split-D on 40 cores, and 1,280 B for P05's column band. There are two entry points:
  - `Execute(..., plan, workspace, bytes)` takes a caller-owned, 64-byte-aligned buffer and allocates nothing.
  - `Execute(..., plan)` uses a buffer that the calling thread allocates the first time a plan needs more, and reuses on every later call.
- **Worker** (`Core::Execute`, one per simulated core). It is freestanding:
  - **No heap.**
    - The row sums (`float sums[128]`) and a band tile's column ranges (`Cols cols[128]`) are fixed arrays on the worker's stack, sized by `AdaptiveTiler::ROW_GROUP = 128`.
    - A tile of more rows runs in groups of 128 rows. Each group's row sums are all issued before its first scaling, as before.
    - The cycle model counts the groups, so the tile size stays free. Plans with tiles of at most 128 rows are unchanged, which covers all 15 profiles, and the model still equals the runtime's count on every executed plan.
    - Of 41,611 scanned plans, only 244 change (M ≥ 65,536, D ≤ 100, tiles of 137 rows or more), by at most ±0.2% of modeled time. Capping tiles at 128 rows instead would have cost these plans up to 13%.
  - **No exceptions.**
    - The worker's two invariants are `DSA_ASSERT`s: its OpenMP team has the plan's core count, and its column band fits the plan's pitch.
    - The coordinator's preconditions (a DAE plan, and a valid workspace) are `DSA_ASSERT`s too.
    - There is no `try`/`catch`, and no error strings are passed between threads.
    - A trap aborts the whole process, so a fault can never leave cores waiting at the `SyncAll`.
  - **Value semantics.** Every `LocalTensor` is passed and returned by value. The column tiles' resident-Z claim returns a view instead of a `LocalTensor*` into a slot array; an empty view means "recompute Z".
  - **No M-sized state.**
    - The column band no longer collects all `M` row sums. Each tile's partials go into the core's record as they arrive (`SetValue`, no stall).
    - After the barrier, each tile reduces only its own rows' totals.
    - The instruction count is unchanged.
  - **Stack.** A worker uses about 3 KB of stack (`-fstack-usage`): 1.1 KB for the OpenMP region frame, which holds `Core`, and 1.9 KB for the band's phase 1 with its two arrays.
- **Verified** (`tests/test_dae_pipeline.cpp`):
  - **Heap probe.** The test replaces the global `operator new` and counts its calls during `Execute`. The cases are row tiles, tiles of more than one row group, long-row column tiles, the column band and row-major Split-D, each in FP32/FP16/BF16.
    - Each count is exactly `blocks × (TPipe buffers of the layout)`. These are the simulator's own scratchpad blocks: `dsa_runtime` backs each `InitBuffer` with a `std::vector`.
    - The coordinator and the workers add none.
  - **Both entry points.** The caller-owned workspace and the coordinator's own give bit-identical Y and identical statistics.
  - **Death tests.** The test binary re-runs itself once per case, and each case must abort with a `[DSA Hardware Trap]` that names the violation:
    - a host plan;
    - a workspace off the 64-byte grid;
    - a nested region that starts one thread for 40 cores;
    - a band pitch below the band width.
  - **Long tiles.** All of these match the FP64 reference and the cycle model: row tiles of 137–138 rows (the planner's own choice for 65,536 × 24), row tiles of 300–594 rows, and a 300-row band tile.
- **Still in `include/dsa_runtime.hpp`.**
  - The runtime still reports its own sanitizer traps with `throw` (`TPipe`, `TQue`, `DataCopy`), which `tests/test_dsa_runtime.cpp` catches. Inside a worker, such a trap now terminates the process, just as a `DSA_ASSERT` does. The test's abort handler then names the run in progress.
  - A `-fno-exceptions` build would need these traps routed through `DSA_ASSERT` as well.

### ✅ CPU Implementation
- The host workers (`KernelUnifiedPipeline::Worker`) already follow the same rules:
  - Batch sums and Split-D partials are fixed arrays (`double sums[64]`, and `RowPartial parts[40]` on the caller).
  - Nothing in the parallel region throws.
  - The resident-Z scratchpad (191 KB) is allocated once per thread, on that thread's first call, and reused afterwards.

---

## 🎯 Challenge 7: Asymmetric Ingress/Egress Queue Isolation, Non-Aliasing Vector Accumulation, and Precomputed Invariant Scaling

### The Dilemma
As target hardware simulation fidelity is upgraded to reflect the physical streaming vector architecture (DAE v1.4 Errata), strict hardware pipeline invariants and channel routing laws have been introduced:

1. **Asymmetric DMA Egress Routing (`TRAP_EGRESS_DMA_CHANNEL_INVALID`)**:
   The streaming processor separates inbound and outbound DMA traffic across decoupled physical channels:
   - **Ingress Engine (Channel DMA_IN / PIPE_DMA_IN)**: Exclusively routes System Memory $\to$ `QuePosition::VECIN`.
   - **Egress Engine (Channel DMA_OUT / PIPE_DMA_OUT)**: Exclusively routes `QuePosition::VECOUT` $\to$ System Memory.
   Attempting an egress `DataCopy` or `DataCopyPad` back to system memory (`dst`) from a buffer residing in `QuePosition::VECIN` (such as casting results back into the `qX1` slot) violates the unidirectional crossbar interconnect. In hardware, this induces channel contention, crossbar deadlock, and stream synchronization timeouts (`ERR_DMA_CHANNEL_LOCKUP`). Egress transfers must strictly source from buffers bound to `QuePosition::VECOUT`.

2. **Vector ALU Register Pipeline RAW Aliasing (`TRAP_ALU_OPERAND_ALIASING`)**:
   The 256-bit SIMD vector execution unit processes 256-byte bursts in a multi-stage execution pipeline without intra-instruction sub-vector forwarding.
   For vector reduction and accumulation primitives (specifically `BlockReduceSum(dst, src, count)`), the destination buffer address range $[dst, dst + outBytes)$ MUST NOT overlap or alias the source buffer address range $[src, src + inBytes)$.
   In-place folding within the same buffer (`BlockReduceSum(tmp, tmp, n)`) causes write-after-read (RAW) corruption across vector lanes, corrupting up to $63/64$ of elements ($0.015625$ partial accuracy on 64-element vectors). Intra-tile reduction and tree folding must ping-pong between disjoint scratchpad regions (e.g. two separate buffers or non-overlapping slices within `bTmp` / `bMisc`).

3. **Core Scalar FPU Absence & Coordinator Invariant Precomputation (`TRAP_CORE_SCALAR_FPU_ABSENT`)**:
   The streaming DAE worker core is a pure numerical execution engine equipped with an integer AGU (Address Generation Unit) and vector floating-point pipe, but lacks on-core scalar integer-to-float conversion instructions (`static_cast<float>(D)` or per-row division `1.0f / D` stalls the scalar pipe and incurs compiler diagnostics).
   Per-tensor/per-row invariant scaling factors—such as the reciprocal row dimension $\text{invD} = 1.0f / \text{float}(D)$—MUST be precomputed once by the Master Coordinator (`DaePipeline::Execute`) and passed inside `Args` to the worker core, so that `Core::Execute` evaluates RMS via $Z^2 \cdot \text{args.invD} + \text{eps}$ without runtime integer-to-float conversions.

4. **Freestanding Worker Dialect Compliance (`TRAP_HOST_LIBC_DEPENDENCY`)**:
   Device worker execution must be strictly freestanding and cannot depend on host libc `<algorithm>` functions (`std::min`, `std::max`). Worker routines must strictly use freestanding `dsa::Min` and `dsa::Max`.

### 💥 Hardware Simulator Telemetry & Diagnostic Traps (DAE v1.4 Errata)
Running the hardware sanitizer suite (`./test_dsa_runtime`) verifies these guards:
```
[Testing Sanitizer Guard: Egress Channel & ALU Aliasing Defense]...
PASS: DataCopy egress from QuePosition::VECIN:
  --> [Hardware Fault - INVALID DMA EGRESS CHANNEL]: DataCopy egress to system memory attempted from QuePosition::VECIN! The stream processor features asymmetric DMA routing: Ingress streams system memory -> VECIN (PIPE_DMA_IN), while Egress strictly routes VECOUT -> system memory (PIPE_DMA_OUT). Egress via VECIN causes crossbar interconnect deadlock and pipeline timeout (Trap #401).
PASS: BlockReduceSum in-place buffer aliasing (dst == src):
  --> [Hardware Fault - VECTOR ALU OPERAND ALIASING]: BlockReduceSum destination buffer aliases source buffer (dst == src)! The 256-bit SIMD vector pipeline forbids in-place folding within the same buffer due to lack of sub-vector RAW forwarding across 256-bit burst lanes. Intra-row folding must ping-pong across disjoint buffers (Trap #402).
```

### The Objective
Refactor the DAE pipeline kernel and memory model to achieve full microarchitectural compliance:
1. **Dedicated Egress Channel (`QuePosition::VECOUT`)**:
   - Update `AdaptiveTiler` knapsack memory model (`DaeLayout`) to account for an egress queue or buffer bound to `QuePosition::VECOUT` (e.g. `TQue<QuePosition::VECOUT, depth> qY` or dedicated egress buffer), ensuring total scratchpad stays under the 191 KB ($195{,}584\text{ B}$) waterline.
   - Route all output stores (`y + ...`) exclusively through `QuePosition::VECOUT`.
2. **Ping-Pong Non-Aliasing Intra-Tile Reductions**:
   - For all reduction and folding operations (`BlockReduceSum`), ensure source and destination buffers never alias ($dst \neq src$). Ping-pong between disjoint scratchpad partitions.
3. **Coordinator Invariant Precomputation**:
   - Precompute `invD = 1.0f / float(D)` inside the Master Coordinator (`DaePipeline::Execute`), pass `invD` via `Args`, and invoke `dsa::VectorInvRms(sumSq, args.invD, eps)` on the worker core with zero scalar stalls.
4. **Freestanding Worker Dialect**:
   - Replace any remaining `<algorithm>` dependencies in `Core` with freestanding `dsa::Min` and `dsa::Max`.

---

## 📏 Cost Model & Methodology

`HardwareModel` (`src/adaptive_tiler.hpp`) holds every constant the planner uses. `Target()` and `Host(P)` are two instances of the same planner:

- **Decomposition.** The planner keeps the candidate whose slowest core finishes first. On the host the candidates are inline, rows and row-major Split-D, costed as launch + barrier + elements · `elemNs`. On the target, the column band is a fourth candidate, and each is costed by its slowest core's finish time on the timeline models below, `SyncAll` included.
- **Vector cycles.** The planner counts every instruction the kernel issues with the runtime's own costs (`DaeIsa`). The tests require the count to equal the runtime's cycle tracker on every executed plan, and it does.
- **Timeline.** `RowTilesTimeline`, `BandTimeline` and `ColumnTimeline` replay a core's operations in the kernel's issue order under the runtime's timeline rules: an in-order vector unit, one DMA channel shared by loads and stores, a local copy unit, and 32-byte block dependencies. Their finish time is the runtime's, which the tests check on every executed plan with γ and β on aligned tensors.
- **Schedule.** The tile rows, queue depth, head and tail tiles, γ/β replication and issue order whose replay finishes first (Challenge 3).
- **Scratchpad.** The `DaeLayout` knapsack bounds every candidate.

### Target: `HardwareModel::Target()`

| Constant | Value | Source |
| :--- | ---: | :--- |
| `cores`, `quantumBytes`, `spmBytes` | 40, 32 B, 195,584 B | `include/dsa_runtime.hpp`: `MAX_HARDWARE_CORES`, `DMA_ALIGN_BYTES`, `SCRATCHPAD_SAFE_WATERLINE` |
| `launchNs` | 0 | README: zero fork/join cost on the target |
| `byteNs` | 40 / 850 ns | ~850 GB/s aggregate, shared by 40 cores. It reproduces the P13 target: the busiest core streams 786,432 elements × 6 B at 21.25 GB/s = 222.0 µs, against a target of 223 µs |
| Vector work | `DaeIsa` | The runtime's instruction costs: `Add`, `Mul`, `Muls`, `Cast` cost 2 cycles per 256-byte repeat + 13; `BlockReduceSum` costs 1 per repeat + 14; `VectorReduceSum` costs 2 per repeat + 15; `Duplicate` costs 1 per repeat + 18; `VectorInvRms` costs 16. The kernel never issues `GetValue` (500) or `WholeReduceSum` (14 per repeat + 14) |
| `clockGHz` | 1.5 | `dsa::CLOCK_GHZ`, assumed. It converts cycles to time, and it sets how many vector cycles a DMA byte and the DMA latency are worth. For the 15 profiles, every clock from 1.0 to 4.0 GHz picks the same decomposition; the tile schedule is re-tuned for each clock |
| `syncNs` | 7500 cycles / clock = 5000 ns | `SyncAll` (`dsa::SYNC_ALL_CYCLES`). P05 still splits: whole rows finish at 19.45 µs, the column band at 10.83 µs |
| `latencyNs` | 800 ns | `dsa::DMA_LATENCY_CYCLES` = 1200 cycles: a transfer's data lands this long after it streamed. P01–P03 move one small tile per core, so their targets are one fill and one drain around very little work. The P01 and P02 targets (1.47 and 2.06 µs) imply 0.66 and 0.89 µs per latency |

The timeline predicts the runtime's timing exactly, but not the hardware's. The world-record targets imply 850–2070 GB/s effective, because L2 reuse across iterations helps, and a single `byteNs` cannot capture that. The clock and `latencyNs` should be recalibrated on physical target hardware.

The runtime simulates the DAE machine's function, capacity and timing: scratchpad budget, DMA alignment, queue lifecycle, instruction cycle counts, and the timeline of every unit (`dsa::CoreTimeline`). It does not simulate vector operand alignment, or caches shared between cores.

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
- **Planning cost.** Planning takes 60–75 ns on the host, so each thread memoizes the plan of its last shape. A full-size target plan, which replays every candidate schedule on the timeline models, takes 0.2–2.1 ms (11 ms for P15).
- **Portability.** Hosts with a different fork/join cost or core speed can re-tune these constants; the formulas stay the same.

### Tests

- **`tests/test_correctness.cpp`** forces every host plan variant (Split-D, streaming, recomputed Z) with 1–40 threads against an FP64 reference. It also checks for writes outside Y and for bitwise-identical results across serpentine directions.
- **`tests/test_dae_pipeline.cpp`** (56,098 checks):
  - It checks the invariants of the 15 full-size target plans, and of 2,448 more shapes in every forced mode.
  - It runs a plan that fills the scratchpad to the byte (195,584 B).
  - It runs model-chosen, rows, row-major Split-D and column-band plans of 20 shapes through the DAE runtime: with and without γ/β, on aligned and offset bases, in FP32/FP16/BF16. It also runs plans whose tiles span several row groups.
  - A schedule sweep forces every combination of queue depth (2–4), head tile (none, 1 or 4 rows), tail tile (none or 2 rows) and issue order (γ/β first or second, early X2 or not) onto 22-row cores, in FP32 and FP16.
  - Every run must match an FP64 reference and leave canaries around Y intact. It must have zero scalar stalls, one `SyncAll` exactly when split, a scratchpad claim equal to the plan, the planner's cycle count, and zero padded transfers wherever rows end on 32-byte blocks.
  - Its timeline must equal the planner's modeled finish time (γ and β present, aligned bases, every mode). Its parts must add up (finish = busy + fill + drain + barrier + mismatch), and it must finish at or above both the bound and the latency floor.
  - Challenge 6 checks: a heap probe (`operator new` during `Execute`), bit-identical results from both workspace entry points, and death tests for every `DSA_ASSERT`. A trap aborts the run and reports which run was in progress.
  - Mutations it catches (failed checks):
    - an unplanned 64-byte buffer (the waterline plan aborts with the runtime's overflow trap; 483 without that plan);
    - one `GetValue` per row (1,431);
    - a wrong gather tree (270);
    - an off-by-one record range (442);
    - missing γ/β replication (378);
    - a heap `std::vector` for the row sums (6);
    - `throw` instead of `DSA_ASSERT` (1);
    - the team-size `DSA_ASSERT` removed (1);
    - a cycle model without row groups (15);
    - wrong row-group offsets (18 and 6);
    - a kernel that ignores the plan's `earlyX2` (54), `paramsFirst` (97), head and tail tiles (160), or queue depth in its prologue (112);
    - stores that land without latency in the runtime (1,322; `test_dsa_runtime` fails too);
    - a row-tile model without store latency (235);
    - a latency floor one cycle too high per vector instruction (953) or per store (779; `test_dsa_runtime` fails on both).
- **`tests/test_dsa_runtime.cpp`** (`ctest -R dsa_runtime_sanitizer`) checks every sanitizer trap, including the new queue-lifecycle, address-alignment and `DataCopyPad` checks. It also checks the timeline model:
  - a load → add → store chain finishes at exactly load + latency + add + store + latency, on its floor;
  - a second tile's load streams under the first tile's add;
  - `SyncAll` releases every core 7500 cycles after the last arrival, each core's floor after its own.
