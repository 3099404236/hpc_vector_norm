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
- **Flattened 32-byte decomposition.** The whole $M \times D$ tensor is cut into $U = \lceil MD / q \rceil$ units of one 32-byte DMA block ($q = 32 / s$ elements). Core $t$ of $P$ gets units $[\lfloor Ut/P \rfloor, \lfloor U(t+1)/P \rfloor)$ (`AdaptiveTiler::Range`), so any two cores differ by at most one block for every $M$, $D$ and $P$. The busiest core holds $\lceil U/P \rceil$ blocks, the minimum any split into 32-byte blocks can reach. The boundaries are offsets into the flattened tensor, so with a 32-byte-aligned base every boundary is a 32-byte-aligned address, even when $D \cdot s$ is not a multiple of 32. Per-row slicing cannot guarantee that.
- **P05 solved exactly.** With $U = 16384$ and $P = 40$, every row lands on exactly 5 cores. The boundaries are $\{x_0, \dots, x_5\} = \{0, 6544, 13104, 19648, 26208, 32768\}$, so each slice is 6544 or 6560 elements (409 or 410 blocks): 32-byte aligned and within 32 bytes of the other slices. An exactly equal split does not exist, since $16384 / 40 = 409.6$. The same code balances any shape to one block, for example 3 × 100003 FP16 → 7488–7504 elements per core, 7 × 5000 → 864–880, 13 × 12288 → 3984–4000.
- **Reduction tree.** Each core writes one 32-byte record $\{\Sigma_0, \Sigma_1, \text{row}_0, \text{row}_1\}$ covering its at most two shared rows to global memory, then passes one `SyncAll`. The owners of row $r$ are cores $\text{owner}(u_0) \dots \text{owner}(u_1)$, where $u_0$ and $u_1$ are the row's first and last units and $\text{owner}(u) = \lfloor ((u+1)P - 1) / U \rfloor$ (`RowOwners`). Each owner reads the owners' records with one contiguous `DataCopy`, sums them in core order, and finishes its fragment. The 5 owners of a P05 row therefore compute an identical $\sigma$. Every core reaches the `SyncAll`, including one whose sanitizer trapped, so a fault cannot deadlock the barrier.
- **Resident Z.** Sweep 1 writes each fragment's FP32 Z to the scratchpad: up to $\min(\text{share}, 3D)$ elements, 26 KB for P05. Sweep 2 normalizes from there and reads only γ, so X1/X2 are never read from global memory a second time.
- **When to split.** The planner compares busiest-core times, `t_split = sync + ⌈U/P⌉·q·c` against `t_rows` for whole-row units. At P05, rows cost 9.25 µs because only 8 of the 40 cores are busy, while split costs 2 + 1.85 µs. P05 is the only one of the 15 profiles that splits.
- **Verified** (`tests/test_dae_pipeline.cpp`):
  - 16 shapes, including the non-power-of-2 widths 5 × 7, 3 × 100, 7 × 200, 17 × 4097, 41 × 5000, 1 × 70001 and 3 × 100003;
  - each run in forced Split-D, forced-rows and model-chosen modes;
  - with and without γ/β, on 32-byte-aligned and offset bases, in FP32/FP16/BF16;
  - through the sanitizer-enabled runtime, checked against an FP64 reference.

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
- **An exact memory model.** `DaeLayout` lists the buffers the kernel claims through `TPipe`, and the runtime traps above 195,584 B, so the planner's arithmetic is checked on every run. For a row tile of $B$ rows (each buffer rounded up to 32 bytes):
  $$\text{MemoryModel}(B, D, s) = \underbrace{2 \cdot 2 \cdot BDs}_{\text{X1, X2 double-buffered}} + \underbrace{4BD\,[s < 4]}_{\text{FP32 Z tile}} + \underbrace{8D}_{\gamma,\ \beta\ \text{(FP32)}} + \underbrace{8192}_{\text{fold scratch}} + \underbrace{2048}_{\text{records}}$$
- **20 B → 12 B per element.**
  - Y has no buffer of its own: the result is cast back into the X1 slot and sent from there.
  - $Z^2$ has no buffer either: the squares pass through the 8 KB scratch chunk and `BlockReduceSum` folds, 8 → 1 per fold while the length stays a multiple of 8. For $D = 512$ the folds go 512 → 64 → 8 → 1, which plays the role of the 64-element streaming reduction. Each fold costs 1 cycle per repeat, and the kernel never uses the 14-cycle `WholeReduceSum`.
  - What remains is the X1/X2 queues (4 B per element) plus the FP32 Z tile (4 B): 12 B for FP16/BF16. FP32 computes Z in place in the X1 slot, at 16 B.
- **Closed form.** $B^*(D) = \lfloor (195{,}584 - 10{,}240 - 8D) / (bD) \rfloor$ with $b = 12$ for 16-bit dtypes and 16 for FP32, less the 32-byte rounding:

  | $D$ (FP16) | 128 | 400 | 512 | 576 | 1024 |
  | :--- | ---: | ---: | ---: | ---: | ---: |
  | $B^*$ (rows) | 120 | 37 | 29 | 26 | 14 |

  At $D = 512$, 29 rows take 192,512 B, so P08 is feasible anywhere in $B = 24 \dots 29$. The plan uses 26 rows (174 KB), because $n^* = 10$ tiles (Challenge 3) sets the tile below the cap.
- **Column tiles** (rows longer than a row tile, and Split-D fragments) cost $(6s + 4)$ B per element, plus the resident FP32 Z.

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
- **Ping-pong through the runtime's queues.** X1 and X2 use `TQue<VECIN, 2>`. The tile iterator (`Tiles`) issues the `DataCopy` for tile $k+1$ into the free slot before it dequeues tile $k$. The prologue loads tile 0. In the steady state, the load of tile $k+1$ overlaps the compute of tile $k$ and the store of tile $k-1$. The epilogue drains the last tile. Interior tile boundaries sit on the 32-byte grid.
- **Pipeline depth from the model.** With $n$ tiles through three stages that each cost $W/n + \text{tileNs}$ per tile, $T(n) = (n+2)(W/n + \text{tileNs})$ and $n^* = \sqrt{2W/\text{tileNs}}$. The Challenge 2 knapsack caps it. For P04, each core's 20 rows ($W = 1.08$ µs) give $n^* = 2$, so the core runs 2 × 10-row tiles and the second tile's load overlaps the first tile's compute. γ/β (0.8 KB) are loaded once per core and stay resident. $n^*$ sizes the tiles for P01–P08; from P09 on, the scratchpad bound sets the tile (full plan table in the README).
- **A runtime defect had silently disabled double buffering.** `TQue::AllocTensor` returned slot `(tail + allocatedCount) % depth` and ignored slots that were still enqueued. After `EnQue(tile k)`, the prefetch of tile $k+1$ therefore got tile $k$'s buffer and overwrote it before `DeQue`. The queue now tracks each slot (FREE → ALLOCATED → ENQUEUED → DEQUEUED) in FIFO order and traps misuse (`tests/test_dsa_runtime.cpp`). With the original queue, 701 of the DAE suite's 1056 checks fail.

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
- Every vector instruction covers a whole tile or an 8 KB scratch chunk (`Add`, `Mul`, `Muls`, `Cast`, `BlockReduceSum`), so the 256-byte repeat pipeline issues long runs of repeats with no per-element branches.
- The runtime's cycle tracker charges the busiest core between 0.24 and 0.53 vector cycles per element:

  | Profile | P09 | P13 | P08 | P05 | P04 |
  | :--- | ---: | ---: | ---: | ---: | ---: |
  | Cycles per element | 0.24 | 0.33 | 0.39 | 0.42 | 0.53 |

### ✅ CPU Implementation
- Explicit intrinsics in `src/simd.hpp`: AVX-512 (16 lanes, masked tails, no scalar remainder loops), AVX2 + FMA + F16C (8 lanes), and a portable scalar path. On this host AVX-512 is 1.3–1.6× faster per element than AVX2.
- The inner loops have no branches besides the loop back-edge. A null bias/gamma becomes a stride-0 constant vector (address mask 0) instead of a per-element test. An `objdump` audit of the pass-1 and pass-2 loop bodies shows only loads, conversions, add/mul/FMA and stores, with no stack traffic.

---

## 📏 Cost Model & Methodology

`HardwareModel` (`src/adaptive_tiler.hpp`) holds every constant the planner uses. `Target()` and `Host(P)` are two instances of the same equations:

- **Decomposition.** With $c = \text{elemNs} + 3s \cdot \text{byteNs}$, the planner compares `t_inline = MD·c`, `t_rows = launch + MaxLoad(rows)·c` and `t_split = launch + sync + MaxLoad(32 B)·c`.
- **Pipeline depth.** $n^* = \sqrt{2W/\text{tileNs}}$.
- **Scratchpad.** The `DaeLayout` knapsack.

### Target: `HardwareModel::Target()`

| Constant | Value | Source |
| :--- | ---: | :--- |
| `cores`, `quantumBytes`, `spmBytes` | 40, 32 B, 195,584 B | `include/dsa_runtime.hpp`: `MAX_HARDWARE_CORES`, `DMA_ALIGN_BYTES`, `SCRATCHPAD_SAFE_WATERLINE` |
| `launchNs` | 0 | README: zero fork/join cost on the target |
| `byteNs` | 40 / 850 ns | ~850 GB/s aggregate, shared by 40 cores. It reproduces the P13 target: the busiest core streams 786,432 elements × 6 B at 21.25 GB/s = 222.0 µs, against a target of 223 µs |
| `elemNs` | 0 | Assumes vector work overlaps DMA. The busiest core's vector work (Challenge 5 table: 0.24–0.53 cycles per element) stays below the 0.28 ns DMA share of a 16-bit element at clocks above ~1.2 GHz (P13) to ~1.9 GHz (P04). The runtime defines no clock |
| `tileNs` | 800 ns | P01–P03 run one tile per core and exceed their streaming time $W$ by 1.45–1.96 µs, which is one fill plus one drain of 0.73–0.98 µs each. The tilings that Challenges 2 and 3 ask for bracket it independently. P04 in two tiles ($n^* = 2$) requires $347 < \text{tileNs} \le 964$ ns. P08 in tiles of ≥ 24 rows ($n^* \le 11$) requires $\text{tileNs} > 560$ ns |
| `syncNs` | 2000 ns | P05's target minus its streaming time and one fill and drain: 5.39 − 1.85 − 2 × 0.8 = 1.94 µs. It only selects split vs rows, and P05 splits for any value below 7.4 µs |

The model ranks decompositions and tile sizes; it does not predict latency. The world-record targets imply 850–2070 GB/s effective, because L2 reuse across iterations helps, and a single `byteNs` cannot capture that. The two inferred constants (`tileNs`, `syncNs`) should be recalibrated on silicon.

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
- **Planning cost.** Planning takes 60–75 ns on the host (150 ns for the target), so each thread memoizes the plan of its last shape.
- **Portability.** Hosts with a different fork/join cost or core speed can re-tune these constants; the formulas stay the same.

### Tests

- **`tests/test_correctness.cpp`** forces every host plan variant (Split-D, streaming, recomputed Z) with 1–40 threads against an FP64 reference. It also checks for writes outside Y and for bitwise-identical results across serpentine directions.
- **`tests/test_dae_pipeline.cpp`** checks the invariants of the 15 full-size target plans. It then runs model-chosen, forced-split and forced-rows plans of 16 shapes through the DAE runtime: with and without γ/β, on aligned and offset bases, in FP32/FP16/BF16. It requires zero padded transfers wherever rows end on 32-byte blocks.
- **`tests/test_dsa_runtime.cpp`** (`ctest -R dsa_runtime_sanitizer`) checks every sanitizer trap, including the new queue-lifecycle, address-alignment and `DataCopyPad` checks.
