# hpc_vector_norm (High-Performance Fused Vector Normalization Engine)

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![OpenMP](https://img.shields.io/badge/Parallel-OpenMP-green.svg)](https://www.openmp.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An ultra-high-throughput, cache-conscious, vectorized C++ math kernel library for **Fused Residual Vector Normalization (`FusedResidualNormalize`)** targeting modern multi-core symmetric CPU architectures (up to 40 hardware execution threads with AVX-2 / AVX-512 SIMD vectorization).

---

## 🎯 Project Mission & Challenge

In modern signal processing, scientific simulations, and large tensor computations, normalizing multi-channel vectors while accumulating residual streams is a fundamental computational primitive:

$$Z_{i} = X_{1,i} + X_{2,i} + \text{bias}$$
$$\sigma_i = \sqrt{\frac{1}{D} \sum_{j=0}^{D-1} Z_{i,j}^2 + \epsilon}$$
$$Y_{i,j} = \frac{Z_{i,j}}{\sigma_i} \cdot \gamma_j$$

### 🏛️ Target Hardware Laws vs Host CI Environment (Crucial Directive)

Contributors and autonomous AI agents must distinguish between the **Target Hardware Architecture** and the **Host CI Testbed**:

| Architectural Dimension | Host CI Testbed (Development Environment) | Target Hardware Architecture (Physical Ground Truth) |
| :--- | :--- | :--- |
| **Execution Concurrency** | 4-core Cascade Lake VM (used for CI & logic verification) | **40 Dedicated Symmetric Cores (`P = 40`)** with zero fork/join cost |
| **DMA Memory Quantum** | 64-byte x86 Cache Line | **Strict 32-Byte DMA Block (`DMA_ALIGN_BYTES = 32`)** |
| **Scratchpad Buffer** | 191 KB L1 resident Z scratchpad | **Strict 191 KB (195,584 B) Scratchpad (SPM)** per core |
| **Memory Bandwidth** | ~40 GB/s DDR4 (host bus limitation) | **Aggregated High-Throughput Memory Subsystem (~850 GB/s)** |
| **SIMD Instruction Width** | AVX-512 (64B) / AVX2 (32B) | **256-Byte Repeat SIMD Vector Pipeline (2048-bit)** |

> **⚠️ Directives for Tiler & Kernel Optimization**:
> 1. **Do NOT overfit to the 4-core host!** While `AdaptiveTiler` should gracefully handle `threads <= 4` on the host to avoid OS thrashing during tests, the **mathematical planning model must be explicitly architected for 40 symmetric cores**.
> 2. **Alignment must honor 32 bytes**: The hardware DMA engine transfers memory in 32-byte blocks. All dimension slicing in Split-D should support 32-byte granularity.
> 3. **Latency Targets Reflect Peak Theoretical Memory Roofline**: Target latencies (e.g. P13 $223\ \mu\text{s}$) assume an aggregated ~850 GB/s memory subsystem roofline. On the 4-core VM, reaching host memory saturation (~40 GB/s) confirms the algorithm is optimal!
> 4. **Coordinator-Worker Decoupling & Zero-Allocation Freestanding Execution**:
>    - The Master Coordinator (`DaePipeline::Execute`) evaluates `TilingConfig` on the master CPU thread before the OpenMP region and manages a pre-allocated 64-byte aligned reduction workspace buffer (`float* workspace`) passed to worker threads.
>    - Worker threads must be purely freestanding: **zero dynamic allocation** (`malloc`, `new`, `std::vector`), zero exception unwinding (`throw`), and pass `LocalTensor` by value.

---

### 🛡️ DAE Stream Pipeline Runtime Engine (`include/dsa_runtime.hpp`)

To bridge the gap between high-level C++ and the target decoupled access-execute (DAE) processor, we provide an authentic **C++ Hardware Simulation Model** in [`include/dsa_runtime.hpp`](include/dsa_runtime.hpp):

- **DAE Pipeline Semantics**: Direct mapping for `TPipe`, `TQue<QuePosition, depth>`, `LocalTensor<T>`, `DataCopy`, `Add`, `Mul`, `BlockReduceSum`, and `SyncAll<true>()`.
- **Integrated Hardware Sanitizer Traps**:
  - **Scratchpad Budget Guard**: Instantly aborts if total allocated scratchpad exceeds **191 KB (195,584 bytes)**.
  - **DMA Alignment Guard**: Instantly aborts if DMA transfers are not aligned to **32 bytes**.
  - **Queue Hazard Guard**: Validates queue depth and double-buffering lifecycle.
  - **Egress Channel Guard** (DAE v1.4, Trap #401): a `DataCopy`/`DataCopyPad` to system memory must read a `QuePosition::VECOUT` buffer, never a VECIN one.
  - **ALU Aliasing Guard** (DAE v1.4, Trap #402): `BlockReduceSum` must not fold a buffer onto itself.
  - **DAE v1.5 guards**:
    - `ReduceSum` reduces whole 64-lane repeats only (Trap #408), and its destination, source and workpad must be disjoint (Trap #402).
    - A `Brcb` destination must hold whole 64-lane bursts (Trap #410).
    - `ValidateLaunchArgs` rejects a launch structure larger than the 32-byte constant frame (Trap #409).
- **Queue sequencer** (DAE v1.5): every `TQue` lifecycle step (`AllocTensor`, `EnQue`, `DeQue`, `FreeTensor`) costs 625 cycles. They count in the core's total cycles but not on its timeline. `LocalMemAllocator` hands out static scratchpad buffers with no queue and no such cost.
- **Hardware Virtual Cycle Tracker**:
  - Automatically profiles instruction costs: `Add`(2 cycles/repeat), `Mul`(2 cycles/repeat), `BlockReduceSum`(1 cycle/repeat), `WholeReduceSum`(14 cycles/repeat).
  - Run verification via `ctest -R dsa_runtime_sanitizer` or `./build/test_dsa_runtime`.
- **Timeline model** (`dsa::CoreTimeline`, reported by `./hpc_vector_norm_bench --timeline`). Every primitive also advances a per-core timeline, so the runtime knows *when* each operation runs, not only what it costs:
  - Three in-order units per core. **VECTOR** runs every vector instruction for its cycle cost. **DMA** is one system-memory channel shared by loads and stores: 850 GB/s over 40 cores is 21.25 GB/s, 14.2 B/cycle at the assumed 1.5 GHz, and each transfer's data lands 800 ns (1200 cycles) after it has streamed, so back-to-back transfers overlap their latency. **LOCAL** runs scratchpad-to-scratchpad copies at 256 B/cycle.
  - A 32-byte block scoreboard orders the units: a block is read once its last write has landed, and rewritten once its last read has ended.
  - `SyncAll`: a core arrives when its vector and local units are idle and its stores have landed; every core leaves 7500 cycles after the last arrival.
  - `TimelineSummary` splits the finish time of a core into busy time plus **fill**, **drain**, **mismatch** and **barrier** idle time of its critical unit, and computes a **latency floor** (below).

---

## 📊 Benchmark Suite (15 Test Profiles)

The benchmark harness tests across 15 diverse real-world tensor profiles ($M \times D$). The current baseline and target world-record latency limits are:

| Profile | Dimensions ($M \times D$) | Data Type | Payload Size | Baseline Latency | World Record Target | Status / Focus Area |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **P01** | $1 \times 64$ | FP16 | 256 B | 2.22 $\mu$s | **1.47 $\mu$s** | Launch-bound single thread bypass |
| **P02** | $7 \times 200$ | FP32 | 11.2 KB | 3.76 $\mu$s | **2.06 $\mu$s** | 🏆 Already optimal |
| **P03** | $128 \times 256$ | FP32 | 262 KB | 3.26 $\mu$s | **2.54 $\mu$s** | Balanced small block |
| **P04** | $768 \times 192$ | FP16 | 589 KB | 9.75 $\mu$s | **3.23 $\mu$s** | ⚠️ **Pain Point**: Pipeline startup bubble |
| **P05** | $8 \times 32768$ | FP16 | 1.05 MB | 9.80 $\mu$s | **5.39 $\mu$s** | 🔥 **Boss Challenge**: 32/40-thread Split-D & resident local buffer |
| **P06** | $1536 \times 576$ | FP16 | 3.54 MB | 14.87 $\mu$s | **8.58 $\mu$s** | 🏆 Highly overlapping pipeline |
| **P07** | $10240 \times 400$ | FP16 | 16.4 MB | 19.36 $\mu$s | **16.10 $\mu$s** | Non-power-of-2 vector folding |
| **P08** | $10240 \times 512$ | FP16 | 21.0 MB | 39.43 $\mu$s | **17.57 $\mu$s** | ⚠️ **Pain Point**: Compress scratchpad state to fit 24-row batch |
| **P09** | $4096 \times 1536$ | FP32 | 50.3 MB | 69.78 $\mu$s | **41.70 $\mu$s** | 🏆 FP32 double-buffer stream |
| **P10** | $8192 \times 1024$ | FP16 | 33.6 MB | 59.22 $\mu$s | **47.34 $\mu$s** | 4-stage binary reduction |
| **P11** | $4096 \times 3072$ | FP32 | 100.7 MB | 170.83 $\mu$s | **132.31 $\mu$s** | Bus saturation state |
| **P12** | $4096 \times 4096$ | BF16 | 67.1 MB | 95.64 $\mu$s | **48.62 $\mu$s** | ⚠️ **Pain Point**: 32MB L2/L3 cache-conscious tiling |
| **P13** | $10240 \times 3072$ | FP16 | 125.8 MB | 582.98 $\mu$s | **223.00 $\mu$s** | 🏆 Large-scale throughput |
| **P14** | $2\text{M} \times 128$ | FP16 | 1.05 GB | 4059.64 $\mu$s | **1322.97 $\mu$s** | 🏆 Ultra-long streaming |
| **P15** | $115\text{K} \times 8192$ | FP16 | 3.77 GB | 9537.56 $\mu$s | **6081.74 $\mu$s** | 🏆 Saturated memory bandwidth |

---

## ⚡ Quick Start: Build & Benchmark

```bash
# 1. Clone repository
git clone https://github.com/your-username/hpc_vector_norm.git
cd hpc_vector_norm

# 2. Build with GCC/Clang with OpenMP and AVX-2/AVX-512
#    (-march=native by default; -DHPC_NATIVE_ARCH=OFF builds a portable AVX2 + FMA + F16C binary)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

# 3. Tests (3 suites): host correctness sweep (every dtype x plan x team size vs. an FP64 reference),
#    dsa_runtime sanitizer traps, and the 40-core target plans executed on the DAE simulation
ctest --output-on-failure

# 4. Run full 15-profile benchmark suite (FP16/BF16 profiles use real 16-bit storage)
./hpc_vector_norm_bench            # host latency, then the 40-core target plan of every profile
./hpc_vector_norm_bench --fp32     # every profile in FP32, the workload of the original benchmark
./hpc_vector_norm_bench --target   # also run each target plan on the DAE simulation (sanitizer on, output checked)
./hpc_vector_norm_bench --timeline # --target plus the pipeline timeline of each profile: bound, fill, drain, mismatch, bubble ratio, floor
```

The public entry point is `hpc::FusedResidualNormalize(x1, x2, gamma, bias, y, rows, cols, dtype, eps)`
(`include/hpc_vector_norm.hpp`). FP16/BF16 tensors are raw `uint16_t` storage; `gamma`/`bias` may be null.
The target executor is `hpc::DaePipeline<Codec>::Execute(..., plan)` (`src/kernel_unified.hpp`), with
`plan = AdaptiveTiler::Plan(M, D, elemBytes, HardwareModel::Target())`. `Execute(..., plan, workspace, bytes)` takes a
caller-owned, 64-byte-aligned reduction workspace of `DaePipeline<Codec>::WorkspaceBytes(plan, M)` bytes and allocates nothing.

---

## 🧠 Implementation: One Planning Model, Two Executors

`AdaptiveTiler` (`src/adaptive_tiler.hpp`) is one set of closed-form equations over a `HardwareModel`. The 40-core target is
the model's primary instance. The CI host runs the same equations with its own core count and measured costs, so the planner
has no shape special cases and nothing specific to 4 cores:

| `HardwareModel` field | `Target()`: deployment | `Host(P)`: CI testbed |
| :--- | :--- | :--- |
| `cores` | 40 (`dsa::MAX_HARDWARE_CORES`) | `min(P, 40)`, `P = omp_get_max_threads()` (1 inside a parallel region) |
| `quantumBytes` | 32 (`dsa::DMA_ALIGN_BYTES`) | 32 |
| `spmBytes` | 195,584 (`dsa::SCRATCHPAD_SAFE_WATERLINE`) | 195,584 (per-thread resident-Z scratchpad) |
| `launchNs` (fork/join) | 0 | 3500, measured |
| Vector work | The runtime's cycle costs, instruction by instruction (`DaeIsa`) | `elemNs` = 0.3 ns per element, measured |
| `clockGHz` | 1.5 (`dsa::CLOCK_GHZ`), assumed: converts cycles to time | — |
| `syncNs` (one all-core barrier) | `SyncAll`'s 7500 cycles / clock = 5000 | 1000, measured |
| `byteNs` (streaming, per byte and core) | 40 / 850: ~850 GB/s shared by 40 cores (`dsa::DMA_BYTES_PER_CYCLE`) | 0: cache-resident rows are compute-bound |
| `latencyNs` (a DMA transfer's data lands this long after it streamed) | 800 (`dsa::DMA_LATENCY_CYCLES` = 1200 cycles), inferred from the P01–P03 targets | 0: hardware prefetchers, no explicit DMA |

On the target, the planner minimizes the slowest core's finish time on the runtime's own timeline: it replays each candidate
schedule in the kernel's issue order under the timeline rules above (see **2.** below). It counts vector cycles instruction by
instruction with the runtime's costs (`DaeIsa`):

| Instruction | Cycles |
| :--- | :--- |
| `Add`, `Adds`, `Mul`, `Muls`, `Cast` (strided multi-row `Add`/`Mul` too) | 2 per 256-byte repeat + 13 |
| `BlockReduceSum` | 1 per repeat + 14 |
| `VectorReduceSum`, `ReduceSum` | 2 per repeat + 15 |
| `Rsqrt` | 2 per repeat + 14 |
| `Brcb` | 1 per 64-lane repeat + 8 |
| `Duplicate` | 1 per repeat + 18 |
| `VectorInvRms` | 16 |
| `SyncAll` | 7500 |
| `TQue` lifecycle step | 625, on the queue sequencer (not a vector cycle, not on the timeline) |

`tests/test_dae_pipeline.cpp` requires the planner's cycle count to equal the runtime's on every plan it executes, and its
modeled finish time to equal the runtime timeline's (γ and β present, aligned tensors), in every mode. Both hold.
The clock sets how many vector cycles a DMA byte and the DMA latency are worth. For each of the 15 profiles, every clock from 1.0 to
4.0 GHz picks the same decomposition; the tile schedule is re-tuned for each clock.
[`docs/ARCHITECTURE_CHALLENGES.md`](docs/ARCHITECTURE_CHALLENGES.md#-cost-model--methodology) derives every constant. The clock and
`latencyNs` should be recalibrated on physical target hardware; the models stay the same.

**1. Decomposition, balanced to one DMA block (Challenge 1).** The planner treats the `M·D` elements as one flattened stream and
cuts it into units. Core `t` of `n` gets units `[⌊U·t/n⌋, ⌊U·(t+1)/n⌋)`, so any two cores differ by at most one unit for every
`M`, `D` and core count. There are three kinds of unit:

- `ROW_PARALLEL` uses `p = 32 / gcd(32, D·s)` whole rows. `p = 1` whenever a row is a whole number of 32-byte blocks. Cores never share a row, so they never communicate.
- `SPLIT_D` uses one 32-byte DMA block (`q = 32 / s` elements) in row-major order. A core shares at most two rows with its neighbours, at the cost of one `SyncAll`. It reads γ/β once per element it owns.
- `SPLIT_COLUMNS` (target only) numbers the same 32-byte blocks column-major: block `j` of row `i` is unit `j·M + i`. The balance is the same, but each core's share becomes a column band of every row, so it reads γ/β for that band once instead of once per element. It needs rows on the 32-byte grid (`D·s % 32 = 0`), and the band's FP32 Z must fit in the scratchpad.

A balanced partition gives its busiest core `⌈U/P⌉` units, the minimum for that unit size, so no split along 32-byte blocks does
better. For P05 (8 × 32768 FP16), 16384 blocks go to 40 cores: 24 cores take 410 blocks (6560 elements) and 16 take 409
(6544 elements). An exactly equal split does not exist, because 16384 / 40 = 409.6.

Both Split-D numberings meet the challenge's objective of 32-byte-aligned slices within 32 bytes of each other. Row-major lands
every row on exactly 5 cores, with boundaries `{0, 6544, 13104, 19648, 26208, 32768}`. The planner picks the column band: each
core owns 51 or 52 blocks (816–832 columns) of all 8 rows. That cuts a core's γ/β traffic from 26 KB to 3.3 KB and its system
traffic from 66 KB to 44 KB, with one `SyncAll` either way.

**2. Pipeline schedule (Challenge 3, pipeline bubbles).** Each core streams its share in tiles: DMA in, vector, DMA out. Three
timeline models (`RowTilesTimeline`, `BandTimeline`, `ColumnTimeline`) replay a core's tiles in the kernel's exact issue order
under the runtime's timeline rules, and the planner keeps the schedule whose slowest core finishes first. For row tiles it
searches:

- `depth`: 2–4 tiles in flight per input queue. Deeper queues hide the 800 ns latency, at the price of smaller tiles in the same 191 KB.
- `B`: body tile rows, up to the largest tile the layout admits (Challenge 2), and `rep`, the replicated γ/β rows.
- `head`, `tail`: a small first tile starts the vector unit sooner (prologue fill), and a small last tile lands the final store sooner (epilogue drain).
- `outDepth`: 1–2 egress buffers (VECOUT, DAE v1.4): a second one lets a tile's result stream out while the next is written.
- Issue order: γ/β before or after tile 0 (`paramsFirst`), and X1/X2 of a later tile as soon as the tile's Z is built (`earlyLoads`) rather than after its store.

Column bands search the band tile rows, `rep` and `depth`; column tiles search the tile size and the resident Z. The search
stays fast: uniform body tiles make the replay a max-plus linear recurrence, so it jumps over whole periods, and a lower bound
cuts each candidate short once it cannot win. A full-size profile plans in 0.15–3.5 ms (14 ms for P15, whose cores run 2,875
one-row tiles each). P04's 20 rows per core now run as tiles of 4 + 9 + 6 + 1 rows, three in flight, with γ/β issued first:
3.32 µs, against 3.62 µs for the previous 2 × 10 rows.

**3. Scratchpad knapsack (Challenge 2).** Every candidate the schedule search tries must fit the 191 KB layout.
The layout (`DaeLayout`) lists exactly the buffers the kernel claims through `TPipe`. The tests require the claim to equal the
plan byte for byte, so the runtime's 191 KB trap checks the planner's arithmetic.

- **Row tiles.** X1 and X2 each hold `depth` tiles in the native dtype (`2·depth·s` B per element), and Y leaves from `outDepth` egress buffers (`outDepth·s`). For 16-bit dtypes, one FP32 Z tile adds 4 B per element; FP32 builds Z in its egress buffer. So a double-buffered row tile with one egress buffer costs `b = 14` B per element for FP16/BF16 and 20 B for FP32, and each further level of depth adds `2s`.
  - γ/β stay resident in FP32, replicated into `rep` rows (`2·Align32(4·rep·D)` B, at most 16 KB). Bias and γ then take one instruction per `rep` rows instead of one per row.
  - The 8 KB scratch chunk holds the widened X2, the squares, and the staged γ/β; an 8 → 1 fold of the squares lands in a 1 KB partition after it.
  - So `B*(D) = ⌊(195,584 − 9,216 − 2·Align32(4·rep·D)) / (b·D)⌋` rows: 25 rows at `D = 512` without replication, 23 with `rep = 4` (double-buffered, one egress buffer).
  - The search tries both options. Replication wins wherever spare scratchpad pays for it; P14 spends 10 of its 121 possible rows on it to cut 22% of its vector cycles.
- **Column band** (`SPLIT_COLUMNS`). Tiles hold `k` band rows at the band's pitch. The band's FP32 Z for all `M` rows stays resident across the barrier, next to the replicated γ/β band and the partial-sum records.
- **Column tiles** (row-major Split-D fragments, and rows too long for a row tile). X1, X2, a γ/β chunk and the egress buffer are double-buffered, plus FP32 Z: `8s + 4` B per element. The segment's FP32 Z stays resident when it fits, so the normalize sweep reads nothing from main memory a second time.

- **Direct kernel** (Challenge 8). A share of at most 512 B per tensor runs from fixed static buffers: 13,056 B for 16-bit data, 8,448 B for FP32.

The full-size target plans are below; `./hpc_vector_norm_bench` prints them. Tiles lists the busiest core's tiles in rows
(head + body + tail). Model cycles are the slowest core's vector cycles (the barrier excluded), and model µs is its finish time on
the timeline at 1.5 GHz. `tests/test_dae_pipeline.cpp` checks that every plan obeys the rules below: these 15, 2,448 other shapes
(each also forced into every mode), and every tensor of at most 512 bytes. Every plan:

- uses at most 40 cores and at most 195,584 B per core;
- balances the cores to within one unit;
- cuts units and tiles on 32-byte blocks;
- uses `min(40, U)` cores.

| Profile | Mode | Cores | Unit | Busiest core (elements) | Tiles (rows) | Depth | Out | `rep` | SPM / core | Model cycles | Model µs |
| :--- | :---: | ---: | :---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| P01 1×64 FP16 | **direct** | 1 | 1 row | 64 | 1 | — | 1 | — | 13,056 B | 297 | 1.81 |
| P02 7×200 FP32 | rows | 7 | 1 row | 200 | 1 | 2 | 1 | 1 | 14,816 B | 144 | 1.87 |
| P03 128×256 FP32 | rows | 40 | 1 row | 1,024 (min 768) | 4 × 1 | 4 | 2 | 1 | 21,504 B | 576 | 2.23 |
| P04 768×192 FP16 | rows | 40 | 1 row | 3,840 (min 3,648) | 4 + 9 + 6 + 1 | 3 | 1 | 9 | 54,144 B | 2,362 | 3.32 |
| P05 8×32768 FP16 | **band** | 40 | 32 B | 6,560 (min 6,544) | 4 × 2 (× 832 columns) | 4 | 2 | 2 | 83,744 B | 2,917 + 7,500 | 10.83 |
| P06 1536×576 FP16 | rows | 40 | 1 row | 22,464 (min 21,888) | 2 + 12 × 3 + 1 | 4 | 1 | 3 | 61,056 B | 9,358 | 8.14 |
| P07 10240×400 FP16 | rows | 40 | 1 row | 102,400 | 4 + 24 × 10 + 8 + 4 | 3 | 1 | 5 | 97,216 B | 44,579 | 31.99 |
| P08 10240×512 FP16 | rows | 40 | 1 row | 131,072 | 1 + 20 × 12 + 11 + 4 | 4 | 2 | 4 | 173,056 B | 52,375 | 38.03 |
| P09 4096×1536 FP32 | rows | 40 | 1 row | 158,208 (min 156,672) | 34 × 3 + 1 | 3 | 2 | 1 | 168,960 B | 38,256 | 90.72 |
| P10 8192×1024 FP16 | rows | 40 | 1 row | 209,920 (min 208,896) | 28 × 7 + 5 + 4 | 4 | 2 | 1 | 189,440 B | 77,943 | 60.26 |
| P11 4096×3072 FP32 | rows | 40 | 1 row | 316,416 (min 313,344) | 1 + 51 × 2 | 2 | 2 | 1 | 181,248 B | 70,407 | 180.64 |
| P12 4096×4096 BF16 | rows | 40 | 1 row | 421,888 (min 417,792) | 51 × 2 + 1 | 2 | 2 | 1 | 173,056 B | 135,019 | 120.85 |
| P13 10240×3072 FP16 | rows | 40 | 1 row | 786,432 | 1 + 127 × 2 + 1 | 3 | 2 | 1 | 156,672 B | 260,622 | 223.43 |
| P14 2M×128 FP16 | rows | 40 | 1 row | 6,710,912 (min 6,710,784) | 8 + 557 × 94 + 55 + 8 | 2 | 1 | 16 | 194,048 B | 4,426,534 | 2,954.6 |
| P15 115K×8192 FP16 | rows | 40 | 1 row | 23,552,000 | 2,875 × 1 | 2 | 1 | 1 | 189,440 B | 7,363,413 | 6,653.4 |

Out: egress buffers (VECOUT). Direct: the core's rows run in one shot from static buffers, with no queue (Challenge 8). Issue
order: P04 and P06 load γ/β before tile 0; P11, P12, P13 and P15 refill X1/X2 as soon as a tile's Z is built. The model µs of the
benchmark-sized P14 (50,000 rows) and P15 (5,000 rows) are 72.90 and 292.59.

P05 is the only profile that splits. The model puts its column band at 10.83 µs, against 11.40 µs for the row-major split and
20.11 µs for whole rows. Everywhere else, whole rows already balance to within one row, and one row costs less than a `SyncAll`.

**DAE executor** (`DaePipeline<Codec>`, the target path at the end of `src/kernel_unified.hpp`). Each OpenMP thread is one
simulated core (`GetCoreIdx`), and everything goes through `include/dsa_runtime.hpp`:

- **No scalar stalls.** The scalar unit never reads the scratchpad (`GetValue`, a 500-cycle V→S stall). Row sums reach it through `VectorReduceSum`, and Split-D partials are combined by vector adds. Every row costs one `VectorReduceSum`, one `VectorInvRms` and one `Muls`. The `VectorReduceSum` is preceded by one 8 → 1 `BlockReduceSum` fold over a whole chunk of rows when the cost model says that is cheaper. The kernel never uses `WholeReduceSum`.
- **Tile-wide instructions.** Widening X1, widening and adding X2, the squares and the narrowing each cover a whole tile or an 8 KB chunk. Bias and γ cover `rep` replicated rows per instruction.
- **Prologue overlap (Challenge 3).** Row tiles issue the loads of their first `depth` tiles and of γ/β (before tile 0 or right after it, as planned) before the first vector instruction, and only then widen γ/β. 16-bit γ/β are staged in the Z tile, which is free until tile 0, so both arrive in one piece each. Replicating γ/β is scratchpad-to-scratchpad DMA, with no vector cycles. From then on, both input buffers of a tile refill right after its Z is built when the plan says `earlyLoads`, else after its store. The column band's sweep 2 narrows each tile into the next egress buffer, so narrowing never waits for the previous tile's stores to stream out.
- **DAE v1.4 errata (Challenge 7).**
  - Every result and every published record leaves from a VECOUT buffer (`qY`, `bRec`), never from an input queue: the egress DMA channel reads VECOUT only.
  - FP32 row tiles build Z in their egress buffer; 16-bit tiles narrow into it. Results rotate through the `outDepth` egress buffers, because a worker keeps each one until the next result has taken its own.
  - An 8 → 1 fold writes a 1 KB partition after the chunk it reads, never the chunk itself.
  - The coordinator passes `invD = 1/D` in `Args`, so no row pays the worker's 20-cycle integer-to-float stall.
  - The worker uses `dsa::Min`/`dsa::Max`, not `<algorithm>`.
  - All 15 profiles run under the v1.4 guards with 0 traps and 0 scalar stalls. Their outputs are bit-identical to the pre-errata kernel on 11 profiles; on the other four, 204 of 133.5 M FP16 values differ by one ulp.
- **DAE v1.5 (Challenge 8).**
  - **Direct kernel.** A `TQue` lifecycle step costs the queue sequencer 625 cycles, and a row tile takes ten: X1 and X2 four each, the egress buffer two. A core whose share fits the direct kernel is a single tile with nothing to overlap them with, so it runs on `DirectCore` instead. A share fits when its rows take at most 512 B per tensor and their squares, each row padded to whole 64-lane repeats, fit 1,024 floats (`AdaptiveTiler::DirectFits`, physical properties only). Every tensor of at most 512 B fits. `DirectCore` claims static `LocalMemAllocator<Hardware::Scratchpad>` buffers on the worker's stack: no `TPipe`, no `TQue`, no heap. Its inputs are tagged VECIN and its result VECOUT, so the v1.4 egress guard still covers it.
  - **Scoreboard tokens, no flushes.** Each load sets its own `MTE2_V` flag on a literal event ID (`EVENT_ID0`–`EVENT_ID3`: without a `TPipe` there is none to fetch). The vector unit waits on each flag just before that input's first use, so X1 is widened while X2, β and γ are still landing. `V_MTE3` hands the result to the egress channel, and `MTE3_S` ends the kernel. The queue kernel's stores take the same `V_MTE3` token, since its egress buffers are never enqueued. No kernel issues `PIPE_ALL`.
  - **64-lane reductions.** Each row's squares go to a row of whole 64-lane repeats whose pad lanes are zeroed first. The zeroing has no operand to wait for, so it runs under the DMA latency. One `ReduceSum` per row then writes that row's lane of the row-sum partition. Destination, source and workpad never overlap. The mean, inverse RMS, Newton-Raphson term and `Brcb` destination each have a partition of their own.
  - **Inverse RMS without the scalar unit.** The kernel runs `Muls` by the coordinator's `invD`, `Adds` ε, then `Rsqrt`. v1.5's `Rsqrt` is a table of about 11 bits, so Newton-Raphson steps follow until the bits exceed the output's significand: one step for 16-bit outputs, two for FP32. `Brcb` then broadcasts each row's value across its repeats for the scaling `Mul`, so the scalar unit never reads the scratchpad.
  - **Strided rows.** β and γ apply to every row in one strided `Add`/`Mul` when FP32 rows are whole 32-byte blocks, because the repeat stride counts blocks. Otherwise they take one instruction per row.
  - **Flat launch.** The coordinator launches the kernel with 64-bit addresses and 32-bit scalars only; the plan travels as the address of its tiling data. A `static_assert` rejects any other argument at compile time, and `ValidateLaunchArgs` checks each argument at run time (Trap #409).
  - **Descriptors and converters.** Padded transfers carry a `DataCopyExtParams` descriptor. Widening and narrowing stay on the runtime's `Cast` with the codec's exact converters, at the same cycle cost. The simulator's `dsa::half` converts all 2,046 FP16 subnormals to garbage, and its `FromFloat` truncates instead of rounding.
  - **P01:** 1.81 µs with no queue step. The queue kernel took 1.73 µs on the timeline plus 6,250 sequencer cycles, 5.90 µs end to end. Its total cycle count goes from 6,433 to 297.
- **Column band.**
  - Each core loads its band one whole-block DMA per row and keeps the band's Z resident.
  - After sweep 1 it publishes one record of `M` partials in 32-byte blocks and passes the single `SyncAll`.
  - It then fetches all 40 records in one DMA and sums them with a fixed tree of 6 vector adds. Every core runs the same tree, so all owners of a row compute the same σ.
  - Sweep 2 normalizes the resident Z.
- **Row-major Split-D.**
  - Each core publishes `{Σ₀, 0 ×7}{Σ₁, 0 ×7}` in two 32-byte records.
  - The owners of a row fetch the contiguous record range that holds the row's partials, with zeros in between, and reduce it with one `VectorReduceSum`.
  - The first γ chunk of sweep 2 is already in flight during the `SyncAll`.
- **Coordinator and freestanding workers (Challenge 6).**
  - `DaePipeline::Execute` runs on the calling thread. It checks the plan, owns the reduction workspace (caller-provided, or its own buffer reused across calls) and launches every core with a flat frame of addresses and scalars (Challenge 8).
  - Each core runs `Core::Execute` with no heap allocation and no exceptions. Row sums and band columns sit in fixed 128-row arrays on the worker's stack; longer tiles run in groups of 128 rows, and the cycle model counts the groups.
  - Invariants are `DSA_ASSERT`s, and `LocalTensor` views are passed by value.
  - A trap aborts the whole process, so no core is ever left waiting at the `SyncAll`.
  - A heap probe in the tests confirms that `Execute` allocates nothing beyond the simulator's scratchpad buffers.
- **Exact scratchpad claim.** The kernel claims only the layout's buffers. The previous merge added a 64-byte placeholder Z buffer when a plan has none, which pushed plans that fill the scratchpad to the byte over the waterline. 100,000 × 128 FP32 trapped at 195,648 B, for example. A regression test now runs such a plan.
- **DMA.** Every transfer is a 32-byte `DataCopy`. `DataCopyPad` is used only where a transfer does not end on a 32-byte block (`D·s % 32 ≠ 0`) or starts off the 32-byte grid.
- **Results with `--target`.** On all 15 profiles the output matches the host kernel, with 0 hardware traps, 0 scalar stalls and 0 padded transfers. Every core claims exactly its planned scratchpad (at most 191 KB). The runtime's cycle count and its timeline's finish time both equal the model's. P01 takes no queue step.

Busiest-core vector cycles on the DAE runtime: before and after the stall removal of the previous rounds, and now, with the
timeline-scheduled tiles. These are the benchmark's sizes (P14 is 50,000 rows and P15 is 5,000), so they differ from the
full-size model above:

| Profile | Stalling kernel: total (scalar stalls) | Stall-free | Now | Stalling → now |
| :--- | ---: | ---: | ---: | ---: |
| P01 1×64 | 713 (500) | 183 | 183 | −74% |
| P02 7×200 | 698 (500) | 144 | 144 | −79% |
| P03 128×256 | 2,562 (2,000) | 420 | 576 | −78% |
| **P04 768×192** | 12,692 (10,000) | 2,180 | **2,362** | **−81%** |
| **P05 8×32768** | 12,859 (2,500; + 7,500 barrier) | 10,365 (+ barrier) | **10,417** (+ barrier) | **−19%** |
| P06 1536×576 | 29,422 (19,500) | 9,085 | 9,358 | −68% |
| P07 10240×400 | 178,028 (128,000) | 44,114 | 44,396 | −75% |
| **P08 10240×512** | 187,480 (128,000) | 52,341 | **52,591** | **−72%** |
| P09 4096×1536 | 92,502 (51,500) | 37,996 | 38,256 | −59% |
| P10 8192×1024 | 182,356 (102,500) | 76,867 | 75,655 | −59% |
| P11 4096×3072 | 127,878 (51,500) | 70,186 | 70,186 | −45% |
| P12 4096×4096 | 191,639 (51,500) | 134,577 | 135,019 | −30% |
| P13 10240×3072 | 401,754 (128,000) | 258,906 | 260,622 | −35% |
| P14 50000×128 | 763,967 (625,000) | 106,222 | 106,300 | −86% |
| P15 5000×8192 | 396,663 (62,500) | 320,689 | 320,663 | −19% |

Smaller tiles cost vector cycles, because every tile repeats seven tile-wide instructions: P03 +37%, P04 +8%, P06 +3%, and under
1% elsewhere (P10 −1.6%: it now replicates γ/β). The timeline below shows what that buys: no profile finishes later, and 13
finish sooner.

What remains on P05 is 72% barrier. `SyncAll` is the only cross-core primitive, and Split-D needs exactly one. A plan without a
barrier would have to read whole rows: 32,768 elements on each of 8 cores, or every core's full row redundantly. Either moves
three times the column band's DMA or more.

### Pipeline bubbles: timeline telemetry and the optimization loop

`./hpc_vector_norm_bench --timeline` executes every target plan and reports the timeline of the core that finishes last:

- **Compute**, the vector unit's busy time, and **Stream**, the DMA channel's. The busier of the two is the critical unit (**Crit**).
- **Bound** `= max(compute, stream) + sync`: the time with every bubble removed.
- **Bubble** `= total − bound`, split into the critical unit's idle time: **fill** (before its first operation), **drain** (after its last), **mismatch** (in between) and **wait** (inside a `SyncAll`, beyond its own 7500 cycles). The **ratio** is bubble / total.
- **Floor** (`TimelineSummary::LatencyFloor`): the same program replayed on a relaxed core with unlimited buffers, so nothing waits for a buffer's previous use, and with loads and stores in separate queues, so a store waiting for its data never holds up a load. It is also at least the channel's total occupancy plus one latency. Every operation starts no later than on the real core, so no run can finish sooner: `tests/test_dae_pipeline.cpp` checks this on every run. **Excess** = total − floor is what buffer reuse, the shared channel's order and `SyncAll` waits cost. The floor itself is DMA latency and data dependencies.

The loop: measure the timeline, find where the critical unit idles, change the schedule or the model, and repeat.

1. **Baseline** (the previous round's plans): 1,176.3 µs over the 15 profiles, mean bubble ratio 29.8%. The telemetry showed three kinds of bubble:
   - DMA-bound cores idled mid-stream (mismatch 3.5–7.8 µs on P08, P10, P12, P13 and P15). With two buffers per queue, the load of tile k+2 must wait for tile k's result to stream out of the X1 slot, and the next load queues behind that store.
   - Vector-bound cores waited for their first large tile (fill 3.0 µs on P07, 3.2 µs on P14) and for their last store (drain 1.7–1.8 µs).
   - The planner's closed-form pipeline cost (`(n + 2)(W/n + tileNs)`) could not see either effect, and it mispredicted the timeline by up to 58%.
2. **Iteration 1: an exact row-tile model and a schedule search.** `RowTilesTimeline` replays the kernel's issue order under the runtime's rules and equals the runtime's timeline to 6e-15 relative error on 372 plans. The kernel gained queue depths 2–4, head and tail tiles, γ/β issued first or second, early X2 prefetch, and γ/β staged in the Z tile. Result: 1,140.4 µs, 25.3%.
3. **Iteration 2: exact band and column models.** `BandTimeline` and `ColumnTimeline` (0 mismatches on 119 and 72 plans) let the band search its depth and tile rows, and let `Plan()` compare decompositions by true finish time. The band's sweep 2 narrows each tile into a fresh X1 buffer. P05: 11.26 → 10.83 µs. Result: 1,140.0 µs, 25.2%.
4. **Iteration 3: traces of what was left.** On P06 and P07 the vector unit runs without a gap from its first instruction on; P15 is limited by its scratchpad. Deriving the planner's constants from the runtime's changed no plan. The gains went from 36 µs to 0.4 µs to nothing: diminishing returns, so the loop stopped.

Before (baseline) and after, at the benchmark's sizes. The timeline's cycles are those of the core that finishes last:

| Profile | Crit | Total µs | Timeline cycles | Bubble ratio | Fill / drain / mismatch µs (after) | Excess µs |
| :--- | :---: | ---: | ---: | ---: | :---: | ---: |
| P01 1×64 | VEC | 1.75 → **1.73** | 2,619 → 2,601 | 93.0% → **93.0%** | 0.81 / 0.81 / 0.00 | 0.00 → 0.00 |
| P02 7×200 | DMA | 1.87 → **1.87** | 2,805 → 2,805 | 89.9% → **89.9%** | 0.00 / 0.80 / 0.88 | 0.00 → 0.00 |
| P03 128×256 | DMA | 2.53 → **2.23** | 3,799 → 3,339 | 73.4% → **69.7%** | 0.00 / 0.80 / 0.75 | 0.00 → 0.00 |
| P04 768×192 | VEC | 3.62 → **3.32** | 5,428 → 4,985 | 59.8% → **52.6%** | 0.82 / 0.82 / 0.11 | 0.00 → 0.00 |
| P05 8×32768 | DMA | 11.26 → **10.83** | 16,892 → 16,249 | 37.2% → **34.7%** | 0.00 / 0.80 / 2.96 | 0.03 → 0.00 |
| P06 1536×576 | DMA | 10.07 → **8.14** | 15,104 → 12,216 | 35.9% → **20.8%** | 0.00 / 0.80 / 0.89 | 0.75 → 0.00 |
| P07 10240×400 | VEC | 35.81 → **31.96** | 53,710 → 47,944 | 17.9% → **7.4%** | 1.44 / 0.88 / 0.05 | 1.65 → 0.00 |
| P08 10240×512 | DMA | 43.07 → **37.90** | 64,606 → 56,857 | 13.9% → **2.1%** | 0.00 / 0.80 / 0.00 | 2.93 → 0.00 |
| P09 4096×1536 | DMA | 90.73 → **90.72** | 136,091 → 136,079 | 0.9% → **0.9%** | 0.00 / 0.80 / 0.00 | 0.01 → 0.00 |
| P10 8192×1024 | DMA | 63.71 → **60.26** | 95,568 → 90,396 | 6.7% → **1.3%** | 0.00 / 0.80 / 0.00 | 3.45 → 0.00 |
| P11 4096×3072 | DMA | 180.64 → **180.64** | 270,958 → 270,958 | 0.4% → **0.4%** | 0.00 / 0.80 / 0.00 | 0.00 → 0.00 |
| P12 4096×4096 | DMA | 126.30 → **120.85** | 189,454 → 181,280 | 5.1% → **0.8%** | 0.00 / 0.80 / 0.16 | 5.61 → 0.16 |
| P13 10240×3072 | DMA | 229.94 → **223.43** | 344,915 → 335,144 | 3.2% → **0.4%** | 0.00 / 0.80 / 0.00 | 6.51 → 0.00 |
| P14 50000×128 | VEC | 75.80 → **72.90** | 113,701 → 109,348 | 6.6% → **2.8%** | 1.00 / 0.85 / 0.18 | 0.00 → 0.00 |
| P15 5000×8192 | DMA | 299.22 → **293.22** | 448,837 → 439,834 | 2.9% → **0.9%** | 0.00 / 0.80 / 1.75 | 7.75 → 1.75 |
| **All 15** | | **1,176.3 → 1,140.0 (−3.1%)** | | **29.8% → 25.2%** (mean) | | **28.7 → 1.9** (2.30% → 0.05% mean) |

Summed over the 15 profiles, the bubble fell from 60.6 to 24.0 µs: mismatch 38.3 → 7.7 µs, fill 8.2 → 4.1 µs, drain 14.1 → 12.2 µs.

**Since the DAE v1.4 errata (Challenge 7)** every result leaves through its own egress buffer, which costs scratchpad and frees X1
early. Twelve profiles are unchanged:

- P07 goes from 31.96 to 31.99 µs, with 10-row tiles instead of 15.
- P08 goes from 37.90 to 38.03 µs, with 12-row tiles instead of 15.
- P15 goes from 293.22 to 292.59 µs: X1 now refills before the store.

The 15 profiles sum to 1,139.5 µs, and the mean bubble ratio stays at 25.2%.

**Since DAE v1.5 (Challenge 8)** the runtime counts 625 queue-sequencer cycles per `TQue` lifecycle step, off the timeline. The
benchmark shows them in its Queue columns. P01's single tile paid ten steps: 1.73 µs on the timeline plus 6,250 cycles (4.17 µs),
5.90 µs end to end. The direct kernel takes 1.81 µs and no queue step. Its vector chain is longer, 297 cycles against 183, because
`ReduceSum`, `Rsqrt`, one Newton-Raphson step and `Brcb` replace `VectorReduceSum` and `VectorInvRms`. The other 14 profiles are
unchanged. The 15 sum to 1,139.6 µs, and the mean bubble ratio is 24.9%.

The streaming profiles still take their queue steps, up to 806,250 sequencer cycles on P13's busiest core. The runtime keeps them
off the timeline, so how much of that a multi-tile core hides is not modeled.

**Why the rest stays.** The excess is gone except on P08 (0.13 µs), P12 (0.16 µs) and P15 (1.12 µs). There a finished tile's store
heads the channel's queue while the next loads wait behind it. P15's 191 KB hold only two 8,192-element tiles per input. Everywhere
else the timeline is its program's latency floor, and what remains is DMA latency:

- **A DMA-bound core keeps at least one latency of bubble:** its last store lands 800 ns after the channel's last byte. P09–P11 and P13 are exactly there (0.80 µs).
- **A vector-bound core keeps at least two:** it computes nothing until its first load lands, and its last result lands 800 ns after it. P01 is there (1.61 µs), P04 is 0.15 µs above (3.32 µs against a 3.23 µs world-record target).
- **The small profiles have too little work to hide either latency.** P01–P04 move less data per core than streams in two latencies, which is why their ratios stay above 50%. A P02 core's single row must land, be computed and be stored before the channel has anything else to do (mismatch 0.88 µs).

The candidates left each gain under 1% on the large profiles:

- a geometric ramp of head tiles (1, 2, 4, … rows) for the vector-bound P07 and P14, whose fill is still 1.44 and 1.00 µs;
- β before γ, with γ widened just before its first use, worth up to 0.8 µs of P15's startup;
- a store queue of its own on hardware that has one.

**Host executor** (`KernelUnifiedPipeline<Codec>`, CI). It uses the same partition code (`AdaptiveTiler::Range` and
`RowOwners`) on `P` threads. The host candidates are inline, rows and row-major Split-D, and their plans are identical to the
previous revision's (compared on 14,664 shapes):

| Stage | Decision | Why |
| :--- | :--- | :--- |
| Team size | `P` threads iff `min(t_rows, t_split) < t_inline`, else inline on the caller (no OpenMP region) | Fork/join costs ~3.5 µs here; resizing an OpenMP team between calls costs 70–330 µs, so the team is either 1 or `P`, never in between |
| Work quantum | Rows, or 32-byte blocks with one barrier when rows would leave threads idle | The target's Split-D |
| Row batching | `B = clamp(2048 / D, 1, 64)` rows issue pass 1 before their pass 2 | Hides the reduce → sqrt → reciprocal latency of short rows |
| Resident Z | FP32 Z kept in a per-thread 191 KB scratchpad; longer segments recompute Z from cache-hot inputs | One DRAM read of X1/X2 per element |
| Stores | Non-temporal Y stores + 512 B software prefetch once `X1+X2+Y > LLC` | Saves the read-for-ownership stream; keeps more line fills in flight |
| Traversal | Serpentine at 64 KB chunk granularity (chunk order flips every call) | The chunks touched last are consumed first while still cached |

The host kernel runs two passes per row. Pass 1 computes `Z = X1 + X2 + bias` with 4 independent FMA accumulators, in FP32 within
4096-element blocks and FP64 across blocks. Pass 2 computes `Y = Z · invRms · gamma`.

The ISA layer (`src/simd.hpp`) provides AVX-512 (masked tails), AVX2 + FMA + F16C, and a portable scalar path. FP16/BF16 are widened
on load and rounded to nearest-even on store. For 16-bit tensors, bias/gamma are widened to FP32 once per call when a thread owns at
least 4 rows. Each thread memoizes the plan for the last shape, so repeated calls skip planning, which takes 60–75 ns.

**Runtime fixes** (`include/dsa_runtime.hpp`, regression tests in `tests/test_dsa_runtime.cpp`). Building the DAE pipeline
exposed two defects in the previous revision:

1. **`TQue` double buffering was silently single-buffered.** `AllocTensor` picked slot `(tail + allocatedCount) % depth` and ignored tensors already enqueued. After an `EnQue`, the next `AllocTensor` therefore returned the same buffer, and the prefetch of tile k+1 overwrote tile k before it was dequeued. The queue is now a per-slot state machine (FREE → ALLOCATED → ENQUEUED → DEQUEUED) with FIFO order. It traps allocation beyond the queue depth, freeing a free or in-flight buffer, and tensors that belong to another queue.
2. **`DataCopy` did not check addresses.** It checked only the transfer size, so a whole-block transfer from a misaligned address passed. Misaligned system memory or local addresses now trap.
3. **`LocalMemAllocator` did not compile with clang** (DAE v1.5). Its pool was declared `uint8_t pool[N] alignas(64)`, which places `alignas` after the array declarator. Clang rejects that, so no file including the runtime built. It now reads `alignas(64) uint8_t pool[N]`, with the same layout under GCC.

The runtime also gains:

- `Cast`, charged at the vector conversion cycle cost;
- `DataCopyPad`, which zero-pads a partial block and is counted;
- per-core DMA byte, transfer and pad counters;
- a trap when `InitBuffer` asks for more buffers than the queue depth.

Size, in lines of code (neither blank nor comment): `adaptive_tiler.hpp` is 1,221. That includes the instruction-level cycle
model, the three streaming timeline models (628), the direct kernel's rule, layout and timeline (80) and the schedule search (159).
`kernel_unified.hpp` is 1,002: the host executor is 248 and the DAE executor is 754, 140 of them the direct kernel. The ISA layer
adds 105. The runtime's timeline model is 208 of `dsa_runtime.hpp`'s 1,145.

### Measured results (4-core Cascade Lake VM, AVX-512, ~40 GB/s DRAM)

All values are µs.

- **Original columns.** The original code, which runs every profile in FP32 as the original benchmark did. "cap 4" is the same code with `MAX_THREADS` set to the machine's 4 cores. These columns are carried over from the previous revision.
- **New columns.** Median of 5 runs of this revision.

The host is shared, so single runs can differ by up to ~50%. Interleaved A/B runs against the previous revision agree within that noise
on every profile except P01, which is 21% faster here (0.14 → 0.11 µs).

| Profile | Type | Original (40 thr) | Original (cap 4) | New, same FP32 work | New, profile dtype | Speedup (as shipped → new) |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: |
| P01 1×64 | FP16 | 0.94 | 1.10 | 0.10 | 0.11 | 8.5× |
| P02 7×200 | FP32 | 746 | 257 | 0.52 | 0.48 | 1550× |
| P03 128×256 | FP32 | 798 | 35.9 | 6.28 | 9.41 | 85× |
| P04 768×192 | FP16 | 1088 | 70.6 | 18.8 | 19.7 | 55× |
| P05 8×32768 | FP16 | 1140 | 364 | 33.9 | 27.0 | 42× |
| P06 1536×576 | FP16 | 955 | 184 | 109 | 73.3 | 13× |
| P07 10240×400 | FP16 | 2637 | 1622 | 1007 | 420 | 6.3× |
| P08 10240×512 | FP16 | 3084 | 2272 | 1326 | 637 | 4.8× |
| P09 4096×1536 | FP32 | 3598 | 2397 | 1659 | 1719 | 2.1× |
| P10 8192×1024 | FP16 | 4604 | 3183 | 2407 | 993 | 4.6× |
| P11 4096×3072 | FP32 | 7077 | 5570 | 4233 | 4091 | 1.7× |
| P12 4096×4096 | BF16 | 8067 | 6949 | 5849 | 3437 | 2.3× |
| P13 10240×3072 | FP16 | 15963 | 13908 | 11470 | 5757 | 2.8× |
| P14 50000×128 | FP16 | 3573 | 5088 | 1866 | 880 | 4.1× |
| P15 5000×8192 | FP16 | 18526 | 17497 | 14430 | 8102 | 2.3× |

For FP32 profiles the two "New" columns are the same workload measured in different runs, so their gap is run-to-run noise.

P01 and P02 beat their targets. P09–P15 exceed the LLC and stream from DRAM at 29–51 GB/s effective, against a measured ~40 GB/s
read-2/write-1 roofline. P09, P10 and P14 get some LLC reuse. The world-record targets assume the target's memory system: P13's
223 µs needs ~850 GB/s, which is exactly the busiest-core streaming time of the 40-core plan above (786,432 elements × 6 B at
850/40 GB/s = 222 µs). They are out of reach for a 4-core host. For the scaled-down P14/P15 runs, the `Target (us)` column is
scaled by the benchmarked row count.

---

## 🧩 Architectural Focus: The 5 Open Tasks

The 5 optimization bottlenecks (detailed in [`docs/ARCHITECTURE_CHALLENGES.md`](docs/ARCHITECTURE_CHALLENGES.md), including how each is solved on the 40-core target and on the CPU host):

1. **Task 1: Split-D Reduction for Profile 5 ($8 \times 32768$)**
   - $M=8$ rows is too small for 40 threads (leaves 32 threads idle if row-parallel).
   - Divide columns across 32 or 40 threads ($8 \times 4 = 32$ threads, 8192 elements per thread).
   - Keep intermediate sum $Z$ resident in thread-local scratchpad ($8192 \times 2 = 16\text{ KB} \ll 191\text{ KB}$).
   - Perform lightweight 32-element inter-thread reduction to compute $\sigma$, broadcast, and finish normalization without any secondary main memory reload!
   - **Target**: Break **5.39 $\mu$s**.
   - ✅ **Target**: the tensor is split into 32-byte DMA blocks, balanced to one block for any $M$ and core count (409.6 blocks per core on average, so 409 or 410: no exactly equal split exists). P05 numbers the blocks column-major: each of the 40 cores owns a 51–52-block band of all 8 rows, reads that band of γ/β once (3.3 KB instead of 26 KB), keeps the band's FP32 Z resident and passes one `SyncAll`. All 40 partial records then come back in one DMA and are summed by 6 vector adds, with no scalar stall. Busiest core: 10,417 cycles, 7,500 of them the barrier; its band streams in tiles of 2 rows, four in flight, and the core finishes at 10.83 µs on the timeline (11.26 µs before the timeline-driven schedule). The `SyncAll` alone is 5 µs of the 5.39 µs target.
   - ✅ **CPU**: the same partition on the host's threads with one barrier. On 4 cores: $1 \times 2^{20}$ FP16 784 → ~190 µs.

2. **Task 2: In-place Sliding Window Reduction for Profile 8 ($10240 \times 512$)**
   - Eliminate full-row 512-element auxiliary buffers.
   - Use a 64-element SIMD accumulator (`acc[64]`) to accumulate 8 chunks of 64 elements inline, then perform bisection folding ($32 \to 16 \to 8 \to 4 \to 2 \to 1$).
   - This drops per-element memory overhead from 20B to 12B, unlocking a batch size of 24 rows without exceeding 191 KB!
   - **Target**: Break **17.57 $\mu$s**.
   - ✅ **Target**: 14 B/element for double-buffered 16-bit row tiles: X1/X2, one FP32 Z tile, and one egress buffer for Y (the DAE v1.4 egress channel reads VECOUT only; before it, Y went out through the X1 slot at 12 B/element). That admits $B^* = 25$ rows at $D = 512$ in 191 KB (23 with γ/β replicated 4 times). Row sums reach the scalar unit through `VectorReduceSum` (one `BlockReduceSum` fold first, into a partition of its own) instead of 500-cycle `GetValue` stalls: 187,480 → 52,375 cycles on the busiest core. The timeline showed that two 26-row buffers per input left the channel idle for 5.2 µs, so P08 now keeps four 12-row tiles (48 rows) in flight, with two egress buffers, a 1-row head and a 4-row tail, in 173,056 B, and applies bias and γ 4 rows per instruction: 43.07 → 38.03 µs, which is its 37.10 µs of streaming, the final store's 0.8 µs latency and 0.13 µs of channel order.
   - ✅ **CPU**: 4 × 16-lane FMA accumulators (the 64-lane window) in registers; 4 B/element of resident state; batch $B^* = \lfloor 2048/D \rfloor$ (4 rows at $D = 512$).

3. **Task 3: Dual-Stage Pipelining Overlap for Profile 4 ($768 \times 192$)**
   - 40 threads handle ~19 rows each. With batch size = 20, execution degenerates into 1 single group (zero pipeline overlap).
   - Split into two batches of 10 rows or interleave parameter loading with input streaming.
   - **Target**: Break **3.23 $\mu$s**.
   - ✅ **Target**: the planner replays every candidate schedule on the runtime's timeline and keeps the one that finishes first. P04 runs each core's 20 rows as tiles of 4 + 9 + 6 + 1 rows, three in flight. γ/β are loaded first, so their widening overlaps the first tile's DMA (parameter loading interleaved with input streaming), the small head tile starts the vector unit sooner, and the one-row tail lands the last store sooner. γ/β are replicated to 9 rows, so bias and γ cost one instruction per tile, and no row sum stalls the pipeline: 12,692 → 2,362 vector cycles, and 3.62 → 3.32 µs on the timeline, against a floor of two DMA latencies plus 1.57 µs of vector work.
   - ✅ **CPU**: the bubble is the per-row reduce → sqrt → reciprocal chain; row batching overlaps it (−20–30% per short row). Hardware prefetchers plus a 512 B software prefetch (DRAM-streaming plans) overlap loads with compute.

4. **Task 4: Cache-Oblivious Traversal for Profile 12 ($4096 \times 4096$)**
   - Benchmark runs over repeated iterations.
   - Reorder thread chunk access pattern (e.g. Morton order or serpentine snake pattern) to maximize shared 32MB L2/L3 cache hit rate.
   - **Target**: Break **48.62 $\mu$s**.
   - ✅ **CPU**: serpentine over 64 KB chunks (row-granular reversal measured 10–15% slower because it breaks prefetch streams), plus non-temporal Y stores so the LLC holds inputs.
   - **Target**: not modeled. The DAE simulation has no shared L2, so it cannot evaluate a traversal order; each core streams its tiles in ascending order.

---

## 📜 License
MIT License. Contributions and PRs welcome!
