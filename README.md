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
- **Hardware Virtual Cycle Tracker**:
  - Automatically profiles instruction costs: `Add`(2 cycles/repeat), `Mul`(2 cycles/repeat), `BlockReduceSum`(1 cycle/repeat), `WholeReduceSum`(14 cycles/repeat).
  - Run verification via `ctest -R dsa_runtime_sanitizer` or `./build/test_dsa_runtime`.

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
| `clockGHz` | 1.5, assumed: the runtime counts cycles, not time | — |
| `syncNs` (one all-core barrier) | `SyncAll`'s 7500 cycles / clock = 5000 | 1000, measured |
| `byteNs` (streaming, per byte and core) | 40 / 850: ~850 GB/s shared by 40 cores | 0: cache-resident rows are compute-bound |
| `tileNs` (fixed cost of one DMA tile) | 800, inferred from the P01–P03 targets and the P04/P08 tilings | 0: hardware prefetchers, no explicit tiles |

On the target, the planner minimizes the slowest core's modeled time: one `SyncAll` for split plans, plus a 3-stage DMA pipeline
over `W = max(vector cycles / clock, DMA bytes · byteNs)`. It counts vector cycles instruction by instruction with the runtime's
costs (`DaeIsa`):

| Instruction | Cycles |
| :--- | :--- |
| `Add`, `Mul`, `Muls`, `Cast` | 2 per 256-byte repeat + 13 |
| `BlockReduceSum` | 1 per repeat + 14 |
| `VectorReduceSum` | 2 per repeat + 15 |
| `Duplicate` | 1 per repeat + 18 |
| `VectorInvRms` | 16 |
| `SyncAll` | 7500 |

`tests/test_dae_pipeline.cpp` requires the planner's count to equal the runtime's count on every plan it executes, and it does.
The clock only converts cycles to time. For each of the 15 profiles, every clock from 1.0 to 4.0 GHz picks the same decomposition,
and every clock from 1.5 to 2.5 GHz picks the identical plan.
[`docs/ARCHITECTURE_CHALLENGES.md`](docs/ARCHITECTURE_CHALLENGES.md#-cost-model--methodology) derives every constant. The clock and
`tileNs` should be recalibrated on physical target hardware; the equations stay the same.

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

**2. Pipeline depth (Challenge 3).** Each core streams its share in `n` double-buffered tiles through three stages: DMA in,
vector, DMA out. Each stage costs `W/n + tileNs` per tile, so `T(n) = (n + 2)(W/n + tileNs)`, which is minimized at
`n* = √(2W / tileNs)`. P04 has `W = 1.45 µs`: 2180 vector cycles, above its 1.12 µs of DMA. That gives `n* = 2`, so each core's
20 rows run as two 10-row tiles, and the second tile's DMA overlaps the first tile's compute.

**3. Scratchpad knapsack (Challenge 2).** The tile is the smaller of the `n*` tile and the largest tile the 191 KB layout admits.
The layout (`DaeLayout`) lists exactly the buffers the kernel claims through `TPipe`. The tests require the claim to equal the
plan byte for byte, so the runtime's 191 KB trap checks the planner's arithmetic.

- **Row tiles.** X1 and X2 are double-buffered in the native dtype (`4s` B per element), and Y is written back through the X1 slot. For 16-bit dtypes, one FP32 Z tile adds 4 B per element; FP32 computes in place. So a row tile costs `b = 12` B per element for FP16/BF16 and 16 B for FP32.
  - γ/β stay resident in FP32, replicated into `rep` rows (`2·Align32(4·rep·D)` B, at most 16 KB). Bias and γ then take one instruction per `rep` rows instead of one per row.
  - The 8 KB scratch chunk holds the widened X2, the squares, and the staged γ/β.
  - So `B*(D) = ⌊(195,584 − 8,192 − 2·Align32(4·rep·D)) / (b·D)⌋` rows: 29 rows at `D = 512` without replication, 27 with `rep = 4`.
  - The planner plans both options and keeps the faster. Replication wins wherever spare scratchpad pays for it; P14 spends 10 of its 121 possible rows on it to cut 22% of its vector cycles.
- **Column band** (`SPLIT_COLUMNS`). Tiles hold `k` band rows at the band's pitch. The band's FP32 Z for all `M` rows stays resident across the barrier, next to the replicated γ/β band and the partial-sum records.
- **Column tiles** (row-major Split-D fragments, and rows too long for a row tile). X1, X2 and a γ/β chunk are double-buffered, plus FP32 Z: `6s + 4` B per element. The segment's FP32 Z stays resident when it fits, so the normalize sweep reads nothing from main memory a second time.

The full-size target plans are below; `./hpc_vector_norm_bench` prints them. Model cycles are the slowest core's vector cycles
(the barrier excluded), and model µs is its modeled time at 1.5 GHz. `tests/test_dae_pipeline.cpp` checks that every plan, for
these and for 2,448 other shapes, each also forced into every mode:

- uses at most 40 cores and at most 195,584 B per core;
- balances the cores to within one unit;
- cuts units and tiles on 32-byte blocks;
- uses `min(40, U)` cores.

| Profile | Mode | Cores | Unit | Busiest core (elements) | Tiles × size | `rep` | Bound | SPM / core | Model cycles | Model µs |
| :--- | :---: | ---: | :---: | ---: | :--- | ---: | :---: | ---: | ---: | ---: |
| P01 1×64 FP16 | rows | 1 | 1 row | 64 | 1 × 1 row | 1 | `n*` | 9,472 B | 183 | 2.8 |
| P02 7×200 FP32 | rows | 7 | 1 row | 200 | 1 × 1 row | 1 | `n*` | 12,992 B | 144 | 3.0 |
| P03 128×256 FP32 | rows | 40 | 1 row | 1,024 (min 768) | 1 × 4 rows | 4 | `n*` | 32,768 B | 420 | 4.4 |
| P04 768×192 FP16 | rows | 40 | 1 row | 3,840 (min 3,648) | 2 × 10 rows | 10 | `n*` | 46,592 B | 2,180 | 6.1 |
| P05 8×32768 FP16 | **band** | 40 | 32 B | 6,560 (min 6,544) | 2 × 4 rows × 832 | 2 | `n*` | 76,064 B | 2,865 + 7,500 | 12.3 |
| P06 1536×576 FP16 | rows | 40 | 1 row | 22,464 (min 21,888) | 4 × 10 rows | 3 | `n*` | 91,136 B | 9,085 | 14.5 |
| P07 10240×400 FP16 | rows | 40 | 1 row | 102,400 | 9 × 29 rows | 5 | `n*` | 163,392 B | 44,114 | 44.7 |
| P08 10240×512 FP16 | rows | 40 | 1 row | 131,072 | 10 × 26 rows | 4 | `n*` | 184,320 B | 52,341 | 54.1 |
| P09 4096×1536 FP32 | rows | 40 | 1 row | 158,208 (min 156,672) | 15 × 7 rows | 1 | SPM | 192,512 B | 37,996 | 115.5 |
| P10 8192×1024 FP16 | rows | 40 | 1 row | 209,920 (min 208,896) | 15 × 14 rows | 1 | SPM | 188,416 B | 76,867 | 81.0 |
| P11 4096×3072 FP32 | rows | 40 | 1 row | 316,416 (min 313,344) | 35 × 3 rows | 1 | SPM | 180,224 B | 70,186 | 219.7 |
| P12 4096×4096 BF16 | rows | 40 | 1 row | 421,888 (min 417,792) | 35 × 3 rows | 1 | SPM | 188,416 B | 134,577 | 156.3 |
| P13 10240×3072 FP16 | rows | 40 | 1 row | 786,432 | 64 × 4 rows | 1 | SPM | 180,224 B | 258,906 | 282.4 |
| P14 2M×128 FP16 | rows | 40 | 1 row | 6,710,912 (min 6,710,784) | 473 × 111 rows | 16 | SPM | 195,072 B | 4,421,607 | 3,340 |
| P15 115K×8192 FP16 | rows | 40 | 1 row | 23,552,000 | 2,875 × 1 row | 1 | SPM | 172,032 B | 7,363,439 | 8,958 |

P05 is the only profile that splits. The model puts its column band at 12.3 µs, against 14.2 µs for the row-major split and
28.1 µs for whole rows. Everywhere else, whole rows already balance to within one row, and one row costs less than a `SyncAll`.

**DAE executor** (`DaePipeline<Codec>`, the target path at the end of `src/kernel_unified.hpp`). Each OpenMP thread is one
simulated core (`GetCoreIdx`), and everything goes through `include/dsa_runtime.hpp`:

- **No scalar stalls.** The scalar unit never reads the scratchpad (`GetValue`, a 500-cycle V→S stall). Row sums reach it through `VectorReduceSum`, and Split-D partials are combined by vector adds. Every row costs one `VectorReduceSum`, one `VectorInvRms` and one `Muls`. The `VectorReduceSum` is preceded by one 8 → 1 `BlockReduceSum` fold over a whole chunk of rows when the cost model says that is cheaper. The kernel never uses `WholeReduceSum`.
- **Tile-wide instructions.** Widening X1, widening and adding X2, the squares and the narrowing each cover a whole tile or an 8 KB chunk. Bias and γ cover `rep` replicated rows per instruction.
- **Prologue overlap (Challenge 3).** Row tiles issue tile 0's DMA, then γ/β, then tile 1's, and only then widen γ/β. So the parameter loads stream in behind the inputs. Replicating γ/β is scratchpad-to-scratchpad DMA, with no vector cycles. From then on, tile k+2 streams in while tile k+1 is computed.
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
  - `DaePipeline::Execute` runs on the calling thread. It checks the plan, owns the reduction workspace (caller-provided, or its own buffer reused across calls) and hands every core a POD `Args`.
  - Each core runs `Core::Execute` with no heap allocation and no exceptions. Row sums and band columns sit in fixed 128-row arrays on the worker's stack; longer tiles run in groups of 128 rows, and the cycle model counts the groups.
  - Invariants are `DSA_ASSERT`s, and `LocalTensor` views are passed by value.
  - A trap aborts the whole process, so no core is ever left waiting at the `SyncAll`.
  - A heap probe in the tests confirms that `Execute` allocates nothing beyond the simulator's scratchpad buffers.
- **Exact scratchpad claim.** The kernel claims only the layout's buffers. The previous merge added a 64-byte placeholder Z buffer when a plan has none, which pushed plans that fill the scratchpad to the byte over the waterline. 100,000 × 128 FP32 trapped at 195,648 B, for example. A regression test now runs such a plan.
- **DMA.** Every transfer is a 32-byte `DataCopy`. `DataCopyPad` is used only where a transfer does not end on a 32-byte block (`D·s % 32 ≠ 0`) or starts off the 32-byte grid.
- **Results with `--target`.** On all 15 profiles the output matches the host kernel, with 0 scalar stalls and 0 padded transfers. Every core claims exactly its planned scratchpad (at most 191 KB). The runtime's cycle count equals the model's.

Busiest-core cycles on the DAE runtime, before and after this round. These are the benchmark's sizes (P14 is 50,000 rows and
P15 is 5,000), so they differ from the full-size model above:

| Profile | Before: total (scalar stalls) | After: total (scalar stalls) | Change |
| :--- | ---: | ---: | ---: |
| P01 1×64 | 713 (500) | 183 (0) | −74% |
| P02 7×200 | 698 (500) | 144 (0) | −79% |
| P03 128×256 | 2,562 (2,000) | 420 (0) | −84% |
| **P04 768×192** | 12,692 (10,000) | **2,180** (0) | **−83%** |
| **P05 8×32768** | 12,859 (2,500; + 7,500 barrier) | **10,365** (0; + 7,500 barrier) | **−19%** |
| P06 1536×576 | 29,422 (19,500) | 9,085 (0) | −69% |
| P07 10240×400 | 178,028 (128,000) | 44,114 (0) | −75% |
| **P08 10240×512** | 187,480 (128,000) | **52,341** (0) | **−72%** |
| P09 4096×1536 | 92,502 (51,500) | 37,996 (0) | −59% |
| P10 8192×1024 | 182,356 (102,500) | 76,867 (0) | −58% |
| P11 4096×3072 | 127,878 (51,500) | 70,186 (0) | −45% |
| P12 4096×4096 | 191,639 (51,500) | 134,577 (0) | −30% |
| P13 10240×3072 | 401,754 (128,000) | 258,906 (0) | −36% |
| P14 50000×128 | 763,967 (625,000) | 106,222 (0) | −86% |
| P15 5000×8192 | 396,663 (62,500) | 320,689 (0) | −19% |

What remains on P05 is 72% barrier. `SyncAll` is the only cross-core primitive, and Split-D needs exactly one. A plan without a
barrier would have to read whole rows: 32,768 elements on each of 8 cores, or every core's full row redundantly. Either moves
three times the column band's DMA or more.

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

The runtime also gains:

- `Cast`, charged at the vector conversion cycle cost;
- `DataCopyPad`, which zero-pads a partial block and is counted;
- per-core DMA byte, transfer and pad counters;
- a trap when `InitBuffer` asks for more buffers than the queue depth.

Size: `adaptive_tiler.hpp` is 492 lines of code, including the instruction-level cycle model. `kernel_unified.hpp` is 800: the
host executor is 248 and the DAE executor is 552. The ISA layer adds 105.

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
   - ✅ **Target**: the tensor is split into 32-byte DMA blocks, balanced to one block for any $M$ and core count (409.6 blocks per core on average, so 409 or 410: no exactly equal split exists). P05 numbers the blocks column-major: each of the 40 cores owns a 51–52-block band of all 8 rows, reads that band of γ/β once (3.3 KB instead of 26 KB), keeps the band's FP32 Z resident and passes one `SyncAll`. All 40 partial records then come back in one DMA and are summed by 6 vector adds, with no scalar stall. Busiest core: 10,365 cycles, 7,500 of them the barrier.
   - ✅ **CPU**: the same partition on the host's threads with one barrier. On 4 cores: $1 \times 2^{20}$ FP16 784 → ~190 µs.

2. **Task 2: In-place Sliding Window Reduction for Profile 8 ($10240 \times 512$)**
   - Eliminate full-row 512-element auxiliary buffers.
   - Use a 64-element SIMD accumulator (`acc[64]`) to accumulate 8 chunks of 64 elements inline, then perform bisection folding ($32 \to 16 \to 8 \to 4 \to 2 \to 1$).
   - This drops per-element memory overhead from 20B to 12B, unlocking a batch size of 24 rows without exceeding 191 KB!
   - **Target**: Break **17.57 $\mu$s**.
   - ✅ **Target**: 12 B/element for 16-bit row tiles: double-buffered X1/X2 plus one FP32 Z tile, with Y written back through the X1 slot. That admits $B^* = 29$ rows at $D = 512$ in 191 KB (27 with γ/β replicated 4 times). P08 runs 10 tiles of 26 rows (180 KB) with bias and γ applied 4 rows per instruction, and row sums reach the scalar unit through `VectorReduceSum` (one `BlockReduceSum` fold first) instead of 500-cycle `GetValue` stalls: 187,480 → 52,341 cycles on the busiest core.
   - ✅ **CPU**: 4 × 16-lane FMA accumulators (the 64-lane window) in registers; 4 B/element of resident state; batch $B^* = \lfloor 2048/D \rfloor$ (4 rows at $D = 512$).

3. **Task 3: Dual-Stage Pipelining Overlap for Profile 4 ($768 \times 192$)**
   - 40 threads handle ~19 rows each. With batch size = 20, execution degenerates into 1 single group (zero pipeline overlap).
   - Split into two batches of 10 rows or interleave parameter loading with input streaming.
   - **Target**: Break **3.23 $\mu$s**.
   - ✅ **Target**: $n^* = \sqrt{2W/\text{tileNs}}$ tiles per core. P04 runs as 2 × 10-row tiles: tile 0, γ/β and tile 1 are all in flight before the first vector instruction (parameter loading interleaved with input streaming), and tile k+2 streams in under tile k+1. γ/β are replicated to 10 rows, so bias and γ cost one instruction per tile, and no row sum stalls the pipeline: 12,692 → 2,180 cycles on the busiest core.
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
