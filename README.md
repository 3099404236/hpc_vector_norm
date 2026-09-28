# hpc_vector_norm (High-Performance Fused Vector Normalization Engine)

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![OpenMP](https://img.shields.io/badge/Parallel-OpenMP-green.svg)](https://www.openmp.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An ultra-high-throughput, cache-conscious, vectorized C++ numerical kernel library for **Fused Residual Vector Normalization (`FusedResidualNormalize`)** targeting high-performance many-core streaming CPU architectures (a 40-core Decoupled Access-Execute symmetric multi-core vector CPU with on-chip software-managed Scratchpad Memory (SPM), 2048-bit SIMD vector pipelines, and asynchronous DMA stream engines).

---

## 🎯 Project Mission & Challenge

In modern numerical scientific computing, digital signal processing, multidimensional physical simulations, and dense linear algebra, normalizing multi-channel vectors while accumulating residual streams is a fundamental computational primitive:

$$Z_{i} = X_{1,i} + X_{2,i}$$
$$\sigma_i = \sqrt{\frac{1}{D} \sum_{j=0}^{D-1} Z_{i,j}^2 + \epsilon}$$
$$Y_{i,j} = \frac{Z_{i,j}}{\sigma_i} \cdot \gamma_j + \text{bias}_j$$

Bias is not part of Z: it never enters the sum of squares, and it is added after the normalization and γ. This is the
target's convention ([`docs/TARGET_MEASUREMENTS.md`](docs/TARGET_MEASUREMENTS.md), section 0). Earlier revisions folded bias
into Z, and their golden references did the same, so the tests agreed with the kernels while both computed a different
function (deviation 3.4 on the target's scoring). The references in `tests/` and `src/benchmark.cpp` now follow the target.

### 🏛️ Target Hardware Laws vs Host CI Environment (Crucial Directive)

Systems programmers and kernel developers must distinguish between the **Target 40-Core Many-Core Streaming Architecture** and the **Host CI Testbed**:

| Architectural Dimension | Host CI Testbed (Development Environment) | Target Hardware Architecture (Physical Ground Truth) |
| :--- | :--- | :--- |
| **Execution Concurrency** | 4-core Cascade Lake VM (used for CI & logic verification) | **40 Dedicated Symmetric Cores (`P = 40`)** with zero fork/join cost |
| **DMA Memory Quantum** | 64-byte x86 Cache Line | **Strict 32-Byte DMA Block (`DMA_ALIGN_BYTES = 32`)** |
| **Scratchpad Buffer** | 191 KB L1 resident Z scratchpad | **Strict 191 KB (195,584 B) Scratchpad (SPM)** per core |
| **Memory Bandwidth** | ~40 GB/s DDR4 (host bus limitation) | **Measured: min(working-set regime, transfer-size ceiling) × min(1, P/27)**, up to 1,090 GB/s for ≥ 6 KB transfers (`dsa::MemorySystem`) |
| **SIMD Instruction Width** | AVX-512 (64B) / AVX2 (32B) | **256-Byte Repeat SIMD Vector Pipeline (2048-bit)** |

> **⚠️ CRITICAL ARCHITECTURAL DIRECTIVE (40-CORE TARGET FOCUS)**:
> 1. **The Primary Optimization Target is the 40-Core Many-Core Architecture (`HardwareModel::Target()`), NOT the host machine.**
> 2. The host testbed (whether a 4-core VM, an 8-core workstation, or a Xeon server) is used solely for functional correctness validation and regression testing. **Do NOT spend time measuring, profiling, or tuning host-specific CPU topology, OS thread scheduling, or spinning thread pools.** All architectural models, cycle calculations, and latency targets are calibrated against the 40-core target hardware specifications.
> 3. **The Target is a 40-Core Decoupled Access-Execute (DAE) Vector CPU**: Each core features an in-order scalar instruction pipeline, a 2048-bit wide SIMD vector unit (64 FP32 lanes), a dedicated DMA stream transfer engine (32-byte burst alignment), and a 191 KB on-chip software-managed Scratchpad Memory (SPM / Local Store). Hardware scoreboard event flags ensure hazard-free synchronization between DMA and vector units.
> 4. **Do NOT overfit to the 4-core host!** While `AdaptiveTiler` should gracefully handle `threads <= 4` on the host to avoid OS thrashing during tests, the **mathematical planning model must be explicitly architected for 40 symmetric cores**.
> 5. **Alignment must honor 32 bytes**: The hardware DMA engine transfers memory in 32-byte blocks. All dimension slicing in Split-D should support 32-byte granularity.
> 6. **Target numbers are measured on the target** ([`docs/TARGET_MEASUREMENTS.md`](docs/TARGET_MEASUREMENTS.md)): the 15 benchmark cases C1–C15, their reference and best-known times, the kernel launch (1.70 µs), `SyncAll` (0.924 µs), the memory system and the vector throughput. The runtime and the planner use those numbers; host CI timings say nothing about the target.
> 7. **Coordinator-Worker Decoupling & Zero-Allocation Freestanding Execution**:
>    - The Master Coordinator (`DaePipeline::Execute`) evaluates `TilingConfig` on the master CPU thread before the OpenMP region and manages a pre-allocated 64-byte aligned reduction workspace buffer (`float* workspace`) passed to worker threads.
>    - Worker threads must be purely freestanding: **zero dynamic allocation** (`malloc`, `new`, `std::vector`), zero exception unwinding (`throw`), and pass `LocalTensor` by value.

---

### 🛡️ DAE Stream Pipeline Runtime Engine (`include/dsa_runtime.hpp`)

To bridge the gap between high-level C++ and the target decoupled access-execute (DAE) processor, we provide an authentic **C++ Hardware Simulation Model** in [`include/dsa_runtime.hpp`](include/dsa_runtime.hpp):

- **The target's API shape** ([`docs/TARGET_API_SHAPE.md`](docs/TARGET_API_SHAPE.md)). Moving a kernel to the target is a rename wherever the two APIs have the same shape, and a rewrite wherever they differ. So the runtime refuses the 18 forms the target does not have: each keeps its declaration and fails to compile, when used, with a `static_assert` that names the constraint and what the target takes instead.
  - Global memory is reached through `GlobalTensor<T>` and `SetGlobalBuffer` only: `DataCopy(LocalTensor, GlobalTensor, count)`, `DataCopyPad(LocalTensor, GlobalTensor, DataCopyExtParams, DataCopyPadExtParams)` and their egress forms. There is no raw-pointer form and no count-only `DataCopyPad`.
  - `Cast(dst, src, RoundMode, count)` takes a rounding mode, not a converter. Narrowing rounds to nearest-even (`CAST_RINT`, and `CAST_NONE` wherever precision is lost), keeps FP16 subnormals and overflows to infinity; `CAST_TRUNC`, `CAST_FLOOR` and `CAST_CEIL` round as named.
  - No reduction returns a value: there is no `VectorReduceSum` and no `VectorInvRms`. `BlockReduceSum(dst, src, repeatTimes, mask, dstRepStride, srcBlkStride, srcRepStride)` and `WholeReduceSum(dst, src, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride)` spell out their repeat structure.
  - There is no `LocalMemAllocator`: scratchpad is `TPipe::InitBuffer(TBuf&, bytes)` and `TBuf::Get<T>()`, sliced by the kernel.
  - `ctest -R target_api_shape` compiles every refused form and requires its `static_assert`, and requires the target's forms to compile.
- **The target's semantics** (`docs/TARGET_MEASUREMENTS.md`, section 6), each a way a kernel could pass the old runtime and fail on the hardware:
  - `Brcb` is a block broadcast. Each repeat reads 8 consecutive values and fills its j-th 32-byte block with value j.
  - `ReduceSum` defines lane 0 of the 32-byte block at its destination and nothing else. The destination must start a block (row i's slot is `dst[i * 8]`), and this runtime fills lanes 1..7 with NaN, so a kernel that consumes the whole slot fails here instead of on the hardware.
  - `Rsqrt` is an 11-bit table value, the rest truncated, so a kernel refines it by Newton-Raphson.
  - The strided `Mul`/`Add(dst, src0, src1, mask, repeatTimes, BinaryRepeatParams)` takes a per-repeat mask (at most 64 lanes of 4 bytes) and block and repeat strides.
  - The scalar and vector units see each other's writes only after a fence: `CrossPipe<V_S>` / `<S_V>` or `PipeBarrier<PIPE_ALL>`, never `PipeBarrier<PIPE_V>`.
  - A new scratchpad buffer reads as NaN until written. A `DataCopyPad` descriptor starts every row on a 32-byte block in the scratchpad.
- **Integrated Hardware Sanitizer Traps**: the scratchpad budget of 191 KB (195,584 B), 32-byte DMA sizes and addresses, the queue lifecycle, egress from `VECOUT` only (Trap #401), no folded or reduced buffer aliased (Trap #402), whole 64-lane `ReduceSum` counts (Trap #408), launch arguments of at most 32 bytes (Trap #409), `Brcb` and strided operands inside their buffers (Trap #410), per-repeat masks and block-aligned reduction destinations.
- **Queue sequencer** (DAE v1.5): every `TQue` lifecycle step (`AllocTensor`, `EnQue`, `DeQue`, `FreeTensor`) costs 625 cycles. They count in the core's total cycles but not on its timeline. The kernels take every buffer from static `TBuf`s and take no `TQue` step at all.
- **Hardware Virtual Cycle Tracker**: `Add`, `Mul` (2 cycles per repeat + 13), `BlockReduceSum` (1 per repeat + 14), `WholeReduceSum` (14 per repeat + 14). Run `ctest -R dsa_runtime_sanitizer` or `./build/test_dsa_runtime`.
- **Timeline model** (`dsa::CoreTimeline`, reported by `./hpc_vector_norm_bench --timeline`). Every primitive also advances a per-core timeline, so the runtime knows *when* each operation runs, not only what it costs:
  - Three in-order units per core. **VECTOR** runs every vector instruction for its cycle cost. **DMA** is one system-memory channel shared by loads and stores. A transfer of `b` bytes occupies it for `b` at the core's share of the measured memory system (`dsa::MemorySystem`, below), and its data lands one DMA latency later, so back-to-back transfers overlap their latency. **LOCAL** runs scratchpad-to-scratchpad copies at 256 B/cycle.
  - **The memory system** (measured, `docs/TARGET_MEASUREMENTS.md` sections 3–5). The `P` cores of a launch share `min(regime(working set), ceiling(b)) × min(1, P/27)` GB/s. The transfer-size ceiling is 277 / 475 / 677 / 1,090 GB/s at 1 / 2 / 4 / ≥ 6 KB. The working-set regime runs from 1,945–2,046 GB/s (30–80 MiB) down to 943 GB/s (5.5 GiB). The two combine by `min`. Below 27 cores each core's DMA issue rate is the limit (8 cores reach 8/27 of the bandwidth).
  - **Time base.** A vector instruction costs 2 cycles per 256-byte repeat, and one repeat of a 4-byte pass measures 64 × 0.0182 ns, so a cycle is 0.582 ns (`CLOCK_GHZ` = 1.717 is this conversion, not a measured clock). The fixed per-instruction terms (+13, +14, +15, +18 cycles) are not measured.
  - **Not measured: the DMA latency.** It is set to 100 ns. The earlier 800 ns (inferred from an older benchmark's targets) contradicts the measured launch: C1 would take at least 3.4 µs and C5 7.9 µs, against 1.70 and 5.20 µs measured. The planner's decompositions are the same for 100 and 800 ns; only the body tiles of C9 (4 → 3 rows) and C14 (60 → 45 rows) change.
  - A 32-byte block scoreboard orders the units: a block is read once its last write has landed, and rewritten once its last read has ended.
  - `SyncAll`: a core arrives when its vector and local units are idle and its stores have landed (loads may keep streaming); every core leaves 0.924 µs (1,587 cycles) after the last arrival. The kernel launch, 1.70 µs whatever the core count, precedes every core's timeline.
  - `TimelineSummary` splits the finish time of a core into busy time plus **fill**, **drain**, **mismatch** and **barrier** idle time of its critical unit, and computes a **latency floor** (below).

---

## 📊 Benchmark Suite (the target's 15 cases)

The 15 cases C1–C15 of the target benchmark ([`docs/TARGET_MEASUREMENTS.md`](docs/TARGET_MEASUREMENTS.md), section 9), with the
reference implementation's time and the best known time measured on the target, next to this revision's modeled time (the
kernel launch, then the slowest core's timeline on the measured constants; a model, not a measurement):

| Case | $M \times D$ | dtype | Row bytes | Reference µs | Best known µs | Model µs | Model / best |
| :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 1 × 64 | FP16 | 128 | 2.13 | 1.70 | 2.12 | 1.25 |
| C2 | 7 × 197 | FP32 | 788 | 2.74 | 2.21 | 2.36 | 1.07 |
| C3 | 128 × 256 | FP32 | 1,024 | 3.11 | 2.47 | 2.99 | 1.21 |
| C4 | 766 × 193 | FP16 | 386 | 8.83 | 6.66 | 3.87 | 0.58 |
| C5 | 8 × 32768 | FP16 | 65,536 | 10.87 | 5.20 | 5.62 | 1.08 |
| C6 | 1508 × 577 | FP16 | 1,154 | 13.42 | 9.62 | 9.38 | 0.98 |
| C7 | 3104 × 397 | FP16 | 794 | 17.85 | 13.90 | 10.96 | 0.79 |
| C8 | 10240 × 512 | FP16 | 1,024 | 34.46 | 30.16 | 31.21 | 1.03 |
| C9 | 4080 × 1536 | FP32 | 6,144 | 71.56 | 68.20 | 71.24 | 1.04 |
| C10 | 8192 × 1024 | FP16 | 2,048 | 52.62 | 46.63 | 48.60 | 1.04 |
| C11 | 3752 × 3083 | FP32 | 12,332 | 161.39 | 131.16 | 130.53 | 1.00 |
| C12 | 3392 × 4096 | BF16 | 8,192 | 80.70 | 76.35 | 79.18 | 1.04 |
| C13 | 10432 × 3079 | FP32 | 12,316 | 565.99 | 392.72 | 356.71 | 0.91 |
| C14 | 3440640 × 128 | FP16 | 256 | 3,943.65 | 3,750.12 | 2,689.44 | 0.72 |
| C15 | 117504 × 8192 | FP16 | 16,384 | 8,903.53 | 8,321.94 | 6,122.85 | 0.74 |

A model / best ratio below 1 says the model is optimistic there, not that the kernel beats the best known run: C4, C7, C14 and
C15 are the shapes where the measured memory system (sections 3–5) does not yet explain the best known time. The model's
measurement noise differs by shape (section 7): C1, C2, C5 and C6 cannot resolve differences below 6–9%, C13–C15 below 1%.

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

# 3. Tests (4 suites): host correctness sweep (every dtype x plan x team size vs. an FP64 reference),
#    dsa_runtime sanitizer traps and target semantics, the 40-core target plans executed on the DAE
#    simulation, and the target's API shape (every refused form must stay a compile error)
ctest --output-on-failure

# 4. Run the 15-case benchmark (FP16/BF16 cases use real 16-bit storage)
./hpc_vector_norm_bench            # host latency, then the 40-core target plan of every case
./hpc_vector_norm_bench --fp32     # every case in FP32
./hpc_vector_norm_bench --target   # also run each target plan on the DAE simulation (sanitizer on, output checked)
./hpc_vector_norm_bench --timeline # --target plus the pipeline timeline of each case: bound, fill, drain, mismatch, bubble ratio, floor
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
| `launchNs` | 1700, measured: the kernel launch, the same for any core count, paid by every plan | 3500 (fork/join), measured |
| Vector work | The runtime's cycle costs, instruction by instruction (`DaeIsa`) | `elemNs` = 0.3 ns per element, measured |
| `clockGHz` | 1.717 (`dsa::CLOCK_GHZ`): 2 cycles per repeat, a repeat of a 4-byte pass measured at 1.165 ns | — |
| `syncNs` (one all-core barrier) | 924, measured (1,587 cycles) | 1000, measured |
| Streaming | `Memory(cores, workingSet)`: the measured memory system (`dsa::MemorySystem`), each transfer priced by its size | none: cache-resident rows are compute-bound |
| `latencyNs` (a DMA transfer's data lands this long after it streamed) | 100, not measured (see the timeline model above) | 0: hardware prefetchers, no explicit DMA |

On the target, the planner minimizes the slowest core's finish time on the runtime's own timeline: it replays each candidate
schedule in the kernel's issue order under the timeline rules above (see **2.** below). It counts vector cycles instruction by
instruction with the runtime's costs (`DaeIsa`):

| Instruction | Cycles |
| :--- | :--- |
| `Add`, `Adds`, `Mul`, `Muls`, `Cast` | 2 per 256-byte repeat + 13 |
| strided `Add`/`Mul` (mask per repeat) | 2 per repeat + 13 |
| `BlockReduceSum` | 1 per repeat + 14 |
| `ReduceSum` | 2 per repeat + 15 |
| `Rsqrt` | 2 per repeat + 14 |
| `Brcb` | 1 per repeat (8 blocks) + 8 |
| `Duplicate` | 1 per repeat + 18 |
| `SyncAll` | 1,587 (0.924 µs, measured) |
| `PipeBarrier<PIPE_ALL>` | drains every pipe, then 21 (12 ns, measured) |
| `GetValue` | 2 (≤ 1.3 ns, measured) once its value is written |
| `TQue` lifecycle step | 625, on the queue sequencer (not a vector cycle, not on the timeline) |

`tests/test_dae_pipeline.cpp` requires the planner's cycle count to equal the runtime's on every plan it executes, and its
modeled finish time to equal the runtime timeline's (γ and β present, aligned tensors), in every mode. Both hold. A plan's
modeled time is the kernel launch plus that timeline.
The per-repeat cost is measured; the fixed per-instruction terms and the DMA latency are not, so no decision should rest on
their magnitude alone.

**1. Decomposition, balanced to one DMA block (Challenge 1).** The planner treats the `M·D` elements as one flattened stream and
cuts it into units. Core `t` of `n` gets units `[⌊U·t/n⌋, ⌊U·(t+1)/n⌋)`, so any two cores differ by at most one unit for every
`M`, `D` and core count. There are three kinds of unit:

- `ROW_PARALLEL` uses one row. A row off the 32-byte grid starts a block of its own in the scratchpad (a multi-row `DataCopyPad` descriptor lays rows out at `RowPitch`, D rounded up to the grid), so any row count balances to one row. Cores never share a row, so they never communicate.
- `SPLIT_D` uses one 32-byte DMA block (`q = 32 / s` elements) in row-major order. A core shares at most two rows with its neighbours, at the cost of one `SyncAll`. It reads γ/β once per element it owns.
- `SPLIT_COLUMNS` (target only) numbers the same 32-byte blocks column-major: block `j` of row `i` is unit `j·M + i`. The balance is the same, but each core's share becomes a column band of every row, so it reads γ/β for that band once instead of once per element. It needs rows on the 32-byte grid (`D·s % 32 = 0`), and the band's FP32 Z must fit in the scratchpad.

A balanced partition gives its busiest core `⌈U/P⌉` units, the minimum for that unit size, so no split along 32-byte blocks does
better. For C5 (8 × 32768 FP16), 16384 blocks go to 40 cores: 24 cores take 410 blocks (6560 elements) and 16 take 409
(6544 elements). An exactly equal split does not exist, because 16384 / 40 = 409.6. The planner runs C5 row-major (5.62 µs):
8 cores would reach only 8/27 of the bandwidth (section 4).

**2. Pipeline schedule (Challenge 3, pipeline bubbles).** Each core streams its share in tiles: DMA in, vector, DMA out. Three
timeline models (`RowTilesTimeline`, `BandTimeline`, `ColumnTimeline`) replay a core's tiles in the kernel's exact issue order
under the runtime's timeline rules, and the planner keeps the schedule whose slowest core finishes first. For row tiles it
searches:

- `depth`: 2–4 tiles in flight per input ring, at the price of smaller tiles in the same 191 KB.
- `B`: body tile rows, up to the largest tile the layout admits (Challenge 2).
- `head`, `tail`: a small first tile starts the vector unit sooner (prologue fill), and a small last tile lands the final store sooner (epilogue drain).
- `outDepth`: 1–2 egress buffers (VECOUT, DAE v1.4): a second one lets a tile's result stream out while the next is written.
- Issue order: γ/β before or after tile 0 (`paramsFirst`), and X1/X2 of a later tile as soon as the tile's Z is built (`earlyLoads`) rather than after its store.

Column bands search the band tile rows (a multiple of 8) and `depth`; column tiles search the tile size and the resident Z. The
search stays fast: uniform body tiles make the replay a max-plus linear recurrence, so it jumps over whole periods, and a lower
bound cuts each candidate short once it cannot win. This is the cross-chunk software pipelining of section 11: every core keeps
`depth` tiles in flight on static rings, each slot's flags set and waited once per tile, and no `PIPE_ALL` between tiles, for any
number of tiles (C15's cores run over 2,900 one-row tiles each).

**3. Scratchpad knapsack (Challenge 2).** Every candidate the schedule search tries must fit the 191 KB layout.
The layout (`DaeLayout`) lists exactly the buffers the kernel claims through `TPipe`. The tests require the claim to equal the
plan byte for byte, so the runtime's 191 KB trap checks the planner's arithmetic.

- **Row tiles.** X1 and X2 each hold `depth` tiles in the native dtype (`2·depth·s` B per element at the row pitch), and Y leaves from `outDepth` egress buffers (`outDepth·s`). For 16-bit dtypes, one FP32 Z tile adds 4 B per element; FP32 builds Z in its egress buffer. γ/β stay resident as one FP32 row each.
- **Scratch partitions** (15,136 B, a `TBuf` each): an 8 KB chunk (casts, squares, staged γ/β), its 1 KB fold partition, a row group's slots (8 partials per row, then the Brcb blocks), the packed sums, inverse RMS and Newton-Raphson term of a group of up to 128 rows, and a 256 B workpad.
- **Column band** (`SPLIT_COLUMNS`). Tiles hold `k` band rows at the band's pitch. The band's FP32 Z for all `M` rows stays resident across the barrier, next to the γ/β band and the partial-sum records.
- **Column tiles** (row-major Split-D fragments, and rows too long for a row tile). X1, X2, a γ/β chunk pair and the egress buffer are double-buffered, plus FP32 Z: `10s + 4` B per element, and the segment's FP32 Z resident when it fits (every later tile on the 32-byte grid).

The full-size target plans (`./hpc_vector_norm_bench`): Tile is the busiest core's body tile; model cycles are its vector
cycles (the barrier excluded), and model µs the launch plus its timeline.

| Case | Mode | Cores | Unit | Busiest core (elements) | Tile | SPM / core | Model cycles | Model µs |
| :--- | :---: | ---: | :---: | ---: | :--- | ---: | ---: | ---: |
| C1 1×64 FP16 | rows | 1 | 1 row | 64 | 1 row | 16.2 KB | 310 | 2.12 |
| C2 7×197 FP32 | rows | 7 | 1 row | 197 | 1 row | 20.2 KB | 383 | 2.36 |
| C3 128×256 FP32 | rows | 40 | 1 row | 1,024 (min 768) | 4 rows | 36.8 KB | 628 | 2.99 |
| C4 766×193 FP16 | rows | 40 | 1 row | 3,860 (min 3,667) | 18 rows | 67.6 KB | 2,206 | 3.87 |
| C5 8×32768 FP16 | split | 40 | 32 B | 6,560 (min 6,544) | 4,032 el + Z | 137.7 KB | 3,027 | 5.62 |
| C6 1508×577 FP16 | rows | 40 | 1 row | 21,926 (min 21,349) | 12 rows | 130.4 KB | 10,896 | 9.38 |
| C7 3104×397 FP16 | rows | 40 | 1 row | 30,966 (min 30,569) | 15 rows | 99.9 KB | 14,024 | 10.96 |
| C8 10240×512 FP16 | rows | 40 | 1 row | 131,072 | 16 rows | 178.8 KB | 47,659 | 31.21 |
| C9 4080×1536 FP32 | rows | 40 | 1 row | 156,672 | 4 rows | 170.8 KB | 43,918 | 71.24 |
| C10 8192×1024 FP16 | rows | 40 | 1 row | 209,920 (min 208,896) | 8 rows | 182.8 KB | 77,317 | 48.60 |
| C11 3752×3083 FP32 | rows | 40 | 1 row | 289,802 (min 286,719) | 2 rows | 183.7 KB | 91,682 | 130.53 |
| C12 3392×4096 BF16 | rows | 40 | 1 row | 348,160 (min 344,064) | 2 rows | 174.8 KB | 124,086 | 79.18 |
| C13 10432×3079 FP32 | rows | 40 | 1 row | 803,619 (min 800,540) | 2 rows | 183.2 KB | 254,063 | 356.71 |
| C14 3440640×128 FP16 | rows | 40 | 1 row | 11,010,048 | 60 rows | 165.8 KB | 3,766,240 | 2,689.44 |
| C15 117504×8192 FP16 | rows | 40 | 1 row | 24,068,096 (min 24,059,904) | 1 row | 190.8 KB | 8,382,652 | 6,122.85 |

`tests/test_dae_pipeline.cpp` checks that every plan obeys the rules: these 15, 2,448 other shapes (each also forced into every
mode), and every tensor of at most 1,024 bytes. Every plan uses at most 40 cores and 195,584 B per core, balances the cores to
within one unit, cuts units and tiles on 32-byte blocks, uses `min(40, U)` cores, and finishes no later than the plan that runs
each core's rows in one tile.

**DAE executor** (`DaePipeline<Codec>`, the target path at the end of `src/kernel_unified.hpp`). Each OpenMP thread is one
simulated core (`GetBlockIdx`), and everything goes through `include/dsa_runtime.hpp` in the target's API shapes:

- **Global memory through `GlobalTensor`.** The kernel's arguments are global-memory addresses (`GM_ADDR`) and 32-bit scalars. Each address becomes a `GlobalTensor` through `SetGlobalBuffer`, the only place a typed pointer is formed, and every transfer is a `DataCopy` or `DataCopyPad` between a `LocalTensor` and a `GlobalTensor` indexed by element. The coordinator says in a launch flag which buffers start on the 32-byte grid: a transfer of whole blocks at both ends is a `DataCopy`, anything else one `DataCopyPad` descriptor. The tiling data travels as one more address and is copied in first, as `GET_TILING_DATA` does. The device's element types are the target's `half` and `bfloat16_t`, so every widening and narrowing is a `Cast` with a rounding mode (`CAST_NONE` in, `CAST_RINT` out).
- **Rows at the 32-byte pitch.** A tile's rows lie at `RowPitch`, as one multi-row `DataCopyPad` descriptor lays them out: every row starts a block, the padded load fills the rest of each row's last block with zeros, which add nothing to the row's sum, and every vector operand starts a block. One descriptor moves a whole tile each way.
- **Sums of squares without `ReduceSum`.** `ReduceSum` defines lane 0 of its slot only, so a row group's sums are built from blocks the kernel defines itself: each row's squares are added column-wise down to one 64-lane repeat (one strided `Add` per 64-lane column for every row of an 8 KB chunk, or, for rows of whole repeats and at least 512 lanes, after one 8 → 1 `BlockReduceSum` of the whole chunk, which cuts the column adds eightfold), one `BlockReduceSum` folds each row into 8 partials, and a second packs 8 rows' partials into their sums. Rows longer than the chunk go in 8 KB pieces, each folded or halved to one repeat and accumulated.
- **Inverse RMS in lanes.** `Muls` by the coordinator's `invD`, `Adds` ε, `Rsqrt` (11 bits), then Newton-Raphson steps until the bits exceed the output's significand: one for 16-bit outputs, two for FP32. One `Brcb` per 8 rows spreads the group's inverse RMS over one block per row, and the scaling is one strided `Mul` per 64-lane column that reads each row's block (`src1BlkStride` 0). γ and β apply the same way from their resident rows (`src1RepStride` 0), or row by row where that is cheaper (a few wide rows). The scalar unit never reads the scratchpad: 0 scalar stalls on every run.
- **Split-D.** Column tiles reduce each segment with `ReduceSum` into lane 0 of a slot and read lane 0 alone. Each core publishes its two 32-byte records `{Σ₀, 0 ×7}{Σ₁, 0 ×7}`, built in VECOUT from the slots' lanes 0; the owners of a row fetch the records that hold its partials, zero-pad them to whole repeats and reduce them in the same order, so every owner computes the same σ. The column band publishes one record of `M` partials, packed by the group folds, sums all 40 records with a fixed tree of 6 vector adds and normalizes its resident Z. The first γ/β chunk pair of sweep 2 is in flight during the `SyncAll`.
- **Pipelining.** Row tiles issue the loads of their first `depth` tiles and of γ/β before the first vector instruction and only then widen γ/β (16-bit γ/β are staged in the Z tile, which is free until tile 0). From then on, both input buffers of a tile refill right after its Z is built when the plan says `earlyLoads`, else after its store. Results rotate through the `outDepth` egress buffers.
- **DAE v1.4 errata (Challenge 7).** Every result and every published record leaves from a VECOUT buffer (`qY`, `bRec`), never from an input ring. A fold never writes what it reads. The coordinator passes `invD = 1/D`. The worker uses `dsa::Min`/`dsa::Max`, not `<algorithm>`.
- **DAE v1.5 (Challenge 8).** X1/X2 tiles, γ/β chunks and egress buffers come from static `TBuf` rings (`BufferRing`, a `TBuf` per slot and way) that hand out slots as the queues did, so no sequencer cycle is spent. Scoreboard flags order the pipes, one literal event ID per slot, each flag set is waited once, and `V_MTE3` hands every result to the egress channel. The launch frame holds only addresses and 32-bit scalars (Trap #409).
- **Coordinator and freestanding workers (Challenge 6).** `DaePipeline::Execute` runs on the calling thread: it checks the plan, owns the reduction workspace (caller-provided, or its own buffer reused across calls) and launches every core. Each core runs `Core::Execute` with no heap allocation and no exceptions; invariants are `DSA_ASSERT`s, `LocalTensor` views are passed by value, and a trap aborts the whole process, so no core is ever left waiting at the `SyncAll`. A heap probe in the tests confirms that `Execute` allocates nothing beyond the simulator's scratchpad buffers.
- **Results with `--target`.** On all 15 cases the output matches the host kernel, with 0 hardware traps, 0 scalar stalls and 0 queue steps. Every core claims exactly its planned scratchpad, and the runtime's cycle count and timeline finish both equal the model's. Padded transfers appear only where rows are off the 32-byte grid (C2, C4, C6, C7, C11, C13).

**Against the kernel it replaces.** The previous kernel called `VectorReduceSum` and `VectorInvRms`, which the target does not
have, flooded `Brcb` and ran shares of up to 1 KB on a `LocalMemAllocator` kernel. Its modeled times are therefore not reachable
on the target; they are the reference the target-legal kernel was tuned against, on the same measured constants:

| Case | Previous µs | Now µs | Now / previous |
| :--- | ---: | ---: | ---: |
| C1 | 2.04 | 2.12 | 1.04 |
| C2 | 2.85 | 2.36 | 0.83 |
| C3 | 2.95 | 2.99 | 1.01 |
| C4 | 5.05 | 3.87 | 0.77 |
| C5 | 5.48 | 5.62 | 1.03 |
| C6 | 11.22 | 9.38 | 0.84 |
| C7 | 12.53 | 10.96 | 0.87 |
| C8 | 33.36 | 31.21 | 0.94 |
| C9 | 71.24 | 71.24 | 1.00 |
| C10 | 48.40 | 48.60 | 1.00 |
| C11 | 218.57 | 130.53 | 0.60 |
| C12 | 79.06 | 79.18 | 1.00 |
| C13 | 598.61 | 356.71 | 0.60 |
| C14 | 3,396.65 | 2,689.44 | 0.79 |
| C15 | 6,122.51 | 6,122.85 | 1.00 |
| **geomean** | | | **0.874** |

- C11 and C13 took row-major Split-D before, because the old row unit was the 8-row group whose bytes end on the 32-byte grid; one-row units at the 32-byte pitch run them as row tiles.
- C2, C4, C6, C7, C8 and C14 gain from the fold-based sums: a row costs a share of a few strided adds and two folds instead of one `ReduceSum` of its own.
- C1, C3 and C5 lose 0.04–0.14 µs to the inverse RMS in lanes (`Rsqrt`, Newton-Raphson, `Brcb`), which replaces the non-existent 16-cycle `VectorInvRms`; that is below their measurement noise (6–9%).

### Pipeline bubbles: timeline telemetry

`./hpc_vector_norm_bench --timeline` executes every target plan and reports the timeline of the core that finishes last:

- **Compute**, the vector unit's busy time, and **Stream**, the DMA channel's. The busier of the two is the critical unit (**Crit**).
- **Bound** `= max(compute + sync, stream)`: the time with every bubble removed.
- **Bubble** `= total − bound`, split into the critical unit's idle time: **fill** (before its first operation), **drain** (after its last), **mismatch** (in between) and **wait** (inside a `SyncAll`). The **ratio** is bubble / total (launch excluded).
- **Floor** (`TimelineSummary::LatencyFloor`): the same program replayed on a relaxed core with unlimited buffers, so nothing waits for a buffer's previous use, and with loads and stores in separate queues, so a store waiting for its data never holds up a load. It is also at least the channel's total occupancy plus one latency. Every operation starts no later than on the real core, so no run can finish sooner: `tests/test_dae_pipeline.cpp` checks this on every run. **Excess** = total − floor is what buffer reuse, the shared channel's order and `SyncAll` waits cost. The floor itself is DMA latency and data dependencies.

At the benchmark's sizes (C14 with 50,000 rows, C15 with 5,000), times after the launch:

| Case | Crit | Compute µs | Stream µs | Bubble µs (fill / drain / mismatch / wait) | Ratio | Excess µs |
| :--- | :---: | ---: | ---: | :--- | ---: | ---: |
| C1 | VEC | 0.18 | 0.06 | 0.24 (0.11 / 0.11 / 0.01 / 0) | 56.7% | 0 |
| C2 | DMA | 0.22 | 0.39 | 0.27 (0 / 0.10 / 0.17 / 0) | 40.7% | 0 |
| C3 | DMA | 0.37 | 1.02 | 0.27 (0 / 0.10 / 0.17 / 0) | 20.9% | 0 |
| C4 | DMA | 1.28 | 1.30 | 0.87 (0 / 0.10 / 0.77 / 0) | 40.0% | 0 |
| C5 | DMA | 1.76 | 2.69 | 1.23 (0 / 0.10 / 0.70 / 0.43) | 31.3% | 0 |
| C6 | VEC | 6.35 | 5.26 | 1.33 (0.25 / 0.36 / 0.72 / 0) | 17.3% | 0 |
| C7 | VEC | 8.17 | 7.10 | 1.10 (0.22 / 0.39 / 0.49 / 0) | 11.8% | 0 |
| C8 | DMA | 27.76 | 29.16 | 0.35 (0 / 0.10 / 0.25 / 0) | 1.2% | 0.25 |
| C9 | DMA | 25.58 | 69.44 | 0.10 (0 / 0.10 / 0 / 0) | 0.1% | 0 |
| C10 | DMA | 45.03 | 46.57 | 0.34 (0 / 0.10 / 0.24 / 0) | 0.7% | 0.24 |
| C11 | DMA | 53.40 | 128.73 | 0.10 (0 / 0.10 / 0 / 0) | 0.1% | 0 |
| C12 | DMA | 72.27 | 77.26 | 0.22 (0 / 0.10 / 0.12 / 0) | 0.3% | 0.12 |
| C13 | DMA | 147.97 | 354.91 | 0.10 (0 / 0.10 / 0 / 0) | 0.0% | 0 |
| C14 | DMA | 32.13 | 35.30 | 0.13 (0 / 0.10 / 0.03 / 0) | 0.4% | 0.03 |
| C15 | DMA | 208.01 | 226.67 | 1.32 (0 / 0.10 / 1.22 / 0) | 0.6% | 1.22 |

Mean bubble ratio 14.8%, mean excess over the latency floor 0.14%. What is left is DMA latency: a DMA-bound core keeps at least
one latency of drain (its last store lands 100 ns after the channel's last byte), a vector-bound one two, and the small cases
move too little data per core to hide either. Only C8, C10, C12, C14 and C15 keep any excess, where a finished tile's store heads
the channel's queue while the next loads wait behind it.

**Host executor** (`KernelUnifiedPipeline<Codec>`, CI). It uses the same partition code (`AdaptiveTiler::Range` and
`RowOwners`) on `P` threads. The host candidates are inline, rows and row-major Split-D:

| Stage | Decision | Why |
| :--- | :--- | :--- |
| Team size | `P` threads iff `min(t_rows, t_split) < t_inline`, else inline on the caller (no OpenMP region) | Fork/join costs ~3.5 µs here; resizing an OpenMP team between calls costs 70–330 µs, so the team is either 1 or `P`, never in between |
| Work quantum | Rows, or 32-byte blocks with one barrier when rows would leave threads idle | The target's Split-D |
| Row batching | `B = clamp(2048 / D, 1, 64)` rows issue pass 1 before their pass 2 | Hides the reduce → sqrt → reciprocal latency of short rows |
| Resident Z | FP32 Z kept in a per-thread 191 KB scratchpad; longer segments recompute Z from cache-hot inputs | One DRAM read of X1/X2 per element |
| Stores | Non-temporal Y stores + 512 B software prefetch once `X1+X2+Y > LLC` | Saves the read-for-ownership stream; keeps more line fills in flight |
| Traversal | Serpentine at 64 KB chunk granularity (chunk order flips every call) | The chunks touched last are consumed first while still cached |

The host kernel runs two passes per row. Pass 1 computes `Z = X1 + X2` and its sum of squares with 4 independent FMA
accumulators, in FP32 within 4096-element blocks and FP64 across blocks. Pass 2 computes `Y = Z · invRms · gamma + bias`.

The ISA layer (`src/simd.hpp`) provides AVX-512 (masked tails), AVX2 + FMA + F16C, and a portable scalar path. FP16/BF16 are widened
on load and rounded to nearest-even on store. For 16-bit tensors, bias/gamma are widened to FP32 once per call when a thread owns at
least 4 rows. Each thread memoizes the plan for the last shape, so repeated calls skip planning, which takes 60–75 ns.

**Runtime fixes** (`include/dsa_runtime.hpp`, regression tests in `tests/test_dsa_runtime.cpp`):

1. **`TQue` double buffering was silently single-buffered.** `AllocTensor` picked slot `(tail + allocatedCount) % depth` and ignored tensors already enqueued, so the prefetch of tile k+1 overwrote tile k before it was dequeued. The queue is now a per-slot state machine (FREE → ALLOCATED → ENQUEUED → DEQUEUED) with FIFO order that traps allocation beyond the queue depth, freeing a free or in-flight buffer, and tensors of another queue.
2. **`DataCopy` did not check addresses**, only the transfer size. Misaligned system memory or local addresses now trap.
3. **The API-shape commit did not compile.** Two edits had pasted a `static_assert` and a function body into default arguments (`DataCopyPad` and `Brcb`), `VectorInvRms` held a `static_assert(false)` that fires in any C++17 file that includes the header, and the target's own forms (`DataCopy` / `DataCopyPad` / `LoadPad` / `StorePad` through a `GlobalTensor`, `Cast` with a rounding mode) called the refused forms, so no kernel could compile. The refused forms now fail only when used; the target's forms work; the 7-argument `BlockReduceSum` and `WholeReduceSum` the refusals point to exist, with their repeat structure.
4. **`half` converted FP16 subnormals to garbage** (0x0001 read as −1.3e36; 2,046 of the 65,536 values) and truncated instead of rounding. Every row that held one tiny input came out NaN once the kernel widened through the target's types. Both directions now match IEEE conversion bit for bit.

Size, in lines of code (neither blank nor comment): `adaptive_tiler.hpp` is 1,141, including the instruction-level cycle
model and the three streaming timeline models. `kernel_unified.hpp` is 1,033: the host executor is 244 and the DAE executor 789.
The ISA layer adds 105, and `dsa_runtime.hpp` is 1,351.

### Measured results (4-core Cascade Lake VM, AVX-512, ~40 GB/s DRAM)

The host is not the target (see the directive above); these runs check the host executor, not the target's latency. Median of 3
runs, µs, C14 with 50,000 rows and C15 with 5,000. The host is shared, so single runs can differ by up to ~50%.

| Case | dtype | Case dtype | Same case in FP32 |
| :--- | :---: | ---: | ---: |
| C1 1×64 | FP16 | 0.11 | 0.14 |
| C2 7×197 | FP32 | 0.54 | 0.57 |
| C3 128×256 | FP32 | 12.10 | 9.75 |
| C4 766×193 | FP16 | 29.73 | 30.20 |
| C5 8×32768 | FP16 | 33.24 | 45.27 |
| C6 1508×577 | FP16 | 140.23 | 136.28 |
| C7 3104×397 | FP16 | 204.53 | 234.93 |
| C8 10240×512 | FP16 | 839.50 | 1,822.24 |
| C9 4080×1536 | FP32 | 2,288.86 | 2,213.73 |
| C10 8192×1024 | FP16 | 1,607.21 | 3,200.95 |
| C11 3752×3083 | FP32 | 4,908.64 | 5,140.53 |
| C12 3392×4096 | BF16 | 3,290.03 | 5,763.12 |
| C13 10432×3079 | FP32 | 13,699.45 | 13,827.25 |
| C14 50000×128 | FP16 | 1,052.40 | 2,173.62 |
| C15 5000×8192 | FP16 | 9,369.83 | 15,891.91 |

For FP32 cases the two columns are the same workload measured in different runs, so their gap is run-to-run noise. From C8 on,
the tensors exceed the LLC and stream from DRAM at the host's ~40 GB/s read-2/write-1 roofline, far below the target's.

---

## 🧩 Architectural Focus: The 5 Open Tasks

The optimization bottlenecks, detailed in [`docs/ARCHITECTURE_CHALLENGES.md`](docs/ARCHITECTURE_CHALLENGES.md) with how each is
solved on the 40-core target and on the CPU host. The task statements name the profiles of the original benchmark (P01–P15); the
target's benchmark is C1–C15, and C5 = P05's shape.

1. **Task 1: Split-D Reduction for Profile 5 ($8 \times 32768$)**
   - $M=8$ rows is too small for 40 threads (leaves 32 threads idle if row-parallel).
   - Divide columns across 32 or 40 threads and keep the intermediate $Z$ resident, then reduce across threads and finish without reloading.
   - ✅ **Target**: the tensor is split into 32-byte DMA blocks, balanced to one block for any $M$ and core count (409.6 blocks per core on average, so 409 or 410). C5 runs row-major Split-D on all 40 cores: each core's two fragments keep their FP32 Z resident, publish two 32-byte records and pass one `SyncAll`; the owners of a row reduce its records with one `ReduceSum` in the same order. 5.62 µs modeled against 5.20 µs best known, of which the `SyncAll` is 0.92 µs.
   - ✅ **CPU**: the same partition on the host's threads with one barrier.

2. **Task 2: In-place Sliding Window Reduction for Profile 8 ($10240 \times 512$)**
   - Eliminate full-row auxiliary buffers; accumulate 64-lane windows and fold them.
   - ✅ **Target**: the squares of a chunk of rows fold 8 → 1 into a 1 KB partition, their 64-lane columns add up with one strided `Add` per column for all rows, and two `BlockReduceSum`s give the packed row sums: no per-row buffer beyond the 8 KB chunk. Double-buffered 16-bit row tiles cost 14 B per element (X1/X2, the FP32 Z tile, one egress buffer). C8 runs 16-row tiles and finishes at 31.21 µs modeled, its streaming plus 0.35 µs, against 30.16 µs best known.
   - ✅ **CPU**: 4 × 16-lane FMA accumulators (the 64-lane window) in registers; batch $B^* = \lfloor 2048/D \rfloor$.

3. **Task 3: Dual-Stage Pipelining Overlap for Profile 4 ($768 \times 192$)**
   - With one batch per core, execution degenerates into one group with zero pipeline overlap.
   - ✅ **Target**: the planner replays every candidate schedule on the runtime's timeline and keeps the one that finishes first: queue depth, egress buffers, head and tail tiles, γ/β before or after tile 0, early refills. C4 (766 × 193, rows off the 32-byte grid, one multi-row `DataCopyPad` per tile) finishes at 3.87 µs modeled.
   - ✅ **CPU**: row batching overlaps the per-row reduce → sqrt → reciprocal chain (−20–30% per short row).

4. **Task 4: Cache-Oblivious Traversal for Profile 12 ($4096 \times 4096$)**
   - Reorder the chunk access pattern to maximize the shared cache hit rate across iterations.
   - ✅ **CPU**: serpentine over 64 KB chunks (row-granular reversal measured 10–15% slower because it breaks prefetch streams), plus non-temporal Y stores so the LLC holds inputs.
   - **Target**: not modeled. The DAE simulation has no shared L2, so it cannot evaluate a traversal order; each core streams its tiles in ascending order.

---

## 📜 License
MIT License. Contributions and PRs welcome!
