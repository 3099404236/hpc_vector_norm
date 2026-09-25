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
| **Memory Bandwidth** | ~40 GB/s DDR4 (host bus limitation) | **~850 GB/s HBM3 High-Bandwidth Channel** |
| **SIMD Instruction Width** | AVX-512 (64B) / AVX2 (32B) | **256-Byte Repeat SIMD Vector Pipeline (2048-bit)** |

> **⚠️ Directives for Tiler & Kernel Optimization**:
> 1. **Do NOT overfit to the 4-core host!** While `AdaptiveTiler` should gracefully handle `threads <= 4` on the host to avoid OS thrashing during tests, the **mathematical planning model must be explicitly architected for 40 symmetric cores**.
> 2. **Alignment must honor 32 bytes**: The hardware DMA engine transfers memory in 32-byte blocks. All dimension slicing in Split-D should support 32-byte granularity.
> 3. **Latency Targets are HBM3 Roofline**: Target latencies (e.g. P13 $223\ \mu\text{s}$) assume 850 GB/s bandwidth. On the 4-core VM, reaching memory saturation (~40 GB/s) confirms the algorithm is optimal!

---

### 🛡️ DAE Stream Pipeline Runtime Engine (`include/dsa_runtime.hpp`)

To bridge the gap between high-level C++ and the target decoupled access-execute (DAE) processor, we provide an authentic **C++ Hardware Simulation Model** in [`include/dsa_runtime.hpp`](include/dsa_runtime.hpp):

- **DAE Pipeline Semantics**: Direct mapping for `TPipe`, `TQue<QuePosition, depth>`, `LocalTensor<T>`, `DataCopy`, `Add`, `Mul`, `BlockReduceSum`, and `SyncAll<true>()`.
- **Integrated Hardware Sanitizer Traps**:
  - `pass_spm_budget`: Instantly aborts if total allocated scratchpad exceeds **191 KB (195,584 bytes)**.
  - `pass_dma_align`: Instantly aborts if DMA transfers are not aligned to **32 bytes**.
  - `pass_async_hazard`: Validates queue depth and double-buffering lifecycle.
- **Hardware Virtual Cycle Tracker**:
  - Automatically profiles instruction costs: `VADD`(2 cycles/repeat), `VMUL`(2 cycles/repeat), `VCGADD`(1 cycle/repeat), `VREDUCEV2`(14 cycles/repeat).
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

# 3. Correctness sweep (every dtype x tiling path x team size vs. an FP64 reference)
ctest --output-on-failure

# 4. Run full 15-profile benchmark suite (FP16/BF16 profiles use real 16-bit storage)
./hpc_vector_norm_bench
./hpc_vector_norm_bench --fp32   # every profile in FP32, the workload of the original benchmark
```

The public entry point is `hpc::FusedResidualNormalize(x1, x2, gamma, bias, y, rows, cols, dtype, eps)`
(`include/hpc_vector_norm.hpp`). FP16/BF16 tensors are raw `uint16_t` storage; `gamma`/`bias` may be null.

---

## 🖥️ CPU Implementation

The engine adapts to the host at run time: `P = omp_get_max_threads()` (capped at 256), the LLC size from `sysconf`, and the
widest SIMD ISA the build targets. Every decision is a closed-form function of `(M, D, dtype, P, LLC)` (no shape special-cases):

| Stage | Decision (`src/adaptive_tiler.hpp`) | Why |
| :--- | :--- | :--- |
| Team size | `P` threads iff `M·D·c > τ·P/(P−1)`, else inline on the caller (no OpenMP region) | Fork/join costs ~3.5 µs here; resizing an OpenMP team between calls costs 70–330 µs, so the team is either 1 or `P`, never in between |
| Work quantum | Whole rows, or 64-byte lines with one barrier when row granularity would idle threads (`(⌈M/T⌉ − M/T)·D·c > barrier`) | Generalized Split-D: any `M`, any `T`, balanced to one cache line |
| Row batching | `B* = clamp(2048 / D, 1, 64)` rows issue pass 1 before their pass 2 | Hides the reduce → sqrt → reciprocal latency of short rows |
| Resident Z | FP32 Z kept in a per-thread 191 KB scratchpad; longer segments recompute Z from cache-hot inputs | One DRAM read of X1/X2 per element |
| Stores | Non-temporal Y stores + 512 B software prefetch once `X1+X2+Y > LLC` | Saves the read-for-ownership stream; keeps more line fills in flight |
| Traversal | Serpentine at 64 KB chunk granularity (chunk order flips every call) | The chunks touched last are consumed first while still cached |

The kernel (`src/kernel_unified.hpp`) runs two passes per row: `Z = X1 + X2 + bias` with 4 independent FMA accumulators (FP32 within
4096-element blocks, FP64 across blocks), then `Y = Z · invRms · gamma`. The ISA layer (`src/simd.hpp`) provides AVX-512 (masked tails),
AVX2 + FMA + F16C, and a portable scalar path; FP16/BF16 are widened on load and rounded to nearest-even on store. For 16-bit tensors,
bias/gamma are widened to FP32 once per call when a thread owns at least 4 rows.

Size: `adaptive_tiler.hpp` + `kernel_unified.hpp` = 312 lines of code (403 with comments); the ISA layer adds 105.

### Measured results (4-core Cascade Lake VM, AVX-512, ~40 GB/s DRAM)

Median of 3 runs, µs. The first two columns are the original code (every profile in FP32, as the original benchmark did);
"cap 4" is the same code with `MAX_THREADS` set to the 4 cores this machine has. The host is shared, so single runs can differ by up to ~50%.

| Profile | Type | Original (40 thr) | Original (cap 4) | New, same FP32 work | New, profile dtype | Speedup (as shipped → new) |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: |
| P01 1×64 | FP16 | 0.94 | 1.10 | 0.13 | 0.14 | 6.7× |
| P02 7×200 | FP32 | 746 | 257 | 0.55 | 0.53 | 1400× |
| P03 128×256 | FP32 | 798 | 35.9 | 9.45 | 11.3 | 70× |
| P04 768×192 | FP16 | 1088 | 70.6 | 16.7 | 20.7 | 53× |
| P05 8×32768 | FP16 | 1140 | 364 | 37.3 | 23.6 | 48× |
| P06 1536×576 | FP16 | 955 | 184 | 126 | 74.2 | 13× |
| P07 10240×400 | FP16 | 2637 | 1622 | 1073 | 436 | 6.1× |
| P08 10240×512 | FP16 | 3084 | 2272 | 1366 | 639 | 4.8× |
| P09 4096×1536 | FP32 | 3598 | 2397 | 2015 | 1874 | 1.9× |
| P10 8192×1024 | FP16 | 4604 | 3183 | 2729 | 1362 | 3.4× |
| P11 4096×3072 | FP32 | 7077 | 5570 | 5189 | 4283 | 1.7× |
| P12 4096×4096 | BF16 | 8067 | 6949 | 5809 | 3212 | 2.5× |
| P13 10240×3072 | FP16 | 15963 | 13908 | 11191 | 6167 | 2.6× |
| P14 50000×128 | FP16 | 3573 | 5088 | 1774 | 913 | 3.9× |
| P15 5000×8192 | FP16 | 18526 | 17497 | 15117 | 8440 | 2.2× |

For FP32 profiles the two "New" columns are the same workload measured in different runs, so their gap is run-to-run noise.

P01 and P02 beat their targets. Profiles P09–P15 exceed the LLC and stream from DRAM at 29–42 GB/s effective, against a measured
~40 GB/s read-2/write-1 roofline (P09 and P14 get some LLC reuse). The world-record targets assume far more memory bandwidth (P13's 223 µs needs ~850 GB/s),
so they are out of reach for a 4-core host. The `Target (us)` column for the scaled-down P14/P15 runs is scaled by the benchmarked row count.

---

## 🧩 Architectural Focus: The 5 Open Tasks

The 5 optimization bottlenecks (detailed in [`docs/ARCHITECTURE_CHALLENGES.md`](docs/ARCHITECTURE_CHALLENGES.md), including how each is solved on CPU):

1. **Task 1: Split-D Reduction for Profile 5 ($8 \times 32768$)**
   - $M=8$ rows is too small for 40 threads (leaves 32 threads idle if row-parallel).
   - Divide columns across 32 or 40 threads ($8 \times 4 = 32$ threads, 8192 elements per thread).
   - Keep intermediate sum $Z$ resident in thread-local scratchpad ($8192 \times 2 = 16\text{ KB} \ll 191\text{ KB}$).
   - Perform lightweight 32-element inter-thread reduction to compute $\sigma$, broadcast, and finish normalization without any secondary main memory reload!
   - **Target**: Break **5.39 $\mu$s**.
   - ✅ **CPU**: rows split into 64-byte lines for any $M$ and $T$ (at $8 \times 32768$ with 40 threads every thread gets 204–205 lines), one barrier, resident Z. On 4 cores: $1 \times 2^{20}$ FP16 784 → 197 µs.

2. **Task 2: In-place Sliding Window Reduction for Profile 8 ($10240 \times 512$)**
   - Eliminate full-row 512-element auxiliary buffers.
   - Use a 64-element SIMD accumulator (`acc[64]`) to accumulate 8 chunks of 64 elements inline, then perform bisection folding ($32 \to 16 \to 8 \to 4 \to 2 \to 1$).
   - This drops per-element memory overhead from 20B to 12B, unlocking a batch size of 24 rows without exceeding 191 KB!
   - **Target**: Break **17.57 $\mu$s**.
   - ✅ **CPU**: 4 × 16-lane FMA accumulators (the 64-lane window) in registers; 4 B/element of resident state; batch $B^* = \lfloor 2048/D \rfloor$ (4 rows at $D = 512$).

3. **Task 3: Dual-Stage Pipelining Overlap for Profile 4 ($768 \times 192$)**
   - 40 threads handle ~19 rows each. With batch size = 20, execution degenerates into 1 single group (zero pipeline overlap).
   - Split into two batches of 10 rows or interleave parameter loading with input streaming.
   - **Target**: Break **3.23 $\mu$s**.
   - ✅ **CPU**: the bubble is the per-row reduce → sqrt → reciprocal chain; row batching overlaps it (−20–30% per short row). Hardware prefetchers plus a 512 B software prefetch (DRAM-streaming plans) overlap loads with compute.

4. **Task 4: Cache-Oblivious Traversal for Profile 12 ($4096 \times 4096$)**
   - Benchmark runs over repeated iterations.
   - Reorder thread chunk access pattern (e.g. Morton order or serpentine snake pattern) to maximize shared 32MB L2/L3 cache hit rate.
   - **Target**: Break **48.62 $\mu$s**.
   - ✅ **CPU**: serpentine over 64 KB chunks (row-granular reversal measured 10–15% slower because it breaks prefetch streams), plus non-temporal Y stores so the LLC holds inputs.

---

## 📜 License
MIT License. Contributions and PRs welcome!
