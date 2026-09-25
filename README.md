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

### 🛠️ Hardware Constraints & Optimization Rules

Our target symmetric multi-core hardware platform imposes **strict physical laws**:

1. **40 Execution Threads (`P = 40`)**: Symmetric core pool with zero-cost thread pinning.
2. **Strict L1 Scratchpad Limit (`191 KB`)**:
   - Each thread possesses a dedicated fast L1 local scratchpad buffer strictly bounded at **191 KB (195,584 bytes)**.
   - Exceeding 191 KB results in cache thrashing, stack spills, and massive performance degradation.
3. **32-Byte Quantum Alignment**:
   - All memory burst streaming must be 32-byte aligned (`alignas(32)` / 256-bit SIMD block).
4. **256-Byte SIMD Vector Registers**:
   - Vector operations operate on 64 FP32 elements or 128 FP16 elements per instruction cycle.
5. **No Hardcoded Case Branches! (The Golden Rule)**:
   - **Do NOT write `if (M == 8 && D == 32768)`!**
   - The entire implementation must be governed by an **Adaptive 2D Tiling Model (`AdaptiveTiler`)** that dynamically computes thread distribution, batch sizes, and tile widths in closed-form mathematical equations based on tensor shape $(M, D)$.
   - Target total implementation size: **under 350 lines of clean C++**.

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
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

# 3. Run full 15-profile benchmark suite
./hpc_vector_norm_bench
```

---

## 🧩 Architectural Focus: The 5 Open Tasks

We invite contributors and AI autonomous agents to solve the 5 open optimization bottlenecks (detailed in [`docs/ARCHITECTURE_CHALLENGES.md`](docs/ARCHITECTURE_CHALLENGES.md)):

1. **Task 1: Split-D Reduction for Profile 5 ($8 \times 32768$)**
   - $M=8$ rows is too small for 40 threads (leaves 32 threads idle if row-parallel).
   - Divide columns across 32 or 40 threads ($8 \times 4 = 32$ threads, 8192 elements per thread).
   - Keep intermediate sum $Z$ resident in thread-local scratchpad ($8192 \times 2 = 16\text{ KB} \ll 191\text{ KB}$).
   - Perform lightweight 32-element inter-thread reduction to compute $\sigma$, broadcast, and finish normalization without any secondary main memory reload!
   - **Target**: Break **5.39 $\mu$s**.

2. **Task 2: In-place Sliding Window Reduction for Profile 8 ($10240 \times 512$)**
   - Eliminate full-row 512-element auxiliary buffers.
   - Use a 64-element SIMD accumulator (`acc[64]`) to accumulate 8 chunks of 64 elements inline, then perform bisection folding ($32 \to 16 \to 8 \to 4 \to 2 \to 1$).
   - This drops per-element memory overhead from 20B to 12B, unlocking a batch size of 24 rows without exceeding 191 KB!
   - **Target**: Break **17.57 $\mu$s**.

3. **Task 3: Dual-Stage Pipelining Overlap for Profile 4 ($768 \times 192$)**
   - 40 threads handle ~19 rows each. With batch size = 20, execution degenerates into 1 single group (zero pipeline overlap).
   - Split into two batches of 10 rows or interleave parameter loading with input streaming.
   - **Target**: Break **3.23 $\mu$s**.

4. **Task 4: Cache-Oblivious Traversal for Profile 12 ($4096 \times 4096$)**
   - Benchmark runs over repeated iterations.
   - Reorder thread chunk access pattern (e.g. Morton order or serpentine snake pattern) to maximize shared 32MB L2/L3 cache hit rate.
   - **Target**: Break **48.62 $\mu$s**.

---

## 📜 License
MIT License. Contributions and PRs welcome!
