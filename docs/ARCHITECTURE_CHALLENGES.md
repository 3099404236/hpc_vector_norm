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

---

## 🎯 Challenge 4: Cache-Conscious Space-Filling Traversal (Morton / Serpentine Order)

### The Dilemma
For large matrices evaluated over repeated iterations (e.g., Profile 12: $4096 \times 4096$), sequential row access suffers from cold cache eviction when returning to row 0.
The measured bandwidth of $1.06\text{ TB/s}$ is bounded by external memory, while the theoretical record achieves **$2.07\text{ TB/s}$** via 32MB L2/L3 cache residency.

### The Objective
Implement a cache-oblivious or cache-conscious index permutation:
$$\pi: [0, M-1] \to [0, M-1]$$
Using serpentine (snake-like bi-directional) traversal or Morton (Z-order) tiling, maximizing cache hit rate between successive evaluation epochs.

---

## 🎯 Challenge 5: Extreme Vectorization & Register Reuse (Zero-Spill Core)

### The Objective
Refactor the reduction tree and normalization pass to ensure:
1. Zero stack spills verified by `-Wframe-larger-than` or compiler assembly audit.
2. 100% vector instruction saturation (FMA instructions without branch jumps).
