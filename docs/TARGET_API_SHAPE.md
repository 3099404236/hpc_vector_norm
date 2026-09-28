# API shape: what the target actually has

## The rule

Moving a kernel from this runtime to the target is a **rename** wherever the two
APIs have the same *shape*, and a **manual rewrite** wherever they differ. Names
are cheap to map with a script. Shapes are not.

So `dsa_runtime.hpp` now refuses the forms the target does not have. Every one of
them keeps its declaration; calling it produces a `static_assert` that names the
constraint and what the target requires instead.

**This is a blacklist, not a mandate.** It says what the hardware cannot do. It
does not say how to write a kernel. Anything the target can express is still
open, including designs nobody has tried here.

## What this changes for you right now

> **Status:** resolved on this branch. The DAE kernel was rewritten to the target's
> forms and compiles against this header; `ctest -R target_api_shape` compiles each
> refused form below and requires its `static_assert`. The table records the state
> before the rewrite.

The current kernel sources will not compile against this header any more. That is
the point: every error marks a place that was silently non-translatable. Measured
on `src/kernel_unified.hpp` + `src/adaptive_tiler.hpp` before this change:

| Form | Occurrences |
| :--- | --: |
| **raw global-memory pointer parameters** (`const S*`, `S*`, `float*`) | **75** |
| `Cast(dst, src, count, converter)` | 11 |
| `DataCopy` / `DataCopyPad` with raw pointers | 6 |
| `VectorReduceSum` returning a `float` | 5 |
| `Brcb` (semantics differ) | 5 |
| `LocalMemAllocator` | 5 |
| `LocalTensor::GetData()` | 4 |
| `BlockReduceSum` (3 args vs 7) | 2 |
| `VectorInvRms` | 1 |
| `LocalTensor::pos` assignment | 1 |

The first row is the structural one: the whole worker addresses global memory by
pointer arithmetic on `x1 + e`. On the target a device function cannot cast a
global-memory pointer to a typed pointer at all — the compiler says

```
casting a global-memory pointer to a typed pointer is not allowed in a device function
```

Global memory is reached through `GlobalTensor<T>` + `SetGlobalBuffer`, and every
offset becomes an index into that tensor.

## The 18 forms now refused

**Global memory**

| Refused | On the target |
| :--- | :--- |
| `DataCopy(LocalTensor, const T*, count)` | `DataCopy(LocalTensor, GlobalTensor, count)` |
| `DataCopy(T*, LocalTensor, count)` | `DataCopy(GlobalTensor, LocalTensor, count)` |
| `DataCopyPad(LocalTensor, const T*, …)` ×2 | `DataCopyPad(LocalTensor, GlobalTensor, DataCopyExtParams, DataCopyPadExtParams)` |
| `DataCopyPad(T*, LocalTensor, …)` ×2 | `DataCopyPad(GlobalTensor, LocalTensor, DataCopyExtParams)` |
| `DataCopyPad(dst, src, count)` ×2 (no ExtParams) | the target always takes a `DataCopyExtParams` |
| `LoadPad(LocalTensor, const T*, count)` | `LoadPad(LocalTensor, GlobalTensor, count)` |
| `StorePad(T*, LocalTensor, count)` | `StorePad(GlobalTensor, LocalTensor, count)` |

**Conversion**

| Refused | On the target |
| :--- | :--- |
| `Cast(dst, src, count, converter)` | `Cast(dst, src, RoundMode::CAST_NONE, count)` — a rounding mode, and it comes before the count |
| `Cast(dst, src, count)` | same |

**Reduction**

| Refused | On the target |
| :--- | :--- |
| `VectorReduceSum(src, count) -> float` | nothing on the target returns a value. `ReduceSum(dst, src, work, count)` returns void, writes **lane 0 only** of a 32-byte slot, and needs a work tensor. Reading the scalar back costs a vector→scalar fence — inside a per-row loop that is one fence per row. Offset multi-row destinations by `dst[i * 8]` for 4-byte elements. |
| `VectorInvRms(...)` ×2 | no such instruction. `Rsqrt` on a tensor is a **low-precision table lookup**; refine it with Newton-Raphson or lose accuracy. |
| `BlockReduceSum(dst, src, count)` | `BlockReduceSum(dst, src, repeatTimes, mask, dstRepStride, srcBlkStride, srcRepStride)` |
| `WholeReduceSum(dst, src, count)` | same treatment — the repeat structure is explicit |

**Scratchpad**

| Refused | On the target |
| :--- | :--- |
| `LocalMemAllocator<Hardware::X>` | `TPipe::InitBuffer(TBuf&, bytes)` then `TBuf::Get<T>()`, sliced by the kernel. Round **every** segment start up to 32 bytes — row widths are not always even (197 / 397 / 577 / 3079 / 3083 all appear), so slicing sequentially by `k * D * sizeof(T)` produces misaligned starts, and the copy engine requires 32-byte alignment. |

## Two behaviour fixes (not blacklist entries)

These ran fine before and produced results the hardware does not produce, so a
kernel could pass `ctest` and still be wrong.

**`Brcb`** previously read `src.data[0]` and flooded the destination with that one
value. The target reads **eight consecutive** values per repeat and fills the
j-th 32-byte block with the j-th value. Code written against the old behaviour —
broadcasting one row's inverse RMS with `Brcb(bc, inv[i], padded / 64, {1, 8})` —
splices rows i..i+7 into a single row on the hardware, and accuracy collapses.

**`ReduceSum`** now fills lanes 1..7 of the destination slot with NaN. The target
writes lane 0 only and leaves the rest holding whatever was there. A kernel that
consumes the whole slot folds stale values into its sum; measured on the hardware
that showed up as accuracy **0.2526** (four rows per group, only row 0 correct) —
a number that reads like a numerical bug rather than a memory-reuse bug.

## Still present here, absent on the target

Left in place because this runtime's own trap checks depend on them. A kernel that
uses them will not translate:

- `LocalTensor<T>::GetData()` — the target exposes no raw pointer
- `LocalTensor<T>::pos` — no such field; `t.pos = …` does not compile there
- `GetCoreIdx` / `GetCoreNum` / `GetThreadIdx` / `GetThreadNum` — use `GetBlockIdx` / `GetBlockNum`
- `CrossPipe` — not a target API; it is a two-line helper (`SetFlag` then `WaitFlag` on the same event id), so carry the helper across

## Also worth knowing

`Rsqrt` on the target is a low-precision table lookup, not an exact reciprocal
square root. Two Newton-Raphson steps are the usual refinement. A CPU simulator
that returns the exact value will hide a real accuracy loss.

Measured constants, the scoring rules, the 15 benchmark shapes and the
per-shape measurement noise are in [`TARGET_MEASUREMENTS.md`](TARGET_MEASUREMENTS.md).
