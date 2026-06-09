# PTI Interleaved Activation Patch for llama.cpp

> **STATUS — 2026-06-08: READY TO BENCHMARK**
>
> All blockers resolved. `pti_4seq` shows **100% 4-ACCEPT at 38.1 tok/s** on
> Qwen3.6-27B-UD-Q6_K_XL (MI50, greedy, -n 80).
>
> **Fixes applied:**
> - `reinit_seq()`: rm_all + seq_cp chain for REJECT/2-ACCEPT/3-ACCEPT paths
> - SSM cell priming: seq0 run single-seq before B/C/D inits to establish correct
>   recurrent cell allocation order for the quad-batch gather
>
> **Next:** benchmark `GGML_PTI_INTERLEAVED=0` vs `GGML_PTI_INTERLEAVED=1` on
> Qwen3.6-27B-UD-Q6_K_XL to measure the interleaved activation kernel speedup.

**Target file**: `ggml/src/ggml-cuda/mmvq.cu`
**Status**: kernel written, benchmark blocked (reinit incompatibility with hybrid model)

## What this does

Eliminates the 2× overhead that `mul_mat_vec_q` incurs at `ncols_dst=4`
(the N=4 PTI decode path) by changing the activation memory access pattern.

### Root cause

For each weight block at position `kby`, the N=4 kernel reads four activation
blocks from addresses 5760 bytes apart (one per column, `stride_col_y` apart):

```
y[0 * 5760 + kby]   col 0 activation at K-position kby
y[1 * 5760 + kby]   col 1 activation at K-position kby
y[2 * 5760 + kby]   col 2 activation at K-position kby
y[3 * 5760 + kby]   col 3 activation at K-position kby
```

These four reads hit four different cache lines, saturating L2 bandwidth.

### Fix

Pack the four columns into interleaved layout before the GEMV:

```
y_intlv[kby * 4 + j]   all four activation blocks adjacent in 3 cache lines
```

The inner loop then reads `ybase[j]` where `ybase = &y[kby * ncols_dst]`.
All four column reads are within 144 consecutive bytes instead of 4 × 5760 B.

## How to apply

```bash
# From your llama.cpp root
git apply llama.cpp-patch/0001-pti-interleaved-activation.patch
cmake --build build --config Release -j $(nproc)
```

## How to test

```bash
# Correctness: output tokens must be bit-identical to baseline
GGML_PTI_INTERLEAVED=0 ./pti_4seq ...  > baseline.txt
GGML_PTI_INTERLEAVED=1 ./pti_4seq ...  > patched.txt
diff baseline.txt patched.txt   # must be empty

# Performance: compare step times
GGML_PTI_INTERLEAVED=0 ./pti_4seq -m model.gguf -p "hello" -n 80 2>&1 | grep tok/s
GGML_PTI_INTERLEAVED=1 ./pti_4seq -m model.gguf -p "hello" -n 80 2>&1 | grep tok/s
```

## Expected result

| Config | step time | tok/s | HBM eff BW |
|---|---|---|---|
| N=4 baseline (no patch) | ~105 ms | 38.1 | 177 GB/s |
| N=4 interleaved (patched) | ~52 ms | ~67 | ~330 GB/s |
| N=1 baseline (reference) | 47 ms | 19.4 | 393 GB/s |

## Files changed

| File | Change |
|---|---|
| `ggml/src/ggml-cuda/mmvq.cu:396` | Add `bool pti_interleaved = false` to `mul_mat_vec_q` template |
| `ggml/src/ggml-cuda/mmvq.cu:495` | Add `if constexpr (pti_interleaved)` inner loop branch |
| `ggml/src/ggml-cuda/mmvq.cu:689` | Add `bool pti_interleaved = false` to `mul_mat_vec_q_switch_fusion` |
| `ggml/src/ggml-cuda/mmvq.cu:714` | Launch interleaved kernel variant when `pti_interleaved` |
| `ggml/src/ggml-cuda/mmvq.cu:756` | Add `bool pti_interleaved = false` to `mul_mat_vec_q_switch_ncols_dst` |
| `ggml/src/ggml-cuda/mmvq.cu:868` | Dispatch PTI variant in `case 4` |
| `ggml/src/ggml-cuda/mmvq.cu:926` | Add `bool pti_interleaved = false` to `mul_mat_vec_q_switch_type` |
| `ggml/src/ggml-cuda/mmvq.cu:1072` | Add `pti_interleave_q8_1_kernel<N>` GPU kernel |
| `ggml/src/ggml-cuda/mmvq.cu:1178` | Env-var gated dispatch in `ggml_cuda_mul_mat_vec_q` |

All changes are backward compatible. `pti_interleaved = false` is the default
everywhere; the env var `GGML_PTI_INTERLEAVED=1` activates the new path only
for non-MoE batches with exactly `ncols_dst = 4`.
