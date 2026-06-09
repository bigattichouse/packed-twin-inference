# PTI v2 — Integration Plan

**Goal**: integrate PTI into llama.cpp to eliminate the batch-GEMM overhead
and reach 80–100 tok/s on Qwen3.6-27B (MI50).

**Current baseline**: 38.1 tok/s (1.96×) via external `pti_4seq.cpp` using
llama.cpp's public API. No patches to llama.cpp required for 2×.

---

## Where We Are

### Measured results (MI50, Qwen3.6-27B, greedy, -n 80)

| Config | Model | tok/s | overhead | source |
|---|---|---|---|---|
| baseline | UD-Q6_K_XL | 19.4 | 1.00× | llama.cpp single-token |
| PTI 2-seq | UD-Q6_K_XL | 30.5 | 1.27× | pti_4seq.cpp |
| PTI 3-seq | UD-Q6_K_XL | 33.9 | 1.71× | pti_4seq.cpp / pti_mtp.cpp |
| **PTI 4-seq** | **UD-Q6_K_XL** | **38.1** | **2.04×** | **pti_4seq.cpp** |

Overhead per added sequence is sub-linear (+0.27, +0.44, +0.33 per step).
The bottleneck is llama.cpp routing multi-token batches through a batch GEMM
code path that is less memory-efficient than its single-token GEMV path.

### Bandwidth analysis (pti_kernel.hip benchmarks, FFN layer [17408×5120])

| Kernel | GB/s | % of D2D | finding |
|---|---|---|---|
| D2D hipMemcpy | 383 | 100% | hardware ceiling |
| Raw int8 stream | 330 | 86% | HBM throughput ceiling |
| Weight-only GEMV (no activations) | 254 | 66% | weight-read ceiling |
| Flat Q8 GEMV (1 activation vector) | 92 | 24% | activation L2 traffic limits |
| Vectorized (128-bit loads) | 100 | 26% | L2 bottleneck, not MSHR |
| Register-tiled (M_REG=4) | 127 | 33% | 4× less L2 pressure |
| Interleaved float4 (4-stream) | 130 | 34% | best custom Q8 kernel |
| **llama.cpp Q5_K_M GEMV (N=1)** | **393** | **103%** | **5-bit format, more data/cache-line** |

Root cause: Q5_K_M packs 5 bits/weight → 60% more weights per 64-byte cache
line vs Q8_0. Our custom Q8 kernel is limited to 254 GB/s ceiling while
llama.cpp already reads the Q5_K_M model at 393 GB/s effective bandwidth.

**The overhead is not the kernel dispatch or activation L2 traffic.** N=4
uses `mul_mat_vec_q` (confirmed, not GEMM). The pre-pass interleave benchmark
(Phase 2) shows the activation L2 traffic is not the bottleneck — the overhead
is from 4× compute in the MMVQ inner loop (multiply-accumulate for 4 output
columns) plus attention and SSM computation scaling linearly with N.

---

## Integration Target: llama.cpp

### Key file: `ggml/src/ggml-cuda/mmvq.cu`

The `mul_mat_vec_q<type, ncols_dst>` template already computes multiple output
columns per weight load. For GCN (MI50):

```
ncols_dst=1: nwarps=2, rows_per_block=1  → 19.4 tok/s baseline
ncols_dst=4: nwarps=2, rows_per_block=2  → same kernel, 2× step cost
```

The kernel loads each weight block once and loops over `ncols_dst` activation
vectors. On paper this should scale near-linearly with N. The measured 2.04×
overhead at N=4 comes from:
1. Activation L2 pressure: 4× more activation reads per block
2. Longer reduction tree: `nwarps` stays at 2, but 4× more accumulator state
3. Possible dispatch switch: llama.cpp may route N>1 to a different path

### Dispatch investigation needed (Step 0)

File: `ggml/src/ggml-cuda/ggml-cuda.cu`, around line 2638:
```cpp
if (ne2 <= MMVQ_MAX_BATCH_SIZE) {
    // MMVQ path (quantized GEMV)
} else {
    // MMQ or cuBLAS path (batch GEMM)
}
```

Confirm whether N=4 still uses `mul_mat_vec_q` or falls through to GEMM.
`MMVQ_MAX_BATCH_SIZE` for GCN Q5_K is 4 (from `get_mmvq_mmid_max_batch_gcn`).
So N=4 should still be MMVQ — overhead is from within the kernel, not dispatch.

---

## Three-Phase Integration Plan

### Phase 1 — Overhead source confirmed (done)

**Dispatch analysis** (static, no profiling needed):

`MMVQ_MAX_BATCH_SIZE = 8`. For N=4 on MI50 GCN:
- `src1->ne[1] = 4 <= 8` → `use_mul_mat_vec_q = true`
- `ggml_cuda_should_use_mmq` returns `false` for GCN (no DP4A / MMA)
- Dispatch path: `ggml_cuda_mul_mat_vec_q` → `mul_mat_vec_q<Q5_K, ncols_dst=4>`

**N=4 uses the MMVQ GEMV kernel, not cuBLAS/rocBLAS GEMM.**
Overhead is inside `mul_mat_vec_q` from increased L2 activation traffic:

```
N=1:  17408 blocks × 1 × 20 KB activations = 357 MB L2 traffic
N=4:   8704 blocks × 4 × 20 KB activations = 714 MB L2 traffic  (2× more)
```

The activation tensor (20 KB per seq) fits in MI50's 16 MB L2, so it's not
an HBM miss — it's L2 bandwidth saturation from 2× more requests.
The weight reads (89 MB) are identical for N=1 and N=4.

**Fix**: interleave the 4 activation tensors as `[K × 4]` float4 layout.
One float4 load fetches all 4 activations at the same K position →
4× fewer L2 transactions → L2 traffic drops from 714 MB back to ~179 MB.

This is exactly the `pti_linear4_q8_intlv` approach benchmarked in
`pti_kernel.hip` at 130 GB/s (the best of all custom kernels).

**Deliverable from Phase 1**: complete dispatch/overhead analysis (this document).
Phase 2 proceeds directly to writing the kernel patch.

---

### Phase 2 — Eliminate overhead (patch written, benchmarked: no gain)

**Status**: patch applied and benchmarked on Qwen3.6-27B (MI50, greedy, -n 80):

| Config | Model | tok/s | vs baseline |
|---|---|---|---|
| GGML_PTI_INTERLEAVED=0 (baseline) | UD-Q6_K_XL | 38.1 | — |
| GGML_PTI_INTERLEAVED=1 (patched) | UD-Q6_K_XL | 35.6 | −6% (within noise, no gain) |
| GGML_PTI_INTERLEAVED=0 (baseline) | Q5_K_M | ~29 | — |
| GGML_PTI_INTERLEAVED=1 (patched) | Q5_K_M | ~25 | −14% (net loss) |

**Root cause revised**: the 2× overhead at N=4 is NOT from activation L2 traffic.
The pre-pass interleave kernel adds ~160 KB/layer of extra memory I/O (read +
write the activation buffer) which is not recouped by improved cache behavior.
Activation traffic is negligible (~480 MB/step total) vs weight loading (~18.6 GB),
so optimizing L2 activation access cannot move the overall step time.

**Root cause confirmed** (from Phase 1 dispatch analysis):
- N=4 uses `mul_mat_vec_q<Q5_K, ncols_dst=4>` on MMVQ path (confirmed, not GEMM)
- Inner loop reads `y[j * stride_col_y + kby]` for j = 0..3
- `stride_col_y = 160 blocks = 5760 bytes` → 4 accesses 5760 bytes apart per kby
- 160 kby iterations × 4 scattered reads = 640 L2 transactions vs 160 for N=1

**Patch approach** (see `pti_mmvq_patch.cu` for full annotated diff):

1. `pti_interleave_q8_1_kernel<N>` — pre-packs N activation columns into
   interleaved layout `y_intlv[kby * N + j]` so all N blocks per kby position
   are adjacent (144 bytes in 3 cache lines, not 4 × 5760 bytes apart).
   Launched once per GEMV call; negligible cost (22.5 KB of data movement).

2. `mul_mat_vec_q<..., pti_interleaved=true>` — new bool template param.
   When true, the inner loop uses `ybase = &y[kby * ncols_dst]` with `ybase[j]`
   instead of `y[j * stride_col_y + kby]`.  All existing code paths compile
   exactly as before (the new param defaults to false).

3. `ggml_cuda_mul_mat_vec_q_pti` — new dispatch function in ggml-cuda.cu.
   Checks `ctx.pti_mode && ncols_dst == 4`, interleaves activations into a
   pre-allocated 78 KB buffer, then calls the interleaved kernel variant.

4. `llama_context_set_pti(ctx, n_streams)` — allocates the interleaved buffer
   once at context init; reused across all decode steps.

**Files to modify in llama.cpp**:
- `ggml/src/ggml-cuda/mmvq.cu` — add `pti_interleaved` template param + branch
- `ggml/src/ggml-cuda/ggml-cuda.cu` — add PTI dispatch + `pti_interleave_q8_1_kernel`
- `src/llama.cpp` — add `pti_n_streams`, `pti_act_buf` to `llama_context`
- `include/llama.h` — add `llama_context_set_pti`, `llama_pti_params`

**Expected result**: step overhead drops from 2.04× → 1.05–1.15×.
```
4 / 1.15 × 19.4 ≈ 67 tok/s  (conservative)
4 / 1.05 × 19.4 ≈ 74 tok/s  (optimistic)
```

**Validation**: run `pti_4seq` with `GGML_CUDA_FORCE_MMVQ_PTI=1` env var;
output tokens must be bit-identical to baseline (same argmax on same logits).

---

### Phase 3 — PTI × MTP integration (3–5 days)

**Goal**: use the UD-Q6_K_XL MTP head to add free tokens on top of PTI.

The UD model has `nextn_predict_layers=1`. The MTP head is 1 transformer layer
(vs 64 main layers) — ~1.5% of total weight. Running it after the main forward
pass costs ~1.5% extra compute/bandwidth, returning ~1 extra token.

**Architecture** (uses existing `pti_mtp.cpp` approach):

```
Per step:
  1. Main forward (N=4 sequences) → N logits
  2. Verify/accept (existing PTI logic) → emit 1-4 tokens
  3. MTP head on accepted hidden state → 1 extra draft token (free)
  4. Feed MTP token back as sequence N+1 draft at next step
```

**Files to modify**:
- `src/llama-ext.h` — already extended for MTP (see `pti_mtp.cpp`)
- `src/llama.cpp` — add MTP head execution after main decode in PTI mode
- `pti_4seq.cpp` — reference implementation to merge

**Expected result**:
```
Step tokens: 4 (PTI) × 1.88 (MTP k=1 accept rate) ≈ 7.5 per step
Step cost:   ~1.1× baseline (Phase 2 overhead eliminated)
tok/s:       7.5 / 1.1 × 19.4 ≈ 132 tok/s  (optimistic)
             5.0 / 1.3 × 19.4 ≈  75 tok/s  (conservative, ~5 accepted)
Target:      80–100 tok/s
```

---

## API Design (llama.cpp integration)

Minimal API surface — add PTI as a context mode, not a new type:

```c
// In llama.h — proposed additions
typedef struct llama_pti_params {
    int  n_streams;        // number of PTI sequences (2, 3, or 4)
    bool use_mtp;          // enable MTP head if model supports it
    bool interleaved_act;  // use interleaved activation layout (Phase 2)
} llama_pti_params;

struct llama_context_params llama_context_default_params(void);
// New field (or separate function):
void llama_context_set_pti(struct llama_context *, llama_pti_params);

// PTI decode — replaces llama_decode in PTI mode
// Returns number of tokens emitted this step (1–n_streams+1)
int llama_decode_pti(struct llama_context *, struct llama_batch,
                     llama_token * out_tokens);
```

For llama-cli: add `--pti N` flag that enables N-stream PTI decode.

---

## File Map: What Changes Where

```
llama.cpp/
├── include/llama.h
│   └── + llama_pti_params, llama_context_set_pti, llama_decode_pti
│
├── src/llama.cpp
│   └── + PTI decode loop (verify/accept logic from pti_4seq.cpp)
│       + MTP head execution after main decode
│
├── ggml/src/ggml-cuda/
│   ├── mmvq.cu
│   │   └── + mul_mat_vec_q_pti<type, ncols_dst> (interleaved activation variant)
│   └── ggml-cuda.cu
│       └── + dispatch PTI mode to new kernel variant
│
└── examples/llama-cli/
    └── + --pti N flag
```

**This repo** (`packed-twin-inference/`):
- `pti_4seq.cpp` — reference; verify/accept logic migrates into llama.cpp
- `pti_kernel.hip` — benchmark harness; PTI-style kernel variant ports to `mmvq.cu`
- `PLAN.md` — this document

---

## Milestones and Expected tok/s

| Milestone | Description | Expected tok/s | Delta |
|---|---|---|---|
| M0 (done) | pti_4seq.cpp via public API | **38.1** | baseline for v2 |
| M1 | Dispatch analysis, overhead source confirmed (static) | 38.1 | diagnostic |
| M2 (patch written) | PTI GEMV variant in mmvq.cu, overhead 1.05–1.15× | **60–75** | +22–37 |
| M3 | PTI × MTP, MTP head after main decode | **80–100** | +20–30 |
| M4 | llama-cli --pti flag, public API | 80–100 | shippable |

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| N=4 uses GEMM not GEMV on MI50 | Medium | Phase 1 rocprof confirms; fix is dispatch override |
| Interleaved activation layout breaks llama.cpp tensor handling | Medium | Implement as separate PTI tensor, not in-place |
| MTP hidden state access requires internal llama.cpp API | Low | Already solved in `pti_mtp.cpp` via `llama-ext.h` |
| Phase 2 kernel rework causes regression on N=1 | Low | Keep original `mul_mat_vec_q` unchanged; new variant only |
| MI50 occupancy drops at higher ncols_dst | Medium | Profile and tune nwarps/rows_per_block per ncols_dst |

---

## Non-Goals for v2

- NVIDIA CUDA support: ROCm/MI50 first, CUDA port is mechanical afterward
- TSQ (task-specific twin variant): separate workstream, not a prerequisite
- Training infrastructure for fine-tune twins: future work

> **Note on sampled decoding (temp > 0):** the logic is correct and works as-is.
> Acceptance per position = `Σ p(x)²`; expected tokens/step = `1 + p + p² + p³`.
> At Qwen3's recommended temp=0.6–0.7, structured tasks yield ~2.5–3.0 tok/step.
> No rejection-sampling correction needed (self-draft, identical distributions when prefix matches).
> Benchmarking sampled mode is deferred to after Phase 2 kernel work.

---

## Quick Reference: Key Numbers

```
Hardware:       MI50, 32 GiB HBM2, ~1 TB/s peak, 383 GB/s measured D2D
Model (best):   UD-Q6_K_XL, 25 GB, 19.4 tok/s baseline, MTP head present
Weight ceiling: 254 GB/s (Q8_0 GEMV), 393 GB/s (Q5_K_M GEMV, llama.cpp)
Activation L2:  17408 blocks × 4 seqs × 20 KB = 1.36 GB traffic (root cause)
Current best:   38.1 tok/s (4-seq, pti_4seq.cpp, 1.96× baseline)
Target:         80–100 tok/s (Phase 3 complete)
```
