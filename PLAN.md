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

**The overhead is not the kernel — it's the dispatch.** llama.cpp uses its
fast GEMV path (`mul_mat_vec_q` in `mmvq.cu`) for N=1, but routes N=4 through
a different code path with higher overhead. The GEMV kernel itself already
supports multi-column output via the `ncols_dst` template parameter.

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

### Phase 1 — Measure overhead source (1–2 days)

**Goal**: confirm exactly what causes the 2.04× overhead at N=4.

1. Add `--pti 4` flag to `llama-cli` that passes `n_seq_max=4` and submits
   4-token batches. This mirrors what `pti_4seq.cpp` does but from inside llama.cpp.

2. Run rocprof on the 4-seq decode loop and identify which kernel consumes
   the extra 1.04× step time vs baseline:
   ```bash
   rocprof --stats ./llama-cli --pti 4 -m ../gguf/Qwen3.6-27B-UD-Q6_K_XL.gguf \
           -p "The key to faster LLM inference is" -n 80 -ngl 99
   ```

3. Expected findings:
   - If `mul_mat_vec_q` is called with `ncols_dst=4` → overhead is within kernel
   - If GEMM kernel appears instead → dispatch threshold is the fix

**Deliverable**: profiler trace pinpointing the overhead. Determines which fix to implement.

**Expected result from this phase alone**: 0 extra tok/s, but unblocks Phase 2.

---

### Phase 2 — Eliminate overhead (3–5 days)

**Option A (likely)**: overhead is within `mul_mat_vec_q` at `ncols_dst=4`.

The GCN table uses `nwarps=2` for both N=1 and N=4. The kernel processes
`rows_per_block=2` at N=4 but `rows_per_block=1` at N=1. Each row-block
reads all 4 activation vectors once per K iteration, so L2 pressure grows 4×.

Fix: add a PTI-specific kernel variant that reads 4 activation vectors as a
single `float4` (interleaved layout), matching the `pti_linear4_q8_intlv` approach
already benchmarked at 130 GB/s in `pti_kernel.hip`:

```cpp
// In mmvq.cu — new kernel variant
template <ggml_type type, int ncols_dst, bool pti_interleaved>
__global__ void mul_mat_vec_q_pti(...)
{
    // Existing weight load (unchanged)
    // Activation load: if pti_interleaved, use float4 from [K × ncols_dst] layout
    float4 xv = ((const float4*)vy_interleaved)[k];
    // xv.x=xa, xv.y=xb, xv.z=xc, xv.w=xd — ONE VMEM instruction for 4 streams
}
```

**Option B (less likely)**: dispatch routes N=4 to GEMM.

Fix: raise `MMVQ_MAX_BATCH_SIZE` or override the dispatch for PTI mode to
force the GEMV path for all N ≤ 8.

**Files to modify**:
- `ggml/src/ggml-cuda/mmvq.cu` — kernel variant
- `ggml/src/ggml-cuda/ggml-cuda.cu` — dispatch if needed
- `src/llama.cpp` — PTI context mode (sets interleaved activation layout)

**Expected result**: step overhead drops from 2.04× → 1.1–1.3×.
```
4 / 1.15 × 19.4 ≈ 67 tok/s  (conservative, 1.15× overhead)
4 / 1.05 × 19.4 ≈ 74 tok/s  (optimistic, 1.05× overhead)
```

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
| M1 | rocprof trace, overhead source confirmed | 38.1 | diagnostic |
| M2 | PTI GEMV variant in mmvq.cu, overhead 1.1–1.3× | **60–75** | +22–37 |
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
- Sampled decoding (top-p): greedy 100% accept is the baseline; sampling is additive
- TSQ (task-specific twin variant): separate workstream, not a prerequisite
- Training infrastructure for fine-tune twins: future work

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
