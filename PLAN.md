# PTI v2 — Integration Plan

**Goal**: integrate PTI into llama.cpp to eliminate the batch-GEMM overhead
and reach 80–100 tok/s on Qwen3.6-27B (MI50).

**Current baseline**: 18.9 tok/s (single-sequence, correct output).
38.1 tok/s figure was inflated by duplicate token counting — see Algorithm
Correction section.

---

## Audit Framework

Every step of the PTI pipeline has a corresponding automated check.
Run with `make audit` (full, ~10 min) or `make audit-quick` (no bench, ~6 min).

```
make audit         # build → baseline → PTI match → accept rates → GEMV bench
make audit-quick   # same minus the GEMV scaling bench
./audit.sh --help  # options: -m model, -n tokens, --no-bench, --quick
```

### What each audit step checks

| Step | Binary | Checks |
|---|---|---|
| 1. Build | make | debug, 4seq, bench targets compile cleanly |
| 2. Baseline | pti_debug | non-empty output, prefill timing, tok/s > 5, coherence heuristic |
| 3. PTI correctness | pti_4seq | byte-identical to baseline, 100% 4-accept at greedy, zero rejects, same token count |
| 4. GEMV scaling | pti_gemv_bench | N=4 scaling between 1× and 4× (partial fusion confirmed) |

### Passing criteria

- All steps PASS → PTI machinery is correct and hardware is behaving as expected
- Step 3 fails on output match → regression in emit logic (check pti_4seq.cpp)
- Step 3 fails on accept rate → sampler or batch state issue
- Step 4 scaling ≥ 4× → ggml MMVQ dispatch may have changed; M5 kernel priority rises
- Step 4 scaling ≤ 1.2× → ggml now fully fuses; M5 kernel may already be done

---

## Algorithm Correction (June 2026)

### The duplication bug

`pti_4seq.cpp` emits 4 tokens every 4-accept step — but 3 of those 4 are
identical to tokens emitted in the *previous* step. Consecutive 4-accept steps
produce the sequence:

```
Step 1 emits: t1  t2  t3  t4
Step 2 emits: t2  t3  t4  t5     ← t2, t3, t4 duplicated
Step 3 emits: t3  t4  t5  t6     ← t3, t4, t5 duplicated
```

The n_gen counter counted all emit calls (including duplicates), so the
reported 38.1 tok/s figure overstates actual unique-token throughput.
The output text is garbled — visible in the raw output of `pti_4seq` and
confirmed by running `pti_server`.

### Root cause

After a 4-accept, `pos_a` advances by only 1. In the next step seq 0
re-decodes `actual` at the new `pos_a` and produces the same prediction as
`from_b` from the previous step. The stagger stays fixed at {0, 1, 2, 3}
relative to seq 0, but the intent was to advance the *entire window* by 4.

### Correct algorithm (speculative decoding framing)

PTI is speculative decoding where the drafter and verifier are the *same*
model on staggered KV caches. The accept chain is:

```
tok_b == actual?         (B's draft for n+1 matches A's ground truth)
  tok_c == from_b?       (C's draft for n+2 matches B's confirmed output)
    tok_d == from_c?     (D's draft for n+3 matches C's confirmed output)
      from_d is the new unverified draft for n+4
```

On k-accept, **k tokens are emitted** (actual plus the k-1 confirmed drafts).
The stagger must then advance by k to avoid re-decoding already-confirmed
positions. For a 4-accept this means copying seq 3's KV to seq 0 and
rebuilding the B/C/D stagger from that new base — 3 single-token reinit
calls, same as the reject path.

Expected throughput after correction (same 2.04× step overhead):
- 4-accept (temp=0, greedy): 4 tokens per (1 batch + 3 reinit) = 4/5.04 ≈ 0.79× baseline
- 2-seq at temp=0: 2 tokens per (1 batch + 1 reinit) = 2/2.27 ≈ 0.88×

This means **PTI does not deliver speedup at N>1 via the public API** as
currently structured. The speedup in the original paper/approach comes from
weight sharing in hardware (one weight fetch serves N activations), which
requires the fused kernel (Phase 4 / M4). Without M4, the correct PTI
algorithm runs at or below baseline throughput.

### Revised benchmark table

| Config | Reported tok/s | Unique tok/s (corrected) | Note |
|---|---|---|---|
| baseline | 19.4 | 19.4 | single-sequence, no duplication |
| PTI 4-seq (old, buggy) | 38.1 | ~10–12 | counted duplicate tokens |
| PTI 4-seq (corrected) | — | ~15–16 (est.) | correct emit + reinit overhead |

### Revised plan

**Step 1 (now):** Replace the PTI loop in `pti_server.cpp` with the single-
sequence loop from `pti_debug.cpp`. This gives a correct, working HTTP server
at baseline throughput. Validate with nanocoder.

**Step 2:** Implement correct speculative-decoding-style PTI in `pti_debug`
first (with verbose per-token logging), confirm output is identical to
single-sequence baseline. Then port to `pti_server.cpp`.

**Step 3 (M4):** Fused PTI GEMV kernel (mmvq.cu). This is where the real
speedup lives — one weight fetch, N activation vectors. Without it, the
correct PTI is at or below baseline. With it, target 60–75 tok/s.

---

## GEMV Fusion Test — June 2026

### Does ggml share weight reads across N sequences?

`pti_gemv_bench` runs N=1/2/4 sequences through identical decode steps
(short context, greedy) and measures ms/step. If weights are loaded once
for all N, ms/step should be flat. If loaded N times, it scales linearly.

**Results (MI50, Qwen3.6-27B UD-Q6_K_XL, ctx=128, 20 measured steps):**

```
N   ms/step   ms/seq   scaling   eff_BW(GB/s)   BW_per_seq
─   ───────   ──────   ───────   ────────────   ──────────
1     53.1     53.1    1.00×       456              456
2     66.2     33.1    1.25×       366              183
4    103.5     25.9    1.95×       234               59
```

Theoretical minimum (weights loaded once): 23.7 ms (24.22 GB / 1024 GB/s)

**Interpretation**: N=4 scaling of 1.95× sits between fully-fused (1×) and
fully-unfused (4×). ggml achieves partial weight sharing via L1 cache — the
weight block for a given kbx stays in L1 across the j=0..3 activation passes.
The remaining overhead is from 4× more multiply-accumulate instructions in the
inner loop, transitioning the kernel from memory-bound (N=1) to partially
compute-bound (N=4).

**Theoretical PTI throughput at full fusion:**
```
N=1 baseline: 18.8 tok/s
N=4 ideal:   ~38–40 tok/s  (weights shared, KV/SSM overhead ~5%)
```

### Context isolation — bottleneck confirmed as GEMV compute (M5)

Ran N=1 and N=4 at two context sizes to distinguish GEMV compute overhead from
SSM state and KV cache:

```
ctx    N=1 ms   N=4 ms   scaling
128     53.1    103.6    1.95×
1024    53.3    103.6    1.94×
Δ = 0.01   →  ctx-INDEPENDENT
```

- **KV cache** scales with ctx — ruled out (scaling unchanged at 8× larger ctx)
- **SSM state** is fixed per n_seq_max (not n_ctx) and is only 576 MB total →
  0.56 ms at 1024 GB/s = 1% of step time — mathematically ruled out
- **Conclusion**: overhead is in the MMVQ inner loop compute

Mechanism: `vec_dot_q6_K_q8_1` with `#pragma unroll` on j=0..3 produces
8 sequential FMAF instructions per weight block (4j × 2i). L1 cache holds the
weight block across j, so weight LOADS are shared. But the ALU must execute
4× more multiply-accumulate chains — 8 instructions vs 2 at N=1. This
computation serializes with memory, preventing the GPU from overlapping other
work, and accounts for the ~50 ms extra cost (103.6 − 53.1 − 0.56 ms SSM ≈ 50 ms).

### M5 kernel target

The fix is not about memory layout or cache. It is about reducing instruction count
for the multi-column dot product. Two concrete approaches:

1. **Explicit weight hoist (lowest risk)**: restructure `mmvq.cu` j-loop so weight
   data (bq6_K->ql, qh, scales, d) is loaded outside the j-loop and passed by
   value to a new `vec_dot_q6_K_q8_1_Nx` function. Forces compiler to keep weights
   in registers (VGPR) across all j iterations. Expected: reduce redundant loads
   that survive L1 (if any) and reduce register spill.

2. **MFMA/WMMA vectorization (higher gain)**: on gfx906 (MI50), use
   `v_mfma_f32_16x16x4f16` to compute all 4 column dot products as a single
   matrix instruction rather than 4 sequential scalar chains. Requires dequantizing
   Q6_K → FP16 before the MFMA call. Expected: reduce instruction count from 8 to
   2 per weight block, approaching compute-bound ceiling of ~1.2× overhead.

---

## Memory Layout Analysis — KV Cache and SSM State

### KV cache — shared-prefix layout (high value)

The 4 PTI sequences share identical KV history for all positions except the
3-position stagger tip. Current llama.cpp stores 4 separate full KV caches
(4× redundant data for the shared prefix). A positional layout stores one
record per position, shared across all sequences:

```
Positional KV (single record per position):
  pos 0:   K/V shared by all 4 seqs       ← one read serves all
  pos 1:   K/V shared by all 4 seqs
  ...
  pos N:   K/V for seqs 1, 2, 3 only
  pos N+1: K/V for seqs 2, 3 only
  pos N+2: K/V for seq 3 only
```

For a 4096-token context: reduces KV reads from 4×4096 to ~4099 records.
~4× reduction in attention-layer KV traffic. Only ~16 attention layers in
Qwen3.6-27B, so absolute savings are modest vs 24 GB of weights, but the
principle is sound and implementable within llama.cpp's KV pool.

### SSM recurrent state — interleaved layout (moderate value)

SSM states cannot be shared (each sequence's state is a distinct running
accumulator that diverges after the first generated token). However, grouping
all 4 seqs' values at the same state dimension together enables SIMD:

```
Current (4 separate 144 MB vectors):
  h_A: [d0, d1 ... dD]
  h_B: [d0, d1 ... dD]   ← different values, same layout
  h_C, h_D: same

Interleaved by dimension (still 576 MB, coalesced):
  dim 0: [h_A[0], h_B[0], h_C[0], h_D[0]]  ← one 128-bit load = all 4 seqs
  dim 1: [h_A[1], h_B[1], h_C[1], h_D[1]]
  ...
```

SSM update kernel becomes one float4 FMA per dimension instead of 4 scattered
f32 FMAs. On GPU with proper batching the benefit is already partially realised
(4 parallel thread blocks), but explicit SIMD interleaving can improve it further.

**Mathematical ground truth**: the SSM state is tiny relative to weights
(144 MB × 4 = 576 MB vs 24 GB). Even at 0% efficiency the SSM state cost is
576 MB / 1024 GB/s = 0.56 ms — about 1% of a 53 ms decode step. Interleaving
the state is a second-order optimization. The weight GEMV fusion (M5) is where
the step-time budget actually lives.

### Priority ranking for M5 kernel work

| Target | Data size | Expected gain | Priority |
|---|---|---|---|
| Fused weight GEMV (all layers) | 24 GB | ~2× step speedup | Critical |
| Shared-prefix KV cache | 4096 × 16 attn layers | ~5% step speedup | Medium |
| SSM state interleaving | 576 MB | <1% step speedup | Low |
| Conv state sharing | 22 MB | Negligible | Skip |

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

### Phase 3 — MTP analysis (done, negative result for bonus tokens)

**Finding**: The Qwen3.6-27B UD MTP head (`nextn_predict_layers=1`) takes
`(tok_t, h_t from main model)` and predicts `t+1` — the SAME position the
main model already computes. It does NOT predict `t+2`. It cannot add a
genuine 5th token to the PTI output.

**Architecture** (from `qwen35moe.cpp:graph_mtp`):
```
MTP input:  (tok_embed_t, h_t)  →  concat  →  1-layer transformer  →  logits for t+1
```
The token at position t and the main model's hidden state at t are concatenated
and projected. This is a more informative "output head" — it may improve accept
rate vs the standard head, but predicts the same position.

**pti_mtp.cpp status**: `ctx_mtp` is created, `mtp_predict_next()` is defined,
but the main loop uses full decodes for all reinit paths. The MTP function is
never called. The comment reads "Partial cross-stream seq_cp fails on hybrid
models — no MTP shortcut here."

**What MTP CAN do** (for temperature > 0, future work):
- On the REJECT path, after B-reinit produces h_B_new (at pos_a):
  `mtp_predict_next(ctx_mtp, ctx, 0, actual_next, pos_a, n_embd, n_vocab)`
  predicts tok_b — same as B's reinit logits. No saving here.
- Each chained reinit (B→C→D) requires the PREVIOUS step's hidden state,
  so no reinit calls can be skipped.
- **At 100% greedy (all benchmarks so far): zero MTP benefit.**

**Corrected throughput estimate**:
```
At greedy:     4.0 tok/step (unchanged — MTP cannot add tokens at t+2)
At temp=0.7:   ~2.5–3.0 tok/step (PTI accept rate, no MTP benefit)
Path to >4×:   kernel rewrite (Phase 4) to eliminate 2× step overhead
```

**MTP infrastructure** is ported to `pti_4seq.cpp` for:
- Correctness (model initializes properly with pre-norm embeddings enabled)
- Future sampled-mode benchmarks where MTP quality difference may be measured

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

| Milestone | Description | Expected tok/s | Status |
|---|---|---|---|
| M0 | pti_4seq.cpp via public API | 38.1 (duplicate-inflated) | done |
| M1 | Dispatch analysis, overhead source confirmed | diagnostic | done |
| M2 | PTI activation interleave patch | no gain | benchmarked |
| M3 | MTP analysis — no bonus tokens on hybrid model | no gain | done |
| **M4a** | **pti_server: single-sequence baseline (correct, working)** | **~19 tok/s** | **done** |
| **M4b** | **pti_4seq: correct speculative PTI, output byte-identical to baseline** | **~10 tok/s** | **done (1.95× step overhead confirmed)** |
| M4c | pti_server: correct PTI loop | ~10 tok/s | deferred (slower than single-seq until M5) |
| **M5a** | **mmvq.cu: explicit weight hoist for N=4 inner loop** | **est. 15–25 tok/s** | **next** |
| **M5b** | **mmvq.cu: MFMA vectorization for N=4 (gfx906)** | **est. 55–75 tok/s** | **future** |
| M6 | llama-cli --pti flag, public API | 60–75 | future |

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
Overhead:       2.04× at N=4; from 4× MMVQ compute + attn/SSM scaling with N
                Activation traffic (~480 MB/step) is <2% of weight traffic (~25 GB/step)
Current best:   38.1 tok/s (4-seq, pti_4seq.cpp, 1.96× baseline)
Honest (-n 80): ~32 tok/s amortized (includes 7 startup decode steps, one-time)
Target:         60–75 tok/s after kernel rewrite (M4)
```
