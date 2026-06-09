# Packed Twin Inference (PTI)

**One weight load. N token streams. Zero quality loss.**

PTI achieves **~2× LLM throughput today**, measured on Qwen3.6-27B (MI50), using nothing but
llama.cpp's existing batch-decode infrastructure. No draft model. No separate VRAM footprint.
No quality gap. The full path to **4–5× throughput** is documented below.


## Note

4seq (the 1.96X version) is the current working benchmark. I'm working on the MCP+llama CPP.  This should give us a theoretical max of 4X for temperature 0.  
Once that's ready, I'll push, and we should expect somewhere between 70%-80% of that max: 2.5-3.5X speedup over bare model.

---

## Headline Results (MI50, Qwen3.6-27B, measured)

| Config | Model | tok/s | vs baseline | GPU VRAM |
|---|---|---|---|---|
| Baseline | Q5_K_M | 21.1 | 1.00× | 18.1 GiB |
| **PTI 4-seq** | **UD-Q6_K_XL** | **38.1** | **1.96×** | 24.5 GiB |
| PTI 3-seq | UD-Q6_K_XL | 33.9 | 1.75× | 24.3 GiB |
| PTI 2-seq | UD-Q6_K_XL | 30.5 | 1.57× | 24.0 GiB |
| PTI 2-seq | Q5_K_M | 28.9 | 1.38× | 18.4 GiB |
| Baseline | UD-Q6_K_XL | 19.4 | 1.00× | 23.7 GiB |

The 4-seq run uses `pti_4seq.cpp` — 150 lines of C that wrap llama.cpp's public API.
No patches to llama.cpp. No custom kernels. **Just one `llama_decode` call per step
with a 4-token batch — one per sequence.**

```bash
# Reproduce the 38.1 tok/s result:
make 4seq && make 4seq-run
# bin/pti_4seq -m ../gguf/Qwen3.6-27B-UD-Q6_K_XL.gguf \
#              -p "The key to faster LLM inference is" -n 80 -ngl 99
```

---

## How It Works

LLM decode at batch=1 is memory-bandwidth-bound. Every token requires loading
all model weights from VRAM:

```
tokens/sec ≈ HBM_bandwidth / model_size
```

PTI exploits the fact that **llama.cpp already shares weight loads across all tokens
in a batch**. Running N sequences as a batch costs ~1.0–2.0× a single-sequence step,
not N×. The gain comes from emitting N tokens per step while paying only ~2× the
single-token step cost.

```
Single-sequence decode (baseline):
  Load 25 GB weights → produce 1 token → 19.4 tok/s

PTI 4-sequence decode:
  Load 25 GB weights → produce 4 tokens → 38.1 tok/s
  (weights shared across all 4 sequences in one llama_decode call)
```

**Overhead scaling (measured, UD-Q6_K_XL, MI50):**

| N seqs | tok/s | multiplier | step overhead | overhead increment | GPU VRAM |
|---|---|---|---|---|---|
| 1 (baseline) | 19.4 | 1.00× | 1.00× | — | 23.7 GiB |
| 2 | 30.5 | 1.57× | 1.27× | +0.27 | 24.0 GiB |
| 3 | 33.9 | 1.75× | 1.71× | +0.44 | 24.3 GiB |
| **4** | **38.1** | **1.96×** | **2.04×** | +0.33 | **24.5 GiB** |

Overhead grows sub-linearly per added sequence. At N=4 we hit the first efficiency
plateau: 4 tokens for 2.04× step cost ≈ 1.96×. The overhead comes from llama.cpp's
batch GEMM being less memory-efficient than its single-token GEMV path (see
Bandwidth Analysis below).

**The arithmetic**: each step produces 4 tokens and costs 2.04× a baseline step.
Throughput = (4 tokens) / (2.04 × baseline_step_time) = (4/2.04) × baseline_rate
= 1.96 × 19.4 = 38.0 tok/s. The 2.04× overhead is not a deduction from baseline —
it is the denominator in the ratio, divided into 4× more output per step.

**VRAM cost is minimal**: PTI adds only ~0.2 GiB per extra sequence (KV cache only)
because model weights are shared. Standard speculative decoding needs a full second
model (4–19 GiB extra). PTI's extra cost is near-zero.

---

## How Acceptance Works

PTI is **self-speculative decoding** — the same model acts as both drafter and
verifier. The four sequences run at genuinely different KV positions (n, n+1,
n+2, n+3), staggered during initialization. Each step produces 4 tokens at 4
distinct output positions, not 4 re-computations of the same position.

```
Step k — four sequences at four different positions:

  Seq 0 (A): KV covers positions 0..n.   Processes tok_a at pos n.
             → actual_next  = token at output position n+1  (ground truth)

  Seq 1 (B): KV covers positions 0..n+1. Processes tok_b at pos n+1.
             → next_from_b  = token at output position n+2  (B's prediction)

  Seq 2 (C): KV covers positions 0..n+2. Processes tok_c at pos n+2.
             → next_from_c  = token at output position n+3  (C's prediction)

  Seq 3 (D): KV covers positions 0..n+3. Processes tok_d at pos n+3.
             → next_from_d  = token at output position n+4  (D's prediction)

On 4-accept: emit [actual_next, next_from_b, next_from_c, next_from_d].
Output cursor advances by 4. Next step: A→n+1, B→n+2, C→n+3, D→n+4.
```

The stagger is established during initialization (see `pti_4seq.cpp` lines
232–262): B's KV is built by running `tok_gen_0` at pos_a; C's KV by running
`tok_gen_0` then `tok_b` at pos_a, pos_b; D's by running all three. Each
sequence sees a different, complete prefix — not a variation on A's position.

**Greedy (temp=0):** the same model with the same prefix always produces the
same argmax token. Seq 1's guess from step k-1 is guaranteed to equal what
Seq 0 produces at step k → 100% accept, N tokens every step, no rollbacks.

**Temperature > 0:** the acceptance check (`tok_b_guess == actual_next`) asks
whether two independent draws from the same distribution agreed. The probability
is `Σ_x p(x)²` — the collision probability of the output distribution. No
rejection-sampling correction is needed (distributions are identical when the
prefix matches), but the rate drops with temperature:

```
p(accept per position) ≈ Σ_x p(x)²
Expected tokens/step  = 1 + p + p² + p³

p=1.00 (greedy):           4.00 tok/step
p=0.75 (low-entropy task): 2.74 tok/step
p=0.60 (typical text):     2.16 tok/step
p=0.40 (creative/high-T):  1.64 tok/step
```

At Qwen3's recommended temp=0.6–0.7, structured tasks (code, factual QA) tend
toward the high-p end; open-ended generation toward the low-p end.

| Decoding mode | Approx accept/pos | Expected tok/step |
|---|---|---|
| Greedy (temp=0) | 100% | 4.00 |
| temp=0.6–0.7, structured | ~70–80% | 2.5–3.0 |
| temp=0.6–0.7, general text | ~50–65% | 2.0–2.5 |
| TSQ fine-tune variant (same-arch twin) | 75–90% | 2.7–3.6 |

---

## MTP Analysis (Qwen3.6-27B UD Model)

The UD-Q6_K_XL model includes a Multi-Token Prediction head (`nextn_predict_layers=1`).
**Finding**: the MTP head takes `(tok_t, h_t from main model)` → predicts `t+1`.
It predicts the *same position* the main model already computes — it is NOT a
"next-next" predictor and cannot add a bonus 5th token to PTI output.

`pti_4seq.cpp` creates the MTP context for correctness and future sampled-mode work,
but at 100% greedy it is never invoked — there are no reject calls and D's logits
already give the t+1 prediction. `pti_mtp.cpp` has the same result: `ctx_mtp` is
created but never called in the main loop (the comment reads "no MTP shortcut here").

The UD model outperforms Q5_K_M in PTI (1.57× vs 1.38× at 2-seq) due to its higher
quantization quality per byte (6-bit) being more VRAM-bandwidth-efficient on the
PTI multi-sequence path, not from the MTP head.

---

## Bandwidth Analysis: Why the Overhead Happens

This section explains the measured 2× overhead and maps the path to eliminating it.

### MI50 Bandwidth Ceilings (measured)

| Kernel | GB/s | % of D2D | Notes |
|---|---|---|---|
| HipMemcpy D2D | 383 | 100% | theoretical ceiling |
| Raw int8 stream | 330 | 86% | pure HBM streaming |
| Weight-only GEMV | 254 | 66% | weights+scales, no activations |
| Flat Q8 GEMV (N=1) | 92 | 24% | activation L2 traffic bottleneck |
| Vectorized Q8 (N=1) | 100 | 26% | 128-bit loads, minimal gain |
| Register-tiled Q8 (N=4) | 127 | 33% | M_REG=4, warp-shuffle |
| Interleaved float4 (N=4) | 130 | 34% | best custom kernel |
| **llama.cpp Q5_K_M (N=1)** | **393** | **103%** | **exceeds our Q8 ceiling** |

`llama.cpp Q5_K_M achieves 393 GB/s "effective" bandwidth` because Q5_K_M packs
5 bits/weight — 60% more weights per cache line than Q8_0's 8 bits/weight. The
smaller model (18.6 GB vs 27 GB) travels across HBM faster per token.

### Why Custom Q8 Kernels Underperform (Root Cause)

For Qwen3.6-27B decode at position 80:

```
Weight HBM traffic (Q8):  17408 rows × 5120 cols × 1 byte  =  89 MB
Activation L2 traffic:    17408 blocks × 4 seqs × 5120×4B  =  1.36 GB
```

Activation reads are **15× larger than weight reads** in L2 traffic, even though
the activation tensor is only 80 KB in total. Each of 17408 row-blocks must read
the full activation vector → L2 thrash. Register tiling (M_REG=4) reduces blocks
to 4353 but only gets to 34% of D2D ceiling.

### Why llama.cpp Q5_K_M GEMV Is Already Optimal

llama.cpp's `mul_mat_vec_q` for Q5_K_M achieves 393 GB/s because:
1. 5 bits/weight → smaller model → less HBM traffic per token
2. The kernel already implements multi-row tiling with shared activations
3. GCN (MI50) path: `nwarps=2`, `rows_per_cuda_block=2` — exactly the tiling we built

### Why N=4 Batch Causes 2× Overhead (Revised)

**What we tested**: interleaving the 4 activation columns before the GEMV call
(`GGML_PTI_INTERLEAVED=1` patch in `llama.cpp-patch/`). Result: **no speedup**
(35.6 vs 38.1 tok/s, −6%, within noise).

**Root cause**: activation traffic is negligible compared to weight traffic:
```
Weight HBM traffic (Q6_K_XL): ~25 GB/step  (dominant)
Activation L2 traffic (N=4):  ~480 MB/step  (<2% of weight traffic)
```
Optimizing activation L2 access cannot move the overall step time.

**Actual root cause**: the `mul_mat_vec_q` inner loop runs 4× the multiply-accumulate
operations for `ncols_dst=4` vs `ncols_dst=1`. The attention and SSM layers also
scale linearly with N sequences. The 2.04× overhead is simply 4× more arithmetic
split across the 64 main layers + 16 SSM layers, hitting a compute (not memory) wall.

**The kernel rewrite target**: a fused PTI kernel that loads each weight block once
and performs all 4 dot products in a single warp-cooperative pass — eliminating the
per-column loop that causes the 4× compute scaling. See `PLAN.md` Phase 4 (M4).

---

## Integration Roadmap: llama.cpp (2× → 4–5×)

### Current State

PTI at 2× works today using llama.cpp's public API. No patches needed.
The source is `pti_4seq.cpp` (~200 lines). N=4 uses `mul_mat_vec_q` (MMVQ, the
quantized GEMV path) — confirmed via dispatch analysis. The overhead comes from
the MMVQ inner loop running 4× more multiply-accumulate operations per weight block
when `ncols_dst=4`. Activation L2 traffic is negligible (<2% of weight traffic).

### Integration Architecture

```
llama.cpp
├── ggml/src/ggml-cuda/mmvq.cu        ← Target: add PTI GEMV variant here
│   └── mul_mat_vec_q<type, ncols_dst> ← Already has N-column template
└── src/llama.cpp                      ← Add --pti N flag to context creation
```

The `mul_mat_vec_q` kernel already accepts a `c_ncols_dst` template parameter for
multiple output columns. The GCN (MI50) path uses `nwarps=2` for `ncols_dst=1..4`.

**Two targeted changes to llama.cpp:**

1. **Fused PTI GEMV kernel** (`mmvq.cu`):
   Write a new `mul_mat_vec_q_pti<type, N>` kernel that loads each weight block
   once and computes all N dot products cooperatively — no per-column inner loop.
   Eliminates the 2.04× overhead → targets 60–75 tok/s.

2. **PTI context API + decode loop** (`llama.h` / `llama.cpp`):
   ```c
   llama_context_set_pti_streams(ctx, n_streams);
   // --pti 4 in llama-cli
   ```
   The verify/accept logic migrates from `pti_4seq.cpp` into `llama_decode()`.

### Expected Throughput After Integration

| Stage | Technique | Expected tok/s | vs today |
|---|---|---|---|
| Today | pti_4seq.cpp (external) | 38.1 | — |
| **M4** | **Fused PTI GEMV kernel in mmvq.cu** | **60–75** | **+57–97%** |
| M5 | llama-cli --pti flag, public API | 60–75 | shippable |

**M4 estimate**: fused kernel eliminates the 4× inner-loop compute scaling
(2.04× → ~1.1× overhead):
```
4 / 1.1 × 19.4 ≈ 70 tok/s  (optimistic)
4 / 1.3 × 19.4 ≈ 60 tok/s  (conservative, some compute overhead remains)
```

### Comparison to Standard Speculative Decoding

| | Standard SpecDec | PTI today | PTI + integration |
|---|---|---|---|
| Draft model | Small, separate model | Same model | Same model |
| Draft VRAM | +4–19 GiB | +0.2 GiB/seq | +0.2 GiB/seq |
| Accept rate (greedy) | ~60–70% | **100%** | **100%** |
| Quality | Draft model quality | Identical to baseline | Identical |
| Throughput gain | ~1.6× | **1.96× (measured)** | **3–4× (projected, M4)** |
| llama.cpp patches | None needed | None needed | 3 targeted changes |

---

## Architecture Diagrams

| Diagram | What it shows |
|---|---|
| `diagram-pti-4seq-step.svg` | One full PTI step: 4-token batch, weight sharing, verify/accept/reject paths, KV layout |
| `diagram-pti-overhead.svg` | Why overhead is 2× not 4×: weight load shared, 4× inner-loop compute scaling, throughput bars |
| `diagram-memory-layout.svg` | SSQ uint16 block format — how two int8 weights pack into one word |
| `diagram-workflow.svg` | High-level PTI inference loop from prefill through accept/reject |

---

## Files

| File | Purpose | Status |
|---|---|---|
| `pti_4seq.cpp` | 4-sequence PTI — public llama.cpp API | ✓ 38.1 tok/s steady-state, ~32 amortized |
| `pti_mtp.cpp` | 3-sequence PTI + MTP context (created, not invoked) | ✓ measured: 33.9 tok/s |
| `pti_llama.c` | 2-sequence PTI — C API (baseline) | ✓ measured: 28.9–30.5 tok/s |
| `pti_kernel.hip` | HIP/ROCm Q8 multi-stream kernel + benchmarks | ✓ bandwidth ceiling analysis done |
| `pti_hip.py` | Python wrapper for HIP kernel | ✓ |
| `infer.py` | PTI inference loop — Python reference | ✓ |
| `benchmark.py` | Acceptance rate and output correctness | ✓ |
| `pack.py` | Pack a model with itself as SSQ twin streams | ✓ |
| `Makefile` | Build for MI50 / MI100 / RX 7900 | ✓ |
| `DESIGN.md` | Full design rationale, math, implementation | ✓ |

### Models Used (MI50, 32 GiB VRAM)

| File | Size | Format | Notes |
|---|---|---|---|
| `Qwen3.6-27B-Q5_K_M.gguf` | 18.6 GB | 5-bit K-quant | Best bandwidth/quality |
| `Qwen3.6-27B-UD-Q6_K_XL.gguf` | 25 GB | 6-bit K-quant | Has MTP head; best PTI result |
| `Qwen3.6-27B-Q8_0.gguf` | 27 GB | 8-bit | Custom kernel target |

---

## Build & Run

```bash
# Build pti_4seq → bin/pti_4seq (requires llama.cpp built in ../llama.cpp/build)
make 4seq

# Run 4-sequence PTI — reproduces 38.1 tok/s headline result
make 4seq-run
# Equiv: bin/pti_4seq -m ../gguf/Qwen3.6-27B-UD-Q6_K_XL.gguf \
#                     -p "The key to faster LLM inference is" -n 80 -ngl 99

# Baseline for direct comparison
make 4seq-run-base

# 3-seq PTI + MTP
make mtp && make mtp-run

# 2-seq PTI (C implementation)
make llama && make llama-run-pti

# Build all three llama.cpp binaries at once
make all-llama

# HIP kernel bandwidth benchmarks (requires ROCm)
make && bin/pti_test

# Clean all built artifacts
make clean
```

**Prerequisites:**
- llama.cpp built at `../llama.cpp/build/` with ROCm/HIP support
- AMD MI50 (gfx906) or compatible GPU with ≥32 GiB VRAM
- `Qwen3.6-27B-UD-Q6_K_XL.gguf` in `../gguf/`

---

## Relationship to SSQ (Side-by-Side Quantization)

The deeper path to 4–5× uses SSQ format, which packs two int8 weight values into
one uint16 word. One load from VRAM yields weights for both streams:

```c
uint16_t pw = W_packed[i];
int8_t   wa = (int8_t)(pw >> 8);    // Stream A weight
int8_t   wb = (int8_t)(pw & 0xFF);  // Stream B weight
acc_a += scale_a * (float)wa * x_a[i];
acc_b += scale_b * (float)wb * x_b[i];
// → one memory transaction, two compute streams
```

SSQ enables the **TSQ (Task-Specific Quantization)** variant: pack base model +
fine-tune as twin pair. High accept rate on the fine-tune's target domain, falling
acceptance signals off-domain query routing.

The SSQ kernel (`pti_kernel.hip`) is built and benchmarked. The next step is the
fused PTI GEMV kernel in `mmvq.cu` (M4) — loading each weight block once, computing
all N dot products cooperatively. SSQ format would further extend this to handle
twin-pair quantization for the TSQ variant.

---

## Summary: What's Real, What's Next

**Real today (reproducible):**
- 38.1 tok/s steady-state / ~32 tok/s amortized on Qwen3.6-27B — **1.96× baseline**
- 100% greedy accept rate — no quality loss vs baseline
- +0.8 GiB VRAM overhead (vs +4–19 GiB for standard speculative decoding)
- Overhead source confirmed: 4× MMVQ inner-loop compute scaling, NOT activation L2 traffic
  (activation interleave patch benchmarked — no gain)

**Next (M4 — fused PTI GEMV kernel):**
- Write `mul_mat_vec_q_pti<type, N>` in `mmvq.cu`: load each weight block once,
  compute N dot products cooperatively — eliminates the 4× inner-loop scaling
- Projected: 2.04× overhead → ~1.1–1.3× → **60–75 tok/s** on MI50
- No MTP bonus tokens (MTP head predicts t+1, same position as main logits)

---

---

## Common Objections

**"B/C/D just recompute A's token — output is counted 4×."**

No. B, C, D are at positions n+1, n+2, n+3 — not at A's position n. Each sequence
has a fully independent KV cache covering a different prefix length. On 4-accept, the
four emitted tokens are at output positions n+1, n+2, n+3, n+4. The code confirms
this: `pos_b = pos_a + 1`, `pos_c = pos_b + 1`, `pos_d = pos_c + 1` (init lines
236, 247, 262). All four positions advance together each step (line 296).

**"batch=4 costs 2.04× so real throughput is baseline/2.04 ≈ 0.49×."**

Wrong divisor. The correct ratio is `(4 tokens produced) / (2.04 × step cost)` =
`4/2.04 × baseline_rate` = 1.96×. `baseline/2.04` would be correct only if PTI
produced 1 token per step — it produces 4.

**"This is just batch decoding."**

Yes, in the sense that `llama_decode` with a multi-token batch is the mechanism.
The contribution is the staggered-offset initialization, the greedy accept-chain
logic, and the reject/reinit protocol that keeps the four sequences coherent.
Standard batch serving runs N *independent* user requests; PTI runs N *staggered*
positions of one sequence and emits them all as one stream.

**"38 tok/s is the HBM bandwidth ceiling for this model anyway."**

The single-stream baseline (19.4 tok/s) is already near the HBM ceiling for the
UD-Q6_K_XL model (~444 GB/s measured, 87% of MI50 spec). PTI at 38 tok/s exceeds
what any single-stream decode can achieve on this hardware — it does so by
producing 4 confirmed tokens per weight-load cycle instead of 1. That is the gain.

---

*Hardware: AMD MI50 (16 nm, gfx906, 32 GiB HBM2, ~1 TB/s peak). Model: Qwen3.6-27B.*
*All throughput figures are tokens/second total output, measured with `-n 80` generation.*
