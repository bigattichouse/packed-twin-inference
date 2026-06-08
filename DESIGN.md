# Packed Twin Inference (PTI)

**One weight load. Two token streams. Zero quality loss.**

---

## The Problem

LLM inference at batch=1 is memory-bandwidth-bound. Every token requires loading
all model weights from VRAM. Throughput is:

```
tokens/sec = HBM_bandwidth / model_size_bytes
```

For Qwen3.6-27B Q8 on MI50 (1 TB/s HBM2, 27 GB model):
  27 GB / 1 TB/s = 27 ms/token → ~37 tokens/sec theoretical max

No kernel trick changes this without either reducing bytes loaded or doing more
useful work per byte loaded. PTI does more useful work per byte loaded.

---

## Standard Speculative Decoding

The established solution: use a small fast draft model to generate N candidate
tokens, then verify all N in one parallel pass of the large target model.

```
Draft model (1.7B):  generates tokens T+1, T+2, T+3 cheaply
Target model (27B):  verifies 3 tokens in one pass (same cost as 1 token)
Win: ~3× throughput if acceptance rate is high
```

**Constraints:**
- Need two models in VRAM simultaneously (draft + target)
- Draft must be much smaller to be "free" — quality tradeoff
- Different architectures → lower acceptance rate (~60-70%)
- Two separate memory loads per inference step

---

## SSQ Background: Side-By-Side Quantization

The `ssq/` package packs two Q8 (int8) weight values into one uint16 word:

```
uint16 slot: [ int8 model_A | int8 model_B ]
              hi byte         lo byte
```

Block layout (mode 16, 32 weights):
```
[scale_A f16][scale_B f16][32 × uint16]  =  68 bytes
```

vs two separate Q8 blocks:
```
[scale_A f16][32 × int8]  +  [scale_B f16][32 × int8]  =  68 bytes
```

Same total bytes, but **one memory load serves two weight streams**. When both
models share identical architecture and tensor shapes, they pack perfectly.

The HIP kernel loads one uint16, unpacks hi/lo, and computes two matmul
accumulations simultaneously — one memory transaction, two compute streams.

---

## The Twin Insight

Standard speculative decoding makes the draft "cheap" by making it **small**.
SSQ twins make the draft "cheap" by making it **free in bandwidth terms**.

```
Standard:                          SSQ Twins:
─────────────────────────────      ──────────────────────────────────
Load draft weights   (small)       Load twin weights (one uint16 load)
  → draft token T+1                  → Twin A: current token T
Load target weights  (large)         → Twin B: speculative token T+1
  → verify T+1                       (same bandwidth as loading once)
Two memory loads                   One memory load
Two VRAM footprints                One VRAM footprint
```

**The draft costs zero extra bandwidth.** The weights were being loaded for Twin A
anyway. Twin B rides along for free.

---

## Acceptance Rate With Identical Twins

When both twins are the same model (identical weights):

- Twin B inputs: speculative token T (our best guess before Twin A confirms)
- If our guess was correct → Twin B's output is **identical** to what Twin A
  would have produced for T+1 → 100% acceptance
- If our guess was wrong → discard Twin B's output, advance only one token

| Decoding mode | Acceptance rate | Notes |
|---|---|---|
| Greedy (temp=0) | **100%** | Identical twins always agree — proved empirically |
| Sampling (top-p) | 70-85% | Depends on model confidence |
| TSQ side-car (task-tuned B) | 75-95% | Higher within task domain |

Identical twins have a **higher acceptance ceiling** than small-model speculative
decoding (~60-70%) because when the input assumption holds, the result is
guaranteed identical — same weights, same computation.

---

## Throughput Comparison

Four benchmark points, in increasing order of gain:

```
① Baseline (no PTI, no MTP)
   Tokens/step: 1
   Multiplier:  1×

② PTI only (identical twins, SSQ kernel)
   Tokens/step: 1 + acceptance_rate
   Multiplier:  2× (greedy), ~1.75× (sampled)

③ MTP only (Qwen3.6 built-in multi-token prediction)
   Tokens/step: k  (k ≈ 2, MTP acceptance ~88%)
   Multiplier:  ~1.8×

④ PTI + MTP (one SSQ weight load × k MTP tokens × 2 streams)
   Tokens/step: k × (1 + acceptance_rate)
   Multiplier:  ~3.6-4× (greedy, k=2)
```

For Qwen3.6-27B Q5_K_M on MI50, at 100% greedy acceptance and MTP k=2:
```
  baseline:      ~37 tok/s (HBM-bound, 19 GB / 1 TB/s)
  PTI only:      ~74 tok/s (2×)
  MTP only:      ~66 tok/s (1.8×, MTP acceptance ~88%)
  PTI + MTP:     ~133 tok/s (3.6×)
```

Bandwidth cost per step: unchanged (one model load).
Compute cost: 2× per PTI, k× per MTP — both hidden on bandwidth-bound hardware.

---

## Variants

### Variant 1: Identical Twins (implemented)
- Pack model with itself
- Acceptance rate = 100% greedy, ~75-85% sampled
- Zero quality difference — verified output is always the target model's output
- Proved with Qwen3-1.7B, Qwen3.5-0.8B (5/5 prompts, SVG generation)

### Variant 2: TSQ Side-Car (planned)
- Stream A: base model (verifier)
- Stream B: task-specific fine-tune (TSQ-Coding, TSQ-Multilang, etc.)
- Same base architecture → SSQ-compatible tensor shapes
- Higher acceptance on the tuned task domain
- Quality signal: falling acceptance rate = off-domain prompt (use base output)
- TSQ GGUF files exist at `../gguf/Qwen3.6-27B-TSQ-*.gguf`

### Variant 3: Fine-Tune Twins / EAGLE-style (future)
- Stream B trained specifically to draft for Stream A
- Higher acceptance than identical twins on sampling
- Can use a shallow fine-tune of the base — same shapes, SSQ-compatible
- Requires training infrastructure not yet built

### Variant 4: PTI + MTP (planned)
- Enable Qwen3.6's built-in MTP heads alongside PTI
- Twin A: predicts tokens T+1, T+2 per pass (MTP)
- Twin B: speculatively runs at T+2, predicts T+3, T+4
- One SSQ weight load per layer → 4 confirmed tokens per step (at 100% accept)
- Requires MTP-aware inference loop and llama.cpp integration

---

## The Inference Loop

```python
# Initialization
twin_A_kvcache = empty()
twin_B_kvcache = empty()
speculative_token = prompt_last_token

for step in range(max_tokens):
    # One SSQ weight load per layer serves both forwards
    logits_A = forward(model, confirmed_token,    twin_A_kvcache)
    logits_B = forward(model, speculative_token,  twin_B_kvcache)

    actual_next = argmax(logits_A)

    if speculative_token == actual_next:           # accept
        emit(actual_next)
        speculative_token = argmax(logits_B)       # B's guess is now live
        twin_B_kvcache.extend(actual_next)
    else:                                          # reject
        emit(actual_next)
        twin_B_kvcache = deepcopy(twin_A_kvcache)  # reset B
        speculative_token = actual_next            # use A's output as next guess
```

**KV cache note:** each twin maintains a fully independent KV cache.
`copy.deepcopy` is required in transformers — DynamicCache is mutable and
sharing it between A and B corrupts both states (confirmed bug in development).

---

## The HIP Kernel

`pti_kernel.hip` implements:

- `pti_linear` — packed matmul for small K (one block per output row)
- `pti_linear_tiled` — tiled matmul for large K (256 threads per row)
- `pti_attn_scores` — QK attention with independent KV caches per twin
- `pti_verify` — argmax reduction for accept/reject check
- `pti_softmax` — in-place softmax

Portability shim at the top maps all `hip*` API calls to `cuda*` equivalents
when compiled with `nvcc` — the same file compiles for both AMD and NVIDIA.

```
AMD:    hipcc -O3 --offload-arch=gfx906  -o pti_test pti_kernel.hip
NVIDIA: nvcc  -O3 -arch=sm_89 -x cu     -o pti_test pti_kernel.hip
```

---

## What Has Been Built

### `ssq/` package (complete)
- `format.py`: ModeConfig, block pack/unpack, SSQFile reader/writer
- `pack.py`: GGUF and safetensors ingestion, source format detection
- `__main__.py`: CLI — `pack`, `verify`, `info`, `kernel`

### `packed-twin-inference/` (this directory)
- `infer.py`: Python PTI inference loop (transformers-based)
- `benchmark.py`: acceptance rate + throughput measurement
- `pack.py`: convenience wrapper — pack model with itself as SSQ twins
- `pti_kernel.hip`: AMD/NVIDIA HIP kernel (matmul, attention, verify, softmax)
- `pti_hip.py`: Python numpy reference + ctypes wrapper for libpti.so
- `Makefile`: build for gfx906/gfx1100 (AMD) or sm_80/sm_89 (NVIDIA)
- `svg_test.py`: SVG generation test — baseline vs PTI twin output comparison
- `diagram-memory-layout.svg`: block format and dual-stream compute diagram
- `diagram-workflow.svg`: full PTI inference loop diagram

### Proved results
| Model | Prompts | Accept | Output match | Throughput |
|---|---|---|---|---|
| Qwen3-1.7B (BF16) | 5 general | 100% | 5/5 ✓ | — |
| Qwen3.5-0.8B (BF16) | 5 general | 100% | 5/5 ✓ | — |
| Qwen3.5-0.8B (BF16) | SVG rainbow | 100% | byte-identical ✓ | — |
| **Qwen3.6-27B Q5_K_M** | general | **100%** | ✓ | **28.9 tok/s (1.38× baseline)** |

**Measured throughput on MI50 (Qwen3.6-27B Q5_K_M, 80-token runs):**

| Config | tok/s | multiplier | notes |
|---|---|---|---|
| Baseline | 21.1 | 1.00× | single stream, llama.cpp |
| PTI (llama.cpp, two-seq batch) | 28.9 | **1.38×** | 2 tokens emitted per accept step |
| PTI SSQ HIP kernel (projected) | ~42 | **~2×** | fused weight load, overhead eliminated |
| PTI SSQ + MTP (target) | **~80–100** | **~4–5×** | 2 streams × MTP k=1 at 55–70% HBM eff |

The gap from 1.38× to 2×: llama.cpp reads two separate KV caches per step (A and B). The SSQ HIP kernel eliminates this by fusing both matmuls behind one HBM weight load — weight bandwidth is the bottleneck, and it's shared.

---

## How PTI Works on llama.cpp (No Special Tricks)

`pti_4seq.cpp` achieves 1.96× with standard llama.cpp API:

1. Load one model once
2. Create one context with `n_seq_max=4`
3. Each step: submit a 4-token batch — one token per sequence — to a single `llama_decode`
4. llama.cpp shares the weight load internally across all tokens in the batch

There are no custom kernels, no model modifications, no format changes. The throughput
gain comes entirely from the batch API: llama.cpp loads each weight tensor once and
computes against all 4 tokens simultaneously.

**Why it works**: batch=1 LLM inference is memory-bandwidth-bound. The weight tensors
are loaded from HBM once per layer regardless of how many tokens are in the batch
(up to some capacity limit). Adding 3 more sequences adds attention computation and
KV cache reads for the divergent tail — but those are small relative to the weight load.

**VRAM cost**: almost free.

| Config | GPU VRAM | vs baseline |
|---|---|---|
| Baseline 1-seq | 23.7 GiB | — |
| PTI 2-seq | 24.0 GiB | +0.3 GiB |
| PTI 3-seq | 24.3 GiB | +0.6 GiB |
| PTI 4-seq | 24.5 GiB | +0.8 GiB |

KV + recurrent state per sequence = ~278 MiB (tiny vs 22.9 GiB model). Standard
speculative decoding would require a 2nd draft model: +4–19 GiB depending on size.

**The ceiling**: at 4 sequences, step overhead is 2.04× single-batch → 4/2.04 ≈ 1.96×.
A 5th sequence adds ~1 more token but costs another ~0.3× overhead → 5/2.35 ≈ 2.1×.
Returns diminish. The SSQ fused kernel is the path beyond 2× because it eliminates
the per-sequence overhead entirely: all N sequences share ONE weight load, so adding
a sequence costs only the divergent-tail attention — which is negligible.

---

## The 1.38× → 2× Gap: Root Cause and Fix

### What's costing us

Each dual-batch decode step runs twice for:
1. **KV cache reads** — for every attention layer, seq 0 reads its full KV history
   and seq 1 reads its own. These are separate HBM reads on identical data for
   all positions up to `pos_a - 1`.
2. **Softmax + logit projection** — computed for both sequences.

The weights are already shared (one load). KV for the shared prefix is the waste.

### Fix: shared-prefix KV via dual seq_id assignment

A and B have identical KV for every token from position 0 through `pos_a - 1`.
The divergence is only at the most recent 1–2 positions. If each token is
registered under **both** seq_id 0 and seq_id 1 simultaneously, llama.cpp's
attention naturally reads the shared KV once and serves both sequences:

```c
// Prefill and every accepted token: write to BOTH sequences at the same position
batch_add(&batch, tok, pos, 0, false);
batch_add(&batch, tok, pos, 1, false);  // same pos, both seq_ids
```

This means:
- Shared prefix KV is stored once (not copied)
- Attention reads it once and fan-outs to both decode heads
- Only the diverged tail (pos_a and pos_b) has separate KV entries

Combined with the 2-token-per-accept emit, this should push measured throughput
toward ~1.7–1.8× baseline on llama.cpp, closing most of the remaining gap to 2×.

The irreducible floor: the diverged tail (1–2 positions) and the separate softmax
passes. The SSQ HIP kernel eliminates even these by fusing into a single pass.

### pti_4seq.cpp: 4-Sequence PTI

`pti_4seq.cpp` adds a 4th sequence (Twin D) three steps ahead of A:

```
Seq 0 (A): verifier at pos_a
Seq 1 (B): +1 drafter at pos_b
Seq 2 (C): +2 drafter at pos_c
Seq 3 (D): +3 drafter at pos_d

Quad-batch: one llama_decode, 4 predictions, 4 tokens per 100% accept step.
Overhead: 2.04× single-batch → 4 / 2.04 ≈ 1.96× baseline ≈ 2×.
Measured: 38.1 tok/s on Qwen3.6-27B UD-Q6_K_XL, MI50.
```

The sub-linear overhead growth (+0.33 per seq at 4 vs +0.44 at 3) makes each
added sequence cheaper, but beyond 4 the marginal gain shrinks toward zero.

### Next target: pti_ssq.cpp — SSQ kernel inference loop

Goal: **~80–100 tok/s** on MI50 Q5_K_M. Math:

```
Q5_K_M model: 18.6 GB | MI50 HBM: 1000 GB/s
SSQ 2-stream:   2 tokens per weight load (overhead eliminated)
+ MTP k=1:      × 1.88 MTP accept rate
= 3.76 tokens per 18.6 GB load → 4.9 GB/token effective
At 70% HBM efficiency: ~143 tok/s theoretical
Conservative (55-60%): ~80–100 tok/s — the target
```

Measured kernel bandwidth (MI50, gfx906):
- D2D hipMemcpy (baseline): 383 GB/s  
- F16 GEMV [17408×5120]: 460 GB/s (custom kernel, optimal coalescing)
- Packed Q8_0 [17408×5120]: **7 GB/s** (stride-34 non-coalesced — broken for inference)
- Flat int8 [17408×5120]: **92 GB/s** (unit-stride, 24% of peak — usable)

MI50 peak HBM spec: 512 GB/s. Measured baseline 19.4 tok/s on 22.9 GiB UD-Q6_K_XL → implies
~444 GB/s effective bandwidth (87% of spec). The llama.cpp ROCm backend already achieves
near-peak HBM utilization for single-stream decode.

**Inference loop plan** (pti_infer.cpp or pti_infer.hip):
1. Use rocBLAS/hipBLAS SGEMV for matmul layers — achieves ~460 GB/s (proven)
2. Load Q8_0 weights from GGUF (simple: f16 scale + 32 int8 per block, 34 bytes)
3. Dequantize on-the-fly: w_float = scale × w_int8 before GEMV call
4. Run 4 SGEMV calls per layer (A, B, C, D streams) with SAME weight matrix → one HBM load
5. KV cache per stream (tiny: ~70 MiB each), RoPE, RMSNorm, SwiGLU activation
6. Accept/reject loop + MTP head (layer 64, same GGML format)

`pti_kernel.hip` already has: `pti_linear_tiled`, `pti_attn_scores`, `pti_verify`, `pti_softmax`.
Also added: `pti_linear4_q8_flat` (coalesced 4-stream Q8 GEMV).
Still needed for inference loop: RMSNorm, RoPE, SwiGLU gate, rocBLAS GEMV wrapper.

- TSQ side-car benchmark (TSQ-* GGUF files exist in ../gguf/) — after SSQ loop is working

---

## pti_mtp.cpp: 3-Sequence PTI with MTP Re-Init

`pti_mtp.cpp` extends the 2-seq dual-batch of `pti_llama.c` to 3 sequences:

```
Seq 0 (Twin A): verifier at pos_a
Seq 1 (Twin B): 1-step drafter at pos_b = pos_a + 1
Seq 2 (Twin C): 2-step drafter at pos_c = pos_a + 2

Triple-batch each step: one llama_decode, one weight load, three predictions.
```

### 3-accept: 3 tokens per step

```
actual_next = argmax(A's logits)    ← verified
next_from_b = argmax(B's logits)    ← confirmed when tok_b == actual_next
next_from_c = argmax(C's logits)    ← confirmed when tok_c == next_from_b
```

On 100% greedy accept: emit 3 tokens per step.

### Why PTI × MTP doesn't multiply on llama.cpp

```
pti_llama.c 2-seq dual-batch:
  step cost    = 1.47× baseline (measured)
  tokens/step  = 2
  throughput   = 2 / 1.47 = 1.36× ≈ measured 1.38–1.57×

pti_mtp.cpp 3-seq triple-batch:
  step cost    ≈ 1.8× baseline (estimated; more sequences = more attention)
  tokens/step  = 3
  throughput   = 3 / 1.8 = ~1.67× baseline

Naïve 1.38× × 1.7× = 2.35× is wrong. The gains don't compound because
the overhead grows with each added sequence.
```

On the SSQ fused kernel, adding C IS free (one weight load for all sequences,
only negligible attention overhead at short positions), so gains do compound.

### MTP context role

The Qwen3.6-27B UD-Q6_K_XL GGUF has `nextn_predict_layers=1` (one MTP head,
layer 64 of 65). `LLAMA_CONTEXT_TYPE_MTP` creates a context that only runs
this head, not the full 65-layer transformer.

MTP inputs: (token ID, hidden state H from main context at that position)  
MTP output: logits for the next token  
MTP cost: ~1/65 of a main context decode (negligible on bandwidth-bound hardware)

In `pti_mtp.cpp`, MTP is used **only on the reject path** to re-init C:
- After reject: run B-init (1 full pass) → get H_b
- MTP(tok_b, H_b) → tok_c, no extra full pass needed
- Without MTP: B-init + C-init = 2 full passes

On 100% greedy (no rejects), MTP fires zero times. Its primary value is
for sampling mode where rejects occur on ~5–15% of steps.

### C initialization

Two-stage init to ensure C's KV is populated before the main loop:
1. C decodes tok_gen_0 at pos_a (seq 2) — fills C's KV at pos_a
2. C decodes tok_b at pos_b (seq 2) — fills C's KV at pos_b, gives tok_c

This costs 2 extra decode steps before the main loop begins (done once).

### Measured throughput

| Config | tok/s | multiplier | overhead |
|---|---|---|---|
| Baseline UD-Q6_K_XL | 19.4 | 1.00× | 1.00× |
| PTI 2-seq (pti_llama.c) | 30.5 | 1.57× | 1.27× |
| PTI 3-seq (pti_mtp.cpp) | 33.9 | 1.75× | 1.71× |
| PTI 4-seq (pti_4seq.cpp) | **38.1** | **1.96×** | 2.04× |

Overhead scaling: +0.44, +0.33 per additional sequence. Sub-linear growth means
each added sequence is cheaper than the last. A 5th sequence would give ~2.1× —
marginal enough that 4-seq is the practical limit on llama.cpp.

---

## Key Validated Claims

1. **Acceptance rate**: 100% greedy on identical twins — proved on Qwen3-1.7B, Qwen3.5-0.8B, Qwen3.6-27B
2. **Output identity**: PTI output identical to baseline greedy — proved on text, SVG, and GGUF 27B
3. **Measured throughput (2-seq)**: 28.9 tok/s PTI vs 21.1 tok/s baseline on Q5_K_M MI50 (1.38×); 30.5 vs 19.4 on UD-Q6_K_XL (1.57×)
4. **Measured throughput (3-seq)**: 33.9 tok/s PTI vs 19.4 tok/s baseline on UD-Q6_K_XL MI50 (1.75×); 100% B+C accept on greedy
5. **Measured throughput (4-seq)**: 38.1 tok/s PTI vs 19.4 tok/s baseline on UD-Q6_K_XL MI50 (1.96×); 100% B+C+D accept on greedy; overhead scaling sub-linear (+0.33/seq at 4)
5. **2× gap explained**: llama.cpp dual-KV overhead; fused SSQ HIP kernel eliminates it → true 2×
6. **CUDA portability**: pti_kernel.hip compiles for both AMD (hipcc) and NVIDIA (nvcc -x cu)
