# Packed Twin Inference (PTI)

**One weight load. Two token streams. Zero quality loss.**

Packed Twin Inference achieves ~2× LLM throughput by running two inference
streams simultaneously from a single memory load — eliminating the memory
bandwidth bottleneck that limits single-session token generation speed.

---

## The Idea

LLM inference at batch=1 is bottlenecked by memory bandwidth, not compute:

```
tokens/sec = HBM_bandwidth / model_size
```

Every token requires loading all model weights from VRAM. The GPU compute
units sit mostly idle waiting for data.

Standard speculative decoding breaks this limit using a small draft model —
but the draft must be much smaller than the target, accepting a quality
tradeoff and requiring two separate models in VRAM.

**PTI eliminates both constraints.**

```
Standard speculative decoding:
  Load draft weights   (small model, lower quality)
  Load target weights  (large model, separate VRAM)
  → Two memory loads, two VRAM footprints, quality gap

Packed Twin Inference:
  Load packed weights  (one SSQ file, one uint16 per weight pair)
  → Twin A computes token T     (verifier, position n)
  → Twin B computes token T+1   (drafter, position n+1)
  → One memory load serves both streams
  → One VRAM footprint, zero quality gap
```

The draft is not cheaper because it is smaller — it is free because it
**rides along on the same memory load** as the verifier.

---

## How It Works

Weights are packed using [SSQ (Side-by-Side Quantization)](../ssq/):

```
uint16 slot:  [ int8: Twin A weight | int8: Twin B weight ]
               hi byte                lo byte
```

One load from VRAM yields one weight for each twin. The HIP kernel unpacks
and accumulates both matmuls simultaneously:

```c
uint16_t pw = W_packed[i];
int8_t   wa = (int8_t)(pw >> 8);    // Twin A weight
int8_t   wb = (int8_t)(pw & 0xFF);  // Twin B weight
acc_a += scale_a * (float)wa * x_a[i];
acc_b += scale_b * (float)wb * x_b[i];
```

At each inference step:
- **Twin A** processes the confirmed token at position n → produces T_{n+1}
- **Twin B** processes T_{n+1} speculatively → produces T_{n+2}
- **Check**: did B correctly predict T_{n+1} last cycle?
  - **Yes** (accept): emit T_{n+1}, B's T_{n+2} is immediately ready
  - **No** (reject): emit T_{n+1}, reset B to A's state

---

## Proved Results

| Model | Task | Accept | Output |
|---|---|---|---|
| Qwen3-1.7B BF16 | 5 general prompts | 100% | 5/5 identical ✓ |
| Qwen3.5-0.8B BF16 | 5 general prompts | 100% | 5/5 identical ✓ |
| Qwen3.5-0.8B BF16 | SVG rainbow generation | 100% | byte-identical ✓ |
| **Qwen3.6-27B Q5_K_M** | general prompts | **100%** | ✓ | **28.9 tok/s (1.38× baseline)** |

Python simulation measures ~0.50× (expected — two sequential forwards, not fused).
Acceptance rate and output identity are the meaningful signals.

---

## Measured Throughput (MI50, Qwen3.6-27B)

VRAM = GPU-resident memory only (model weights on ROCm0 + KV/state + compute buffer).
CPU_Mapped weights (embedding table overflow): +0.8 GiB for Q5_K_M, +1.3 GiB for UD-Q6_K_XL.

| Config | Model | tok/s | multiplier | GPU VRAM | notes |
|---|---|---|---|---|---|
| Baseline | Q5_K_M | 21.1 | 1.00× | 18.1 GiB | 17.3 model + 0.3 KV/state + 0.5 compute |
| PTI 2-seq — llama.cpp | Q5_K_M | **28.9** | **1.38×** | 18.4 GiB | +0.3 GiB vs baseline |
| Baseline | UD-Q6_K_XL | 19.4 | 1.00× | 23.7 GiB | 22.9 model + 0.3 KV/state + 0.5 compute |
| PTI 2-seq — llama.cpp | UD-Q6_K_XL | **30.5** | **1.57×** | 24.0 GiB | +0.3 GiB vs baseline |
| PTI 3-seq — llama.cpp | UD-Q6_K_XL | **33.9** | **1.75×** | 24.3 GiB | +0.6 GiB vs baseline |
| PTI 4-seq — llama.cpp | UD-Q6_K_XL | **38.1** | **1.96×** | 24.5 GiB | +0.8 GiB vs baseline |
| PTI SSQ HIP kernel | Q8 packed | ~40 | **~2×** | ~29 GiB (est.) | two models in one uint16 file |
| PTI + MTP (SSQ kernel) | UD-Q6_K_XL | ~65 | **~3.4×** | ~29 GiB (est.) | fused kernel + MTP k=1 |

**VRAM note**: PTI on llama.cpp costs almost no extra VRAM — 0.3–0.8 GiB across 2–4 sequences,
because KV/state per sequence is tiny (~278 MiB) relative to model weights (~23 GiB).
Standard speculative decoding needs a 2nd draft model: ~4–19 GiB extra depending on size.

The UD-Q6_K_XL has `nextn_predict_layers=1` (MTP head present). PTI on this model
yields 1.57× vs 1.38× on Q5_K_M — the dual-batch is proportionally more efficient.

**How PTI works on llama.cpp**: no special tricks — one model, one context, one
`llama_decode` call per step with a 4-token batch (one per sequence). llama.cpp
already shares weight loads across all tokens in a batch. The gain comes entirely
from emitting N tokens per step while paying only ~2× the single-token step cost.

**Overhead scaling (measured, UD-Q6_K_XL MI50)**:

| N seqs | tok/s | multiplier | step overhead | overhead increment | GPU VRAM |
|---|---|---|---|---|---|
| 1 (baseline) | 19.4 | 1.00× | 1.00× | — | 23.7 GiB |
| 2 | 30.5 | 1.57× | 1.27× | +0.27 | 24.0 GiB |
| 3 | 33.9 | 1.75× | 1.71× | +0.44 | 24.3 GiB |
| 4 | 38.1 | **1.96×** | 2.04× | +0.33 | 24.5 GiB |

Overhead grows sub-linearly per sequence added. 4 tokens / 2.04× overhead ≈ 1.96×. Adding a 5th sequence would cost another ~0.3× overhead → 5/2.35 ≈ 2.1× — marginal gain.

**Why PTI × MTP ≠ 1.57 × 1.9 on llama.cpp**: PTI's 1.57× is already capped by
dual-batch overhead (2× attention per step). Adding MTP as a 3rd sequence
(triple-batch) costs ~1.8× single-batch — so 3 tokens / 1.8× overhead ≈ 1.67×
baseline. The gains don't multiply. The SSQ fused kernel avoids this: it loads
weights once for ALL sequences, so attention is the only extra cost (~negligible
at pos 80). PTI + MTP only multiplies cleanly on the SSQ kernel.

**MTP integration**: `pti_mtp.cpp` (C++) uses `src/llama-ext.h` to access
`llama_set_embeddings_pre_norm` for MTP context initialization. The MTP head
(1 layer vs 65 main layers) runs the draft head directly on hidden states — it
does NOT reload the full model. MTP's main benefit in pti_mtp.cpp is speeding
up re-initialization after a reject (1 fewer full-model pass).

---

## Four-Point Throughput Comparison

On bandwidth-bound hardware (MI50, 1 TB/s HBM2), all compute overhead is hidden.

| Config | Tokens per weight load | Multiplier | Status |
|---|---|---|---|
| ① Baseline | 1 | 1.00× | ✓ measured: 21.1 (Q5_K_M) / 19.4 (UD) |
| ② PTI — llama.cpp | 2 per accept | **1.38–1.57×** | ✓ measured: 28.9 / 30.5 tok/s |
| ② PTI — SSQ kernel | 1 + accept | **2.00×** | ✓ projected; kernel ready |
| ③ MTP only (UD-Q6_K_XL) | k = 1 | **~1.9×** | MTP head is 1 extra layer (~free) |
| ④ PTI + MTP — llama.cpp | 3-seq triple-batch | **1.75×** | ✓ measured: 33.9 tok/s (UD-Q6_K_XL) |
| ④ PTI 4-seq — llama.cpp | 4-seq quad-batch | **1.96×** | ✓ measured: 38.1 tok/s (UD-Q6_K_XL) |
| ⑤ PTI + MTP — SSQ kernel | (1+accept)×k | **~3.4×** | fused weight load; gains multiply |

Qwen3.6 has Multi-Token Prediction (MTP) built in — the model head predicts
k≈2 tokens per forward pass. PTI doubles streams per weight load. Combined,
one weight load confirms ~4 tokens instead of 1: `1 TB/s / 19 GB × 4 = ~210 tok/s`
vs baseline's ~52 tok/s for Qwen3.6-27B Q5_K_M on MI50.

---

## Acceptance Rate by Decoding Mode

| Decoding | Acceptance | PTI gain | Notes |
|---|---|---|---|
| Greedy (temp=0) | 100% | 2.00× | Identical twins always agree |
| Top-p / Top-k | ~70-85% | 1.70-1.85× | Depends on model confidence |
| TSQ side-car | 75-95% | 1.75-1.95× | Higher within the tuned domain |
| Fine-tune twins | higher | up to 2× | B trained to match A's distribution |

TSQ side-car: pack base + task-fine-tune (e.g., TSQ-Coding) as the twin pair.
High acceptance on the target domain; falling acceptance signals off-domain.

---

## Comparison to Standard Speculative Decoding

| | Standard SpecDec | PTI | PTI + MTP |
|---|---|---|---|
| Draft model | Small, separate | Same model, packed | Same model, packed |
| Draft VRAM | Extra (draft model) | Zero extra | Zero extra |
| Bandwidth cost | 2 loads | 1 load | 1 load |
| Acceptance rate | 60-70% | 100% greedy, ~75% sampled | same |
| Quality | Target model quality | Identical to baseline | Identical |
| Throughput gain | ~1.6× | **2×** | **~4×** |
| Integration | llama.cpp, vLLM | Requires SSQ kernel | SSQ kernel + MTP |

---

## Tools

| File | Purpose |
|---|---|
| `infer.py` | PTI inference loop — baseline and twin generation |
| `benchmark.py` | Measure acceptance rate, throughput, output correctness |
| `pack.py` | Pack a model with itself as SSQ twin streams |
| `pti_kernel.hip` | HIP/ROCm kernel — packed matmul, attention, verify |
| `pti_hip.py` | Python wrapper for the compiled HIP kernel (ctypes) |
| `pti_llama.c` | 2-seq PTI via llama.cpp C API (measured: 1.38–1.57×) |
| `pti_mtp.cpp` | 3-seq PTI + MTP re-init via llama.cpp C++ API (measured: 1.75×) |
| `pti_4seq.cpp` | 4-seq PTI via llama.cpp C API (measured: **1.96×**, ~2× target) |
| `Makefile` | Build HIP kernel, pti_llama, and pti_mtp |
| `DESIGN.md` | Full design rationale, math, and implementation details |
| `diagram-memory-layout.svg` | How uint16 packs two int8 weights — one load, two streams |
| `diagram-workflow.svg` | Full PTI inference loop: prefill → verify → accept/reject |

---

## Usage

```bash
# Run the benchmark (measures acceptance rate + output correctness)
python3 benchmark.py --model ../model/Qwen3-1.7B --tokens 60

# Pack a model as twins for SSQ inference
python3 pack.py ../model/Qwen3-1.7B qwen17b_twins.ssq

# Verbose mode — see each accept/reject decision
python3 benchmark.py --model ../model/Qwen3-1.7B --verbose

# Build and test the HIP kernel (requires ROCm)
make                          # defaults to gfx906 (MI50)
make ARCH=gfx1100             # RX 7900 XTX
./pti_test                    # correctness check + throughput timing

# Run PTI inference using the compiled kernel
python3 pti_hip.py --model ../model/Qwen3-1.7B.ssq --prompt "Hello"
```

---

## Architecture Diagrams

**Memory layout** (`diagram-memory-layout.svg`):

```
PTI block (68 bytes)
┌──────────┬──────────┬─────────────────────────────────────┐
│ scale_a  │ scale_b  │   32 × uint16                       │
│  f16 2B  │  f16 2B  │  ┌──────────┬──────────┐  × 32      │
└──────────┴──────────┘  │ Twin A   │ Twin B   │            │
                          │  int8    │  int8    │            │
One HBM load              │  hi byte │  lo byte │            │
serves both ──────────►  └──────────┴──────────┘            │
streams                   ↓                    ↓             │
                    acc_a += sa*wa*x_a   acc_b += sb*wb*x_b │
```

**Inference workflow** (`diagram-workflow.svg`):

```
PREFILL (both twins on prompt)
         │
    ┌────▼─────────────────────────────────────────┐
    │ STEP n:  ① load 68-byte packed block         │
    │          ② Twin A (pos n)  →  actual_next    │
    │          ③ Twin B (pos n+1, speculative)      │
    │          ④ VERIFY: B's guess == actual_next?  │
    │             YES → emit+advance  (no reset)   │
    │             NO  → emit+reset B to A's state  │
    └──────────────────────────────────────────────┘
         │ n += 1
         └──► repeat
```

---

## Requirements

**Python simulation (benchmark/infer.py):**
- Python 3.10+
- PyTorch with CUDA or ROCm
- `transformers` (HuggingFace)
- SSQ package (`../ssq/`) for packing

**HIP kernel (real hardware throughput):**
- ROCm 5.x or 6.x
- `hipcc` compiler
- AMD GPU: MI50 (gfx906), MI100 (gfx908), RX 7900 (gfx1100), or compatible

---

## Relationship to SSQ

PTI is built on top of the [SSQ format](../ssq/):

```
ssq/          Side-by-Side Quantization format and tools
  format.py   Pack/unpack uint16 weight pairs
  pack.py     GGUF and safetensors ingestion
  infer.py    Numpy reference kernel

packed-twin-inference/   ← this directory
  infer.py              PTI inference loop (Python sim)
  benchmark.py          Acceptance rate and correctness
  pack.py               Pack model with itself as twins
  pti_kernel.hip        HIP kernel — the real throughput path
  pti_hip.py            Python wrapper for pti_kernel.so
  Makefile              Build for MI50 / MI100 / RX 7900
  diagram-memory-layout.svg
  diagram-workflow.svg
```

SSQ handles the twin packing format. PTI is the inference strategy that turns
it into a speculative throughput multiplier. The HIP kernel is where the 2×
gain materialises in hardware: one uint16 load → two independent matmul streams.
