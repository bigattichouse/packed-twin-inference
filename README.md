# Packed Twin Inference (PTI)

**Self-speculative decoding using staggered KV caches — same model, N sequences, one weight load.**

PTI packs N staggered token streams into a single `llama_decode` call. Each step produces N candidate
tokens via a verify/accept chain. At 100% accept (greedy decoding) it emits up to N tokens per step.
The overhead per step determines whether this beats baseline.

> **June 2026 — the investigation concluded** (see `KERNEL_PLAN.md`): the stagger design is
> mathematically bounded below baseline via the public API, and the GEMV kernel is at its
> instruction-issue floor on gfx906. The machinery built for PTI (checkpoint sequences, batched
> verification, byte-identical audit) was redirected into **`pti_lookup`** — n-gram lookup
> speculative decoding with SSM-safe rollback and adaptive draft length — which delivers
> **1.85× on real code-editing tasks (35.5 vs 19.2 tok/s), byte-identical output, and a
> bounded ~0.96× worst case on hostile text.**

---

## Honest Status (June 2026)

**The 38.1 tok/s headline from earlier versions of this repo was wrong.** The original
`pti_4seq.cpp` had a duplicate-token counting bug: on 4-accept it emitted 4 tokens per step but
re-emitted 3 of them at the start of the next step. After correcting the algorithm:

| Config | Measured tok/s | vs baseline | Status |
|---|---|---|---|
| Baseline (single-seq) | 19.0 | 1.00× | correct |
| PTI 2-seq, multi-emit | 16.3 | 0.86× | **best measured** |
| PTI 4-seq, multi-emit (M5.3) | 14.6 | 0.77× | correct |
| PTI 4-seq, single-emit | ~10.2 | 0.54× | correct but slow |
| ~~PTI 4-seq (old, buggy)~~ | ~~38.1~~ | ~~2.02×~~ | counted duplicate tokens |

**PTI is currently slower than baseline regardless of N.** The fundamental bottleneck is
mathematical: for any N-seq multi-emit step, total cost = 1 batch-decode + (N-1) reinit calls.
Each reinit is one full N=1 baseline decode (52.6ms). So the per-token cost is always:

```
(batch + (N-1)×52.6ms) / N  ≥  (1×52.6ms + (N-1)×52.6ms) / N  =  52.6ms/tok  =  baseline
```

The batch-decode costs > 1× baseline (1.25× for N=2, 1.86× for N=4), so PTI is always
**strictly slower** than baseline with the current public-API reinit design. No value of N helps.

- 2-seq: (66.2 + 52.6ms) / 2 = 59.4ms/tok → 16.8 tok/s theoretical, **16.3 measured**
- 4-seq: (98 + 158ms) / 4 = 64ms/tok → 15.6 tok/s theoretical, **14.6 measured**
- 2-seq is optimal N; adding more sequences just adds more reinit calls without proportional gain.

**What we measured and confirmed:**
- N=4 MMVQ scaling: 1.86× (after M5.1 loop restructure, down from 1.95×); N=2: 1.25×
- GEMV bottleneck: compute (inner-loop FMAFs), NOT activation L2 traffic
- SSM state: cannot tolerate stale updates — speculative reinit causes 47% D-miss rate
- Accept chain: correct, byte-identical to baseline at greedy, 15/15 audit PASS
- 2-seq (optimal N): 16.3 tok/s measured, 100% 2-accept, byte-identical, 25/25 PASS

---

## Audited Results

Run with `make audit` (full, ~10 min) or `make audit-quick` (skip GEMV bench):

```
━━ 2. Baseline single-sequence (pti_debug)
      baseline: 19.0 tok/s  (261.9 ms prefill)
      output looks coherent (unique-4gram ratio: 0.95)

━━ 3. PTI 4-seq correctness (pti_4seq)
  PASS  PTI output identical to baseline (byte-for-byte)
        4-acc=13  3-acc=0  2-acc=0  rej=0  tokens=50
  PASS  zero rejects at greedy
  PASS  no partial accepts (all 4-accept at greedy)
  PASS  PTI token count correct (50)
        PTI reported: 14.6 tok/s (steady)  13.1 tok/s (amortized)
```

---

## How It Works

### Stagger structure

Four sequences share the same model and KV/SSM state up to the prefill, then diverge:

```
Seq 0 (A): verifier    at pos_a          — ground truth each step
Seq 1 (B): 1-step drafter at pos_a + 1  — prediction for pos_a+1
Seq 2 (C): 2-step drafter at pos_a + 2  — prediction for pos_a+2
Seq 3 (D): 3-step drafter at pos_a + 3  — prediction for pos_a+3
```

Each step is one `llama_decode` call with all 4 tokens in one batch.

### Accept chain (greedy)

```
actual_next  = A's prediction  (ground truth)
next_from_b  = B's prediction

tok_b == actual_next?           → B was correct
  tok_c == next_from_b?         → C was correct
    tok_d == next_from_c?       → D was correct → 4-accept
    else                        → 3-accept, reinit D
  else                          → 2-accept, reinit C+D
else                            → reject, reinit B+C+D
```

At greedy (temp=0): 100% 4-accept. Output is identical to single-seq baseline.

### Multi-emit on 4-accept

On 4-accept the 4 confirmed output tokens are:
`actual_next, next_from_b, next_from_c, next_from_d`

Emitting all 4 requires rebuilding the stagger from D's confirmed state:
1. Copy D → A (free, just KV/SSM bookkeeping)
2. reinit B from A: decode `next_from_d` at `pos_d` → `tok_b` (one N=1 decode)
3. reinit C from B: decode `tok_b` at `pos_b` → `tok_c` (one N=1 decode)
4. reinit D from C: decode `tok_c` at `pos_c` → `tok_d` (one N=1 decode)

These 3 reinit calls are **irreducible** for a forward-stagger hybrid SSM model:
- Each depends on the previous (sequential dependency)
- SSM state diverges by 3 steps if reinit is skipped (47% miss rate measured)
- A pure-transformer model might tolerate speculative reinit; Qwen3.6's Mamba layers cannot

### Cost breakdown (measured, MI50, 19.0 tok/s baseline)

```
Baseline step:          52.6 ms  (1 token)  → 19.0 tok/s

PTI 4-accept step:
  quad-decode (N=4):    98.0 ms  (1.86× baseline, M5.1 improved from 1.95×)
  3 reinit calls (N=1): 158 ms   (3 × 52.6 ms)
  ─────────────────────────────
  Total:                256 ms   for 4 tokens
  Per-token cost:       64 ms/token → 15.6 tok/s theoretical, 14.6 measured

Reinit cost dominates: 62% of total step time.
```

**Hard ceiling with current design**: even with zero quad-decode overhead, 3 reinit
calls cost `3 × 52.6ms = 158ms` for 4 tokens → 25.4 tok/s max (1.34× baseline), not 2×.

---

## GEMV Fusion Analysis

`pti_gemv_bench` measures whether ggml shares weight reads across N decode sequences.

**Results (MI50, Qwen3.6-27B UD-Q6_K_XL, ctx=128):**

```
N   ms/step   scaling
─   ───────   ───────
1     53.1    1.00×
2     66.2    1.25×
4    103.5    1.95× (pre-M5.1)  → 98.8ms / 1.86× (post-M5.1 loop restructure)
```

Context isolation test: Δscaling = 0.01 at ctx=128 vs ctx=1024 → bottleneck is GEMV
compute (inner-loop FMAFs), **not** SSM state or KV cache.

VGPR analysis: `mul_mat_vec_q<Q6_K, ncols_dst=4>` uses **63 VGPRs → 4 waves/SIMD**
vs ncols_dst=1 at **27 VGPRs → 9 waves/SIMD**. The 4:9 occupancy reduction accounts
for most of the 1.86× overhead.

---

## Path to Beating Baseline

To beat 19.0 tok/s with 4-seq multi-emit, total cost per 4-emit step must be < 210ms.
Current cost: 256ms. The gap: 46ms.

**Option A — MFMA kernel (M5.4):** reduce quad-decode from 98ms → ~58ms (1.1×).
Reinit still costs 158ms. Total: 216ms → 18.5 tok/s. *Barely matches, does not beat.*

**Option B — MFMA + speculative batch reinit:** run the 3 reinits as one N=4 batch
(~58ms instead of 158ms). Total: 116ms for 4 tokens → 34.5 tok/s (1.82× baseline).
*Requires empirical validation that C/D speculative states converge for this model.*

**Option C — 2-seq (current best):** 1 reinit call per 2-emit step.
Measured: 16.3 tok/s (0.86× baseline). Best without a kernel. *Still below baseline.*

**Why no N-seq can beat baseline with the public API**: total cost = batch + (N-1) reinits
= N × ≥52.6ms for N tokens → ≥52.6ms/token ≥ baseline. The batch always costs > baseline
(it processes N sequences), making the inequality strict. PTI via public API is bounded above
by baseline throughput.

**Option D — Custom fused kernel outside public API:** a kernel that loads each weight
block once and computes N dot products in a single warp pass, eliminating the 4× inner-loop
compute overhead. This is the original plan (PLAN.md Phase 4 / M5.4).

---

## MTP Analysis (Qwen3.6-27B UD Model)

The UD-Q6_K_XL model includes a Multi-Token Prediction head (`nextn_predict_layers=1`).
**Finding**: the MTP head takes `(tok_t, h_t from main model)` → predicts `t+1` — the
**same position** the main model already computes. It is NOT a "next-next" predictor and
cannot add a bonus 5th token to PTI output.

`pti_4seq.cpp` creates the MTP context for correctness and future sampled-mode work,
but at 100% greedy it is never invoked.

---

## Comparison to Standard Speculative Decoding

| | Standard SpecDec | PTI (correct, current) | PTI + MFMA kernel (projected) |
|---|---|---|---|
| Draft model | Small, separate model | Same model, staggered | Same model |
| Extra VRAM | +4–19 GiB | +0.8 GiB | +0.8 GiB |
| Accept rate (greedy) | ~60–70% | **100%** | **100%** |
| Quality | Draft model quality | Identical to baseline | Identical |
| Throughput (greedy) | ~1.6× | 0.77× baseline today | ~1.8× with MFMA+spec-reinit |

---

## Build & Run

```bash
# Prerequisites: llama.cpp built at ../llama.cpp/build with ROCm/HIP support
make 4seq          # bin/pti_4seq
make debug         # bin/pti_debug (single-seq baseline for comparison)

# Run PTI 4-seq
bin/pti_4seq -m ../gguf/Qwen3.6-27B-UD-Q6_K_XL.gguf \
             -p "The key to faster LLM inference is" -n 80

# Run baseline for comparison
bin/pti_4seq -m ../gguf/Qwen3.6-27B-UD-Q6_K_XL.gguf \
             -p "The key to faster LLM inference is" -n 80 --baseline

# Full correctness + performance audit (includes GEMV bench, ~10 min)
make audit

# Quick audit (build + baseline + PTI output match, ~6 min)
make audit-quick
```

---

## Files

| File | Purpose | Status |
|---|---|---|
| `pti_lookup.cpp` | **M6.4 n-gram lookup spec-dec — beats baseline on real tasks** | ✓ **35.5 tok/s on code edit (1.85×), byte-identical** |
| `pti_kbatch_bench.cpp` | M6.0 k-token batch cost curve + chain-match probe | ✓ batch verify exact, sub-linear cost |
| `pti_q6k_bench.hip` | M6.1/2 Q6_K MMVQ replica + variant experiments | ✓ kernel floor documented |
| `pti_2seq.cpp` | 2-seq PTI — optimal N for stagger design, public API | ✓ 16.3 tok/s, byte-identical |
| `pti_4seq.cpp` | 4-sequence PTI — public llama.cpp API, correct algorithm | ✓ 14.6 tok/s, byte-identical |
| `pti_debug.cpp` | Single-sequence baseline for comparison | ✓ |
| `pti_gemv_bench.cpp` | GEMV scaling + context isolation benchmark | ✓ |
| `pti_mtp.cpp` | 3-sequence PTI + MTP context (created, not invoked at greedy) | ✓ |
| `pti_llama.c` | 2-sequence PTI — C API (earlier prototype) | ✓ |
| `pti_server.cpp` | HTTP server (single-seq baseline, 19 tok/s, correct) | ✓ |
| `pti_kernel.hip` | HIP/ROCm Q8 multi-stream kernel + benchmarks | ✓ |
| `audit.sh` | End-to-end correctness + performance audit | ✓ 15/15 PASS |
| `PLAN.md` | Full investigation log, kernel analysis, roadmap | ✓ current |
| `DESIGN.md` | Architecture rationale, SSQ format | ✓ |

---

## Key Numbers

```
Hardware:          MI50, 32 GiB HBM2, ~1 TB/s peak
Model:             Qwen3.6-27B UD-Q6_K_XL (25 GB, 6-bit)
Baseline:          19.0 tok/s  (52.6 ms/step, single-seq)
PTI 2-seq (best):  16.3 tok/s  (59.5 ms/token, 0.86× baseline)
PTI 4-seq:         14.6 tok/s  (64 ms/token, 0.77× baseline)
N=2 scaling:       1.25×
N=4 scaling:       1.86×  (post-M5.1 loop restructure; was 1.95×)
Reinit cost:       62% of 4-accept step time (3 × 52.6ms = 158ms)
VGPR pressure:     27 → 63 VGPRs at N=4 (9 → 4 waves/SIMD, 44% occupancy)
SSM sensitivity:   47% D-miss rate with 1 stale update step → speculative reinit ruled out
Public API ceiling: cannot exceed baseline throughput (proof: batch > 1× + (N-1) reinits = N×)
```

---

*Hardware: AMD MI50 (gfx906, 32 GiB HBM2). Model: Qwen3.6-27B. All throughput figures
are unique tokens/second, measured with `-n 50` at greedy (temp=0).*
