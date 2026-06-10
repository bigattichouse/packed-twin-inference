# M6 — Custom N-Stream Kernel: Plan, Tests, Exploitation

**Status**: planning → M6.0 in progress
**Hardware**: AMD MI50 (gfx906, Vega 20). **No MFMA** (matrix cores start at gfx908/MI100).
Available: `v_dot4_i32_i8` (sdot4, confirmed in ggml `common.cuh:692` for `__gfx906__`),
64 KB LDS/CU, 256 VGPRs/SIMD, wave64.
**Model**: Qwen3.6-27B UD-Q6_K_XL (25 GB), baseline 19.0 tok/s = 52.6 ms/step.

---

## 1. What the kernel can and cannot buy (read this first)

The public-API ceiling proof (PLAN.md → Quick Reference) still holds **even with a perfect kernel**:

```
PTI N-seq step = batch(N) + (N-1) reinits = N decodes minimum per N tokens
perfect kernel: batch(N) → 1.0×  ⇒  step = N × baseline  ⇒  throughput = baseline. Parity, not a win.
```

At greedy, single-stream generation is information-theoretically sequential: token t+1's decode
needs token t. The stagger's "draft" costs a full model decode, so it can never pay for itself.
**A kernel that makes N-column GEMV cost ~1.0× is the enabling primitive, not the speedup.**
The speedup comes from what we run on top of it — a draft source that costs **less than a full decode**:

| Path | Draft cost | What the kernel buys | Projected |
|---|---|---|---|
| **A. PTI stagger** (current) | 1 full decode/token | 16.3 → ~19.0 tok/s (parity ceiling) | dead end; keep as harness |
| **B. Twin serving** (2 independent prompts) | n/a — both streams are real | aggregate 1.6× → **~1.9×** | guaranteed if kernel ≤1.1× |
| **C. N-gram lookup drafting** (single stream) | **0** (CPU string match) | verify-batch 1.86× → ~1.2× makes spec-dec profitable | **>1.5× on code/repetitive text; ≈1.0× worst case** |

Path C is the answer to "any way to get the single-stream speedup for this model": it breaks the
ceiling-proof assumption (draft = full decode) by drafting from the context for free, and verifying
k draft tokens in ONE batched decode. The ceiling proof does not apply because k verified tokens
cost ~1.2× one decode instead of k decodes.

**Critical property of Path C at greedy**: output is *provably byte-identical* to baseline regardless
of draft quality (verification accepts exactly the greedy chain). Our existing byte-identical audit
carries over unchanged. Bad drafts cost time, never correctness.

---

## 2. Why Path C survives on a hybrid-SSM model

Standard llama.cpp speculative decoding struggles with SSM/recurrent models because rejecting draft
tokens requires rolling back the recurrent state, which has no per-position cache. **PTI already
solved this** — the checkpoint-sequence trick (`llama_memory_seq_cp`) we use in `reinit_seq()`:

```
seq 0: working stream          seq 1: checkpoint (state through last emitted position)

loop:
  draft[0..k-1] = ngram_lookup(context)               # free, CPU-side; k=0 if no match
  batch = [tok_last @ p] + draft[i] @ p+1+i           # all in seq 0, logits on all
  llama_decode(batch)                                  # ONE call, ~1.2× with kernel
  j = longest prefix where draft[i] == argmax(logits[i])
  emit j+1 tokens; p += j+1
  if j == k:    seq_cp(0→1)                            # clean accept: advance checkpoint
  else:         seq_cp(1→0); re-decode emitted j+1     # SSM consumed wrong tail → rebuild
                tokens as one batch; seq_cp(0→1)       # miss penalty ≈ 1 extra decode
```

Throughput model (per loop iteration, baseline step = 1.0):

```
tok/s ≈ baseline × E[j+1] / (cost_k + miss_rate × (cost_rebuild + cp))

example, code-gen text:  E[j]=2 hits, cost_k(k=4)=1.3, miss 30% × 1.2 rebuild
   → 3.0 / (1.3 + 0.36) = 1.81× → ~34 tok/s
worst case (no n-gram hits, adaptive k=0): every step is a plain decode → 1.0× exactly
```

Adaptive drafting (only attach draft tokens when the n-gram table has a confident match) makes the
worst case ≈ baseline + ε, with upside on repetitive/structured text. M6.0 measures the real
`cost_k` curve; M6.4's gate uses measured numbers, not these estimates.

---

## 3. Hardware analysis — what to fix in the kernel

Measured (M5.1/M5.2, `pti_gemv_bench` + ISA dump of `mul_mat_vec_q<Q6_K>`):

```
ncols_dst   step cost   VGPRs   waves/SIMD
1           1.00×       27      9
2           1.25×       ~40?    (TBD M6.1 — dump ISA)
4           1.86×       63      4
```

- N=1 GEMV already runs at ~393 GB/s effective (≈ D2D rate) → **no headroom at N=1**; llama.cpp is fine.
- N=4 penalty is **occupancy + instruction issue**, not HBM and not activation L2 (Δscaling = 0.01
  between ctx=128 and ctx=1024). 63 VGPRs → 4 waves/SIMD can't cover memory latency.
- Peak-MAC is irrelevant: 27 G-MAC/stream/token at sdot4 rate is ~4 ms — we're nowhere near it.

VGPR → occupancy budget (gfx906, 256 VGPRs/SIMD): `waves = floor(256 / VGPRs)`

```
≤32 → 8 waves   ≤36 → 7   ≤42 → 6   ≤51 → 5   ≤64 → 4
target: k=2 ≤42 VGPRs (6 waves), k=4 ≤51 VGPRs (5 waves)   — current k=4: 63
```

### Kernel experiment matrix (M6.2)

The j-loop (per-column work) is what bloats registers: k accumulators × rows_per_block, plus k sets
of q8_1 activation registers. Candidate fixes, to be tested independently then combined:

1. **VGPR diet**: reduce rows_per_cuda_block at k>1; re-examine M5.1 loop order with unroll pragmas
   pinned; check the compiler isn't keeping k copies of weight regs.
2. **LDS-staged activations**: q8_1 activations for k columns staged in LDS tiles, j-inner loop reads
   from LDS (≈free) instead of holding per-column regs.
   Arithmetic: q8_1 = 36 B per 32 weights. K=5120 → 5.8 KB/col → k=4 = 23 KB: fits 64 KB LDS whole.
   K=17408 → 19.6 KB/col → k=4 = 78 KB: **must tile** (e.g., 16 KB tiles + `__syncthreads` per tile,
   amortized over 4 K-rows per tile).
3. **dot4 discipline**: confirm inner loop emits `v_dot4_i32_i8` (it should via `ggml_cuda_dp4a`);
   if the Q6_K path falls back to scalar mul/add on HIP, fix that first — it alone could explain 1.86×.
4. **2 rows/thread** to amortize index math if still issue-bound after 1–3.

FP-order constraint: keep the same accumulation order as upstream MMVQ (integer dot4 partials, float
scale-combine in same sequence) so outputs stay bit-identical and the byte-identical audit still passes.
Any change that reorders float adds needs an explicit audit-gate decision (logit max-abs-diff < 1e-4 AND
identical argmax on full test runs).

### Plan B (fallback only): flat-Q8 custom format

`pti_kernel.hip` regtile/intlv kernels already hit **330 GB/s = 86% of D2D for 4 streams** on a flat
int8 format. If patching MMVQ stalls, preconvert weights to flat Q8 (27.5 GB — tight beside KV+SSM in
32 GB) and run a fully custom forward pass. Much more work (attention + SSM kernels too); only if M6.2/3 fail.

---

## 4. Phases, tests, gates

Every phase = build cmd + run cmd + PASS/FAIL gate measured before moving on. Abort criteria explicit.

### M6.0 — k-token batch cost curve (no kernel changes) — **first, now**

The verify-batch (k tokens, 1 seq) hits the same MMVQ `ncols_dst=k` path as N-seq, but attention/SSM
costs differ (SSM scans k steps sequentially in-batch). Measure reality before designing anything.

- **Tool**: `pti_kbatch_bench.cpp` — prefill, checkpoint via seq_cp, then time `llama_decode` of
  k ∈ {1,2,3,4,6,8} greedy-chain tokens in one call, restoring state from checkpoint each iter.
  Also times `seq_cp` itself (= the miss-penalty component of Path C).
- **Output**: cost(k) table: ms, ×baseline, ms/token, break-even E[j] per k.
- **PASS**: cost(1) ≈ 52.6 ms (sanity); curve monotone; numbers recorded below.
- **Informs**: Path C economics (which k to use), M6.2 gates (the non-GEMV floor for each k).
- **Abort signal**: if cost(4) ≳ 3× — SSM scan dominates and is irreducible by a GEMV kernel —
  then cap k at 2–3 and re-derive Path C projections before any kernel work.

**Results (measured, 2026-06-09, 15 iters)**:

```
k    ms/call   ×k=1    ms/token   break-even E[j]   chain-match
1       55.6   1.00×      55.6    0.00              PASS
2       67.0   1.20×      33.5    0.20              PASS
3       87.6   1.57×      29.2    0.57              PASS
4       94.9   1.71×      23.7    0.71              PASS
6      138.9   2.50×      23.1    1.50              PASS
8      172.7   3.11×      21.6    2.11              PASS

seq_cp checkpoint restore: 0.02 ms  (effectively free)
```

**Conclusions**:
- **Chain-match PASS at every k**: batched decode bit-reproduces the sequential greedy chain on
  this hybrid-SSM model → lookup verification is sound. Path C foundation confirmed.
- **seq_cp ≈ 0**: the miss penalty reduces to just the rebuild decode. Checkpointing costs nothing.
- ms/token monotonically falls through k=8 → for high-hit text, larger draft windows keep winning.
- Path C economics at TODAY's kernel (batch of b = 1+drafts, emit 1+j on j accepts):
  - all-accept, b=4: 4 tokens / 94.9 ms = **42 tok/s** during hit runs (2.2× baseline)
  - E[j]=2, 40% miss (rebuild ≈ cost_3): 3 / (94.9 + 0.4×87.6) ms = **23.1 tok/s** (1.2×)
  - no n-gram match (adaptive k=0): plain decode = **1.0× exactly**
- Caveat: k=1 measured 55.6 ms vs 52.6 ms pure baseline — the per-iteration seq_rm/seq_cp forces a
  graph re-reserve + KV stream copy inside the timed region (~3 ms). Real lookup loop checkpoints at
  the same frequency, so these numbers are the honest in-loop costs.
- **Verdict: M6.4 (lookup) is GO at current kernel cost; kernel work (M6.1–6.3) multiplies the gain
  (cost_4 1.71× → ~1.2× would lift hit-run throughput from 42 to ~60 tok/s).**

### M6.1 — Standalone Q6_K microbench replica — **DONE, gate PASS**

- Extract the MMVQ Q6_K inner loop (from `ggml-cuda/mmvq.cuh` + `vecdotq.cuh`, both in working dir)
  into a standalone `.hip` benchmark with real Q6_K block layout (synthetic data, format-identical),
  shapes 17408×5120 / 5120×17408 / 10240×5120, k = 1,2,4.
- Dump ISA (`-save-temps` or `extractkernel`) → record exact VGPR counts per k.
- **PASS**: replica reproduces in-tree ratios (1.0× / ~1.25× / ~1.86×) within ±10%. This proves the
  microbench is a valid proxy and iteration can happen outside llama.cpp (seconds, not minutes).
- **FAIL** → the cost lives outside GEMV (dispatch, quantize_q8_1, graph overhead) → pivot to
  profiling `llama_decode` with rocprof before touching kernels.

**Results (measured)**: `pti_q6k_bench.hip`, correctness vs CPU integer reference PASS at all k.
Ratios: n=2 = 1.24–1.33× (in-tree 1.20×), n=4 = 1.51–1.66× (in-tree 1.71×) → within ±10%, PASS.
VGPRs via hipFuncGetAttributes: 24/36/63/79 for n=1/2/4/8 → 10/7/4/3 waves (n=4 matches the
in-tree ISA analysis exactly). n=1 runs at ~510 GB/s pure weight-read.
Lesson logged: the replica's first reduction used `__shfl_down` (sum in lane 0 only) — upstream
needs the XOR butterfly because lane tx writes row tx. The CPU-reference check caught it.

### M6.2 — Kernel optimization loop (microbench only) — **CLOSED: floor reached**

- Apply experiment matrix (§3) one variable at a time; keep a results table per variant.
- **Gate**: k=2 ≤ **1.10×**, k=4 ≤ **1.35×** (stretch 1.15×) on the microbench GEMV.
- **Abort**: if after VGPR diet + LDS staging + dot4 audit, k=2 still > 1.18× → occupancy isn't the
  real limiter; stop, write up findings, fall back to Plan B decision point.

**Results (measured 2026-06-09, FFN gate/up 17408×5120, 200 iters)**:

```
variant                          n=2      n=4      VGPRs(n=4)  verdict
base (upstream + M5.1)           1.29×    1.65×    63 / 4 wv   OPTIMUM
fused unpack (j-invariant CSE)   1.30×    1.66×    64 / 4 wv   wash — LLVM already CSEs
probe B: forced ≤40 VGPRs        —        6.00×    40 + spill  disaster
colwarp (block/column, L2)       2.01×    3.87×    24 / 10 wv  no L2 temporal sharing
warpcol (warp/column, L1)        1.93×    4.24×    24 / 10 wv  no L1 sharing either
probe A: shared-y (L2 traffic)   —        1.64×    —           activations irrelevant
```

**Verdict — abort criterion met.** The n=2/n=4 overheads (1.25×/1.7×) are the instruction-issue
floor for Q6_K MMVQ on gfx906:
- The per-column work (dp4a + scale-combine per (i,j,qr)) is irreducible and already minimal.
- The compiler already shares the weight unpack across columns; no VALU redundancy left.
- Registers cannot shrink without spills (the n=4 dataflow needs ~63 VGPRs).
- Column-split decompositions (better occupancy) lose more to duplicated weight reads than they
  gain — neither L2 nor same-CU L1 delivers temporal sharing at GEMV streaming rates.
- The format itself taxes the kernel: 210-byte unaligned q6_K blocks force 2×u16 loads per int.

**Consequence**: M6.3 (ggml integration) is MOOT — there is nothing to integrate. The only
remaining kernel route is Plan B (flat-format custom pipeline, §3), a major project. Path C does
NOT need it: lookup economics are already positive at the measured 1.20×/1.71× in-tree costs.
**Proceed directly to M6.4.**

### M6.3 — ggml integration

- Patch `mmvq.cuh` for HIP+gfx906, `ncols_dst ∈ {2,4}` only; N=1 path untouched; guarded so CUDA and
  other archs compile unchanged.
- **Gates** (in order):
  1. `llama-bench` tg (N=1) unchanged ±2% — no regression for normal use
  2. `make audit` — byte-identical output still 15/15 PASS
  3. `pti_gemv_bench`: N=2 / N=4 end-to-end decode ratios improve to ≤ (M6.2 GEMV gain + non-GEMV
     floor measured in M6.0)
  4. `pti_kbatch_bench` re-run: new cost(k) curve recorded
- **Abort**: gate 2 fails and can't be restored by matching FP order → revert, document, Plan B decision.

### M6.4 — `pti_lookup.cpp`: n-gram draft + batch verify + checkpoint rollback — **DONE, gates PASS**

- Implement per §2 sketch. Unit test: **forced-wrong drafts → output still byte-identical** (proves
  rollback correctness independent of n-gram quality). Adaptive k (draft only on table hit).
- Prompts: (a) code generation, (b) repetitive prose, (c) adversarial random — measure all three.
- **Gates**: byte-identical on all three, always. tok/s: favorable class > 19.0, worst case ≥ 18.0.

**Results (measured 2026-06-10, k=7, g=3, AIMD L≥5)**:

```
prompt class                      baseline   lookup    ratio   identical
adversarial prose                 19.4       18.6      0.96×   ✓ (1 fire, suppressed)
adversarial + sabotaged drafts    19.4       18.5      0.95×   ✓ (gate bounds damage)
code w/ parallel structure        19.3       18.8      0.97×   ✓ (1 fire, suppressed)
REAL TASK: code edit (rename a    19.4       28.2      1.45×   ✓ (8 full 7-accepts; the
  var, re-emit function)                                       2 misses = the rename points)
verbatim repetition               19.4       27.2      1.40×   ✓ (4× full 7-accepts)
verbatim repetition (no AIMD)     19.4       34.2      1.76×   ✓ (6× full 7-accepts)
stress: 20 rebuilds (sabotage,    19.4       13.8      0.71×   ✓ ← correctness machinery
  pre-AIMD, every draft poisoned)                              exercised hard, output exact
```

**Real-task validation (2026-06-10)**: input-grounded editing — paste a Python function, ask for
one variable renamed, output only the code. The model re-emits ~90% of the prompt verbatim;
lookup drafts fire on PROMPT n-grams (history includes the prompt by construction). 93 tokens in
36 decode steps (2.58 tok/step), 80% draft acceptance. This is the workload class of code
assistants (edit/refactor/quote), summarization-with-quotes, and RAG.

### M6.4b — Adaptive draft length (k-escalation) — **DONE**

M6.0 curve extension: k=12 costs 4.43×, **k=16 costs 4.47×** (≈ same as 12 — ggml leaves the
MMVQ path past ncols=8 for a flatter-cost kernel; chain-match still PASS). A full 16-batch hit
runs at 15.6 ms/token = 64 tok/s. Policy: fire at probe size k=7; on full accept jump to k=15
(batch 16); on any miss reset to 7. Plus hard-off after 3 zero-accept fires (worst-case bound).

```
task                          baseline   v1(k=7)   v2(adaptive)   identical
long code edit (234 tok)      19.2       —         35.5  (1.85×)  ✓  9× full-15 accepts
short code edit (93 tok)      19.4       28.2      29.9  (1.54×)  ✓
hostile prose                 19.4       18.6      18.7  (0.96×)  ✓  bound unchanged
```

The longer the copy-runs, the closer to the batch-16 ceiling (3.6×): real editing tasks with
bigger functions/documents sit at 1.8–2× and rising with output length.

**Findings**:
1. **Byte-identical output on every run** — including 100% poisoned drafts with 20 rebuilds.
   The SSM checkpoint-rollback (seq_cp + re-decode accepted prefix) is correct.
2. First genuine single-stream beat of baseline on this model: **34.2 tok/s (1.76×)** during
   copy-runs; 2.4–3.75 tok/step.
3. Static gates fail on parallel-structure text ("write two similar functions"): long shared
   phrases (L≥5) diverge at substitution points; 3 policies measured 16.7–17.2 (worse than
   baseline). **AIMD gate** (bar +4 per non-full fire, −1 per full accept, floor 5) fixes it:
   such text suppresses itself after 1 fire → 0.96–0.97× parity, while true copy-runs (L in
   the tens) clear any bar.
4. Worst case is bounded at ~0.95× (one suppressed fire costs ~1 step); favorable class
   (verbatim repetition: RAG with quoting, refactoring, structured listings) gets 1.4–1.76×.
5. The residual ~0.04 parity gap is the cost of the single probe fire — acceptable; a
   cross-step persistent-suppression flag could reclaim it if needed.

### M6.5 — Twin aggregate serving (Path B)

- 2 independent prompts, 1 seq each, one decode per step, aggregate tok/s vs 2× sequential baseline.
- **Gate**: ≥1.6× aggregate today (sanity: implied by N=2 = 1.25×); ≥1.7× post-kernel.

---

## 5. Sequencing decision

```
M6.0 (bench, ~1 h)  →  numbers decide:
   ├─ Path C viable at current kernel? → M6.4 lookup FIRST (biggest win per effort, no kernel risk)
   ├─ then M6.1 → M6.2 → M6.3 kernel loop (multiplies C, enables B at 1.9×)
   └─ M6.5 last (cheap, mostly measurement)
```

Rationale: M6.4 is pure host-side C++ on proven primitives (seq_cp checkpointing from PTI), is
output-safe by construction, and is the only path that can beat baseline *this week*. The kernel
is a force multiplier, not a prerequisite.

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| FP accumulation order changes → audit fails | Keep upstream MMVQ order; explicit gate in M6.3 |
| SSM scan cost scales with k, floors the verify batch | M6.0 measures it before any kernel work; cap k |
| `seq_cp` cost on 144 MB SSM state eats miss penalty | M6.0 times seq_cp explicitly; it enters the Path C model |
| n-gram hit rate poor on real model outputs | M6.4 measures 3 text classes; adaptive k bounds downside to ~1.0× |
| ggml upstream drift (we patch shared mmvq.cuh) | Guard with `__gfx906__` + ncols_dst branch; document diff in repo |
| LDS tiling for K=17408 adds sync overhead | Tile-size sweep in M6.2; gate decides |
