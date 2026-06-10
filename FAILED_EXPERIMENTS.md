# Failed Experiments

Negative results from this project, kept specific so they don't get re-tried. Each entry:
what we believed, what we built/measured, why it failed, and the lesson.

Hardware/model for all entries: AMD MI50 (gfx906), Qwen3.6-27B UD-Q6_K_XL, llama.cpp,
greedy decoding, baseline 19.0–19.4 tok/s (52.6 ms/step).

---

## 1. The 38.1 tok/s headline — duplicate-token counting bug

**Claim**: early `pti_4seq.cpp` reported 38.1 tok/s (2× baseline).
**Reality**: on each 4-accept step it emitted 4 tokens but re-emitted 3 of the previous step's
tokens. `n_gen` counted all emit calls. Output text was garbled; unique-token throughput was far
lower. **Lesson**: every throughput claim must be anchored by a byte-diff against baseline output.
All later results enforce this.

---

## 2. PTI forward stagger — mathematically bounded below baseline

The repo's namesake idea: N staggered sequences of the same model in one batch, seq i running
i steps ahead, verify/accept chain emitting up to N tokens per step.

**Measured**: 4-seq multi-emit 14.6 tok/s (0.77×); 2-seq 16.3 tok/s (0.86×) — best possible N.
Output byte-identical, machinery correct. Still slower than baseline.

**Proof of the ceiling** (public API): after an N-emit, rebuilding the stagger needs N−1
sequential single-token decodes (each depends on the previous; SSM state forbids shortcuts):

```
step_cost = batch(N) + (N-1)×reinit(1) > 1×baseline + (N-1)×baseline = N×baseline
⇒ tok/s < baseline, for every N
```

The deeper reason: at greedy, the stagger's "draft" costs one full model decode per token —
the draft is exactly as expensive as the thing it speculates. Self-speculation with the same
model at full price can never pay for itself. **Lesson**: a draft must cost less than a decode;
that constraint led to n-gram lookup (free drafts), which works.

---

## 3. Speculative (stale-state) reinit — 47% D-miss on SSM

**Hypothesis**: skip the sequential reinit decodes; rebuild B/C/D from slightly stale states and
let verification catch errors.
**Measured**: 47% D-miss rate, 13.7 tok/s (worse than the exact 14.6). Qwen3.6's Mamba/SSM
layers cannot tolerate even one missed position update.
**Lesson**: hybrid-SSM state is exact-or-nothing. Any scheme that consumes a wrong token into a
recurrent state must rebuild that state from a checkpoint (the trick `pti_lookup` uses).

---

## 4. Ring buffer of past states for stagger rebuild — direction error

**Plan (pre-measurement)**: keep A's past states in a ring buffer to reinitialize B/C/D cheaply
(~41 tok/s projected).
**Reality**: after a 4-emit, the new B/C/D positions are 1–3 steps *ahead* of the new anchor.
Past states are all *behind*. A ring buffer of history cannot provide forward states.
**Lesson**: the forward stagger needs future state, which only sequential decodes produce.

---

## 5. ~~MTP head as a bonus-token source~~ — **CONCLUSION OVERTURNED by M7.0**

The UD-Q6_K_XL model ships a Multi-Token-Prediction head (`nextn_predict_layers=1`).
**Original claim (analytical)**: the head takes `(tok_t, h_t)` and predicts t+1 — redundant
with the main logits, not a next-next predictor.

**What was actually wrong**: that conclusion came from reading the graph wiring, which is
input-agnostic. The same wiring fed with the *just-emitted* token (DeepSeek-V3 semantics) is
a t+2 drafter. Never tested empirically — until M7.0 (`pti_mtp_probe`, 2026-06-10):

```
Variant A (same-index → t+1):  72.7%   not ~99% ⇒ NOT a redundant output head
Variant B (shifted    → t+2):  88.6%   ⇒ genuine next-next drafter, 3.5 ms/call
```

**Real lesson (replaces the old one)**: an analytical reading of graph wiring cannot determine
what a head predicts — the input convention decides, and only an empirical probe settles it.
The probe cost ~30 minutes; the wrong conclusion sat unchallenged for weeks because it was
never promoted from "analysis" to "measurement." See POSITIVE_RESULTS.md and KERNEL_PLAN.md M7.

---

## 6. GEMV kernel restructuring on gfx906 — instruction-issue floor

**Goal**: cut the N-column MMVQ overhead (n=2 = 1.25×, n=4 = 1.7–1.9×) toward 1.0× so batched
decode would be nearly free.

Validated standalone replica (`pti_q6k_bench.hip`, CPU-reference-checked, reproduces in-tree
ratios and exact VGPR counts). Then every structural variant lost:

```
variant                          n=2      n=4      verdict
base (upstream + M5.1)           1.29×    1.65×    optimum — nothing beat it
fused unpack (share across j)    1.30×    1.66×    wash: LLVM already CSEs the unpack
forced ≤40 VGPRs (waves=6)       —        6.00×    scratch spills; regs are genuinely needed
block-per-column (share via L2)  2.01×    3.87×    no temporal sharing at streaming rates
warp-per-column (share via L1)   1.93×    4.24×    warps drift; 16KB L1 can't bridge them
probe: all columns share one y   —        1.64×    activation traffic is irrelevant
```

Root cause: irreducible per-column dot-product issue (dp4a + scale-combine) at 4 waves/SIMD,
plus Q6_K's 210-byte unaligned blocks forcing 2×u16 loads per int.
Also: the original plan said "MFMA kernel" — **gfx906 has no MFMA** (matrix cores start at
gfx908/MI100); only sdot4 exists, and the kernel already uses it.
**Lessons**: (a) measure with cheap probes before designing (the shared-y and forced-occupancy
probes killed two whole design branches in minutes); (b) upstream MMVQ is already at the floor
for this format/arch; reopening this requires the flat-format custom pipeline (Plan B in
KERNEL_PLAN.md), not MMVQ tweaks.

---

## 7. Static gates for n-gram lookup drafting — lose to parallel-structure text

Three static firing policies for `pti_lookup`, all measured **below** baseline (19.3) on a
code prompt with deliberate parallel structure ("write two similar functions"):

```
policy                              tok/s   why it failed
g=3 probe with g=2 fallback         17.2    coincidental 2-gram matches, 33% accept
strict g=3 (no fallback)            16.7    3-gram matches still coincidental
static suffix-length gate L≥5       16.9    shared phrases pass L≥5 then diverge at
                                            the substitution points (row→key)
```

The cost asymmetry makes partial accepts net-negative: a miss pays the full verify batch plus
a rebuild decode. With k=3: acc=3 → +115 ms, acc=2 → −24, acc=1 → −57, acc=0 → −98.

**Fix that worked (AIMD)**: ratchet the suffix-length bar +4 on every non-full fire, decay −1 on
full accepts. True copy-runs have suffix matches in the tens and clear any bar; coincidence-prone
text self-suppresses after one fire → 0.96–0.97× parity, upside intact.
**Lesson**: no static threshold separates copy-runs from parallel structure; the *response to
failure* has to be part of the policy.
