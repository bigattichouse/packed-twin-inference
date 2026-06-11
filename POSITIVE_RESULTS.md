# Positive Results

What worked, with the measurements that back it. Counterpart to `FAILED_EXPERIMENTS.md`.

Hardware/model for all entries: AMD MI50 (gfx906), Qwen3.6-27B UD-Q6_K_XL, llama.cpp public API,
greedy decoding, baseline 19.0–19.4 tok/s (52.6 ms/step).

---

## 1. n-gram lookup speculative decoding beats baseline on real tasks — `pti_lookup`

The headline result. Drafts cost zero model decodes (string match against token history,
which includes the prompt); one batched decode verifies all drafts; SSM-safe rollback on miss.

```
task                                  baseline   lookup    ratio   output
code edit, 25-line function (234 t)   19.2       35.5      1.85×   byte-identical
code edit, short function (93 t)      19.4       29.9      1.54×   byte-identical
verbatim repetition (best case)       19.4       34.2      1.76×   byte-identical
hostile prose (worst case)            19.4       18.7      0.96×   byte-identical
```

Why it escapes the PTI ceiling proof: the proof assumes a draft costs one full decode per
token. A string match costs nothing, so every accepted draft is pure profit.
The real-task accept histogram is the strongest evidence it does real work: on the rename-edit
task, 8 full 7-token accepts on unchanged lines and exactly 2 misses — at the two rename points.

## 1b. MTP head drafting — 2.06× on real tasks, novel text flipped positive (M7)

The model's `nextn` head, probed empirically (M7.0), is a **t+2 drafter at 88.6%** when fed
the just-emitted token — overturning the earlier analytical conclusion (FAILED_EXPERIMENTS §5).
One MTP call = 3.5 ms vs 52.6 ms full pass. Integrated as the novel-text draft source
(lookup keeps copy-runs):

```
                  hostile prose      long code edit
lookup            18.8  (0.97×)      37.0  (1.93×)
lookup+MTP        23.6  (1.22×)      39.6  (2.06×)    ← 2× on a real task
MTP alone         24.3  (1.25×)      —
```

All byte-identical, including sabotage of both draft sources (MTP self-disables via its
<30%-accept guard). Live MTP accept: 81–87%. Caveat discovered: multi-token batches in the
MTP context produce garbage (≈48% live accept, alternating pattern) — worked around with
last-pair-only feeds (flat 3.5 ms/step, −4 accept points); root-cause fix in the ext graph
is open.

## 1c. MTP arbitration — mtp mode becomes the floor of pti mode (M7.3)

The MTP candidate predicts the same position as the first lookup token, so when the
n-gram fires at the probe rung, the two draft sources vote: agreement keeps the lookup
draft; disagreement vetoes it as coincidence and fires the MTP draft instead. Escalated
ladder rungs (15/31) skip the veto — earned trust outweighs one ~85%-accurate vote.

```
                  base   lookup-only   mtp-only   pti(arbitrated)
code edit         19.3   37.5          25.2       39.9  (2.07×)
fresh prose       19.1   18.9          23.7       23.7  (1.24×)
structured        18.1   18.4          23.7       24.8  (1.37×)
```

pti went from losing to mtp-only on novel text to best-or-tied on every class — and the
veto *improved* the edit task (doomed probe fires became MTP 2-emits).

## 1d. Sampled verification — full speedup at coding temperatures (M7.4)

Sample-and-match: every verified position samples from the target logits; drafts are
accepted while the sample agrees. For deterministic drafts this is the optimal rejection
scheme — output is exactly plain temperature sampling. Position-keyed RNG
(`splitmix64(seed ^ pos)`) makes seeded runs reproducible and lets all modes consume
randomness identically.

```
code edit  τ=0.25:  base 18.5 → pti 37.0  (2.00×)  byte-identical to seeded baseline
prose      τ=0.70:  0.96× (floor holds; MTP accept drops to 56%)
sabotage   τ=0.25:  all drafts poisoned → byte-identical
server API τ=0.25:  35.6 tok/s end-to-end
```

## 2. Batched verification is EXACT on a hybrid-SSM model, and sub-linear in cost

`pti_kbatch_bench` chain-match: a k-token single-sequence batch reproduces the sequential
greedy chain argmax-for-argmax at every k tested — the foundational soundness requirement for
batch verification on a recurrent model, confirmed empirically.

```
batch size:   1      2      4      8      12     16
cost:         1.00×  1.20×  1.71×  3.11×  4.43×  4.47×
ms/token:     55.7   33.5   23.7   21.6   20.6   15.6
chain-match:  PASS   PASS   PASS   PASS   PASS   PASS
```

Bonus discovery: batch 16 costs the same as batch 12 (ggml leaves the MMVQ path past
ncols=8 for a flatter-cost kernel) → a full 16-batch hit runs at 15.6 ms/token.

## 3. SSM-safe rollback via checkpoint sequence — `seq_cp` is free

Recurrent state cannot rewind per-position, which is why stock lookup decoding can't run on
hybrid models. Pattern that works: keep a checkpoint sequence; on miss, `seq_cp(ckpt→work)`
plus ONE batched re-decode of the accepted prefix rebuilds exact state.
Measured: `llama_memory_seq_cp` ≈ **0.02 ms** on a 27B hybrid model — checkpointing costs
nothing; the miss penalty is only the rebuild decode.

## 4. The sabotage test — correctness machinery proven independent of draft quality

`--sabotage` corrupts every draft token. Result: 60 poisoned drafts, 0 accepted, 20 rebuilds —
**output still byte-identical** (13.8 tok/s; bad drafts cost time, never correctness).
This is the control that distinguishes a real lossless speculator from a lucky one, and it
validated the rollback path under maximum stress.

## 5. AIMD confidence gate — bounded worst case without losing the upside

Static firing thresholds all lost to parallel-structure text (FAILED_EXPERIMENTS.md §7).
Adaptive bar: +4 per non-full fire, −1 per full accept, floor 5, hard-off after 3 dead fires.
Measured: hostile text fires once then self-suppresses → 0.96× parity; copy-runs (suffix
matches in the tens) clear any bar → upside intact. The insight: the match-length
*distribution* separates copy-runs (L≈tens) from coincidence (L≈5–10); the response to failure
must be part of the policy.

## 6. Adaptive draft length — escalate to batch 16 inside confirmed runs

Probe at k=7; on full accept jump to k=15 (batch 16, same cost as 12); reset on miss.
Long edit task improved 28.2 → 35.5 tok/s (1.45× → 1.85×); 9 full 15-draft accepts.

## 7. Byte-diff audit methodology

Every throughput claim is anchored by `diff` of stdout against the baseline run (born from the
38.1 tok/s counting-bug burn, FAILED_EXPERIMENTS.md §1). Three guards on every result:
unique-count anchored by identical bytes, monotonic wall-clock around the loop, and a control
that reports its own penalty (sabotage). This methodology caught real bugs (e.g., the replica
kernel's wrong warp reduction, §9).

## 8. PTI stagger machinery is correct (just not faster)

The 4-seq/2-seq stagger emits byte-identical output at 100% accept with multi-emit and exact
reinit — the accept-chain logic, SSM cell priming, and state bookkeeping all work. 16.3 tok/s
(2-seq) is the proven optimum of that design class; the ceiling is mathematical, not a bug
(FAILED_EXPERIMENTS.md §2). The checkpoint/rebuild machinery built here is what `pti_lookup`
reuses.

## 9. Standalone kernel replica methodology — `pti_q6k_bench.hip`

A CPU-reference-checked replica of ggml's `mul_mat_vec_q<Q6_K>` reproduced in-tree scaling
ratios within ±10% and exact VGPR counts (63 @ n=4 → 4 waves/SIMD). Iteration time per kernel
experiment dropped from a llama.cpp rebuild (~minutes) to ~30 s, and the CPU reference
immediately caught a reduction bug (shfl_down vs the XOR butterfly that upstream's
lane-indexed epilogue requires). The kernel search itself failed (floor reached), but the
methodology is reusable for any future kernel work.

## 11. Serving the speedup: pti_server + the exactness engineering it took (M7.2)

OpenAI-compatible server (`pti_server`, `llama-server-pti.sh`) with three switchable modes —
base / mtp / pti — for controlled A/B in an editor. Measured through the HTTP API:
edit task base 18.0 → **pti 36.6 tok/s (2.03×)**, all modes byte-identical.

Two findings required to get cross-mode identity, both measured here:
- **Q8_0 KV + flash attention is batch-size-dependent**: the same position decoded in a
  1-token vs 2-token batch can argmax differently. f16 KV is required for exactness.
- **Even at f16, genuine fp near-ties flip with batch shape** (different kernel configs
  differ by ~1e-3 on logits; the `<think>` open/close decision on chat templates is a
  reliable knife-edge). Fix: deterministic tie-breaking — among tokens within ε=0.05 of
  the max logit, lowest id wins, applied identically in every mode. Restores byte-identity
  unconditionally.
- Also: `kv_unified=true` flips ties too; the proven-exact config is non-unified + f16
  (usable context = n_ctx/2 because the checkpoint takes the second stream).

## 10. M5.1 — i-outer/j-inner loop order in MMVQ (the one kernel win)

Reordering the inner loops so the weight-block address is loop-invariant across columns let
LLVM hoist the block load: N=4 decode overhead 1.95× → 1.86×. Small, real, upstreamable —
and the only positive result from the entire kernel investigation.

## 12. Packed agents — four cooperating streams at ~2× aggregate (PA.0)

The packed-batch economics that verify drafts also run *independent* agents. Four unrelated
prompts decode in ONE `llama_context`, one batched `llama_decode` per step (one token per
live stream); finished streams drop out and the survivors continue.

```
            tok/s   tokens   wall
sequential  19.3    384      19.8s    ← 4 prompts one at a time (= baseline)
packed      37.7    384      10.2s    ← 4 streams, one batched decode/step
aggregate   1.95×            (predicted 2.15× from M5.1's 1.86× 4-seq step)
```

This is the **all-in** number: it includes the per-step 4× full-vocab argmax (~151k logits
× 4) and string bookkeeping on the CPU side, so the effective 4-seq step came to ~2.05× one
stream (vs the 1.86× microbench) — the gap is real per-step overhead, not the kernel. One
model in VRAM (~24 GB, loaded once); the four lanes add only KV/SSM state, which at fixed
`n_ctx` is a *subdivision* of an already-paid pool. Substrate for the cooperative roadmap
(`PACKED_AGENTS.md` → `spec/PACKED_AGENTS_DESIGN.md`: boss decomposes → 3 workers → gather).
Build/run: `make agents && make agents-run`. **Byte-identity gate: PASS** (2026-06-11) —
packed lanes byte-identical to solo on all 4 streams (96 tok each), asserted in-binary
(`kv_unified=false` + f16 KV + ε=0.05 tie-break; the binary exits non-zero on divergence).
This is the correctness foundation for cooperation: a lane's output does not depend on which
siblings share its batch.

### Scaling streams: `-s` is configurable (1..16), but 4 is the sweet spot — measured

Sweeping lanes (UD-Q6_K_XL, n=96/lane), packed *absolute* throughput and the gate:

| N  | packed tok/s | speedup vs true 19.3 baseline | byte-identity gate |
|----|------|------|------|
| 4  | **37.4** | **1.93×** | PASS (4/4 identical) |
| 8  | 21.5 | 1.11× | FAIL (2/8 lanes) |
| 16 | 26.5 | 1.37× | FAIL (3/16 lanes) |

More lanes do **not** win, for two measured reasons:

1. **Idle-sequence SSM tax.** The *sequential* (one-live-lane) rate itself falls as
   `n_seq_max` grows — 19.3 → 13.6 → 7.5 tok/s at N=4/8/16 — though only one lane is ever
   live. On this hybrid-SSM model with non-unified KV, the recurrent state is processed for
   **every allocated sequence each decode step**, live or not. So packed throughput peaks
   near N=4, and the in-binary "aggregate" (packed ÷ same-config sequential) is *inflated*
   at high N because its denominator is degraded by the same tax (N=16 prints 3.54× but is
   only 1.37× vs the real baseline). The k-batch microbench has no such tax — k tokens share
   ONE sequence — which is why it over-predicted high-N gains.
2. **Byte-identity breaks past 4.** At N=8/16 a few lanes diverge from their solo run, but
   the divergences are **equally-valid near-ties** (e.g. solo `' etc'` vs packed `' but'`) —
   the same "fp near-tie wider than ε=0.05" bound the project ships under, just more frequent
   at higher batch occupancy. Verification correctness is intact; strict byte-*determinism*
   is not, above 4.

Conclusion: keep `-s` for experiments, but **4 is the operating point** — it maximizes packed
throughput *and* is the largest gate-clean (byte-identical) lane count. This matches the
coordinator + 3-workers design; now measured, not assumed.
