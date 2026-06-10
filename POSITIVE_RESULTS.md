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

## 10. M5.1 — i-outer/j-inner loop order in MMVQ (the one kernel win)

Reordering the inner loops so the weight-block address is loop-invariant across columns let
LLVM hoist the block load: N=4 decode overhead 1.95× → 1.86×. Small, real, upstreamable —
and the only positive result from the entire kernel investigation.
