# packed-twin-inference

**Lossless speculative decoding for hybrid-SSM models — 1.85× on real editing tasks,
byte-identical output, public llama.cpp API only.**

Target: Qwen3.6-27B (hybrid attention + Mamba/SSM) on AMD MI50 (gfx906, 32 GiB).
The working result is **`pti_lookup`** — n-gram lookup speculative decoding with SSM-safe
rollback. No custom kernels, no model changes, no quality loss.

## Results (June 2026, greedy, measured)

| task | baseline | pti_lookup | ratio | output |
|---|---|---|---|---|
| Code edit, 25-line function (234 tok) | 19.2 | **37.0** | **1.93×** | byte-identical |
| Code edit, short function (93 tok) | 19.4 | 29.9 | 1.54× | byte-identical |
| Verbatim repetition (best case) | 19.4 | 34.2 | 1.76× | byte-identical |
| Hostile prose (worst case) | 19.4 | 18.8 | 0.97× | byte-identical |
| Sabotage: 100% poisoned drafts | 19.4 | 18.8 | 0.97× | **byte-identical** |

The speedup comes from **copy-runs** — spans the model re-emits from the prompt or its own
output. Editing, refactoring, summarize-with-quotes, RAG answers quoting passages, structured
output: real gains that grow with output length. Novel free-form prose: no gain, bounded ~4%
loss. The sabotage row is the correctness control: every draft deliberately corrupted, output
still exact (slower is the only penalty bad drafts can inflict).

## How it works

```
seq 0: working stream        seq 1: checkpoint (clean state at position p)

loop:
  draft[0..k-1] = n-gram match against token history     # CPU string match — FREE
  decode [tok_last@p, draft tokens] in ONE batch          # sub-linear: batch16 = 4.47× cost
  accept the longest verified prefix, emit accepted+1 tokens
  full accept  → state is clean, continue (next fire jumps to k=15)
  partial/miss → seq_cp(checkpoint→working), re-decode accepted prefix
```

Four pieces make it work on a hybrid-SSM model where stock lookup decoding can't:

1. **Batched verification is exact** — a k-token batch reproduces the sequential greedy chain
   bit-for-bit (`pti_kbatch_bench` chain-match, PASS at every k), and costs far less than k
   decodes: batch 4 = 1.71×, batch 8 = 3.11×, batch 16 = 4.47×, **batch 32 = 6.93×**
   (12.1 ms/token — a full 32-hit runs at 82 tok/s).
2. **SSM-safe rollback** — recurrent state can't rewind per-position. A checkpoint sequence
   (`llama_memory_seq_cp`, measured 0.02 ms) restores exact state after a miss; the accepted
   prefix rides as logits-free tokens at the front of the next batch (merged rebuild — one
   sub-linear call instead of two).
3. **AIMD confidence gate** — every failed fire raises the suffix-match bar (+4), full accepts
   decay it (−1). True copy-runs (matches in the tens of tokens) clear any bar; coincidence-prone
   text self-suppresses after one fire. Static thresholds lose (see FAILED_EXPERIMENTS.md §7).
4. **Draft-length ladder** — probe at k=7; full accepts escalate 7 → 15 → 31; any miss resets.
   Deep batches only fire inside runs proven at the previous rung. Hard-off after 3 dead fires
   bounds hostile text.

**Why output is provably identical**: every emitted token is the argmax of a logit row whose
input prefix is fully verified. Drafts only decide *which positions get computed in the same
batch* — never what gets emitted.

## Quick start

```bash
# prerequisite: llama.cpp built at ../llama.cpp/build (ROCm/HIP)
make lookup

# speculative run
bin/pti_lookup -m ../gguf/Qwen3.6-27B-UD-Q6_K_XL.gguf -p "your prompt" -n 200

# reference run (also the byte-diff source)
bin/pti_lookup -m ... -p "your prompt" -n 200 --baseline

# correctness stress: poison every draft, output must not change
bin/pti_lookup -m ... -p "your prompt" -n 200 --sabotage

# flags: -k max draft (7), -g probe n-gram (3), -L initial bar (5), --verbose per-fire log
```

`pti_lookup` prints generated text to stdout and stats (accept histogram, rebuilds, tok/s) to
stderr, so `diff <(run --baseline) <(run)` is the audit.

## Roadmap

- [x] Draft window to k=31 — ladder 7→15→31, full 32-batch hit = 82 tok/s
- [x] Merged rebuild — miss penalty folded into the next verify batch
- [ ] Possible next: persistent cross-prompt n-gram cache; sampled (non-greedy) verification
- Twin aggregate serving (2 users, one weight stream, ~1.6×) — understood, not pursued

The drafted-step cost is now near-optimal; remaining time is dominated by the novel-text
portions of the output, which is the structural floor for lookup-based drafting.

## Project history

This repo began as Packed Twin Inference: N staggered sequences of the same model
self-speculating in one batch. The investigation proved that design is bounded **below**
baseline through the public API (the draft costs exactly what it saves), and that the gfx906
GEMV kernel is at its instruction-issue floor. The machinery built along the way — checkpoint
sequences, batched verification, byte-identical auditing — became `pti_lookup`, where the
draft is free and the math flips positive.

- `POSITIVE_RESULTS.md` — what worked, with the measurements that back it
- `FAILED_EXPERIMENTS.md` — every negative result, with measurements and lessons
- `KERNEL_PLAN.md` — the M6 investigation log: cost curves, kernel experiments, lookup gates
- `PLAN.md` — full PTI-era log: stagger design, ceiling proof, audit framework
- `DESIGN.md` — original architecture rationale

## Files

| file | what | status |
|---|---|---|
| `pti_lookup.cpp` | lookup spec-dec, SSM-safe — **the result** | ✓ 1.85× real tasks, identical |
| `pti_kbatch_bench.cpp` | batch cost curve + chain-match probe | ✓ foundation measurements |
| `pti_q6k_bench.hip` | Q6_K MMVQ replica + kernel experiments | ✓ floor documented |
| `pti_2seq.cpp` / `pti_4seq.cpp` | PTI stagger (correct, slower than baseline) | archived result |
| `pti_debug.cpp` | single-seq baseline driver | ✓ |
| `pti_gemv_bench.cpp` | N-seq decode scaling bench | ✓ |
| `pti_server.cpp` | HTTP server (single-seq) | ✓ |
| `audit.sh` | end-to-end correctness/perf audit (PTI era) | ✓ 15/15 |

## Key numbers

```
Hardware:        MI50 (gfx906), 32 GiB HBM2, no MFMA, wave64
Model:           Qwen3.6-27B UD-Q6_K_XL (25 GB), hybrid attention+SSM
Baseline:        19.0–19.4 tok/s (52.6 ms/step)
Batch decode:    b2 1.20×  b4 1.71×  b8 3.11×  b16 4.47×  b24 5.90×  b32 6.93×  (chain-exact)
seq_cp:          0.02 ms (checkpointing is free)
pti_lookup:      37.0 tok/s on real code edit (1.93×); 0.97× bounded worst case
Hit-run ceiling: 12.1 ms/token in a full 32-batch = 4.3× during copy-runs
```
