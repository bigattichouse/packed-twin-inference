# Packed Twin Inference (PTI)

**Draft forward. Verify in a packed batch. Fall back to your twin.**

Make a local LLM up to 2× faster — with provably identical output. Runs Qwen3.6-27B on a
single AMD MI50 (32 GiB) through llama.cpp. No retraining, no quality loss, no custom GPU
kernels.

The name is the design:

- **Packed** — where the speed comes from. Verifying a 32-token draft costs 6.93× one
  token, not 32×, because the 25 GB weight read is shared across every position in the
  batch. One weight load, many useful computations.
- **Twin** — why it's safe. Every draft is verified against, and healed from, a twin of
  the working sequence's state. That twin checkpoint is what makes speculation possible
  on hybrid-SSM models, where recurrent state cannot rewind — the piece stock lookup
  decoding lacks.

| what you're generating | plain | lookup only | MTP only | **PTI (lookup+MTP)** |
|---|---|---|---|---|
| Code edits (rewrite a function) | 19.3 | 37.5 | 25.2 | **39.9 (2.07×)** |
| Fresh prose / new code | 19.1 | 18.9 | 23.7 | **23.7 (1.24×)** |
| Structured / patterned output | 18.1 | 18.4 | 23.7 | **24.8 (1.37×)** |

*(tok/s, one consistent build, 2026-06-10.)* `pti` mode is best-or-tied on every text
class: the two draft sources **arbitrate** — when the MTP head disagrees with a fresh
n-gram match, the match is treated as coincidence and the MTP draft fires instead, so
MTP-only performance is the floor and copy-run performance is the ceiling.

**Exactness, precisely**: every emitted token is verified against the model's own logits
before emission — corrupted drafts can never change the output (the `--sabotage` test
proves this unconditionally). Byte-identity across modes additionally holds everywhere
except at genuine floating-point near-ties (logit gap below kernel batch-shape noise),
where modes may take different — equally greedy — branches; deterministic tie-breaking
(ε=0.05) makes these rare (one occurrence across the table above).

## The idea, in plain words

A 27B model generates one token per forward pass, and each pass reads ~25 GB of weights
from GPU memory. That memory streaming is the bottleneck — the GPU's arithmetic units are
mostly idle. The trick: **checking several proposed tokens in one pass costs barely more
than generating one** (a 16-token check costs just 4.5× one token, not 16×).

So if we can *guess* upcoming tokens for free, we batch-verify the guesses and emit every
one that matches what the model would have said anyway. Wrong guesses are discarded and
recomputed — output never changes, only the number of passes does.

Two free guess sources, combined:

1. **Lookup** — when the model is re-typing text it has already seen (editing code,
   quoting a document, structured output), the continuation is sitting right there in the
   context. A string match proposes up to 31 tokens at once.
2. **MTP head** — Qwen3.6 ships a small built-in "next-next token" predictor (one extra
   layer, ~3.5 ms vs 52.6 ms for a full pass). It guesses 1 token ahead with ~85%
   accuracy on text the lookup can't help with.

Three safety mechanisms keep it exact and fast: every guess is verified against the
model's real output before it is emitted; a confidence gate stops guessing on text where
guesses keep missing (worst case ≈ plain speed); and the recurrent-state problem that
normally breaks speculation on hybrid SSM models like Qwen3.6 is solved with a cheap
checkpoint/rollback (0.02 ms).

## Quick start

```bash
# prerequisite: upstream llama.cpp (master >= ad2775726, has the MTP/ext APIs)
# at ../llama.cpp, built with ROCm/HIP; model in ../gguf/. One OPTIONAL perf patch:
./llama.cpp-patch/patch.sh --build     # details: llama.cpp-patch/README.md
# (modified files also mirrored in llama.cpp-files/ for reading or copying)
make lookup server chat

# terminal demo (great for screen recording) — full speed, MTP-only, or plain
./pti-cli.sh pti  "Rewrite this function with better names: ..." 300
./pti-cli.sh base "same prompt" 300                  # plain llama.cpp speed
./pti-cli.sh compare "same prompt" 300               # both, back to back

# OpenAI-compatible server for your editor (port 8080)
./llama-server-pti.sh pti     # lookup + MTP   (~2× on edits)
./llama-server-pti.sh mtp     # MTP only       (the "fair" baseline if you'd always use MTP)
./llama-server-pti.sh base    # plain decode   (= stock llama-server behavior)
```

The server speaks `/v1/chat/completions` with SSE streaming. **Set temperature 0 in your
editor** — verification is greedy; requests with temperature > 0 fall back to the plain
path (logged per request). Every request logs its mode, token count, accept rates, and
tok/s to the server console, so A/B comparison is one config switch in your editor.

```bash
# correctness audit, any time:
bin/pti_lookup -m <model> -p "prompt" -n 200 --baseline > a.txt   # reference
bin/pti_lookup -m <model> -p "prompt" -n 200 --mtp      > b.txt   # fast
diff a.txt b.txt    # always empty
```

## Where the speed comes from (and where it doesn't)

- **Best case**: editing/refactoring code, summarize-with-quotes, RAG answers that cite
  passages, JSON/tables — anything where output overlaps the prompt or itself. The longer
  the overlap, the closer to the 4.3× per-pass ceiling.
- **Typical novel text**: ~1.2× from the MTP head alone.
- **Worst case**: adversarial text where every guess misses → the gates shut speculation
  off and you get ~0.97× of plain speed. You can never lose more than a few percent.
- Requires a model with an MTP head for the novel-text gains (Qwen3.6-UD has one); lookup
  works on any model.

## Prior art, and what's actually new here

The components are established, and we cite them rather than claim them:
**speculative decoding** (draft-then-verify: Leviathan et al. / Chen et al., 2023),
**prompt-lookup drafting** (Saxena, 2023 — mainline llama.cpp ships `llama-lookup`),
**MTP-head drafting** (DeepSeek-V3 proposes reusing its MTP module for speculation;
vLLM ships it for DeepSeek models on datacenter GPUs).

What this repo adds on top:

1. **Draft-verify on a hybrid-SSM model.** Recurrent state cannot rewind, which is why
   stock lookup decoding doesn't run on models like Qwen3.6. The twin-checkpoint
   rollback + merged rebuild makes it work, measured at ~2× — to our knowledge the
   first working lossless spec-dec on a hybrid-SSM model in the llama.cpp ecosystem.
2. **Byte-identity made real, not aspirational.** "Lossless" claims in spec-dec usually
   mean "same distribution in theory." We enforce `diff`-clean and documented what it
   took: Q8_0 KV is batch-size-dependent under flash attention, unified KV flips fp
   near-ties, and exact ties need deterministic tie-breaking. Anyone shipping spec-dec
   silently hits these; we measured them.
3. **The measured system around the drafts**: AIMD confidence gating, the 7→15→31 draft
   ladder, merged rebuild, MTP probe methodology — plus an honest trail
   (`FAILED_EXPERIMENTS.md`) of everything that didn't work, including the project's
   own original idea.

## Status & roadmap

Working and validated end-to-end (CLI + server, all byte-identical): the numbers in the
table above. Open items, in value order:

1. **Multi-token MTP-context batches produce garbage** (llama-ext graph bug; the
   last-pair-only workaround costs ~4 accept points and leaves cache gaps)
2. **Sampled verification** — temperature > 0 currently falls back to plain decode;
   proper speculative sampling would bring the speedup to default chat settings
3. **Context ceiling** — 49k usable today (f16, exact); a lazier/smaller MTP context or
   `--kv-q8` (which trades away cross-mode byte-identity) reaches further
4. **Persistent cross-request n-gram cache** — editor sessions resend similar code;
   drafts could fire from earlier requests' history

Declined/parked: twin aggregate serving (2-user throughput), flat-Q8 custom kernel
(the only path past the current verify-cost curve; major project).

## Repo guide

| file | what |
|---|---|
| `pti_lookup.cpp` | the algorithm — CLI with `--baseline/--mtp/--sabotage` audit modes |
| `pti_server.cpp` | OpenAI-compatible server with `--mode base/mtp/pti` |
| `pti-cli.sh`, `llama-server-pti.sh` | launch scripts |
| `pti_kbatch_bench.cpp`, `pti_mtp_probe.cpp`, `pti_q6k_bench.hip` | the measurements behind the design |
| `POSITIVE_RESULTS.md` | every win, with numbers |
| `FAILED_EXPERIMENTS.md` | every dead end, with numbers (read this before "improving" the kernel) |
| `KERNEL_PLAN.md`, `PLAN.md`, `DESIGN.md` | full investigation logs |

## History, briefly

PTI originally meant something stricter — running staggered *active* copies of the model
in one batch to self-speculate. Measurement proved that idea is mathematically bounded
*below* plain speed (the draft cost equals the savings), and that the GPU kernel had no
headroom left on gfx906. The machinery built along the way — the twin checkpoint, packed
batch verification, byte-diff auditing — became the foundation for what actually works: free
drafts from lookup + MTP. The numbers: baseline 52.6 ms/pass; batch verify b4 = 1.71×,
b16 = 4.47×, b32 = 6.93×; checkpoint 0.02 ms; MTP head 88.6% t+2 accuracy at 3.5 ms.
