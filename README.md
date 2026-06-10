# packed-twin-inference

**Make a local LLM up to 2× faster — with provably identical output.**

Runs Qwen3.6-27B on a single AMD MI50 (32 GiB) through llama.cpp. No retraining, no
quality loss, no custom GPU kernels.

| what you're generating | plain llama.cpp | this project | speedup |
|---|---|---|---|
| Code edits (rewrite a function) | 19.2 tok/s | **39.6 tok/s** | **2.06×** |
| Fresh prose / new code | 19.4 tok/s | **23.6 tok/s** | **1.22×** |
| Highly repetitive output | 19.4 tok/s | 34.2 tok/s | 1.76× |

Every number above produced **byte-identical output** to plain llama.cpp — checked with
`diff` on every run, including a stress test where every speculative guess is deliberately
corrupted (the output still comes out exact; bad guesses only cost time).

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
# prerequisite: llama.cpp built at ../llama.cpp/build (ROCm/HIP), model in ../gguf/
make lookup server

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

This repo started as "Packed Twin Inference" — running staggered copies of the model in
one batch to self-speculate. Measurement proved that idea is mathematically bounded
*below* plain speed (the draft cost equals the savings), and that the GPU kernel had no
headroom left on gfx906. The machinery built along the way — checkpointing, batched
verification, byte-diff auditing — became the foundation for what actually works: free
drafts from lookup + MTP. The numbers: baseline 52.6 ms/pass; batch verify b4 = 1.71×,
b16 = 4.47×, b32 = 6.93×; checkpoint 0.02 ms; MTP head 88.6% t+2 accuracy at 3.5 ms.
