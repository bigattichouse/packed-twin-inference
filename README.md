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
proves this unconditionally, at any temperature). Byte-identity across modes additionally
holds everywhere except at genuine floating-point near-ties (logit gap below kernel
batch-shape noise), where modes may take different — equally valid — branches;
deterministic tie-breaking (ε=0.05) makes these rare. At temperature > 0, sampling is
position-seeded: a fixed seed reproduces a run byte-for-byte (τ=0.25 code edit: pti
output identical to the seeded plain run), with occasional cross-mode flips at high
temperatures where samples land near CDF boundaries.

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

# OpenAI-compatible server for your editor (port 8080) — any temperature
./llama-server-pti.sh pti     # lookup + MTP   (~2× on edits)
./llama-server-pti.sh mtp     # MTP only       (the "fair" baseline if you'd always use MTP)
./llama-server-pti.sh base    # plain decode   (= stock llama-server behavior)

# interactive chat with live mode/temp switching (llama-cli equivalent)
bin/pti_chat -m ../gguf/Qwen3.6-27B-UD-Q6_K_XL.gguf --mode pti -t 0.25
#   > your message ...      /mode base|mtp|pti    /temp 0.7    /clear    /quit
```

The server speaks `/v1/chat/completions` with SSE streaming, **at any temperature** —
sampled verification keeps speculation active (τ=0.25 code edit: 35.6 tok/s through the
API; a fixed request `"seed"` reproduces a run exactly). Every request logs its mode,
temperature, seed, accept rates, and tok/s to the server console, so A/B comparison is
one config switch in your editor.

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

1. **Prompt cache for the server** *(next up)* — pti_server re-prefills every request;
   at big contexts the prefill dominates editor latency (llama-server reuses slots).
   Hybrid-SSM constraint: recurrent state can't rewind, so the cache can fast-path
   conversation *extensions* (delta prefill) and must full-prefill on divergence.
2. **Multi-token MTP-context batches produce garbage** (upstream graph issue; the
   last-pair-only workaround costs ~4 accept points and leaves cache gaps)
3. **Context ceiling** — 49k usable exact; `--kv-q8 -c 196608` verified working (~96k
   usable, MTP active) at the cost of cross-mode byte-identity
4. ~~Sampled verification~~ — **done** (M7.4): speculation active at any temperature;
   τ=0.25 code edit keeps the full 2.0×, seeded runs reproduce byte-for-byte
5. **Packed agents** *(active direction)* — the same packed-batch economics run *cooperating*
   agents, not just drafts: a boss decomposes a coding task, workers build in parallel, one
   batched decode per step. PA.0 substrate measured **1.95× aggregate** (4 streams; one model
   in VRAM, lanes add only KV). Now a full autonomous arc on the `--tools --no-stream` path:
   - **PA.0–PA.2** plan → parallel → gather to `--out` (2026-06-14).
   - **PA.3** MTP spec-dec (behind `--mtp`; net-slower on packed — `spec/PA3_MTP_DESIGN.md`).
   - **PA.4** coordination: test-gen → verify → L1 repair → boss arbiter (work-order rework). Fresh
     sessions get the **full triad + collaborator code** (goal/contract + blueprint + module + test +
     the modules it calls); thinking repair lanes; escalate to the boss early when L1 stalls; the
     **designer dictates libraries** (contract pins deps/CommonJS); untested modules re-queue test-gen.
   - **PA.5** tool calls (`create_file`/`execute_bash`, nanocoder convention).
   - **PA.6** staged pipeline: triage → parallel design → reconcile (contract) → implement →
     test-gen → verify → repair (`spec/PA6_PIPELINE_DESIGN.md`); **prefix-cached pools** (shared
     goal/contract/blueprints prefill once per stage, delta-prefill per item — fixes the scale
     prefill bottleneck, smoke-confirmed); GPU end-to-end validating.
   - **PA.7** *(core built; integration pending)* **eager scheduling**: dissolve the stage barriers —
     artifact-gated ready-queue, reconcile = the first rework pass. Pure scheduler + makespan sim
     (`--eager-test`, **eager 18% < barriers** on the measured spread) and active-retrieval primitives
     (`read_file`/`abandon`, `--gather-test`) are in; the run_pool mid-stream loop is next.
     (`spec/PA7_PIPELINING_DESIGN.md`)
   - Harnesses: `flappy_validate.sh` (greenfield, coupled — packed's worst case) and
     `scale_validate.sh` (many independent modules — packed's sweet spot).

   Design: `PACKED_AGENTS.md` → `spec/PACKED_AGENTS_DESIGN.md` → the per-milestone `spec/PA*.md`.

Declined/parked: twin aggregate serving (2-user throughput), flat-Q8 custom kernel
(the only path past the current verify-cost curve; major project).

## Repo guide

| file | what |
|---|---|
| `pti_lookup.cpp` | the algorithm — CLI with `--baseline/--mtp/--sabotage/-t` audit modes |
| `pti_server.cpp` | OpenAI-compatible server, `--mode base/mtp/pti`, any temperature |
| `pti_chat.cpp` | interactive chat (llama-cli equivalent): live `/mode`, `/temp`, per-turn stats |
| `pti_agents.cpp` | packed-agents binary (PA.0–PA.6): boss triage → parallel design/implement → reconcile → test-gen → verify → repair; `--tools --no-stream` staged pipeline + gather to `--out` |
| `pti-cli.sh`, `llama-server-pti.sh` | launch scripts |
| `llama.cpp-patch/`, `llama.cpp-files/` | the one (optional) llama.cpp patch + mirrored files |
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
