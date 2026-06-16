# Packed Agents — spec (draft)

**One model, one GPU, four cooperating agent streams in a single batched decode.**
Coordinator + 3 workers progress simultaneously; a 4-stream step costs **1.86×** one
stream (measured, M5.1) — so four agents run at **2.15× aggregate efficiency**, and
splittable tasks finish in roughly the time of their longest piece instead of the sum.
**PA.0 demo measured 1.95× end-to-end (2026-06-10), all-in with CPU-side argmax overhead.**
Detailed design: [`spec/PACKED_AGENTS_DESIGN.md`](spec/PACKED_AGENTS_DESIGN.md).

## Why this is feasible today (measured foundations)

| fact | source |
|---|---|
| 4 sequences decode in ONE `llama_decode` at 1.86× cost | pti_4seq / M5.1, measured |
| Each seq has independent KV + SSM state | pti_4seq machinery, proven |
| Mid-flight token injection into any seq = just prefill into that seq | same plumbing as our delta prefill |
| Per-stream MTP drafting stacks: batch 8 @ 3.11× → ~2.4× aggregate | M6.0 curve + M7.1 accept rates |
| Position-keyed sampling: per-stream seeds, reproducible runs | M7.4 |

The original "packed twin inference" idea returns — but instead of self-speculation
(proven bounded below baseline) or independent users (declined), the streams **cooperate
on one task**. The economics that killed the stagger don't apply: every stream does real,
wanted work.

## Honest economics (before anyone says 4×)

- Worker phase wall-clock vs one sequential agent doing all three pieces:
  `3·L sequential vs max(L_i)·1.86 packed` → **~1.6× wall speedup** for balanced pieces.
- With the coordinator productively occupied during the worker phase (tests, docs,
  integration scaffolding): **4 / 1.86 ≈ 2.15× effective work rate**.
- Stacking per-stream MTP drafts (batch 8 ≈ 3.11×, ~1.9 tok/step/stream):
  `4 × 1.9 / 3.11 ≈ **2.4× aggregate token rate**` — before any lookup hits.
- "4×" is the in-flight parallelism, not the speedup. The speedup is ~1.6–2.4×,
  which is still the largest single lever on the board for splittable tasks.

## Architecture

```
seq 0: COORDINATOR        seq 1..3: WORKERS
(one llama_context, n_seq_max = 4..8; per-stream sampling seeds)

Phase 1  PLAN      coordinator alone: emit a JSON work order
                   {shared_context, pieces: [{id, instructions, interface}, ×3]}
Phase 2  FAN-OUT   prefill each worker seq: shared preamble + its piece
                   (workers' prompts share the task header → batched prefill)
Phase 3  PARALLEL  one llama_decode per step, one token per LIVE stream;
                   streams hit EOG independently; batch shrinks as they finish
         REFILL    coordinator keeps a backlog; a finished worker slot gets the
                   next piece injected (keeps the batch full — stragglers are
                   the main efficiency leak)
Phase 4  GATHER    workers' outputs are injected into the coordinator as
                   messages; coordinator integrates / resolves conflicts /
                   emits the final artifact
```

### Coordination primitives (all already exist in our codebase)

- **inject(seq, text)**: tokenize + prefill into that seq at its current position —
  identical mechanics to the M7.5 delta prefill.
- **broadcast**: coordinator finishes the class skeleton in Phase 1; it is part of
  every worker's preamble. Mid-flight broadcast (v2): pause at a checkpoint round,
  inject into all workers, resume.
- **checkpoint rounds (v2)**: every N tokens, coordinator inspects worker tails and may
  inject guidance ("stop, the signature changed") — possible because all streams pause
  between steps anyway; it's one batch boundary.

### Speculation stacking (v2)

Each stream can carry its own MTP draft token: batch = up to 8 (4 real + 4 drafts) at
3.11× → ~2.4× aggregate. Lookup drafting also applies per stream (workers writing
similar functions hit each other's... no — histories are per-stream; lookup hits within
a stream's own preamble+output). Rollback needs a checkpoint seq per stream →
n_seq_max = 8, usable ctx = n_ctx/8: speculation costs context. v1 ships WITHOUT
speculation (plain 4-stream, n_seq_max=4, ctx/4 each); v2 adds it behind a flag.

## Interface (v1)

```
bin/pti_agents -m model.gguf -p "task description" [-n max-per-piece] [--workers 3]
```
- stdout: phase-tagged stream (`[plan] … [w1] … [w2] … [gather] …`), final artifact last
- stderr: per-phase stats (tok/s per stream, aggregate, refills)
- exit artifact: gather output written to a file via --out

## Risks / open questions

1. **Split quality** — the whole win depends on the coordinator producing genuinely
   independent pieces with clean interfaces. Mitigation: strict plan schema; pieces
   must declare their exported symbols; gather phase resolves collisions.
2. **Stragglers** — unequal pieces idle the batch. Mitigation: work-queue refill; plan
   prompt asks for similar-sized pieces.
3. **Worker context isolation** — workers can't see each other mid-flight (by design);
   overlapping helpers get deduped at gather. Acceptable for function-level splits.
4. **Memory** — n_seq_max=4 non-unified → usable ctx/4 (16k each at -c 64000 q8).
   Worker contexts are small (preamble + piece); the coordinator carries the big context.
   Asymmetric ctx per seq is not supported by llama.cpp — all streams pay the same
   reservation. Fine at v1 scale.
5. **Failure modes** — a worker going off the rails burns its stream until EOG/cap;
   checkpoint rounds (v2) give the coordinator a kill/retry switch.

## Milestones

- **PA.0** ✓ *(2026-06-10)* — plumbing demo (`pti_agents.cpp`): 4 independent prompts, one
  context, parallel decode. **Measured 1.95× aggregate** (19.3 → 37.7 tok/s); four
  independent buffers. **Byte-identity gate PASS** (2026-06-11, N=4).
- **PA.1** ✓ *(2026-06-14)* — phased pipeline plan → fan-out → parallel → **gather**.
  PA.1a+b: boss decomposes → lanes generate (`-p "task"`). **PA.1c**: boss merges the pieces
  into one `--out` artifact (both pipelines); `--gather-test` 10/10; streaming `--out`
  smoke-tested. Straggler + split-quality remain the open risks (design §7).
- **PA.2** ✓ *(2026-06-11)* — work-queue refill (`--pool`, 34.2 tok/s) + PA.2.1 prefix cache;
  Q8 KV → 128k verified.
- **PA.3** ✓ *(2026-06-15, behind `--mtp`)* — speculation stacking (MTP second context + nextn head,
  checkpoint-seq rollback). **MEASURED net-slower on packed** — doubling `n_seq_max` taxes every lane's
  SSM state more than spec-dec saves; kept as an opt-in + diagnostic. `--mtp-test` 8/8. See
  `spec/PA3_MTP_DESIGN.md`.
- **PA.4** ✓ *(2026-06-15/16)* — coordination as a **verify→repair loop**, not just Q&A: harness
  test-gen → run tests → L1 worker amend → **L2 boss arbiter** (re-queues rework via a **work-order**,
  not a bespoke format — the boss reliably emits PLAN/PIECE). Fresh sessions get the **full triad +
  collaborator code** (goal/contract + blueprint + module + test + error + the modules it calls); repair
  lanes think; **escalate to the boss early** when L1 stalls; the **designer dictates libraries**
  (contract pins deps = CommonJS, no jsdom/jest); **untested modules re-queue test-gen** (no silent
  pass). `--coord-test` 12/12. See `spec/PA4_COORDINATION_DESIGN.md`.
- **PA.5** ✓ *v1 (2026-06-14)* — worker tool-calls: nanocoder-style `<create_file>` /
  `<execute_bash>` (`--tools` / `--allow-run`), sandboxed to `--work-dir`. create_file verified
  end-to-end (workers wrote stack.js + test.js; the generated tests pass).
- **PA.6** ✓ *(2026-06-15)* — staged pipeline (`--tools --no-stream`): triage → parallel **design**
  (blueprints) → **reconcile** (interface contract) → parallel **implement** → test-gen → verify →
  repair. Parallelizes the serial plan-think (the ~80%-of-wall cost); reconcile lifted verify 1/3→2/3.
  GPU end-to-end validating. See `spec/PA6_PIPELINE_DESIGN.md`.
- **PA.7** *(core built 2026-06-16; integration pending)* — **eager scheduling**: dissolve the stage
  barriers into an artifact-gated ready-queue (idle lanes pull the next ready item across stages);
  **reconcile becomes the first rework pass**; **active retrieval** (`read_file`/`abandon`, NOT_FOUND →
  re-queue) gates discovered deps. Pure scheduler + makespan sim (`--eager-test` 8/8: **eager 18% <
  barriers** on the measured spread) and active-tool primitives (`--gather-test` T5–T8) are in; the
  `run_pool` mid-stream loop + ready-queue orchestration are next. See `spec/PA7_PIPELINING_DESIGN.md`.
