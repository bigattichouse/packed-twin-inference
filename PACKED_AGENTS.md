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
  pass); a **repair journal** (`ATTEMPT:` notes accumulated per-component, read by the next worker +
  the arbiter), a **multi-round arbiter**, and **RESPEC-on-stuck** (rewrite the living blueprint when
  the spec is the bug). `--coord-test` 14/14. See `spec/PA4_COORDINATION_DESIGN.md`.
- **PA.5** ✓ *v1 (2026-06-14)* — worker tool-calls: nanocoder-style `<create_file>` /
  `<execute_bash>` (`--tools` / `--allow-run`), sandboxed to `--work-dir`. create_file verified
  end-to-end (workers wrote stack.js + test.js; the generated tests pass).
- **PA.6** ✓ *(2026-06-15/16)* — staged pipeline (`--tools --no-stream`): triage → parallel **design**
  (blueprints) → **reconcile** (interface contract) → parallel **implement** → test-gen → verify →
  repair. Parallelizes the serial plan-think (the ~80%-of-wall cost); reconcile lifted verify 1/3→2/3.
  **Prefix-cached pools (§6.1):** the shared goal/contract/blueprints prefill **once** per stage
  (`stage_prefix` → `run_pool`), delta-prefill per item — fixes the scale prefill bottleneck (the
  108-min/0.22× run); smoke-confirmed `cloned+delta` all stages. See `spec/PA6_PIPELINE_DESIGN.md`.
- **PA.7** *(core built 2026-06-16; integration pending)* — **eager scheduling**: dissolve the stage
  barriers into an artifact-gated ready-queue (idle lanes pull the next ready item across stages);
  **reconcile becomes the first rework pass**; **active retrieval** (`read_file`/`abandon`, NOT_FOUND →
  re-queue) gates discovered deps. Pure scheduler + makespan sim (`--eager-test` 8/8: **eager 18% <
  barriers** on the measured spread) and active-tool primitives (`--gather-test` T5–T8) are in; the
  `run_pool` mid-stream loop + ready-queue orchestration are next. See `spec/PA7_PIPELINING_DESIGN.md`.
- **Next (design, 2026-06-22)** — sharpening the verify→repair loop (`spec/PA4_COORDINATION_DESIGN.md`
  §4.6–4.7), grounded in the `stringUtils.truncate` scale specimen: a per-component **repair history
  bundle** (persist the arbiter's FULL diagnosis, not just its one-line work-order; carry the cross-round
  error trajectory; fix the RESPEC `.blueprint` journal-key orphan); **executed-truth over reasoning** (the
  harness owns the runtime → inject the *executed* actual + the spec rule and rank them above any
  hand-derived expected; never source expected values from the artifact-under-test → tautology) + a
  **self-contradiction lint** (two asserts, same input, different expected → straight to L2); and a **test
  hierarchy** — integration tests *one or more levels up* that compose the **real** sibling subtree and stub
  only the external boundary (DOM/canvas/clock/RNG), failures routed straight to the boss arbiter.
- **§4.6/§4.7 IMPLEMENTED + merged to main (2026-06-24)** — the repair-quality trio + integration-test
  layer + reconcile gate (§6.3, `N≥2·lanes`) + cap-truncation guard; `--coord-test` 23/23; §4.7+§6.3
  GPU-validated (v47b clean 5/5). §4.6's contradiction/arbiter paths remain GPU-unexercised (no wrong test
  arose across runs) — to be validated with a deterministic fixture.
- **PA.8 — server / remote deployment (design, 2026-06-24)** — run cognition on a shared inference box;
  **percolate execution + resources to the calling client** (the LLM-outside-firewall / DB-inside case); a
  durable client-side **`.blackboard/`** (survives session close, isolated from cwd); a **stateless
  resumable server** (checkpoint coordination state → abandon-and-re-derive); client-declared **stack
  profiles** (de-JS-ify → SQL/Python/…); Board-on-the-wire status. One principle: *client owns the
  environment, server owns the cognition*. See [`spec/PA8_SERVER_DESIGN.md`](spec/PA8_SERVER_DESIGN.md).
- **Minimal-files directive + measured end-to-end run (2026-06-28)** — added a no-bloat directive to the
  **worker** preamble ("write the LEAST code that satisfies the spec — no speculative features / defensive
  bloat / over-engineering; small, cleanly-organized files") and the **test-gen** prompt ("small and focused
  — pin the contract + key edge cases, no padding"). The boss SIZE FLOOR (≥ a cohesive class/module, never a
  single function — tiny items waste per-item overhead) is unchanged: the middle ground. Run (staged
  `--tools`, Qwen3.6-27B-UD-Q6_K_XL, MI50/gfx906, 128k ctx, `-s 3 -n 24000`, KV-store w/ TTL+LRU task):
  - **Truncation solved.** Tight files (`entry.js` 682 B, `kvstore.js` 2099 B); the prior run's
    `create_file` cap-truncation (which left **0** tests on disk → verify had nothing to measure) is gone.
  - **Measurement works end-to-end.** 3 test files materialized (0 untested), node ran them, and the §4.7
    integration test caught a **real cross-module bug** — `entry.isExpired is not a function`: the DESIGN
    stage flip-flopped Entry's API (`get expired()` getter vs `isExpired()` method; `lastAccessTime` vs
    `lastAccessed`) and reconcile missed the getter-vs-method shape mismatch, so Entry and KVStore were
    built against different contracts.
  - **Repair did NOT converge (1/3, GIVEN UP).** The L2 arbiter saw the 2 failures but "requested 0 rework
    items" → no fix issued. Two separable gaps: (a) **design reconciliation** doesn't lint interface-shape
    mismatches (getter-vs-method, name drift) *before* implement; (b) **arbiter convergence** — it must
    reliably emit a rework envelope on real, attributable failures instead of punting. The
    decompose→implement→test→**measure** half is solid and honestly catches real bugs; closing the
    **repair** half (and pushing reconciliation upstream to prevent the drift) is the next work.
- **Repair loop closes — first all-green build (2026-06-29).** `build_boss_arbiter_prompt` was declared but
  never wired (both split and monolith fed the arbiter the generic triage prompt → it re-emitted the contract
  instead of fix-pieces → 0 reworks → give up). Implemented it as a dedicated REPAIR-ARBITER prompt: reason in
  `<think>`, emit ONLY a work-order, MINIMAL/targeted reworks (never touch a passing file), adapt to the cause
  (flaky test → make deterministic; fragile design → harden). Added **best-result preservation** in
  `finalize_verify` — snapshot the highest-passing version (a *sibling* dir, out of module discovery) and
  revert-to-best + stop on any regression, so the bounded loop always gives up at its **best** point, never a
  thrashed one. Result (Q6, MI50, 128k, `-s3 -n24000`, KV-store task): arbiter went 0 → 2 targeted reworks/round
  (down from an over-aggressive 4 mid-iteration), fixed the getter-as-method bug AND the flaky
  timestamp-collision LRU test, and converged **3/3 — DONE (all green)** — the first fully verified end-to-end
  build. Next: proactive **reviewer agents** (design/code/test review in the queue) to catch interface-shape
  drift, contract violations, and flaky tests *before* verify; and a side-git history
  (`.blackboard/history.git`, git-dir excluded from itself) for real per-rework diffs to the arbiter +
  checkpoint/resumability (PA.8).
- **Critics land — design / code / test reviewers in the queue (2026-06-30, CPU-validated, GPU-pending).** The
  "proactive reviewer agents" named above, built as three CONTRACT-anchored critic passes (see PA6 §12): DESIGN
  REVIEW pins call-syntax into the contract; the CODE CRITIC reviews each module vs the contract (blind); the
  TEST CRITIC reviews each test vs the contract (never the code — no tautology). All **advisory** — notes feed
  the arbiter beside the `contradictory_asserts` lint (cheapest-tier-first). A test the critic flags that
  *currently passes* (the wrong-test class execution + the lint both miss) forces one arbiter adjudication —
  critic nominates, boss decides — with a one-round `flag_grace` so best-preservation doesn't revert the
  intended dip. Contract cached into the system turn (`build_critic_system`) so critic pools clone+delta like
  every other pool. `--coord-test` 34/34. **GPU-validated (2026-07-01, 27B Qwen, KV-store w/ an `isExpired`
  getter):** all three critics fired + prefix-cached; DESIGN REVIEW pinned the getter's call-syntax and the
  implementers obeyed (→ CODE CRITIC correctly clean — the review *prevented* the bug); TEST CRITIC made 2
  correct CONTRACT-MISMATCH catches on generated tests, routed to the arbiter. Still to prove via a planted
  fixture (the wrong tests here *failed*, not passed): the flagged-**passing** adjudication cascade + a
  code-critic *catch*. Cross-model diversity (2nd same-size model) deferred — single-model reviewer
  discipline, from the Onklaud-5 pipeline note (`pipelines.txt`).
