# PA.6 — Staged design→build pipeline (parallelize the thinking)

**Goal**: kill the dominant cost — the **serial boss plan-think** (measured ~80% of wall, ~18 min
on the multi-file Flappy task) — by **parallelizing the design phase**, and keep **sampling as the
production default**. The packed team's win was never raw decode speed; it's doing the *thinking*
(design) and the *implementation* in parallel, then shipping **verified** multi-file code.

**Status**: design + **IMPLEMENTED (2026-06-15, increment 1–4)** — `run_pipeline_staged` (triage →
parallel design pool → parallel implement pool → test-gen + verify) on the `--tools --no-stream`
path; build clean + GPU-free tests green; GPU end-to-end verification pending. Synthesizes the
2026-06-15 conversation. Builds on PA.2 (pool+refill), PA.4 (files-on-disk + test verifier +
repair), PA.5 (tools). Supersedes the "one boss thinks the whole plan" model in
PACKED_AGENTS_DESIGN §5.

---

## 1. The problem this fixes

Measured on the multi-file task (both thinking @ 0.6, 128k):

- Packed wall 1369 s, of which the **boss plan-think alone was 1105 s (~80%)** — one serial stream.
- Worker pool aggregate was only ~12.8 tok/s — **but that was STRAGGLERS, not sampling** (§6):
  the 5 items were 388/479/508/384/**1290** tok, so the 1290-tok module ran ~890 tokens **1-wide**
  after the small ones finished. A uniform-item A/B shows sampled 4-wide = **36.5 tok/s (1.89×)**,
  at the ceiling.
- Net: the two real costs are **serial plan-think (80% of wall)** and **straggler tails from uneven
  pieces** — both attacked by this pipeline (parallel design + more, smaller, balanced pieces).

The boss doing *all* the design thinking, alone, is the bottleneck. Parallelize it.

---

## 2. The pipeline (stages; parallel inside each)

```
Boss TRIAGE (light, fast — little/no deep thinking)
    → GOAL BLUEPRINT: the component map + how they connect + per-worker assignments
        ↓   every lane carries the goal blueprint + "you are X, here is where you fit"
DESIGN pool   (parallel, THINKING)   each designer writes design/<comp>.blueprint
        ↓
RECONCILE     (boss, serial-bounded) unify the blueprints → design/INTERFACE.md (the CONTRACT)
        ↓   harness seeds each lane with the goal blueprint + the CONTRACT (+ blueprints)
IMPLEMENT pool (parallel, instruct)  each writes src/<comp>.js against the CONTRACT
        ↓
TEST-GEN pool (parallel)             a test per module (given the goal/CONTRACT + its blueprint + the
                                     file; tests the SPEC, not just the code-as-written)  [PA.4b/4.2]
        ↓
VERIFY → done when green; failures → repair loop  [PA.4c]
```

**Why RECONCILE (added 2026-06-15, validated):** the first PA.6 run proved that parallel designers
keep the high-level *structure* coherent (via the goal blueprint) but **drift on signatures** — the
verify failed with `this.canvas.addEventListener is not a function` (a canvas-vs-context mismatch
between `index` and `renderer`), exactly as the blueprint review predicted. So after the parallel
DESIGN pool, the boss does ONE bounded **reconcile** pass that merges the proposed blueprints into a
single authoritative interface contract (canonical signatures + global decisions: **module system =
CommonJS** so `node` tests run, wiring, input/score owner, canvas/ctx). Implementers obey the
contract, not five mutually-inconsistent specs. Reconcile is serial but *merges* (it doesn't invent),
so it's far lighter than the old upfront full-plan think.

**MEASURED (2026-06-15):** reconcile lifted the verifier from **1/3 → 2/3** passing on the Flappy
task. The `index` integration failure (`addEventListener is not a function`, the canvas-vs-ctx drift)
was **fixed** by the contract, which also picked **CommonJS** so `node` tests actually run. The lone
remaining failure (`renderer`) is a per-module *logic* bug (most asserts pass, one fails) — not
integration → that's the **repair loop (PA.4c)**'s job. Cost: reconcile's serial boss pass pushed
wall ~1498→1914 s — exactly what the parallel pairwise-reconcile (above) would recover.

Only the **triage** is serial, and it is deliberately *light* (a component map, not a full design).
The expensive **design thinking is parallel** — exactly where packing wins, and where the per-token
sampling cost is amortized across lanes.

---

## 3. Goal blueprint + position (coherence without shared code)

Workers never see each other's code (isolation = parallelism). They stay coherent via two context
layers, both cheap (small, and the goal blueprint is the cached shared prefix):

- **Global — goal blueprint**: triage emits a component map, e.g. *"`engine` drives the loop and
  calls `bird.update()/flap()`; `renderer` draws all; `pipes` owns spawn + collision."* Every lane
  gets it. This is the existing `shared` block **elevated** from a flat interface dump to a
  component map + connections.
- **Local — where you fit**: each assignment names the component, its exports/contract, and its
  neighbors (*"you own `bird`; `engine` calls your `flap()`/`update(dt)`; `renderer` reads
  `getBounds()`"*).

This closes PACKED_AGENTS_DESIGN risk #6 (isolation drift): **shared goal + known position, no
shared code.**

---

## 4. Blueprints are files — the blackboard (no special tool)

Each component is its **own blueprint file** (`design/<comp>.blueprint`): a small structured doc —
component name + a JSON-ish tree of params/types + short docs (the same shape as a task entry).
Written with the existing `create_file`; no new tool.

"Query an interface" (*"how do I move the bird"*) is the **read-file→inject** primitive PA.4b
already uses for the test task: when the harness dispatches a designer/implementer, it seeds the
prompt with the relevant **sibling blueprints off disk**. Blueprints can be updated (a later stage
overwrites) and re-read. `design/` *is* the queryable interface registry — the blackboard, as files.
The boss's monolithic `shared` block thus **emerges from the parallel blueprints** (optionally
reconciled in a quick pass) instead of being authored serially.

---

## 5. Sampling is the production default

Greedy did the theory/byte-identity legwork; it remains **only** for the MTP spec-dec path and as a
diagnostic. Real output uses **sampling** (Qwen model-card params, PA.4 §10) — *creative and
realistic use needs it*. So the response to "sampling is slow" is **make sampling fast, never drop
it** (§6).

---

## 6. Sampler cost — MEASURED: not the bottleneck (do NOT regress to greedy)

**Measured 2026-06-15 (`--pool 8 -s 4 -n 128`, the A/B):**

| mode | tokens | decode | aggregate |
|---|---|---|---|
| **sampled** (Qwen instruct 0.7/0.80/top_k20/presence1.5) | 1024 | 28.1 s | **36.5 tok/s (1.89×)** |
| greedy (`--greedy`, argmax) | 939 | 31.2 s | 30.1 tok/s (1.56×) |

**Sampling is essentially free** — sampled 4-wide is at the ~37 ceiling, even *faster* than greedy
here (greedy's items hit EOG earlier → 939 tok → more 1-wide tail → lower aggregate; a lane-occupancy
artifact, not a sampler win). So the earlier "sampler caused 37→13" hypothesis was **wrong**.

The flappy pool's 12.8 tok/s was **stragglers**: uneven item sizes (one 1290-tok module vs ~400-tok
ones) → a long 1-wide tail. The fix is **balanced, smaller pieces + refill** (more items than lanes,
even sizes) — which the per-component granularity of this pipeline naturally provides — plus parallel
design to kill the plan-think. **No sampler optimization needed.** Greedy stays only as a
diagnostic/bench reference + the MTP path; sampling is the product.

## 6.1 MEASURED 2026-06-16: the SCALE bottleneck is PER-ITEM PREFILL, not stragglers

`scale_validate.sh` (13 independent modules — packed's supposed sweet spot, items ≫ lanes) was run to
test the hypothesis that *more, balanced, independent items recover throughput toward the ~1.9×
ceiling*. **It did the opposite:**

| stage | flappy (5 coupled) | scale (13 independent) |
|---|---|---|
| design aggregate | 0.69× | **0.64×** |
| implement aggregate | 0.50× | **0.22×** |
| packed wall | ~55 min | **108 min (6486 s), killed mid-test-gen — never reached verify** |

More items made aggregate **worse** and the wall **brutal**. Root cause, visible in the logs as
`full prefill` on every item: **`run_pipeline_staged` calls `run_pool(items, …, "")` with an EMPTY
shared prefix**, so all N items **re-prefill the entire shared context** (the goal + the reconciled
`INTERFACE.md` contract + the blueprint) from scratch. At 13 modules the contract is large and there
are 13 of them → **prefill dominates the wall** and decode-aggregate tok/s craters. This is NOT
stragglers (flappy) and NOT scheduling (PA.7) — it's **prefill bloat from re-sending the shared
prefix N times.**

**Fix — wire PA.2.1 prefix caching into the staged pools — IMPLEMENTED 2026-06-16.** The mechanism
already existed (`POSITIVE_RESULTS` §PA.2.1): prefill the shared prefix **once** into a base seq,
`seq_cp`-clone it per lane, and **delta-prefill only the per-item part**. `run_pipeline_staged` simply
never passed a shared prefix (`""`). Fix: the **shared content moves into the SYSTEM turn** so it is a
literal token-prefix of every item (`stage_prefix()`/`stage_item()` share identical system text), and
each stage passes it to `run_pool`:
- **design** — prefix = `goal`; delta = the per-component assignment.
- **implement** — prefix = `goal` + contract + **ALL blueprints** (read once, not re-sent per item —
  this was the 0.22× case); delta = a tiny "implement component X" user.
- **test-gen** — prefix = goal/contract; delta = blueprint + module + collaborators (`build_test_task`
  → `test_user`).

**SMOKE-CONFIRMED on GPU** (3-module task): design / implement / test-gen / test-gen-repair all log
`cloned+delta` (cache hit) instead of `full prefill`. A full re-run of `scale_validate.sh` will give
the new wall/aggregate numbers.

**Revised priority (done):** prefix-cache landed; **PA.7 eager scheduling is next** (recovers the
residual straggler tail on top). The "sweet spot" claim stays retracted until the scale re-run with
caching reports.

---

## 7. Risks & mitigations

| risk | mitigation |
|---|---|
| Triage too heavy (re-introduces the serial tax) | keep triage to a component map; cap its tokens; light/▽no deep thinking |
| Designers diverge (incompatible interfaces) | shared goal blueprint; a quick reconcile pass; collision check on declared exports |
| Stage latency (design→implement is serial across stages) | stages are few; each is parallel inside; only triage is truly serial |
| Implementer can't find a sibling interface | harness injects sibling blueprints (read-file→inject); blueprint is required output of design |
| Sampling stays slow | §6 measure+optimize; do not drop sampling |
| More moving parts | reuse PA.2 pool for every stage; blueprints/tests/modules are all just files |
| **Triage builds instead of assigns** (emits `*.test` pieces → implementers write `src/*.test.js` that clash with the harness test-gen; also `exports=none` false-collides) | **the COORDINATOR assigns work, it does NOT build code or tests itself** (user, 2026-06-15): triage prompt says so + forbids test pieces; harness drops any `.test` piece anyway; collision check skips the `none` sentinel (index.html, `*.test.js`). Tests are owned by the test-gen stage (§4.2), not the triage |

---

## 8. Acceptance criteria

1. Triage produces a goal blueprint (component map) in « the old full-plan time.
2. Design pool writes one `design/<comp>.blueprint` per component, in parallel.
3. Implement pool writes `src/<comp>.js`, each having seen the goal + sibling blueprints.
4. Test-gen + verify (PA.4) run; report tests passed.
5. End-to-end wall on the multi-file task is **lower** than the serial-plan baseline (1369 s), with
   the design phase parallelized — and a real sampled tok/s number.

---

## 9. Test plan (GPU-free, `--pipeline-test`)

Pure helpers, unit-tested like `--gather-test`/`--mtp-test`:
- `parse_goal_blueprint` (triage output → component list + assignments).
- `build_design_task` / `build_impl_task` (seed prompt contains goal blueprint + named siblings).
- blueprint file round-trip (write `design/x.blueprint`, read it back into a sibling prompt).
- stage sequencing bookkeeping (design done → implement dispatched; counts).

Integration (GPU): the multi-file Flappy task end-to-end; assert design/ has N blueprints, src/ has
N modules, test/ has N tests, verifier runs, wall < serial baseline.

---

## 10. Build order

1. ~~Sampler optimization~~ — **DONE/N-A**: measured free (§6); the pool slowness was stragglers.
2. **Staged pipeline** (§2): triage → design pool → implement pool (reuse PA.2 `run_pool` per stage;
   blueprints + goal-blueprint context via read-file→inject). This also fixes stragglers by producing
   **more, smaller, balanced pieces** (keep lanes full) and kills the serial plan-think.
3. **Repair loop** (PA.4c): failing tests → amend items → re-run until green / budget.

Out of scope (v1): mid-flight blueprint renegotiation, a dedicated critic lane, streaming-path
staging.

### Future idea — parallel pairwise reconcile (round-robin / orthogonal array) (user, 2026-06-15)

The v1 reconcile is one serial boss pass (bounded, but serial — the new long pole). Most interface
conflicts are **pairwise** (A's assumed view of B vs B's actual). So reconcile could itself be
parallelized: schedule **pairwise reconcile tasks** round-robin-tournament style — `(A,B),(C,D)…` in
round 1, rotate to `(A,C),(B,D)…` in round 2 — so **N/2 disjoint pairs run in parallel each round**,
and N−1 rounds cover all C(N,2) pairs. A **Taguchi / orthogonal array** prunes that to the few rounds
that still cover the important interactions (no full factorial). Each pair agrees its shared boundary
and writes it to the contract (the blackboard); a final light merge assembles the global contract.
Caveats: pairwise agreements can be **globally inconsistent** → iterate to convergence; and schedule
by the **actual dependency graph** from the goal blueprint (a hub like `engine` must pair with
everyone; a mesh benefits most from the orthogonal-array pruning). Explore after single-pass reconcile
is validated.

### Future idea — worker self-splits an oversized piece (user, 2026-06-15)

A worker (designer or implementer) that judges its assigned piece **too big** could itself emit a
*split*: spawn sub-pieces back onto the queue (recursive decomposition) rather than produce one
giant file. This turns the static triage→design→implement depth into an **adaptive** tree — the
boss does a coarse map, and lanes refine where needed. Fits the existing queue/blackboard (a split
is just "push N smaller work items + their blueprints"). Explore after the repair loop (PA.4c);
needs a depth/anti-runaway budget so splitting can't recurse forever.
