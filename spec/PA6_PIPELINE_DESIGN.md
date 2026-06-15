# PA.6 — Staged design→build pipeline (parallelize the thinking)

**Goal**: kill the dominant cost — the **serial boss plan-think** (measured ~80% of wall, ~18 min
on the multi-file Flappy task) — by **parallelizing the design phase**, and keep **sampling as the
production default**. The packed team's win was never raw decode speed; it's doing the *thinking*
(design) and the *implementation* in parallel, then shipping **verified** multi-file code.

**Status**: design. Synthesizes the 2026-06-15 conversation. Builds on PA.2 (pool+refill), PA.4
(files-on-disk + test verifier + repair), PA.5 (tools). Supersedes the "one boss thinks the whole
plan" model in PACKED_AGENTS_DESIGN §5.

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
        ↓   harness seeds each lane with the goal blueprint + sibling blueprints (read-file→inject)
IMPLEMENT pool (parallel, instruct)  each writes src/<comp>.js against its + siblings' blueprints
        ↓
TEST-GEN pool (parallel)             a test per module (given the file + its blueprint)  [PA.4b]
        ↓
VERIFY → done when green; failures → repair loop  [PA.4c]
```

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
