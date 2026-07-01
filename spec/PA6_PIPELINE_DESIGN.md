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
        ↓
DESIGN REVIEW (boss)                 pin exact call-syntax (getter/method/field) into the CONTRACT  [§12]
        ↓   harness seeds each lane with the goal blueprint + the CONTRACT (+ blueprints)
IMPLEMENT pool (parallel, instruct)  each writes src/<comp>.js against the CONTRACT
        ↓
CODE CRITIC   (parallel, THINKING)   each module vs the CONTRACT, blind → design/CODE_REVIEW.md  [§12]
        ↓
TEST-GEN pool (parallel)             a test per module (given the goal/CONTRACT + its blueprint + the
                                     file; tests the SPEC, not just the code-as-written)  [PA.4b/4.2]
        ↓
TEST CRITIC   (parallel, THINKING)   each test vs the CONTRACT → design/TEST_REVIEW.md  [§12]
        ↓
VERIFY → done when green; failures → repair loop (critic notes feed the arbiter)  [PA.4c / §12]
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
residual straggler tail on top).

## 6.2 Clean scale re-run — RESULTS (2026-06-16, all four fixes)

`scale_validate.sh`, 13-component utility library, with prefix-cache + test-placement + contract-in-
repair + living-blueprint:

- **PACKED: 9/12 green, CLEAN** — 12 modules + 12 tests all in `test/` (zero `src/` dup), every module
  tested, auto-repaired. First pass 6/12 → L1 stalled (escalate-early fired) → **boss arbiter requested
  6 reworks, correctly MIXED**: `test/*.test.js` for judged test-bugs, `src/*.js` for module-bugs →
  **6→9**. 3 hard fails survived the budget (`dateUtils`, `numberUtils`, `stringUtils`). Cache **100%**
  (`cloned+delta`, 0 `full prefill`); **implement aggregate 0.22×→~0.68×** (prefill fix); design 0.79×.
  Wall ~116 min (serial reconcile + thinking-design stragglers remain — PA.7 / parallel-reconcile).
- **SINGLE: unscored** — fast (30 min) but in thinking mode it spent most of its budget *sketching*
  the 24-file output (`...` placeholders) and the harness splitter extracted 0 files. Real signal:
  **single is great at a fast single-file one-shot; cramming 24 complete files into one thinking pass
  is awkward** (output-length + reasoning budget). Packed's decomposition isn't bound by one context's
  output length — that's its structural edge at multi-file scale.

**Takeaway:** packed delivered a clean, fully test-covered, auto-repaired multi-file product (9/12);
single owns the fast small one-shot. They have different sweet spots — see "How to improve" (§11).

> **Reproducibility NOTE (verified 2026-06-22).** The committed artifacts in `validate/scale_packed`
> currently re-run at **6/12** — they are the **pre-arbiter first-pass snapshot** (`scale_packed.log` ends
> exactly at `══ PA.4d ARBITER (escalation 1/2) ══`); the **6→9 arbiter recovery was not preserved on
> disk**, so 9/12 is not reproducible from the repo as-is (re-run `scale_validate.sh` to regenerate). The 6
> first-pass survivors are dominated by **test-bugs + spec-ambiguities** — `stringUtils.truncate` (two
> contradictory assertions on the same input, both hand-reasoned wrong), `numberUtils.toInt(null)`,
> `objectUtils` Map-handling, `validationUtils` password policy, `colorUtils` float-rounding, `dateUtils`
> tokens — exactly the class that PA4 §4.6 (executed-truth + history bundle) and the L2 arbiter target. This
> is *why* §11.A (pass-rate) is the priority.

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

## 6.3 Parallel reconcile — IMPLEMENTED (2026-06-16)

The serial boss reconcile was the wall pole at scale (~23 min for 13 modules; §6.2). **v1 parallel
reconcile**: partition the blueprints into **G = min(lanes, N) balanced groups** (`reconcile_groups`,
pure + `--coord-test` R15), reconcile each group **in parallel on the pool** (`build_partial_reconcile_user`,
thinking lanes) → G partial contracts, then **one light boss MERGE** (`build_merge_user`) assembles
them + resolves cross-group/global decisions → `design/INTERFACE.md`. Small N (≤1 group) keeps the
single serial pass (no merge overhead). The boss MERGE retains final authority, so cross-group
consistency is preserved (the merge sees all partials); the win is that the per-group conflict-
resolution — the bulk — runs concurrently, and the merge reads *pre-digested* partials, not raw
blueprints. Biggest win on **independent** components (partials trivial → fast merge); ~neutral on
heavily-coupled (merge still resolves cross-group). **GPU MEASURED 2026-06-22 (`seq_vs_packed_ab.sh`): at
small N this HURT.** 6 blueprints triggered the parallel path (`min(lanes,N)=4` groups) → **5 boss
generations** (4 partials + a merge) vs **1** serial pass — inflating wall with no coherence gain on
independent components. The `min(lanes,N)` trigger is **too loose**: gate parallel reconcile on **N >>
lanes** (e.g. `N >= 2*lanes`) and keep the single serial pass otherwise. **IMPLEMENTED 2026-06-23** (`reconcile_parallel_g(N,lanes)`,
`--coord-test` R19): N < 2·lanes now takes the single serial pass (6 blueprints / 4 lanes → 1 gen, not 5).
Scale before/after (13 modules) still pending.

### Future — sharper scheduling (round-robin / orthogonal array, dependency-graph-aware)

v1 uses contiguous groups. The fuller idea: schedule **pairwise** tasks round-robin
(`(A,B),(C,D)…` → rotate) so N/2 disjoint pairs run per round across N−1 rounds covering all C(N,2)
pairs, **Taguchi/orthogonal-array**-pruned to the important interactions, and **scheduled by the
dependency graph** (a hub pairs with everyone; a mesh benefits most from pruning). Caveat: pairwise
agreements can be globally inconsistent → iterate to convergence. Explore if v1's contiguous grouping
proves too coarse on coupled projects.

### Future idea — worker self-splits an oversized piece (user, 2026-06-15)

A worker (designer or implementer) that judges its assigned piece **too big** could itself emit a
*split*: spawn sub-pieces back onto the queue (recursive decomposition) rather than produce one
giant file. This turns the static triage→design→implement depth into an **adaptive** tree — the
boss does a coarse map, and lanes refine where needed. Fits the existing queue/blackboard (a split
is just "push N smaller work items + their blueprints"). Explore after the repair loop (PA.4c);
needs a depth/anti-runaway budget so splitting can't recurse forever.

---

## 11. How to improve the agents (roadmap, value-ordered after the 2026-06-16 scale verdict)

The clean scale run (§6.2) showed the system *works* — a clean, fully test-covered, auto-repaired
9/12 multi-file product — and pinned where the value leaks. In priority order:

**A. Close the pass-rate gap (the "great tested product" priority).** 3 of 12 survived the repair
budget. Levers:
- **Multi-round arbiter** — today the boss escalates *once*; let it iterate (bounded by an L2 budget),
  re-judging after each rework.
- **RESPEC on repeated failure** — when a module fails repair *and* re-amend, the spec itself may be
  ambiguous → arbiter rewrites the **living blueprint** (capability shipped; encourage its use when a
  module is stuck, not just test-vs-module).
- **Minimal failing case** — feed the rework worker the *specific* failing assertion + values, not the
  whole test output, so it targets precisely.
- **Smarter budget** — more rounds for the few hard cases rather than a flat cap.
- **Full repair-history bundle (PA4 §4.6)** — persist the **arbiter's full diagnosis** (today only its
  one-line work-order survives `strip_think`+parse); carry the **cross-round error trajectory**; fix the
  RESPEC `.blueprint` **journal-key orphan**. A rework worker should see the *entire* history of its item.
- **Executed-truth over reasoning (PA4 §4.6)** — the harness owns the runtime, so inject the **executed
  actual** (already in node's `actual !== expected`) + the **spec rule** and rank them above any
  hand-derived expected; a **self-contradiction lint** (same input, two expecteds) jumps straight to L2.
  (The `stringUtils.truncate` survivor was a value the test-writer *reasoned* wrong twice.)

**B. Make workers genuinely autonomous — the live tool loop (PA.7b).** `read_file`/`abandon` primitives
are built; wire the **mid-stream round-trip** so a worker fetches exactly the context it needs (and
`abandon`→re-queue on a not-yet-built dep). This is the real-agent capability and unblocks eager.

**C. Throughput (secondary per user; wall ~116 min at 13 modules).** Two serial/straggler poles remain
after prefix-cache: **serial reconcile** (→ parallel pairwise reconcile) and **thinking-design
stragglers** (→ PA.7 eager scheduling). Both decode-bound, both documented.

**D. Right tool for the job — a packed-vs-single router.** Single owns the fast single-file one-shot
(§6.2); packed owns decomposed multi-file projects (not bound by one context's output length). Triage
should **route**: a small/single-file ask → one fast single pass; a multi-file project → the packed
pipeline. Don't run the 116-min pipeline for what single does in seconds.

**E. Right-size the pieces** — worker self-split (§future above) for oversized components; tighter
triage granularity so pieces are balanced (helps both quality and the straggler tail).

**F. A TEST HIERARCHY — integration tests one or more levels up (PA4 §4.7).** The unit-only layer
(collaborators stubbed) can't catch composition bugs (the flappy canvas-vs-ctx class) — both sides of a mock
drift. Add integration tests that **use the real sibling subtree and stub only the external boundary**
(DOM/canvas/clock/RNG), mirroring the dependency DAG (leaf=unit, internal node=integration, root=smoke);
integration reds route **straight to L2** (only the arbiter can attribute a multi-module failure).
Self-targeting: zero cost on the independent-utils scale task, all value on coupled apps. Restores the boss's
original `smokeTest` intent. Sequenced **after** A's history/executed-truth work, which is what makes
integration failures repairable.

---

## 12. Critics — quality gates at design / code / test (2026-06-30; design→build, CPU-validated, GPU-pending)

Three **critic** touchpoints, one per artifact kind, each a bounded pass that judges an artifact **against
the CONTRACT (the spec), never against sibling artifacts**. This is the ensemble-of-perspectives idea from
the Onklaud-5 pipeline note (`pipelines.txt`), adapted to our one-model, throughput-bound world: we bank the
*separate-reviewer* discipline (a fresh context + a reviewer persona + thinking mode breaks the author's
anchoring) but NOT cross-architecture diversity — that needs a 2nd same-size model (deferred; § future). It
is also the "proactive reviewer agents" named as *Next* in `PACKED_AGENTS.md` after the first all-green build.

- **DESIGN REVIEW** (after reconcile, `pti_pipeline.cpp`) — appends an EXACT CALL SYNTAX addendum
  (getter/method/field) to the CONTRACT so isolated implementers can't mis-call a sibling's surface (the
  `x is not a function` class — directly answers the getter-vs-method drift called out in §11 / PACKED_AGENTS).
- **CODE CRITIC** (after implement, before test-gen) — reviews each module vs the CONTRACT, **blind** to the
  author's reasoning (contract + code + siblings only; the worker's `<think>` is already stripped). Classes:
  CALL-SYNTAX / SIGNATURE / MISSING-EXPORT / CONTRACT-DRIFT. Writes `design/CODE_REVIEW.md`.
- **TEST CRITIC** (after test-gen, before verify) — reviews each test vs the CONTRACT + blueprint, **never
  the code** (the executed-truth rule — a test bent to match buggy code is a tautology). Classes:
  CONTRADICTION / CONTRACT-MISMATCH / OVER-MOCK / TAUTOLOGY. Writes `design/TEST_REVIEW.md`. It is the
  model-level generalization of the syntactic `contradictory_asserts()` lint (PA4 §4.6).

**Advisory, not destructive.** No critic rewrites an artifact. Their notes ride into the **arbiter
failblock** (beside the deterministic contradiction lint) so repair can tell a wrong TEST from wrong CODE.
Cheapest-tier-first is preserved: the $0 lints (compile, `contradictory_asserts`) run first; the model
critic is the escalation, not the front line.

**Flag adjudication — the one wrong-test class only a spec-anchored critic can catch.** A test that
CONTRACT-MISMATCHes the spec but *currently passes* means the TEST and MODULE agree with each other while
both disagree with the CONTRACT — invisible to execution (green) and to the contradiction lint (a
self-contradictory test can't pass). So a flagged **passing** test now forces ONE arbiter escalation
(bounded by `ARBITER_BUDGET`): the critic **nominates**, the boss **decides** (rework the test to match the
contract, or reject the flag). If reworked, the corrected test starts failing → surfaces the real module
bug → the normal repair cascade. A one-round `flag_grace` stops best-result preservation (§ PA.4c) from
reverting that INTENDED pass-count dip. A boss-rejected pure-flag escalation ends **DONE**, not GIVEN UP.

**Cost + caching.** Each critic is one thinking-mode pass per artifact — real cost on SSM-taxed HW. The big
shared chunk (the CONTRACT) is placed in the SYSTEM turn (`build_critic_system`) so `run_pool`
prefix-caches it once and delta-prefills only the per-item user turn (blueprint/test or code/siblings) — the
same PA.2.1 clone+delta the other pools use.

**Status.** Parsers + the cacheable system builder are `--coord-test` R32–R34 (34/34, clean build).
**GPU-validated 2026-07-01** (27B Qwen, KV-store task with an `isExpired` getter, `-s3 -c49152 --kv-q8`): all
three critics fired end-to-end with the prefix cache active (529 tok cloned/lane). DESIGN REVIEW pinned the
call-syntax addendum (`Entry.isExpired — GETTER — read as entry.isExpired (✓), NEVER entry.isExpired() (✗)`)
and the implementers obeyed it → CODE CRITIC correctly found both modules clean (design review *prevented* the
getter bug rather than the critic catching it). TEST CRITIC made two correct, spec-anchored CONTRACT-MISMATCH
catches on generated tests (a negative-TTL test asserting *not expired* vs the contract's `ttlMs<0` = already
expired; a lazy-delete test whose `Date.now()` stub left the entry un-expired), notes routed to the arbiter.
**Still unexercised** (both want a planted fixture, not dice): the flagged-**passing** adjudication cascade
(here the wrong tests *failed* → 0 flagged-passing) and the code-critic-catches-a-real-violation path (design
review pre-empted it). Wiring lives in `finalize_verify` (`pti_verify.cpp`); design review in
`run_pipeline_staged`.
