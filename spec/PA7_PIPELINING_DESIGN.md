# PA.7 — Eager scheduling (dissolve the stage barriers; reconcile = the first rework pass)

**Goal**: stop idling lanes at **stage barriers**. PA.6 runs design / implement / test-gen / verify as
**bulk-synchronous stages** — every item in a stage must finish before the next stage starts — so a
lane that finishes its piece sits idle through the stage's straggler tail *and* the hand-off. Replace
the stages with **eager scheduling**: a dependency-keyed ready-queue where a free lane immediately
pulls the highest-priority item whose inputs already exist. *If part of the design is done and there
are work items in the queue — start working.*

**Status**: design (2026-06-15) + **CORE IMPLEMENTED (2026-06-16)** — the pure scheduler logic
(`EItem`, `eager_dep_met`/`eager_ready` readiness, `eager_prio`, `eager_thinks` per-item mode, and a
makespan simulator `eager_simulate`/`staged_simulate`) plus `--eager-test` **8/8**. The simulator
**quantifies the win**: on the measured Flappy spread (design 6–12, impl 8–17 units) eager = **27** vs
barriers = **33 → 18% faster**, with the dispatcher proven work-conserving (never idle while ready work
exists). Decode-loop integration (run_pool → ready-queue) lands after the PA.6 GPU baseline.
Builds on PA.2 (pool + refill), PA.4 (the **rework primitive** —
full-triad `build_rework_user`/`build_amend_user` + verify→repair→arbiter loop), PA.6 (the stages it
fuses). Supersedes PA.6's bulk-synchronous staging; keeps PA.6's artifacts (goal blueprint, per-comp
blueprints, contract, per-module tests). Captures the user's design: *"eager scheduling… work on the
code, reconciling the contract is a small rework step before we get to post-test rework."*

---

## 1. The problem this fixes — barriers idle lanes (MEASURED)

PA.6 design pool, live Flappy run (2026-06-15), 5 designers on 4 lanes:

- items were **678 / 941 / 1231 / 624 / 1385** tok → the lane that drew the 624-tok blueprint idled
  ~760 tokens waiting for the 1385-tok one. Aggregate **13.4 tok/s = 0.69× baseline** — *below* a
  single stream, purely from the straggler tail + the barrier.
- Then a **hard hand-off**: *all* designs must finish → boss reconciles → only then does implement
  start. Two more barriers (implement→test-gen→verify) behind it.

Thinking lanes have the widest token spread (design, repair), so they suffer the worst tails — and
they're exactly the stages PA.6 makes everyone wait on.

---

## 2. Eager model — readiness is per-artifact, not per-stage

One global **ready-queue**. An item is *ready* when its **input artifacts exist on disk** (the
blackboard, PA.4), not when its "stage" completes. A free lane pulls the highest-priority ready item.

| item | ready when | lane mode |
|---|---|---|
| `design:X` | immediately (X in the goal map) | thinking (coding 0.6) |
| `implement:X` | `design/X.blueprint` exists | instruct |
| `reconcile` | runs concurrently as blueprints land; **emits `rework:X`** (see §3) | boss / thinking |
| `testgen:X` | `src/X.js` exists | instruct |
| `verify:X` | `src/X.js` **and** `test/X.test.js` exist | (harness, no lane) |
| `rework:X` | `verify:X` red **or** reconcile overrode X (§3) | thinking (coding 0.6) |

**No barrier except true data deps.** `design:bird` done → `implement:bird` ready *now*, even while
`design:engine` is still thinking. `src/bird.js` written → `testgen:bird` ready while `pipes.js` is
still being coded. Lanes never idle while *any* item anywhere is ready.

**Priority** (which ready item to pull): prefer items that unblock the most downstream — designs
before implements, and **critical-path / hub** components first (a hub like `engine` that everyone
depends on). v1 can approximate with: `design > rework > implement > testgen`, ties broken by the
goal-map fan-out. Refine later (orthogonal-array / dependency-graph scheduling).

---

## 3. Reconcile = the first rework pass (the key simplification)

PA.6's reconcile is a **barrier**: wait for all blueprints → boss writes `INTERFACE.md` → implement.
PA.7 flips it to a **rework trigger**, identical machinery to post-test repair:

1. Implementers start **eagerly** against their own blueprint + whatever sibling blueprints exist
   (read-file→inject). They do *not* wait for the contract.
2. **Reconcile runs concurrently** (boss, incrementally as blueprints land). When the contract
   finalizes, the harness **diffs it against each implementation's assumptions**; each component that
   actually drifted gets a **`rework:X`** item.
3. `rework:X` flows through the **same rework primitive** as post-test repair — `build_rework_user`
   (full triad: contract + blueprint + module + the drift note) on a **thinking** lane, output via
   `create_file`. The verify→repair→arbiter loop (PA.4) is unchanged.

So there are **two rework triggers, one rework operation**:

- **pre-test rework** ← reconcile contract-diff (integration drift; the canvas-vs-ctx class of bug),
- **post-test rework** ← a failing test (per-module logic bug).

Most components won't drift → no rework. Only the few the contract overrides get a cheap touch-up.
**Reconcile stops being a wall and becomes a source of repair items** — which is why eager-start is
safe: coherence is enforced *after the fact* by rework, not *before the fact* by a barrier. This is
viable precisely because PA.4 hardened rework (full-triad context, thinking lanes) this session.

---

## 3.1 Active retrieval + DISCOVERED dependencies (user, 2026-06-16)

Static injection is a guessing game — measured 2026-06-16, three failures in a row were a fresh
worker missing context (repair lacked spec+test → §4.2; test-gen lacked goal+spec → §4.2; then
**test-gen lacked the *collaborator code*** — `engine.test.js` had to mock the canvas API that
`renderer.js` actually calls, but the test-writer never saw `renderer.js`, so it mocked blind and
rework whack-a-moled one missing method at a time). The robust fix is to let the worker **fetch what
it decides it needs**, two layers:

- **Passive (PA.7a, now):** the harness injects the **collaborator modules** a component references
  (not just its own code) into test-gen / rework. Cheap (reuses read-file→inject), solves the
  integration-mock class. Still a *guess* about what to include → prune by the dep graph at scale.
- **Active (PA.7b, next):** a `read_file` / grep **tool the worker calls mid-stream** ("read
  `renderer.js` to see what I must mock"); the harness returns the content and the lane continues.
  Pulls *exactly* what's needed — the real agent loop. Needs the **mid-stream tool round-trip** (a
  lane that pauses, gets a result, resumes; in the packed batch = re-feed one lane while others
  decode). This is PA.5-v2 / the "live tool loop."

**`NOT_FOUND` is a dynamic dependency (the key semantic).** In eager mode a worker may read a file
that **doesn't exist yet** (its producer hasn't run). The read-tool returns *"not built yet,"* and
the worker **abandons + re-pushes itself as blocked**: *"Original task: <task>. Waiting on
`renderer.js`."* The scheduler gates the re-pushed item on that file and makes it ready the moment the
file appears. This turns §2's **static** dep table into a **discovered-at-runtime** graph — the worker
finds a dependency we never modelled and gates itself on it (same "worker pushes work onto the queue"
primitive as worker-self-split). Anti-runaway: cap re-push depth + total blocked-requeues per item.

**Automate the re-push via an `abandon` tool (user, 2026-06-16).** The worker doesn't reconstruct
anything — the **harness already holds the dispatch prompt** for every in-flight item. So the worker
just calls `abandon(reason)` (e.g. *"waiting on renderer.js"*); the harness **re-enqueues the held
prompt with the reason appended** (*"…Previously abandoned: waiting on renderer.js — now available
below:"*) and gates it on the named file (from the `read` NOT_FOUND, or parsed from the reason). So
the PA.7b worker toolset is: `create_file`, `execute_bash`, **`read_file`** (→ content or NOT_FOUND),
and **`abandon(reason)`** (→ harness-automated blocked re-push). No prompt-reconstruction by the model;
the harness owns continuity.

---

## 4. The one real engine change — per-item lane mode

Today the think/instruct mode + sampler is set **per stage**, globally (`g_worker_think`,
`g_worker_sp`, saved/restored around each pool). A single queue interleaves **thinking** items
(design, rework) with **instruct** items (implement, testgen), so mode must move **onto the item**:
each queue entry carries its `think` flag + `SParams`, and `run_pool` builds that lane's sampler
(and `<think>` stub) **per item** instead of from the globals. Bounded, but it touches the hot loop.

Everything else reuses existing parts: blueprints/modules/tests/contract are all just files (PA.4
blackboard); `run_worker_tools` stores outputs; `module_for_test` / `read_file_str` already map
components ↔ files; the rework builders already exist.

---

## 5. Risks & mitigations

| risk | mitigation |
|---|---|
| **Repair churn / cascade** — contract changes a hub (`engine`) → its dependents re-touch | bound by `--repair-budget`; re-queue only *direct* dependents (goal-map edges), not the world; most drift is leaf-local |
| Eager-implement drifts more than PA.6's barrier | that's the point — caught by reconcile-rework (§3) + verify-repair; net wall still wins if drift is rare |
| Starvation / priority inversion (lanes always pull cheap items, hub starves) | priority favors high-fan-out/critical-path items (§2); cap in-flight per component |
| Non-determinism / hard to follow vs clean stages | log every enqueue/dequeue with item id + trigger; `--eager-test` pins the schedule (§7) |
| Mixed-mode pool complexity | §4 per-item mode is the only engine change; keep it small + unit-tested |
| Reconcile races implement (contract lands after X is already built) | expected — that's exactly what the contract-diff → `rework:X` handles |

---

## 6. Acceptance criteria

1. A lane is **never idle while any ready item exists** (assert in `--eager-test`: with N items > L
   lanes and uneven sizes, occupancy stays full until the ready-set empties).
2. End-to-end wall on the multi-file task is **lower than the PA.6 staged baseline** (the run this
   doc was written during), with a real sampled tok/s.
3. Verify pass-rate **≥** PA.6 (eager-start + reconcile-rework must not *lose* coherence vs the
   barrier — drift is fixed, not shipped).
4. Reconcile drift produces `rework:X` items for *only* the drifted components (not all).

---

## 7. Test plan (GPU-free, `--eager-test`)

Pure scheduler/dataflow helpers, unit-tested like `--gather-test` / `--coord-test`:

- **readiness**: given a set of on-disk artifacts, compute the ready-set; assert `implement:X` is
  ready iff `X.blueprint` exists, `testgen:X` iff `src/X.js` exists, etc.
- **priority**: given a ready-set, the scheduler pops design/hub items before leaf implements.
- **no-idle invariant**: simulate N uneven items on L lanes; assert no lane idles while the ready-set
  is non-empty (the §6.1 property, on a mock clock).
- **reconcile-diff → rework**: given a contract + a set of blueprints, the components whose
  interface the contract changed (and only those) yield `rework:X` items.
- **per-item mode**: a thinking item and an instruct item in the same queue select their own
  sampler/`<think>` (verifies §4).

Integration (GPU): the Flappy task end-to-end; assert wall < PA.6 baseline, verify ≥ PA.6, and
(from logs) lanes stayed full across the old design→implement boundary.

---

## 8. Build order

1. **Per-item lane mode** in `run_pool` (§4) — the prerequisite engine change; unit-test it alone.
2. **Ready-queue scheduler** (§2): dependency-keyed readiness over the on-disk blackboard + priority;
   replace the bulk `run_pool`-per-stage calls in `run_pipeline_staged` with one eager loop.
3. **Reconcile-as-rework** (§3): run reconcile concurrently; diff contract vs blueprints → `rework:X`
   via the existing rework primitive.
4. **Measure** vs the PA.6 baseline (wall, occupancy, verify pass-rate).

Out of scope (v1): cross-task scheduling, mid-flight blueprint renegotiation beyond the one reconcile
diff, a dedicated critic lane.

### Enables (already-documented) follow-ons

The same live ready-queue is what two PA.6 future ideas need, so they fall out of this infra:
- **Worker self-split** (PA.6 §future): an oversized piece pushes sub-pieces back onto the queue —
  just new ready items.
- **Parallel pairwise reconcile** (PA.6 §future): reconcile incrementally as blueprints land
  (Taguchi / round-robin pairs) — the concurrent-reconcile of §3, parallelized. This is also the
  *only* lever for the design-pool straggler (§1), which eager scheduling alone can't remove because
  the contract genuinely needs all blueprints.
