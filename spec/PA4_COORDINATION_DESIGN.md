# PA.4 — Coordination: verify/repair loop + live task board (blackboard)

**Goal**: turn the fire-and-forget pool into a **closed agentic loop**. Workers write files +
tests; a *test agent* runs the tests; failures re-queue as **amend** items; the boss reasons over
a shared **Board** (the blackboard) and reports progress to the user as a **live checklist**. The
packed team's payoff is not raw speed — it's **autonomously verified, test-passing, multi-file
output** that a one-shot model cannot self-correct.

**Status**: §3 verify/repair (PA.4c), §4.1 boss arbiter (PA.4d), and §4.2 fresh-session full-context
enrichment **IMPLEMENTED (2026-06-15)** — `finalize_verify` (test-gen → verify → L1 amend → L2 boss
re-queue), GPU-free `--coord-test` 10/10 green; GPU end-to-end (3/3) pending. §1/§2 Board live-checklist
still design. Builds on PA.2 (pool + refill queue), PA.5 (`--tools`: `create_file`/`execute_bash`),
PA.1c (gather). Realizes design §8.2 (bidirectional coordination), §8.4 (two queues / blackboard),
§10 (boss = single voice to the user).

This doc captures the user's design (2026-06-15): tool calls that *run tests and amend files*; a
*test agent that finds a problem and pushes it back to be amended*; workers *report DONE so the
boss can report progress*; a *live list* (add → check off → warnings); and the realization that
the list is *a data structure the boss has access to*.

---

## 1. The Board — harness-owned blackboard

The single source of truth. **Owned and written by the harness (exactly one writer ⇒ no races**,
per §8.4 "exactly one lane holds the boss"). Read by the boss; rendered for the user.

```cpp
struct BoardItem {
    std::string id;                 // "w1", "w2", ...
    std::string spec;               // instruction + blueprint for the piece
    std::vector<std::string> exports;
    std::string file;               // path written via create_file (if any)
    enum Status { QUEUED, RUNNING, DONE, FAILED, GIVEN_UP } status = QUEUED;
    std::string test_cmd;           // how to verify (e.g. "node bird.test.js")
    std::string test_output;        // last run's output/error (on FAILED)
    int repair_count = 0;           // repair rounds spent on this item
};
struct Board { std::vector<BoardItem> items; /* + index by id */ };
```

**Three consumers, one writer:**
- **Harness** — the WRITER. Mutates the Board on events (§2).
- **Boss** — READER + decider. Gets a snapshot injected into its context; emits *intents* (§4).
  The boss never writes the Board directly.
- **User** — VIEWER. The Board renders as a live checklist (§5).

---

## 2. Events → Board transitions (the harness writes)

| event | transition |
|---|---|
| boss plans a `<<<PIECE>>>` | add item → `QUEUED` |
| a lane pops the item | `RUNNING` (record lane) |
| worker emits `DONE` / hits EOG | `DONE`; record files written via `create_file` |
| test agent runs the test, **pass** | stays `DONE` (✓ verified) |
| test agent runs the test, **fail** | `FAILED` + `test_output`; if `repair_count < budget` → re-queue an **amend** item (`QUEUED`, `repair_count++`); else `GIVEN_UP` |

The DONE signal is the event the user described ("workers report back when they complete") — it is
already detected today (EOG / `max_new`); PA.4 promotes it from a stderr line to a Board write.

---

## 2.1 Worker file lifecycle — craft → write → store → done → next (PA.5 timing change)

User's model (2026-06-15): *"workers should be able to save their own files, so other workers can
edit them if there's a problem. craft, write, store, done … next work item."*

Today (PA.5 v1) tool calls execute **post-completion, in gather** — so files only appear at the very
end, too late for one worker to edit another's. PA.4 moves execution **per-lane, on DONE**:

```
worker:  craft (generate) → emit create_file → DONE
harness: store file to shared work-dir → (verify, §3) → Board update → refill lane with next item
```

So files **accumulate on disk during the run**. That is precisely what lets another worker **edit**
them: an amend item (§3) targets file X; the harness reads X's *current* contents off disk, injects
them + the test error into the amend prompt; any free lane pops it and rewrites X via `create_file`
(overwrite). The **work-dir (files) + the Board (statuses) together are the blackboard** — workers
coordinate through saved files, the boss/harness through the Board.

Implementation: in the pool loop's `finished`-lane handling, run that lane's tool calls *before*
refilling it (instead of batching all tool calls in `finish_gather`). A small `read_file` helper (or
just harness-side disk read) supplies the current contents for amend items. Gather then *verifies
and assembles* real files rather than re-emitting and flattening them (which is what stripped the
tests in the single-file run).

---

## 3. Verify → repair loop (the test agent)

After a worker's files land (`create_file`, executed in the tool pass), the harness runs that
piece's test (`execute_bash`, e.g. `node bird.test.js`), captures pass/fail + output:

- **pass** → item `DONE`/verified.
- **fail** → build an **amend item** and push it onto the worker queue:
  ```
  AMEND <id>: your file FAILED its tests. Fix ONLY what's needed to pass.
  <<<CURRENT_FILE path=bird.js>>> …current contents… <<<END_FILE>>>
  <<<TEST_OUTPUT>>> …assertion failures… <<<END_TEST_OUTPUT>>>
  Re-emit the corrected file with create_file.
  ```
  A lane pops it, rewrites the file, re-test. Loop until pass or **repair budget** (default 3) →
  `GIVEN_UP`. The budget is **mandatory** (the loop's only termination guarantee besides `-n`).

**The "test agent":**
- **v1**: the *harness* runs tests + writes the Board (simplest, deterministic).
- **v2**: a dedicated **critic lane** that runs/reads tests and posts amend requests to the boss
  queue — a real agent role. (§8.4 boss-queue.)

---

## 3.2 Gather → finalize/verify (often nothing to do)

User's question (2026-06-15): *"so gather shouldn't really happen in this new world?"* — correct.

In the file-per-worker world the **text-merge gather is redundant and should not run**: the
deliverable is the files on disk, which the workers already wrote. Re-emitting a merged blob is both
*expensive* (a full boss generation **with thinking** — one of packed's two big serial costs, the
other being the plan) and *lossy* — the single-file run's gather **stripped every test** and drifted
the code to TypeScript.

Gather collapses into a cheap **finalize/verify** bookend (matches §8.3 "boss verifies the assembled
tree and runs the smoke test"):
- ensure the entry point exists (`index.html`) — authored as its own small piece by a worker, or a
  short boss glue file, **never** a re-emit of everything;
- run the **full test suite** on the assembled tree (one `execute_bash`) → record on the Board;
- emit the **Board summary** to the user (pass/fail per module, repairs spent).

No code regeneration. `--out` becomes "point at the work-dir" (or a zip), not "the boss's merged
blob". The legacy merge-gather (PA.1c) stays available only for the genuine single-blob mode (and
even there, prefer *one worker writes the entry file that includes the others* over a boss merge).

This is also a **performance** result: deleting the gather-regeneration removes one of packed's two
big serial costs (plan-think + gather-think) on top of the correctness win — directly narrowing the
wall-clock gap with the single model in the multi-file regime.

---

## 4. Boss reads the Board + acts (coordination turns)

At a coordination cadence (a batch boundary, after a verify round) the harness injects a Board
snapshot into the boss's context:

```
<<<BOARD>>>
[✓] bird.js     done (tests pass)
[⚠] engine.js   FAILED  engine.test.js 2/5  (repair 1/3)
[ ] hud.js      queued
<<<END_BOARD>>>
```

The boss emits **intents**; the harness applies them to the Board (boss never writes it):
- `<<<REQUEUE id: hint>>>` — re-queue with extra guidance.
- `<<<RESPEC id: new spec>>>` — replace the piece's spec (e.g. the *test* is wrong, or try a
  different approach).
- `<<<GIVE_UP id>>>` — stop repairing (budget/utility call).
- `<<<DONE>>>` — board acceptable → proceed to gather.

**v1 runs the loop WITHOUT boss intents** (harness auto-amends against the original spec until the
budget); **boss-driven coordination is v2** (`--boss-coord`). This keeps the first cut deterministic
and testable; the boss-as-coordinator layer rides on top once the mechanics are proven.

### 4.1 PA.4d — escalate to the boss on repair-budget exhaustion (user, 2026-06-15)

Two-level repair hierarchy:
- **L1 worker repair** (PA.4c): amend the *module*, N rounds — cheap, parallel.
- **L2 boss arbiter** (PA.4d): when L1 hits `--repair-budget`, instead of a flat `GIVEN UP`, hand the
  boss the **original plan + interface contract + the failing module + its test + the error/attempt
  history** and let it **decide**: *quit* (accept/drop the piece, report) or *try something different*
  — re-spec the module, **fix the TEST**, or re-approach.

**Why it matters (measured 2026-06-15):** the full-arc run gave up 2/3 on `renderer` — but the
"failure" was a **bug in the test**, not the module: the arc spy used `originalArc.apply(this, ...args)`
(spread → apply's 2nd arg becomes a number → `CreateListFromArrayLike`). L1 repair is *module-only*,
so it could never fix it. The boss arbiter — seeing the correct module + the buggy test + the error —
can fix the test (or accept). This is the authority/context a worker lacks (§8.2 GUIDE/KILL/RESPEC,
§10 single voice). Cost is bounded: workers grind cheaply in parallel; the serial boss pass fires
**only on exhaustion** (rare).

**The boss decides and DELEGATES — it pushes rework back onto the queue (user, 2026-06-15).** The
arbiter doesn't fix things itself; it emits **rework requests** that the harness enqueues onto the
worker queue, and the **parallel pool** does the work — the §8.4 blackboard fully realized (boss
produces items, workers pop them). The arbiter's verbs are queue ops:
- `REWORK file=<path>: <guidance>` — re-queue a task to rewrite that file (module **or** test);
- `RESPEC <comp>: <new spec>` — re-design then re-implement the component;
- `SPLIT <comp> → <a,b,…>` — break an oversized piece into new pieces (boss-driven version of the
  "worker self-split" idea);
- `KILL <comp>` — accept/drop it and report.

Flow: on `GIVEN_UP`, build the arbiter turn `{goal, contract, failing module+test, error/attempt
history}` → boss emits rework items → harness **enqueues** them → pool runs (parallel) → re-verify →
possibly escalate again, bounded by an **overall L2 budget** so it can't loop forever. This unifies
repair with the queue: L1 is the harness auto-amending the module; L2 is the boss requeuing *any*
rework (fix the test, re-spec, split, or quit). It also fixes today's blocker — the boss requeues
"fix test/renderer.test.js: the arc spy spreads `...args` into apply" and the pool corrects it.

### 4.2 Every fresh worker gets the FULL triad — "we're building THIS, here's your part" (user, 2026-06-15) — IMPLEMENTED

Each worker/arbiter/repair lane is a **fresh session** (`run_pool`'s `start_lane` does `seq_rm(L)` —
no infection from prior work). Isolation buys parallelism, but it means a lane knows **only what the
harness injects**. The first PA.4d arbiter run *correctly* targeted the test, yet the rework didn't
land — the fresh worker saw only the file + a one-line guidance, not enough to fix anything. Fix:
inject the same context an implementer had, so every repair/test lane has *goal → my part → code*:

- **L1 worker amend** (`build_amend_user`): module + test + error **+ `design/<comp>.blueprint` (spec)**.
- **L2 boss rework** (`build_rework_user`): target + **spec** + module + test + **the matching error** +
  the boss's guidance — the full triad, derived per rework file (component ← target name, spec ←
  blueprint, test+error ← the failing `fails` entry).
- **Test-gen** (`build_test_task`): was the worst-contextualized lane — it saw only its module + the
  *thin triage* `shared` line. Now it gets **the project goal/contract (reconciled `INTERFACE.md`, else
  triage shared) + this component's `blueprint` + the code**, and is told to test against the **spec,
  not just the code as-written** ("we're building THIS; your part is `<comp>`; here is its spec").

**Repair/rework lanes THINK** (coding 0.6), like designers — debugging a failing test is precise
reasoning, not mechanical transcription. The verify→repair loop saves/sets `g_worker_think=true`,
`g_worker_sp=qwen_params(true,false)` for its duration and restores after (mirrors the design pool).
Covered GPU-free by `--coord-test` R3/R9/R10 (amend/rework/test-gen all carry their full context).

### 4.3 The arbiter speaks WORK-ORDERS, not a bespoke REWORK format (MEASURED 2026-06-15/16)

First full GPU run with §4.2 enrichment: **2/4** on Flappy (`bird`,`renderer` pass; `engine`,`pipes`
fail). Both failures were **test bugs, not module bugs** — `engine.test.js`'s `MockContext` lacked
`fillRect` (which `render()` calls); `pipes.test.js` asserted `p.x === 400` *after* `update()` had
already moved the pipe left. So:

- **L1 (module-amend) structurally can't fix them** — it spent both repair rounds rewriting already-
  correct modules. Test bugs need L2's authority (only the boss may edit a test — anti-gaming).
- **The §4.2 enrichment WORKED at the reasoning layer**: given the full triad, the boss diagnosed
  both correctly — it even computed `400 − 22.5 = 377.5` to prove the pipes assertion wrong, and
  noted "MockContext must implement `fillRect`". The context did its job.
- **But the arbiter's output never parsed.** The boss emitted its decision as the **work-order
  envelope it knows** (`<<<PLAN>>>/<<<PIECE …>>>`), while the old `parse_rework` only matched a
  bespoke `<<<REWORK file=…>>>…<<<END>>>`. Result: **0 rework items → GIVEN UP.** The mechanism
  reasoned right and then spoke the wrong language.

**Fix (user choice, 2026-06-16): reuse the work-order envelope.** The arbiter prompt now asks for a
PLAN/PIECE work-order (one PIECE per file to rewrite, path in `exports=`, fix in the instruction; may
target a TEST file), and `reworks_from_plan()` maps each PIECE → `(target, guidance)` via the existing
`parse_work_order` (path-like tokens from `exports`/instruction; dedup; `..` rejected). This unifies on
**"the boss always emits work-orders; the harness turns pieces into queue items"** — the same model as
PA.7 eager scheduling. Covered GPU-free by `--coord-test` R11/R12.

**Escalate to L2 early when L1 stalls (IMPLEMENTED 2026-06-16).** L1 (module-amend) can't fix *test*
bugs, so on the 2/4 run it burned the full budget rewriting correct modules before the arbiter fired.
Now the loop tracks the failing count across rounds: if an L1 repair round does **not reduce** the
number of failing tests, it escalates to the boss arbiter immediately instead of spending more L1
rounds (`no_progress` → `RA_GIVEUP`). Combined with §4.2 collaborator injection, the boss's rework
then gets the context L1 lacked.

### 4.4 Designer dictates libraries; untested modules re-queue test-gen (user, 2026-06-16 — IMPLEMENTED)

The 2nd flappy run (2/3) showed §4.2 collaborator injection *working* — the engine test went from a
blind, incomplete canvas mock to recognizing it needed a real DOM/canvas — but it **over-reached**:
`import { JSDOM } from 'jsdom'` (not installed) + ESM `import` (violating the CommonJS contract). And
`pipes` got a module but **no test**, which the verifier passed over silently. Two fixes:

- **The designer DICTATES the libraries when the task didn't (user).** Nothing forbade `jsdom`, so the
  test-writer invented a dependency. Reconcile now pins a **DEPENDENCY POLICY** in the contract's TECH
  DECISIONS: if the task didn't name libraries, the designer chooses — default **zero external
  packages**; tests use only Node built-ins + `console.assert`, **CommonJS `require`** (not `import`),
  **no `jsdom`/`jest`**, collaborators stubbed directly. `build_test_task` reinforces the same. One
  authoritative decision flows to every implementer and test, instead of each worker guessing.
- **Untested modules trigger test-gen jobs in the repair phase (user).** A module with no
  `test/<comp>.test.js` is now counted as an open *issue* (not a silent pass) and **re-queued for
  test-gen each repair round** (`gen_tests_for(untested)`), bounded by `--repair-budget`; if it still
  can't get a test, it ends in `GIVEN UP`, reported — never green-by-omission. Test-gen is now a
  reusable harness job, fired both initially and on demand during repair.
- **Implementer tests are allowed, but in the PROPER place; test-gen fills only the gaps (user,
  2026-06-16).** The scale run showed implementers writing `*.test.js` **into `src/`** (same-dir
  `require`), duplicating the harness's `test/` tests. Implementers may write a first-pass test (it
  gets reviewed by verify), but `impl_user` now directs it to **`test/<comp>.test.js`** (`require('../src/<comp>')`),
  never inside `src/`. And the **initial** test-gen now generates **only for modules that still lack a
  test** (same "fill the gaps" rule as repair) — so implementer-written tests are kept and reviewed,
  not overwritten or duplicated.
- **Repair reads the CONTRACT, and the blueprint is a LIVING doc (user, 2026-06-16).** The spec a
  repair/rework worker reads should be the *current* spec, not a frozen original. So: (1) both repair
  levels now fold the **authoritative `INTERFACE.md` contract** (reconcile-evolved, overrides
  blueprints) into the `spec`, alongside the per-component blueprint; and (2) the **blueprint is
  rewritable** — the arbiter may target `design/<comp>.blueprint` in its work-order to **RE-SPEC** a
  component (`reworks_from_plan` now accepts `.blueprint` targets; `build_arbiter_user` lists it as an
  option). Because repair re-reads the blueprint fresh each round, a RESPEC in round N is picked up
  when round N+1 re-implements the module against it — the spec evolves, repair tracks it. Covered by
  `--coord-test` R13.

### 4.5 Repair journal + multi-round arbiter + RESPEC-on-stuck (user, 2026-06-16 — IMPLEMENTED)

Diagnosis of the clean scale run's 3 survivors (`spec/PA6_PIPELINE_DESIGN.md` §6.2): **1 test bug +
2 spec ambiguities** (`toInt(null)`, `formatDate` unknown-tokens), all of which the arbiter *should*
crack — but they survived because each repair attempt was **blind to prior attempts** and the arbiter
**escalated only once**. Three coupled fixes:

- **Repair journal (user).** Each repair/rework worker writes one line `ATTEMPT: <what's wrong + what I
  changed>` before its tool call; the harness (`extract_attempt`) accumulates these **per component**
  (in-memory `journal[comp]` for now; files/git for cross-session survival is the documented next step).
  The next worker reads the history **before the code** ("don't repeat these"), and the arbiter sees it
  too. Stops the repeated-approach churn.
- **Multi-round arbiter.** The boss now escalates up to `ARBITER_BUDGET` (=2) times, re-judging after
  each rework with the journal in hand — instead of one-and-done.
- **RESPEC-on-stuck.** The arbiter's failblock flags, per component, "PRIOR ATTEMPTS … if these failed
  the SPEC may be ambiguous → RESPEC `design/<comp>.blueprint`" — steering it to fix the *spec* (the
  living blueprint, §4.4) for the spec-ambiguity class, not just re-poke the test/module.

GPU-free coverage: `--coord-test` R3/R9 (journal + `ATTEMPT:` in the repair prompts), R14
(`extract_attempt`). Future (user, deferred): journal survives sessions; `git init` the work-dir so
each worker sees real per-write diffs.

### 4.6 The repair HISTORY bundle + executed-truth-over-reasoning (user, 2026-06-22 — DESIGN)

Diagnosis of the scale run's surviving failures (`spec/PA6_PIPELINE_DESIGN.md` §6.2) sharpened into two
coupled principles, both grounded in one specimen — `stringUtils.truncate` (the 13-module scale run,
`validate/scale_packed`):

**The specimen.** `truncate('hello', 6)` — `'hello'` is 5 chars, fits in 6, so the stated rule ("truncate
only if `len > max`") yields `'hello'`; the module is correct. The harness-generated test asserted it
**twice, contradictorily** — `=== 'hello.'` then `=== 'hel...'` — with a literal self-correcting comment
between them (`// Wait, default suffix is '...' (len 3)… = 'hel...'`) whose correction was *also* wrong. The
test-writer **reasoned** the expected value, in its head, and was wrong both times, when
`node -e 'truncate("hello",6)'` would have ended it. L1 (module-only) is **structurally incapable** here —
no module can satisfy two mutually-exclusive assertions on the same input — so it burns its budget
corrupting a correct module until `no_progress` (§4.3) escalates.

**Principle 1 — a rework worker needs the ENTIRE history of its item, as one persisted object (user):**
*"you need to see the entire history and notes related — how that item fits in the plan, the code, the
tests, the comments, arb, etc."* Today the repair context is **re-assembled fresh each round** (spec/
contract + module + test + error + collaborators + `journal[comp]`), and the two richest history pieces are
**dropped**:

- **The arbiter's diagnosis is discarded.** `reworks_from_plan(strip_think(boss_generate(...)))` keeps only
  the one-line `instruction:` from each PIECE — the boss's actual reasoning (its `<think>` block + prose:
  *"these two asserts contradict; the module is right; rewrite the test to expect `'hello'`"*) is thrown
  away. The rework worker and the *next* arbiter round never see **why**. On flappy (§4.1) the doc celebrated
  the boss computing `400 − 22.5 = 377.5` to prove a test wrong — then deleted that proof.
- **The cross-round trajectory is not assembled** — only the latest `r.out`. A worker that fixed the first
  assertion → re-ran → now fails the second isn't told *"you already changed this; a **different** assertion
  on the same input broke"* — the exact signature of a multiply-wrong test. `journal[comp]` was meant to be
  this memory but it's thin (fires only when a worker emits `ATTEMPT:`, often "(no note)", and is not
  file-attributed).

Fix: a **per-component history bundle**, harness-owned, threaded into every amend/rework/arbiter prompt —
the round trajectory `{file changed, the boss's FULL diagnosis (not the one-liner), what was tried, error
before→after}`, **file-attributed**, blueprint/contract under the right key. One persisted object, not six
re-derived with the two best dropped.

**A concrete keying bug this exposes.** The L2 attempt recorder strips `.test.js`/`.js` to key the journal
by component, but a **RESPEC target (`design/X.blueprint`) matches neither** → it is journaled under
`"X.blueprint"` while every reader looks up `journal["X"]`. So living-blueprint RESPEC history (§4.4) is
**orphaned under the wrong key** and invisible to the next module/test repair — precisely the thread you
most want when a spec keeps drifting. Fix the key extraction to also strip `.blueprint`.

**Principle 2 — an executed/verifiable value always beats a reasoned one (user):** *"tool calls should
always win over reasoned values — running a little command-line calculation is verifiable."* The project
ethos ("MEASURED, not predicted") pushed **into the agents**. The boundary that keeps this from being
self-defeating:

> **Execute to get a FACT; something INDEPENDENT must supply the ORACLE. Never run the artifact-under-test
> to SOURCE the test's expected values** — that yields `assert(fn(x) === fn(x))`, a tautology that catches
> zero module bugs and silently converts a spec-test into a characterization-test (violating §4.2 "test the
> SPEC, not the code-as-written").

The executed fact is **already in hand and merely not privileged.** Node's assert prints `'hello' !==
'hello.'`: the left side is the **executed actual** (a verified fact — what the module really returns), the
right is the **reasoned expected** (the test-writer's hand-derivation). The harness dumps it raw and lets
the arbiter *re-reason*. Instead, frame it as a ranking:

```
EXECUTED actual = 'hello'  ·  test-expected = 'hello.'  ·  spec rule (truncate iff len>max) endorses 'hello'  →  the TEST is wrong.
```

Execution supplies the fact, the **spec rule** supplies the oracle, the reasoned value loses — and the boss
never has to do arithmetic in its head (the exact failure mode that bit the test-writer). The **harness
already owns the runtime** (`run_all_tests`, popen) so it can run the failing call against the module and
inject `EXECUTED: …` into the failblock **today**, before the PA.7b live tool loop; the agent-driven version
(a worker calls `execute_bash` mid-stream and continues) is a prime PA.7b motivation.

**Principle 2b — a self-contradiction lint.** Two assertions, same input, different expected is
**unsatisfiable by any module** — detectable by reading the test file: a hard *"the test is the bug → jump
straight to L2"* signal (no reasoning, no L1 budget burned).

**Build (the "trio"; GPU-free fixture = the `stringUtils.truncate` case, new `--coord-test` cases):**
(1) history bundle — persist the arbiter's full diagnosis into the journal; fix the `.blueprint` keying;
carry the error trajectory; (2) executed-ground-truth injection — harness runs the failing call, frames
actual-vs-expected-vs-spec, journals the *executed* fact over any reasoned claim; (3) self-contradiction
lint → immediate L2.

**IMPLEMENTED 2026-06-23 (commit adb8a90):** the trio is built + wired + unit-tested (`--coord-test`
R16–R18 → 18/18). `comp_key()` fixes the journal-key orphan (both recorders); `contradictory_asserts()`
(paren-aware) routes a self-contradictory failing test **straight to L2**; `frame_executed_truth()` is
prepended to the error in the arbiter failblock + L1 amend + L2 rework prompts; and `arbiter_diag`
persists the arbiter's full reasoning across escalations (round 2 refines round 1). **Partial:**
executed-truth currently re-frames node's error text (which already carries the actual) — having the
harness *run the failing call* for values the error omits is the deeper follow-up. End-to-end GPU
confirmation (a contradictory test escalating + getting fixed) still pending.

### 4.7 Test HIERARCHY — integration tests one or more levels up (user, 2026-06-22 — DESIGN)

Today the test layer is **one level deep**: `test/<comp>.test.js` requires one module and asserts on it,
**collaborators stubbed**. That stubbing is the bug source for *coupled* tasks: the flappy
`addEventListener is not a function` / canvas-vs-ctx failure (§4.1, PA6 §2) lives in **no single module** —
it is in how `index` wires the *real* `renderer`. Two modules each pass their unit test against their own
*mock* of each other and still don't compose, because **both sides of the mock drifted** — the "blind mock,
whack-a-mole one missing method at a time" of PA7 §3.1, unfixable at the unit level by construction.

**The rule: mock only the external boundary; use real siblings inside the tree.** A dependency that
resolves to a generated `src/*.js` is *internal* → use it for real; anything outside the generated tree
(canvas, DOM, clock, RNG, fs, network) is *external* → stub it. The test layers then **mirror the dependency
DAG**:

- **leaves** → unit tests, external boundary stubbed (have this);
- **internal nodes** (`engine`, `index`) → **integration tests** composing the *real* subtree below them,
  stubbing only the external edge;
- **root** → a **smoke test** that boots the whole thing.

"One or more levels up" = walk up the DAG; each non-leaf gets a test against real children. This is *less*
work than the current path, not more: the whole collaborator-injection thread (PA.7a injects collaborator
code so the unit test can mock it *accurately*; PA.7b fetches more to mock better) exists to make mocks
faithful — the endpoint of "be collaborator-aware" is **stop mocking collaborators**. It also **restores the
boss's original `smokeTest` SELF-piece** (PACKED_AGENTS_DESIGN §4), dropped when gather collapsed into
finalize/verify (§3.2) — but properly: graph-structured + harness-generated, not a boss blob.

Three consequences, each coupling to the rest of the design:

1. **The oracle rises with the level.** A unit test's oracle is the function's spec rule; an integration
   test's oracle is the **contract's wiring section** + the goal-blueprint connections ("engine calls
   `bird.update()/flap()`; renderer reads `getBounds()`"). The harness owns both (`INTERFACE.md` + the goal
   map); the generator is seeded with the *wiring*, and the edges to test are exactly what `collab_for`
   already computes.
2. **Integration failures route straight to L2.** When `engine`-with-real-`renderer` fails, L1 (module-only)
   can't tell *which* module — or the wiring, or the contract — is wrong; only the arbiter has the graph +
   contract to attribute it. So an integration red **skips L1**. This is why §4.6 (history bundle +
   executed-truth) is the prerequisite: integration failures are the hardest to attribute, so they need the
   fullest history and a *run-the-composition-and-observe* fact over reasoning across five modules.
3. **Cost is self-targeting.** The 12-independent-utils scale task has **zero internal edges** → zero
   integration tests → nothing added to the throughput showcase; the coupled flappy app is almost all
   internal edges → all the value lands there. Readiness is the natural eager rule one level up from
   `testgen:X`: an integration test for node N is **ready when N's whole internal subtree exists on disk**
   (PA.7 §2).

**Build order (after the §4.6 trio, which makes integration failures repairable):** derive the internal-dep
DAG (reuse `collab_for` edges / parse the goal blueprint) → identify non-leaf nodes → generate one
integration test per internal node (real siblings, external-only stubs, seeded with the contract wiring) +
a root smoke test → integration reds skip to the arbiter. GPU-free fixture: a 2-module compose (one real
sibling + one external stub). Rationale for sequencing: building integration first would only generate
harder-to-attribute failures into a loop that cannot yet repair them.

**IMPLEMENTED 2026-06-23:** `module_refs()` (whole-word dep-edge detection, `--coord-test` R20) classifies
leaf vs internal node; `build_integration_task()` (R21) composes the REAL target + REAL siblings, stubbing
only the external boundary, output → `test/<node>.integration.test.js`. `finalize_verify` generates one per
internal node after unit test-gen (independent modules reference no siblings → none, so **zero cost on the
scale task**); a failing test that maps to no single module now **escalates straight to L2** (was a silent
break). **Partial:** integration-rework now gets the subtree-root module + siblings (`integration_base`,
`--coord-test` R22) so an integration failure is repairable with full context; no dedicated root smoke
test yet; GPU end-to-end pending.

---

## 5. Boss reports to the user — the live board

The boss is the single voice (§10). The harness renders the Board as a **live checklist** to stderr
(items appear as planned, flip `[ ]→[~]→[✓]`, show `[⚠]` on failure), and the boss gives the final
summary: *"4/5 modules pass; engine.js given up after 3 repairs — collision test still failing."*
One structure, three consumers (§1): harness writes, boss decides, user watches.

---

## 6. Flags

| flag | effect |
|---|---|
| `--verify` | run each piece's test + the repair loop (implies `--tools --allow-run`) |
| `--repair-budget N` | max repair rounds per item (default 3) |
| `--board` | render the live checklist (default on under `--verify`) |
| `--critic` (v2) | dedicated test-agent lane instead of harness-run tests |
| `--boss-coord` (v2) | boss reads the Board snapshot and emits intents |

---

## 7. Risks & mitigations

| risk | mitigation |
|---|---|
| Repair never terminates | **mandatory** `--repair-budget` cap → `GIVEN_UP` |
| Test agent runs untrusted model code | already sandboxed: `--work-dir`, `--allow-run` gate, destructive-cmd guard (PA.5) + `timeout` |
| The *test* is wrong, not the file | repairs loop to budget → `GIVEN_UP`; v2 boss `RESPEC` fixes the test |
| Board races | harness is the single writer; boss is read-only + intents |
| Repair cost (each round = a worker regen + a test run) | budget bounds it; report total rounds + wall in the summary |
| Coordination serializes (every round is a serial step) | same tax as §8.2; keep the default path quiet (verify only on tool output) |

---

## 8. Acceptance criteria

1. GPU-free `--coord-test` passes: Board transitions, repair-budget counting, amend-item builder,
   test-output parse.
2. On a task with a deliberately-failing module, the loop repairs it within budget and the Board
   shows `⚠ → ✓`.
3. A truly-broken piece hits the budget → `GIVEN_UP`, reported, and the run still completes.
4. The live checklist renders; the boss gives a final pass/fail summary.
5. Sandbox honored — no writes outside `--work-dir`; `execute_bash` gated by `--allow-run`.

---

## 9. Test plan

### 9.1 GPU-free unit tests (`--coord-test`, like `--gather-test` / `--mtp-test`)

| # | function | case | expected |
|---|---|---|---|
| C1 | board transition | plan → pop → DONE | status QUEUED→RUNNING→DONE |
| C2 | board transition | DONE → test fail (budget left) | FAILED + re-queued amend, repair_count=1 |
| C3 | board transition | fail at repair_count==budget | GIVEN_UP, no re-queue |
| C4 | `parse_test_result` | "✓ 5/5" / "OK" | pass=true |
| C5 | `parse_test_result` | "2 assertions failed" / "AssertionError" | pass=false |
| C6 | `build_amend_prompt` | file + error | contains `<<<CURRENT_FILE`, `<<<TEST_OUTPUT`, the error text |
| C7 | `render_board` | mixed statuses | `[✓]/[~]/[ ]/[⚠]` lines, repair counts |
| C8 | boss intent parse (v2) | `<<<REQUEUE w1: hint>>>` | {action=REQUEUE, id=w1, hint} |
| **R9** | `build_rework_user` (§4.2) | target+spec+module+test+error+guidance | all six present (fresh worker full triad) |
| **R10** | `build_test_task` (§4.2) | goal/contract + blueprint + module | all three present (test the spec) |

### 9.2 Integration (GPU)

- Module whose test fails once then passes after amend → Board `⚠→✓`, repair_count=1.
- Module that always fails → `GIVEN_UP` after N, run still finishes + reports.
- Live checklist renders during the run; boss final summary correct.

---

## 10. Build order

- **PA.4a** — Board + event transitions + live checklist render. Workers→DONE→✓ (visibility only).
- **PA.4b** — test agent verify: run each piece's test, mark ✓/⚠ on the Board. **DONE (2026-06-15).**
- **PA.4c** — amend re-queue + repair budget (the closed loop). **DONE (2026-06-15)** — wired into
  `finalize_verify`; ran 2 rounds on the full-arc flappy test (mechanism ✓), but module-only repair
  can't fix a buggy *test* (§4.1).
- **PA.4d** (next) — **escalate to boss on budget exhaustion (§4.1):** boss arbiter fixes the test /
  re-specs / accepts. Then boss-reads-Board intents, a `--critic` lane, and the movable-boss-lock (§8.4).

**Out of scope (v1)**: boss-intent coordination (PA.4d), the movable-boss-lock, streaming-path
(`run_pipeline`) coordination. v1 = harness-driven verify/repair on the pool path + the live Board.
