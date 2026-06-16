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
