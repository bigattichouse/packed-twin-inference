# PA.4 — Coordination: verify/repair loop + live task board (blackboard)

**Goal**: turn the fire-and-forget pool into a **closed agentic loop**. Workers write files +
tests; a *test agent* runs the tests; failures re-queue as **amend** items; the boss reasons over
a shared **Board** (the blackboard) and reports progress to the user as a **live checklist**. The
packed team's payoff is not raw speed — it's **autonomously verified, test-passing, multi-file
output** that a one-shot model cannot self-correct.

**Status**: design. Builds on PA.2 (pool + refill queue), PA.5 (`--tools`: `create_file`/
`execute_bash`), PA.1c (gather). Realizes design §8.2 (bidirectional coordination), §8.4 (two
queues / blackboard), §10 (boss = single voice to the user).

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

### 9.2 Integration (GPU)

- Module whose test fails once then passes after amend → Board `⚠→✓`, repair_count=1.
- Module that always fails → `GIVEN_UP` after N, run still finishes + reports.
- Live checklist renders during the run; boss final summary correct.

---

## 10. Build order

- **PA.4a** — Board + event transitions + live checklist render. Workers→DONE→✓ (visibility only).
- **PA.4b** — test agent verify: run each piece's test, mark ✓/⚠ on the Board.
- **PA.4c** — amend re-queue + repair budget (the closed loop). *This is the headline.*
- **PA.4d** (v2) — boss reads the Board snapshot + emits intents (`REQUEUE`/`RESPEC`/`GIVE_UP`);
  dedicated `--critic` lane; the movable-boss-lock (§8.4).

**Out of scope (v1)**: boss-intent coordination (PA.4d), the movable-boss-lock, streaming-path
(`run_pipeline`) coordination. v1 = harness-driven verify/repair on the pool path + the live Board.
