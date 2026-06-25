# PA.8 — Server / remote deployment: the client owns the environment, the server owns the cognition

**Goal**: run the packed-agent **cognition** (model + boss + lanes + planning) on a shared inference box,
while every **side effect and resource access** — writing files, running tests, reaching a database —
happens on the **calling client** (a programmer's coding tool), in the programmer's real environment. One
inference server, many dev machines. Captures the 2026-06-24 design conversation (user).

**Status**: DESIGN (conversation captured; not implemented). This is the brain/hands split — the brain is
packed remote inference instead of a co-located harness. It *generalizes existing hooks*, it does not
invent new mechanism: the Board (PA.4 §4.1/§5), the two-queue broker (PACKED_AGENTS_DESIGN §8.4), the
communication channels (§10), the mid-stream tool loop (PA.7b), the external-boundary test split (PA.4 §4.7),
and movable boss state (§8.4).

---

## 1. The one principle

**The client owns the environment; the server owns the cognition.** Three things the current *local* design
bakes into the server are the **same mistake** wearing three hats — the server presuming the client's world:

- **server-side execution** (running tests/tools where the model runs),
- **a server-side blackboard** (the work-dir on the inference box),
- **JS-hardcoded prompts + verify** (`CommonJS`, `module.exports`, `node *.test.js`).

Push all three to the client — client-side execution, a client-side `.blackboard/`, a client-declared stack
profile — and the three special cases collapse into one architecture. The "integrated shell tool" (the
user's own use) is just the degenerate config where client == server and the round-trips are free.

## 2. Why the client must execute — the firewall case (hard constraint, not a preference)

The motivating example (user): the LLM runs **outside** the firewall; the database it must test against is
**inside**. The check *can only* run on the dev machine. So client-side execution isn't a preference — it
is a **hard constraint** for any tool that touches a client resource (a DB, the real filesystem, the real
shell, an internal network). The server emits a tool *request*; the client executes it where the resource is
reachable and posts the result back.

## 3. What runs where

| concern | lives | why |
|---|---|---|
| plan / design / reconcile / rework decisions | **server** | pure reasoning over context — the client can't help *think* |
| Board, repair journal, arbiter diagnosis (§4.6 history) | **server**, checkpointed to client | coordination metadata *about* the work |
| `create_file`, `execute_bash`, run-tests, DB, real-fs reads | **client** | real side effects / real resources (§2) |
| code-under-construction + isolated test runs | **client `.blackboard/`** | durability + resource reach + no cwd churn (§4) |

Pure cognition never round-trips; everything with a side effect or a resource dependency does.

## 4. The client-side `.blackboard/`

A **durable working area on the dev machine, separate from the live source tree** — a `.blackboard/` dir in
the project (gitignored) or an external per-project dir. It is where agents **craft → iterate → run isolated
tests** without churning the user's working directory (the model's habitual `/tmp`, but durable, on the
client, and project-scoped). Verified artifacts are **promoted** into the real project as a reviewable diff —
the programmer sees the *result*, not the three repair iterations.

This one placement answers three worries at once:
- **durability** — survives a server session close (§5);
- **resource reach** — on the dev machine, so client tools (the DB check) run against it;
- **no churn** — isolated from the live tree.

## 5. Session durability — a stateless, resumable server

A shared inference box should be **stateless / poolable**: a client may drop, time out, or be re-assigned to
a different server. The expensive thing that *cannot* survive — the packed KV/SSM context — is GPU memory and
is intentionally **not** preserved. Instead:

- the server **checkpoints its coordination state** (Board, plan, blueprints, contract, journal — all small
  text) into the client's `.blackboard/` at each batch boundary;
- on resume, a fresh server session **reloads `.blackboard/` and re-derives** — it re-dispatches unfinished
  items rather than restoring lane state.

This is the existing **abandon-and-re-decode-fresh** model (PA.7b §3.1, chosen for SSM safety) scaled up:
missing-file-abandon and whole-session-resume are the *same move* — never preserve mid-generation state,
always re-derive from the durable blackboard. It is also the **movable-boss-state** idea (§8.4) with the
client's disk as the parking lot. Statelessness is what later enables one box to pool many clients (§11).

## 6. Status feedback — the Board on the wire

The Board (PA.4 §4.1) is *already* "one structure, three consumers: harness writes, boss decides, user
watches" (§5), rendered today as a stderr checklist. The server renders the same deltas as a client **event
stream** (SSE-style):

- `{type: board, item: bird, status: done, file: …}` — structured progress;
- `{type: voice, text: "design complete; implementing 4 modules"}` — the boss's natural-language status.

**Narrate cheaply.** The *harness* templates the structured Board deltas (deterministic, free); the **boss
generates prose only at genuine decision points** (the final summary, or "I need your input"). A boss that
narrates every stage re-introduces exactly the serial-boss cost the throughput work fought
(PACKED_AGENTS_DESIGN §1.1). Structured events free; boss generation reserved.

## 7. The tool round-trip — the broker routes execution

A lane emits a tool call; the broker (§8.4, at the batch boundary) classifies it:

- **server-local cognition** → no round-trip;
- **cached read** → serve from the server's read-through cache of `.blackboard/` (refresh round-trip if stale);
- **client execution** → emit a `tool_request` event, **park the lane** (the §4.2 blocked mechanism — other
  lanes keep decoding), inject the client's posted result, resume.

This is the **PA.7b mid-stream tool loop generalized to the wire**. A parked lane waits on a network + client
round-trip, but the batch does not stall (others advance) — the same "park the asker, others continue"
already in §4.2. There is a throughput *upside*: offloading test execution to the client keeps the GPU
**decoding cognition** instead of idling on `node`/`psql` subprocesses (today's serial verify step). The
server keeps a **read-through cache** of `.blackboard/` (client is source of truth) so context-rich prompts —
collaborator injection (§4.7), rework-with-current-module — don't pay a round-trip per read.

## 8. Local vs. remote client — one protocol, a routing knob

Client **locality decides the staging policy** (resolving the earlier local/remote question):

- **local client** (same machine / IPC): percolate *everything* — every write and test runs in the real
  `.blackboard/` immediately; round-trips ~free; maximally transparent.
- **remote client** (network): the server **stages** the iterative craft+unit-test loop against the cache +
  mocks (no per-iteration round-trips) and percolates only the **real-resource steps** (integration tests
  against the real DB — §4.7's external boundary) **+ the final promotion**. Amortizes latency.

Same protocol, same brain; the broker chooses "execute here vs. round-trip" from a declared client
**capability + locality** hint. The two use cases stop being two architectures.

## 9. Language-genericity — a client-declared stack profile

The prompts and verify are JS-baked, and the fix is the same principle (§1): the **client owns the stack**.
A **stack profile** the client declares:

- file conventions (extensions, test-file naming),
- **how to run a test + how to detect a pass** (the client owns "how to test here" — verify is a client tool
  call anyway),
- module / dependency conventions feeding the contract's TECH DECISIONS.

The server's prompts become stack-neutral — "in `<language>` following `<conventions>`, tested by
`<runner>`" — with specifics injected from the profile. **SQL example (user):** stack = SQL, deliverable =
`.sql`, test = run against a test DB via the client's `psql`, pass = exit code / expected rows. Same
pipeline, different profile.

**The de-JS-ification refactor** (replace hardcoded constants with profile lookups) threads through every
place JS is currently assumed: the `node *.test.js` verify command → `profile.test_cmd`; the `*.test.js`
glob → `profile.test_glob`; `comp_key`'s extension list; the reconcile contract's TECH DECISIONS; the
test-gen / integration-test prompts. Well-defined, but broad. **It pays off even in the local path today** —
a Python or SQL task currently can't be tested.

**Layer 1a IMPLEMENTED 2026-06-24 (`--lang`, `--coord-test` R24/R25):** a `StackProfile`
(lang / src_ext / test_suffix / runner) with `javascript` (default — a no-op), `python`, `sql`. The four
**core mechanics** are now profile-driven: file detection (`is_src_file`/`is_test_file`), the run command
(`run_test_cmd`), `module_for_test`, and `comp_key`. JS path verified unchanged (25/25; gather/eager/mtp
green). **Layer 1b (next):** the remaining path-string construction (`test/<comp>.test.js` literals,
integration-test naming) + the comp-extraction call sites. **Layer 2:** the prompts themselves
(`CommonJS`/`module.exports`/"node …" → profile-templated) — where cross-language behavior actually changes.
(The `--coord-test` fixtures are JS strings, so the suite pins `javascript`; the profile mechanism itself is
covered by R24/R25.)

## 9.1 Variables / mini-memories — the resolved-decision store (user, 2026-06-24)

The stack profile (§9) is the typed slice of a more general idea: a **session variable store** — named
key-value facts that accumulate and travel with every request, so a decision is resolved **once** and
injected everywhere, deterministically (no per-lane re-inference + drift).

**Three tiers, by lifetime:**
- **user-level** = **memories** (survive across sessions; seed every new one) — *"I always use pytest + ruff."*
- **project-level** = `.blackboard/` (persist for the project; brownfield reload) — the detected stack/conventions.
- **session-level** = the live task's dialog-resolved values (checkpointed to `.blackboard/` for resume).

**Prompts become templates — this is Layer 2.** Prompts carry `<var>` placeholders — *"write a
`<test_framework>` test for `<lang>`, asserting via `<assertion_style>`, runnable by `<runner>`"* — filled
from the store. The current JS constants become the **default values**, so an empty store = today's
behavior, and a populated store retargets without editing prompt text. This promotes reconcile's "TECH
DECISIONS" from prose to **typed, inspectable, overridable** key-values — and the block sits in the shared
system prefix, so it's **prefix-cache-friendly** (free per item).

**Variables are DISCOVERED mid-run, not just set up front (user — the MySQL/MyISAM case).** A worker or
tester can find a fact *during execution* — *"the database is MySQL / MyISAM"* — and write it back:
`db_engine=mysql`, `storage=MyISAM`, `supports_transactions=false`. The broker (§8.4) routes a `set_var`
marker into the store at the batch boundary; every subsequent lane carries the updated values. This is the
**discovered-dependency** model (PA.7b `NOT_FOUND` → re-queue) generalized from *files* to *facts*.

**Discovered > configured > inferred — the §4.6 principle, again.** For *environment facts* (DB engine,
runtime version), an **empirically-discovered** value (a worker actually connected) outranks a config-file
reading (which can be stale) which outranks a guess — exactly "executed-truth over reasoning" (PA4 §4.6)
applied to the fact-base. For *preferences* (test framework, style), **user-explicit** outranks
project-convention outranks inference. So precedence is **per-variable-kind**: *intent* variables defer to
the user; *fact* variables defer to observation. (And the three entry modes — greenfield / mid / brownfield
— are just which tier seeds the store first: model-inference / user-explicit / project-detection.)

**A late discovery can invalidate prior work → rework.** If code was written assuming transactions and then
`supports_transactions=false` is discovered, the transaction-using modules are **re-queued** — the same
rework operation as a failing test (PA.4) or contract drift (PA.7 §3). So there are now **three rework
triggers, one operation**: a failing test, a contract drift, and a **variable invalidation**. The store is a
*living* fact-base (like the living blueprint, §4.4): discoveries update it, downstream conforms, upstream
violators get a touch-up.

**Schema — the consequential variables** (what prompts template against), vs. freeform notes the boss may
stash but that don't drive generation: `lang`, `test_framework`, `runner`, `test_suffix`, `module_system`,
`assertion_style`, `deps_policy` (the stack slice — what §9's `StackProfile` already carries), plus
discovered environment facts (`db_engine`, `runtime_version`, …). The known set keeps the store from
becoming an untyped junk drawer; freeform stays allowed but non-templating.

**Build-order impact:** Layer 2 reframes from "edit every prompt" to **(a)** define the consequential-variable
schema + the store, **(b)** variabilize the prompts against it, **(c)** populate from the three tiers + the
`set_var` discovery marker + the variable-invalidation rework trigger. The `StackProfile` (§9 Layer 1a) is
the first typed entries.

**Store IMPLEMENTED 2026-06-24 (`--coord-test` R26-R28).** Format decision (user): **one line per item**,
not JSON — `key = value`, in `<project>/.blackboard/memory` (default `<work-dir>/.blackboard`, or
`--blackboard DIR` for the user's home-project location). Rationale: git-diffable (one var changed = one
line), hand-editable (`#` comments + blank lines ignored), and the **same syntax as the `SET_VAR key=value`
discovery marker** — store line and marker are identical. **Multiline:** schema vars are single-line by
nature; a rare freeform note escapes newline → `\n` (and `\\`), so one-line-per-item always holds. Split on
the **first** `=` (values may contain `=`, e.g. a connection string). Wiring: `vars_seed_from_stack` +
`vars_load` at startup (stored > stack-default); `vars_render()` prepends the block to the shared goal +
the test-gen/repair context (cached prefix); `run_worker_tools` runs `vars_absorb` on every worker output
and `vars_save`s discoveries; the worker preamble advertises `SET_VAR`. **Still Layer 2:** the prompts'
hardcoded `console.assert`/`node`/`*.test.js` aren't yet templated against the vars (needs
`assertion_style`/`test_framework` entries); the **variable-invalidation rework trigger** isn't wired
(discovery updates the store + flows to later lanes, but doesn't yet re-queue already-built violators).

## 10. The session-start contract

The client announces, up front (MCP-shaped): its **tools + schemas**, its **locality** (local/remote), its
**permission policy** (what it will let the server run), and its **stack profile**. That single declaration
is what makes the broker's local/remote routing (§8) and the stack-neutral prompts (§9) deterministic.

## 11. Risks / open questions

- **Trust both ways.** The server runs model-written code in `.blackboard/` (sandbox there — the existing
  `--work-dir` + destructive guard); the client executes server-emitted tool requests (needs its own
  approval/sandbox layer — the Claude Code permission model is the reference). Neither side blindly trusts.
- **Concurrency / VRAM.** Weights load once, but KV/SSM is per session and the `n_seq_max=4` packing is per
  session. Many clients = many contexts or time-sharing. One active session per server for v1; statelessness
  (§5) is the path to pooling.
- **Cache coherence.** The server's read cache of `.blackboard/` must invalidate on client writes; a stale
  sibling injected into a prompt is a silent drift bug.
- **Round-trip latency for remote clients.** The staging policy (§8) mitigates; the park-vs-abandon rule
  (PA.7b §3.1) decides whether a slow client tool idles a lane or frees it.
- **Do design artifacts (blueprints/contract) sync to the client or stay behind the event stream?** The line
  between "the agent litters my repo with design files" and "the agent shows its thinking but only commits
  code." Open (user).

## 12. Build order (when pursued)

1. **Stack profile** (§9) — de-JS-ify. Benefits the *local* path immediately (Python/SQL today), no server
   needed yet.
2. **`.blackboard/` + promotion** (§4) — client-side durable workspace, separate from cwd; verified-artifact
   promotion.
3. **Event stream** (§6) — Board deltas + boss voice over SSE; harness-narrated.
4. **Tool round-trip over the wire** (§7) — the PA.7b loop to a remote executor; broker routing (§8); the
   session contract (§10).
5. **Stateless resume** (§5) — checkpoint coordination state to `.blackboard/`; resume re-derives.

Out of scope (v1): multi-client concurrency on one server; cross-session scheduling.
