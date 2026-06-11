# Packed Agents — Detailed Design

**One model, one GPU, four cooperating agent streams in a single batched decode.**
A *boss* (seq 0) decomposes a coding task into independent pieces; three *workers*
(seq 1–3) implement them simultaneously; the boss integrates. All four advance in one
`llama_decode` per step.

Status: design (v1 + v2 arc). Substrate (`pti_agents.cpp`, PA.0) written, **not yet built
or run** — the 2.15× headline is still a prediction from M5.1, not a number this binary
has produced. See [§9 Milestones](#9-milestones); PA.0 is gate 0.

Companion docs: [`PACKED_AGENTS.md`](../PACKED_AGENTS.md) (1-page summary),
[`blueprint-prompt.md`](blueprint-prompt.md) (the spec language the team speaks).

---

## 1. Goal & honest economics

The win is **wall-clock on splittable coding tasks**, bought with idle batch capacity we
already proved exists. It is *not* 4×.

| quantity | value | source |
|---|---|---|
| single-stream baseline | 19.0–19.4 tok/s | M-series baseline |
| 4-seq packed step cost | **1.86×** one stream | pti_4seq / M5.1, measured |
| ⇒ aggregate token rate, 4 lanes | 2.15× derived → **1.95× measured** | M5.1 + **PA.0 run (2026-06-10)** |
| worker-phase wall-clock, 3 balanced pieces | `3·L` seq vs `max(Lᵢ)·1.86` packed ≈ **1.6×** | derived |
| with boss productively busy (4th lane) | **2.15×** | the boss is not idle — see §5 |
| + per-stream MTP/lookup drafts (v2) | `4·1.9 / 3.11 ≈ **2.4×**` | M6.0 curve + M7.1 accept |

**The 1.6× vs 2.15× gap is the boss.** If the boss waits while three workers run, you
get ~1.6×. If the boss takes a fourth piece (integration glue, tests, scaffolding) during
the parallel phase, all four lanes do wanted work and you get 2.15×. The design treats the
boss as a worker-that-also-coordinates, never as an idle dispatcher. This is the single
most important economic decision in the doc.

"4×" is the *in-flight parallelism* (four token streams alive at once). The *speedup* is
1.6–2.4×. State it that way everywhere; it is still the largest lever on the board for
splittable work.

---

## 2. Substrate contract (what the hardware/llama.cpp give us)

Everything below is measured and lives in the codebase already.

- **Batched decode, per-seq state.** N sequences advance in one `llama_decode`; each seq
  has independent KV **and** SSM state (hybrid model). Proven in `pti_4seq`.
- **Cost curve** (one batched step vs one single-token step): 1.20×@b2, 1.71×@b4,
  3.11×@b8, 4.47×@b16, 6.93×@b32; ≈12.1 ms/tok at b32. 4 real lanes sit at b4 = 1.71×
  for decode, ~1.86× including the per-step overhead measured end-to-end.
- **inject(seq, text)** = tokenize + prefill into that seq at its current position. Same
  mechanics as the M7.5 delta prefill. This is how fan-out, refill, broadcast, and gather
  all move tokens into a stream.
- **Context budget is divided, not shared.** Non-unified KV ⇒ `usable_ctx = n_ctx /
  n_seq_max`. v1: `n_seq_max = 4` ⇒ ctx/4 per lane. v2 with a checkpoint seq per stream:
  `n_seq_max = 8` ⇒ ctx/8. **llama.cpp reserves equal ctx per seq — asymmetric budgets
  are not supported** — so the boss's context need sets the reservation everyone pays.
- **VRAM barely moves — this is the whole feasibility argument.** The model weights
  (~24 GB for UD-Q6_K_XL) load **once** and are shared by every lane; only KV + SSM state is
  per-sequence. At a **fixed `n_ctx`** the KV pool is a fixed reservation *subdivided* by
  `n_seq_max`, so going 1→4 lanes adds **≈0** VRAM — you are spending batch/context slots
  already paid for. (If instead you grow `n_ctx` to keep each lane's length constant, total
  KV grows ~linearly in `n_seq_max` — but the 24 GB of weights still loads once.) Four agents
  for the price of one model + a slice of KV is why this fits on a single 32 GB MI50.
- **Per-stream sampling.** Position-keyed RNG `splitmix64(seed ^ pos)` (M7.4): each lane
  carries its own seed → independent, reproducible sampling at any temperature.
- **Checkpoint/rollback** (for v2 speculation): `seq_cp` ≈ 0.02 ms; cheap.

### 2.1 The exactness constraint (do not skip — this is correctness, not polish)

In a packed batch, **a worker's output must not depend on which siblings are co-resident
or on the batch shrinking as lanes finish.** If it does, the artifact a worker produces
becomes a function of timing, and the run is neither reproducible nor reviewable.

With non-unified KV there is no cross-seq attention, so lane A attends only to its own KV;
A's logits are *mathematically* independent of B. What remains is floating-point
nondeterminism from batch shape — the exact effect the project already tamed:

> **Hard rules (M7.2, regress at your peril):** KV must be **f16** (Q8_0 KV is
> batch-size-dependent under FA), **`kv_unified = false`**, and argmax needs the
> **ε = 0.05 lowest-id tie-break**. All `pti_*` binaries share this argmax — `pti_agents`
> already has `ARGMAX_EPS = 0.05f`; keep it in sync.

**Scope of the guarantee, stated honestly:** a worker's output is byte-identical packed-vs-
solo *modulo floating-point near-ties wider than ε* (observed elsewhere at `<think>`
boundaries and repetition forks). That is the same honesty bound the rest of the project
ships under. **PA.0's acceptance gate is to verify this empirically** — run each stream
packed and again alone, assert identical output (§9). The cooperative model is built on
this holding; do not assume it, measure it. **Measured 2026-06-11 (PA.0): PASS** — 4 lanes
byte-identical packed-vs-solo at n=96, with `kv_unified=false` + f16 KV + ε=0.05 tie-break.

---

## 3. The team's shared language: BluePrint

The boss and workers communicate in **BluePrint** (`spec/blueprint-prompt.md`). This is the
spine of the design and it reconciles the two choices "BluePrint contract" + "harness-routed":

- The **boss runs the full BluePrint framework** (`spec/blueprint-prompt.md` is its system
  prompt) — it operates *specification mode* and authors a mini-spec per lane.
- A **worker does NOT carry the framework.** It receives only the boss's BluePrint *output*
  for its piece, plus that piece's **inputs / outputs / language** and a one-line "produce
  JUST this part" instruction. That is enough to compile against — and it keeps each worker's
  context tiny: the ~16 KB framework prompt lives **once**, in the boss, not ×3 in lanes that
  must fit in ctx/N. The `compile` semantics are baked into the lean worker preamble (§5.2),
  not shipped as the whole framework.
- The **harness routes**: it does not understand BluePrint semantics, only the thin
  envelope that wraps each spec (§4). Spec richness for the model, machine boundaries for
  the router.

Why this is better than freeform sub-prompts: BluePrint specs carry **declared exported
symbols, interfaces, pre/postconditions, and test descriptions**. The boss declares each
piece's exports up front → the harness does a collision check before fan-out, and the
gather phase has a contract to diff actual output against (§6).

---

## 4. Work-order envelope (the one thing the harness parses)

The boss emits exactly one planning block. The harness splits it on delimiters; everything
*inside* a piece is BluePrint (or prose) the harness passes through verbatim.

```
<<<PLAN strategy=function lang=js>>>
shared:
<blueprint>
  // the "public params of the current class" — what every lane may rely on
  World { pipes: Pipe[], bird: {x,y,vy}, score: int, gravity: float }
  Pipe  { x: float, gapY: float, gapH: float, passed: bool }
  deps: []   // allowed external libraries; [] = vanilla js, no imports
</blueprint>
<<<PIECE id=w1 exports=gravityStep>>>
instruction: implement this function given the following public params of the current class, in js
<blueprint> gravityStep(bird, dt) -> void { apply gravity then velocity to bird } </blueprint>
<<<PIECE id=w2 exports=spawnPipe>>>
instruction: implement this function given the following public params of the current class, in js
<blueprint> spawnPipe(world) -> void { append a Pipe with a random gap to world.pipes } </blueprint>
<<<PIECE id=w3 exports=checkCollision>>>
instruction: implement this function given the following public params of the current class, in js
<blueprint> checkCollision(bird, pipes) -> bool { true if bird hits a pipe or the ground } </blueprint>
<<<SELF id=boss exports=stepWorld,smokeTest>>>
instruction: implement this object in js — wire the three functions into the game loop and a smoke test
<blueprint> Integration { stepWorld(world, dt) -> void, smokeTest() -> bool } </blueprint>
<<<END>>>
```

Rules the parser enforces (and the prompt promises):
- 1 `shared` block + **≤3 `PIECE`** + exactly 1 `SELF` block (the boss's 4th lane).
- `exports=` is a comma list of every top-level symbol the piece will define. Union must be
  **collision-free**; a collision is a hard error → re-plan (v1: warn + abort; PA.1+: one
  re-plan round).
- `lang=` on `PLAN` is the default language; a `PIECE` may override with its own `lang=`.
  Each worker is handed only its spec body + its `exports`/signature + `lang` + a fixed
  "produce only this piece" instruction — **not** the BluePrint framework (§3). The boss's
  spec for a piece must therefore be self-contained.
- `instruction:` (one line per piece) is the lean imperative the worker reads verbatim.
  Canonical forms: *"implement this function given the following public params of the current
  class, in `<lang>`"* (function strategy, the shared block **is** those public params) and
  *"implement this object in `<lang>`"* (file/object strategy). The worker's whole prompt is
  **standing preamble (§5.2) + this line + the spec body** — nothing else.
- Delimiters chosen for parse-robustness. The model already reliably emits XML-ish markup
  (`<function=…><parameter=…>`, M7.8) despite the template; `<<<…>>>` line-anchored markers
  are in the same comfort zone. **Validate delimiter reliability empirically in PA.1** — a
  malformed envelope is the most likely v1 failure and needs a fallback (re-ask the boss
  with the parse error injected).

### 4.1 Data structures (extend `pti_agents.cpp`)

```cpp
struct Piece {
    std::string id;                       // "w1".."w3", "boss"
    std::vector<std::string> exports;     // declared top-level symbols (the outputs/signature)
    std::string language;                 // target language (defaults from PLAN lang=)
    std::string instructions;             // worker-facing imperative, e.g.
                                          // "implement this function given the class's public params, in js"
    std::string blueprint;                // spec body; carries inputs/outputs via signature
    llama_seq_id seq;                     // assigned lane
};
struct WorkOrder {
    std::string  strategy;                // function | file | role
    std::string  shared;                  // frozen interface, prepended to every lane
    std::vector<Piece> pieces;            // [w1,w2,w3,boss]; boss is pieces.back()
};
// Stream already exists in pti_agents.cpp (prompt_toks, text, tok_last, pos, n_gen, live).
// Add: const Piece* piece;  bool is_boss;  bool blocked=false;  int q_budget=2;
```

### 4.2 Coordination messages (the chat channel)

The default is one-shot: the boss assigns a clear spec, the worker implements, no talking —
that silence is what keeps lanes parallel. A worker **genuinely blocked** by an ambiguous
spec may raise one structured question; the boss answers; the worker resumes. Two shapes,
both caught by the harness at a batch boundary:

```
worker → boss:   <<<ASK to=boss>>> short question <<<END>>>
boss   → worker: <<<REPLY to=w2>>> short answer  <<<END>>>     (also GUIDE / KILL, §8.2)
```

A worker emitting `ASK` is parked (`blocked = true`, dropped from the batch) until its reply
is injected. Questions are **budgeted** (`q_budget`, default 2/worker): the channel is an
exception path, not a conversation. Scheduling and the (serial) cost are in §8.2.

---

## 5. Phase machine

```
seq 0 = BOSS, seq 1..3 = WORKERS, one llama_context, n_seq_max = 4 (v1) / 8 (v2)
```

### Phase 1 — PLAN (boss alone)

Only seq 0 is live. Boss runs the **decomposition prompt** (§5.1) over the user task and
emits the work-order envelope. Harness parses → `WorkOrder`, runs the collision check,
assigns seqs (`w1→1, w2→2, w3→3, boss→0`). The boss keeps `SELF` as its own piece.

Cost: this phase is single-stream (1×), unavoidable serial latency before any parallelism.
Keep the decomposition prompt tight; the plan is short (specs, not code).

### Phase 2 — FAN-OUT (batched prefill)

For each lane i, build `worker_prompt(i) = worker_preamble + shared + piece[i].blueprint`.
The `worker_preamble + shared` prefix is identical across workers but **cannot be KV-shared**
(non-unified). Mitigation: prefill all lanes in **one batched, chunked pass** (tokens tagged
per seq) so the shared-prefix recompute *overlaps* across lanes instead of running 3×
serially. Honest: the recompute cost is still paid; batching only hides it behind the batch.

The boss's `SELF` piece is prefilled into seq 0 too — boss starts its integration/test lane
now, so all four lanes enter Phase 3 hot.

### Phase 3 — PARALLEL (the packed loop)

Exactly the PA.0 loop: one token per live stream, one `llama_decode` per step; each lane
samples with its own seed; lanes hit EOG independently; the batch shrinks as they finish.
This is the 2.15× region. The boss lane writes glue/tests *against the declared interfaces*
(it cannot see worker bodies yet — that is Phase 4). A worker may raise one bounded
clarifying question here (§4.2 / §8.2), which parks its lane until the boss replies; v1
disables the channel (one-shot), PA.4 enables it.

### Phase 3.5 — REFILL (v1.1 / PA.2)

If the plan produced a backlog (>3 worker pieces), a finished worker slot gets the next
piece injected at its current position (`inject(seq, next.blueprint)`), keeping the batch
full. Stragglers are the dominant efficiency leak (§7), so the plan prompt asks for
similar-sized pieces and refill recovers the rest. v1 ships **without** a backlog (exactly
3 workers); refill is PA.2.

### Phase 4 — GATHER (boss integrates)

Each finished worker's output is injected into seq 0 as a tagged message:
`inject(0, "// worker w1 produced:\n" + stream[1].text)`. With all three bodies plus its
own glue in context, the boss: (a) diffs actual exports against declared `exports=` (drift
detection), (b) resolves symbol collisions and dedups overlapping helpers, (c) emits the
final artifact. Output written via `--out`.

### 5.1 Boss prompt — decompose · coordinate · integrate (draft)

> You are the **coordinator** (lane `boss`, seq 0) of a packed-agent team: you plus **three
> workers `w1 w2 w3`** that decode **in parallel on one GPU context**. Workers cannot see
> each other or you once they start — each sees only the shared interface, its own spec, and
> a one-line instruction. Plan *and* coordinate accordingly.
>
> Given a coding task, produce a **work order**:
> 1. Choose a split **strategy** and say why:
>    - **function** — independent functions sharing one header/interface. Cleanest
>      isolation; prefer when the task is a set of pure-ish functions.
>    - **file** — each lane owns a whole file/module. Use for multi-file features; expect
>      bigger, less even pieces.
>    - **role** — impl / tests / docs over one target. Use when the interface is already
>      frozen and the value is doing all three at once.
> 2. Write a **shared** block: the frozen interface (types, signatures, layout) **and the
>    allowed external libraries** (`deps`) every lane may use. This is the environment
>    contract — complete it; it is what prevents both merge conflicts *and* dependency
>    sprawl. Workers use only the libraries you list here unless they ask.
> 3. Split the work into **≤3 worker pieces + 1 piece you keep** (`SELF`: integration,
>    smoke tests, scaffolding — you will be busy while they work, not waiting).
> 4. For each piece author a **BluePrint mini-spec** (see the BluePrint framework) and
>    declare its **inputs, outputs, target language, and every top-level symbol it exports**
>    — the worker receives only this, not the framework, so each spec must be self-contained.
>
> Hard constraints: pieces must be **independent** (no shared mutable state mid-flight);
> **exported symbols must not collide**; keep pieces **similar-sized** (a slow lane idles
> the batch). Emit the envelope exactly as specified, then stop.
>
> **Then coordinate while they work.** After `<<<END>>>` you begin your own `SELF` piece.
> Between steps you may receive worker messages:
> - `// wN asks: <q>` → reply with one `<<<REPLY to=wN>>> … <<<END>>>`: brief, decisive,
>   just enough to unblock — do **not** redesign the interface mid-flight.
> - A common ask is a **library request** (`need: <lib>`). Grant only if a listed `dep`
>   doesn't already cover it; **record it for the manifest**, and `<<<GUIDE>>>` the same
>   choice to the other lanes if consistency matters (one HTTP client, one date lib).
> - If a worker drifts, `<<<GUIDE to=wN>>> … <<<END>>>`; if it is unrecoverable,
>   `<<<KILL to=wN>>><<<END>>>` to retry that piece.
> - Every word here is **serial** — it pauses the asker and detours you — so keep it short.
>
> **Finally integrate.** When the workers finish you receive their outputs; reconcile them
> against the declared interfaces, dedup overlapping helpers, and emit the final artifact.
>
> *(v1 runs plan + integrate only; the coordinate block is exercised from PA.4 on.)*

### 5.2 Worker preamble (v1 draft)

> You implement one piece of a larger program. You are given: a frozen shared interface, a
> BluePrint spec for **your piece only**, and its **inputs**, **outputs**, and **language**.
> Produce **just that piece** — the implementation and its tests for your declared exports,
> in the given language, and **nothing else**: no prose, no other functions, no re-declaring
> the shared interface. Match the declared signature exactly, and **use only the libraries in the shared `deps`** —
> if you need another, emit `<<<ASK to=boss>>> need: <lib> for <why> <<<END>>>` rather than
> importing it. If the spec is genuinely
> ambiguous and you cannot proceed, emit one `<<<ASK to=boss>>>…<<<END>>>` and wait — but
> prefer to implement against the spec as written; a question costs the whole team time.

(Note the asymmetry, per §3: this lean preamble is the worker's *entire* standing prompt.
The BluePrint framework is the **boss's** system prompt, never loaded into a worker lane.)

---

## 6. Gather & conflict resolution

- **Declared-export collision (plan time):** union of all `exports=` must be unique. The
  shared interface's symbols are reserved. Caught before any GPU work is spent on fan-out.
- **Actual-vs-declared drift (gather time):** the boss diffs each worker's real top-level
  defs against its declared `exports=`. Drift (worker defined an undeclared symbol, or
  skipped a declared one) is flagged into the boss's gather context for reconciliation.
- **Overlapping helpers:** two workers independently writing the same private helper is
  expected and acceptable (workers are isolated by design). The boss dedups at gather.
  Function-level splits make this rare; role splits make it near-impossible.
- **Dependency manifest:** the boss owns the union of declared `deps` + any libraries it
  granted mid-flight, and emits it as the artifact's manifest (e.g. `package.json`),
  resolving duplicates and version conflicts (two libs for one job → pick one, rewrite the
  loser's import). Without a single owner, four isolated lanes ship four ad-hoc dep sets.
- **v1 reconciliation is the boss's job in natural language** (it has all bodies in
  context). A structured merge (AST-level) is explicitly out of scope — out of altitude
  for a single-model team and unnecessary at function granularity.

---

## 7. Risks / open questions

1. **Split quality** — the entire win depends on the boss producing genuinely independent
   pieces with clean interfaces. Mitigation: strict envelope schema, declared exports,
   collision check, gather drift-diff. Measured in PA.1 (artifact quality gate).
2. **Stragglers** — unequal pieces idle the batch; a 4-lane batch with 1 live lane is back
   to ~1× and you have paid 1.86× overhead for nothing. Mitigation: "similar-sized" in the
   prompt + work-queue refill (PA.2) + checkpoint kill of runaways (PA.4).
3. **Plan latency** — Phase 1 is serial (1×) before any parallelism. For small tasks the
   plan tax can erase the worker-phase win. Open: a minimum-task-size threshold below which
   `pti_agents` should just run one lane. Measure the crossover in PA.1.
4. **Envelope parse failures** — most likely v1 break. Mitigation: robust delimiters +
   re-ask-with-error fallback (§4). Needs a retry budget so a stubborn boss can't loop.
5. **Context reservation** — equal ctx/seq means the boss's big context is charged to every
   lane (×4, ×8 with speculation). At `-c 64000` q8 that is ~16k/lane (v1) or ~8k/lane
   (v2). Worker pieces are small; the boss's shared+task context is the binding constraint.
   Open: is ctx/8 enough for the boss on real tasks, or does v2 speculation need a bigger
   `-c`?
6. **Worker isolation vs. consistency** — workers can't see each other mid-flight (by
   design). If two pieces turn out coupled, only gather catches it, by which point both
   lanes' tokens are spent. Checkpoint broadcast (v2, §8.2) is the mid-flight escape hatch.
7. **Q&A serialization** — every worker question is a serial boss round-trip; a chatty team
   collapses toward ~1×. Mitigation: clear specs by construction (§5.1), the `q_budget` cap,
   and treating budget-exhaustion as a split-quality failure, not a feature (§8.2).
8. **Dependency sprawl** — isolated workers reaching for libraries independently yield a
   bloated or conflicting manifest (two HTTP clients, mismatched versions). Mitigation: an
   allowed-`deps` list in `shared`; new libs require an `ASK` (`need: <lib>`); the boss owns
   the merged manifest at gather (§6).

---

## 8. v2 levers

### 8.1 Speculation stacking (PA.3)

Each lane carries its own draft token (MTP head or n-gram lookup), verified in the same
batched step: batch up to **8** (4 real + 4 draft) at 3.11× → ~**2.4× aggregate** before
any lookup hits. Constraints carried from the M7 work:

- **MTP multi-token-context bug is unfixed** — multi-token MTP-context batches produce
  garbage (~48% accept). Use **last-pair-only feeds per lane**, exactly as `pti_lookup
  --mtp` does today.
- **Rollback needs a checkpoint seq per stream** ⇒ `n_seq_max = 8` ⇒ **usable ctx/8**.
  Speculation costs context; that is the trade. v1 ships **without** speculation
  (`n_seq_max = 4`, ctx/4); v2 adds it behind a flag.
- **Per-stream lookup** hits within a lane's *own* preamble+output history (histories are
  per-stream; workers do not share an n-gram table). Workers writing similar functions do
  **not** warm each other's tables — do not design as if they do.
- Position-keyed sampled verification (M7.4) keeps each lane reproducible under speculation.

### 8.2 Bidirectional coordination (PA.4)

Default flow is assign → implement → gather with no mid-flight talk: clear subtasks are the
boss's job (§5.1) and silence keeps the four lanes parallel. The coordination channel is the
**bounded** exception. Every batch boundary is a free sync point (all lanes pause between
steps anyway), so both directions ride the same machinery.

**Worker → boss (questions).** A blocked worker emits `<<<ASK>>>` (§4.2):
1. Harness parks the lane (`blocked`, out of the batch) and posts the question to the boss:
   `inject(0, "// w2 asks: <q>\n// reply with <<<REPLY to=w2>>>…")`.
2. The boss answers at its next turn; the harness extracts the `REPLY` body and `inject`s it
   into w2 at its parked position; `blocked = false`; w2 resumes.
3. **Cost, plainly:** w2 idles for the round-trip and the boss's SELF lane takes a detour —
   a Q&A round is **serial**, so it erodes the 2.15×. Hence `q_budget`. A worker that
   exhausts its budget is a **plan-quality signal** (the spec was underspecified) — log it
   for the split-quality metric (risk #1), don't treat it as normal.

**Boss → worker (guidance / kill).** At a checkpoint cadence N the boss may, unprompted,
inspect worker tails and:
- **GUIDE** a drifting worker ("the signature changed, re-read shared"),
- **broadcast** a correction into all workers (inject at each lane's checkpoint),
- **KILL/retry** a runaway (`seq_rm` the lane, re-inject the piece, or reassign).

Both directions need the checkpoint-seq machinery (§8.1) and a boss-side policy for *when* to
intervene. Open: cadence N, the intervention heuristic, and whether a parked worker should
**hard-wait** (simple, idles the lane) or **speculate** past the question on an assumed
answer and roll back if wrong (faster, needs the v2 checkpoint seq). v1 stays one-shot (no
channel); this is PA.4. **PA.4 is independent of PA.3 (speculation)** — if conversation
quality matters more than the extra ~0.25×, do PA.4 first.

---

## 9. Milestones

| id | deliverable | acceptance gate |
|---|---|---|
| **PA.0** | plumbing demo (`pti_agents.cpp`): 4 prompts, one context, packed vs sequential | **DONE: ~1.9× aggregate** (1.87–1.95× run-to-run; 19.3 → ~37 tok/s, 4 independent buffers) and **byte-identity gate PASS (2026-06-11)** — all 4 lanes byte-identical packed-vs-solo, asserted in-binary (`kv_unified=false`, exits non-zero on divergence). Survivors-continue-after-EOG path coded, not yet exercised (equal-cap run). |
| **PA.1** | phased pipeline: plan → fan-out → parallel → gather on a canned task ("class with 3 methods"); boss authors BluePrint, harness routes | end-to-end artifact compiles & passes the boss's smoke test; envelope parses reliably over ≥20 plans; measure plan-tax crossover (risk #3) |
| **PA.2** | work-queue refill + straggler stats | batch stays full with a >3-piece backlog; report idle-lane-fraction |
| **PA.3** | speculation stacking (MTP/lookup per stream, `n_seq_max=8`) | ~2.4× aggregate; per-lane output still reproducible |
| **PA.4** | bidirectional coordination: worker `ASK`→boss `REPLY`, plus boss GUIDE/KILL (independent of PA.3) | a worker blocked on an ambiguous spec gets an answer and finishes; boss kills+retries a sabotaged runaway within N tokens; Q&A stays within `q_budget` |

**Immediate next action:** PA.0 is built (`make agents`), measured (**~1.9× aggregate**,
1.87–1.95× run-to-run), and the **byte-identity gate PASSES** (2026-06-11; 4 lanes identical
packed-vs-solo, `kv_unified=false`, exits non-zero on divergence). Next: make the lane count
configurable (boss picks **4–16** by cleanest split, bounded by the measured speed/ctx
tradeoff), then **PA.1** (boss → workers → gather pipeline).

---

## 10. Decisions on record

- **Scope:** full arc (v1 core + v2 speculation/checkpoint) — this doc.
- **Contract:** BluePrint mini-specs inside a thin routing envelope. Boss authors, workers
  compile, harness routes only the envelope.
- **Orchestration:** one C++ binary (`pti_agents`), harness-routed; all lanes in one
  `llama_context`. No tool-call round-trips, no external orchestrator in v1.
- **Task shape:** boss chooses function / file / role per task via a selection rubric
  (§5.1); not hard-coded.
- **Boss is a fourth worker,** not an idle dispatcher — this is what buys 2.15× over 1.6×.
- **Communication:** v1 is one-shot assign → implement → gather (no mid-flight talk — keeps
  lanes parallel). A **bounded** bidirectional channel (worker questions + boss guidance/
  kill, `q_budget`-capped) is **PA.4**, independent of speculation. Clear subtasks are the
  default; chat is the exception, and every round is serial.
- **Memory:** one model in VRAM (~24 GB, loaded once); lanes add only KV/SSM state, and at
  fixed `n_ctx` that is a *subdivision* of an already-paid pool, not new allocation.
