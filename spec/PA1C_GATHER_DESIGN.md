# PA.1c — Gather Phase Design

**Goal**: After all worker pieces are generated, assemble them into a single final artifact
(a runnable, cohesive file) and write it to `--out`.

**Current state**: Workers produce individual outputs printed to stdout with `───── item <id> ─────` headers. No assembly happens. The boss SELF piece generates glue code against declared interfaces only.

---

## Design Decision: Simple Concatenation vs. Boss-Led Integration

Two approaches:

### A. Simple concatenation (no additional decode)

Concatenate all pieces in order: shared interface → worker outputs → boss glue. No model
involved in the merge step.

Pros:
- Zero cost, zero risk
- Deterministic

Cons:
- Workers may emit prose wrappers ("Here is your function:") → garbage in the artifact
- Boss glue is written against declared signatures, not actual worker output → may not compile
- No dedup of overlapping helpers
- No dependency manifest

### B. Boss-led integration (additional decode pass)

Inject all worker outputs into the boss's context, let the boss write the final merged
artifact.

Pros:
- Boss sees actual implementations, can reconcile drift
- Boss can strip prose, dedup helpers, write a proper manifest
- Single coherent output from a single model perspective

Cons:
- One additional serial decode (the boss's gather pass)
- Boss context grows significantly (shared + all worker outputs + gather prompt)
- Adds serial latency after the parallel phase

### Decision: B (boss-led integration)

The design doc is explicit about this (§6: *"v1 reconciliation is the boss's job in natural language"*). The parallel phase already did the expensive work; the gather pass is a single
stream at the end. The quality difference is substantial — a boss that sees actual code
produces compilable glue.

---

## Architecture

```
Phase 1  PLAN       boss alone → work order (already works)
Phase 2  FAN-OUT    prefill all lanes (already works)
Phase 3  PARALLEL   packed decode, workers + boss SELF (already works)
Phase 4  GATHER     boss alone → final artifact (NEW)
```

Phase 4 is **serial** (one stream), so it runs at baseline speed. The total wall-clock is:

```
T_total = T_plan + max(T_workers) + T_gather
```

Where `max(T_workers)` is the longest piece (the parallel phase). For balanced pieces, this
is negligible overhead. For a 1000-token artifact, gather adds ~50s at 19 tok/s — acceptable
given the quality gain.

### Phase 4 mechanics

1. **Collect all worker outputs** (already done in `outv`)
2. **Inject them into the boss's context** using `inject(BOSS_SEQ, text)` — same mechanics
   as delta prefill (PA.2.1)
3. **Append a gather prompt** instructing the boss to produce the final artifact
4. **Decode the boss alone** until EOG or cap
5. **Post-process**: extract the final artifact, write to `--out`

### Inject mechanics

All worker outputs are injected as tagged messages into the boss's seq at its current
position. The injection format uses the same marker envelope for parseability:

```
<<<WORKER_DONE id=w1 exports=gravityStep>>>
[actual worker output here]
<<<END_WORKER>>>
<<<WORKER_DONE id=w2 exports=spawnPipe>>>
[actual worker output here]
<<<END_WORKER>>>
...
<<<GATHER_INSTRUCTION>>>
You have received all worker outputs. Produce a single, complete, runnable file:
- Include the shared interface
- Include all worker implementations (dedup overlapping helpers, keep one copy)
- Include your integration/glue code, updated to match actual worker signatures
- Add a dependency manifest if needed (package.json, etc.)
- Output only code, wrapped in a single ```<lang> ... ``` block
- If any worker's output contains prose, strip it — only include the code
<<<END_GATHER>>>
```

The boss sees the actual code, diffs it against what it expected (declared exports), and
produces a unified artifact.

### Context budget

The boss's seq already contains: BOSS_PROMPT + task + plan output + SELF piece. We need
room for: all worker outputs + gather prompt + gather output.

Budget calculation (non-unified KV, n_ctx / n_seq_max per lane):

```
n_ctx = 131072 (default with Q8)
n_seq_max = 5 (4 lanes + 1 base cache)
per-lane budget = ~26k tokens

Boss's pre-gather context ≈ plan (~2k) + SELF piece (~1k) = ~3k
Worker outputs ≈ M pieces × ~300 tok = ~2.4k (for M=8)
Gather prompt + output ≈ 500 + 2k = ~2.5k

Total boss context ≈ 8k — well within 26k budget.
```

For large tasks with many pieces, the boss context could approach the budget. Mitigation:
`max_new` for gather is separate from worker `max_new` (default 2048 for gather).

---

## Implementation Plan

### Step 1: Code extraction utility

Workers may emit prose. Add a utility to extract code from ```fenced``` blocks, with a
fallback to the full text if no fences are found.

```cpp
// Extract content from ```lang ... ``` blocks.
// If no fences found, return the full text (fallback).
static std::string extract_code_fence(const std::string &text);
```

This is used both for the worker outputs injected into the boss (so the boss sees clean
code) and for the final gather output (so `--out` gets only code).

### Step 2: Gather prompt builder

```cpp
// Build the gather injection text: worker outputs + gather instruction
static std::string build_gather_prompt(
    const WorkOrder &wo,
    const std::vector<std::string> &worker_outputs,  // by piece id
    const std::string &boss_self_output);
```

Format: each worker's output wrapped in `<<<WORKER_DONE>>>...<<<END_WORKER>>>` markers,
followed by the `<<<GATHER_INSTRUCTION>>>` block.

### Step 3: Gather decode

After the parallel loop completes (all lanes done), inject the gather prompt into the boss's
seq and run a single-stream decode:

```cpp
static std::string run_gather(const std::string &gather_prompt, int max_gather) {
    // Inject gather_prompt into boss seq at current position (delta prefill)
    // Decode until EOG or max_gather
    // Return the boss's gather output
}
```

This reuses the existing `inject` (delta prefill) and single-stream decode machinery.

### Step 4: `--out` flag

Add `--out <path>` to the CLI. Write the final gather output to the specified file.

If no gather phase runs (PA.0 mode, `--pool` mode), `--out` writes the concatenated worker
outputs with headers.

### Step 5: Wire into both pipelines

**Non-streaming path** (`--no-stream`):
After `run_pool` returns, `wo.pieces` and `outputs` are available. Build gather prompt,
run gather, write output.

**Streaming path** (`run_pipeline`):
After the pipeline loop completes, `queue` and `outv` are available. Build gather prompt,
run gather, write output. The boss's seq is still live in context — just inject at its
current position.

### Step 6: Strip boss SELF from worker pool

Currently, the boss's SELF piece runs as a worker lane during the parallel phase. Its output
should be captured separately (as `boss_self_output`) and included in the gather prompt, not
treated as a regular worker piece.

---

## Code Changes (pti_agents.cpp)

### New functions

```cpp
// Extract ```lang ... ``` fenced code from text; fallback to full text
static std::string extract_code_fence(const std::string &text);

// Build the gather injection text
static std::string build_gather_prompt(
    const WorkOrder &wo,
    const std::vector<std::pair<std::string, std::string>> &worker_results,  // id -> output
    const std::string &boss_self_output);

// Run the gather phase: inject prompt, decode boss, return output
static std::string run_gather_phase(int boss_seq, int max_gather);
```

### Modifications to existing code

**`run_pool`**: Return worker outputs keyed by piece id (already returns `out` vector;
add a parallel `ids` vector for piece id mapping).

**`run_pipeline`**: After the decode loop, call `run_gather_phase`. The boss's seq state is
still live — inject at current position.

**`main()` (non-streaming path)**: After `run_pool`, call gather, write `--out`.

**`main()` argument parser**: Add `--out <path>`.

### Gather prompt template

```cpp
static const char *GATHER_PROMPT =
    "You have received the outputs from all workers. Your job is to produce a SINGLE, "
    "COMPLETE, RUNNABLE artifact.\n\n"
    "For each worker below, you see their actual implementation. Use these implementations "
    "AS-IS — do not rewrite them. If a worker's output contains prose, extract only the code.\n\n"
    "Your output must:\n"
    "1. Include the shared interface (types, signatures)\n"
    "2. Include every worker's implementation (dedup overlapping helpers — keep one copy)\n"
    "3. Include your integration/glue code, UPDATED to match actual worker signatures\n"
    "4. Add a dependency manifest if needed\n"
    "5. Be wrapped in a single code fence: ```<lang> ... ```\n\n"
    "Output ONLY the code fence, nothing else.\n\n";
```

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Boss context overflow | Gather decode fails | Hard cap on gather input size; truncate worker outputs if needed |
| Boss produces poor merge | Artifact is broken | `--no-gather` fallback: skip gather, print pieces separately |
| Worker prose in output | Boss has to clean up | `extract_code_fence` pre-processes worker output before injection |
| Gather is slow | Adds serial latency | Separate `--gather-n` cap; keep gather prompt tight |
| Boss hallucinates symbols | Wrong glue | Boss prompt includes declared exports for drift detection |

---

## Acceptance Criteria

1. `bin/pti_agents -m <model> -p "task" --out result.js` produces a single file containing:
   - All worker implementations
   - Integration/glue code
   - No prose wrappers
2. The output file is syntactically valid (no obvious breakage)
3. Gather phase completes without context overflow on the default vector-math task (8 items)
4. `--no-gather` flag skips gather, prints pieces separately (fallback for debugging)
5. Both streaming and non-streaming paths support gather

---

## Testing Plan

### Unit tests (GPU-free, fast, `--parse-test` style)

All pure-text utilities are testable without loading a model. Add a `--gather-test` flag
(like the existing `--parse-test`) that exercises every text-processing function and exits.

| # | Function | Test case | Expected |
|---|---|---|---|
| U1 | `extract_code_fence` | Input: ````js\nfunction foo() {}\n```` | Returns `function foo() {}` |
| U2 | `extract_code_fence` | Input: ````javascript\n...code...\n```` | Strips `javascript` lang tag, returns inner code |
| U3 | `extract_code_fence` | Input: no fences, raw code | Returns full text unchanged (fallback) |
| U4 | `extract_code_fence` | Input: multiple fences | Returns content of **first** fence only |
| U5 | `extract_code_fence` | Input: fence with prose before/after | Returns fenced content only, strips surrounding prose |
| U6 | `extract_code_fence` | Input: empty string | Returns empty string |
| U7 | `extract_code_fence` | Input: unclosed fence | Returns full text unchanged (graceful degradation) |
| U8 | `build_gather_prompt` | WorkOrder with 3 pieces + boss SELF | Output contains all `<<<WORKER_DONE>>>` markers, `<<<GATHER_INSTRUCTION>>>`, and piece ids |
| U9 | `build_gather_prompt` | WorkOrder with 0 pieces (edge case) | Output contains `<<<GATHER_INSTRUCTION>>>` but no `<<<WORKER_DONE>>>` markers |
| U10 | `build_gather_prompt` | Worker output contains `<<<` markers | Worker output is escaped or placed so markers don't confuse the boss (verify no false marker matches) |

The existing `--parse-test` pattern is the model: `parse_self_test()` returns 0/2, runs
against `SAMPLE_ENVELOPE`. `--gather-test` will run a battery of `assert` calls with
stderr output for each pass/fail.

### Integration tests (require GPU, slower)

| # | Scenario | How | Expected |
|---|---|---|---|
| I1 | Gather on canned task | `bin/pti_agents -m <model> -p "write a Vec2 class with add, sub, mag methods" --out /tmp/vec2.js --no-think` | File exists, contains all 3 methods, no duplicate helpers, no prose |
| I2 | `--no-gather` produces separate pieces | Same as I1 with `--no-gather` | stdout has `───── item w1 ─────` headers, no `--out` file written |
| I3 | `--out` in non-streaming mode | `bin/pti_agents -m <model> -p "task" --out /tmp/out.js --no-stream --no-think` | File written, contains merged output |
| I4 | `--out` in streaming mode (default) | `bin/pti_agents -m <model> -p "task" --out /tmp/out.js --no-think` | File written, contains merged output |
| I5 | Small task (2 items) | Boss emits only 2 PIECEs | Gather handles it, no crash on small piece count |
| I6 | Worker emits prose | Worker preamble weakened to allow prose | `extract_code_fence` strips it; artifact contains only code |
| I7 | Boss SELF included in gather | Boss SELF ran as lane during parallel | Gather prompt includes boss SELF output separately from workers |

### Smoke test for CI (fastest possible GPU test)

A minimal integration that runs with `-n 16` (tiny output per worker) to verify the full
pipeline doesn't crash:

```bash
bin/pti_agents -m <model> -p "Write a js function double(x)" -n 16 --out /tmp/smoke.js --no-think --no-gather
# Verify: exits 0, /tmp/smoke.js not created (--no-gather)
bin/pti_agents -m <model> -p "Write a js function double(x)" -n 16 --out /tmp/smoke.js --no-think
# Verify: exits 0, /tmp/smoke.js exists, non-empty
```

---

## Out of Scope (future)

- AST-level merge (the design doc explicitly rejects this)
- Worker drift detection beyond the boss's natural language reconciliation
- Dependency manifest generation beyond what the boss can write
- PA.4 `<<<ASK>>>`/`<<<REPLY>>>` channel during gather
