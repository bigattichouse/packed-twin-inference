# pti_agents.cpp — Gaps for Generating Actual Code Output

**Date**: 2026-06-13
**Based on**: `pti_agents.cpp`, `PACKED_AGENTS.md`, `POSITIVE_RESULTS.md`, `spec/PACKED_AGENTS_DESIGN.md`

---

## What's working well

The code implements PA.0 through PA.2.1 comprehensively:

- **PA.0** — 4-stream packed decode vs sequential baseline with byte-identity gate
- **PA.1a** — Work-order envelope parser with export collision detection
- **PA.2** — Pool mechanism with refill-on-DONE and prefix cache (PA.2.1)
- **PA.3 (streaming pipeline)** — Boss plans while workers execute, live piece enqueuing

The plumbing (packed decode loop, seq management, batch construction, argmax with tie-breaking, Q8/f16 KV) is solid and mirrors the proven machinery from `pti_lookup.cpp`.

---

## Gaps for generating actual code output

### 1. No final artifact assembly (Gather phase is missing)

The design's Phase 4 is: *"workers' outputs are injected into the coordinator as messages; coordinator integrates / resolves conflicts / emits the final artifact."*

What the code actually does: prints each piece to stdout with a `───── item <id> ─────` header, then exits. There is **no step that merges the worker outputs into a single runnable file**. The boss SELF piece generates integration code *blind* (against declared interfaces, not actual worker bodies), so its output is a standalone glue module — not a merged artifact.

**Missing**: A post-pool pass that either:
- Concatenates all outputs into one file (shared interface + worker implementations + boss glue), or
- Injects worker outputs back into the boss for a final integration pass, or
- At minimum, provides a structured merge script/flag

### 2. `--out` flag not implemented

The PACKED_AGENTS.md interface spec says: *"exit artifact: gather output written to a file via --out"*. The flag does not exist in `main()`'s argument parser. All output goes to stdout only.

### 3. Boss SELF piece has no access to actual worker output

The boss's SELF lane sees only the *declared exports* of workers:

```cpp
for (auto &q : wo.pieces) {
    if (q.is_boss) continue;
    for (auto &e : q.exports) { user += ... + e; }
}
```

It does not receive the *actual generated code*. So the boss writes integration code against type signatures it declared, not against the workers' actual implementations. If a worker deviates from spec (wrong parameter name, extra dependency), the integration code won't match. This is acknowledged in the design ("workers are in ISOLATION"), but it means the output is not a cohesive, compilable unit without manual assembly.

### 4. No code fencing / language tagging on output

Worker outputs are raw model text (with `</think>` stripped). There's no enforcement that workers emit fenced code blocks, and no post-processing to extract code from prose. The worker preamble says *"Output only code"* but there's no guardrail. If the model emits "Here is your function:" before the code, that prose becomes part of the output.

**Missing**: A simple post-processing step to extract ```lang ... ``` blocks, or a stronger worker prompt that uses a parseable output envelope (similar to the work-order markers).

### 5. No re-plan on parse failure

If the boss emits a malformed envelope, the code exits with code 2. The design doc acknowledges this: *"a malformed envelope is the most likely v1 failure and needs a fallback (re-ask the boss with the parse error injected)."*

**Missing**: A retry loop that feeds the parse error back into the boss with a "fix your output" instruction.

### 6. No worker termination safeguard

Workers run until EOG or `max_new`. There's no checkpoint round (PA.4) to kill a runaway worker, no quality check, no `<<<ASK>>>` channel. A worker that starts emitting garbage will burn its entire token budget. The design acknowledges this as a PA.4 item, but for "generating actual code," even a basic safeguard (e.g., checking for EOG + minimum output length) would help.

### 7. Per-stream temperature sampling not implemented

The design calls out per-stream seeds for reproducible runs at any temperature. The code uses `argmax_f` (greedy, temperature=0) everywhere. If you want diverse or higher-temperature output (useful for creative code), the sampling infrastructure isn't there.

This is a v2 feature (stacking MTP + sampling), but it's worth noting if you want anything other than greedy decode.

### 8. Context budget management is fragile

With non-unified KV and `n_seq_max = n_streams + 1`, each lane gets `n_ctx / (n_streams + 1)` tokens. At default settings (128k ctx, 5 seqs), that's ~25k per lane. The boss's planning phase can consume a lot of this budget if `--no-think` isn't set. The code has a `plan_cap` calculation but no hard guard — if the boss exceeds its budget, `llama_decode` will fail silently.

### 9. Output collection in `run_pipeline` is incomplete

The `run_pipeline` function collects `outv` for worker items, but the boss's generated text is only in `lanes[BOSS].text` (which contains the raw planning output, not a separate SELF piece output). After the pipeline completes, the boss's contribution is printed inline with `outv`, but there's no distinction between the plan text and any integration code the boss might produce as a SELF worker.

---

## Summary of what to add for "actual code output"

| Priority | What | Why |
|---|---|---|
| **Critical** | Final artifact assembly (gather) | Without this, you get N disconnected snippets, not a usable program |
| **Critical** | `--out` flag | Design promises it; needed for editor integration |
| **High** | Code extraction from worker output | Workers may emit prose; need fence stripping |
| **High** | Re-plan on parse failure | Boss occasionally emits bad envelopes; retry is essential for reliability |
| **Medium** | Boss sees worker output | Integration code won't match if workers deviate from spec |
| **Medium** | Worker termination guard | Runaway workers waste tokens; basic length check would help |
| **Low** | Per-stream temperature sampling | Greedy works for code; temperature is a v2 nicety |
| **Low** | `<<<ASK>>>` / `<<<REPLY>>>` channel | PA.4 feature; valuable for ambiguous specs |

The single biggest gap is **the missing gather phase**. The code produces individual pieces but never assembles them into the "final artifact" that PACKED_AGENTS.md promises. Everything else is incremental.
