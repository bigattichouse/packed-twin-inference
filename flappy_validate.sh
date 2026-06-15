#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# flappy_validate.sh — packed-agents vs single-model head-to-head (multi-file).
#
# Methodology (apples-to-apples):
#   - Same task: a multi-file Flappy Bird browser game (canvas + vanilla JS), one
#     module per concern, an index.html, AND a unit-test file per module.
#   - Same settings on the two THINKING roles: the packed BOSS and the SINGLE model
#     both run thinking @ Qwen3.6 coding sampling (temp 0.6 / top_p 0.95 / top_k 20).
#     Packed WORKERS run instruct (no-think, 0.7/0.80/presence 1.5) — they implement
#     a fully-specified piece, where reasoning adds latency for little gain.
#   - Both at 128k context. No MTP (it's a net loss on this hybrid card; see
#     spec/PA3_MTP_DESIGN.md). Greedy is intentionally avoided (Qwen guidance).
#
# What each side produces, and how we score it:
#   - PACKED: pti_agents --no-stream --tools --allow-run. The boss plans a dir
#     structure + per-module tests; workers create_file their module + <name>.test.js
#     to validate/packed/; the harness "test verifier" (finalize_verify) runs every
#     *.test.js. No merge-gather (it stripped tests / drifted code — PA.4 §3.2).
#   - SINGLE: one llama-cli pass emits the whole project (files tagged "// FILE:"),
#     which we split into validate/single/ and run through the SAME verifier.
#
# Output (gitignored) lands in validate/packed/ and validate/single/ for later
# examination; both dirs are wiped at the start of each run.
#
# Usage:   ./flappy_validate.sh            (uses the default model below)
#          MODEL=/path/to/model.gguf ./flappy_validate.sh
# ─────────────────────────────────────────────────────────────────────────────
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
MODEL="${MODEL:-$DIR/../gguf/Qwen3.6-27B-UD-Q6_K_XL.gguf}"
LLAMA_CLI="${LLAMA_CLI:-$DIR/../llama.cpp/build/bin/llama-cli}"
PK="$DIR/validate/packed"
SG="$DIR/validate/single"
TASK="Build a playable Flappy Bird browser game as a MULTI-FILE project (HTML5 canvas, vanilla JavaScript). Use a clean directory structure. Produce a module per concern (bird, pipes, renderer, game engine/loop), an index.html that wires them with <script> tags, AND a unit-test file per module (NAME.test.js) using console.assert that calls process.exit(1) on any failure so it runs with 'node NAME.test.js'. Every file complete and runnable."

rm -rf "$PK" "$SG"; mkdir -p "$PK" "$SG"   # fresh output each head-to-head

echo "######## PACKED (workers write files+tests to validate/packed; harness verifier runs them) ########"
t0=$(date +%s)
"$DIR/bin/pti_agents" -m "$MODEL" -p "$TASK" --no-stream -s 4 -n 2600 \
  --tools --allow-run --work-dir "$PK" > "$DIR/validate/packed.log" 2>&1
echo "packed exit=$?  WALL=$(( $(date +%s) - t0 ))s"
grep -E 'boss plan done|PARSE OK|item.*done|aggregate|STORE|VERIFY|PASS|FAIL' "$DIR/validate/packed.log" | tail -24
echo "--- validate/packed ---"; ls -R "$PK" 2>/dev/null

echo "######## SINGLE (one model emits the whole multi-file project + tests) ########"
SINGLE_TASK="$TASK Output EACH file in its own fenced code block, immediately preceded by a line: // FILE: <relative/path>"
t1=$(date +%s)
"$LLAMA_CLI" -m "$MODEL" --jinja -rea on -st \
  -p "$SINGLE_TASK" --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 -s 42 -n 14000 -ngl 99 -c 131072 \
  > "$DIR/validate/single.out" 2> "$DIR/validate/single.log"
echo "single exit=$?  WALL=$(( $(date +%s) - t1 ))s"
# split the single output into validate/single/ by the // FILE: markers
python3 - "$SG" < "$DIR/validate/single.out" <<'PY'
import sys, re, os
out_dir = sys.argv[1]; text = sys.stdin.read()
pat = re.compile(r'//\s*FILE:\s*(\S+)\s*\n```[^\n]*\n(.*?)```', re.S)
n = 0
for m in pat.finditer(text):
    path = m.group(1).strip().lstrip('/')
    if '..' in path: continue
    fp = os.path.join(out_dir, path)
    os.makedirs(os.path.dirname(fp) or out_dir, exist_ok=True)
    open(fp, 'w').write(m.group(2)); n += 1
print(f"split {n} files into {out_dir}")
PY
echo "--- validate/single ---"; ls -R "$SG" 2>/dev/null
echo "--- single verifier: run every *.test.js (same as packed) ---"
sp=0; st=0
while IFS= read -r tf; do
  st=$((st+1)); rel="${tf#$SG/}"
  if (cd "$SG" && timeout 30 node "$rel" >/tmp/_sv.out 2>&1); then echo "  [PASS] $rel"; sp=$((sp+1));
  else echo "  [FAIL] $rel"; sed -n '1,4p' /tmp/_sv.out | sed 's/^/      /'; fi
done < <(find "$SG" -name '*.test.js' | sort)
echo "== SINGLE VERIFY: $sp/$st passed =="
echo "######## DONE ########"
