#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# scale_validate.sh — packed vs single on packed's SWEET SPOT: many independent,
# balanced, loosely-coupled work items (not a 5-file greenfield app).
#
# Why this exists (2026-06-16): flappy_validate.sh tests greenfield generation of
# a small, dependency-COUPLED app — nearly the worst case for batch parallelism
# (few items, unbalanced, reconcile barrier, high coordination ratio). Packed's
# economics win when there are MANY INDEPENDENT items so lanes stay full (items >>
# lanes → refill hides stragglers) and coordination amortizes to ~nothing. This
# harness builds a 12-module utility library with NO cross-module deps — 12 even,
# independent pieces — to see whether packed finally beats single on WALL and holds
# its ~1.9× aggregate. Single must emit all 12 serially; packed does ~4 at a time.
#
# Apples-to-apples: same task, same Qwen coding sampling on the thinking roles
# (packed boss + single both think @ 0.6/0.95/k20); packed workers instruct. Both
# 128k. No MTP (net-loss on packed — spec/PA3). Same verifier (node *.test.js).
#
# Reports, per side: VERIFY pass-rate + DETAIL metrics (files, tests, LOC,
# assertions) + WALL. Output (gitignored) → validate/scale_packed, validate/scale_single.
#
# Usage:   ./scale_validate.sh          (MODEL=/path/to.gguf to override)
# ─────────────────────────────────────────────────────────────────────────────
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
MODEL="${MODEL:-$DIR/../gguf/Qwen3.6-27B-UD-Q6_K_XL.gguf}"
LLAMA_CLI="${LLAMA_CLI:-$DIR/../llama.cpp/build/bin/llama-cli}"
PK="$DIR/validate/scale_packed"
SG="$DIR/validate/scale_single"

TASK="Build a JavaScript utility library as a MULTI-FILE Node.js (CommonJS) project. Produce 12 INDEPENDENT modules, each in its own file under src/ with NO cross-module dependencies: stringUtils, arrayUtils, mathUtils, objectUtils, dateUtils, numberUtils, validationUtils, urlUtils, colorUtils, randomUtils, sortUtils, statUtils. Each module exports 4-6 small PURE functions (module.exports = { ... }). Provide a unit-test file per module (NAME.test.js) that requires the module, checks its functions with console.assert, prints a line per check, and calls process.exit(1) on ANY failure so it runs with 'node NAME.test.js'. Every file complete and runnable."

# ── shared helpers ───────────────────────────────────────────────────────────
metrics() {  # detail: files / tests / LOC / assertions in $1
  local d="$1"
  local files tests loc asserts
  files=$(find "$d" -name '*.js' ! -name '*.test.js' 2>/dev/null | wc -l)
  tests=$(find "$d" -name '*.test.js' 2>/dev/null | wc -l)
  loc=$(find "$d" -name '*.js' -exec cat {} + 2>/dev/null | wc -l)
  asserts=$(find "$d" -name '*.test.js' -exec grep -ho 'console\.assert' {} + 2>/dev/null | wc -l)
  echo "files=$files  tests=$tests  LOC=$loc  assertions=$asserts"
}
verify() {  # run every *.test.js in $1; print pass/total
  local d="$1" sp=0 st=0 rel
  while IFS= read -r tf; do
    st=$((st+1)); rel="${tf#$d/}"
    if (cd "$d" && timeout 30 node "$rel" >/tmp/_scv.out 2>&1); then sp=$((sp+1));
    else echo "    [FAIL] $rel"; sed -n '1,3p' /tmp/_scv.out | sed 's/^/        /'; fi
  done < <(find "$d" -name '*.test.js' | sort)
  echo "  VERIFY: $sp/$st passed"
}

rm -rf "$PK" "$SG"; mkdir -p "$PK" "$SG"

# ── PACKED ───────────────────────────────────────────────────────────────────
echo "######## PACKED (12 independent modules; workers write files+tests; harness verifies) ########"
t0=$(date +%s)
"$DIR/bin/pti_agents" -m "$MODEL" -p "$TASK" --no-stream -s 4 -n 2600 \
  --tools --allow-run --work-dir "$PK" > "$DIR/validate/scale_packed.log" 2>&1
echo "packed exit=$?  WALL=$(( $(date +%s) - t0 ))s"
grep -E 'PARSE OK|DESIGN —|IMPLEMENT —|TEST-GEN|aggregate|VERIFY RESULT' "$DIR/validate/scale_packed.log" | tail -16
echo "  DETAIL: $(metrics "$PK")"
verify "$PK"

# ── SINGLE ───────────────────────────────────────────────────────────────────
echo "######## SINGLE (one model emits all 12 modules + tests in one pass) ########"
SINGLE_TASK="$TASK Output EACH file in its own fenced code block, immediately preceded by a line: // FILE: <relative/path>"
t1=$(date +%s)
"$LLAMA_CLI" -m "$MODEL" --jinja -rea on -st \
  -p "$SINGLE_TASK" --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 -s 42 -n 24000 -ngl 99 -c 131072 \
  > "$DIR/validate/scale_single.out" 2> "$DIR/validate/scale_single.log"
echo "single exit=$?  WALL=$(( $(date +%s) - t1 ))s"
python3 - "$SG" < "$DIR/validate/scale_single.out" <<'PY'
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
print(f"  split {n} files into {out_dir}")
PY
echo "  DETAIL: $(metrics "$SG")"
verify "$SG"
echo "######## DONE — compare WALL, VERIFY pass-rate, and DETAIL across the two ########"
