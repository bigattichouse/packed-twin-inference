#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# seq_vs_packed_ab.sh — does INTERLEAVING actually buy throughput, or would
# looping ONE agent at a time be faster?  (the "crossover" A/B)
#
# Why this exists (2026-06-22): the measured agentic stages run BELOW a single
# stream (design 0.79×, implement 0.68× aggregate — PACKED_AGENTS_DESIGN §1.1),
# because the hybrid model's idle-seq SSM tax processes EVERY allocated lane's
# recurrent state each step, plus straggler/barrier idle. A sequential single-
# agent loop (-s 1) runs every item at the full baseline (~19.3 tok/s, no
# allocation tax / straggler / barrier) and IS 1× by definition. So packed beats
# sequential ONLY if its aggregate clears 1×. This script measures that crossover
# DIRECTLY: same task, same flags, two lane counts, compare WALL.
#
# Runs strictly SEQUENTIALLY (packed side fully exits before the sequential side
# starts) so the two never share the GPU — otherwise the wall comparison is junk.
#
# Reports per side: WALL, VERIFY pass-rate, DETAIL (files/tests/LOC/asserts), and
# the binary's own per-stage aggregate tok/s; then the crossover verdict.
#
# Usage:   ./seq_vs_packed_ab.sh
#   env overrides: MODEL=/path.gguf  LANES_PACKED=4  LANES_SEQ=1  NMAX=2000  TASK="..."
# ─────────────────────────────────────────────────────────────────────────────
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
MODEL="${MODEL:-$DIR/../gguf/Qwen3.6-27B-UD-Q6_K_XL.gguf}"
BIN="${BIN:-$DIR/bin/pti_agents}"
LANES_PACKED="${LANES_PACKED:-4}"
LANES_SEQ="${LANES_SEQ:-1}"
NMAX="${NMAX:-2000}"
PK="$DIR/validate/ab_packed"
SQ="$DIR/validate/ab_seq"

# 5 INDEPENDENT modules → items > packed lanes (refill + straggler tail), but no
# cross-module coupling, so the comparison isolates the decode/scheduling cost.
TASK="${TASK:-Build a small JavaScript utility library as a MULTI-FILE Node.js (CommonJS) project. Produce 5 INDEPENDENT modules, each in its own file under src/ with NO cross-module dependencies: stringUtils, arrayUtils, mathUtils, objectUtils, numberUtils. Each module exports 4-5 small PURE functions (module.exports = { ... }). Provide a unit-test file per module (NAME.test.js) using console.assert that exits non-zero on any failure. Every file complete and runnable.}"

metrics() {  # files / tests / LOC / assertions in $1
  local d="$1" files tests loc asserts
  files=$(find "$d" -name '*.js' ! -name '*.test.js' 2>/dev/null | wc -l)
  tests=$(find "$d" -name '*.test.js' 2>/dev/null | wc -l)
  loc=$(find "$d" -name '*.js' -exec cat {} + 2>/dev/null | wc -l)
  asserts=$(find "$d" -name '*.test.js' -exec grep -ho 'console\.assert' {} + 2>/dev/null | wc -l)
  echo "files=$files  tests=$tests  LOC=$loc  assertions=$asserts"
}
verify() {  # run every *.test.js in $1; echo "pass/total"
  local d="$1" sp=0 st=0 rel
  while IFS= read -r tf; do
    st=$((st+1)); rel="${tf#$d/}"
    (cd "$d" && timeout 30 node "$rel" >/dev/null 2>&1) && sp=$((sp+1)) || echo "      [FAIL] $rel" >&2
  done < <(find "$d" -name '*.test.js' 2>/dev/null | sort)
  echo "$sp/$st"
}

run_side() {  # $1=lanes  $2=workdir  $3=logfile  → echoes wall seconds
  local lanes="$1" wd="$2" log="$3" s e rc
  rm -rf "$wd"; mkdir -p "$wd"
  s=$(date +%s)
  timeout 5400 "$BIN" -m "$MODEL" -p "$TASK" --no-stream -s "$lanes" -n "$NMAX" \
    --tools --allow-run --work-dir "$wd" > "$log" 2>&1
  rc=$?; e=$(date +%s)
  echo "$((e-s)) $rc"
}

echo "######## seq_vs_packed_ab — model=$(basename "$MODEL")  NMAX=$NMAX ########"
[ -x "$BIN" ] || { echo "FATAL: binary not built at $BIN (run: make agents)"; exit 1; }
[ -f "$MODEL" ] || { echo "FATAL: model not found at $MODEL"; exit 1; }

echo "## PACKED  (-s $LANES_PACKED) ##"
read -r W_PK RC_PK < <(run_side "$LANES_PACKED" "$PK" "$DIR/validate/ab_packed.log")
echo "  exit=$RC_PK  WALL=${W_PK}s   VERIFY=$(verify "$PK")   DETAIL: $(metrics "$PK")"
grep -hiE 'tok/s|aggregate' "$DIR/validate/ab_packed.log" 2>/dev/null | sed 's/^/    · /' | tail -8

echo "## SEQUENTIAL  (-s $LANES_SEQ) ##"
read -r W_SQ RC_SQ < <(run_side "$LANES_SEQ" "$SQ" "$DIR/validate/ab_seq.log")
echo "  exit=$RC_SQ  WALL=${W_SQ}s   VERIFY=$(verify "$SQ")   DETAIL: $(metrics "$SQ")"
grep -hiE 'tok/s|aggregate' "$DIR/validate/ab_seq.log" 2>/dev/null | sed 's/^/    · /' | tail -8

echo "######## CROSSOVER VERDICT ########"
awk -v p="$W_PK" -v q="$W_SQ" 'BEGIN{
  if (p<=0||q<=0){ print "  (bad wall numbers — check logs)"; exit }
  printf "  packed=%ds  sequential=%ds\n", p, q;
  spd=q/p;   # packed speedup over sequential
  if (spd>1.0) printf "  → PACKED is %.2f× faster than the sequential loop (interleaving wins here)\n", spd;
  else         printf "  → SEQUENTIAL is %.2f× faster (packed runs BELOW 1× — interleaving LOSES here)\n", 1/spd;
}'
echo "  (logs: validate/ab_packed.log, validate/ab_seq.log; trees: $PK, $SQ)"
