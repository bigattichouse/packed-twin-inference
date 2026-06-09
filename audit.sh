#!/usr/bin/env bash
# audit.sh — end-to-end correctness and performance audit for PTI pipeline
#
# Tests each step: build → baseline coherence → PTI token match →
# accept rates → GEMV fusion scaling.
#
# Runtime: ~8–10 min (multiple model loads on MI50).
#
# Usage: ./audit.sh [-m model] [-n tokens] [--no-bench] [--quick]
#   -m   path to model gguf     (default: ../gguf/Qwen3.6-27B-UD-Q6_K_XL.gguf)
#   -n   tokens to generate     (default: 50)
#   --no-bench  skip GEMV scaling bench (saves ~3 min)
#   --quick     -n 20, -w 2 -s 5 for bench (faster, less accurate)

set -uo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
MODEL="../gguf/Qwen3.6-27B-UD-Q6_K_XL.gguf"
N_TOK=50
DO_BENCH=1
BENCH_WARMUP=3
BENCH_STEPS=10
BENCH_CTX=128

while [[ $# -gt 0 ]]; do
    case $1 in
        -m)         MODEL="$2";      shift 2 ;;
        -n)         N_TOK="$2";      shift 2 ;;
        --no-bench) DO_BENCH=0;      shift   ;;
        --quick)    N_TOK=20; BENCH_WARMUP=2; BENCH_STEPS=5; shift ;;
        *)          echo "Unknown: $1"; exit 1 ;;
    esac
done

PROMPT=$'<|im_start|>user\nWrite a Python fibonacci function<|im_end|>\n<|im_start|>assistant\n'

# ── state ────────────────────────────────────────────────────────────────────
PASS=0
FAIL=0
FAILURES=()

# ── colour ────────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    GRN='\033[0;32m'; RED='\033[0;31m'; CYN='\033[0;36m'; NC='\033[0m'
else
    GRN=''; RED=''; CYN=''; NC=''
fi

hdr() { printf "\n${CYN}━━ %s${NC}\n" "$1"; }
ok()  { printf "  ${GRN}PASS${NC}  %s\n" "$1";       PASS=$((PASS+1)); }
bad() { printf "  ${RED}FAIL${NC}  %s\n" "$1";       FAIL=$((FAIL+1)); FAILURES+=("$1"); }
inf() { printf "        %s\n"            "$1"; }
sep() { printf "        %-50s  " "$1"; }  # for inline results

float_gt() { awk "BEGIN{exit !($1 > $2)}"; }  # float_gt 3.5 2.0
float_lt() { awk "BEGIN{exit !($1 < $2)}"; }

# ── 1. Build ──────────────────────────────────────────────────────────────────
hdr "1. Build"

build_targets=(debug 4seq)
[[ $DO_BENCH -eq 1 ]] && build_targets+=(bench)

for t in "${build_targets[@]}"; do
    if make -s "$t" 2>/dev/null; then
        ok "make $t"
    else
        bad "make $t (check compiler/llama.cpp paths)"
    fi
done

# Verify binaries exist and are executable
for bin in bin/pti_debug bin/pti_4seq; do
    [[ -x "$bin" ]] && ok "$bin exists" || bad "$bin missing"
done
[[ $DO_BENCH -eq 1 ]] && { [[ -x bin/pti_gemv_bench ]] && ok "bin/pti_gemv_bench exists" \
                                                         || bad "bin/pti_gemv_bench missing"; }

# ── 2. Baseline single-sequence ───────────────────────────────────────────────
hdr "2. Baseline single-sequence (pti_debug)"

# pti_debug -n N generates N+1 tokens (1 from prefill + N loop iterations).
# pti_4seq  -n N generates N   tokens (1 tok_gen0 + N-1 loop iterations).
# Give debug one fewer so both produce N_TOK total output tokens.
BL_N=$((N_TOK - 1))
PTI_N=$N_TOK

bl_out=$(mktemp)
bl_err=$(mktemp)

if bin/pti_debug -m "$MODEL" -p "$PROMPT" -n $BL_N >"$bl_out" 2>"$bl_err"; then
    ok "pti_debug exited cleanly"
else
    bad "pti_debug non-zero exit"
fi

# Output length
bl_len=$(wc -c < "$bl_out")
if [[ $bl_len -gt 10 ]]; then
    ok "output non-empty ($bl_len bytes)"
else
    bad "output too short ($bl_len bytes)"
fi

# Prefill completed
prefill_ms=$(grep -oP 'Prefill done \(\K[0-9.]+' "$bl_err" 2>/dev/null || echo 0)
if float_gt "$prefill_ms" 0; then
    ok "prefill completed"
    inf "prefill: ${prefill_ms} ms"
else
    bad "prefill timing not found in stderr"
fi

# Token count
bl_tokens=$(grep -oP '(\d+) tokens' "$bl_err" 2>/dev/null | grep -oP '\d+' | tail -1 || echo 0)
if [[ $bl_tokens -gt 0 ]]; then
    ok "generated $bl_tokens tokens"
else
    bad "could not parse token count"
fi

# tok/s sanity check (>5 tok/s = model is on GPU)
bl_rate=$(grep -oP '[\d.]+ tok/s' "$bl_err" 2>/dev/null | grep -oP '^[\d.]+' | tail -1 || echo 0)
if float_gt "$bl_rate" 5; then
    ok "tok/s in range"
    inf "baseline: ${bl_rate} tok/s  (${prefill_ms} ms prefill)"
else
    bad "tok/s too low — model may be on CPU (got ${bl_rate})"
fi

# Garbling heuristic: unique 4-grams / total 4-grams > 0.4
bl_text=$(cat "$bl_out")
total4=$(echo "$bl_text" | grep -oP '(?<=.)...' | wc -l || echo 1)
uniq4=$(echo "$bl_text" | grep -oP '(?<=.)...' | sort -u | wc -l || echo 0)
ratio=$(awk "BEGIN{printf \"%.2f\", $uniq4/$total4}")
if float_gt "$ratio" 0.4; then
    ok "output looks coherent (unique-4gram ratio: $ratio)"
else
    bad "output may be garbled (unique-4gram ratio: $ratio)"
fi

inf "sample: $(head -c 100 "$bl_out" | tr '\n' ' ')"

# ── 3. PTI 4-seq correctness ──────────────────────────────────────────────────
hdr "3. PTI 4-seq correctness (pti_4seq)"

pti_out=$(mktemp)
pti_err=$(mktemp)

if bin/pti_4seq -m "$MODEL" -p "$PROMPT" -n $PTI_N >"$pti_out" 2>"$pti_err"; then
    ok "pti_4seq exited cleanly"
else
    bad "pti_4seq non-zero exit"
fi

# Token-level output match
if diff -q "$bl_out" "$pti_out" >/dev/null 2>&1; then
    ok "PTI output identical to baseline (byte-for-byte)"
else
    bad "PTI output differs from baseline"
    inf "--- baseline"
    inf "+++ pti_4seq"
    diff "$bl_out" "$pti_out" | head -8 | while IFS= read -r line; do inf "$line"; done
fi

# Accept rates at greedy (all should be 4-accept, zero rejects)
acc4=$(grep -oP '4-accept:\s+\K\d+' "$pti_err" 2>/dev/null || echo -1)
acc3=$(grep -oP '3-accept:\s+\K\d+' "$pti_err" 2>/dev/null || echo -1)
acc2=$(grep -oP '2-accept:\s+\K\d+' "$pti_err" 2>/dev/null || echo -1)
rej=$(grep  -oP 'Rejects:\s+\K\d+'  "$pti_err" 2>/dev/null || echo -1)
pti_n=$(grep -oP 'Tokens generated:\s+\K\d+' "$pti_err" 2>/dev/null || echo 0)

inf "4-acc=$acc4  3-acc=$acc3  2-acc=$acc2  rej=$rej  tokens=$pti_n"

# At temp=0 (greedy), all steps should be 4-accept
if [[ $acc4 -ge 0 && $rej -eq 0 ]]; then
    ok "zero rejects at greedy"
else
    bad "unexpected rejects: $rej"
fi

if [[ $acc3 -eq 0 && $acc2 -eq 0 ]]; then
    ok "no partial accepts (all 4-accept at greedy)"
else
    bad "partial accepts at greedy: 3-acc=$acc3 2-acc=$acc2"
fi

# PTI token count should equal N_TOK (pti_4seq counts tok_gen0 + loop)
if [[ $pti_n -eq $N_TOK ]]; then
    ok "PTI token count correct ($pti_n)"
else
    bad "PTI token count wrong: expected $N_TOK got $pti_n"
fi

# PTI timing
pti_ss=$(grep -oP 'Steady-state:\s+\K[\d.]+' "$pti_err" 2>/dev/null || echo 0)
pti_am=$(grep -oP 'Amortized:\s+\K[\d.]+' "$pti_err" 2>/dev/null || echo 0)
inf "PTI reported: ${pti_ss} tok/s (steady)  ${pti_am} tok/s (amortized)"
inf "PTI step cost vs baseline: $(awk "BEGIN{printf \"%.2f\", $bl_rate / ($pti_ss+0.001)}")× slower per output token"

# ── 4. GEMV fusion scaling ────────────────────────────────────────────────────
if [[ $DO_BENCH -eq 1 ]]; then
    hdr "4. GEMV fusion scaling (pti_gemv_bench)"

    bench_err=$(mktemp)
    bin/pti_gemv_bench -m "$MODEL" \
        -w $BENCH_WARMUP -s $BENCH_STEPS -ctx $BENCH_CTX \
        >/dev/null 2>"$bench_err"

    # Parse table
    ms1=$(awk '/^\s+1\s/{print $2; exit}' "$bench_err" 2>/dev/null || echo 0)
    ms2=$(awk '/^\s+2\s/{print $2; exit}' "$bench_err" 2>/dev/null || echo 0)
    ms4=$(awk '/^\s+4\s/{print $2; exit}' "$bench_err" 2>/dev/null || echo 0)
    s4=$(grep -oP 'N=4 scaling: \K[\d.]+' "$bench_err" 2>/dev/null || echo 0)

    inf "N=1: ${ms1} ms   N=2: ${ms2} ms   N=4: ${ms4} ms   scaling: ${s4}×"

    # Partial fusion: 1.0 < scaling < 4.0
    if float_gt "$s4" 1.0 && float_lt "$s4" 4.0; then
        ok "GEMV scaling is partial fusion (${s4}×)"
    else
        bad "GEMV scaling out of expected range: ${s4}×"
    fi

    # Theoretical ceiling if fully fused
    ideal=$(awk "BEGIN{printf \"%.1f\", $bl_rate * 4 * ($ms1/$ms4)}")
    inf "Theoretical PTI at full fusion: ~${ideal} tok/s"
    inf "M5 kernel target: close the $(awk "BEGIN{printf \"%.2f\", $s4}")× gap to 1×"

    if float_lt "$s4" 1.5; then
        inf "→ Near-full fusion. PTI benefits already realised by ggml."
    elif float_lt "$s4" 2.5; then
        inf "→ Partial fusion. M5 kernel would close the remaining gap."
    else
        inf "→ Minimal sharing. M5 kernel is the critical bottleneck."
    fi

    rm -f "$bench_err"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
hdr "Summary"
printf "  ${GRN}Passed: %d${NC}   ${RED}Failed: %d${NC}\n" "$PASS" "$FAIL"

if [[ ${#FAILURES[@]} -gt 0 ]]; then
    printf "\n  Failed:\n"
    for f in "${FAILURES[@]}"; do printf "    • %s\n" "$f"; done
fi
echo

# ── Cleanup ───────────────────────────────────────────────────────────────────
rm -f "$bl_out" "$bl_err" "$pti_out" "$pti_err"

exit $FAIL
