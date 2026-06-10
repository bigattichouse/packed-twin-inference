#!/usr/bin/env bash
# pti-cli.sh — terminal demo with raw streaming output (made for screen recording).
#
#   ./pti-cli.sh base    "prompt" [tokens]     # plain llama.cpp speed
#   ./pti-cli.sh mtp     "prompt" [tokens]     # + MTP head drafting
#   ./pti-cli.sh pti     "prompt" [tokens]     # + lookup drafting (full stack)
#   ./pti-cli.sh compare "prompt" [tokens]     # all three, back to back
#
# Tokens stream to the screen as they are emitted; the stats block at the end
# shows tok/s and accept rates. Output text is byte-identical across modes.
#
# Good demo prompt for the speedup (paste-and-edit triggers the lookup path):
#   ./pti-cli.sh compare "$(cat demo_edit_prompt.txt)" 300

MODE=${1:-pti}
PROMPT=${2:-"Write a Python function that parses a CSV file and prints each row."}
NTOK=${3:-300}
DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL="${MODEL:-$DIR/../gguf/Qwen3.6-27B-UD-Q6_K_XL.gguf}"
BIN="$DIR/bin/pti_lookup"

run_one() {
    local label="$1"; shift
    echo
    echo "════════════════════════════════════════════════════════════"
    echo "  $label"
    echo "════════════════════════════════════════════════════════════"
    "$BIN" -m "$MODEL" -p "$PROMPT" -n "$NTOK" "$@"
    echo
}

case "$MODE" in
  base)    run_one "BASELINE — plain decode"            --baseline ;;
  mtp)     run_one "MTP — head drafting only"           --mtp --no-ngram ;;
  pti)     run_one "PTI — lookup + MTP (full stack)"    --mtp ;;
  compare)
           run_one "1/3  BASELINE — plain decode"       --baseline
           run_one "2/3  MTP — head drafting only"      --mtp --no-ngram
           run_one "3/3  PTI — lookup + MTP"            --mtp
           ;;
  *) echo "usage: $0 [base|mtp|pti|compare] \"prompt\" [tokens]"; exit 1 ;;
esac
