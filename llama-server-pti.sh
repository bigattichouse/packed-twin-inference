#!/usr/bin/env bash
# llama-server-pti.sh — OpenAI-compatible server for the speedup experiment.
#
#   ./llama-server-pti.sh            # pti  = lookup + MTP   (~2x on edits)
#   ./llama-server-pti.sh mtp        # MTP drafting only     (the "fair" baseline)
#   ./llama-server-pti.sh base       # plain decode          (= stock llama-server)
#   ./llama-server-pti.sh pti 8081   # custom port
#
# Point your editor at http://localhost:8080/v1 (same as llama-server).
# Speculation works at ANY temperature (sampled verification): coding temps
# (0.2-0.4) keep ~the full speedup; it degrades gracefully toward parity at
# chat temps. A fixed request "seed" makes runs reproducible.
#
# You can also switch modes per request without restarting, by adding
# "pti_mode": "base" | "mtp" | "pti" to the request body.
#
# NOTE: uses the UD-Q6_K_XL model — it carries the MTP (nextn) head that the
# mtp/pti modes need. Q5_K_M does not have one.

MODE=${1:-pti}
PORT=${2:-8080}
DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL="${MODEL:-$DIR/../gguf/Qwen3.6-27B-UD-Q6_K_XL.gguf}"
CTX="${CTX:-16384}"

exec "$DIR/bin/pti_server" \
  -m "$MODEL" \
  -ngl 99 \
  -c "$CTX" \
  -p "$PORT" \
  --mode "$MODE" \
  --temp 0.0
