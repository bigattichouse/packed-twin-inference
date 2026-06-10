#!/usr/bin/env bash
# patch.sh — apply this repo's llama.cpp modifications.
#
# Usage:
#   ./patch.sh                        # apply to ../llama.cpp (default)
#   ./patch.sh /path/to/llama.cpp     # apply to a specific checkout
#   ./patch.sh --build                # apply + rebuild (ROCm 7.2.1 / gfx906)
#   ./patch.sh /path/to/llama.cpp --build
#
# What it applies (see README.md in this directory):
#   0001-mmvq-i-outer-j-inner.patch — MMVQ loop-order fix (M5.1). OPTIONAL
#   performance patch: N=4 batched-decode overhead 1.95x -> 1.86x on gfx906.
#   Everything else PTI needs (MTP context type, pre-norm embeddings ext)
#   is already upstream as of the base commit below — no patch required.
#
# Idempotent: re-running detects already-applied patches and skips them.
# Patch won't apply (upstream drift)? Copy the full modified files instead —
# ../llama.cpp-files/ mirrors the llama.cpp directory structure:
#   cp -r llama.cpp-files/* /path/to/llama.cpp/

set -euo pipefail

BASE_COMMIT=ad2775726   # upstream ggerganov/llama.cpp master this repo was built against

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse args
LLAMA_DIR=""
DO_BUILD=0
for arg in "$@"; do
    case "$arg" in
        --build) DO_BUILD=1 ;;
        *)       LLAMA_DIR="$arg" ;;
    esac
done

# Default: look for llama.cpp one level up from packed-twin-inference
if [[ -z "$LLAMA_DIR" ]]; then
    LLAMA_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)/llama.cpp"
fi

echo "Target: $LLAMA_DIR"

# Sanity check
if [[ ! -f "$LLAMA_DIR/ggml/src/ggml-cuda/mmvq.cu" ]]; then
    echo "ERROR: $LLAMA_DIR does not look like a llama.cpp checkout."
    echo "       (ggml/src/ggml-cuda/mmvq.cu not found)"
    exit 1
fi

if ! git -C "$LLAMA_DIR" merge-base --is-ancestor "$BASE_COMMIT" HEAD 2>/dev/null; then
    echo "WARNING: checkout does not contain base commit $BASE_COMMIT."
    echo "         PTI needs the upstream MTP + pre-norm ext APIs (master >= $BASE_COMMIT)."
    echo "         The patch may still apply cleanly; continuing."
fi

for PATCH_FILE in "$SCRIPT_DIR"/*.patch; do
    NAME="$(basename "$PATCH_FILE")"
    if git -C "$LLAMA_DIR" apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
        echo "skip:  $NAME (already applied)"
    elif git -C "$LLAMA_DIR" apply --check "$PATCH_FILE" 2>/dev/null; then
        git -C "$LLAMA_DIR" apply "$PATCH_FILE"
        echo "apply: $NAME"
    else
        echo "FAIL:  $NAME does not apply cleanly (upstream drift?)." >&2
        echo "       Fallback — copy the mirrored full files:" >&2
        echo "         cp -r $SCRIPT_DIR/../llama.cpp-files/* $LLAMA_DIR/" >&2
        exit 1
    fi
done

if [[ "$DO_BUILD" -eq 1 ]]; then
    echo ""
    echo "Building llama.cpp (ROCm 7.2.1, gfx906)..."
    cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
        -DGGML_HIP=ON \
        -DCMAKE_C_COMPILER=/opt/rocm-7.2.1/lib/llvm/bin/clang \
        -DCMAKE_CXX_COMPILER=/opt/rocm-7.2.1/lib/llvm/bin/clang++ \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CCACHE=OFF \
        -DGGML_NATIVE=ON \
        -DGGML_OPENMP=ON \
        -DCMAKE_HIP_ARCHITECTURES=gfx906 \
        -DCMAKE_HIP_COMPILER_FORCED=ON
    cmake --build "$LLAMA_DIR/build" --config Release -j "$(nproc)"
    echo ""
    echo "Build done. Binaries: $LLAMA_DIR/build/bin/"
else
    echo ""
    echo "Done. Rebuild llama.cpp before building PTI (or re-run with --build)."
fi
