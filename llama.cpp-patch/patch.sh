#!/usr/bin/env bash
# patch.sh — apply the PTI interleaved activation patch to a llama.cpp checkout
#
# Usage:
#   ./patch.sh                        # apply to ../llama.cpp (default)
#   ./patch.sh /path/to/llama.cpp     # apply to a specific checkout
#   ./patch.sh --build                # apply + rebuild (ROCm 7.2.1 / gfx906)
#   ./patch.sh /path/to/llama.cpp --build
#
# The patch adds the GGML_PTI_INTERLEAVED=1 env var gate to mul_mat_vec_q.
# No existing behavior changes when the env var is unset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$SCRIPT_DIR/0001-pti-interleaved-activation.patch"

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

# Check patch hasn't been applied already
if grep -q "pti_interleave_q8_1_kernel" "$LLAMA_DIR/ggml/src/ggml-cuda/mmvq.cu"; then
    echo "Patch already applied — skipping."
else
    echo "Applying patch..."
    git -C "$LLAMA_DIR" apply "$PATCH_FILE"
    echo "Patch applied successfully."
fi

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
fi

echo ""
echo "Test with:"
echo "  GGML_PTI_INTERLEAVED=0 ./pti_4seq -m model.gguf ...   # baseline"
echo "  GGML_PTI_INTERLEAVED=1 ./pti_4seq -m model.gguf ...   # patched"
