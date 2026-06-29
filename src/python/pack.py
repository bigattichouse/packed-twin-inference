"""
Pack a model as SSQ twins (mode 16, same model in both streams).

Usage:
    python pack.py ../model/Qwen3-1.7B qwen17b_twin.ssq
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from ssq.pack import pack


def pack_twins(model_path: str, out_path: str, verbose: bool = True):
    """Pack a single model with itself as SSQ twin streams (mode 16)."""
    return pack(
        [model_path, model_path],
        out_path,
        mode=16,
        verbose=verbose,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pack model as SSQ twins")
    parser.add_argument("model", help="Path to model (GGUF or safetensors dir)")
    parser.add_argument("output", help="Output .ssq file")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    pack_twins(args.model, args.output, verbose=not args.quiet)
