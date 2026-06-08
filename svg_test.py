"""
SVG generation test: baseline vs PTI twin.

Both models receive the same prompt asking for SVG rainbow code.
Greedy decoding → PTI output must be identical to baseline.
Saves output SVG files for visual inspection.
"""

import argparse
import sys
import torch
from pathlib import Path
from infer import load_model, generate_baseline, generate_twin

PROMPT = (
    "Write a complete SVG image of a colorful rainbow with a blue sky background. "
    "Include all six rainbow bands in their correct colors. "
    "Output only the SVG code, starting with <svg and ending with </svg>. "
    "No explanation, no markdown fences."
)


def extract_svg(text: str) -> str:
    """Pull the SVG element out of the model output."""
    start = text.find("<svg")
    end   = text.rfind("</svg>")
    if start != -1 and end != -1:
        return text[start:end + 6]
    return text.strip()


def run(model_path: str, max_tokens: int, device: str):
    print(f"\nLoading {model_path} ...")
    tokenizer, model = load_model(model_path, device)
    print(f"Device: {next(model.parameters()).device}")
    print(f"Dtype:  {next(model.parameters()).dtype}\n")

    print("─── Prompt ──────────────────────────────────────────────")
    print(PROMPT)
    print()

    # ── Baseline ──────────────────────────────────────────────────
    print("Running baseline ...", flush=True)
    text_b, n_b, t_b = generate_baseline(model, tokenizer, PROMPT, max_tokens)
    svg_b = extract_svg(text_b)
    tps_b = n_b / t_b
    print(f"  {n_b} tokens  {t_b:.1f}s  {tps_b:.1f} tok/s")

    # ── Twin ──────────────────────────────────────────────────────
    print("Running PTI twin ...", flush=True)
    text_t, n_t, t_t, acc = generate_twin(model, tokenizer, PROMPT, max_tokens, verbose=False)
    svg_t = extract_svg(text_t)
    tps_t = n_t / t_t
    print(f"  {n_t} tokens  {t_t:.1f}s  {tps_t:.1f} tok/s  accept={acc:.1%}")

    # ── Compare ───────────────────────────────────────────────────
    print()
    match = text_b.strip() == text_t.strip()
    print(f"Output identical: {'YES ✓' if match else 'NO ✗  (see diff below)'}")
    if not match:
        # Show first divergence
        for i, (a, b) in enumerate(zip(text_b, text_t)):
            if a != b:
                print(f"  First diff at char {i}:")
                print(f"    baseline: ...{repr(text_b[max(0,i-20):i+20])}...")
                print(f"    twin:     ...{repr(text_t[max(0,i-20):i+20])}...")
                break

    print(f"Acceptance rate:  {acc:.1%}")
    print(f"Theoretical gain: {1+acc:.2f}×  on SSQ HIP hardware")
    print()

    # ── Save SVGs ─────────────────────────────────────────────────
    out_dir = Path(__file__).parent
    base_svg_path = out_dir / "rainbow_baseline.svg"
    twin_svg_path = out_dir / "rainbow_twin.svg"

    with open(base_svg_path, "w") as f:
        f.write(svg_b if svg_b.startswith("<svg") else
                f'<!-- raw output -->\n<svg xmlns="http://www.w3.org/2000/svg" width="400" height="200">'
                f'<text x="10" y="20">{svg_b[:200]}</text></svg>')

    with open(twin_svg_path, "w") as f:
        f.write(svg_t if svg_t.startswith("<svg") else
                f'<!-- raw output -->\n<svg xmlns="http://www.w3.org/2000/svg" width="400" height="200">'
                f'<text x="10" y="20">{svg_t[:200]}</text></svg>')

    print(f"Saved: {base_svg_path}")
    print(f"Saved: {twin_svg_path}")

    if match:
        print("\nFiles are byte-identical — PTI produced exactly the same SVG as baseline.")
    else:
        print("\nFiles differ — check acceptance rate and verbose mode for details.")

    print()
    print("─── Baseline SVG (first 400 chars) ──────────────────────")
    print(svg_b[:400])
    if match:
        print("\n(Twin SVG is identical.)")
    else:
        print("\n─── Twin SVG (first 400 chars) ──────────────────────────")
        print(svg_t[:400])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="../model/Qwen3-1.7B")
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run(args.model, args.tokens, args.device)


if __name__ == "__main__":
    main()
