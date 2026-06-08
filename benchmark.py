"""
Benchmark Packed Twin Inference (PTI) vs baseline greedy on Qwen3-1.7B.

Usage:
    python benchmark.py
    python benchmark.py --model ../model/Qwen3-1.7B --tokens 100 --verbose
"""

import argparse
import torch
from infer import load_model, generate_baseline, generate_twin

PROMPTS = [
    "The key difference between a transformer and an RNN is",
    "To make a great cup of coffee, you need to",
    "The French Revolution began in 1789 because",
    "In Python, a generator function differs from a regular function in that",
    "The speed of light is approximately 299,792 km/s, which means",
]


def run(model_path, max_tokens, device, verbose):
    print(f"\nLoading {model_path} ...")
    tokenizer, model = load_model(model_path, device)
    print(f"Device: {next(model.parameters()).device}")
    print(f"Dtype:  {next(model.parameters()).dtype}")
    print()

    baseline_tps = []
    twin_tps     = []
    accept_rates = []
    texts_match  = []

    for i, prompt in enumerate(PROMPTS):
        print(f"── Prompt {i+1}/{len(PROMPTS)} ──────────────────────────────")
        print(f"  {prompt[:60]}...")

        # Baseline
        text_b, n_b, t_b = generate_baseline(model, tokenizer, prompt, max_tokens)
        tps_b = n_b / t_b
        baseline_tps.append(tps_b)
        print(f"  Baseline:  {n_b:3d} tokens  {t_b:.2f}s  {tps_b:.1f} tok/s")

        # Twin
        text_t, n_t, t_t, acc = generate_twin(
            model, tokenizer, prompt, max_tokens, verbose=verbose
        )
        tps_t = n_t / t_t
        twin_tps.append(tps_t)
        accept_rates.append(acc)

        match = text_b.strip() == text_t.strip()
        texts_match.append(match)
        print(f"  Twin:      {n_t:3d} tokens  {t_t:.2f}s  {tps_t:.1f} tok/s"
              f"  accept={acc:.1%}  match={'✓' if match else '✗'}")

        if verbose or not match:
            print(f"\n  Baseline output: {text_b[:120]}")
            print(f"  Twin output:     {text_t[:120]}")
        print()

    print("══ Summary ══════════════════════════════════════════════")
    avg_b   = sum(baseline_tps) / len(baseline_tps)
    avg_t   = sum(twin_tps)     / len(twin_tps)
    avg_acc = sum(accept_rates) / len(accept_rates)
    n_match = sum(texts_match)

    print(f"  Baseline avg:      {avg_b:.1f} tok/s")
    print(f"  Twin avg:          {avg_t:.1f} tok/s")
    print(f"  Measured speedup:  {avg_t/avg_b:.2f}×  "
          f"(theoretical at {avg_acc:.0%} accept: "
          f"{1 + avg_acc:.2f}×)")
    print(f"  Avg acceptance:    {avg_acc:.1%}")
    print(f"  Output identical:  {n_match}/{len(PROMPTS)} prompts")
    print()
    print("  Note: measured speedup here reflects Python overhead of running")
    print("  two sequential forward passes rather than one fused SSQ kernel.")
    print("  The acceptance rate and theoretical multiplier are the real signal.")
    print("  On hardware with SSQ HIP kernel: throughput ≈ baseline × (1 + accept_rate)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="../model/Qwen3-1.7B")
    parser.add_argument("--tokens",  type=int, default=60)
    parser.add_argument("--device",  default="auto")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run(args.model, args.tokens, args.device, args.verbose)


if __name__ == "__main__":
    main()
