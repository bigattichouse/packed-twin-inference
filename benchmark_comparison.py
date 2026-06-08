"""
Four-point PTI throughput comparison benchmark.

Measures all four configurations for a given model:
  ① Baseline          — standard greedy decoding
  ② PTI only          — twin speculative, SSQ kernel (simulated here)
  ③ MTP only          — multi-token prediction heads (Qwen3.6 native)
  ④ PTI + MTP         — both combined

Usage:
    # Python simulation (acceptance rate only, throughput is synthetic):
    python3 benchmark_comparison.py --model ../model/Qwen3-1.7B

    # When llama.cpp integration is ready (for ③ and ④):
    python3 benchmark_comparison.py --model ../model/Qwen3-1.7B --llama-cli ../llama.cpp/build/bin/llama-cli

    # Full 27B GGUF run (requires llama.cpp):
    python3 benchmark_comparison.py --gguf ../gguf/Qwen3.6-27B-Q5_K_M.gguf --llama-cli ../llama.cpp/build/bin/llama-cli
"""

import argparse
import time
import subprocess
import sys
import re
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

PROMPTS = [
    "The key difference between a transformer and an RNN is",
    "To make a great cup of coffee, you need to",
    "The French Revolution began in 1789 because",
    "In Python, a generator function differs from a regular function in that",
    "Write an SVG image of a colorful rainbow with a blue sky background. Output only the SVG code.",
]

# MI50 constants for theoretical calculation
HBM_BANDWIDTH_GB   = 1000   # 1 TB/s
MTP_K              = 2      # Qwen3.6 MTP depth (tokens per pass)
MTP_ACCEPT_RATE    = 0.88   # empirical MTP acceptance rate


# ── ① Baseline + ② PTI (Python simulation) ────────────────────────────────────

def run_transformers_comparison(model_path: str, max_tokens: int, device: str):
    from infer import load_model, generate_baseline, generate_twin

    print(f"\nLoading {model_path} ...")
    tokenizer, model = load_model(model_path, device)
    print(f"  Device: {next(model.parameters()).device}")
    print(f"  Dtype:  {next(model.parameters()).dtype}\n")

    baseline_tps   = []
    accept_rates   = []
    output_matches = []

    for i, prompt in enumerate(PROMPTS):
        print(f"  Prompt {i+1}/{len(PROMPTS)}: {prompt[:50]}...")

        text_b, n_b, t_b = generate_baseline(model, tokenizer, prompt, max_tokens)
        text_t, n_t, t_t, acc = generate_twin(model, tokenizer, prompt, max_tokens)

        tps_b = n_b / t_b
        baseline_tps.append(tps_b)
        accept_rates.append(acc)
        output_matches.append(text_b.strip() == text_t.strip())

        status = "✓" if text_b.strip() == text_t.strip() else "✗"
        print(f"    baseline {tps_b:.1f} tok/s  accept={acc:.0%}  match={status}")

    avg_tps    = sum(baseline_tps) / len(baseline_tps)
    avg_accept = sum(accept_rates) / len(accept_rates)
    n_match    = sum(output_matches)

    return avg_tps, avg_accept, n_match


# ── ③ MTP (llama-cli subprocess) ──────────────────────────────────────────────

def run_llama_mtp(llama_cli: str, model_path: str, max_tokens: int) -> tuple:
    """
    Runs llama-cli with MTP enabled and parses throughput from stderr.
    Returns (tokens_per_sec, estimated_mtp_k).

    llama.cpp reports MTP acceptance via --verbose-prompt or log output.
    """
    cmd = [
        llama_cli,
        "-m", model_path,
        "-n", str(max_tokens),
        "--temp", "0",          # greedy
        "--mtp",                # enable multi-token prediction
        "--log-disable",
        "-p", PROMPTS[0],
    ]
    try:
        t0 = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        elapsed = time.perf_counter() - t0

        # Parse tok/s from llama.cpp stderr: "llama_print_timings: eval time = X ms / Y tokens"
        m = re.search(r"eval time\s*=\s*([\d.]+) ms\s*/\s*(\d+) tokens", result.stderr)
        if m:
            ms_per_tok = float(m.group(1)) / int(m.group(2))
            tps = 1000 / ms_per_tok
            return tps, MTP_K
        return None, None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None, None


# ── Theoretical projections ────────────────────────────────────────────────────

def theoretical_tps(baseline_tps: float, model_gb: float,
                    accept_rate: float, mtp_k: int = 1,
                    mtp_accept: float = 1.0) -> dict:
    """
    Project throughput for all four configs from measured baseline.

    For bandwidth-bound inference: throughput ∝ work_per_byte_loaded.
      baseline:   1 token per load
      PTI:        (1 + accept_rate) tokens per load
      MTP:        mtp_k * mtp_accept tokens per load
      PTI+MTP:    mtp_k * mtp_accept * (1 + accept_rate) tokens per load
    """
    t1 = baseline_tps                                          # ① baseline
    t2 = baseline_tps * (1 + accept_rate)                     # ② PTI
    t3 = baseline_tps * mtp_k * mtp_accept                    # ③ MTP
    t4 = baseline_tps * mtp_k * mtp_accept * (1 + accept_rate) # ④ PTI+MTP

    hbm_bound = (HBM_BANDWIDTH_GB * 1e9) / (model_gb * 1e9)   # absolute HBM limit

    return {
        "baseline":  (t1, 1.0),
        "pti":       (t2, t2 / t1),
        "mtp":       (t3, t3 / t1),
        "pti_mtp":   (t4, t4 / t1),
        "hbm_limit": hbm_bound,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(model_name: str, baseline_tps: float, accept_rate: float,
                 n_match: int, n_prompts: int, model_gb: float,
                 mtp_tps=None):
    proj = theoretical_tps(baseline_tps, model_gb, accept_rate,
                           mtp_k=MTP_K, mtp_accept=MTP_ACCEPT_RATE)

    print("\n" + "═" * 60)
    print(f"  PTI Four-Point Benchmark — {model_name}")
    print("═" * 60)
    print(f"  Acceptance rate (greedy):  {accept_rate:.1%}")
    print(f"  Output identical:          {n_match}/{n_prompts}")
    print()
    print(f"  {'Config':<22} {'tok/s (projected)':<20} {'multiplier'}")
    print(f"  {'─'*22} {'─'*20} {'─'*10}")

    configs = [
        ("① Baseline",  "baseline",  "measured"),
        ("② PTI only",  "pti",       "SSQ kernel (projected)"),
        ("③ MTP only",  "mtp",       "Qwen3.6 MTP k=2 (projected)"),
        ("④ PTI + MTP", "pti_mtp",   "combined (projected)"),
    ]

    for label, key, note in configs:
        tps, mul = proj[key]
        measured = ""
        if key == "baseline":
            measured = f"  ← measured"
        elif key == "mtp" and mtp_tps is not None:
            measured = f"  ← measured {mtp_tps:.1f}"
        print(f"  {label:<22} {tps:>8.1f} tok/s   {mul:>5.2f}×   {note}{measured}")

    print()
    print(f"  HBM limit ({HBM_BANDWIDTH_GB} GB/s ÷ {model_gb:.0f} GB): "
          f"{proj['hbm_limit']:.0f} tok/s theoretical max")
    print()
    print("  Note: Python simulation tok/s is ~0.5× baseline (2 sequential")
    print("  forwards). Projections assume bandwidth-bound MI50 with SSQ kernel.")
    print("═" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PTI four-point throughput benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model",      default="../model/Qwen3-1.7B",
                        help="HuggingFace safetensors model dir")
    parser.add_argument("--gguf",       default=None,
                        help="GGUF model path (for llama-cli runs)")
    parser.add_argument("--llama-cli",  default=None,
                        help="Path to llama-cli binary")
    parser.add_argument("--tokens",     type=int, default=60)
    parser.add_argument("--device",     default="auto")
    parser.add_argument("--model-gb",   type=float, default=None,
                        help="Model size in GB (for HBM projection). Auto-detected if omitted.")
    args = parser.parse_args()

    # Auto-detect model size
    model_path = args.gguf or args.model
    if args.model_gb is None:
        p = Path(model_path)
        if p.is_file():
            args.model_gb = p.stat().st_size / 1e9
        elif p.is_dir():
            args.model_gb = sum(f.stat().st_size for f in p.rglob("*.safetensors")) / 1e9
        else:
            args.model_gb = 1.0
        print(f"Model size: {args.model_gb:.1f} GB")

    # ① + ② Python sim
    if not args.gguf:
        avg_tps, avg_accept, n_match = run_transformers_comparison(
            args.model, args.tokens, args.device
        )
    else:
        print("GGUF path provided — skipping transformers simulation.")
        print("Run with --llama-cli for ③ MTP measurement.")
        avg_tps    = (HBM_BANDWIDTH_GB * 1e9) / (args.model_gb * 1e9)
        avg_accept = 1.0
        n_match    = 0

    # ③ MTP measurement via llama-cli (optional)
    mtp_tps = None
    if args.llama_cli and (args.gguf or args.model):
        print("\nRunning MTP measurement via llama-cli ...")
        mtp_path = args.gguf or args.model
        mtp_tps, _ = run_llama_mtp(args.llama_cli, mtp_path, args.tokens)
        if mtp_tps:
            print(f"  MTP measured: {mtp_tps:.1f} tok/s")
        else:
            print("  Could not parse MTP throughput from llama-cli output.")

    model_name = Path(model_path).name
    print_report(model_name, avg_tps, avg_accept,
                 n_match, len(PROMPTS), args.model_gb, mtp_tps)


if __name__ == "__main__":
    main()
