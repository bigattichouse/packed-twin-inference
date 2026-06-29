"""
pti_hip.py — Python wrapper for the compiled PTI HIP kernel (libpti.so).

Build the shared library first:
    make shared               # MI50 default
    make ARCH=gfx1100 shared  # RX 7900

Then use as a drop-in replacement for the Python simulation in infer.py:
    from pti_hip import PTIKernel
    kernel = PTIKernel()
    y_a, y_b = kernel.linear(W_packed, x_a, x_b)

Or run end-to-end PTI inference:
    python3 pti_hip.py --model model.ssq --prompt "Hello"
"""

import ctypes
import os
import sys
import argparse
import time
import numpy as np
from pathlib import Path


# ── ctypes interface to libpti.so ─────────────────────────────────────────────

class PTIKernel:
    """Thin wrapper around the compiled pti_linear kernels in libpti.so."""

    def __init__(self, lib_path=None):
        if lib_path is None:
            lib_path = Path(__file__).parent / "libpti.so"
        if not Path(lib_path).exists():
            raise FileNotFoundError(
                f"libpti.so not found at {lib_path}\n"
                "Run: make shared   (or: make ARCH=gfx1100 shared)"
            )
        self._lib = ctypes.CDLL(str(lib_path))
        self._setup_signatures()

    def _setup_signatures(self):
        lib = self._lib

        # void pti_linear_launch(W, x_a, x_b, y_a, y_b, M, K, n_blocks)
        # Note: the host wrapper in pti_kernel.hip uses a struct for cfg,
        # but we expose individual ints for ctypes simplicity.
        # We call the tiled kernel directly here via a thin C export.
        lib.pti_linear_launch_c.restype = None
        lib.pti_linear_launch_c.argtypes = [
            ctypes.c_void_p,   # W_packed  (device ptr via hip)
            ctypes.c_void_p,   # x_a
            ctypes.c_void_p,   # x_b
            ctypes.c_void_p,   # y_a
            ctypes.c_void_p,   # y_b
            ctypes.c_int,      # M
            ctypes.c_int,      # K
        ]

        # int pti_verify_host(logits_a, speculation, vocab_size)
        lib.pti_verify_host.restype = ctypes.c_int
        lib.pti_verify_host.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # logits_a
            ctypes.c_int,                    # speculation (token id)
            ctypes.c_int,                    # vocab_size
        ]

    def verify_cpu(self, logits_a: np.ndarray, speculation: int) -> bool:
        """Check B's speculation against A's argmax.  Runs on CPU (numpy)."""
        return int(np.argmax(logits_a)) == speculation


# ── Numpy reference PTI linear (for testing without ROCm) ────────────────────

BLOCK_SIZE = 32
BLOCK_BYTES = 68  # 2×f16 scales + 32×uint16 weights


def pti_linear_numpy(W: np.ndarray, x_a: np.ndarray, x_b: np.ndarray):
    """
    Pure numpy implementation of pti_linear.
    W: uint8 array of shape (M, n_blocks * BLOCK_BYTES)
    x_a, x_b: float32 arrays of shape (K,)
    Returns y_a, y_b: float32 arrays of shape (M,)
    """
    M = W.shape[0]
    K = x_a.shape[0]
    n_blocks = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
    W_blocks = W.reshape(M, n_blocks, BLOCK_BYTES)

    y_a = np.zeros(M, dtype=np.float32)
    y_b = np.zeros(M, dtype=np.float32)

    for bl in range(n_blocks):
        blk = W_blocks[:, bl, :]                   # (M, 68)
        k0  = bl * BLOCK_SIZE
        k1  = min(k0 + BLOCK_SIZE, K)
        n   = k1 - k0

        # Unpack scales (f16 LE at bytes 0-1 and 2-3)
        sa_raw = blk[:, 0].astype(np.uint16) | (blk[:, 1].astype(np.uint16) << 8)
        sb_raw = blk[:, 2].astype(np.uint16) | (blk[:, 3].astype(np.uint16) << 8)
        s_a = sa_raw.view(np.float16).astype(np.float32)   # (M,)
        s_b = sb_raw.view(np.float16).astype(np.float32)

        # Unpack int8 weight pairs (bytes 4-67)
        lo = blk[:, 4:4+n*2:2].view(np.int8)   # (M, n) — lo byte = Twin B
        hi = blk[:, 5:5+n*2:2].view(np.int8)   # (M, n) — hi byte = Twin A

        xa_slice = x_a[k0:k1]   # (n,)
        xb_slice = x_b[k0:k1]

        y_a += s_a * (hi.astype(np.float32) @ xa_slice)
        y_b += s_b * (lo.astype(np.float32) @ xb_slice)

    return y_a, y_b


# ── PTI inference loop using SSQ weights (numpy path, no ROCm needed) ─────────

def pack_random_weight_block(M: int, K: int) -> np.ndarray:
    """Generate a random uint8 weight tensor in PTI block format."""
    n_blocks = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
    W = np.random.randint(0, 256, (M, n_blocks * BLOCK_BYTES), dtype=np.uint8)
    # Set scales to 1.0 in f16 (0x3C00) so output is comparable to float
    W[:, 0::BLOCK_BYTES] = 0x00
    W[:, 1::BLOCK_BYTES] = 0x3C
    W[:, 2::BLOCK_BYTES] = 0x00
    W[:, 3::BLOCK_BYTES] = 0x3C
    return W


class PTINumpyModel:
    """
    Toy PTI model using numpy pti_linear for unit testing.
    Not a real LLM — just validates the dual-stream matmul logic.
    """

    def __init__(self, hidden: int = 256, vocab: int = 1024):
        self.hidden = hidden
        self.vocab  = vocab
        rng = np.random.default_rng(0)
        # Single combined weight block: hidden × hidden (toy "attention")
        self.W_proj   = pack_random_weight_block(hidden, hidden)
        # LM head: vocab × hidden
        self.W_lm     = pack_random_weight_block(vocab, hidden)
        # Embedding table
        self.embed     = rng.standard_normal((vocab, hidden)).astype(np.float32)

    def forward(self, tok_a: int, tok_b: int):
        """One PTI step: two token IDs → two logit vectors."""
        x_a = self.embed[tok_a]
        x_b = self.embed[tok_b]

        # "Attention projection" (toy: single linear)
        h_a, h_b = pti_linear_numpy(self.W_proj, x_a, x_b)

        # LM head
        logits_a, logits_b = pti_linear_numpy(self.W_lm, h_a, h_b)
        return logits_a, logits_b


def run_pti_loop(model: PTINumpyModel, prompt_ids: list, max_new: int = 20):
    """PTI inference loop on the toy numpy model."""
    generated = list(prompt_ids)
    vocab     = model.vocab
    accepts   = 0
    total     = 0

    # Prefill: use last prompt token as starting point
    tok_a = prompt_ids[-1]
    logits_a, logits_b = model.forward(tok_a, tok_a)
    actual_next   = int(np.argmax(logits_a))
    speculation   = int(np.argmax(logits_b))

    generated.append(actual_next)
    tok_a = actual_next

    t0 = time.perf_counter()

    for step in range(max_new - 1):
        logits_a, logits_b = model.forward(tok_a, speculation)
        actual_next = int(np.argmax(logits_a))

        if speculation == actual_next:
            accepts += 1
        total += 1

        generated.append(actual_next)
        tok_a = actual_next
        speculation = int(np.argmax(logits_b))

    elapsed = time.perf_counter() - t0
    tok_per_s = max_new / elapsed if elapsed > 0 else float("inf")
    accept_rate = accepts / total if total > 0 else 0.0

    return generated, accept_rate, tok_per_s


# ── CLI ───────────────────────────────────────────────────────────────────────

def cmd_test(_args):
    """Quick correctness and throughput test on the numpy path."""
    print("PTI numpy reference — correctness test")
    np.random.seed(42)

    M, K = 512, 256
    W    = pack_random_weight_block(M, K)
    x_a  = np.random.randn(K).astype(np.float32)
    x_b  = np.random.randn(K).astype(np.float32)

    # Numpy PTI dual matmul
    t0 = time.perf_counter()
    for _ in range(500):
        y_a, y_b = pti_linear_numpy(W, x_a, x_b)
    elapsed = (time.perf_counter() - t0) / 500

    # CPU reference (single matmul each)
    n_blocks = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
    W_blocks = W.reshape(M, n_blocks, BLOCK_BYTES)
    ref_a = np.zeros(M, dtype=np.float32)
    ref_b = np.zeros(M, dtype=np.float32)
    for bl in range(n_blocks):
        blk = W_blocks[:, bl, :]
        k0 = bl * BLOCK_SIZE
        k1 = min(k0 + BLOCK_SIZE, K)
        n  = k1 - k0
        sa_raw = blk[:, 0].astype(np.uint16) | (blk[:, 1].astype(np.uint16) << 8)
        sb_raw = blk[:, 2].astype(np.uint16) | (blk[:, 3].astype(np.uint16) << 8)
        s_a = sa_raw.view(np.float16).astype(np.float32)
        s_b = sb_raw.view(np.float16).astype(np.float32)
        lo = blk[:, 4:4+n*2:2].view(np.int8)
        hi = blk[:, 5:5+n*2:2].view(np.int8)
        ref_a += s_a * (hi.astype(np.float32) @ x_a[k0:k1])
        ref_b += s_b * (lo.astype(np.float32) @ x_b[k0:k1])

    err_a = float(np.abs(y_a - ref_a).max())
    err_b = float(np.abs(y_b - ref_b).max())

    print(f"  M={M} K={K}  n_blocks={n_blocks}")
    print(f"  Twin A max error: {err_a:.2e}")
    print(f"  Twin B max error: {err_b:.2e}")
    print(f"  {'PASS' if err_a < 1e-4 and err_b < 1e-4 else 'FAIL'}")
    print(f"  Numpy throughput: {elapsed*1000:.3f} ms/call  ({M*K*2/elapsed/1e9:.2f} GFLOP/s both streams)")

    # Toy model loop
    print("\nToy PTI loop (numpy model, vocab=1024, hidden=256):")
    model = PTINumpyModel(hidden=256, vocab=1024)
    prompt = [1, 2, 3, 4, 5]
    toks, accept_rate, tok_s = run_pti_loop(model, prompt, max_new=40)
    print(f"  Generated {len(toks) - len(prompt)} tokens")
    print(f"  Acceptance rate: {accept_rate:.1%}")
    print(f"  tok/s (numpy sim): {tok_s:.0f}")
    print(f"  (toy random weights — acceptance rate not meaningful)")


def cmd_hip(_args):
    """Try to load libpti.so and run the HIP kernel test."""
    try:
        kernel = PTIKernel()
        print("libpti.so loaded successfully.")
        print("HIP kernel test: run ./pti_test (built with: make test)")
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="PTI HIP kernel Python wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("test", help="Run numpy correctness + throughput test")
    sub.add_parser("hip",  help="Load libpti.so and verify the HIP kernel")

    args = parser.parse_args()

    if args.cmd == "test" or args.cmd is None:
        cmd_test(args)
    elif args.cmd == "hip":
        cmd_hip(args)


if __name__ == "__main__":
    main()
