"""
Packed Twin Inference (PTI) — identical model as both drafter and verifier.

SSQ insight: one weight load from VRAM serves two forward passes simultaneously
(A at position n, B at position n+1). In this Python simulation we run two
sequential forwards — so measured speed is slower than baseline. The real
signal is acceptance rate: on actual SSQ HIP hardware,

    throughput ≈ baseline × (1 + acceptance_rate)

For greedy decoding with identical weights: acceptance = 100%, throughput = 2×.
For temperature sampling: acceptance < 100%, but still > 1×.
"""

import time
import copy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache


def load_model(model_path: str, device: str = "auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def _clone_cache(past):
    """Deep-copy a KV cache so A and B have fully independent state."""
    return copy.deepcopy(past)


@torch.no_grad()
def _forward(model, token_id, past):
    """Single cached forward step. Returns (next_token_id, new_past)."""
    ids = token_id.reshape(1, 1) if token_id.numel() == 1 else token_id.unsqueeze(0)
    out = model(
        input_ids=ids,
        past_key_values=past,
        use_cache=True,
    )
    next_tok = out.logits[0, -1, :].argmax()
    return next_tok, out.past_key_values


@torch.no_grad()
def generate_baseline(model, tokenizer, prompt: str, max_new_tokens: int = 50):
    """Standard greedy generation — single forward per token."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # Prefill
    out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    token = out.logits[:, -1, :].argmax(dim=-1).squeeze()

    generated = [token.item()]
    t0 = time.perf_counter()

    for _ in range(max_new_tokens - 1):
        if token.item() == tokenizer.eos_token_id:
            break
        token, past = _forward(model, token, past)
        generated.append(token.item())

    elapsed = time.perf_counter() - t0
    text = tokenizer.decode(generated, skip_special_tokens=True)
    # elapsed covers tokens 1..N (not the prefill)
    return text, len(generated), elapsed


@torch.no_grad()
def generate_twin(model, tokenizer, prompt: str, max_new_tokens: int = 50,
                  verbose: bool = False):
    """
    Twin speculative decoding.

    Pipeline each cycle (one SSQ weight load on real hardware):
      Twin A: processes confirmed token at position n   → produces T_{n+1}
      Twin B: processes T_{n+1} speculatively           → produces T_{n+2}_spec

    Check (next cycle): did B's speculation T_{n+1} match A's actual T_{n+1}?
      YES → acceptance, B's speculative T_{n+2} is already ready (free bonus)
      NO  → reset B, no bonus

    For greedy + identical weights: speculation is always correct → 100% acceptance.
    Measured Python speed will be slower (2 sequential forwards); acceptance rate
    is the meaningful number.
    """
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # ── Prefill ──────────────────────────────────────────────────────────────
    out = model(input_ids=ids, use_cache=True)
    past_a = out.past_key_values
    token_a = out.logits[:, -1, :].argmax(dim=-1).squeeze()   # T_0 confirmed

    # B runs one step ahead: processes T_0 to predict T_1
    # Clone so A and B have independent KV state (DynamicCache mutates in-place)
    token_b, past_b = _forward(model, token_a, _clone_cache(past_a))

    generated = [token_a.item()]
    accepted = 0
    cycles = 0

    t0 = time.perf_counter()

    # ── Main loop ─────────────────────────────────────────────────────────────
    # At start of each cycle:
    #   token_a  = last confirmed token (A will process it)
    #   token_b  = B's speculation for NEXT confirmed token
    #   past_a   = KV covering everything before token_a
    #   past_b   = KV covering everything before token_b (one step ahead of A)

    for _ in range(max_new_tokens - 1):
        if token_a.item() == tokenizer.eos_token_id:
            break

        # Twin A: confirm next token
        actual_next, past_a = _forward(model, token_a, past_a)
        cycles += 1

        # Did B correctly predict actual_next?
        if token_b.item() == actual_next.item():
            # ── Accept ───────────────────────────────────────────────────────
            # B's past already covers the correct context for actual_next.
            # B advances: produces speculation for T_{n+2}.
            spec_next, past_b = _forward(model, actual_next, past_b)
            accepted += 1

            if verbose:
                ta = tokenizer.decode([actual_next.item()])
                tb = tokenizer.decode([spec_next.item()])
                print(f"  ACCEPT  confirmed={repr(ta)}  next_spec={repr(tb)}")
        else:
            # ── Reject ───────────────────────────────────────────────────────
            # Reset B to A's new confirmed state (clone so they stay independent).
            spec_next, past_b = _forward(model, actual_next, _clone_cache(past_a))

            if verbose:
                ta = tokenizer.decode([actual_next.item()])
                tb = tokenizer.decode([token_b.item()])
                print(f"  REJECT  actual={repr(ta)}  B_had={repr(tb)}")

        generated.append(actual_next.item())
        token_a = actual_next
        token_b = spec_next   # B's next speculation

    elapsed = time.perf_counter() - t0
    text = tokenizer.decode(generated, skip_special_tokens=True)
    accept_rate = accepted / max(cycles, 1)
    return text, len(generated), elapsed, accept_rate
