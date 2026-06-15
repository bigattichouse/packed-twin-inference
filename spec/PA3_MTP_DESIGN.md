# PA.3 — Per-stream MTP drafting (packed speculation)

**Goal**: give every packed lane its own MTP (multi-token-prediction) draft so each lane
emits >1 token per accepted step, the same lever `pti_lookup --mtp` already pulls for a single
stream. This is what lets a packed-agents run and a single-model run be compared **both with
MTP + thinking** (the user's requirement).

**Status**: design. Implementation target = the `--no-stream` path (`boss_generate` plan +
`run_pool` workers). Streaming (`run_pipeline`) MTP is a follow-up — its dynamic lane churn
makes the per-lane checkpoint bookkeeping much harder; the pool path is static enough to get
correct first.

---

## 1. Reference: how `pti_lookup --mtp` works (single stream)

Verified by reading `pti_lookup.cpp` (M7.1). The mechanism:

- **Two contexts on one model.** `ctx` (main, working seq 0 + checkpoint seq 1) and `ctx_mtp`
  (`ctx_type = LLAMA_CONTEXT_TYPE_MTP`, `n_seq_max = 1`, embeddings). The MTP context runs the
  model's 1-layer `nextn` head. APIs (fork extension, `llama.cpp/src/llama-ext.h`):
  `llama_set_embeddings_pre_norm(ctx,true,false)` on main exposes pre-norm hidden states;
  `llama_get_embeddings_pre_norm_ith(ctx, idx)` reads them; the MTP ctx is created with
  `cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP`.
- **Draft = 1 token (k=1).** `mtp_feed(ctx_mtp, ctx, (emitted_tok, h_idx), pos)` feeds the
  emitted token + its source hidden state into the nextn head; `argmax` of the result is the
  candidate for the next position. ~88.6% accurate (M7.0 probe); one MTP decode ≈ 3.5 ms vs
  ~52 ms full pass.
- **Verify in one batch.** Decode `[tok_last @ p (logits), draft @ p+1 (logits)]` on seq 0.
  `a0 = argmax(logits@p)` is the *real* next token; if `draft == a0`, `a1 = argmax(logits@p+1)`
  is the next-next, accepted for free. So a step emits 1 or 2 tokens.
- **SSM-safe rollback.** The draft was decoded into the KV/SSM state; a wrong draft corrupts it
  and CANNOT be truncated (recurrent state advanced). So: before the step, `seq_cp(0→1)` saves
  `S(p)`; on a miss, `seq_rm(0); seq_cp(1→0)` restores, and the correct token is re-decoded
  (pti_lookup merges this as "pending" into the next batch; we do a simple separate re-decode in
  v1). **This is mandatory for correctness, not just byte-identity** — it is why MTP needs a
  checkpoint seq per lane.

---

## 2. Packed design (N lanes)

Each lane runs the single-stream loop above, in parallel, in one `llama_decode` per step.

### 2.1 Contexts & sequence layout

- **`G.ctx_mtp`** — created once when `--mtp` is set: `ctx_type = LLAMA_CONTEXT_TYPE_MTP`,
  `n_seq_max = n_lanes`, embeddings on. One MTP seq per working lane (the nextn cache per lane
  stays contiguous; last-pair-only feeds leave position gaps, which the head tolerates — M7.1).
  If creation fails (model has no nextn head) → log and fall back to greedy (no MTP).
- **Main `G.ctx` seq map** (was: lanes `0..n_lanes-1`, base `n_lanes`):

  | role | seq ids |
  |---|---|
  | working lanes | `0 .. n_lanes-1` |
  | checkpoint of lane L | `n_lanes + L` |
  | prefix cache (PA.2.1) | `2*n_lanes` |

  ⇒ **`n_seq_max = 2*n_lanes + 1`** when `--mtp` (else the current `n_lanes + 1`). `main()` sizes
  the context accordingly. `boss_generate` (single-stream plan) uses working seq 0 + checkpoint
  seq `n_lanes` — same pool, so no extra reservation.

### 2.2 Per-step loop (`run_pool` with MTP)

For each live lane L (working seq `sL`, checkpoint `cL`, candidate `cand_L`):

1. **Refresh checkpoint** (clean lanes only): `seq_rm(cL); seq_cp(sL→cL)` ⇒ `cL = S(p_L)`.
2. **Main batch**: per lane add `tok_last_L @ p_L` (logits) and, if `cand_L != -1`,
   `cand_L @ p_L+1` (logits). One `llama_decode`.
3. **Verify + emit** per lane: `a0 = argmax(logits@tok_last_idx)`; emit. If `cand_L` present and
   `a0 == cand_L`: `a1 = argmax(logits@draft_idx)`; emit; `e_L = 2` (full accept). Else `e_L = 1`
   (miss). Track EOG / `max_new`.
4. **MTP feed** (all lanes, **before** any rebuild decode overwrites main's pre-norm buffer):
   `cand_L = mtp_feed(ctx_mtp, ctx, last_emitted_L, h_idx = last_emitted's main-batch index, p_L+e_L)`.
5. **Rollback + rebuild** missed lanes: `seq_rm(sL); seq_cp(cL→sL)` (restore `S(p_L)`), then add
   `tok_last_old_L @ p_L` to a rebuild batch; one `llama_decode` advances all missed lanes to
   `S(p_L+1)`. (Full-accept lanes are already clean at `S(p_L+2)`.)
6. **Advance**: `p_L += e_L`; `tok_last_L = a0` (miss) or `a1` (full); refill/stop as today.

When a lane refills (new pool item): `seq_rm(sL)`, `seq_rm(cL)`, re-prefill, reseed `cand_L`
from the prefill's last hidden.

### 2.3 Sampler

`pti_agents` is greedy (`argmax_f`, temp 0). Verify = exact argmax match, so accepted drafts are
byte-identical to plain greedy by construction (no sampling RNG needed). Thinking is orthogonal:
just don't pass `--no-think`.

---

## 3. Flags

| flag | effect |
|---|---|
| `--mtp` | enable per-lane MTP drafting (creates `ctx_mtp`, doubles `n_seq_max`). Default off. |
| (thinking) | on by default; `--no-think` disables it. The comparison runs **without** `--no-think`. |

---

## 4. Honest cost model (the thing to MEASURE)

MTP is **not free** here. Doubling `n_seq_max` (4→9 at `-s 4`) raises the **idle-sequence SSM
tax** (§2 of the main design): per-decode cost scales with allocated seqs, so the base per-lane
rate drops (the M-series saw ~19→~13 tok/s going N=4→8). MTP buys ~1.8 tok/accepted-step. So the
net could be a **wash, a modest win, or even a loss** — exactly what this milestone must measure,
not assume. Acceptance is "MTP fires and is accepted at a sane rate, output unchanged, and we
have a real tok/s number to compare", not a guaranteed speedup.

**MEASURED (2026-06-15, `-s 2`, Q8, Stack+Queue task, greedy/thinking-off smoke):**

| run | n_seq_max | aggregate tok/s | boss plan | MTP accept |
|---|---|---|---|---|
| greedy (no `--mtp`) | 3 | **24.1** (1.25×) | 37.4 s | — |
| `--mtp` | 5 | **14.2** (0.73×) | 57.9 s | **87%** (102/117) |

So the MTP head is correct (87% accept ≈ the 88.6% probe; output differs from greedy by **one**
near-tie-flipped token = the §2.1 `n_seq_max`-shape FP effect, **not** a rollback bug), **but the
checkpoint-seq tax makes it a net loss on this hybrid model.** The tell: the *greedy* boss plan
also slowed 37→58 s purely from `n_seq_max` 3→5 — i.e. the tax is the dominant cost and hits even
the un-drafted phase. Batching the per-lane MTP feeds into one `ctx_mtp` decode (§8 below) would
trim some overhead but cannot recover the `n_seq_max` tax. **Conclusion: on gfx906 + this hybrid
model, per-lane MTP via live checkpoint seqs does not pay; MTP's win is asymmetric — it helps the
single-stream baseline (`n_seq_max=2`, ~2× per pti_lookup) but hurts the packed pool.** Kept
behind `--mtp` (default off); the value is the measurement, per the milestone's framing.

---

## 5. Risks & mitigations

| risk | mitigation |
|---|---|
| Model lacks the nextn head | `ctx_mtp` creation fails → log, run greedy (no crash) |
| Idle-seq SSM tax eats the gain (§4) | measure; `--mtp` stays opt-in; report the honest delta |
| Per-lane rollback corrupts SSM state | mirror `pti_lookup`'s seq_cp checkpoint exactly; verify output matches non-MTP on a fixed task |
| MTP cache gaps across multi-emit steps | last-pair-only feed, per-lane MTP seq (matches M7.1; head tolerates gaps) |
| pre-norm buffer overwritten before feed | do **all** `mtp_feed`s before the rebuild decode (§2.2 step 4 before 5) |
| Streaming path churn | v1 targets `run_pool` only; `run_pipeline` MTP is a documented follow-up |

---

## 6. Acceptance criteria

1. `--mtp-test` (GPU-free) passes: verify/accept bookkeeping + seq-layout math.
2. Build clean; `ctx_mtp` reports "active" on the Qwen3.6 model.
3. `pti_agents --no-stream --mtp` on a fixed task produces the **same** worker outputs as without
   `--mtp` (greedy determinism), with a non-zero MTP accept rate logged per run.
4. A reported aggregate tok/s for `--mtp` vs no-`--mtp` (whatever the sign).
5. No crash on lane refill or EOG with MTP on.

---

## 7. Test plan

### 7.1 GPU-free unit tests (`--mtp-test`, like `--gather-test`)

The model-free logic is the verify/accept decision and the seq-layout math. Extract pure
helpers and assert:

| # | function | case | expected |
|---|---|---|---|
| M1 | `mtp_verify` | draft present, `a0==draft` | `e=2, acc=1, full=true` |
| M2 | `mtp_verify` | draft present, `a0!=draft` | `e=1, acc=0, full=false` |
| M3 | `mtp_verify` | no draft (`cand=-1`) | `e=1, acc=0, full=false` |
| M4 | `mtp_verify` | `a0` is EOG | `e=1, stop=true` |
| M5 | `mtp_verify` | full accept but `a1` is EOG | `e=2, stop=true` |
| M6 | `mtp_ckpt_seq(L,n)` | lane 0..n-1 | `n + L` |
| M7 | `mtp_base_seq(n)` | n lanes | `2*n` |
| M8 | `mtp_seqmax(n, mtp_on)` | on / off | `2n+1` / `n+1` |

### 7.2 GPU smoke

```
# determinism: same output with/without MTP (greedy)
pti_agents -m <model> -p "<fixed task>" --no-stream --no-gather -n 64        > /tmp/a.txt
pti_agents -m <model> -p "<fixed task>" --no-stream --no-gather -n 64 --mtp  > /tmp/b.txt
diff <(grep '── item' -A99 /tmp/a.txt) ...   # worker bodies identical
# accept + speed: stderr reports MTP fires/accepted % and aggregate tok/s for both
```

---

## 8. Out of scope (v1) / future optimizations

- **Batched MTP feeds**: do one `ctx_mtp` decode per step for all lanes (one batch, per-seq)
  instead of N sequential `mtp_feed1` calls. Trims per-step overhead but not the `n_seq_max` tax.
- **Host-serialized checkpoints** (`llama_state_seq_get_data/set_data`) instead of live checkpoint
  seqs — avoids doubling `n_seq_max` (and its SSM tax) at the cost of a state memcpy per lane/step.
  This is the only path that could make packed MTP pay on this hybrid model; measure before doing.
- Streaming (`run_pipeline`) MTP.
- n-gram/lookup drafting per lane, AIMD gates, draft ladders (7→15→31), pending-merge rebuild —
  all `pti_lookup` refinements; v1 is pure k=1 MTP with a simple separate rebuild.
- Sampled (temperature) verification — pti_agents is greedy.
