# llama.cpp patches

PTI builds against **upstream `ggerganov/llama.cpp` master @ `ad2775726`** (or later).
Almost everything PTI needs is already upstream at that commit:

- MTP / nextn context support (`LLAMA_CONTEXT_TYPE_MTP` — PRs #22673, #23198)
- pre-norm embeddings ext (`src/llama-ext.h`: `llama_set_embeddings_pre_norm`,
  `llama_get_embeddings_pre_norm_ith`)

## The one local patch

| patch | what | required? |
|---|---|---|
| `0001-mmvq-i-outer-j-inner.patch` | MMVQ loop-order fix (M5.1): makes the weight-block address loop-invariant across columns so LLVM hoists the load. N=4 batched-decode overhead 1.95× → 1.86× on gfx906. | **Optional** — performance only. PTI is correct without it; verify batches just cost a few % more. |

(An earlier "interleaved activation" patch lived here; it was benchmarked at no gain
and removed — see `FAILED_EXPERIMENTS.md` in the repo root for the post-mortem.)

## Apply

```bash
./llama.cpp-patch/patch.sh                  # apply to ../llama.cpp
./llama.cpp-patch/patch.sh --build          # apply + rebuild (ROCm 7.2.1 / gfx906)
./llama.cpp-patch/patch.sh /path/to/llama.cpp
```

Idempotent — re-running skips already-applied patches.

## If the patch doesn't apply (upstream drift)

`../llama.cpp-files/` mirrors the llama.cpp directory structure with the full modified
files — read them in the same place you'd find them upstream, or copy the tree over:

```bash
cp -r llama.cpp-files/* /path/to/llama.cpp/
```

Current contents: `ggml/src/ggml-cuda/mmvq.cu` (base commit + patch applied).
