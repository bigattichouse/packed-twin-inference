/*
 * pti_kbatch_bench.cpp — M6.0: k-token batch decode cost curve
 *
 * Measures the cost of decoding k consecutive tokens of ONE sequence in a
 * single llama_decode call (the "verify batch" of lookup speculative decoding),
 * for k in {1,2,3,4,6,8}. Also measures the seq_cp checkpoint-restore cost
 * (the miss-penalty component of the lookup loop).
 *
 * Doubles as a correctness probe: the k-batch logits at position i must
 * argmax to the same token the sequential single-decode chain produced
 * (g[i+1]). If batch-decode != sequential-decode on this hybrid-SSM model,
 * lookup verification is unsound — this test would catch it.
 *
 * Method:
 *   1. Prefill prompt in seq 0; checkpoint state via seq_cp(0→1).
 *   2. Generate a 9-token greedy chain g[0..8] via single decodes (reference).
 *   3. For each k: restore seq 0 from checkpoint, decode g[0..k-1] at
 *      p0..p0+k-1 in ONE call, time it, verify argmax(i) == g[i+1].
 *
 * Build:  make kbench
 * Run:    bin/pti_kbatch_bench -m ../gguf/Qwen3.6-27B-UD-Q6_K_XL.gguf
 */

#include "../llama.cpp/include/llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#define MAX_TOKENS  4096
#define DEFAULT_CTX 2048
#define CHAIN_LEN   33      // g[0..32]: inputs g[0..31], expected outputs g[1..32]

static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int32_t argmax_f(const float *v, int32_t n) {
    int32_t best = 0;
    for (int32_t i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    return best;
}

static void batch_add(struct llama_batch *b, llama_token tok,
                      llama_pos pos, llama_seq_id seq, bool want_logits) {
    b->token   [b->n_tokens] = tok;
    b->pos     [b->n_tokens] = pos;
    b->n_seq_id[b->n_tokens] = 1;
    b->seq_id  [b->n_tokens][0] = seq;
    b->logits  [b->n_tokens] = want_logits ? 1 : 0;
    b->n_tokens++;
}

static void batch_clear(struct llama_batch *b) { b->n_tokens = 0; }

int main(int argc, char **argv) {
    char model_path[512] = {};
    char prompt[4096]    = "The key to faster LLM inference is";
    int  n_gpu_layers    = 99;
    int  n_iters         = 15;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-m")   && i+1 < argc) strncpy(model_path, argv[++i], sizeof(model_path)-1);
        else if (!strcmp(argv[i], "-p")   && i+1 < argc) strncpy(prompt,     argv[++i], sizeof(prompt)-1);
        else if (!strcmp(argv[i], "-ngl") && i+1 < argc) n_gpu_layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i")   && i+1 < argc) n_iters      = atoi(argv[++i]);
        else { fprintf(stderr, "Usage: %s -m <model> [-p prompt] [-ngl n] [-i iters]\n", argv[0]); return 1; }
    }
    if (model_path[0] == '\0') { fprintf(stderr, "Error: -m <model_path> required\n"); return 1; }

    llama_backend_init();

    fprintf(stderr, "\nLoading model: %s\n", model_path);
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    struct llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = DEFAULT_CTX * 2;
    cparams.n_batch    = DEFAULT_CTX;
    cparams.n_seq_max  = 2;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }
    llama_memory_t mem = llama_get_memory(ctx);

    llama_token prompt_toks[MAX_TOKENS];
    int n_prompt = llama_tokenize(vocab, prompt, (int32_t)strlen(prompt),
                                  prompt_toks, MAX_TOKENS, true, true);
    if (n_prompt < 0) { fprintf(stderr, "Tokenize failed\n"); return 1; }
    fprintf(stderr, "Prompt: %d tokens\n", n_prompt);

    // ── 1. Prefill seq 0, checkpoint to seq 1 ───────────────────────────────
    struct llama_batch batch = llama_batch_init(n_prompt + CHAIN_LEN + 4, 0, 2);
    batch_clear(&batch);
    for (int i = 0; i < n_prompt; i++)
        batch_add(&batch, prompt_toks[i], (llama_pos)i, 0, i == n_prompt - 1);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "Prefill failed\n"); return 1; }

    llama_pos p0 = (llama_pos)n_prompt;
    llama_token g[CHAIN_LEN];
    g[0] = (llama_token)argmax_f(llama_get_logits_ith(ctx, batch.n_tokens - 1), n_vocab);

    llama_memory_seq_cp(mem, 0, 1, 0, -1);   // checkpoint = state through prompt

    // ── 2. Reference chain: 8 sequential single decodes ─────────────────────
    for (int i = 0; i + 1 < CHAIN_LEN; i++) {
        batch_clear(&batch);
        batch_add(&batch, g[i], p0 + i, 0, true);
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "Chain decode failed\n"); return 1; }
        g[i + 1] = (llama_token)argmax_f(llama_get_logits_ith(ctx, 0), n_vocab);
    }
    fprintf(stderr, "Greedy chain:");
    for (int i = 0; i < CHAIN_LEN; i++) fprintf(stderr, " %d", g[i]);
    fprintf(stderr, "\n\n");

    // ── 3. Cost curve: k-token batch from restored checkpoint ───────────────
    const int ks[]  = {1, 2, 4, 8, 16, 24, 32};
    const int n_ks  = (int)(sizeof(ks) / sizeof(ks[0]));
    double mean_ms[16] = {};
    int    mism  [16] = {};
    double cp_ms_total = 0.0;
    int    cp_count    = 0;

    for (int ki = 0; ki < n_ks; ki++) {
        const int k = ks[ki];
        double t_total = 0.0;

        for (int it = 0; it < n_iters + 2; it++) {        // 2 warmup
            // Restore working state from checkpoint (timed separately)
            double tc0 = now_sec();
            llama_memory_seq_rm(mem, 0, 0, -1);
            llama_memory_seq_cp(mem, 1, 0, 0, -1);
            llama_synchronize(ctx);
            double tc1 = now_sec();
            if (it >= 2) { cp_ms_total += (tc1 - tc0) * 1000.0; cp_count++; }

            batch_clear(&batch);
            for (int i = 0; i < k; i++)
                batch_add(&batch, g[i], p0 + i, 0, true);

            double t0 = now_sec();
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "k=%d decode failed\n", k); return 1; }
            llama_synchronize(ctx);
            double t1 = now_sec();
            if (it >= 2) t_total += (t1 - t0) * 1000.0;

            // Correctness: batch logits must reproduce the sequential chain
            if (it == 0) {
                for (int i = 0; i < k; i++) {
                    llama_token got = (llama_token)argmax_f(llama_get_logits_ith(ctx, i), n_vocab);
                    if (got != g[i + 1]) {
                        mism[ki]++;
                        fprintf(stderr, "  [k=%d] MISMATCH at pos %d: batch=%d chain=%d\n",
                                k, i, got, g[i + 1]);
                    }
                }
            }
        }
        mean_ms[ki] = t_total / n_iters;
        fprintf(stderr, "k=%d done: %.1f ms\n", k, mean_ms[ki]);
    }

    // ── Report ───────────────────────────────────────────────────────────────
    double base = mean_ms[0];
    double cp_mean = cp_ms_total / cp_count;

    fprintf(stderr, "\n════════════════════════════════════════════════════════════\n");
    fprintf(stderr, "  M6.0 k-batch cost curve  (%d iters, %d-token prompt)\n", n_iters, n_prompt);
    fprintf(stderr, "════════════════════════════════════════════════════════════\n");
    fprintf(stderr, "  k    ms/call   ×k=1    ms/token   break-even E[j]   chain-match\n");
    for (int ki = 0; ki < n_ks; ki++) {
        const int k = ks[ki];
        // break-even accepted drafts per step for lookup: emit (1+j) tokens at
        // cost(k); profitable vs baseline when (1+j)/cost_ratio > 1 → j > ratio-1
        double ratio = mean_ms[ki] / base;
        fprintf(stderr, "  %d    %7.1f   %.2f×   %7.1f    %.2f              %s\n",
                k, mean_ms[ki], ratio, mean_ms[ki] / k, ratio - 1.0,
                mism[ki] == 0 ? "PASS" : "FAIL");
    }
    fprintf(stderr, "\n  seq_cp checkpoint restore: %.2f ms mean (%d samples)\n", cp_mean, cp_count);
    fprintf(stderr, "  baseline step (k=1):       %.1f ms → %.1f tok/s\n", base, 1000.0 / base);
    fprintf(stderr, "\n  Lookup throughput model: tok/s = (1+E[j]) / (cost_k + miss×(rebuild+cp))\n");
    fprintf(stderr, "  e.g. k=4, E[j]=2, miss=30%%: %.1f tok/s\n",
            1000.0 * 3.0 / (mean_ms[2] + 0.3 * (mean_ms[2] + cp_mean)));
    fprintf(stderr, "════════════════════════════════════════════════════════════\n");

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
