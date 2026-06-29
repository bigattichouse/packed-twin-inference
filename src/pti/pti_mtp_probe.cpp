/*
 * pti_mtp_probe.cpp — M7.0: what does the Qwen3.6 MTP head actually predict?
 *
 * The Phase-3 analysis read the graph wiring — (token_emb, h) → 1-layer
 * transformer → logits — and concluded it predicts t+1 (redundant with the
 * main head). But the SAME wiring fed with the just-emitted token instead of
 * the consumed token is a DeepSeek-V3-style t+2 drafter. The graph cannot
 * tell us which; the inputs decide. This probe measures both, empirically:
 *
 *   Variant A (same-index, the old claim):
 *     MTP(token = consumed tok @ p,  h after consuming it) vs the main
 *     model's own emission for p+1.  High agreement ⇒ fancy output head.
 *
 *   Variant B (shifted, DeepSeek semantics):
 *     MTP(token = emitted tok @ p+1, same h) vs the NEXT step's emission
 *     (the true t+2).  Agreement ≥ ~30% ⇒ viable free draft for novel text
 *     (verify batch b=2 costs 1.20× ⇒ breakeven accept ≈ 20%).
 *
 * Two separate MTP contexts so the variants' caches stay coherent.
 * Note: no MTP prefill is run — the MTP block attends only over the
 * generated region. This degrades BOTH variants equally; a strong B here is
 * a lower bound on properly-integrated accuracy.
 *
 * Build:  make mtpprobe
 */

#include "../../llama.cpp/include/llama.h"
#include "../../llama.cpp/src/llama-ext.h"   // llama_set_embeddings_pre_norm, llama_get_embeddings_pre_norm_ith

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#define MAX_TOKENS  8192
#define DEFAULT_CTX 2048

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

// Same call pattern as pti_mtp.cpp::mtp_predict_next — one MTP decode with
// the main model's pre-norm hidden state attached.
static llama_token mtp_predict(
        llama_context *ctx_mtp,
        const float   *h,
        llama_token    tok,
        llama_pos      pos,
        int32_t        n_embd,
        int32_t        n_vocab)
{
    if (!ctx_mtp || !h) return -1;

    struct llama_batch b = llama_batch_init(1, n_embd, 1);
    b.token = (llama_token *)malloc(sizeof(llama_token));

    b.n_tokens     = 1;
    b.token[0]     = tok;
    b.pos[0]       = pos;
    b.n_seq_id[0]  = 1;
    b.seq_id[0][0] = 0;
    b.logits[0]    = 1;
    memcpy(b.embd, h, (size_t)n_embd * sizeof(float));

    llama_token result = -1;
    if (llama_decode(ctx_mtp, b) == 0) {
        float *logits = llama_get_logits_ith(ctx_mtp, 0);
        if (logits) result = (llama_token)argmax_f(logits, n_vocab);
    }

    free(b.token);
    b.token = nullptr;
    llama_batch_free(b);
    return result;
}

int main(int argc, char **argv) {
    char model_path[512] = {};
    char prompt[4096]    = "The key to faster LLM inference is";
    int  max_new         = 150;
    int  n_gpu_layers    = 99;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-m")   && i+1 < argc) strncpy(model_path, argv[++i], sizeof(model_path)-1);
        else if (!strcmp(argv[i], "-p")   && i+1 < argc) strncpy(prompt,     argv[++i], sizeof(prompt)-1);
        else if (!strcmp(argv[i], "-n")   && i+1 < argc) max_new      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-ngl") && i+1 < argc) n_gpu_layers = atoi(argv[++i]);
        else { fprintf(stderr, "Usage: %s -m <model> [-p prompt] [-n tokens]\n", argv[0]); return 1; }
    }
    if (model_path[0] == '\0') { fprintf(stderr, "Error: -m required\n"); return 1; }

    llama_backend_init();

    fprintf(stderr, "\nLoading model: %s\n", model_path);
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    struct llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);
    int32_t n_embd  = llama_model_n_embd(model);

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = DEFAULT_CTX * 2;
    cparams.n_batch    = DEFAULT_CTX;
    cparams.n_seq_max  = 1;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create main context\n"); return 1; }
    llama_set_embeddings_pre_norm(ctx, true, false);

    // Two MTP contexts: one per input variant, so caches stay monotone
    struct llama_context *mtpA = nullptr, *mtpB = nullptr;
    {
        struct llama_context_params mp = llama_context_default_params();
        mp.n_ctx     = DEFAULT_CTX;
        mp.n_batch   = DEFAULT_CTX;
        mp.n_seq_max = 1;
        mp.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
        mp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        mtpA = llama_init_from_model(model, mp);
        mtpB = llama_init_from_model(model, mp);
        if (mtpA) llama_set_embeddings_pre_norm(mtpA, true, true);
        if (mtpB) llama_set_embeddings_pre_norm(mtpB, true, true);
    }
    if (!mtpA || !mtpB) {
        fprintf(stderr, "MTP context unavailable — model lacks MTP head?\n");
        return 1;
    }
    fprintf(stderr, "MTP contexts: active\n");

    llama_token toks[MAX_TOKENS];
    int n_prompt = llama_tokenize(vocab, prompt, (int32_t)strlen(prompt),
                                  toks, MAX_TOKENS, true, true);
    if (n_prompt < 0) { fprintf(stderr, "Tokenize failed\n"); return 1; }
    fprintf(stderr, "Prompt: %d tokens, generating %d\n\n", n_prompt, max_new);

    struct llama_batch batch = llama_batch_init(n_prompt + 4, 0, 1);
    batch_clear(&batch);
    for (int i = 0; i < n_prompt; i++)
        batch_add(&batch, toks[i], (llama_pos)i, 0, i == n_prompt - 1);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "Prefill failed\n"); return 1; }

    llama_pos   p   = (llama_pos)n_prompt;
    llama_token cur = (llama_token)argmax_f(llama_get_logits_ith(ctx, batch.n_tokens - 1), n_vocab);

    int a_hits = 0, a_total = 0;     // variant A: agree with t+1 (this step's emission)
    int b_hits = 0, b_total = 0;     // variant B: agree with t+2 (next step's emission)
    llama_token pend_B = -1;         // B's prediction awaiting next emission
    double mtp_ms = 0.0; int mtp_calls = 0;

    for (int step = 0; step < max_new; step++) {
        if (llama_vocab_is_eog(vocab, cur)) break;

        batch_clear(&batch);
        batch_add(&batch, cur, p, 0, true);
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "Decode failed\n"); break; }

        llama_token nxt = (llama_token)argmax_f(llama_get_logits_ith(ctx, 0), n_vocab);
        const float *h  = llama_get_embeddings_pre_norm_ith(ctx, 0);

        // settle B's pending prediction against the true t+2 (= nxt)
        if (pend_B != -1) {
            b_total++;
            if (pend_B == nxt) b_hits++;
        }

        double t0 = now_sec();
        // Variant A: consumed token @ p with h(p)  → claims to predict p+1
        llama_token cand_A = mtp_predict(mtpA, h, cur, p,     n_embd, n_vocab);
        // Variant B: emitted token @ p+1 with h(p) → claims to predict p+2
        llama_token cand_B = mtp_predict(mtpB, h, nxt, p + 1, n_embd, n_vocab);
        mtp_ms += (now_sec() - t0) * 1000.0; mtp_calls += 2;

        if (cand_A != -1) { a_total++; if (cand_A == nxt) a_hits++; }
        pend_B = cand_B;

        cur = nxt;
        p++;
    }

    fprintf(stderr, "\n════════════════════════════════════════════════════════\n");
    fprintf(stderr, "  M7.0 MTP head probe (%d steps)\n", a_total);
    fprintf(stderr, "  Variant A (same-index → t+1):  %d/%d = %.1f%%\n",
            a_hits, a_total, a_total ? 100.0 * a_hits / a_total : 0.0);
    fprintf(stderr, "      high ⇒ redundant output head (old conclusion correct)\n");
    fprintf(stderr, "  Variant B (shifted    → t+2):  %d/%d = %.1f%%\n",
            b_hits, b_total, b_total ? 100.0 * b_hits / b_total : 0.0);
    fprintf(stderr, "      ≥30%% ⇒ viable free draft for novel text (breakeven ≈ 20%%)\n");
    fprintf(stderr, "  MTP call cost: %.2f ms mean (%d calls)\n",
            mtp_calls ? mtp_ms / mtp_calls : 0.0, mtp_calls);
    fprintf(stderr, "════════════════════════════════════════════════════════\n");

    llama_batch_free(batch);
    llama_free(mtpA); llama_free(mtpB);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
