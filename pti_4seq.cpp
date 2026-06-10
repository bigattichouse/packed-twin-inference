/*
 * pti_4seq.cpp — Packed Twin Inference, 4-sequence quad-batch
 *
 *   Seq 0 (A): verifier at pos_a
 *   Seq 1 (B): 1-step drafter at pos_b = pos_a + 1
 *   Seq 2 (C): 2-step drafter at pos_c = pos_a + 2
 *   Seq 3 (D): 3-step drafter at pos_d = pos_a + 3
 *
 * Quad-batch each step: one llama_decode, one weight load, 4 predictions.
 * On 100% greedy accept: 4 tokens per step.
 *
 * Overhead scaling (measured, UD-Q6_K_XL, MI50):
 *   2-seq: 1.27× baseline step cost → 2/1.27 = 1.57×  (30.5 tok/s)
 *   3-seq: 1.71× baseline step cost → 3/1.71 = 1.75×  (33.9 tok/s)
 *   4-seq: 2.04× baseline step cost → 4/2.04 = 1.96×  (38.1 tok/s) ← measured
 *
 * Build:
 *   g++ -O2 -std=c++17 -o pti_4seq pti_4seq.cpp \
 *       -I../llama.cpp/include -I../llama.cpp/ggml/include \
 *       -L../llama.cpp/build/bin -lllama \
 *       -Wl,-rpath,../llama.cpp/build/bin -lm
 */

#include "../llama.cpp/include/llama.h"
#include "../llama.cpp/src/llama-ext.h"  // llama_set_embeddings_pre_norm, llama_get_embeddings_pre_norm_ith

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <cfloat>

#define MAX_TOKENS   4096
#define DEFAULT_CTX  2048

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

static void print_token(const struct llama_vocab *vocab, int32_t tok) {
    char buf[256];
    int len = llama_token_to_piece(vocab, tok, buf, (int32_t)sizeof(buf), 0, true);
    if (len > 0) { fwrite(buf, 1, len, stdout); fflush(stdout); }
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

// MTP prediction: takes (tok_t, h_t from main context at batch_idx) → predicts t+1.
// NOTE: this predicts the SAME position as the main model's logits for batch_idx.
// It does NOT add a bonus token — it is an alternative 1-layer prediction of the
// same next-token position, useful as a fast approximate predictor on reject paths.
// Returns -1 on failure.
static llama_token mtp_predict_next(
        llama_context *ctx_mtp,
        llama_context *ctx_main,
        int32_t        batch_idx,
        llama_token    tok_at_pos,
        llama_pos      pos,
        int32_t        n_embd,
        int32_t        n_vocab)
{
    if (!ctx_mtp) return -1;
    const float *h = llama_get_embeddings_pre_norm_ith(ctx_main, batch_idx);
    if (!h) return -1;

    struct llama_batch mb = llama_batch_init(1, n_embd, 1);
    mb.token = (llama_token *)malloc(sizeof(llama_token));
    mb.n_tokens     = 1;
    mb.token[0]     = tok_at_pos;
    mb.pos[0]       = pos;
    mb.n_seq_id[0]  = 1;
    mb.seq_id[0][0] = 0;
    mb.logits[0]    = 1;
    memcpy(mb.embd, h, (size_t)n_embd * sizeof(float));

    llama_token result = -1;
    if (llama_decode(ctx_mtp, mb) == 0) {
        float *logits = llama_get_logits_ith(ctx_mtp, 0);
        if (logits) result = (llama_token)argmax_f(logits, n_vocab);
    }
    free(mb.token);
    mb.token = nullptr;
    llama_batch_free(mb);
    return result;
}

// Reinit sequence dst for hybrid attention+SSM models:
//   rm_all (p0=0) always succeeds even when partial rollback cannot.
//   seq_cp copies BOTH attention KV and recurrent state from src to dst.
//   Then decode tok at pos and return the predicted next token.
//   Sets *pos_out = pos + 1.
static llama_token reinit_seq(
        llama_memory_t        mem,
        struct llama_context *ctx,
        struct llama_batch   *b,
        int32_t               n_vocab,
        llama_seq_id          src,
        llama_seq_id          dst,
        llama_token           tok,
        llama_pos             pos,
        llama_pos            *pos_out)
{
    llama_memory_seq_rm(mem, dst, 0, -1);
    llama_memory_seq_cp(mem, src, dst, 0, -1);
    batch_clear(b);
    batch_add(b, tok, pos, dst, true);
    llama_token next = 0;
    if (llama_decode(ctx, *b) == 0)
        next = (llama_token)argmax_f(llama_get_logits_ith(ctx, 0), n_vocab);
    *pos_out = pos + 1;
    return next;
}

struct PTIArgs {
    int  max_new      = 80;
    int  n_gpu_layers = 99;
    bool verbose      = false;
    char model_path[512] = {};
    char prompt[4096]    = {};
};

static int run_pti4(const PTIArgs *args) {
    fprintf(stderr, "\nLoading model: %s\n", args->model_path);

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args->n_gpu_layers;
    struct llama_model *model = llama_model_load_from_file(args->model_path, mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);
    int32_t n_embd  = llama_model_n_embd(model);
    fprintf(stderr, "Vocab: %d  n_embd: %d\n", n_vocab, n_embd);

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = DEFAULT_CTX * 4;
    cparams.n_batch    = DEFAULT_CTX;
    cparams.n_seq_max  = 4;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    // Enable pre-norm embeddings (needed for mtp_predict_next on reject paths)
    llama_set_embeddings_pre_norm(ctx, true, false);

    // MTP context — 1-layer MTP head for fast next-token prediction.
    // At greedy (100% 4-ACCEPT) this context is created but never invoked.
    // It is useful for temperature>0 reject paths.
    struct llama_context *ctx_mtp = nullptr;
    {
        struct llama_context_params mp = llama_context_default_params();
        mp.n_ctx     = DEFAULT_CTX;
        mp.n_batch   = DEFAULT_CTX;
        mp.n_seq_max = 1;
        mp.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
        mp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        ctx_mtp = llama_init_from_model(model, mp);
        if (ctx_mtp) {
            llama_set_embeddings_pre_norm(ctx_mtp, true, true);
            fprintf(stderr, "MTP context: active\n");
        } else {
            fprintf(stderr, "MTP context: unavailable (model may lack MTP heads)\n");
        }
    }

    llama_memory_t mem = llama_get_memory(ctx);

    llama_token prompt_toks[MAX_TOKENS];
    int n_prompt = llama_tokenize(vocab, args->prompt, (int32_t)strlen(args->prompt),
                                  prompt_toks, MAX_TOKENS, true, true);
    if (n_prompt < 0) { fprintf(stderr, "Tokenize failed\n"); return 1; }
    fprintf(stderr, "Prompt: %d tokens\n\n", n_prompt);

    // ── Prefill under seq 0, copy to seqs 1, 2, 3 ───────────────────────────
    struct llama_batch batch = llama_batch_init(n_prompt + 8, 0, 4);
    batch_clear(&batch);
    for (int i = 0; i < n_prompt; i++)
        batch_add(&batch, prompt_toks[i], (llama_pos)i, 0, i == n_prompt - 1);

    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "Prefill failed\n"); return 1; }
    llama_pos pos_a = (llama_pos)n_prompt;

    float *lp = llama_get_logits_ith(ctx, batch.n_tokens - 1);
    llama_token tok_gen_0 = (llama_token)argmax_f(lp, n_vocab);

    llama_memory_seq_cp(mem, 0, 1, 0, -1);
    llama_memory_seq_cp(mem, 0, 2, 0, -1);
    llama_memory_seq_cp(mem, 0, 3, 0, -1);

    fprintf(stderr, "Prefill done. First token: [%d]\n", tok_gen_0);
    fprintf(stderr, "Output: ");
    print_token(vocab, tok_gen_0);

    double t_gen_start = now_sec();  // honest start: includes init overhead below

    // ── Prime seq0 with its own exclusive SSM cell before B/C/D inits ─────────
    // After seq_cp, all 4 seqs share one SSM cell. Without this step, the
    // recurrent memory's gather+re-order in the first quad-batch places seq0 in
    // an unexpected physical slot, causing the SSM layers to read the wrong
    // source state and producing wrong logits for seq0 only. Running seq0 alone
    // first creates an exclusive cell for it; the reset puts it back at the
    // prefill state via seq1's still-untouched cell, leaving the cell pool in
    // an arrangement that the quad-batch gather handles correctly.
    batch_clear(&batch);
    batch_add(&batch, tok_gen_0, pos_a, 0, true);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "seq0 prime failed\n"); return 1; }
    llama_memory_seq_rm(mem, 0, 0, -1);
    llama_memory_seq_cp(mem, 1, 0, 0, -1);

    // ── B init: B processes tok_gen_0 at pos_a ───────────────────────────────
    batch_clear(&batch);
    batch_add(&batch, tok_gen_0, pos_a, 1, true);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "B init failed\n"); return 1; }
    llama_token tok_b = (llama_token)argmax_f(llama_get_logits_ith(ctx, 0), n_vocab);
    llama_pos   pos_b = pos_a + 1;

    // ── C init: C processes tok_gen_0 at pos_a, then tok_b at pos_b ─────────
    batch_clear(&batch);
    batch_add(&batch, tok_gen_0, pos_a, 2, false);  // fill C's KV at pos_a
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "C init stage 1 failed\n"); return 1; }

    batch_clear(&batch);
    batch_add(&batch, tok_b, pos_b, 2, true);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "C init stage 2 failed\n"); return 1; }
    llama_token tok_c = (llama_token)argmax_f(llama_get_logits_ith(ctx, 0), n_vocab);
    llama_pos   pos_c = pos_b + 1;

    // ── D init: D mirrors C's init, then processes tok_c at pos_c ───────────
    batch_clear(&batch);
    batch_add(&batch, tok_gen_0, pos_a, 3, false);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "D init stage 1 failed\n"); return 1; }

    batch_clear(&batch);
    batch_add(&batch, tok_b, pos_b, 3, false);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "D init stage 2 failed\n"); return 1; }

    batch_clear(&batch);
    batch_add(&batch, tok_c, pos_c, 3, true);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "D init stage 3 failed\n"); return 1; }
    llama_token tok_d = (llama_token)argmax_f(llama_get_logits_ith(ctx, 0), n_vocab);
    llama_pos   pos_d = pos_c + 1;

    llama_token tok_a = tok_gen_0;

    int n_gen    = 1;
    int n_4acc   = 0;
    int n_3acc   = 0;
    int n_2acc   = 0;
    int n_reject = 0;

    struct llama_batch step_batch = llama_batch_init(4, 0, 4);
    double t_start = now_sec();
    // NOTE: t_init (the 7 single-seq decode steps to stagger B/C/D before the loop)
    // is NOT included here. See "Init overhead" in the stats below for the honest total.

    for (int step = 0; step < args->max_new - 1; step++) {
        if (n_gen >= args->max_new) break;
        if (llama_vocab_is_eog(vocab, tok_a)) break;

        batch_clear(&step_batch);
        batch_add(&step_batch, tok_a, pos_a, 0, true);
        batch_add(&step_batch, tok_b, pos_b, 1, true);
        batch_add(&step_batch, tok_c, pos_c, 2, true);
        batch_add(&step_batch, tok_d, pos_d, 3, true);

        if (llama_decode(ctx, step_batch) != 0) {
            fprintf(stderr, "\nDecode failed at step %d\n", step);
            break;
        }

        llama_token actual_next  = (llama_token)argmax_f(llama_get_logits_ith(ctx, 0), n_vocab);
        llama_token next_from_b  = (llama_token)argmax_f(llama_get_logits_ith(ctx, 1), n_vocab);
        llama_token next_from_c  = (llama_token)argmax_f(llama_get_logits_ith(ctx, 2), n_vocab);
        llama_token next_from_d  = (llama_token)argmax_f(llama_get_logits_ith(ctx, 3), n_vocab);

        pos_a++; pos_b++; pos_c++; pos_d++;


        if (tok_b == actual_next) {
            if (tok_c == next_from_b) {
                if (tok_d == next_from_c) {
                    // ── 4-accept ───────────────────────────────────────────
                    n_4acc++;
                    if (args->verbose)
                        fprintf(stderr, "\n[4-ACCEPT pos=%d]", pos_a-1);

                    // M5.3: emit all 4 confirmed tokens, then rebuild stagger
                    // from D's state (3 reinit_seq calls). D is confirmed through
                    // pos_d-1; copying D→A bridges A forward 3 positions at once.
                    // Fall back to single-emit near token limit or if eog in chain.
                    bool full_emit = (n_gen + 4 <= args->max_new)
                                  && !llama_vocab_is_eog(vocab, actual_next)
                                  && !llama_vocab_is_eog(vocab, next_from_b)
                                  && !llama_vocab_is_eog(vocab, next_from_c)
                                  && !llama_vocab_is_eog(vocab, next_from_d);

                    if (full_emit) {
                        print_token(vocab, actual_next);  n_gen++;
                        print_token(vocab, next_from_b);  n_gen++;
                        print_token(vocab, next_from_c);  n_gen++;
                        print_token(vocab, next_from_d);  n_gen++;
                        llama_memory_seq_rm(mem, 0, 0, -1);
                        llama_memory_seq_cp(mem, 3, 0, 0, -1);
                        tok_a = next_from_d;
                        pos_a = pos_d;
                        tok_b = reinit_seq(mem, ctx, &step_batch, n_vocab, 0, 1, next_from_d, pos_d, &pos_b);
                        tok_c = reinit_seq(mem, ctx, &step_batch, n_vocab, 1, 2, tok_b,       pos_b, &pos_c);
                        tok_d = reinit_seq(mem, ctx, &step_batch, n_vocab, 2, 3, tok_c,       pos_c, &pos_d);
                    } else {
                        print_token(vocab, actual_next);  n_gen++;
                        tok_a = actual_next;
                        tok_b = next_from_b;
                        tok_c = next_from_c;
                        tok_d = next_from_d;
                    }

                } else {
                    // ── 3-accept: D wrong, re-init D ──────────────────────
                    n_3acc++;
                    if (args->verbose)
                        fprintf(stderr, "\n[3-ACCEPT pos=%d]", pos_a-1);

                    print_token(vocab, actual_next);  n_gen++;

                    tok_a = actual_next;
                    tok_b = next_from_b;
                    tok_c = next_from_c;

                    // D wrong: copy C's confirmed state, decode next_from_c at pos_c
                    tok_d = reinit_seq(mem, ctx, &step_batch, n_vocab,
                                       /*src=*/2, /*dst=*/3, next_from_c, pos_c, &pos_d);
                }
            } else {
                // ── 2-accept: C+D wrong, re-init C and D ──────────────────
                n_2acc++;
                if (args->verbose)
                    fprintf(stderr, "\n[2-ACCEPT pos=%d]", pos_a-1);

                print_token(vocab, actual_next);  n_gen++;

                tok_a = actual_next;
                tok_b = next_from_b;

                // C wrong: copy B's confirmed state, decode next_from_b at pos_b
                tok_c = reinit_seq(mem, ctx, &step_batch, n_vocab,
                                   /*src=*/1, /*dst=*/2, next_from_b, pos_b, &pos_c);

                // D wrong: copy C's post-reinit state (chain), decode tok_c at pos_c
                tok_d = reinit_seq(mem, ctx, &step_batch, n_vocab,
                                   /*src=*/2, /*dst=*/3, tok_c, pos_c, &pos_d);
            }
        } else {
            // ── Reject: B wrong, re-init B, C, D ─────────────────────────
            n_reject++;
            if (args->verbose)
                fprintf(stderr, "\n[REJECT pos=%d]", pos_a-1);

            print_token(vocab, actual_next);  n_gen++;

            // B wrong: copy A's confirmed state, decode actual_next at pos_a
            tok_b = reinit_seq(mem, ctx, &step_batch, n_vocab,
                               /*src=*/0, /*dst=*/1, actual_next, pos_a, &pos_b);

            // C wrong: copy B's post-reinit state (chain), decode tok_b at pos_b
            tok_c = reinit_seq(mem, ctx, &step_batch, n_vocab,
                               /*src=*/1, /*dst=*/2, tok_b, pos_b, &pos_c);

            // D wrong: copy C's post-reinit state (chain), decode tok_c at pos_c
            tok_d = reinit_seq(mem, ctx, &step_batch, n_vocab,
                               /*src=*/2, /*dst=*/3, tok_c, pos_c, &pos_d);
            tok_a = actual_next;
        }

        if (llama_vocab_is_eog(vocab, actual_next)) break;
    }

    double elapsed      = now_sec() - t_start;
    double t_init_cost  = t_start - t_gen_start;   // 7 startup decode steps
    double elapsed_full = elapsed + t_init_cost;    // honest end-to-end
    int total_checks = n_4acc + n_3acc + n_2acc + n_reject;

    fprintf(stderr, "\n\n════════════════════════════════════════\n");
    fprintf(stderr, "  PTI-4seq Results\n");
    fprintf(stderr, "  Tokens generated:  %d\n", n_gen);
    fprintf(stderr, "  4-accept:          %d\n", n_4acc);
    fprintf(stderr, "  3-accept:          %d  (D re-init)\n", n_3acc);
    fprintf(stderr, "  2-accept:          %d  (C+D re-init)\n", n_2acc);
    fprintf(stderr, "  Rejects:           %d\n", n_reject);
    if (total_checks > 0) {
        fprintf(stderr, "  B accept rate:     %.1f%%\n",
                100.0 * (n_4acc + n_3acc + n_2acc) / total_checks);
        fprintf(stderr, "  C accept rate:     %.1f%%  (of B-accepts)\n",
                (n_4acc + n_3acc + n_2acc) > 0
                ? 100.0 * (n_4acc + n_3acc) / (n_4acc + n_3acc + n_2acc) : 0.0);
        fprintf(stderr, "  D accept rate:     %.1f%%  (of C-accepts)\n",
                (n_4acc + n_3acc) > 0
                ? 100.0 * n_4acc / (n_4acc + n_3acc) : 0.0);
    }
    fprintf(stderr, "  Init overhead:     %.0f ms  (7 startup decode steps, one-time)\n",
            t_init_cost * 1000.0);
    fprintf(stderr, "  Steady-state:      %.1f tok/s  (loop only, excludes init)\n",
            n_gen / elapsed);
    fprintf(stderr, "  Amortized:         %.1f tok/s  (init + loop, honest end-to-end)\n",
            n_gen / elapsed_full);
    fprintf(stderr, "════════════════════════════════════════\n");

    llama_batch_free(batch);
    llama_batch_free(step_batch);
    if (ctx_mtp) llama_free(ctx_mtp);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

static int run_baseline(const PTIArgs *args) {
    fprintf(stderr, "\nLoading model (baseline): %s\n", args->model_path);
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args->n_gpu_layers;
    struct llama_model *model = llama_model_load_from_file(args->model_path, mparams);
    if (!model) return 1;

    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = DEFAULT_CTX;
    cparams.n_batch = DEFAULT_CTX;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) return 1;

    llama_token toks[MAX_TOKENS];
    int n_prompt = llama_tokenize(vocab, args->prompt, (int32_t)strlen(args->prompt),
                                  toks, MAX_TOKENS, true, true);

    struct llama_batch batch = llama_batch_init(n_prompt + 2, 0, 1);
    batch_clear(&batch);
    for (int i = 0; i < n_prompt; i++)
        batch_add(&batch, toks[i], (llama_pos)i, 0, i == n_prompt - 1);
    if (llama_decode(ctx, batch) != 0) return 1;

    llama_pos pos = (llama_pos)n_prompt;
    llama_token tok = (llama_token)argmax_f(llama_get_logits_ith(ctx, batch.n_tokens-1), n_vocab);
    fprintf(stderr, "Prompt: %d tokens\nOutput: ", n_prompt);
    print_token(vocab, tok);

    int n_gen = 1;
    struct llama_batch sb = llama_batch_init(2, 0, 1);
    double t0 = now_sec();
    for (int i = 0; i < args->max_new - 1; i++) {
        if (llama_vocab_is_eog(vocab, tok)) break;
        batch_clear(&sb);
        batch_add(&sb, tok, pos++, 0, true);
        if (llama_decode(ctx, sb) != 0) break;
        tok = (llama_token)argmax_f(llama_get_logits_ith(ctx, 0), n_vocab);
        print_token(vocab, tok);
        n_gen++;
    }
    double elapsed = now_sec() - t0;
    fprintf(stderr, "\n\n════════════════════════════════════════\n");
    fprintf(stderr, "  Baseline:  %d tok  %.1f tok/s\n", n_gen, n_gen / elapsed);
    fprintf(stderr, "════════════════════════════════════════\n");

    llama_batch_free(batch);
    llama_batch_free(sb);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  -m <path>    GGUF model path  (required)\n"
        "  -p <text>    Prompt\n"
        "  -n <int>     Max new tokens   (default: 80)\n"
        "  -ngl <int>   GPU layers       (default: 99)\n"
        "  --baseline   Run single-stream baseline\n"
        "  --verbose    Print accept/reject per step\n",
        prog);
}

int main(int argc, char **argv) {
    PTIArgs args;
    strncpy(args.prompt, "The meaning of life is", sizeof(args.prompt) - 1);
    bool do_baseline = false;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-m")   && i+1 < argc) strncpy(args.model_path, argv[++i], sizeof(args.model_path)-1);
        else if (!strcmp(argv[i], "-p")   && i+1 < argc) strncpy(args.prompt,     argv[++i], sizeof(args.prompt)-1);
        else if (!strcmp(argv[i], "-n")   && i+1 < argc) args.max_new      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-ngl") && i+1 < argc) args.n_gpu_layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--baseline")) do_baseline = true;
        else if (!strcmp(argv[i], "--verbose"))  args.verbose = true;
        else if (!strcmp(argv[i], "--help"))     { usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
    }

    if (args.model_path[0] == '\0') {
        fprintf(stderr, "Error: -m <model_path> required\n");
        usage(argv[0]); return 1;
    }

    llama_backend_init();
    int rc = do_baseline ? run_baseline(&args) : run_pti4(&args);
    llama_backend_free();
    return rc;
}
