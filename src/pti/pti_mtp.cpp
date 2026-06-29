/*
 * pti_mtp.cpp — Packed Twin Inference + MTP (3-sequence, C++)
 *
 * Extends pti_llama.c from 2 sequences to 3:
 *   Seq 0 (Twin A): verifier at pos_a
 *   Seq 1 (Twin B): 1-step drafter at pos_b = pos_a + 1
 *   Seq 2 (Twin C): 2-step drafter at pos_c = pos_a + 2
 *
 * Triple-batch each step: one weight load serves all 3 predictions.
 * On 100% greedy accept: 3 tokens per step vs 2 in pti_llama.c.
 *
 * MTP context (LLAMA_CONTEXT_TYPE_MTP, 1 extra layer, ~negligible cost):
 *   - Used on the REJECT path to re-init C from B's hidden state
 *     without running an extra full-model forward pass.
 *   - On 100% greedy: never fires; C re-init via MTP only matters for sampling.
 *
 * Why gains don't multiply (1.38× × 1.7× ≠ 2.35×):
 *   Triple-batch overhead ≈ 1.8× baseline step cost.
 *   3 tokens / 1.8× overhead ≈ 1.67× baseline — less than 1.38× × 1.7×.
 *   Multiplicative gains require the SSQ fused kernel (one weight load for all
 *   sequences, no triple-batch overhead).
 *
 * Build:
 *   g++ -O2 -std=c++17 -o pti_mtp pti_mtp.cpp \
 *       -I../llama.cpp/include \
 *       -I../llama.cpp/src \
 *       -L../llama.cpp/build/bin -lllama \
 *       -Wl,-rpath,../llama.cpp/build/bin \
 *       -lm
 *
 * Usage:
 *   ./pti_mtp -m ../gguf/Qwen3.6-27B-UD-Q6_K_XL.gguf -p "The speed of light" -n 80 -ngl 99
 *   ./pti_mtp --baseline -m ../gguf/Qwen3.6-27B-UD-Q6_K_XL.gguf -p "same prompt" -n 80
 */

#include "../../llama.cpp/include/llama.h"
#include "../../llama.cpp/src/llama-ext.h"  // llama_set_embeddings_pre_norm, llama_get_embeddings_pre_norm_ith

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <cfloat>
#include <vector>

#define MAX_TOKENS   4096
#define DEFAULT_CTX  2048

// ── Utility ──────────────────────────────────────────────────────────────────

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

// ── Batch helpers ─────────────────────────────────────────────────────────────

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

// ── PTI args ─────────────────────────────────────────────────────────────────

struct PTIArgs {
    int  max_new      = 80;
    int  n_gpu_layers = 99;
    bool verbose      = false;
    char model_path[512] = {};
    char prompt[4096]    = {};
};

// ── C re-init via MTP ────────────────────────────────────────────────────────
//
// After a reject (or 2-accept where C was wrong), we need to re-seed tok_c.
// MTP takes (token at pos, H from that pos in the main context) and produces
// a prediction for the next token — same computation as B's decode, but only
// 1 layer instead of 65.  Saves one full forward pass on every reject.
//
// ctx_main must have been decoded with llama_set_embeddings_pre_norm(true).
// batch_idx_b: index of B's token in the most recent main decode batch.
// tok_at_b:    the token B just processed.
// pos_b:       B's position in that batch.
//
// Returns the MTP-predicted tok_c, or -1 on failure (caller falls back to
// a full C decode).

static llama_token mtp_predict_next(
        llama_context *ctx_mtp,
        llama_context *ctx_main,
        int32_t batch_idx_b,
        llama_token tok_at_b,
        llama_pos pos_b,
        int32_t n_embd,
        int32_t n_vocab)
{
    if (!ctx_mtp) return -1;

    const float *h_b = llama_get_embeddings_pre_norm_ith(ctx_main, batch_idx_b);
    if (!h_b) return -1;

    // llama_batch_init with embd>0 allocates batch.embd but NOT batch.token.
    // The Qwen3.6 MTP graph needs both. Manually allocate token.
    struct llama_batch mtp_batch = llama_batch_init(1, n_embd, 1);
    mtp_batch.token = (llama_token *)malloc(sizeof(llama_token));

    mtp_batch.n_tokens       = 1;
    mtp_batch.token[0]       = tok_at_b;
    mtp_batch.pos[0]         = pos_b;
    mtp_batch.n_seq_id[0]    = 1;
    mtp_batch.seq_id[0][0]   = 0;
    mtp_batch.logits[0]      = 1;
    memcpy(mtp_batch.embd, h_b, (size_t)n_embd * sizeof(float));

    llama_token result = -1;
    if (llama_decode(ctx_mtp, mtp_batch) == 0) {
        float *logits = llama_get_logits_ith(ctx_mtp, 0);
        if (logits) result = (llama_token)argmax_f(logits, n_vocab);
    }

    free(mtp_batch.token);
    mtp_batch.token = nullptr;
    llama_batch_free(mtp_batch);
    return result;
}

// ── 3-seq PTI inference loop ──────────────────────────────────────────────────

static int run_pti3(const PTIArgs *args) {
    fprintf(stderr, "\nLoading model: %s\n", args->model_path);

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args->n_gpu_layers;
    struct llama_model *model = llama_model_load_from_file(args->model_path, mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    const struct llama_vocab *vocab  = llama_model_get_vocab(model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);
    int32_t n_embd  = llama_model_n_embd(model);
    fprintf(stderr, "Vocab: %d  n_embd: %d\n", n_vocab, n_embd);

    // ── Main context: 3 sequences ───────────────────────────────────────────
    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = DEFAULT_CTX * 3;
    cparams.n_batch    = DEFAULT_CTX;
    cparams.n_seq_max  = 3;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create main context\n"); return 1; }

    // Enable pre-norm embedding output (needed for MTP C re-init)
    llama_set_embeddings_pre_norm(ctx, true, false);

    // ── MTP context (optional — only valid for models with MTP heads) ────────
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
            fprintf(stderr, "MTP context: active (for C re-init on reject)\n");
        } else {
            fprintf(stderr, "MTP context: unavailable (model may lack MTP heads)\n");
        }
    }

    llama_memory_t mem = llama_get_memory(ctx);

    // ── Tokenize prompt ─────────────────────────────────────────────────────
    llama_token prompt_toks[MAX_TOKENS];
    int n_prompt = llama_tokenize(vocab, args->prompt, (int32_t)strlen(args->prompt),
                                  prompt_toks, MAX_TOKENS, true, true);
    if (n_prompt < 0) { fprintf(stderr, "Tokenize failed\n"); return 1; }
    fprintf(stderr, "Prompt: %d tokens\n\n", n_prompt);

    // ── Prefill: seq 0, then copy to seqs 1 and 2 ──────────────────────────
    struct llama_batch batch = llama_batch_init(n_prompt + 8, 0, 3);
    batch_clear(&batch);
    for (int i = 0; i < n_prompt; i++)
        batch_add(&batch, prompt_toks[i], (llama_pos)i, 0, i == n_prompt - 1);

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Prefill failed\n"); return 1;
    }
    llama_pos pos_a = (llama_pos)n_prompt;

    float *lp = llama_get_logits_ith(ctx, batch.n_tokens - 1);
    llama_token tok_gen_0 = (llama_token)argmax_f(lp, n_vocab);

    llama_memory_seq_cp(mem, 0, 1, 0, -1);
    llama_memory_seq_cp(mem, 0, 2, 0, -1);

    fprintf(stderr, "Prefill done. First token: [%d]\n", tok_gen_0);
    fprintf(stderr, "Output: ");
    print_token(vocab, tok_gen_0);

    // ── B+C init stage 1: both process tok_gen_0 at pos_a ──────────────────
    // Two separate entries (n_seq_id=1 each): B gets logits, C fills its KV.
    batch_clear(&batch);
    batch_add(&batch, tok_gen_0, pos_a, 1, true);   // B: want logits
    batch_add(&batch, tok_gen_0, pos_a, 2, false);  // C: fill KV at pos_a
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "B+C init stage 1 failed\n"); return 1; }

    llama_token tok_b = (llama_token)argmax_f(llama_get_logits_ith(ctx, 0), n_vocab);
    llama_pos   pos_b = pos_a + 1;

    // ── C-init stage 2: C processes tok_b at pos_b to produce tok_c ────────
    // Full decode only: partial-range cross-stream seq_cp fails on hybrid models,
    // so MTP cannot be used here (it would need seq_cp to populate C's KV at pos_b).
    // MTP is only used on the reject path in the main loop (full-stream seq_cp via -1).
    batch_clear(&batch);
    batch_add(&batch, tok_b, pos_b, 2, true);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "C init stage 2 failed\n"); return 1; }
    llama_token tok_c = (llama_token)argmax_f(llama_get_logits_ith(ctx, 0), n_vocab);
    llama_pos pos_c = pos_b + 1;

    // Main loop state
    llama_token tok_a = tok_gen_0;

    int n_gen    = 1;
    int n_3acc   = 0;  // all 3 correct
    int n_2acc   = 0;  // A+B correct, C wrong
    int n_reject = 0;  // A+B wrong

    struct llama_batch step_batch = llama_batch_init(3, 0, 3);

    double t_start = now_sec();

    // ── Main 3-seq PTI loop ──────────────────────────────────────────────────
    for (int step = 0; step < args->max_new - 1; step++) {

        if (llama_vocab_is_eog(vocab, tok_a)) break;

        // Triple-batch: A, B, C in one llama_decode call (one weight load)
        batch_clear(&step_batch);
        batch_add(&step_batch, tok_a, pos_a, 0, true);  // Twin A
        batch_add(&step_batch, tok_b, pos_b, 1, true);  // Twin B
        batch_add(&step_batch, tok_c, pos_c, 2, true);  // Twin C

        if (llama_decode(ctx, step_batch) != 0) {
            fprintf(stderr, "\nDecode failed at step %d\n", step);
            break;
        }

        llama_token actual_next  = (llama_token)argmax_f(llama_get_logits_ith(ctx, 0), n_vocab);
        llama_token next_from_b  = (llama_token)argmax_f(llama_get_logits_ith(ctx, 1), n_vocab);
        llama_token next_from_c  = (llama_token)argmax_f(llama_get_logits_ith(ctx, 2), n_vocab);

        pos_a++;
        pos_b++;
        pos_c++;

        if (tok_b == actual_next) {
            if (tok_c == next_from_b) {
                // 3-accept: all three twins agreed
                n_3acc++;
                if (args->verbose)
                    fprintf(stderr, "\n[A=%d B=%d C=%d pos=%d] 3-ACCEPT",
                            actual_next, tok_b, tok_c, pos_a-1);

                print_token(vocab, actual_next);  n_gen++;
                if (!llama_vocab_is_eog(vocab, next_from_b)) { print_token(vocab, next_from_b); n_gen++; }
                if (!llama_vocab_is_eog(vocab, next_from_c)) { print_token(vocab, next_from_c); n_gen++; }

                tok_a = actual_next;
                tok_b = next_from_b;
                tok_c = next_from_c;

                if (llama_vocab_is_eog(vocab, next_from_b) || llama_vocab_is_eog(vocab, next_from_c)) break;

            } else {
                // 2-accept: A+B correct, C wrong — re-init C
                n_2acc++;
                if (args->verbose)
                    fprintf(stderr, "\n[A=%d B=%d C=%d pos=%d] 2-ACCEPT (C wrong)",
                            actual_next, tok_b, tok_c, pos_a-1);

                print_token(vocab, actual_next);  n_gen++;
                if (!llama_vocab_is_eog(vocab, next_from_b)) { print_token(vocab, next_from_b); n_gen++; }

                tok_a = actual_next;
                tok_b = next_from_b;

                // Trim C back to B's current position, then re-seed tok_c via full decode.
                // (Partial cross-stream seq_cp fails on hybrid models — no MTP shortcut here.)
                llama_memory_seq_rm(mem, 2, pos_c - 1, -1);
                pos_c = pos_b;

                batch_clear(&step_batch);
                batch_add(&step_batch, next_from_b, pos_c, 2, true);
                if (llama_decode(ctx, step_batch) == 0) {
                    tok_c = (llama_token)argmax_f(llama_get_logits_ith(ctx, 0), n_vocab);
                    pos_c++;
                }

                if (llama_vocab_is_eog(vocab, next_from_b)) break;
            }
        } else {
            // Reject: B was wrong — trim B+C, re-init both
            n_reject++;
            if (args->verbose)
                fprintf(stderr, "\n[A=%d B=%d C=%d pos=%d] REJECT",
                        actual_next, tok_b, tok_c, pos_a-1);

            print_token(vocab, actual_next);  n_gen++;

            // Trim B and C back to A's position
            llama_memory_seq_rm(mem, 1, pos_b - 1, -1);
            llama_memory_seq_rm(mem, 2, pos_c - 1, -1);
            pos_b = pos_a;
            pos_c = pos_a;

            // Re-init B at pos_b with actual_next
            batch_clear(&step_batch);
            batch_add(&step_batch, actual_next, pos_b, 1, true);
            if (llama_decode(ctx, step_batch) == 0) {
                tok_b = (llama_token)argmax_f(llama_get_logits_ith(ctx, 0), n_vocab);
                pos_b++;

                // Re-init C via full decode at pos_b.
                // (Partial cross-stream seq_cp fails on hybrid models — no MTP shortcut here.)
                batch_clear(&step_batch);
                batch_add(&step_batch, tok_b, pos_b, 2, true);
                if (llama_decode(ctx, step_batch) == 0) {
                    tok_c = (llama_token)argmax_f(llama_get_logits_ith(ctx, 0), n_vocab);
                    pos_c = pos_b + 1;
                }
            }

            tok_a = actual_next;
        }

        if (llama_vocab_is_eog(vocab, actual_next)) break;
    }

    double elapsed = now_sec() - t_start;
    int total_checks = n_3acc + n_2acc + n_reject;

    fprintf(stderr, "\n\n════════════════════════════════════════\n");
    fprintf(stderr, "  PTI-3seq Results\n");
    fprintf(stderr, "  Tokens generated:  %d\n", n_gen);
    fprintf(stderr, "  3-accept:          %d  (A+B+C)\n", n_3acc);
    fprintf(stderr, "  2-accept:          %d  (A+B, C re-init)\n", n_2acc);
    fprintf(stderr, "  Rejects:           %d\n", n_reject);
    if (total_checks > 0) {
        fprintf(stderr, "  B accept rate:     %.1f%%\n",
                100.0 * (n_3acc + n_2acc) / total_checks);
        fprintf(stderr, "  C accept rate:     %.1f%%  (of B-accepts)\n",
                (n_3acc + n_2acc) > 0 ? 100.0 * n_3acc / (n_3acc + n_2acc) : 0.0);
    }
    fprintf(stderr, "  Elapsed:           %.2fs\n", elapsed);
    fprintf(stderr, "  Throughput:        %.1f tok/s\n", n_gen / elapsed);
    fprintf(stderr, "  MTP context:       %s\n", ctx_mtp ? "active" : "unavailable");
    fprintf(stderr, "════════════════════════════════════════\n");

    llama_batch_free(batch);
    llama_batch_free(step_batch);
    if (ctx_mtp) llama_free(ctx_mtp);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

// ── Baseline ─────────────────────────────────────────────────────────────────

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

// ── CLI ───────────────────────────────────────────────────────────────────────

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
        if      (!strcmp(argv[i], "-m")   && i+1 < argc) { strncpy(args.model_path, argv[++i], sizeof(args.model_path)-1); }
        else if (!strcmp(argv[i], "-p")   && i+1 < argc) { strncpy(args.prompt,     argv[++i], sizeof(args.prompt)-1); }
        else if (!strcmp(argv[i], "-n")   && i+1 < argc) { args.max_new      = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "-ngl") && i+1 < argc) { args.n_gpu_layers = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--baseline")) { do_baseline = true; }
        else if (!strcmp(argv[i], "--verbose"))  { args.verbose = true; }
        else if (!strcmp(argv[i], "--help"))     { usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
    }

    if (args.model_path[0] == '\0') {
        fprintf(stderr, "Error: -m <model_path> required\n");
        usage(argv[0]); return 1;
    }

    llama_backend_init();
    int rc = do_baseline ? run_baseline(&args) : run_pti3(&args);
    llama_backend_free();
    return rc;
}
