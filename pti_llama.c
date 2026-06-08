/*
 * pti_llama.c — Packed Twin Inference via llama.cpp
 *
 * Runs PTI on any GGUF model using one context with two sequences:
 *   Seq 0 (Twin A): verifier at position n
 *   Seq 1 (Twin B): drafter at position n+1  (one step ahead)
 *
 * Batches A and B tokens together each step, sharing weight loads.
 * On reject: trims B's KV back to A's position, re-runs B for new speculation.
 *
 * Build:
 *   gcc -O2 -o pti_llama pti_llama.c \
 *       -I../llama.cpp/include \
 *       -L../llama.cpp/build/src -lllama \
 *       -L../llama.cpp/build/ggml/src -lggml \
 *       -Wl,-rpath,'$ORIGIN/../llama.cpp/build/src' \
 *       -Wl,-rpath,'$ORIGIN/../llama.cpp/build/ggml/src' \
 *       -lm -lstdc++
 *
 * Usage:
 *   ./pti_llama -m ../gguf/Qwen3.6-27B-Q5_K_M.gguf -p "The speed of light" -n 80 -ngl 99
 *   ./pti_llama -m ../gguf/Qwen3.6-27B-Q5_K_M.gguf -p "Write SVG of a rainbow" -n 200
 */

#include "../llama.cpp/include/llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <float.h>

#define MAX_TOKENS   4096
#define DEFAULT_CTX  2048

// ── Utility ──────────────────────────────────────────────────────────────────

static double now_sec(void) {
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

// ── PTI inference loop ────────────────────────────────────────────────────────

typedef struct {
    int    max_new;
    int    n_gpu_layers;
    bool   verbose;
    bool   think_mode;   // enable thinking (no-think suppression skipped)
    char   model_path[512];
    char   prompt[4096];
} PTIArgs;

static int run_pti(const PTIArgs *args) {

    // ── Load model ──────────────────────────────────────────────────────────
    fprintf(stderr, "\nLoading model: %s\n", args->model_path);
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args->n_gpu_layers;

    struct llama_model *model = llama_model_load_from_file(args->model_path, mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);
    fprintf(stderr, "Vocab size: %d\n", n_vocab);

    // ── Create main context (two sequences for PTI) ─────────────────────────
    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = DEFAULT_CTX * 2;  // room for both sequences
    cparams.n_batch    = DEFAULT_CTX;
    cparams.n_seq_max  = 2;                // seq 0 = A, seq 1 = B
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    // MTP note: llama.cpp's MTP draft requires feeding hidden states from the
    // main context into the MTP head via llama_set_embeddings_pre_norm(), which
    // is not in the public C API. MTP integration requires C++ + common/speculative.cpp.

    llama_memory_t mem = llama_get_memory(ctx);

    // ── Tokenize prompt ─────────────────────────────────────────────────────
    llama_token prompt_toks[MAX_TOKENS];
    int n_prompt = llama_tokenize(vocab, args->prompt, (int32_t)strlen(args->prompt),
                                  prompt_toks, MAX_TOKENS, true, true);
    if (n_prompt < 0) { fprintf(stderr, "Tokenize failed\n"); return 1; }
    fprintf(stderr, "Prompt: %d tokens\n\n", n_prompt);

    // ── Prefill (batch all prompt tokens, seq 0 only) ───────────────────────
    // We'll copy seq 0 → seq 1 after prefill
    struct llama_batch batch = llama_batch_init(n_prompt + 4, 0, 2);
    batch_clear(&batch);
    for (int i = 0; i < n_prompt; i++) {
        bool want_logits = (i == n_prompt - 1);  // only last token's logits
        batch_add(&batch, prompt_toks[i], (llama_pos)i, 0, want_logits);
    }
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Prefill decode failed\n"); return 1;
    }

    llama_pos pos_a = (llama_pos)n_prompt;  // A's next write position

    // First generated token from prefill logits
    float *logits0 = llama_get_logits_ith(ctx, batch.n_tokens - 1);
    llama_token tok_gen_0 = (llama_token)argmax_f(logits0, n_vocab);

    // Copy seq 0's KV cache into seq 1
    llama_memory_seq_cp(mem, 0, 1, 0, -1);

    fprintf(stderr, "Prefill done. First token: [%d]\n", tok_gen_0);
    fprintf(stderr, "Output: ");

    print_token(vocab, tok_gen_0);

    // ── B gets one step ahead ────────────────────────────────────────────────
    // B processes tok_gen_0 at pos_a (seq 1) to produce speculation_1
    batch_clear(&batch);
    batch_add(&batch, tok_gen_0, pos_a, 1, true);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "B init failed\n"); return 1; }

    float *logits_b_init = llama_get_logits_ith(ctx, 0);
    llama_token tok_b    = (llama_token)argmax_f(logits_b_init, n_vocab);
    llama_pos   pos_b    = pos_a + 1;  // B is one ahead

    // Main PTI loop state:
    //   tok_a = the confirmed token A needs to process next
    //   tok_b = the speculation B will process next
    //   pos_a = A's next write position
    //   pos_b = B's next write position (= pos_a + 1)
    llama_token tok_a = tok_gen_0;

    int n_gen     = 1;  // tok_gen_0 already emitted
    int n_accept  = 0;
    int n_reject  = 0;

    // 2-token batch: A + B
    struct llama_batch step_batch = llama_batch_init(2, 0, 2);

    double t_start = now_sec();

    // ── Main PTI loop ────────────────────────────────────────────────────────
    for (int step = 0; step < args->max_new - 1; step++) {

        // Check end of generation
        if (llama_vocab_is_eog(vocab, tok_a)) break;

        // ── Dual-stream decode: A at pos_a, B at pos_b ──────────────────────
        // (Two tokens in one llama_decode call — shared weight loads)
        batch_clear(&step_batch);
        batch_add(&step_batch, tok_a, pos_a, 0, true);  // Twin A
        batch_add(&step_batch, tok_b, pos_b, 1, true);  // Twin B

        if (llama_decode(ctx, step_batch) != 0) {
            fprintf(stderr, "\nDecode failed at step %d\n", step);
            break;
        }

        float *la = llama_get_logits_ith(ctx, 0);
        float *lb = llama_get_logits_ith(ctx, 1);

        llama_token actual_next = (llama_token)argmax_f(la, n_vocab);
        llama_token next_from_b = (llama_token)argmax_f(lb, n_vocab);

        pos_a++;
        pos_b++;

        // ── Verify ───────────────────────────────────────────────────────────
        if (tok_b == actual_next) {
            // ACCEPT: B's speculation was correct.
            // B ran on the correct context with identical weights, so next_from_b
            // is also confirmed — emit both tokens this step.
            n_accept++;
            if (args->verbose)
                fprintf(stderr, "\n[A pos=%d tok=%d] [B pos=%d tok=%d] ACCEPT",
                        pos_a-1, actual_next, pos_b-1, tok_b);

            print_token(vocab, actual_next);
            n_gen++;

            if (!llama_vocab_is_eog(vocab, next_from_b)) {
                print_token(vocab, next_from_b);
                n_gen++;
            }

            tok_a = actual_next;
            tok_b = next_from_b;

            if (llama_vocab_is_eog(vocab, next_from_b)) break;

        } else {
            // REJECT: B's speculation was wrong
            n_reject++;
            if (args->verbose)
                fprintf(stderr, "\n[A pos=%d tok=%d] [B pos=%d tok=%d] REJECT",
                        pos_a-1, actual_next, pos_b-1, tok_b);

            print_token(vocab, actual_next);
            n_gen++;

            // Trim B back: remove the wrong token from seq 1
            llama_memory_seq_rm(mem, 1, pos_b - 1, -1);
            pos_b = pos_a;

            // Re-run B at pos_b to produce correct speculation
            batch_clear(&step_batch);
            batch_add(&step_batch, actual_next, pos_b, 1, true);
            if (llama_decode(ctx, step_batch) == 0) {
                float *lb2 = llama_get_logits_ith(ctx, 0);
                tok_b = (llama_token)argmax_f(lb2, n_vocab);
                pos_b++;
            }

            tok_a = actual_next;
        }

        if (llama_vocab_is_eog(vocab, actual_next)) break;
    }

    double elapsed = now_sec() - t_start;

    // ── Report ────────────────────────────────────────────────────────────────
    fprintf(stderr, "\n\n════════════════════════════════════════\n");
    fprintf(stderr, "  PTI Results\n");
    fprintf(stderr, "  Tokens generated:  %d\n", n_gen);
    fprintf(stderr, "  Accepts:           %d\n", n_accept);
    fprintf(stderr, "  Rejects:           %d\n", n_reject);
    int total_checks = n_accept + n_reject;
    double accept_rate = total_checks > 0 ? (double)n_accept / total_checks : 0.0;
    fprintf(stderr, "  Acceptance rate:   %.1f%%\n", accept_rate * 100.0);
    fprintf(stderr, "  Elapsed:           %.2fs\n", elapsed);
    double tok_per_s = n_gen / elapsed;
    fprintf(stderr, "  Throughput:        %.1f tok/s\n", tok_per_s);
    fprintf(stderr, "  Theoretical gain:  %.2f×  (on fused SSQ kernel)\n",
            1.0 + accept_rate);
    fprintf(stderr, "════════════════════════════════════════\n");

    // Cleanup
    llama_batch_free(batch);
    llama_batch_free(step_batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

// ── Baseline (single sequence) for comparison ─────────────────────────────────

static int run_baseline(const PTIArgs *args) {
    fprintf(stderr, "\nLoading model (baseline): %s\n", args->model_path);
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args->n_gpu_layers;

    struct llama_model *model = llama_model_load_from_file(args->model_path, mparams);
    if (!model) return 1;

    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = DEFAULT_CTX;
    cparams.n_batch   = DEFAULT_CTX;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) return 1;

    llama_token prompt_toks[MAX_TOKENS];
    int n_prompt = llama_tokenize(vocab, args->prompt, (int32_t)strlen(args->prompt),
                                  prompt_toks, MAX_TOKENS, true, true);

    struct llama_batch batch = llama_batch_init(n_prompt + 2, 0, 1);
    batch_clear(&batch);
    for (int i = 0; i < n_prompt; i++)
        batch_add(&batch, prompt_toks[i], (llama_pos)i, 0, i == n_prompt-1);

    if (llama_decode(ctx, batch) != 0) return 1;

    llama_pos pos = (llama_pos)n_prompt;
    float *lp = llama_get_logits_ith(ctx, batch.n_tokens - 1);
    llama_token tok = (llama_token)argmax_f(lp, n_vocab);

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
        float *l = llama_get_logits_ith(ctx, 0);
        tok = (llama_token)argmax_f(l, n_vocab);
        print_token(vocab, tok);
        n_gen++;
    }

    double elapsed = now_sec() - t0;
    fprintf(stderr, "\n\n════════════════════════════════════════\n");
    fprintf(stderr, "  Baseline Results\n");
    fprintf(stderr, "  Tokens generated:  %d\n", n_gen);
    fprintf(stderr, "  Elapsed:           %.2fs\n", elapsed);
    fprintf(stderr, "  Throughput:        %.1f tok/s\n", n_gen / elapsed);
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
        "  -p <text>    Prompt text      (default: \"The meaning of life is\")\n"
        "  -n <int>     Max new tokens   (default: 80)\n"
        "  -ngl <int>   GPU layers       (default: 99)\n"
        "  --baseline   Run baseline (single stream) instead of PTI\n"
        "  --verbose    Print accept/reject per step\n"
        "  --help\n",
        prog);
}

int main(int argc, char **argv) {
    PTIArgs args = {
        .max_new      = 80,
        .n_gpu_layers = 99,
        .verbose      = false,
        .think_mode   = false,
    };
    strncpy(args.model_path, "", sizeof(args.model_path));
    strncpy(args.prompt, "The meaning of life is", sizeof(args.prompt));

    bool do_baseline = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i+1 < argc) {
            strncpy(args.model_path, argv[++i], sizeof(args.model_path)-1);
        } else if (strcmp(argv[i], "-p") == 0 && i+1 < argc) {
            strncpy(args.prompt, argv[++i], sizeof(args.prompt)-1);
        } else if (strcmp(argv[i], "-n") == 0 && i+1 < argc) {
            args.max_new = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-ngl") == 0 && i+1 < argc) {
            args.n_gpu_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--baseline") == 0) {
            do_baseline = true;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            args.verbose = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]); return 0;
        } else {
            fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            usage(argv[0]); return 1;
        }
    }

    if (args.model_path[0] == '\0') {
        fprintf(stderr, "Error: -m <model_path> is required\n");
        usage(argv[0]); return 1;
    }

    llama_backend_init();

    int rc = do_baseline ? run_baseline(&args) : run_pti(&args);

    llama_backend_free();
    return rc;
}
