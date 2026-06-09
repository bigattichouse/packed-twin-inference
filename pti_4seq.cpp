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
    fprintf(stderr, "Vocab: %d\n", n_vocab);

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = DEFAULT_CTX * 4;
    cparams.n_batch    = DEFAULT_CTX;
    cparams.n_seq_max  = 4;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

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

    for (int step = 0; step < args->max_new - 1; step++) {
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

                    print_token(vocab, actual_next);  n_gen++;
                    if (!llama_vocab_is_eog(vocab, next_from_b)) { print_token(vocab, next_from_b); n_gen++; }
                    if (!llama_vocab_is_eog(vocab, next_from_c)) { print_token(vocab, next_from_c); n_gen++; }
                    if (!llama_vocab_is_eog(vocab, next_from_d)) { print_token(vocab, next_from_d); n_gen++; }

                    tok_a = actual_next;
                    tok_b = next_from_b;
                    tok_c = next_from_c;
                    tok_d = next_from_d;

                    if (llama_vocab_is_eog(vocab, next_from_b) ||
                        llama_vocab_is_eog(vocab, next_from_c) ||
                        llama_vocab_is_eog(vocab, next_from_d)) break;

                } else {
                    // ── 3-accept: D wrong, re-init D ──────────────────────
                    n_3acc++;
                    if (args->verbose)
                        fprintf(stderr, "\n[3-ACCEPT pos=%d]", pos_a-1);

                    print_token(vocab, actual_next);  n_gen++;
                    if (!llama_vocab_is_eog(vocab, next_from_b)) { print_token(vocab, next_from_b); n_gen++; }
                    if (!llama_vocab_is_eog(vocab, next_from_c)) { print_token(vocab, next_from_c); n_gen++; }

                    tok_a = actual_next;
                    tok_b = next_from_b;
                    tok_c = next_from_c;

                    llama_memory_seq_rm(mem, 3, pos_d - 1, -1);
                    pos_d = pos_c;

                    batch_clear(&step_batch);
                    batch_add(&step_batch, next_from_c, pos_d, 3, true);
                    if (llama_decode(ctx, step_batch) == 0) {
                        tok_d = (llama_token)argmax_f(llama_get_logits_ith(ctx, 0), n_vocab);
                        pos_d++;
                    }

                    if (llama_vocab_is_eog(vocab, next_from_b) ||
                        llama_vocab_is_eog(vocab, next_from_c)) break;
                }
            } else {
                // ── 2-accept: C+D wrong, re-init C and D ──────────────────
                n_2acc++;
                if (args->verbose)
                    fprintf(stderr, "\n[2-ACCEPT pos=%d]", pos_a-1);

                print_token(vocab, actual_next);  n_gen++;
                if (!llama_vocab_is_eog(vocab, next_from_b)) { print_token(vocab, next_from_b); n_gen++; }

                tok_a = actual_next;
                tok_b = next_from_b;

                llama_memory_seq_rm(mem, 2, pos_c - 1, -1);
                llama_memory_seq_rm(mem, 3, pos_b, -1);   // trim to pos_b-1; D stage1 rebuilds from pos_b
                pos_c = pos_b;
                pos_d = pos_b;

                batch_clear(&step_batch);
                batch_add(&step_batch, next_from_b, pos_c, 2, true);
                if (llama_decode(ctx, step_batch) == 0) {
                    tok_c = (llama_token)argmax_f(llama_get_logits_ith(ctx, 0), n_vocab);
                    pos_c++;
                }

                // D re-init: process next_from_b at pos_b, then tok_c at pos_b+1
                batch_clear(&step_batch);
                batch_add(&step_batch, next_from_b, pos_d, 3, false);
                if (llama_decode(ctx, step_batch) == 0) {
                    pos_d++;
                }
                batch_clear(&step_batch);
                batch_add(&step_batch, tok_c, pos_d, 3, true);
                if (llama_decode(ctx, step_batch) == 0) {
                    tok_d = (llama_token)argmax_f(llama_get_logits_ith(ctx, 0), n_vocab);
                    pos_d++;
                }

                if (llama_vocab_is_eog(vocab, next_from_b)) break;
            }
        } else {
            // ── Reject: B wrong, re-init B, C, D ─────────────────────────
            n_reject++;
            if (args->verbose)
                fprintf(stderr, "\n[REJECT pos=%d]", pos_a-1);

            print_token(vocab, actual_next);  n_gen++;

            llama_memory_seq_rm(mem, 1, pos_b - 1, -1);
            llama_memory_seq_rm(mem, 2, pos_c - 1, -1);
            llama_memory_seq_rm(mem, 3, pos_a, -1);       // trim seq3 to confirmed prefix (0..pos_a-1)
            pos_b = pos_a; pos_c = pos_a; pos_d = pos_a;

            batch_clear(&step_batch);
            batch_add(&step_batch, actual_next, pos_b, 1, true);
            if (llama_decode(ctx, step_batch) == 0) {
                tok_b = (llama_token)argmax_f(llama_get_logits_ith(ctx, 0), n_vocab);
                pos_b++;

                batch_clear(&step_batch);
                batch_add(&step_batch, tok_b, pos_b, 2, true);
                if (llama_decode(ctx, step_batch) == 0) {
                    tok_c = (llama_token)argmax_f(llama_get_logits_ith(ctx, 0), n_vocab);
                    pos_c = pos_b + 1;
                }

                // D stage0: replay actual_next on seq3 (fills pos_a; seq3 was trimmed to pos_a-1)
                batch_clear(&step_batch);
                batch_add(&step_batch, actual_next, pos_d, 3, false);
                if (llama_decode(ctx, step_batch) == 0) {
                    pos_d++;
                }
                batch_clear(&step_batch);
                batch_add(&step_batch, tok_b, pos_b, 3, false);
                if (llama_decode(ctx, step_batch) == 0) {
                    pos_d = pos_b + 1;
                }
                batch_clear(&step_batch);
                batch_add(&step_batch, tok_c, pos_d, 3, true);
                if (llama_decode(ctx, step_batch) == 0) {
                    tok_d = (llama_token)argmax_f(llama_get_logits_ith(ctx, 0), n_vocab);
                    pos_d++;
                }
            }
            tok_a = actual_next;
        }

        if (llama_vocab_is_eog(vocab, actual_next)) break;
    }

    double elapsed = now_sec() - t_start;
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
    fprintf(stderr, "  Elapsed:           %.2fs\n", elapsed);
    fprintf(stderr, "  Throughput:        %.1f tok/s\n", n_gen / elapsed);
    fprintf(stderr, "════════════════════════════════════════\n");

    llama_batch_free(batch);
    llama_batch_free(step_batch);
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
