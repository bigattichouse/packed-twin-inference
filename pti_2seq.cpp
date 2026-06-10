/*
 * pti_2seq.cpp — Packed Twin Inference, 2-sequence
 *
 *   Seq 0 (A): verifier at pos_a
 *   Seq 1 (B): 1-step drafter at pos_b = pos_a + 1
 *
 * One llama_decode per step with a 2-token batch.
 * On 2-accept: emit 2 tokens + 1 reinit_seq call to rebuild B.
 * On reject:   emit 1 token  + 1 reinit_seq call to rebuild B.
 *
 * Cost model (MI50, Qwen3.6-27B, 19.0 tok/s baseline):
 *   dual-decode (N=2): 66.2 ms  (1.25× baseline)
 *   1 reinit call:     52.6 ms
 *   ─────────────────────────────────────────────
 *   2-accept:  119 ms / 2 tok = 59.5 ms/tok → 16.8 tok/s  (0.88× baseline)
 *   reject:    119 ms / 1 tok = 119 ms/tok  →  8.4 tok/s
 *
 * At 100% greedy (all 2-accept): ~16.8 tok/s
 *   vs 4-seq (all 4-accept): ~14.6 tok/s
 *   vs baseline:             19.0 tok/s
 *
 * Build:
 *   g++ -O2 -std=c++17 -o bin/pti_2seq pti_2seq.cpp \
 *       -I../llama.cpp/include -I../llama.cpp/ggml/include \
 *       -L../llama.cpp/build/bin -lllama \
 *       -Wl,-rpath,../llama.cpp/build/bin -lm
 */

#include "../llama.cpp/include/llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
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

struct PTI2Args {
    int  max_new      = 80;
    int  n_gpu_layers = 99;
    bool verbose      = false;
    char model_path[512] = {};
    char prompt[4096]    = {};
};

static int run_pti2(const PTI2Args *args) {
    fprintf(stderr, "\nLoading model: %s\n", args->model_path);

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args->n_gpu_layers;
    struct llama_model *model = llama_model_load_from_file(args->model_path, mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);
    fprintf(stderr, "Vocab: %d\n", n_vocab);

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = DEFAULT_CTX * 2;
    cparams.n_batch    = DEFAULT_CTX;
    cparams.n_seq_max  = 2;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    llama_memory_t mem = llama_get_memory(ctx);

    llama_token prompt_toks[MAX_TOKENS];
    int n_prompt = llama_tokenize(vocab, args->prompt, (int32_t)strlen(args->prompt),
                                  prompt_toks, MAX_TOKENS, true, true);
    if (n_prompt < 0) { fprintf(stderr, "Tokenize failed\n"); return 1; }
    fprintf(stderr, "Prompt: %d tokens\n\n", n_prompt);

    // ── Prefill under seq 0, copy to seq 1 ──────────────────────────────────
    struct llama_batch batch = llama_batch_init(n_prompt + 4, 0, 2);
    batch_clear(&batch);
    for (int i = 0; i < n_prompt; i++)
        batch_add(&batch, prompt_toks[i], (llama_pos)i, 0, i == n_prompt - 1);

    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "Prefill failed\n"); return 1; }
    llama_pos pos_a = (llama_pos)n_prompt;

    float *lp = llama_get_logits_ith(ctx, batch.n_tokens - 1);
    llama_token tok_gen_0 = (llama_token)argmax_f(lp, n_vocab);
    llama_memory_seq_cp(mem, 0, 1, 0, -1);

    fprintf(stderr, "Prefill done. First token: [%d]\n", tok_gen_0);
    fprintf(stderr, "Output: ");
    print_token(vocab, tok_gen_0);

    double t_gen_start = now_sec();

    // ── Prime seq0 for SSM cell exclusivity (same as pti_4seq) ──────────────
    batch_clear(&batch);
    batch_add(&batch, tok_gen_0, pos_a, 0, true);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "seq0 prime failed\n"); return 1; }
    llama_memory_seq_rm(mem, 0, 0, -1);
    llama_memory_seq_cp(mem, 1, 0, 0, -1);

    // ── B init: decode tok_gen_0 at pos_a in seq 1 ──────────────────────────
    batch_clear(&batch);
    batch_add(&batch, tok_gen_0, pos_a, 1, true);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "B init failed\n"); return 1; }
    llama_token tok_b = (llama_token)argmax_f(llama_get_logits_ith(ctx, 0), n_vocab);
    llama_pos   pos_b = pos_a + 1;

    llama_token tok_a = tok_gen_0;

    int n_gen    = 1;
    int n_2acc   = 0;
    int n_reject = 0;

    struct llama_batch step_batch = llama_batch_init(2, 0, 2);
    double t_start = now_sec();

    for (int step = 0; step < args->max_new - 1; step++) {
        if (n_gen >= args->max_new) break;
        if (llama_vocab_is_eog(vocab, tok_a)) break;

        batch_clear(&step_batch);
        batch_add(&step_batch, tok_a, pos_a, 0, true);
        batch_add(&step_batch, tok_b, pos_b, 1, true);

        if (llama_decode(ctx, step_batch) != 0) {
            fprintf(stderr, "\nDecode failed at step %d\n", step);
            break;
        }

        llama_token actual_next  = (llama_token)argmax_f(llama_get_logits_ith(ctx, 0), n_vocab);
        llama_token next_from_b  = (llama_token)argmax_f(llama_get_logits_ith(ctx, 1), n_vocab);

        pos_a++; pos_b++;

        if (tok_b == actual_next) {
            // ── 2-accept ─────────────────────────────────────────────────────
            n_2acc++;
            if (args->verbose) fprintf(stderr, "\n[2-ACCEPT pos=%d]", pos_a-1);

            bool full_emit = (n_gen + 2 <= args->max_new)
                          && !llama_vocab_is_eog(vocab, actual_next)
                          && !llama_vocab_is_eog(vocab, next_from_b);

            if (full_emit) {
                print_token(vocab, actual_next);  n_gen++;
                print_token(vocab, next_from_b);  n_gen++;
                // Rebuild: A inherits B's confirmed state, reinit B from A
                llama_memory_seq_rm(mem, 0, 0, -1);
                llama_memory_seq_cp(mem, 1, 0, 0, -1);
                tok_a = next_from_b;
                pos_a = pos_b;
                tok_b = reinit_seq(mem, ctx, &step_batch, n_vocab, 0, 1, next_from_b, pos_b, &pos_b);
            } else {
                print_token(vocab, actual_next);  n_gen++;
                tok_a = actual_next;
                tok_b = next_from_b;
            }
        } else {
            // ── Reject: rebuild B from A's confirmed state ────────────────────
            n_reject++;
            if (args->verbose) fprintf(stderr, "\n[REJECT pos=%d]", pos_a-1);

            print_token(vocab, actual_next);  n_gen++;
            tok_b = reinit_seq(mem, ctx, &step_batch, n_vocab, 0, 1, actual_next, pos_a, &pos_b);
            tok_a = actual_next;
        }

        if (llama_vocab_is_eog(vocab, actual_next)) break;
    }

    double elapsed      = now_sec() - t_start;
    double t_init_cost  = t_start - t_gen_start;
    double elapsed_full = elapsed + t_init_cost;
    int total_checks = n_2acc + n_reject;

    fprintf(stderr, "\n\n════════════════════════════════════════\n");
    fprintf(stderr, "  PTI-2seq Results\n");
    fprintf(stderr, "  Tokens generated:  %d\n", n_gen);
    fprintf(stderr, "  2-accept:          %d\n", n_2acc);
    fprintf(stderr, "  Rejects:           %d\n", n_reject);
    if (total_checks > 0)
        fprintf(stderr, "  B accept rate:     %.1f%%\n",
                100.0 * n_2acc / total_checks);
    fprintf(stderr, "  Init overhead:     %.0f ms\n", t_init_cost * 1000.0);
    fprintf(stderr, "  Steady-state:      %.1f tok/s  (loop only)\n", n_gen / elapsed);
    fprintf(stderr, "  Amortized:         %.1f tok/s  (init + loop)\n", n_gen / elapsed_full);
    fprintf(stderr, "════════════════════════════════════════\n");

    llama_batch_free(batch);
    llama_batch_free(step_batch);
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
        "  --verbose    Print accept/reject per step\n",
        prog);
}

int main(int argc, char **argv) {
    PTI2Args args;
    strncpy(args.prompt, "The meaning of life is", sizeof(args.prompt) - 1);

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-m")   && i+1 < argc) strncpy(args.model_path, argv[++i], sizeof(args.model_path)-1);
        else if (!strcmp(argv[i], "-p")   && i+1 < argc) strncpy(args.prompt,     argv[++i], sizeof(args.prompt)-1);
        else if (!strcmp(argv[i], "-n")   && i+1 < argc) args.max_new      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-ngl") && i+1 < argc) args.n_gpu_layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--verbose"))  args.verbose = true;
        else if (!strcmp(argv[i], "--help"))     { usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
    }

    if (args.model_path[0] == '\0') {
        fprintf(stderr, "Error: -m <model_path> required\n");
        usage(argv[0]); return 1;
    }

    llama_backend_init();
    int rc = run_pti2(&args);
    llama_backend_free();
    return rc;
}
