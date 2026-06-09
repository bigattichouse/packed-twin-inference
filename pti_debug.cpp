/*
 * pti_debug.cpp — single-sequence generation with verbose per-token logging
 *
 * Used to verify baseline generation is correct before adding PTI.
 * Outputs: step | pos | token_id | token_str | top5_logits
 *
 * Build: make debug
 * Run:   bin/pti_debug -m ../gguf/Qwen3.6-27B-UD-Q6_K_XL.gguf -p "Hello" -n 20
 */

#include "../llama.cpp/include/llama.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <random>
#include <vector>

static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void batch_add(struct llama_batch *b, llama_token tok,
                      llama_pos pos, llama_seq_id seq, bool want_logits) {
    b->token      [b->n_tokens] = tok;
    b->pos        [b->n_tokens] = pos;
    b->n_seq_id   [b->n_tokens] = 1;
    b->seq_id     [b->n_tokens][0] = seq;
    b->logits     [b->n_tokens] = want_logits ? 1 : 0;
    b->n_tokens++;
}

static void batch_clear(struct llama_batch *b) { b->n_tokens = 0; }

static int32_t argmax_f(const float *v, int32_t n) {
    int32_t best = 0;
    for (int32_t i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    return best;
}

// Sample with temperature. Returns argmax when temp <= 0.
static llama_token sample(const float *logits, int32_t n_vocab, float temp,
                           std::mt19937 &rng) {
    if (temp <= 0.0f)
        return (llama_token)argmax_f(logits, n_vocab);

    float max_l = -FLT_MAX;
    for (int32_t i = 0; i < n_vocab; i++)
        if (logits[i] > max_l) max_l = logits[i];

    std::vector<float> probs(n_vocab);
    float sum = 0.0f;
    for (int32_t i = 0; i < n_vocab; i++) {
        probs[i] = expf((logits[i] - max_l) / temp);
        sum += probs[i];
    }
    for (int32_t i = 0; i < n_vocab; i++) probs[i] /= sum;
    std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
    return (llama_token)dist(rng);
}

static std::string tok_str(const struct llama_vocab *vocab, llama_token tok) {
    char buf[256];
    int len = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
    if (len <= 0) return "<unk>";
    std::string s(buf, len);
    // escape for log readability
    std::string out;
    for (char c : s) {
        if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out;
}

// Print top-k logits for a position
static void print_top(const struct llama_vocab *vocab, const float *logits,
                      int32_t n_vocab, int k = 3) {
    // partial sort
    std::vector<int32_t> idx(n_vocab);
    for (int32_t i = 0; i < n_vocab; i++) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    fprintf(stderr, "  top%d:", k);
    for (int i = 0; i < k; i++)
        fprintf(stderr, " [%d]%s(%.2f)", idx[i],
                tok_str(vocab, idx[i]).c_str(), logits[idx[i]]);
    fprintf(stderr, "\n");
}

int main(int argc, char **argv) {
    const char *model_path = nullptr;
    const char *prompt     = "The capital of France is";
    int         max_new    = 30;
    int         ngl        = 99;
    float       temp       = 0.0f;
    bool        verbose    = false;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-m")  && i+1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "-p")  && i+1 < argc) prompt     = argv[++i];
        else if (!strcmp(argv[i], "-n")  && i+1 < argc) max_new    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-ngl")&& i+1 < argc) ngl        = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp")&&i+1<argc)  temp       = atof(argv[++i]);
        else if (!strcmp(argv[i], "-v"))                verbose    = true;
    }

    if (!model_path) {
        fprintf(stderr, "Usage: %s -m <model> [-p <prompt>] [-n <tokens>] "
                        "[-ngl <layers>] [--temp <t>] [-v]\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    fprintf(stderr, "Loading %s ...\n", model_path);
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    const llama_vocab *vocab  = llama_model_get_vocab(model);
    int32_t            n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx        = 4096;
    cparams.n_batch      = 2048;
    cparams.n_seq_max    = 1;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.type_k       = GGML_TYPE_Q8_0;
    cparams.type_v       = GGML_TYPE_Q8_0;

    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    std::mt19937 rng(std::random_device{}());

    // ── tokenise ─────────────────────────────────────────────────────────────
    std::vector<llama_token> toks(4096);
    int n_prompt = llama_tokenize(vocab, prompt, (int32_t)strlen(prompt),
                                  toks.data(), 4096, true, true);
    if (n_prompt <= 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    fprintf(stderr, "Prompt: %d tokens\n", n_prompt);

    // ── prefill ───────────────────────────────────────────────────────────────
    llama_batch batch = llama_batch_init(n_prompt + max_new + 4, 0, 1);
    batch_clear(&batch);
    for (int i = 0; i < n_prompt; i++)
        batch_add(&batch, toks[i], (llama_pos)i, 0, i == n_prompt - 1);

    double t0 = now_sec();
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "prefill failed\n"); return 1;
    }
    fprintf(stderr, "Prefill done (%.1f ms)\n\n",
            (now_sec() - t0) * 1000.0);

    // ── generation loop ───────────────────────────────────────────────────────
    fprintf(stderr, "%-5s %-6s %-8s %-20s\n", "step", "pos", "tok_id", "token");
    fprintf(stderr, "%-5s %-6s %-8s %-20s\n", "----", "---", "------", "-----");

    llama_pos cur_pos = (llama_pos)n_prompt;
    llama_token cur_tok = -1;

    // sample first token from prefill logits
    {
        float *logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        cur_tok = sample(logits, n_vocab, temp, rng);
        fprintf(stderr, "%-5s %-6d %-8d %-20s\n",
                "pre", (int)(cur_pos - 1), cur_tok, tok_str(vocab, cur_tok).c_str());
        if (verbose) print_top(vocab, logits, n_vocab);
        // print to stdout (the actual generation output)
        char buf[256];
        int len = llama_token_to_piece(vocab, cur_tok, buf, sizeof(buf), 0, true);
        if (len > 0) fwrite(buf, 1, len, stdout);
        fflush(stdout);
    }

    double t_gen = now_sec();
    int n_gen = 1;

    for (int step = 0; step < max_new && !llama_vocab_is_eog(vocab, cur_tok); step++) {
        batch_clear(&batch);
        batch_add(&batch, cur_tok, cur_pos, 0, true);

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed at step %d\n", step);
            break;
        }

        float *logits = llama_get_logits_ith(ctx, 0);
        llama_token next = sample(logits, n_vocab, temp, rng);

        fprintf(stderr, "%-5d %-6d %-8d %-20s\n",
                step, (int)cur_pos, next, tok_str(vocab, next).c_str());
        if (verbose) print_top(vocab, logits, n_vocab);

        char buf[256];
        int len = llama_token_to_piece(vocab, next, buf, sizeof(buf), 0, true);
        if (len > 0) { fwrite(buf, 1, len, stdout); fflush(stdout); }

        cur_pos++;
        cur_tok = next;
        n_gen++;
    }

    double elapsed = now_sec() - t_gen;
    fprintf(stderr, "\n\n%d tokens  %.1f tok/s\n", n_gen, n_gen / elapsed);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
