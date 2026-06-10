/*
 * pti_lookup.cpp — M6.4: n-gram lookup speculative decoding for hybrid-SSM
 *
 * Single stream, single model. Drafts cost ZERO model decodes: candidate
 * tokens are copied from earlier occurrences of the current n-gram in the
 * token history (prompt-lookup decoding). A batch of (1 + k) tokens verifies
 * all drafts in ONE llama_decode (measured cost: 1.20×@b2, 1.71×@b4, 3.11×@b8
 * vs 4.0×/8.0× sequential — see KERNEL_PLAN.md M6.0).
 *
 * Why this escapes the PTI ceiling proof: the proof assumes the draft costs
 * one full model decode per token (the stagger). A string match costs nothing,
 * so accepted drafts are pure profit.
 *
 * Output is BYTE-IDENTICAL to greedy baseline by construction: every emitted
 * token is the argmax of a logit row whose input prefix is fully verified.
 * Bad drafts cost time, never correctness.
 *
 * SSM rollback (the reason llama.cpp's stock lookup can't do this on hybrid
 * models): recurrent state cannot be rewound per-position. We keep a
 * checkpoint sequence (seq 1) and on partial accept rebuild seq 0 with one
 * batched re-decode of the accepted prefix — the trick proven in pti_2seq.
 *
 *   seq 0: working   seq 1: checkpoint S(p) = state with tokens < p consumed
 *
 *   loop:
 *     draft[0..k-1] = ngram_lookup(history)          // CPU, free; k=0 if no hit
 *     if k>0: seq_cp(0→1)                            // refresh checkpoint (0.02 ms)
 *     decode [tok_last@p, d0@p+1 .. dk-1@p+k]        // ONE call, logits on all
 *     a_i = argmax(logits_i); emit a_0..a_e-1 where e = 1 + accepted prefix len
 *     if e == 1+k: state clean (all consumed tokens were correct)
 *     else:        seq_cp(1→0); re-decode the e accepted tokens (no logits)
 *
 * Flags:
 *   --baseline   plain single-token loop (for byte-diff + tok/s reference)
 *   --sabotage   corrupt every draft token: output must STILL be identical
 *   -k N         max draft tokens per step (default 3 → batch of 4)
 *   -g N         max n-gram match length (default 3, falls back to 2)
 *
 * Build:  make lookup
 */

#include "../llama.cpp/include/llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#define MAX_TOKENS   8192
#define DEFAULT_CTX  2048
#define MAX_DRAFT    15

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

// ── n-gram lookup draft ───────────────────────────────────────────────────────
// Probe: match the last g tokens of history at an earlier position (most
// recent first). Gate: extend the match BACKWARD to its full suffix length L.
// Fire only when L ≥ L_min — a long suffix match means we are inside a
// verbatim repeated span (copy-run), where the continuation is reliable.
// Short matches (L≈g) are coincidence: measured 33% token-accept on code
// text, and partial accepts cost more than they save (rebuild penalty).
static int ngram_draft(const llama_token *hist, int n_hist,
                       int g, int L_min, int k, llama_token *out) {
    if (n_hist < g + 1) return 0;
    const llama_token *tail = hist + n_hist - g;
    for (int m = n_hist - 2; m >= g - 1; m--) {         // m = end index of candidate match
        int match = 1;
        for (int t = 0; t < g; t++) {
            if (hist[m - g + 1 + t] != tail[t]) { match = 0; break; }
        }
        if (!match) continue;

        // extend backwards: L = total suffix match length around m
        int L = g;
        while (m - L >= 0 && hist[m - L] == hist[n_hist - 1 - L]) L++;
        if (L < L_min) continue;                        // weak evidence — keep scanning

        int avail   = n_hist - 1 - m;                   // tokens after the match
        int n_draft = avail < k ? avail : k;
        if (n_draft < 1) continue;
        for (int t = 0; t < n_draft; t++) out[t] = hist[m + 1 + t];
        return n_draft;
    }
    return 0;
}

struct LookupArgs {
    int  max_new      = 120;
    int  n_gpu_layers = 99;
    int  draft_k      = 7;   // long drafts: only confident fires happen, and
                             // batch-8 hit-runs amortize best (3.11× for 8 tok)
    int  ngram_g      = 3;   // probe length
    int  ngram_L      = 5;   // min suffix-match length to fire (copy-run gate)
    bool baseline     = false;
    bool sabotage     = false;
    bool verbose      = false;
    char model_path[512] = {};
    char prompt[4096]    = {};
};

static int run_lookup(const LookupArgs *args) {
    fprintf(stderr, "\nLoading model: %s\n", args->model_path);

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args->n_gpu_layers;
    struct llama_model *model = llama_model_load_from_file(args->model_path, mparams);
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

    static llama_token hist[MAX_TOKENS];
    int n_hist = llama_tokenize(vocab, args->prompt, (int32_t)strlen(args->prompt),
                                hist, MAX_TOKENS, true, true);
    if (n_hist < 0) { fprintf(stderr, "Tokenize failed\n"); return 1; }
    fprintf(stderr, "Prompt: %d tokens   mode: %s%s   k=%d g=%d L>=%d\n\n",
            n_hist, args->baseline ? "BASELINE" : "LOOKUP",
            args->sabotage ? "+SABOTAGE" : "", args->draft_k, args->ngram_g, args->ngram_L);

    // ── Prefill seq 0 ────────────────────────────────────────────────────────
    struct llama_batch batch = llama_batch_init(n_hist + MAX_DRAFT + 2, 0, 2);
    batch_clear(&batch);
    for (int i = 0; i < n_hist; i++)
        batch_add(&batch, hist[i], (llama_pos)i, 0, i == n_hist - 1);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "Prefill failed\n"); return 1; }

    llama_pos   p        = (llama_pos)n_hist;
    llama_token tok_last = (llama_token)argmax_f(llama_get_logits_ith(ctx, batch.n_tokens - 1), n_vocab);
    hist[n_hist++] = tok_last;

    fprintf(stderr, "Output: ");
    print_token(vocab, tok_last);
    int n_gen = 1;

    // stats
    int n_steps = 0, n_drafted_steps = 0, n_rebuilds = 0;
    int n_accept_hist[MAX_DRAFT + 2] = {};   // index = accepted drafts in a drafted step
    int n_draft_tok = 0, n_draft_acc = 0;

    double t_start = now_sec();

    if (args->baseline) {
        // ── plain greedy loop ────────────────────────────────────────────────
        while (n_gen < args->max_new && !llama_vocab_is_eog(vocab, tok_last)) {
            batch_clear(&batch);
            batch_add(&batch, tok_last, p, 0, true);
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "\nDecode failed\n"); break; }
            tok_last = (llama_token)argmax_f(llama_get_logits_ith(ctx, 0), n_vocab);
            p++;
            print_token(vocab, tok_last);
            hist[n_hist++] = tok_last;
            n_gen++; n_steps++;
        }
    } else {
        // ── lookup speculative loop ──────────────────────────────────────────
        // AIMD confidence gate: every non-full fire raises the suffix-match
        // bar (+4); every full accept decays it (−1, floor L_min). True
        // copy-runs have L in the tens and clear any bar; parallel-structure
        // text (long shared phrases, divergent continuations) ratchets the
        // bar up after a couple of misses and stops firing → baseline parity.
        int L_dyn = args->ngram_L;
        while (n_gen < args->max_new && !llama_vocab_is_eog(vocab, tok_last)) {
            llama_token draft[MAX_DRAFT];
            int k = ngram_draft(hist, n_hist, args->ngram_g, L_dyn, args->draft_k, draft);
            if (k + 1 > args->max_new - n_gen) k = args->max_new - n_gen - 1;
            if (k < 0) k = 0;
            if (args->sabotage) {
                for (int i = 0; i < k; i++) draft[i] = (draft[i] + 1) % n_vocab;
            }

            if (k > 0) {
                // refresh checkpoint: seq1 = S(p)
                llama_memory_seq_rm(mem, 1, 0, -1);
                llama_memory_seq_cp(mem, 0, 1, 0, -1);
            }

            batch_clear(&batch);
            batch_add(&batch, tok_last, p, 0, true);
            for (int i = 0; i < k; i++)
                batch_add(&batch, draft[i], p + 1 + i, 0, true);

            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "\nDecode failed\n"); break; }
            n_steps++;
            if (k > 0) { n_drafted_steps++; n_draft_tok += k; }

            // verify: emit a_0..a_{e-1}; a_i valid while the draft prefix matches
            llama_token tok_last_old = tok_last;
            int e = 0;
            bool stop = false;
            for (int i = 0; i <= k; i++) {
                llama_token a = (llama_token)argmax_f(llama_get_logits_ith(ctx, i), n_vocab);
                print_token(vocab, a);
                hist[n_hist++] = a;
                n_gen++; e++;
                tok_last = a;
                if (llama_vocab_is_eog(vocab, a) || n_gen >= args->max_new) { stop = true; break; }
                if (i < k && draft[i] != a) break;   // a is the correction; rest invalid
            }
            int acc = e - 1;                          // accepted drafts this step
            if (k > 0) {
                n_accept_hist[acc < MAX_DRAFT+1 ? acc : MAX_DRAFT+1]++;
                n_draft_acc += acc;
                if (acc == k) L_dyn = L_dyn > args->ngram_L ? L_dyn - 1 : args->ngram_L;
                else          L_dyn += 4;
            }
            if (args->verbose && k > 0)
                fprintf(stderr, "\n[draft k=%d acc=%d L_dyn=%d]", k, acc, L_dyn);

            if (stop) { p += e; break; }

            if (e == k + 1 || k == 0) {
                // full accept (or plain step): seq0 consumed exactly the
                // verified tokens → state is clean S(p+e)
                p += e;
            } else {
                // partial: seq0 consumed wrong tail tokens → rebuild from checkpoint
                n_rebuilds++;
                llama_memory_seq_rm(mem, 0, 0, -1);
                llama_memory_seq_cp(mem, 1, 0, 0, -1);      // S(p)
                batch_clear(&batch);
                batch_add(&batch, tok_last_old, p, 0, false);
                for (int i = 0; i < e - 1; i++)              // accepted drafts only
                    batch_add(&batch, draft[i], p + 1 + i, 0, false);
                if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "\nRebuild failed\n"); break; }
                p += e;                                      // seq0 = S(p+e) clean
            }
        }
    }

    double elapsed = now_sec() - t_start;

    fprintf(stderr, "\n\n════════════════════════════════════════════════════\n");
    fprintf(stderr, "  PTI-lookup results (%s%s)\n",
            args->baseline ? "baseline" : "lookup", args->sabotage ? "+sabotage" : "");
    fprintf(stderr, "  Tokens generated:   %d\n", n_gen);
    fprintf(stderr, "  Decode steps:       %d  (%.2f tok/step)\n", n_steps, (double)n_gen / n_steps);
    if (!args->baseline) {
        fprintf(stderr, "  Drafted steps:      %d / %d\n", n_drafted_steps, n_steps);
        fprintf(stderr, "  Draft tokens:       %d proposed, %d accepted (%.0f%%)\n",
                n_draft_tok, n_draft_acc, n_draft_tok ? 100.0 * n_draft_acc / n_draft_tok : 0.0);
        fprintf(stderr, "  Rebuilds (miss):    %d\n", n_rebuilds);
        fprintf(stderr, "  Accept histogram:   ");
        for (int i = 0; i <= args->draft_k; i++) fprintf(stderr, "%d:%d ", i, n_accept_hist[i]);
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "  Throughput:         %.1f tok/s  (loop)\n", n_gen / elapsed);
    fprintf(stderr, "════════════════════════════════════════════════════\n");

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  -m <path>    GGUF model path  (required)\n"
        "  -p <text>    Prompt\n"
        "  -n <int>     Max new tokens   (default: 120)\n"
        "  -k <int>     Max draft tokens   (default: 7, batch = k+1)\n"
        "  -g <int>     N-gram probe len   (default: 3)\n"
        "  -L <int>     Min suffix match to fire (default: 5)\n"
        "  -ngl <int>   GPU layers       (default: 99)\n"
        "  --baseline   Plain greedy loop (reference)\n"
        "  --sabotage   Corrupt all drafts (correctness stress)\n"
        "  --verbose    Per-step accept log\n",
        prog);
}

int main(int argc, char **argv) {
    LookupArgs args;
    strncpy(args.prompt, "The meaning of life is", sizeof(args.prompt) - 1);

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-m")   && i+1 < argc) strncpy(args.model_path, argv[++i], sizeof(args.model_path)-1);
        else if (!strcmp(argv[i], "-p")   && i+1 < argc) strncpy(args.prompt,     argv[++i], sizeof(args.prompt)-1);
        else if (!strcmp(argv[i], "-n")   && i+1 < argc) args.max_new      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-k")   && i+1 < argc) args.draft_k      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-g")   && i+1 < argc) args.ngram_g      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-L")   && i+1 < argc) args.ngram_L      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-ngl") && i+1 < argc) args.n_gpu_layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--baseline")) args.baseline = true;
        else if (!strcmp(argv[i], "--sabotage")) args.sabotage = true;
        else if (!strcmp(argv[i], "--verbose"))  args.verbose  = true;
        else if (!strcmp(argv[i], "--help"))     { usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
    }

    if (args.model_path[0] == '\0') {
        fprintf(stderr, "Error: -m <model_path> required\n");
        usage(argv[0]); return 1;
    }
    if (args.draft_k > MAX_DRAFT) args.draft_k = MAX_DRAFT;

    llama_backend_init();
    int rc = run_lookup(&args);
    llama_backend_free();
    return rc;
}
