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

#include "../../llama.cpp/include/llama.h"
#include "../../llama.cpp/src/llama-ext.h"   // pre-norm hidden access for MTP (M7.1)

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#define MAX_TOKENS   131072
#define DEFAULT_CTX  2048
#define PREFILL_CHUNK 1024   // llama_decode caps batches at n_batch; chunk prefill
#define MAX_DRAFT    31     // batch 32 = 6.93× cost, 12.1 ms/token, chain-exact (M6.0)

static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ── quiet llama/ggml logging (default): only WARN+ passes ────────────────────
// Per-step seq ops emit DEBUG spam ("copying KV buffer", "graph_reserve",
// "CUDA Graph id reused") that interleaves with generated text. --verbose
// restores full logs (and per-fire stats).
static bool g_verbose_logs = false;
static enum ggml_log_level g_last_lvl = GGML_LOG_LEVEL_NONE;
static void pti_log_cb(enum ggml_log_level level, const char *text, void *) {
    if (g_verbose_logs) { fputs(text, stderr); return; }
    if (level == GGML_LOG_LEVEL_CONT) {
        if (g_last_lvl >= GGML_LOG_LEVEL_WARN && g_last_lvl != GGML_LOG_LEVEL_CONT)
            fputs(text, stderr);
        return;
    }
    g_last_lvl = level;
    if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

// Greedy pick with deterministic tie-breaking (same rule as pti_server).
// Different batch sizes run different kernel configs whose reductions differ
// by ~1e-3 on logits — invisible except at genuine near-ties (the <think>
// open/close decision is a reliable knife-edge), where plain argmax flips
// between modes and breaks byte-identity. Among tokens within EPS of the
// max, the LOWEST id wins — identical in every mode by construction.
static constexpr float ARGMAX_EPS = 0.05f;

static int32_t argmax_f(const float *v, int32_t n) {
    int32_t best = 0;
    for (int32_t i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    float cut = v[best] - ARGMAX_EPS;
    for (int32_t i = 0; i < best; i++)
        if (v[i] >= cut) return i;        // earliest id within the tie band
    return best;
}

// ── sampled verification (M7.4): sample-and-match ───────────────────────────
// At temp > 0 each verified position SAMPLES from the target logits; the
// draft is accepted while the sample agrees and the sample itself is emitted
// at the first mismatch. Output is exactly plain temperature sampling — the
// drafts only decide what got batched (provably unbiased for deterministic
// drafts: accept prob = p(draft), correction ~ residual — the optimal
// rejection scheme degenerates to this).
//
// The RNG is keyed on (seed, position) — counter-based, not sequential — so
// every mode consumes randomness identically per position and byte-identity
// across modes survives at temp > 0 (same fp near-tie caveat as greedy).

static uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

static int32_t sample_pos(const float *logits, int32_t n, float temp,
                          uint64_t seed, llama_pos pos) {
    if (temp <= 0.0f) return argmax_f(logits, n);

    float max_l = logits[0];
    for (int32_t i = 1; i < n; i++) if (logits[i] > max_l) max_l = logits[i];

    // softmax at temperature; single pass for the normalizer
    double sum = 0.0;
    for (int32_t i = 0; i < n; i++) sum += exp((double)(logits[i] - max_l) / temp);

    double u = (double)(splitmix64(seed ^ (uint64_t)(pos + 1)) >> 11)
             / 9007199254740992.0;                       // [0,1) from top 53 bits
    double acc = 0.0, target = u * sum;
    for (int32_t i = 0; i < n; i++) {
        acc += exp((double)(logits[i] - max_l) / temp);
        if (acc >= target) return i;
    }
    return n - 1;
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
                       int g, int L_min, int k, llama_token *out, int *L_out) {
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
        if (L_out) *L_out = L;
        return n_draft;
    }
    return 0;
}

// ── MTP feed + predict (M7.1) ─────────────────────────────────────────────────
// Feeds the MTP context one batch of (emitted token @ q, h(q-1)) pairs to keep
// its 1-layer attention cache contiguous over the emitted stream, with logits
// on the last pair → candidate for the position after the last emitted token.
// Returns -1 on failure (caller simply drafts nothing).
static llama_token mtp_feed(
        llama_context *ctx_mtp,
        llama_context *ctx_main,
        struct llama_batch *mb,            // pre-allocated, embd-capable
        const llama_token *toks,           // emitted tokens
        const int         *h_idx,          // their logits batch indices in ctx_main
        llama_pos          pos0,           // position of toks[0]
        int                n,
        int32_t            n_embd,
        int32_t            n_vocab)
{
    if (!ctx_mtp || n <= 0) return -1;
    mb->n_tokens = 0;
    for (int j = 0; j < n; j++) {
        const float *h = llama_get_embeddings_pre_norm_ith(ctx_main, h_idx[j]);
        if (!h) return -1;
        mb->token[j]     = toks[j];
        mb->pos[j]       = pos0 + j;
        mb->n_seq_id[j]  = 1;
        mb->seq_id[j][0] = 0;
        mb->logits[j]    = (j == n - 1) ? 1 : 0;
        memcpy(mb->embd + (size_t)j * n_embd, h, (size_t)n_embd * sizeof(float));
        mb->n_tokens++;
    }
    if (llama_decode(ctx_mtp, *mb) != 0) return -1;
    float *logits = llama_get_logits_ith(ctx_mtp, n - 1);
    if (!logits) return -1;
    return (llama_token)argmax_f(logits, n_vocab);
}

struct LookupArgs {
    int  max_new      = 120;
    int  n_ctx        = 16384;  // usable = n_ctx/2 (checkpoint stream)
    int  n_gpu_layers = 99;
    int  draft_k      = 7;   // long drafts: only confident fires happen, and
                             // batch-8 hit-runs amortize best (3.11× for 8 tok)
    int  ngram_g      = 3;   // probe length
    int  ngram_L      = 5;   // min suffix-match length to fire (copy-run gate)
    bool baseline     = false;
    bool sabotage     = false;
    bool verbose      = false;
    bool use_mtp      = false;  // M7.1: MTP t+2 draft on novel-text steps (88.6% probe)
    bool no_ngram     = false;  // disable lookup (isolates MTP: the base+MTP cell)
    float    temperature = 0.0f;   // M7.4: >0 → sampled verification (sample-and-match)
    uint64_t seed        = 42;     // counter-based RNG key; fixed default = reproducible
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
    cparams.n_ctx      = (uint32_t)args->n_ctx;
    cparams.n_batch    = PREFILL_CHUNK;
    cparams.n_seq_max  = 2;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }
    llama_memory_t mem = llama_get_memory(ctx);

    // ── MTP draft context (M7.1) ─────────────────────────────────────────────
    // The nextn head, fed (emb(emitted tok @ q), h(q-1)), predicts q+1 at
    // 88.6% (M7.0 probe). One 1-layer decode ≈ 3.5 ms vs 52.6 ms full pass.
    struct llama_context *ctx_mtp = nullptr;
    int32_t n_embd = llama_model_n_embd(model);
    if (args->use_mtp && !args->baseline) {
        llama_set_embeddings_pre_norm(ctx, true, false);
        struct llama_context_params mp = llama_context_default_params();
        mp.n_ctx     = (uint32_t)args->n_ctx;
        mp.n_batch   = PREFILL_CHUNK;
        mp.n_seq_max = 1;
        mp.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
        mp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        ctx_mtp = llama_init_from_model(model, mp);
        if (ctx_mtp) {
            llama_set_embeddings_pre_norm(ctx_mtp, true, true);
            fprintf(stderr, "MTP draft context: active\n");
        } else {
            fprintf(stderr, "MTP draft context: UNAVAILABLE (no nextn head?) — continuing without\n");
        }
    }

    static llama_token hist[MAX_TOKENS];
    int n_hist = llama_tokenize(vocab, args->prompt, (int32_t)strlen(args->prompt),
                                hist, MAX_TOKENS, true, true);
    if (n_hist < 0) { fprintf(stderr, "Tokenize failed\n"); return 1; }
    fprintf(stderr, "Prompt: %d tokens   mode: %s%s   k=%d g=%d L>=%d\n\n",
            n_hist, args->baseline ? "BASELINE" : "LOOKUP",
            args->sabotage ? "+SABOTAGE" : "", args->draft_k, args->ngram_g, args->ngram_L);

    // ── Budget: clamp generation to the per-stream context ──────────────────
    int usable = args->n_ctx / 2;                 // non-unified: 2 streams
    if (n_hist + MAX_DRAFT + 8 >= usable) {
        fprintf(stderr, "Prompt (%d tokens) does not fit usable context (%d = n_ctx/2). Raise -c.\n",
                n_hist, usable);
        return 1;
    }
    int max_new = args->max_new;
    if (n_hist + max_new + MAX_DRAFT + 8 > usable) {
        max_new = usable - n_hist - MAX_DRAFT - 8;
        fprintf(stderr, "[note] max tokens clamped to %d (usable ctx %d - prompt %d)\n",
                max_new, usable, n_hist);
    }

    // ── Prefill seq 0, chunked to n_batch ───────────────────────────────────
    struct llama_batch batch = llama_batch_init(PREFILL_CHUNK + 160, 0, 2);
    int last_idx = 0;                              // batch index of final prompt token
    for (int i0 = 0; i0 < n_hist; i0 += PREFILL_CHUNK) {
        int nb = n_hist - i0 < PREFILL_CHUNK ? n_hist - i0 : PREFILL_CHUNK;
        batch_clear(&batch);
        for (int j = 0; j < nb; j++)
            batch_add(&batch, hist[i0 + j], (llama_pos)(i0 + j), 0, i0 + j == n_hist - 1);
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "Prefill failed\n"); return 1; }
        last_idx = nb - 1;
    }

    llama_pos   p        = (llama_pos)n_hist;
    llama_token tok_last = (llama_token)sample_pos(llama_get_logits_ith(ctx, last_idx),
                                                   n_vocab, args->temperature, args->seed, p);
    hist[n_hist++] = tok_last;

    fprintf(stderr, "Output: ");
    print_token(vocab, tok_last);
    int n_gen = 1;

    // stats
    int n_steps = 0, n_drafted_steps = 0, n_rebuilds = 0;
    int n_accept_hist[MAX_DRAFT + 2] = {};   // index = accepted drafts in a drafted step
    int n_draft_tok = 0, n_draft_acc = 0;
    int n_mtp_fire = 0, n_mtp_acc = 0, n_veto = 0;

    // MTP state: candidate predicting the token after tok_last (M7.1)
    struct llama_batch mtp_batch = {};
    llama_token mtp_cand = -1;
    if (ctx_mtp) {
        mtp_batch = llama_batch_init(MAX_DRAFT + 40, n_embd, 1);
        mtp_batch.token = (llama_token *)malloc((MAX_DRAFT + 40) * sizeof(llama_token));
        // seed: (tok_last @ p, h of prefill's last token) → candidate for p+1
        int h0 = last_idx;
        mtp_cand = mtp_feed(ctx_mtp, ctx, &mtp_batch, &tok_last, &h0, p, 1, n_embd, n_vocab);
    }

    double t_start = now_sec();

    if (args->baseline) {
        // ── plain greedy loop ────────────────────────────────────────────────
        while (n_gen < max_new && !llama_vocab_is_eog(vocab, tok_last)) {
            batch_clear(&batch);
            batch_add(&batch, tok_last, p, 0, true);
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "\nDecode failed\n"); break; }
            tok_last = (llama_token)sample_pos(llama_get_logits_ith(ctx, 0),
                                               n_vocab, args->temperature, args->seed, p + 1);
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
        //
        // Adaptive draft length ladder: full accept escalates 7 → 15 → 31
        // (batch 16 = 15.6 ms/tok, batch 32 = 12.1 ms/tok — M6.0 curve).
        // Any miss resets to draft_k. Deep batches only fire inside runs
        // that have already proven themselves at the previous rung.
        //
        // Hard-off: after 3 zero-accept fires, drafting is disabled for the
        // rest of the generation — bounds the worst case on hostile text.
        //
        // Merged rebuild: on a miss, the accepted prefix is NOT re-decoded in
        // its own call. It rides as logits-free "pending" tokens at the front
        // of the NEXT batch (batching is sub-linear, so one call of e+1+k
        // beats cost(e) + cost(1+k)). While pending exists, seq0 sits at the
        // checkpoint state and seq1 is not refreshed; a full accept of a
        // merged batch consumes pending and makes seq0 clean again.
        int L_dyn   = args->ngram_L;
        int k_cur   = args->draft_k;
        int n_zero  = 0;
        llama_token pend_tok[128];
        llama_pos   pend_pos[128];
        int n_pend  = 0;
        while (n_gen < max_new && !llama_vocab_is_eog(vocab, tok_last)) {
            llama_token draft[MAX_DRAFT];
            int fire_L = 0;
            int k = (args->no_ngram || n_zero >= 3) ? 0
                  : ngram_draft(hist, n_hist, args->ngram_g, L_dyn, k_cur, draft, &fire_L);

            bool mtp_alive = ctx_mtp && mtp_cand != -1
                          && !(n_mtp_fire >= 10 && n_mtp_acc * 10 < n_mtp_fire * 3);

            // MTP arbitration (M7.3, refined M7.6): the MTP candidate predicts
            // the SAME position as draft[0]; a disagreement marks a fire as
            // suspect. But MTP is only ~89% right, so an unconditional veto
            // discards ~11% of GOOD fires — expensive when fires are rare and
            // land 7-31 tokens (php run: 6 fires ≈ +3 tok/s). L-aware rule:
            // veto only MARGINAL fires (suffix match below L_TRUST); long
            // matches are near-certain copy-runs and override the MTP vote.
            // Escalated rungs (15/31) skip the veto as before (earned trust).
            constexpr int L_TRUST = 10;
            if (k > 0 && mtp_alive && k_cur <= args->draft_k
                && fire_L < L_TRUST && mtp_cand != draft[0]) {
                k = 0;                                     // veto the marginal fire
                n_veto++;
            }

            // MTP fallback (M7.1): no (surviving) n-gram fire → 1-token
            // t+2 draft (88.6% probe accuracy). Disable if it proves <30%.
            bool mtp_fired = false;
            if (k == 0 && mtp_alive) {
                draft[0] = mtp_cand;
                k = 1;
                mtp_fired = true;
            }

            if (k + 1 > max_new - n_gen) k = max_new - n_gen - 1;
            if (k < 0) { k = 0; mtp_fired = false; }
            if (args->sabotage) {
                for (int i = 0; i < k; i++) draft[i] = (draft[i] + 1) % n_vocab;
            }

            // overflow guard: flush pending alone if the merged batch would not fit
            if (n_pend + 1 + k > 120) {
                batch_clear(&batch);
                for (int i = 0; i < n_pend; i++)
                    batch_add(&batch, pend_tok[i], pend_pos[i], 0, false);
                if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "\nFlush failed\n"); break; }
                n_pend = 0;
            }

            if (k > 0 && n_pend == 0) {
                // refresh checkpoint: seq1 = S(p). Skipped while pending exists —
                // seq1 already holds the checkpoint the pending tokens build on.
                llama_memory_seq_rm(mem, 1, 0, -1);
                llama_memory_seq_cp(mem, 0, 1, 0, -1);
            }

            batch_clear(&batch);
            for (int i = 0; i < n_pend; i++)
                batch_add(&batch, pend_tok[i], pend_pos[i], 0, false);
            const int base = n_pend;                 // batch index of first logits token
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
                llama_token a = (llama_token)sample_pos(llama_get_logits_ith(ctx, base + i),
                                                        n_vocab, args->temperature, args->seed, p + i + 1);
                print_token(vocab, a);
                hist[n_hist++] = a;
                n_gen++; e++;
                tok_last = a;
                if (llama_vocab_is_eog(vocab, a) || n_gen >= max_new) { stop = true; break; }
                if (i < k && draft[i] != a) break;   // a is the correction; rest invalid
            }
            int acc = e - 1;                          // accepted drafts this step
            if (k > 0 && mtp_fired) {
                // MTP fires tracked separately — they must not move the AIMD bar
                n_mtp_fire++;
                n_mtp_acc += acc;
            } else if (k > 0) {
                n_accept_hist[acc < MAX_DRAFT+1 ? acc : MAX_DRAFT+1]++;
                n_draft_acc += acc;
                if (acc == k) {
                    L_dyn = L_dyn > args->ngram_L ? L_dyn - 1 : args->ngram_L;
                    k_cur = k_cur <= args->draft_k ? 15 : MAX_DRAFT;   // ladder: 7→15→31
                } else {
                    L_dyn += 4;
                    k_cur = args->draft_k;            // back to probe size
                    if (acc == 0) n_zero++;
                }
            }
            if (args->verbose && k > 0)
                fprintf(stderr, "\n[%s k=%d acc=%d L_dyn=%d k_cur=%d pend=%d]",
                        mtp_fired ? "mtp" : "draft", k, acc, L_dyn, k_cur, n_pend);

            // MTP cache feed + next candidate (M7.1): pair each emitted token
            // a_j (at position p+1+j) with its source hidden state h(p+j) from
            // logits index base+j. Must run before the next main-ctx decode
            // overwrites the pre-norm buffer.
            if (ctx_mtp && !stop) {
                // Single-pair feed, LAST emitted token only (flat 3.5 ms/step).
                // Multi-token MTP batches are broken (~48% accept vs 88.6% probe
                // — graph issue at n_tokens>1; probe/pti_mtp only ever fed one).
                // Per-pair sequential feeds match the probe but cost e×3.5 ms —
                // fatal on 32-token copy-run accepts. Last-pair-only leaves
                // position gaps in the MTP cache for multi-emit steps; the
                // probe already showed the head tolerates missing history.
                int h_idx = base + e - 1;
                mtp_cand = mtp_feed(ctx_mtp, ctx, &mtp_batch,
                                    &tok_last, &h_idx, p + e, 1, n_embd, n_vocab);
            }

            if (stop) { p += e; break; }

            if (e == k + 1 || k == 0) {
                // full accept (or plain step): the batch consumed pending +
                // exactly the verified tokens → seq0 clean at S(p+e)
                n_pend = 0;
                p += e;
            } else {
                // partial: seq0 consumed wrong tail → restore checkpoint and
                // queue the accepted prefix as pending for the next batch
                n_rebuilds++;
                llama_memory_seq_rm(mem, 0, 0, -1);
                llama_memory_seq_cp(mem, 1, 0, 0, -1);       // S(ckpt)
                pend_tok[n_pend] = tok_last_old;  pend_pos[n_pend] = p;  n_pend++;
                for (int i = 0; i < e - 1; i++) {            // accepted drafts only
                    pend_tok[n_pend] = draft[i];  pend_pos[n_pend] = p + 1 + i;  n_pend++;
                }
                p += e;
            }
        }
    }

    double elapsed = now_sec() - t_start;

    fprintf(stderr, "\n\n════════════════════════════════════════════════════\n");
    fprintf(stderr, "  PTI-lookup results (%s%s%s%s)\n",
            args->baseline ? "baseline" : (args->no_ngram ? "mtp-only" : "lookup"),
            (!args->baseline && args->use_mtp && !args->no_ngram) ? "+mtp" : "",
            args->sabotage ? "+sabotage" : "",
            (args->use_mtp && !ctx_mtp) ? " [MTP UNAVAILABLE]" : "");
    fprintf(stderr, "  Tokens generated:   %d\n", n_gen);
    fprintf(stderr, "  Decode steps:       %d  (%.2f tok/step)\n", n_steps, (double)n_gen / n_steps);
    if (!args->baseline) {
        fprintf(stderr, "  Drafted steps:      %d / %d\n", n_drafted_steps, n_steps);
        fprintf(stderr, "  Draft tokens:       %d proposed, %d accepted (%.0f%%)\n",
                n_draft_tok, n_draft_acc, n_draft_tok ? 100.0 * n_draft_acc / n_draft_tok : 0.0);
        if (ctx_mtp)
            fprintf(stderr, "  MTP fires:          %d, accepted %d (%.0f%%)   vetoed n-gram fires: %d\n",
                    n_mtp_fire, n_mtp_acc, n_mtp_fire ? 100.0 * n_mtp_acc / n_mtp_fire : 0.0, n_veto);
        fprintf(stderr, "  Rebuilds (miss):    %d\n", n_rebuilds);
        fprintf(stderr, "  Accept histogram:   ");
        for (int i = 0; i <= MAX_DRAFT; i++)
            if (n_accept_hist[i]) fprintf(stderr, "%d:%d ", i, n_accept_hist[i]);
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "  Throughput:         %.1f tok/s  (loop)\n", n_gen / elapsed);
    fprintf(stderr, "════════════════════════════════════════════════════\n");

    llama_batch_free(batch);
    if (ctx_mtp) {
        free(mtp_batch.token);
        mtp_batch.token = nullptr;
        llama_batch_free(mtp_batch);
        llama_free(ctx_mtp);
    }
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  -m <path>    GGUF model path  (required)\n"
        "  -p <text>    Prompt\n"
        "  -n <int>     Max new tokens   (default: 120; clamped to fit context)\n"
        "  -c <int>     Context size     (default: 16384; usable = c/2)\n"
        "  -k <int>     Max draft tokens   (default: 7, batch = k+1)\n"
        "  -g <int>     N-gram probe len   (default: 3)\n"
        "  -L <int>     Min suffix match to fire (default: 5)\n"
        "  -t <float>   Temperature (default: 0 = greedy; >0 = sampled verification)\n"
        "  --seed <u64> RNG seed, counter-keyed by position (default: 42)\n"
        "  -ngl <int>   GPU layers       (default: 99)\n"
        "  --baseline   Plain greedy loop (reference)\n"
        "  --mtp        MTP t+2 draft on novel-text steps (M7.1)\n"
        "  --no-ngram   Disable lookup drafts (with --mtp: MTP-only mode)\n"
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
        else if (!strcmp(argv[i], "-c")   && i+1 < argc) args.n_ctx        = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-k")   && i+1 < argc) args.draft_k      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-g")   && i+1 < argc) args.ngram_g      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-L")   && i+1 < argc) args.ngram_L      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t")   && i+1 < argc) args.temperature  = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) args.seed       = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-ngl") && i+1 < argc) args.n_gpu_layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--baseline")) args.baseline = true;
        else if (!strcmp(argv[i], "--sabotage")) args.sabotage = true;
        else if (!strcmp(argv[i], "--verbose"))  args.verbose  = true;
        else if (!strcmp(argv[i], "--mtp"))      args.use_mtp  = true;
        else if (!strcmp(argv[i], "--no-ngram")) args.no_ngram = true;
        else if (!strcmp(argv[i], "--help"))     { usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
    }

    if (args.model_path[0] == '\0') {
        fprintf(stderr, "Error: -m <model_path> required\n");
        usage(argv[0]); return 1;
    }
    if (args.draft_k > MAX_DRAFT) args.draft_k = MAX_DRAFT;

    g_verbose_logs = args.verbose;
    llama_log_set(pti_log_cb, nullptr);
    llama_backend_init();
    int rc = run_lookup(&args);
    llama_backend_free();
    return rc;
}
