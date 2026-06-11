/*
 * pti_chat.cpp — interactive terminal chat (llama-cli equivalent) with
 * switchable speculation modes. Made for live demos and screen recording.
 *
 *   bin/pti_chat -m <model.gguf> [--mode base|mtp|pti]
 *
 * Type a message, watch the reply stream; a stats line follows each turn.
 * Commands:
 *   /mode base|mtp|pti   switch speculation mode live (same conversation)
 *   /clear               reset the conversation
 *   /quit                exit  (Ctrl-D works too)
 *
 * Uses the model's chat template. Greedy with deterministic tie-breaking —
 * the SAME question answered in different modes produces byte-identical
 * text, so back-to-back mode comparisons are exact.
 *
 * Each turn re-prefills the full templated conversation (simple and robust;
 * prefill is batched and fast). The speculation loop is the proven M6.4c/M7.1
 * machinery from pti_lookup/pti_server.
 *
 * Build:  make chat
 */

#include "../llama.cpp/include/llama.h"
#include "../llama.cpp/src/llama-ext.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#define MAX_TOKENS  65536
#define MAX_DRAFT   31

enum Mode { MODE_BASE = 0, MODE_MTP = 1, MODE_PTI = 2 };
static const char *mode_name(int m) {
    return m == MODE_PTI ? "pti" : m == MODE_MTP ? "mtp" : "base";
}

static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ── quiet llama/ggml logging (default): only WARN+ passes ────────────────────
// The per-step seq ops make llama emit DEBUG lines ("copying KV buffer",
// "graph_reserve", "CUDA Graph id reused") that interleave with generated
// text. --verbose restores full logs for debugging.
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

// Greedy with deterministic tie-breaking (same rule as pti_server/pti_lookup):
// among tokens within EPS of the max logit, the lowest id wins — batch-shape
// kernel noise (~1e-3) can otherwise flip genuine near-ties between modes.
static constexpr float ARGMAX_EPS = 0.05f;

static int32_t argmax_f(const float *v, int32_t n) {
    int32_t best = 0;
    for (int32_t i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    float cut = v[best] - ARGMAX_EPS;
    for (int32_t i = 0; i < best; i++)
        if (v[i] >= cut) return i;
    return best;
}

// ── sampled verification (M7.4): sample-and-match, position-keyed RNG ────────
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
    double sum = 0.0;
    for (int32_t i = 0; i < n; i++) sum += exp((double)(logits[i] - max_l) / temp);
    double u = (double)(splitmix64(seed ^ (uint64_t)(pos + 1)) >> 11) / 9007199254740992.0;
    double acc = 0.0, target = u * sum;
    for (int32_t i = 0; i < n; i++) {
        acc += exp((double)(logits[i] - max_l) / temp);
        if (acc >= target) return i;
    }
    return n - 1;
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

// ── n-gram lookup draft (as pti_lookup.cpp) ──────────────────────────────────
static int ngram_draft(const llama_token *hist, int n_hist,
                       int g, int L_min, int k, llama_token *out, int *L_out = nullptr) {
    if (n_hist < g + 1) return 0;
    const llama_token *tail = hist + n_hist - g;
    for (int m = n_hist - 2; m >= g - 1; m--) {
        int match = 1;
        for (int t = 0; t < g; t++)
            if (hist[m - g + 1 + t] != tail[t]) { match = 0; break; }
        if (!match) continue;
        int L = g;
        while (m - L >= 0 && hist[m - L] == hist[n_hist - 1 - L]) L++;
        if (L < L_min) continue;
        int avail   = n_hist - 1 - m;
        int n_draft = avail < k ? avail : k;
        if (n_draft < 1) continue;
        for (int t = 0; t < n_draft; t++) out[t] = hist[m + 1 + t];
        if (L_out) *L_out = L;
        return n_draft;
    }
    return 0;
}

// ── MTP 1-pair feed (as pti_lookup.cpp; multi-token MTP batches are broken) ──
static llama_token mtp_feed1(llama_context *ctx_mtp, llama_context *ctx_main,
                             struct llama_batch *mb, llama_token tok, int h_idx,
                             llama_pos pos, int32_t n_embd, int32_t n_vocab) {
    if (!ctx_mtp) return -1;
    const float *h = llama_get_embeddings_pre_norm_ith(ctx_main, h_idx);
    if (!h) return -1;
    mb->n_tokens     = 1;
    mb->token[0]     = tok;
    mb->pos[0]       = pos;
    mb->n_seq_id[0]  = 1;
    mb->seq_id[0][0] = 0;
    mb->logits[0]    = 1;
    memcpy(mb->embd, h, (size_t)n_embd * sizeof(float));
    if (llama_decode(ctx_mtp, *mb) != 0) return -1;
    float *logits = llama_get_logits_ith(ctx_mtp, 0);
    if (!logits) return -1;
    return (llama_token)argmax_f(logits, n_vocab);
}

// ── globals ───────────────────────────────────────────────────────────────────
struct G {
    llama_model       *model   = nullptr;
    const llama_vocab *vocab   = nullptr;
    llama_context     *ctx     = nullptr;
    llama_context     *ctx_mtp = nullptr;
    llama_memory_t     mem     = nullptr;
    const char        *tmpl    = nullptr;
    int32_t n_vocab = 0, n_embd = 0;
    int     mode    = MODE_PTI;
    int     max_new = 1024;
    int     n_ctx_cfg = 98304;
    int     draft_k = 7, ngram_g = 3, ngram_L = 5;
    float    temperature = 0.0f;
    uint64_t seed        = 42;
} g;

static void print_token(llama_token tok, std::string *collect) {
    char buf[256];
    int len = llama_token_to_piece(g.vocab, tok, buf, (int32_t)sizeof(buf), 0, true);
    if (len > 0) {
        fwrite(buf, 1, len, stdout);
        fflush(stdout);
        collect->append(buf, len);
    }
}

// ── one chat turn: prefill the full conversation, then speculate ─────────────
static std::string generate_turn(const std::string &prompt) {
    std::string reply;
    int mode = g.mode;

    llama_memory_seq_rm(g.mem, 0, 0, -1);
    llama_memory_seq_rm(g.mem, 1, 0, -1);
    if (g.ctx_mtp) {
        llama_memory_t mm = llama_get_memory(g.ctx_mtp);
        llama_memory_seq_rm(mm, 0, 0, -1);
    }

    static std::vector<llama_token> hist;
    hist.resize(MAX_TOKENS);
    int n_hist = llama_tokenize(g.vocab, prompt.c_str(), (int32_t)prompt.size(),
                                hist.data(), MAX_TOKENS, true, true);
    if (n_hist <= 0) { fprintf(stderr, "[tokenize failed]\n"); return reply; }

    // clamp this turn's budget to the per-stream context
    int usable  = g.n_ctx_cfg / 2;
    if (n_hist + MAX_DRAFT + 8 >= usable) {
        fprintf(stderr, "[error] conversation (%d tok) exceeds usable context (%d = c/2); /clear or raise -c\n",
                n_hist, usable);
        return reply;
    }
    int max_new = g.max_new;
    if (n_hist + max_new + MAX_DRAFT + 8 > usable)
        max_new = usable - n_hist - MAX_DRAFT - 8;

    static llama_batch batch = {};
    static bool batch_done = false;
    const int chunk = 1024;
    if (!batch_done) { batch = llama_batch_init(chunk + 160, 0, 2); batch_done = true; }

    double t_pre0 = now_sec();
    int last_idx = 0;
    for (int i0 = 0; i0 < n_hist; i0 += chunk) {
        int nb = n_hist - i0 < chunk ? n_hist - i0 : chunk;
        batch_clear(&batch);
        for (int j = 0; j < nb; j++)
            batch_add(&batch, hist[i0 + j], (llama_pos)(i0 + j), 0, i0 + j == n_hist - 1);
        if (llama_decode(g.ctx, batch) != 0) { fprintf(stderr, "[prefill failed]\n"); return reply; }
        last_idx = nb - 1;
    }
    double t_prefill = now_sec() - t_pre0;

    llama_pos   p        = (llama_pos)n_hist;
    llama_token tok_last = (llama_token)sample_pos(llama_get_logits_ith(g.ctx, last_idx),
                                                   g.n_vocab, g.temperature, g.seed, p);
    hist[n_hist++] = tok_last;
    int n_gen = 1, n_steps = 0;
    int ng_fire = 0, ng_acc = 0, mtp_fire = 0, mtp_acc = 0;
    if (!llama_vocab_is_eog(g.vocab, tok_last)) print_token(tok_last, &reply);

    // MTP seed
    static llama_batch mtp_batch = {};
    static bool mtp_batch_init_done = false;
    llama_token mtp_cand = -1;
    if (g.ctx_mtp && mode >= MODE_MTP) {
        if (!mtp_batch_init_done) {
            mtp_batch = llama_batch_init(8, g.n_embd, 1);
            mtp_batch.token = (llama_token *)malloc(8 * sizeof(llama_token));
            mtp_batch_init_done = true;
        }
        mtp_cand = mtp_feed1(g.ctx_mtp, g.ctx, &mtp_batch, tok_last,
                             batch.n_tokens - 1, p, g.n_embd, g.n_vocab);
    }

    int L_dyn = g.ngram_L, k_cur = g.draft_k, n_zero = 0;
    llama_token pend_tok[128]; llama_pos pend_pos[128]; int n_pend = 0;

    double t0 = now_sec();
    while (n_gen < max_new && !llama_vocab_is_eog(g.vocab, tok_last)) {
        llama_token draft[MAX_DRAFT];
        int k = 0;
        int fire_L = 0;
        bool mtp_fired = false;
        if (mode == MODE_PTI && n_zero < 3)
            k = ngram_draft(hist.data(), n_hist, g.ngram_g, L_dyn, k_cur, draft, &fire_L);

        bool mtp_alive = g.ctx_mtp && mode >= MODE_MTP && mtp_cand != -1
                      && !(mtp_fire >= 10 && mtp_acc * 10 < mtp_fire * 3);

        // MTP arbitration (M7.3, L-aware M7.6): veto only marginal fires
        // (suffix match < L_TRUST); long matches override the MTP vote.
        constexpr int L_TRUST = 10;
        if (k > 0 && mtp_alive && k_cur <= g.draft_k
            && fire_L < L_TRUST && mtp_cand != draft[0])
            k = 0;

        if (k == 0 && mtp_alive) {
            draft[0] = mtp_cand; k = 1; mtp_fired = true;
        }
        if (k + 1 > max_new - n_gen) k = max_new - n_gen - 1;
        if (k < 0) { k = 0; mtp_fired = false; }

        if (n_pend + 1 + k > 120) {                       // flush pending
            batch_clear(&batch);
            for (int i = 0; i < n_pend; i++) batch_add(&batch, pend_tok[i], pend_pos[i], 0, false);
            if (llama_decode(g.ctx, batch) != 0) break;
            n_pend = 0;
        }
        if (k > 0 && n_pend == 0) {
            llama_memory_seq_rm(g.mem, 1, 0, -1);
            llama_memory_seq_cp(g.mem, 0, 1, 0, -1);
        }

        batch_clear(&batch);
        for (int i = 0; i < n_pend; i++) batch_add(&batch, pend_tok[i], pend_pos[i], 0, false);
        const int base = n_pend;
        batch_add(&batch, tok_last, p, 0, true);
        for (int i = 0; i < k; i++) batch_add(&batch, draft[i], p + 1 + i, 0, true);

        if (llama_decode(g.ctx, batch) != 0) { fprintf(stderr, "\n[decode failed]\n"); break; }
        n_steps++;
        if (k > 0 && !mtp_fired) ng_fire += k;

        llama_token tok_last_old = tok_last;
        int e = 0; bool stop = false;
        for (int i = 0; i <= k; i++) {
            llama_token a = (llama_token)sample_pos(llama_get_logits_ith(g.ctx, base + i),
                                                    g.n_vocab, g.temperature, g.seed, p + i + 1);
            if (!llama_vocab_is_eog(g.vocab, a)) print_token(a, &reply);
            hist[n_hist++] = a;
            n_gen++; e++; tok_last = a;
            if (llama_vocab_is_eog(g.vocab, a) || n_gen >= max_new) { stop = true; break; }
            if (i < k && draft[i] != a) break;
        }
        int acc = e - 1;
        if (k > 0 && mtp_fired)      { mtp_fire++; mtp_acc += acc; }
        else if (k > 0) {
            ng_acc += acc;
            if (acc == k) { L_dyn = L_dyn > g.ngram_L ? L_dyn - 1 : g.ngram_L;
                            k_cur = k_cur <= g.draft_k ? 15 : MAX_DRAFT; }
            else          { L_dyn += 4; k_cur = g.draft_k; if (acc == 0) n_zero++; }
        }
        if (g.ctx_mtp && mode >= MODE_MTP && !stop)
            mtp_cand = mtp_feed1(g.ctx_mtp, g.ctx, &mtp_batch, tok_last,
                                 base + e - 1, p + e, g.n_embd, g.n_vocab);
        if (stop) { p += e; break; }

        if (e == k + 1 || k == 0) { n_pend = 0; p += e; }
        else {
            llama_memory_seq_rm(g.mem, 0, 0, -1);
            llama_memory_seq_cp(g.mem, 1, 0, 0, -1);
            pend_tok[n_pend] = tok_last_old; pend_pos[n_pend] = p; n_pend++;
            for (int i = 0; i < e - 1; i++) { pend_tok[n_pend] = draft[i]; pend_pos[n_pend] = p + 1 + i; n_pend++; }
            p += e;
        }
    }
    double el = now_sec() - t0;

    fprintf(stderr, "\n\033[2m[%s] %d tok, %.1f tok/s (prefill %.1fs)",
            mode_name(mode), n_gen, el > 0 ? n_gen / el : 0.0, t_prefill);
    if (ng_fire)  fprintf(stderr, "  lookup %d/%d", ng_acc, ng_fire);
    if (mtp_fire) fprintf(stderr, "  mtp %d/%d", mtp_acc, mtp_fire);
    fprintf(stderr, "\033[0m\n");
    return reply;
}

// ── chat template over full message history ─────────────────────────────────
struct Msg { std::string role, content; };

static std::string apply_template(const std::vector<Msg> &msgs) {
    std::vector<llama_chat_message> chat;
    for (auto &m : msgs) chat.push_back({m.role.c_str(), m.content.c_str()});
    int32_t need = llama_chat_apply_template(g.tmpl, chat.data(), chat.size(), true, nullptr, 0);
    if (need < 0) {                       // no template: raw concatenation
        std::string raw;
        for (auto &m : msgs) raw += m.content + "\n";
        return raw;
    }
    std::vector<char> buf(need + 1);
    llama_chat_apply_template(g.tmpl, chat.data(), chat.size(), true, buf.data(), (int32_t)buf.size());
    return std::string(buf.data(), need);
}

int main(int argc, char **argv) {
    char model_path[512] = {};
    int  n_gpu_layers = 99, n_ctx = 98304;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-m")     && i+1 < argc) strncpy(model_path, argv[++i], sizeof(model_path)-1);
        else if (!strcmp(argv[i], "-ngl")   && i+1 < argc) n_gpu_layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c")     && i+1 < argc) n_ctx        = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n")     && i+1 < argc) g.max_new    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mode") && i+1 < argc) {
            const char *m = argv[++i];
            g.mode = !strcmp(m, "base") ? MODE_BASE : !strcmp(m, "mtp") ? MODE_MTP : MODE_PTI;
        }
        else if (!strcmp(argv[i], "-t")     && i+1 < argc) g.temperature = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) g.seed = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--verbose")) g_verbose_logs = true;
        else { fprintf(stderr, "Usage: %s -m <model> [--mode base|mtp|pti] [-t temp] [--seed n] [-n max] [-c ctx] [--verbose]\n", argv[0]); return 1; }
    }
    if (!model_path[0]) { fprintf(stderr, "Error: -m required\n"); return 1; }

    llama_log_set(pti_log_cb, nullptr);
    llama_backend_init();
    fprintf(stderr, "Loading %s ...\n", model_path);
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    g.model = llama_model_load_from_file(model_path, mp);
    if (!g.model) { fprintf(stderr, "load failed\n"); return 1; }
    g.vocab   = llama_model_get_vocab(g.model);
    g.n_vocab = llama_vocab_n_tokens(g.vocab);
    g.n_embd  = llama_model_n_embd(g.model);
    g.tmpl    = llama_model_chat_template(g.model, nullptr);

    llama_context_params cp = llama_context_default_params();
    g.n_ctx_cfg = n_ctx;
    cp.n_ctx = (uint32_t)n_ctx; cp.n_batch = 2048; cp.n_seq_max = 2;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;   // f16 KV, non-unified: exact
    g.ctx = llama_init_from_model(g.model, cp);
    if (!g.ctx) { fprintf(stderr, "context failed\n"); return 1; }
    g.mem = llama_get_memory(g.ctx);
    llama_set_embeddings_pre_norm(g.ctx, true, false);

    {
        llama_context_params mpp = llama_context_default_params();
        mpp.n_ctx = (uint32_t)n_ctx; mpp.n_batch = 2048; mpp.n_seq_max = 1;
        mpp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        mpp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        g.ctx_mtp = llama_init_from_model(g.model, mpp);
        if (g.ctx_mtp) llama_set_embeddings_pre_norm(g.ctx_mtp, true, true);
    }

    fprintf(stderr,
        "\n══ pti_chat ═══════════════════════════════════════════\n"
        "  mode: %s  temp: %.2f  seed: %llu%s\n"
        "  /mode base|mtp|pti   /temp <t>   /clear   /quit\n"
        "═══════════════════════════════════════════════════════\n\n",
        mode_name(g.mode), g.temperature, (unsigned long long)g.seed,
        g.ctx_mtp ? "" : "  [no MTP head: mtp/pti degrade to lookup]");

    std::vector<Msg> msgs;
    char line[8192];
    while (true) {
        fprintf(stderr, "\033[1m> \033[0m");
        if (!fgets(line, sizeof(line), stdin)) break;          // Ctrl-D
        size_t len = strlen(line);
        while (len && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = 0;
        if (!len) continue;

        if (!strcmp(line, "/quit") || !strcmp(line, "/exit")) break;
        if (!strcmp(line, "/clear")) { msgs.clear(); fprintf(stderr, "[conversation cleared]\n"); continue; }
        if (!strncmp(line, "/mode ", 6)) {
            const char *m = line + 6;
            g.mode = !strcmp(m, "base") ? MODE_BASE : !strcmp(m, "mtp") ? MODE_MTP : MODE_PTI;
            fprintf(stderr, "[mode: %s]\n", mode_name(g.mode));
            continue;
        }
        if (!strncmp(line, "/temp ", 6)) {
            g.temperature = atof(line + 6);
            fprintf(stderr, "[temp: %.2f]\n", g.temperature);
            continue;
        }

        msgs.push_back({"user", line});
        std::string reply = generate_turn(apply_template(msgs));
        msgs.push_back({"assistant", reply});
    }

    if (g.ctx_mtp) llama_free(g.ctx_mtp);
    llama_free(g.ctx);
    llama_model_free(g.model);
    llama_backend_free();
    return 0;
}
