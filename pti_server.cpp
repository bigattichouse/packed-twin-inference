/*
 * pti_server.cpp — speculative-decoding HTTP server (OpenAI-compatible)
 *
 * Three modes, one binary — the experiment matrix:
 *   --mode base   plain greedy decode            (= stock llama-server behavior)
 *   --mode mtp    + MTP head drafting            (1-token t+2 drafts, ~85% accept)
 *   --mode pti    + n-gram lookup drafting       (up to 31-token drafts on copy-runs)
 *
 * Speculation is active at ANY temperature (M7.4 sample-and-match): at
 * temp > 0 each verified position samples from the target logits with a
 * position-keyed RNG, so output is exactly plain temperature sampling and
 * a fixed "seed" reproduces a run. Measured at temp 0.25 on a code edit:
 * 2.0× with output byte-identical to the same-seed plain run.
 *
 * Per-request console log: mode, tokens, tok/s, draft accept rates.
 *
 * Build:  make server
 * Run:    bin/pti_server -m ../gguf/Qwen3.6-27B-UD-Q6_K_XL.gguf --mode pti
 */

#define CPPHTTPLIB_THREAD_POOL_COUNT 2
// Resolved via -I$(LLAMA_DIR)/vendor added in the Makefile server rule
#include "cpp-httplib/httplib.h"
#include "nlohmann/json.hpp"

#include "../llama.cpp/include/llama.h"
#include "../llama.cpp/src/llama-ext.h"   // pre-norm hidden access for MTP

#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <functional>
#include <string>
#include <vector>

using json = nlohmann::json;

// ── compile-time limits ───────────────────────────────────────────────────────

static constexpr int MAX_PROMPT_TOKENS = 65536;
static constexpr int MAX_DRAFT         = 31;   // batch 32 = 6.93× cost (M6.0)

enum PtiMode { MODE_BASE = 0, MODE_MTP = 1, MODE_PTI = 2 };
static const char *mode_name(int m) {
    return m == MODE_PTI ? "pti" : m == MODE_MTP ? "mtp" : "base";
}

// ── server args ───────────────────────────────────────────────────────────────

struct ServerArgs {
    char    model_path[512] = {};
    int     port            = 8080;
    int     n_gpu_layers    = 99;
    int     n_ctx           = 98304;   // non-unified: usable ctx = n_ctx/2 = 49k per request
                                       // (~1/4 of layers attend). MTP ctx needs ~1 GB more;
                                       // 131k+ OOMs the MTP context on 32 GB. --kv-q8 stretches.
    bool    kv_q8           = false;   // opt-in: q8 KV → 256k-class ctx, BUT breaks
                                       // byte-identity across modes (batch-size-dependent
                                       // FA numerics flip near-tie tokens)
    int     n_batch         = 1024;
    int     n_ubatch        = 256;
    int     mode            = MODE_PTI;
    float   temperature     = 0.0f;    // greedy default: enables speculation
    int     max_tokens      = 1024;
    int     draft_k         = 7;       // lookup probe size (ladder: 7→15→31)
    int     ngram_g         = 3;
    int     ngram_L         = 5;
};

// ── global model (loaded once at startup) ────────────────────────────────────

struct PTIGlobal {
    llama_model       *model   = nullptr;
    const llama_vocab *vocab   = nullptr;
    int32_t            n_vocab = 0;
    int32_t            n_embd  = 0;
    const char        *tmpl    = nullptr;
    ServerArgs         args;
} G;

static std::atomic<bool> G_busy{false};

// ── helpers ───────────────────────────────────────────────────────────────────

static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ── quiet llama/ggml logging (default): only WARN+ passes ────────────────────
// Per-step seq ops emit DEBUG spam ("copying KV buffer", "graph_reserve",
// "CUDA Graph id reused") that drowns the per-request stats. --verbose
// restores full logs.
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

// Greedy pick with deterministic tie-breaking. Different batch sizes run
// different kernel configurations whose reduction orders differ by ~1e-3 on
// the logits — invisible except at genuine near-ties (e.g. the <think>
// open/close decision on chat templates), where plain argmax flips between
// modes and breaks byte-identity. Rule: among all tokens within EPS of the
// max, the LOWEST token id wins — identical in every mode by construction.
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
// temp > 0: each verified position SAMPLES from the target logits; drafts are
// accepted while the sample agrees. Output is exactly plain temperature
// sampling — drafts only decide what got batched. RNG is keyed on
// (seed, position) so every mode consumes randomness identically.
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
    b->token       [b->n_tokens] = tok;
    b->pos         [b->n_tokens] = pos;
    b->n_seq_id    [b->n_tokens] = 1;
    b->seq_id      [b->n_tokens][0] = seq;
    b->logits      [b->n_tokens] = want_logits ? 1 : 0;
    b->n_tokens++;
}

static void batch_clear(struct llama_batch *b) { b->n_tokens = 0; }

static std::string token_to_str(llama_token tok) {
    char buf[256];
    int len = llama_token_to_piece(G.vocab, tok, buf, (int32_t)sizeof(buf), 0, true);
    return len > 0 ? std::string(buf, len) : std::string{};
}

// ── n-gram lookup draft (same as pti_lookup.cpp) ─────────────────────────────

static int ngram_draft(const llama_token *hist, int n_hist,
                       int g, int L_min, int k, llama_token *out) {
    if (n_hist < g + 1) return 0;
    const llama_token *tail = hist + n_hist - g;
    for (int m = n_hist - 2; m >= g - 1; m--) {
        int match = 1;
        for (int t = 0; t < g; t++) {
            if (hist[m - g + 1 + t] != tail[t]) { match = 0; break; }
        }
        if (!match) continue;
        int L = g;
        while (m - L >= 0 && hist[m - L] == hist[n_hist - 1 - L]) L++;
        if (L < L_min) continue;
        int avail   = n_hist - 1 - m;
        int n_draft = avail < k ? avail : k;
        if (n_draft < 1) continue;
        for (int t = 0; t < n_draft; t++) out[t] = hist[m + 1 + t];
        return n_draft;
    }
    return 0;
}

// ── MTP feed + predict (same as pti_lookup.cpp; 1-pair feeds only —
//    multi-token MTP batches are broken, see KERNEL_PLAN.md M7.1) ─────────────

static llama_token mtp_feed1(
        llama_context *ctx_mtp,
        llama_context *ctx_main,
        struct llama_batch *mb,
        llama_token tok, int h_idx, llama_pos pos,
        int32_t n_embd, int32_t n_vocab)
{
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

// ── generation ────────────────────────────────────────────────────────────────

struct GenStats {
    int    n_gen      = 0;
    int    n_steps    = 0;
    int    ng_fire    = 0, ng_acc  = 0;   // lookup drafts proposed/accepted
    int    mtp_fire   = 0, mtp_acc = 0;   // MTP drafts fired/accepted
    double tok_s      = 0.0;
    const char *mode  = "base";
};

using TokenSink = std::function<bool(const std::string &)>;

static GenStats run_generate(const std::string &prompt, int max_new,
                             float temperature, uint64_t seed, int mode,
                             const TokenSink &sink) {
    GenStats stats;
    stats.mode = mode_name(mode);

    // ── main context ─────────────────────────────────────────────────────────
    // IDENTICAL config for ALL modes (n_seq_max, pre-norm): different cache
    // layouts shift flash-attention numerics enough to flip near-tie tokens,
    // which would break the byte-identical-across-modes guarantee. The only
    // variable between modes is the drafting.
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx        = (uint32_t)G.args.n_ctx;
    cparams.n_batch      = (uint32_t)G.args.n_batch;
    cparams.n_ubatch     = (uint32_t)G.args.n_ubatch;
    cparams.n_seq_max    = 2;                            // working + checkpoint
    cparams.kv_unified   = false;                        // MUST stay false: unified KV
                                                         // shifts FA numerics with batch
                                                         // size → near-tie tokens flip
                                                         // between modes (measured). The
                                                         // proven-exact config matches
                                                         // pti_lookup: non-unified + f16.
                                                         // Cost: n_ctx splits across the
                                                         // 2 streams (usable = n_ctx/2).
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    // KV defaults to f16: Q8_0 KV + flash attention makes attention numerics
    // BATCH-SIZE-DEPENDENT (a 1-token and a 2-token decode of the same
    // position can argmax differently), which breaks byte-identity between
    // modes. f16 KV is chain-exact at every batch size (M6.0). --kv-q8 opts
    // into more context at the cost of that guarantee.
    if (G.args.kv_q8) {
        cparams.type_k = GGML_TYPE_Q8_0;
        cparams.type_v = GGML_TYPE_Q8_0;
    }

    llama_context *ctx = llama_init_from_model(G.model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return stats; }
    llama_memory_t mem = llama_get_memory(ctx);
    llama_set_embeddings_pre_norm(ctx, true, false);

    // ── MTP context (mtp/pti modes) ──────────────────────────────────────────
    llama_context *ctx_mtp = nullptr;
    if (mode >= MODE_MTP) {
        llama_context_params mp = llama_context_default_params();
        mp.n_ctx     = (uint32_t)G.args.n_ctx;
        mp.n_batch   = (uint32_t)G.args.n_batch;
        mp.n_seq_max = 1;
        mp.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
        mp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        ctx_mtp = llama_init_from_model(G.model, mp);
        if (!ctx_mtp)
            fprintf(stderr, "  [note] MTP context unavailable (no nextn head?) — lookup only\n");
    }

    // ── tokenize + prefill ───────────────────────────────────────────────────
    std::vector<llama_token> hist(MAX_PROMPT_TOKENS);
    int n_hist = llama_tokenize(G.vocab, prompt.c_str(), (int32_t)prompt.size(),
                                hist.data(), MAX_PROMPT_TOKENS, true, true);
    if (n_hist <= 0) {
        fprintf(stderr, "Tokenize failed\n");
        if (ctx_mtp) llama_free(ctx_mtp);
        llama_free(ctx);
        return stats;
    }
    hist.resize((size_t)n_hist + max_new + 8);
    fprintf(stderr, "  prompt: %d tokens  mode: %s\n", n_hist, stats.mode);

    int batch_cap = (n_hist > 200 ? n_hist : 200) + 4;
    llama_batch batch = llama_batch_init(batch_cap, 0, 2);
    batch_clear(&batch);
    for (int i = 0; i < n_hist; i++)
        batch_add(&batch, hist[i], (llama_pos)i, 0, i == n_hist - 1);
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Prefill failed\n");
        llama_batch_free(batch);
        if (ctx_mtp) llama_free(ctx_mtp);
        llama_free(ctx);
        return stats;
    }

    llama_pos   p        = (llama_pos)n_hist;
    llama_token tok_last = (llama_token)sample_pos(llama_get_logits_ith(ctx, batch.n_tokens - 1),
                                                   G.n_vocab, temperature, seed, p);
    hist[n_hist++] = tok_last;
    if (!sink(token_to_str(tok_last))) goto done_early;
    stats.n_gen = 1;

    {
        // ── MTP seed + state ─────────────────────────────────────────────────
        struct llama_batch mtp_batch = {};
        llama_token mtp_cand = -1;
        if (ctx_mtp) {
            mtp_batch = llama_batch_init(8, G.n_embd, 1);
            mtp_batch.token = (llama_token *)malloc(8 * sizeof(llama_token));
            mtp_cand = mtp_feed1(ctx_mtp, ctx, &mtp_batch, tok_last,
                                 batch.n_tokens - 1, p, G.n_embd, G.n_vocab);
        }

        // ── speculative loop state (pti_lookup.cpp, M6.4c) ──────────────────
        int L_dyn  = G.args.ngram_L;
        int k_cur  = G.args.draft_k;
        int n_zero = 0;
        llama_token pend_tok[128];
        llama_pos   pend_pos[128];
        int n_pend = 0;

        double t0 = now_sec();

        while (stats.n_gen < max_new && !llama_vocab_is_eog(G.vocab, tok_last)) {
            llama_token draft[MAX_DRAFT];
            int k = 0;
            bool mtp_fired = false;

            if (mode == MODE_PTI && n_zero < 3)
                k = ngram_draft(hist.data(), n_hist, G.args.ngram_g, L_dyn, k_cur, draft);

            bool mtp_alive = ctx_mtp && mtp_cand != -1 && mode >= MODE_MTP
                          && !(stats.mtp_fire >= 10 && stats.mtp_acc * 10 < stats.mtp_fire * 3);

            // MTP arbitration (M7.3): at the probe rung, an MTP disagreement
            // with draft[0] marks the n-gram fire as coincidental — veto it.
            // Makes mtp mode the floor of pti mode (measured: best-or-tied
            // on every text class). Escalated rungs skip the veto.
            if (k > 0 && mtp_alive && k_cur <= G.args.draft_k && mtp_cand != draft[0])
                k = 0;

            if (k == 0 && mtp_alive) {
                draft[0] = mtp_cand;
                k = 1;
                mtp_fired = true;
            }
            if (k + 1 > max_new - stats.n_gen) k = max_new - stats.n_gen - 1;
            if (k < 0) { k = 0; mtp_fired = false; }

            // flush pending if the merged batch would not fit
            if (n_pend + 1 + k > 120) {
                batch_clear(&batch);
                for (int i = 0; i < n_pend; i++)
                    batch_add(&batch, pend_tok[i], pend_pos[i], 0, false);
                if (llama_decode(ctx, batch) != 0) break;
                n_pend = 0;
            }

            if (k > 0 && !mtp_fired) { /* lookup fire */ }
            if (k > 0 && n_pend == 0 && mode != MODE_BASE) {
                llama_memory_seq_rm(mem, 1, 0, -1);
                llama_memory_seq_cp(mem, 0, 1, 0, -1);     // checkpoint S(p)
            }

            batch_clear(&batch);
            for (int i = 0; i < n_pend; i++)
                batch_add(&batch, pend_tok[i], pend_pos[i], 0, false);
            const int base = n_pend;
            batch_add(&batch, tok_last, p, 0, true);
            for (int i = 0; i < k; i++)
                batch_add(&batch, draft[i], p + 1 + i, 0, true);

            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "Decode failed\n"); break; }
            stats.n_steps++;
            if (k > 0 && !mtp_fired) stats.ng_fire += k;

            llama_token tok_last_old = tok_last;
            int e = 0;
            bool stop = false, aborted = false;
            for (int i = 0; i <= k; i++) {
                llama_token a = (llama_token)sample_pos(llama_get_logits_ith(ctx, base + i),
                                                        G.n_vocab, temperature, seed, p + i + 1);
                if (!llama_vocab_is_eog(G.vocab, a)) {
                    if (!sink(token_to_str(a))) { aborted = true; }
                }
                hist[n_hist++] = a;
                stats.n_gen++; e++;
                tok_last = a;
                if (aborted || llama_vocab_is_eog(G.vocab, a) || stats.n_gen >= max_new) { stop = true; break; }
                if (i < k && draft[i] != a) break;
            }
            int acc = e - 1;
            if (k > 0 && mtp_fired) {
                stats.mtp_fire++;
                stats.mtp_acc += acc;
            } else if (k > 0) {
                stats.ng_acc += acc;
                if (acc == k) {
                    L_dyn = L_dyn > G.args.ngram_L ? L_dyn - 1 : G.args.ngram_L;
                    k_cur = k_cur <= G.args.draft_k ? 15 : MAX_DRAFT;
                } else {
                    L_dyn += 4;
                    k_cur = G.args.draft_k;
                    if (acc == 0) n_zero++;
                }
            }

            if (ctx_mtp && !stop) {
                mtp_cand = mtp_feed1(ctx_mtp, ctx, &mtp_batch, tok_last,
                                     base + e - 1, p + e, G.n_embd, G.n_vocab);
            }

            if (stop) { p += e; break; }

            if (e == k + 1 || k == 0) {
                n_pend = 0;
                p += e;
            } else {
                llama_memory_seq_rm(mem, 0, 0, -1);
                llama_memory_seq_cp(mem, 1, 0, 0, -1);
                pend_tok[n_pend] = tok_last_old;  pend_pos[n_pend] = p;  n_pend++;
                for (int i = 0; i < e - 1; i++) {
                    pend_tok[n_pend] = draft[i];  pend_pos[n_pend] = p + 1 + i;  n_pend++;
                }
                p += e;
            }
        }

        double elapsed = now_sec() - t0;
        stats.tok_s = elapsed > 0 ? stats.n_gen / elapsed : 0;

        if (ctx_mtp) {
            free(mtp_batch.token);
            mtp_batch.token = nullptr;
            llama_batch_free(mtp_batch);
        }
    }

    fprintf(stderr, "  → [%s] %d tok in %d steps (%.2f tok/step)  %.1f tok/s",
            stats.mode, stats.n_gen, stats.n_steps,
            stats.n_steps ? (double)stats.n_gen / stats.n_steps : 0.0, stats.tok_s);
    if (stats.ng_fire)  fprintf(stderr, "  lookup %d/%d", stats.ng_acc, stats.ng_fire);
    if (stats.mtp_fire) fprintf(stderr, "  mtp %d/%d (%.0f%%)",
                                stats.mtp_acc, stats.mtp_fire,
                                100.0 * stats.mtp_acc / stats.mtp_fire);
    fprintf(stderr, "\n");

done_early:
    llama_batch_free(batch);
    if (ctx_mtp) llama_free(ctx_mtp);
    llama_free(ctx);
    return stats;
}

// ── chat template ─────────────────────────────────────────────────────────────

static std::string messages_to_prompt(const json &messages) {
    std::vector<llama_chat_message> chat;
    std::vector<std::string> roles, contents;
    for (auto &m : messages) {
        roles.push_back(m.value("role", "user"));
        contents.push_back(m.value("content", ""));
        chat.push_back({roles.back().c_str(), contents.back().c_str()});
    }
    int32_t needed = llama_chat_apply_template(G.tmpl, chat.data(), chat.size(),
                                               /*add_ass=*/true, nullptr, 0);
    if (needed < 0) {
        std::string raw;
        for (auto &m : messages) raw += m.value("content", "") + "\n";
        return raw;
    }
    std::vector<char> buf(needed + 1);
    llama_chat_apply_template(G.tmpl, chat.data(), chat.size(),
                              /*add_ass=*/true, buf.data(), (int32_t)buf.size());
    return std::string(buf.data(), needed);
}

// ── SSE formatting ────────────────────────────────────────────────────────────

static std::string make_id() {
    return "chatcmpl-pti-" + std::to_string((uint32_t)time(nullptr));
}

static std::string sse_chunk(const std::string &text, const std::string &id) {
    json delta  = {{"content", text}};
    json choice = {{"delta", delta}, {"index", 0}, {"finish_reason", nullptr}};
    json chunk  = {{"id", id}, {"object", "chat.completion.chunk"},
                   {"model", "pti-server"}, {"choices", json::array({choice})}};
    return "data: " + chunk.dump() + "\n\n";
}

static std::string sse_done(const GenStats &s, const std::string &id) {
    json choice = {{"delta", json::object()}, {"index", 0}, {"finish_reason", "stop"}};
    json chunk  = {{"id", id}, {"object", "chat.completion.chunk"},
                   {"model", "pti-server"},
                   {"choices", json::array({choice})},
                   {"pti", {{"mode", s.mode}, {"tok_s", s.tok_s},
                            {"tokens", s.n_gen}, {"steps", s.n_steps}}}};
    return "data: " + chunk.dump() + "\n\ndata: [DONE]\n\n";
}

// ── HTTP handlers ─────────────────────────────────────────────────────────────

static void handle_health(const httplib::Request &, httplib::Response &res) {
    res.set_content("{\"status\":\"ok\"}", "application/json");
}

static void handle_chat_completions(const httplib::Request &req, httplib::Response &res) {
    bool expected = false;
    if (!G_busy.compare_exchange_strong(expected, true)) {
        res.status = 503;
        res.set_content("{\"error\":\"server busy\"}", "application/json");
        return;
    }

    json j;
    try { j = json::parse(req.body); }
    catch (...) {
        G_busy.store(false);
        res.status = 400;
        res.set_content("{\"error\":\"invalid JSON\"}", "application/json");
        return;
    }
    if (!j.contains("messages") || j["messages"].empty()) {
        G_busy.store(false);
        res.status = 400;
        res.set_content("{\"error\":\"messages required\"}", "application/json");
        return;
    }

    std::string prompt      = messages_to_prompt(j["messages"]);
    float       temperature = j.value("temperature", G.args.temperature);
    int         max_tokens  = j.value("max_tokens",  G.args.max_tokens);
    // seed: honor the OpenAI-style request field; absent → time-derived
    uint64_t    seed        = j.contains("seed") ? (uint64_t)j["seed"].get<int64_t>()
                                                 : (uint64_t)now_sec() * 1000003ull;
    std::string id          = make_id();

    // per-request mode override: {"pti_mode": "base"|"mtp"|"pti"}
    int mode = G.args.mode;
    if (j.contains("pti_mode")) {
        std::string m = j.value("pti_mode", "");
        if      (m == "base") mode = MODE_BASE;
        else if (m == "mtp")  mode = MODE_MTP;
        else if (m == "pti")  mode = MODE_PTI;
    }

    fprintf(stderr, "[%s] mode=%s temp=%.2f seed=%llu max_tokens=%d\n",
            id.c_str(), mode_name(mode), temperature,
            (unsigned long long)seed, max_tokens);

    res.set_chunked_content_provider("text/event-stream",
        [prompt, temperature, seed, max_tokens, mode, id](size_t /*offset*/,
                                                           httplib::DataSink &sink) -> bool {
            GenStats stats = run_generate(prompt, max_tokens, temperature, seed, mode,
                [&sink, &id](const std::string &text) -> bool {
                    auto chunk = sse_chunk(text, id);
                    return sink.write(chunk.c_str(), chunk.size());
                });
            auto done = sse_done(stats, id);
            sink.write(done.c_str(), done.size());
            sink.done();
            G_busy.store(false);
            return true;
        });
}

// ── main ──────────────────────────────────────────────────────────────────────

static void usage(const char *prog) {
    fprintf(stderr,
        "PTI Server — speculative decoding (lookup + MTP), OpenAI-compatible\n\n"
        "Usage: %s -m <model.gguf> [options]\n\n"
        "  -m <path>      Model path (required; needs nextn head for MTP modes)\n"
        "  -p <port>      HTTP port (default: 8080)\n"
        "  -c <int>       Context size (default: 98304 = 49k usable; f16 KV)\n"
        "  -ngl <int>     GPU layers (default: 99)\n"
        "  --kv-q8        Q8_0 KV cache: ~2x context headroom, but output may\n"
        "                 differ between modes (batch-size-dependent numerics)\n"
        "  --mode <m>     base | mtp | pti (default: pti)\n"
        "  --verbose      Full llama/ggml logs (default: WARN+ only)\n"
        "  --temp <float> Default temperature (default: 0.0 = greedy; >0 uses\n"
        "                 sampled verification — speculation stays active)\n"
        "  -n <int>       Default max tokens (default: 1024)\n\n"
        "Endpoints:\n"
        "  GET  /v1, /health           connectivity check\n"
        "  POST /v1/chat/completions   SSE streaming (OpenAI chat format)\n"
        "                              optional body field: \"pti_mode\": \"base|mtp|pti\"\n\n"
        "Every request logs mode, tok/s and accept rates to this console.\n",
        prog);
}

int main(int argc, char **argv) {
    ServerArgs &args = G.args;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-m")     && i+1 < argc) strncpy(args.model_path, argv[++i], sizeof(args.model_path)-1);
        else if (!strcmp(argv[i], "-p")     && i+1 < argc) args.port         = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c")     && i+1 < argc) args.n_ctx        = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-ngl")   && i+1 < argc) args.n_gpu_layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp") && i+1 < argc) args.temperature  = atof(argv[++i]);
        else if (!strcmp(argv[i], "-n")     && i+1 < argc) args.max_tokens   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kv-q8")) args.kv_q8 = true;
        else if (!strcmp(argv[i], "--verbose")) g_verbose_logs = true;
        else if (!strcmp(argv[i], "--mode") && i+1 < argc) {
            const char *m = argv[++i];
            args.mode = !strcmp(m, "base") ? MODE_BASE : !strcmp(m, "mtp") ? MODE_MTP : MODE_PTI;
        }
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
    }

    if (args.model_path[0] == '\0') {
        fprintf(stderr, "Error: -m <model_path> required\n\n");
        usage(argv[0]); return 1;
    }

    llama_log_set(pti_log_cb, nullptr);
    llama_backend_init();

    fprintf(stderr, "Loading: %s\n", args.model_path);
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = args.n_gpu_layers;
    G.model = llama_model_load_from_file(args.model_path, mparams);
    if (!G.model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    G.vocab   = llama_model_get_vocab(G.model);
    G.n_vocab = llama_vocab_n_tokens(G.vocab);
    G.n_embd  = llama_model_n_embd(G.model);
    G.tmpl    = llama_model_chat_template(G.model, /*name=*/nullptr);

    fprintf(stderr, "Model ready. vocab=%d n_embd=%d template=%s\n",
            G.n_vocab, G.n_embd, G.tmpl ? "(from model)" : "(none — raw prompt)");

    httplib::Server svr;
    svr.Get("/v1",                   handle_health);
    svr.Get("/health",               handle_health);
    svr.Post("/v1/chat/completions", handle_chat_completions);

    fprintf(stderr,
        "\nPTI server on http://0.0.0.0:%d\n"
        "  Mode        : %s   (per-request override: \"pti_mode\")\n"
        "  Temperature : %.2f (sampled verification: speculation active at any temp)\n"
        "  Context     : %d tokens\n"
        "  KV cache    : %s (usable ctx = n_ctx/2 — checkpoint stream)\n\n",
        args.port, mode_name(args.mode), args.temperature, args.n_ctx,
        args.kv_q8 ? "Q8_0 — NOTE: output may differ between modes" : "f16 (exact across modes)");

    svr.listen("0.0.0.0", args.port);

    llama_model_free(G.model);
    llama_backend_free();
    return 0;
}
