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
#include "chat.h"                          // llama.cpp common: jinja templates + tool-call parsing (M7.7)

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <functional>
#include <string>
#include <vector>

// json alias comes from llama.cpp common (chat.h): nlohmann::ordered_json

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
    bool    strip_reasoning = true;    // <think> → message.reasoning_content (clean agent history)
    bool    no_think        = false;   // template enable_thinking=false (faster agent loops)
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

// jinja chat templates + tool-call parser (llama.cpp common). Falls back to
// the builtin llama_chat_apply_template path when init/apply fails.
static common_chat_templates_ptr G_chat_tmpls;
static bool G_common_chat_ok = false;

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
                       int g, int L_min, int k, llama_token *out, int *L_out = nullptr) {
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
        if (L_out) *L_out = L;
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


// ── Qwen-Coder XML tool-call fallback (M7.8) ─────────────────────────────────
// This model family often emits <function=NAME><parameter=KEY>VALUE</parameter>
// ...</function> (optionally wrapped in <tool_call> tags) regardless of the
// template's declared JSON format; the template-derived PEG parser then fails.
// Extract such calls and strip them from the visible content.
static bool parse_xml_function_calls(const std::string &text,
                                     json &tool_calls_out,
                                     std::string &content_out) {
    tool_calls_out = json::array();
    content_out.clear();
    size_t pos = 0, copied = 0;
    int ctr = 0;
    while (true) {
        size_t f = text.find("<function=", pos);
        if (f == std::string::npos) break;
        size_t name_end = text.find('>', f);
        if (name_end == std::string::npos) break;
        std::string name = text.substr(f + 10, name_end - (f + 10));
        size_t fend = text.find("</function>", name_end);
        if (fend == std::string::npos) break;            // incomplete: stop
        std::string body = text.substr(name_end + 1, fend - name_end - 1);

        json args = json::object();
        size_t bp = 0;
        while (true) {
            size_t p0 = body.find("<parameter=", bp);
            if (p0 == std::string::npos) break;
            size_t k_end = body.find('>', p0);
            if (k_end == std::string::npos) break;
            std::string key = body.substr(p0 + 11, k_end - (p0 + 11));
            size_t v_end = body.find("</parameter>", k_end);
            if (v_end == std::string::npos) break;
            std::string val = body.substr(k_end + 1, v_end - k_end - 1);
            if (!val.empty() && val.front() == '\n') val.erase(0, 1);
            if (!val.empty() && val.back()  == '\n') val.pop_back();
            args[key] = val;
            bp = v_end + 12;
        }

        // strip an enclosing <tool_call> wrapper if present
        size_t block_start = f, block_end = fend + 11;
        size_t tc = text.rfind("<tool_call>", f);
        if (tc != std::string::npos && text.find_first_not_of(" \t\n", tc + 11) == f)
            block_start = tc;
        size_t tce = text.find("</tool_call>", block_end);
        if (tce != std::string::npos && text.find_first_not_of(" \t\n", block_end) == tce)
            block_end = tce + 12;

        content_out += text.substr(copied, block_start - copied);
        copied = block_end;
        pos    = block_end;

        tool_calls_out.push_back(json{
            {"index", ctr}, {"id", "call_x" + std::to_string(++ctr) + "_" + std::to_string((uint32_t)time(nullptr))},
            {"type", "function"},
            {"function", {{"name", name}, {"arguments", args.dump()}}}});
    }
    content_out += text.substr(copied);
    return !tool_calls_out.empty();
}

// remove <think>...</think> spans; returns stripped content, appends reasoning
static std::string strip_think(const std::string &text, std::string &reasoning) {
    std::string out;
    size_t pos = 0;
    while (true) {
        size_t t0 = text.find("<think>", pos);
        if (t0 == std::string::npos) { out += text.substr(pos); break; }
        out += text.substr(pos, t0 - pos);
        size_t t1 = text.find("</think>", t0);
        if (t1 == std::string::npos) { reasoning += text.substr(t0 + 7); break; }
        reasoning += text.substr(t0 + 7, t1 - (t0 + 7));
        pos = t1 + 8;
        while (pos < text.size() && text[pos] == '\n') pos++;   // eat the blank line
    }
    return out;
}

// ── generation ────────────────────────────────────────────────────────────────

struct GenStats {
    int    n_gen      = 0;
    int    n_prompt   = 0;
    int    n_steps    = 0;
    int    ng_fire    = 0, ng_acc  = 0;   // lookup drafts proposed/accepted
    int    mtp_fire   = 0, mtp_acc = 0;   // MTP drafts fired/accepted
    int    n_veto     = 0;                // n-gram fires vetoed by MTP disagreement
    double prefill_s  = 0.0;
    double tok_s      = 0.0;
    const char *mode  = "base";
};

using TokenSink = std::function<bool(const std::string &)>;

// ── persistent generation contexts + prompt cache (M7.5) ────────────────────
// Contexts are created once and reused; seq 0 holds the previous request's
// state so conversation EXTENSIONS prefill only the delta. Hybrid-SSM cannot
// rewind, so any divergence from the cached stream = full re-prefill.
struct GenCache {
    llama_context *ctx     = nullptr;
    llama_context *ctx_mtp = nullptr;
    llama_memory_t mem     = nullptr;
    std::vector<llama_token> toks;   // prompt+emitted stream of the last request
    int  clean_c = 0;                // seq0 state is clean through position clean_c;
                                     // toks[clean_c..) are correct but unconsumed
    bool valid   = false;
    // v2 (M7.5b): prompt-base checkpoint. seq 2 holds the state at "base" =
    // the templated prompt MINUS the volatile generation-prompt suffix. Agent
    // loops extend the base every turn even when think-prefills/tool history
    // re-render differently than the lived stream (measured: lived-extension
    // diverges at the genprefill, token 19/23).
    std::vector<llama_token> base_toks;
    bool base_valid = false;
} GC;

static bool ensure_gen_ctx() {
    if (GC.ctx) return true;

    // IDENTICAL config for ALL modes (n_seq_max, pre-norm): different cache
    // layouts shift flash-attention numerics enough to flip near-tie tokens.
    // kv_unified MUST stay false and KV defaults to f16 — both measured to
    // break cross-mode byte-identity otherwise (see KERNEL_PLAN M7.2).
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx        = (uint32_t)G.args.n_ctx;
    cparams.n_batch      = (uint32_t)G.args.n_batch;
    cparams.n_ubatch     = (uint32_t)G.args.n_ubatch;
    cparams.n_seq_max    = 3;                            // working + spec ckpt + base ckpt
    cparams.kv_unified   = false;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (G.args.kv_q8) {
        cparams.type_k = GGML_TYPE_Q8_0;
        cparams.type_v = GGML_TYPE_Q8_0;
    }

    GC.ctx = llama_init_from_model(G.model, cparams);
    if (!GC.ctx) { fprintf(stderr, "Failed to create context\n"); return false; }
    GC.mem = llama_get_memory(GC.ctx);
    llama_set_embeddings_pre_norm(GC.ctx, true, false);

    llama_context_params mp = llama_context_default_params();
    mp.n_ctx     = (uint32_t)G.args.n_ctx;
    mp.n_batch   = (uint32_t)G.args.n_batch;
    mp.n_seq_max = 1;
    mp.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
    mp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    GC.ctx_mtp = llama_init_from_model(G.model, mp);
    if (!GC.ctx_mtp)
        fprintf(stderr, "  [note] MTP context unavailable (no nextn head?) — lookup only\n");
    return true;
}

static GenStats run_generate(const std::string &prompt, const std::string &gen_prompt,
                             int max_new, float temperature, uint64_t seed, int mode,
                             const TokenSink &sink) {
    GenStats stats;
    stats.mode = mode_name(mode);

    if (!ensure_gen_ctx()) return stats;
    llama_context *ctx     = GC.ctx;
    llama_memory_t mem     = GC.mem;
    llama_context *ctx_mtp = (mode >= MODE_MTP) ? GC.ctx_mtp : nullptr;

    // ── tokenize + prefill ───────────────────────────────────────────────────
    std::vector<llama_token> hist(MAX_PROMPT_TOKENS);
    int n_hist = llama_tokenize(G.vocab, prompt.c_str(), (int32_t)prompt.size(),
                                hist.data(), MAX_PROMPT_TOKENS, true, true);
    if (n_hist <= 0) {
        fprintf(stderr, "Tokenize failed\n");
        return stats;
    }
    // ── budget: clamp generation to the per-stream context ───────────────────
    int usable = G.args.n_ctx / 3;                // non-unified: 3 streams
    if (n_hist + MAX_DRAFT + 8 >= usable) {
        fprintf(stderr, "  [error] prompt (%d tok) exceeds usable context (%d = n_ctx/3)\n",
                n_hist, usable);
        return stats;
    }
    if (n_hist + max_new + MAX_DRAFT + 8 > usable) {
        int clamped = usable - n_hist - MAX_DRAFT - 8;
        fprintf(stderr, "  [note] max_tokens %d clamped to %d (usable ctx %d - prompt %d)\n",
                max_new, clamped, usable, n_hist);
        max_new = clamped;
    }
    hist.resize((size_t)n_hist + max_new + 8);
    stats.n_prompt = n_hist;
    fprintf(stderr, "  prompt: %d tokens  mode: %s\n", n_hist, stats.mode);

    // ── new base = templated prompt minus the generation-prompt suffix ──────
    int nb_new = n_hist;                        // base end (token index)
    if (!gen_prompt.empty() && prompt.size() >= gen_prompt.size()
        && prompt.compare(prompt.size() - gen_prompt.size(), gen_prompt.size(), gen_prompt) == 0) {
        std::string base_str = prompt.substr(0, prompt.size() - gen_prompt.size());
        std::vector<llama_token> bt(MAX_PROMPT_TOKENS);
        int n = llama_tokenize(G.vocab, base_str.c_str(), (int32_t)base_str.size(),
                               bt.data(), MAX_PROMPT_TOKENS, true, true);
        // tokenization prefix property must hold or the base is unusable
        if (n > 0 && n <= n_hist && std::equal(bt.begin(), bt.begin() + n, hist.begin()))
            nb_new = n;
    }

    // ── layered cache match: lived-stream extension → base extension → miss ─
    int start = 0;
    const char *how = "cold";
    if (GC.valid
        && n_hist >= (int)GC.toks.size()
        && GC.clean_c < (int)GC.toks.size()
        && std::equal(GC.toks.begin(), GC.toks.end(), hist.begin())) {
        start = GC.clean_c;
        how   = "hit-lived";
    } else if (GC.base_valid
        && !GC.base_toks.empty()
        && n_hist >= (int)GC.base_toks.size()
        && (int)GC.base_toks.size() <= nb_new
        && std::equal(GC.base_toks.begin(), GC.base_toks.end(), hist.begin())) {
        llama_memory_seq_rm(mem, 0, 0, -1);
        llama_memory_seq_cp(mem, 2, 0, 0, -1);   // restore base state
        llama_memory_seq_rm(mem, 1, 0, -1);
        if (GC.ctx_mtp) llama_memory_seq_rm(llama_get_memory(GC.ctx_mtp), 0, 0, -1);
        start = (int)GC.base_toks.size();
        how   = "hit-base";
    } else {
        llama_memory_seq_rm(mem, 0, 0, -1);
        llama_memory_seq_rm(mem, 1, 0, -1);
        llama_memory_seq_rm(mem, 2, 0, -1);
        if (GC.ctx_mtp) llama_memory_seq_rm(llama_get_memory(GC.ctx_mtp), 0, 0, -1);
        if (GC.valid || GC.base_valid) how = "miss";
    }
    fprintf(stderr, "  [cache] %s: reused %d, prefilling %d (base at %d)\n",
            how, start, n_hist - start, nb_new);
    GC.valid = false;        // re-published at finalize
    GC.base_valid = false;   // re-published at the nb_new crossing below

    // ── prefill [start, n_hist), chunked; checkpoint the base at nb_new ─────
    int chunk = G.args.n_batch;
    llama_batch batch = llama_batch_init(chunk + 160, 0, 3);
    int last_idx = 0;
    bool prefill_ok = true;
    double t_pf0 = now_sec();
    {
        int i0 = start;
        if (i0 >= nb_new && nb_new >= 0) {
            // base already covered by reuse: state at start ≥ nb... only safe
            // to (re)publish when start == nb_new exactly (state == S(nb_new))
            if (i0 == nb_new) {
                llama_memory_seq_rm(mem, 2, 0, -1);
                llama_memory_seq_cp(mem, 0, 2, 0, -1);
                GC.base_toks.assign(hist.begin(), hist.begin() + nb_new);
                GC.base_valid = true;
            }
        }
        while (i0 < n_hist && prefill_ok) {
            int lim = (i0 < nb_new && nb_new < n_hist) ? nb_new : n_hist;  // break at base
            int nbt = std::min(chunk, lim - i0);
            batch_clear(&batch);
            for (int j = 0; j < nbt; j++)
                batch_add(&batch, hist[i0 + j], (llama_pos)(i0 + j), 0, i0 + j == n_hist - 1);
            if (llama_decode(ctx, batch) != 0) { prefill_ok = false; break; }
            last_idx = nbt - 1;
            i0 += nbt;
            if (i0 == nb_new) {                  // crossing: seq0 == S(nb_new)
                llama_memory_seq_rm(mem, 2, 0, -1);
                llama_memory_seq_cp(mem, 0, 2, 0, -1);
                GC.base_toks.assign(hist.begin(), hist.begin() + nb_new);
                GC.base_valid = true;
            }
        }
    }
    stats.prefill_s = now_sec() - t_pf0;
    if (!prefill_ok) {
        fprintf(stderr, "Prefill failed\n");
        GC.base_valid = false;
        llama_batch_free(batch);
        return stats;
    }

    llama_pos   p        = (llama_pos)n_hist;
    llama_token tok_last = (llama_token)sample_pos(llama_get_logits_ith(ctx, last_idx),
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
                                 last_idx, p, G.n_embd, G.n_vocab);
        }

        // ── speculative loop state (pti_lookup.cpp, M6.4c) ──────────────────
        int L_dyn  = G.args.ngram_L;
        int k_cur  = G.args.draft_k;
        int n_zero = 0;
        llama_token pend_tok[128];
        llama_pos   pend_pos[128];
        int n_pend = 0;
        llama_pos ckpt_p = -1;       // position of the last checkpoint refresh
        bool stop_dirty = false;     // stopped mid-batch with wrong tail consumed

        double t0 = now_sec();

        while (stats.n_gen < max_new && !llama_vocab_is_eog(G.vocab, tok_last)) {
            llama_token draft[MAX_DRAFT];
            int k = 0;
            int fire_L = 0;
            bool mtp_fired = false;

            if (mode == MODE_PTI && n_zero < 3)
                k = ngram_draft(hist.data(), n_hist, G.args.ngram_g, L_dyn, k_cur, draft, &fire_L);

            bool mtp_alive = ctx_mtp && mtp_cand != -1 && mode >= MODE_MTP
                          && !(stats.mtp_fire >= 10 && stats.mtp_acc * 10 < stats.mtp_fire * 3);

            // MTP arbitration (M7.3, L-aware M7.6): MTP is ~89% right, so an
            // unconditional veto discards ~11% of GOOD fires — costly when
            // fires are rare and land 7-31 tokens. Veto only MARGINAL fires
            // (suffix match < L_TRUST); long matches override the MTP vote.
            constexpr int L_TRUST = 10;
            if (k > 0 && mtp_alive && k_cur <= G.args.draft_k
                && fire_L < L_TRUST && mtp_cand != draft[0]) {
                k = 0;
                stats.n_veto++;
            }

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
                ckpt_p = p;
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

            if (stop) { stop_dirty = (k > 0 && e < k + 1); p += e; break; }

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

        // ── prompt-cache finalize (M7.5): publish a known-clean state ───────
        // clean path: seq0 = S(p), everything in hist consumed except the
        // final emitted token. pending/dirty paths: state == / restored-to
        // the checkpoint; hist[ckpt_p..) are correct-but-unconsumed.
        {
            int clean_c = -1;
            if (stop_dirty) {
                if (ckpt_p >= 0) {
                    llama_memory_seq_rm(mem, 0, 0, -1);
                    llama_memory_seq_cp(mem, 1, 0, 0, -1);
                    clean_c = (int)ckpt_p;
                }
            } else if (n_pend > 0) {
                clean_c = (int)ckpt_p;
            } else {
                clean_c = (int)p;
            }
            if (clean_c >= 0 && clean_c < n_hist) {
                GC.toks.assign(hist.begin(), hist.begin() + n_hist);
                GC.clean_c = clean_c;
                GC.valid   = true;
            }
        }

        if (ctx_mtp) {
            free(mtp_batch.token);
            mtp_batch.token = nullptr;
            llama_batch_free(mtp_batch);
        }
    }

    fprintf(stderr, "  → [%s] %d tok in %d steps (%.2f tok/step)  %.1f tok/s  (prefill %.1fs)",
            stats.mode, stats.n_gen, stats.n_steps,
            stats.n_steps ? (double)stats.n_gen / stats.n_steps : 0.0, stats.tok_s,
            stats.prefill_s);
    if (stats.ng_fire)  fprintf(stderr, "  lookup %d/%d", stats.ng_acc, stats.ng_fire);
    if (stats.mtp_fire) fprintf(stderr, "  mtp %d/%d (%.0f%%)",
                                stats.mtp_acc, stats.mtp_fire,
                                100.0 * stats.mtp_acc / stats.mtp_fire);
    if (stats.n_veto)   fprintf(stderr, "  vetoed %d", stats.n_veto);
    fprintf(stderr, "\n");

done_early:
    llama_batch_free(batch);
    (void)ctx; (void)ctx_mtp;        // persistent (GC) — not freed per request
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

static std::string sse_done(const GenStats &s, const std::string &id,
                            const std::string &finish = "stop") {
    json choice = {{"delta", json::object()}, {"index", 0}, {"finish_reason", finish}};
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

    float       temperature = j.value("temperature", G.args.temperature);
    int         max_tokens  = j.value("max_tokens",  G.args.max_tokens);
    bool        stream      = j.value("stream", false);     // OpenAI default: NOT streamed
    bool        has_tools   = j.contains("tools") && j["tools"].is_array() && !j["tools"].empty();
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

    // ── templating (M7.7): jinja with tools when available ──────────────────
    std::string prompt;
    common_chat_params chat_params;
    bool use_common = false;
    if (G_common_chat_ok) {
        try {
            common_chat_templates_inputs in;
            auto omsgs  = nlohmann::ordered_json::parse(j.at("messages").dump());
            in.messages = common_chat_msgs_parse_oaicompat(omsgs);
            if (has_tools) {
                auto otools = nlohmann::ordered_json::parse(j["tools"].dump());
                in.tools    = common_chat_tools_parse_oaicompat(otools);
            }
            in.add_generation_prompt = true;
            in.use_jinja             = true;
            in.parallel_tool_calls   = j.value("parallel_tool_calls", false);
            in.enable_thinking       = !G.args.no_think;
            chat_params = common_chat_templates_apply(G_chat_tmpls.get(), in);
            prompt      = chat_params.prompt;
            use_common  = true;
        } catch (const std::exception &e) {
            fprintf(stderr, "  [warn] jinja apply failed (%s) — builtin template, tools unparsed\n", e.what());
        }
    }
    if (!use_common) prompt = messages_to_prompt(j["messages"]);
    std::string gen_prompt = use_common ? chat_params.generation_prompt : std::string{};

    fprintf(stderr, "[%s] mode=%s temp=%.2f seed=%llu max_tokens=%d stream=%d tools=%d\n",
            id.c_str(), mode_name(mode), temperature,
            (unsigned long long)seed, max_tokens, (int)stream, (int)has_tools);

    // parse the finished text: template parser → XML-function fallback → raw.
    // Reasoning is stripped to message.reasoning_content unless --reasoning none.
    auto parse_final = [chat_params, use_common](const std::string &full,
                                                 json &message, std::string &finish) {
        message = json{{"role", "assistant"}, {"content", full}};
        finish  = "stop";
        bool parsed = false;
        if (use_common) {
            try {
                common_chat_parser_params pp(chat_params);
                try { pp.parser.load(chat_params.parser); } catch (...) {}
                if (G.args.strip_reasoning) pp.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
                common_chat_msg msg = common_chat_parse(full, /*is_partial=*/false, pp);
                std::vector<std::string> ids;
                int ctr = 0;
                msg.set_tool_call_ids(ids, [&]() { return "call_" + std::to_string(++ctr) + "_" + std::to_string((uint32_t)time(nullptr)); });
                if (!msg.tool_calls.empty()) {
                    message = json::parse(msg.to_json_oaicompat().dump());
                    finish  = "tool_calls";
                    parsed  = true;
                } else if (!msg.content.empty() || !msg.reasoning_content.empty()) {
                    message = json::parse(msg.to_json_oaicompat().dump());
                    parsed  = true;   // may still get XML-fallback below if markup present
                }
            } catch (const std::exception &e) {
                std::string what = e.what();
                if (what.size() > 160) what = what.substr(0, 160) + "...";
                fprintf(stderr, "  [note] template parse failed (%s) — trying XML fallback\n", what.c_str());
            }
        }
        // XML-function fallback: fires when the model used <function=...> markup
        // that the template parser didn't yield tool calls for.
        if (finish != "tool_calls" && full.find("<function=") != std::string::npos) {
            json tcs; std::string stripped;
            if (parse_xml_function_calls(full, tcs, stripped)) {
                std::string reasoning;
                if (G.args.strip_reasoning) stripped = strip_think(stripped, reasoning);
                message = json{{"role", "assistant"},
                               {"content", stripped},
                               {"tool_calls", tcs}};
                if (!reasoning.empty()) message["reasoning_content"] = reasoning;
                finish = "tool_calls";
                fprintf(stderr, "  [note] XML tool-call fallback: %d call(s) extracted\n",
                        (int)tcs.size());
                return;
            }
        }
        if (!parsed && G.args.strip_reasoning) {
            std::string reasoning;
            std::string stripped = strip_think(full, reasoning);
            message["content"] = stripped;
            if (!reasoning.empty()) message["reasoning_content"] = reasoning;
        }
        // template parser does not always extract <think> (e.g. empty blocks):
        // enforce the strip on whatever content we are about to return
        if (G.args.strip_reasoning && message.contains("content") && message["content"].is_string()
            && message["content"].get<std::string>().find("<think>") != std::string::npos) {
            std::string reasoning;
            message["content"] = strip_think(message["content"].get<std::string>(), reasoning);
            if (!reasoning.empty()) message["reasoning_content"] = reasoning;
        }
    };

    if (!stream) {
        // ── non-streamed completion (LangChain/agent clients) ───────────────
        // Sent as chunked JSON with a whitespace heartbeat every ~5 s during
        // generation: leading whitespace is valid JSON, and it keeps client
        // fetch timeouts (nanocoder default: 120 s) from killing long turns.
        res.set_chunked_content_provider("application/json",
            [prompt, gen_prompt, temperature, seed, max_tokens, mode, id, parse_final](
                    size_t /*offset*/, httplib::DataSink &sink) -> bool {
                std::string full;
                double last_beat = now_sec();
                GenStats stats = run_generate(prompt, gen_prompt, max_tokens, temperature, seed, mode,
                    [&](const std::string &text) -> bool {
                        full += text;
                        double now = now_sec();
                        if (now - last_beat > 5.0) {
                            last_beat = now;
                            if (!sink.write(" ", 1)) return false;   // heartbeat
                        }
                        return true;
                    });

                json message; std::string finish;
                parse_final(full, message, finish);

                json resp = {
                    {"id", id}, {"object", "chat.completion"}, {"created", (int64_t)time(nullptr)},
                    {"model", "pti-server"},
                    {"choices", json::array({ json{{"index", 0}, {"message", message}, {"finish_reason", finish}} })},
                    {"usage", {{"prompt_tokens", stats.n_prompt}, {"completion_tokens", stats.n_gen},
                               {"total_tokens", stats.n_prompt + stats.n_gen}}},
                    {"pti", {{"mode", stats.mode}, {"tok_s", stats.tok_s}, {"steps", stats.n_steps}}}
                };
                std::string body = resp.dump();
                sink.write(body.c_str(), body.size());
                sink.done();
                G_busy.store(false);
                return true;
            });
        return;
    }

    // ── streamed (SSE): content flows LIVE through a gate that withholds
    // <think> spans (when stripping) and tool-call markup; parsed tool_calls
    // arrive as a structured delta at the end. Nothing buffers needlessly —
    // long turns show progress immediately (no client timeouts).
    res.set_chunked_content_provider("text/event-stream",
        [prompt, gen_prompt, temperature, seed, max_tokens, mode, id, parse_final](
                size_t /*offset*/, httplib::DataSink &sink) -> bool {
            std::string full;          // everything generated (for final parse)
            std::string pend;          // gate holdback
            bool tool_mode = false;    // saw tool markup: withhold the rest
            bool in_think  = false;
            size_t n_streamed = 0;     // visible chars emitted

            auto emit = [&](const std::string &t) -> bool {
                if (t.empty()) return true;
                n_streamed += t.size();
                auto chunk = sse_chunk(t, id);
                return sink.write(chunk.c_str(), chunk.size());
            };

            // gate: emit visible text; divert think spans (if stripping) and
            // stop emission at the first tool marker. Partial tags at the
            // buffer tail are held back until disambiguated.
            auto feed = [&](const std::string &t) -> bool {
                full += t;
                if (tool_mode) return true;
                pend += t;
                for (;;) {
                    size_t tag = std::string::npos;
                    int    which = -1;   // 0=<think> 1=</think> 2=tool
                    const char *tags[] = {"<think>", "</think>", "<tool_call", "<function="};
                    for (int i = 0; i < 4; i++) {
                        size_t p = pend.find(tags[i]);
                        if (p != std::string::npos && p < tag) { tag = p; which = i >= 2 ? 2 : i; }
                    }
                    if (tag == std::string::npos) break;
                    std::string before = pend.substr(0, tag);
                    if (!(in_think && G.args.strip_reasoning)) { if (!emit(before)) return false; }
                    if (which == 0)      { in_think = true;  pend.erase(0, tag + 7); }
                    else if (which == 1) { in_think = false; pend.erase(0, tag + 8);
                                           while (!pend.empty() && pend[0] == '\n') pend.erase(0, 1); }
                    else                 { tool_mode = true;  pend.clear(); return true; }
                }
                // flush all but a possible partial tag at the tail
                size_t hold = pend.size();
                size_t lt = pend.rfind('<');
                if (lt != std::string::npos && pend.size() - lt <= 12) hold = lt;
                if (hold > 0) {
                    std::string out = pend.substr(0, hold);
                    pend.erase(0, hold);
                    if (!(in_think && G.args.strip_reasoning)) { if (!emit(out)) return false; }
                }
                return true;
            };

            GenStats stats = run_generate(prompt, gen_prompt, max_tokens, temperature, seed, mode, feed);

            json message; std::string finish;
            parse_final(full, message, finish);

            // trailing delta: tool_calls and/or any visible content the gate
            // was still holding (e.g. text after a withheld marker that turned
            // out not to be a tool call)
            json delta = json::object();
            if (finish == "tool_calls" && message.contains("tool_calls")) {
                delta["tool_calls"] = message["tool_calls"];
                int idx = 0;
                for (auto &tc : delta["tool_calls"]) if (!tc.contains("index")) tc["index"] = idx++;
            }
            if (message.contains("content") && message["content"].is_string()) {
                std::string final_content = message["content"].get<std::string>();
                if (final_content.size() > n_streamed)
                    delta["content"] = final_content.substr(n_streamed);
            }
            if (!delta.empty()) {
                json choice = {{"delta", delta}, {"index", 0}, {"finish_reason", nullptr}};
                json chunk  = {{"id", id}, {"object", "chat.completion.chunk"},
                               {"model", "pti-server"}, {"choices", json::array({choice})}};
                std::string out = "data: " + chunk.dump() + "\n\n";
                sink.write(out.c_str(), out.size());
            }
            auto done = sse_done(stats, id, finish);
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
        "  -c <int>       Context size (default: 98304 = 32k usable; f16 KV)\n"
        "  -ngl <int>     GPU layers (default: 99)\n"
        "  --kv-q8        Q8_0 KV cache: ~2x context headroom, but output may\n"
        "                 differ between modes (batch-size-dependent numerics)\n"
        "  --mode <m>     base | mtp | pti (default: pti)\n"
        "  --verbose      Full llama/ggml logs (default: WARN+ only)\n"
        "  --no-think     Template enable_thinking=false (faster agent loops)\n"
        "  --reasoning <m> strip (default: <think> → reasoning_content) | none\n"
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

    // llama-server-compatible flag handling: accept the common llama-server
    // spellings, map what applies, warn-and-ignore what doesn't, and never
    // die on an unknown flag (editor launchers pass their own sets).
    int kv_q8_votes = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      ((!strcmp(a, "-m") || !strcmp(a, "--model")) && i+1 < argc) strncpy(args.model_path, argv[++i], sizeof(args.model_path)-1);
        else if ((!strcmp(a, "-p") || !strcmp(a, "--port"))  && i+1 < argc) args.port  = atoi(argv[++i]);
        else if ((!strcmp(a, "-c") || !strcmp(a, "--ctx-size")) && i+1 < argc) args.n_ctx = atoi(argv[++i]);
        else if ((!strcmp(a, "-ngl") || !strcmp(a, "--n-gpu-layers") || !strcmp(a, "--gpu-layers")) && i+1 < argc) args.n_gpu_layers = atoi(argv[++i]);
        else if ((!strcmp(a, "-b") || !strcmp(a, "--batch-size"))  && i+1 < argc) args.n_batch  = atoi(argv[++i]);
        else if ((!strcmp(a, "-ub") || !strcmp(a, "--ubatch-size")) && i+1 < argc) args.n_ubatch = atoi(argv[++i]);
        else if (!strcmp(a, "--temp") && i+1 < argc) args.temperature = atof(argv[++i]);
        else if ((!strcmp(a, "-n") || !strcmp(a, "--predict") || !strcmp(a, "--n-predict")) && i+1 < argc) args.max_tokens = atoi(argv[++i]);
        else if (!strcmp(a, "--kv-q8")) args.kv_q8 = true;
        else if ((!strcmp(a, "--cache-type-k") || !strcmp(a, "--cache-type-v")) && i+1 < argc) {
            if (!strcmp(argv[++i], "q8_0")) kv_q8_votes++;
        }
        else if (!strcmp(a, "--jinja")) {
            fprintf(stderr, "[note] --jinja accepted: using the model's built-in chat template via\n"
                            "       llama_chat_apply_template (covers Qwen-family templates; full\n"
                            "       Jinja incl. tool-call schemas is not implemented)\n");
        }
        else if ((!strcmp(a, "-t") || !strcmp(a, "--threads")) && i+1 < argc) { ++i; /* full GPU offload: ignored */ }
        else if (!strcmp(a, "--timeout")   && i+1 < argc) ++i;   // accepted, no-op
        else if (!strcmp(a, "--parallel")  && i+1 < argc) {
            if (atoi(argv[++i]) > 1) fprintf(stderr, "[note] --parallel > 1 ignored: single-slot server\n");
        }
        else if (!strcmp(a, "--cache-ram") && i+1 < argc) ++i;   // accepted, no-op
        else if (!strcmp(a, "--no-warmup"))               ;      // accepted, no-op
        else if (!strcmp(a, "--host")      && i+1 < argc) ++i;   // accepted, no-op (binds 0.0.0.0)
        else if (!strcmp(a, "--verbose")) g_verbose_logs = true;
        else if (!strcmp(a, "--no-think")) args.no_think = true;
        else if (!strcmp(a, "--reasoning") && i+1 < argc) args.strip_reasoning = strcmp(argv[++i], "none") != 0;
        else if (!strcmp(a, "--mode") && i+1 < argc) {
            const char *m = argv[++i];
            args.mode = !strcmp(m, "base") ? MODE_BASE : !strcmp(m, "mtp") ? MODE_MTP : MODE_PTI;
        }
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(argv[0]); return 0; }
        else {
            fprintf(stderr, "[warn] unknown arg ignored: %s\n", a);
            if (i+1 < argc && argv[i+1][0] != '-') ++i;   // skip its value too
        }
    }
    if (kv_q8_votes >= 2) {
        args.kv_q8 = true;
        fprintf(stderr, "[note] --cache-type-k/v q8_0 → KV Q8_0 (output may differ between modes)\n");
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
    try {
        G_chat_tmpls = common_chat_templates_init(G.model, "");
        G_common_chat_ok = true;
        fprintf(stderr, "Chat templates: jinja (tools supported)\n");
    } catch (const std::exception &e) {
        fprintf(stderr, "[warn] jinja templates unavailable (%s) — builtin engine, no tools\n", e.what());
    }

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
        "  KV cache    : %s (usable ctx = n_ctx/3 — spec + base checkpoint streams)\n\n",
        args.port, mode_name(args.mode), args.temperature, args.n_ctx,
        args.kv_q8 ? "Q8_0 — NOTE: output may differ between modes" : "f16 (exact across modes)");

    svr.listen("0.0.0.0", args.port);

    llama_model_free(G.model);
    llama_backend_free();
    return 0;
}
