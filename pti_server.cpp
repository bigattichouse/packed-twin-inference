/*
 * pti_server.cpp — PTI-4seq HTTP server
 *
 * Single-user coding demo server. Implements /v1/chat/completions with
 * 4-sequence PTI decode and temperature sampling. Designed for nanocoder
 * integration with Qwen3.6-27B-UD-Q6_K_XL on MI50.
 *
 * Build:  make server
 * Run:    bin/pti_server -m ../gguf/Qwen3.6-27B-UD-Q6_K_XL.gguf
 *
 * Baseline comparison: run llama-server on a different port.
 */

#define CPPHTTPLIB_THREAD_POOL_COUNT 2
// Resolved via -I$(LLAMA_DIR)/vendor added in the Makefile server rule
#include "cpp-httplib/httplib.h"
#include "nlohmann/json.hpp"

// Direct relative path — consistent with pti_4seq.cpp (resolves to ../llama.cpp/)
#include "../llama.cpp/include/llama.h"

#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <random>
#include <string>
#include <vector>

using json = nlohmann::json;

// ── compile-time limits ───────────────────────────────────────────────────────

static constexpr int MAX_PROMPT_TOKENS = 65536;

// ── server args ───────────────────────────────────────────────────────────────

struct ServerArgs {
    char    model_path[512] = {};
    int     port            = 8080;
    int     n_gpu_layers    = 99;
    int     n_ctx           = 32768;
    int     n_batch         = 1024;
    int     n_ubatch        = 256;
    int     pti_streams     = 4;
    float   temperature     = 0.7f;
    int     max_tokens      = 512;
};

// ── global model (loaded once at startup) ────────────────────────────────────

struct PTIGlobal {
    llama_model       *model   = nullptr;
    const llama_vocab *vocab   = nullptr;
    int32_t            n_vocab = 0;
    int32_t            n_embd  = 0;
    const char        *tmpl    = nullptr;   // chat template from model metadata
    ServerArgs         args;
    std::mt19937 rng;
} G;

static std::atomic<bool> G_busy{false};

// ── timing ────────────────────────────────────────────────────────────────────

static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ── sampling ──────────────────────────────────────────────────────────────────

static int32_t argmax_f(const float *v, int32_t n) {
    int32_t best = 0;
    for (int32_t i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    return best;
}

// Temperature sampling. Returns argmax when temp <= 0.
static llama_token sample_token(const float *logits, int32_t n_vocab, float temperature) {
    if (temperature <= 0.0f)
        return (llama_token)argmax_f(logits, n_vocab);

    // Numerically stable softmax with temperature
    float max_l = -FLT_MAX;
    for (int32_t i = 0; i < n_vocab; i++)
        if (logits[i] > max_l) max_l = logits[i];

    std::vector<float> probs(n_vocab);
    float sum = 0.0f;
    for (int32_t i = 0; i < n_vocab; i++) {
        probs[i] = expf((logits[i] - max_l) / temperature);
        sum += probs[i];
    }
    for (int32_t i = 0; i < n_vocab; i++) probs[i] /= sum;

    std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
    return (llama_token)dist(G.rng);
}

// ── batch helpers ─────────────────────────────────────────────────────────────

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

// ── token → UTF-8 string ──────────────────────────────────────────────────────

static std::string token_to_str(llama_token tok) {
    char buf[256];
    int len = llama_token_to_piece(G.vocab, tok, buf, (int32_t)sizeof(buf), 0, true);
    return len > 0 ? std::string(buf, len) : std::string{};
}

// ── chat template ─────────────────────────────────────────────────────────────

// Convert a JSON messages array to a formatted prompt string using the
// model's built-in chat template (llama_model_chat_template + llama_chat_apply_template).
static std::string messages_to_prompt(const json &messages) {
    std::vector<llama_chat_message> chat;
    // Keep role/content strings alive for the duration of this call
    std::vector<std::string> roles, contents;

    for (auto &m : messages) {
        roles.push_back(m.value("role", "user"));
        contents.push_back(m.value("content", ""));
        chat.push_back({roles.back().c_str(), contents.back().c_str()});
    }

    // First call: measure required buffer size
    int32_t needed = llama_chat_apply_template(G.tmpl, chat.data(), chat.size(),
                                               /*add_ass=*/true, nullptr, 0);
    if (needed < 0) {
        fprintf(stderr, "  [warn] llama_chat_apply_template failed, using raw content\n");
        std::string raw;
        for (auto &m : messages) raw += m.value("content", "") + "\n";
        return raw;
    }

    std::vector<char> buf(needed + 1);
    llama_chat_apply_template(G.tmpl, chat.data(), chat.size(),
                              /*add_ass=*/true, buf.data(), (int32_t)buf.size());
    return std::string(buf.data(), needed);
}

// ── PTI state ─────────────────────────────────────────────────────────────────

struct PTIStats {
    int    n_gen    = 0;
    int    n_4acc   = 0;
    int    n_3acc   = 0;
    int    n_2acc   = 0;
    int    n_reject = 0;
    double ss_tok_s = 0.0;   // steady-state tok/s (loop only)
    double am_tok_s = 0.0;   // amortized tok/s (includes init overhead)
};

// Reinitialise sequence dst from src's KV state, decode tok at pos,
// return sampled next token. Used on reject/partial-accept paths.
static llama_token reinit_seq(llama_memory_t mem, llama_context *ctx,
                               llama_batch *b, float temperature,
                               llama_seq_id src, llama_seq_id dst,
                               llama_token tok, llama_pos pos,
                               llama_pos *pos_out) {
    llama_memory_seq_rm(mem, dst, 0, -1);
    llama_memory_seq_cp(mem, src, dst, 0, -1);
    batch_clear(b);
    batch_add(b, tok, pos, dst, true);
    llama_token next = 0;
    if (llama_decode(ctx, *b) == 0)
        next = sample_token(llama_get_logits_ith(ctx, 0), G.n_vocab, temperature);
    *pos_out = pos + 1;
    return next;
}

// ── PTI decode with streaming sink ───────────────────────────────────────────

// Callback: called once per emitted token. Return false to abort generation.
using TokenSink = std::function<bool(const std::string &)>;

static PTIStats run_pti4(const std::string &prompt, int max_new,
                          float temperature, const TokenSink &sink) {
    PTIStats stats;

    // ── context ──────────────────────────────────────────────────────────────
    llama_context_params cparams  = llama_context_default_params();
    cparams.n_ctx                 = (uint32_t)G.args.n_ctx;
    cparams.n_batch               = (uint32_t)G.args.n_batch;
    cparams.n_ubatch              = (uint32_t)G.args.n_ubatch;
    cparams.n_seq_max             = 4;
    cparams.flash_attn_type       = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.type_k                = GGML_TYPE_Q8_0;
    cparams.type_v                = GGML_TYPE_Q8_0;

    llama_context *ctx = llama_init_from_model(G.model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return stats; }

    llama_memory_t mem = llama_get_memory(ctx);

    // ── tokenise prompt ───────────────────────────────────────────────────────
    std::vector<llama_token> prompt_toks(MAX_PROMPT_TOKENS);
    int n_prompt = llama_tokenize(G.vocab, prompt.c_str(), (int32_t)prompt.size(),
                                  prompt_toks.data(), MAX_PROMPT_TOKENS, true, true);
    if (n_prompt <= 0) {
        fprintf(stderr, "Tokenise failed\n");
        llama_free(ctx);
        return stats;
    }
    fprintf(stderr, "  prompt: %d tokens\n", n_prompt);

    // ── prefill ───────────────────────────────────────────────────────────────
    llama_batch batch = llama_batch_init(n_prompt + 8, 0, 4);
    batch_clear(&batch);
    for (int i = 0; i < n_prompt; i++)
        batch_add(&batch, prompt_toks[i], (llama_pos)i, 0, i == n_prompt - 1);

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Prefill failed\n");
        llama_batch_free(batch); llama_free(ctx);
        return stats;
    }

    llama_pos   pos_a    = (llama_pos)n_prompt;
    llama_token tok_gen0 = sample_token(llama_get_logits_ith(ctx, batch.n_tokens - 1),
                                        G.n_vocab, temperature);

    llama_memory_seq_cp(mem, 0, 1, 0, -1);
    llama_memory_seq_cp(mem, 0, 2, 0, -1);
    llama_memory_seq_cp(mem, 0, 3, 0, -1);

    // Emit first token immediately
    if (!sink(token_to_str(tok_gen0))) {
        llama_batch_free(batch); llama_free(ctx);
        return stats;
    }
    stats.n_gen = 1;

    double t_gen_start = now_sec();

    // ── SSM cell prime (Qwen3.6 hybrid model alignment) ───────────────────────
    batch_clear(&batch);
    batch_add(&batch, tok_gen0, pos_a, 0, true);
    llama_decode(ctx, batch);
    llama_memory_seq_rm(mem, 0, 0, -1);
    llama_memory_seq_cp(mem, 1, 0, 0, -1);

    // ── B init ────────────────────────────────────────────────────────────────
    batch_clear(&batch);
    batch_add(&batch, tok_gen0, pos_a, 1, true);
    llama_decode(ctx, batch);
    llama_token tok_b = sample_token(llama_get_logits_ith(ctx, 0), G.n_vocab, temperature);
    llama_pos   pos_b = pos_a + 1;

    // ── C init ────────────────────────────────────────────────────────────────
    batch_clear(&batch);
    batch_add(&batch, tok_gen0, pos_a, 2, false);
    llama_decode(ctx, batch);
    batch_clear(&batch);
    batch_add(&batch, tok_b, pos_b, 2, true);
    llama_decode(ctx, batch);
    llama_token tok_c = sample_token(llama_get_logits_ith(ctx, 0), G.n_vocab, temperature);
    llama_pos   pos_c = pos_b + 1;

    // ── D init ────────────────────────────────────────────────────────────────
    batch_clear(&batch);
    batch_add(&batch, tok_gen0, pos_a, 3, false);
    llama_decode(ctx, batch);
    batch_clear(&batch);
    batch_add(&batch, tok_b, pos_b, 3, false);
    llama_decode(ctx, batch);
    batch_clear(&batch);
    batch_add(&batch, tok_c, pos_c, 3, true);
    llama_decode(ctx, batch);
    llama_token tok_d = sample_token(llama_get_logits_ith(ctx, 0), G.n_vocab, temperature);
    llama_pos   pos_d = pos_c + 1;

    llama_token tok_a = tok_gen0;

    // ── main PTI loop ─────────────────────────────────────────────────────────
    llama_batch step_batch = llama_batch_init(4, 0, 4);
    bool aborted = false;
    double t_loop_start = now_sec();

    for (int step = 0; step < max_new - 1 && !aborted; step++) {
        if (llama_vocab_is_eog(G.vocab, tok_a)) break;

        batch_clear(&step_batch);
        batch_add(&step_batch, tok_a, pos_a, 0, true);
        batch_add(&step_batch, tok_b, pos_b, 1, true);
        batch_add(&step_batch, tok_c, pos_c, 2, true);
        batch_add(&step_batch, tok_d, pos_d, 3, true);

        if (llama_decode(ctx, step_batch) != 0) break;

        llama_token actual   = sample_token(llama_get_logits_ith(ctx, 0), G.n_vocab, temperature);
        llama_token from_b   = sample_token(llama_get_logits_ith(ctx, 1), G.n_vocab, temperature);
        llama_token from_c   = sample_token(llama_get_logits_ith(ctx, 2), G.n_vocab, temperature);
        llama_token from_d   = sample_token(llama_get_logits_ith(ctx, 3), G.n_vocab, temperature);

        pos_a++; pos_b++; pos_c++; pos_d++;

        // Emit helper — returns false to signal abort
        auto emit = [&](llama_token tok) -> bool {
            if (llama_vocab_is_eog(G.vocab, tok)) { aborted = true; return false; }
            if (!sink(token_to_str(tok)))           { aborted = true; return false; }
            stats.n_gen++;
            return true;
        };

        if (tok_b == actual) {
            if (tok_c == from_b) {
                if (tok_d == from_c) {
                    // ── 4-accept ──────────────────────────────────────────
                    stats.n_4acc++;
                    emit(actual); emit(from_b); emit(from_c); emit(from_d);
                    tok_a = actual; tok_b = from_b; tok_c = from_c; tok_d = from_d;
                } else {
                    // ── 3-accept, reinit D ────────────────────────────────
                    stats.n_3acc++;
                    emit(actual); emit(from_b); emit(from_c);
                    tok_a = actual; tok_b = from_b; tok_c = from_c;
                    tok_d = reinit_seq(mem, ctx, &step_batch, temperature,
                                       2, 3, from_c, pos_c, &pos_d);
                }
            } else {
                // ── 2-accept, reinit C+D ──────────────────────────────────
                stats.n_2acc++;
                emit(actual); emit(from_b);
                tok_a = actual; tok_b = from_b;
                tok_c = reinit_seq(mem, ctx, &step_batch, temperature,
                                   1, 2, from_b, pos_b, &pos_c);
                tok_d = reinit_seq(mem, ctx, &step_batch, temperature,
                                   2, 3, tok_c,  pos_c, &pos_d);
            }
        } else {
            // ── reject, reinit B+C+D ──────────────────────────────────────
            stats.n_reject++;
            emit(actual);
            tok_b = reinit_seq(mem, ctx, &step_batch, temperature,
                               0, 1, actual, pos_a, &pos_b);
            tok_c = reinit_seq(mem, ctx, &step_batch, temperature,
                               1, 2, tok_b,  pos_b, &pos_c);
            tok_d = reinit_seq(mem, ctx, &step_batch, temperature,
                               2, 3, tok_c,  pos_c, &pos_d);
            tok_a = actual;
        }

        if (llama_vocab_is_eog(G.vocab, actual)) break;
    }

    double t_end       = now_sec();
    double loop_time   = t_end - t_loop_start;
    double total_time  = t_end - t_gen_start;
    stats.ss_tok_s     = loop_time  > 0 ? stats.n_gen / loop_time  : 0;
    stats.am_tok_s     = total_time > 0 ? stats.n_gen / total_time : 0;

    int total_steps = stats.n_4acc + stats.n_3acc + stats.n_2acc + stats.n_reject;
    fprintf(stderr,
        "  → %d tok  %.1f tok/s (%.1f amortized)\n"
        "     4-acc=%d  3-acc=%d  2-acc=%d  rej=%d  (of %d steps)\n",
        stats.n_gen, stats.ss_tok_s, stats.am_tok_s,
        stats.n_4acc, stats.n_3acc, stats.n_2acc, stats.n_reject, total_steps);

    llama_batch_free(batch);
    llama_batch_free(step_batch);
    llama_free(ctx);
    return stats;
}

// ── SSE formatting ────────────────────────────────────────────────────────────

static std::string make_id() {
    return "chatcmpl-pti-" + std::to_string((uint32_t)time(nullptr));
}

static std::string sse_chunk(const std::string &text, const std::string &id) {
    json delta  = {{"content", text}};
    json choice = {{"delta", delta}, {"index", 0}, {"finish_reason", nullptr}};
    json chunk  = {{"id", id}, {"object", "chat.completion.chunk"},
                   {"model", "pti-4seq"}, {"choices", json::array({choice})}};
    return "data: " + chunk.dump() + "\n\n";
}

static std::string sse_done(const PTIStats &s, const std::string &id) {
    json choice = {{"delta", json::object()}, {"index", 0}, {"finish_reason", "stop"}};
    json chunk  = {{"id", id}, {"object", "chat.completion.chunk"},
                   {"model", "pti-4seq"}, {"choices", json::array({choice})}};
    return "data: " + chunk.dump() + "\n\ndata: [DONE]\n\n";
}

// ── HTTP handlers ─────────────────────────────────────────────────────────────

static void handle_health(const httplib::Request &, httplib::Response &res) {
    res.set_content("{\"status\":\"ok\"}", "application/json");
}

static void handle_chat_completions(const httplib::Request &req, httplib::Response &res) {
    // Reject if already busy
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
    std::string id          = make_id();

    fprintf(stderr, "[%s] temp=%.2f max_tokens=%d\n", id.c_str(), temperature, max_tokens);

    res.set_chunked_content_provider("text/event-stream",
        [prompt, temperature, max_tokens, id](size_t /*offset*/,
                                               httplib::DataSink &sink) -> bool {
            PTIStats stats = run_pti4(prompt, max_tokens, temperature,
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
        "PTI Server — 4-sequence PTI decode for nanocoder / coding demos\n\n"
        "Usage: %s -m <model.gguf> [options]\n\n"
        "  -m <path>      Model path (required)\n"
        "  -p <port>      HTTP port (default: 8080)\n"
        "  -c <int>       Context size (default: 32768)\n"
        "  -ngl <int>     GPU layers (default: 99)\n"
        "  --temp <float> Default temperature (default: 0.7)\n"
        "  --pti <int>    PTI streams, currently fixed at 4\n"
        "  -n <int>       Default max tokens (default: 512)\n\n"
        "Endpoints:\n"
        "  GET  /v1                  connectivity check\n"
        "  POST /v1/chat/completions SSE streaming (OpenAI chat format)\n\n"
        "KV cache: Q8_0 (quantized). For baseline, run llama-server on another port.\n",
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
        else if (!strcmp(argv[i], "--pti")  && i+1 < argc) args.pti_streams  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n")     && i+1 < argc) args.max_tokens   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
    }

    if (args.model_path[0] == '\0') {
        fprintf(stderr, "Error: -m <model_path> required\n\n");
        usage(argv[0]); return 1;
    }

    llama_backend_init();

    fprintf(stderr, "Loading: %s\n", args.model_path);
    llama_model_params mparams   = llama_model_default_params();
    mparams.n_gpu_layers         = args.n_gpu_layers;
    G.model = llama_model_load_from_file(args.model_path, mparams);
    if (!G.model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    G.vocab   = llama_model_get_vocab(G.model);
    G.n_vocab = llama_vocab_n_tokens(G.vocab);
    G.n_embd  = llama_model_n_embd(G.model);
    G.tmpl    = llama_model_chat_template(G.model, /*name=*/nullptr);
    G.rng.seed(std::random_device{}());

    fprintf(stderr, "Model ready. vocab=%d n_embd=%d template=%s\n",
            G.n_vocab, G.n_embd, G.tmpl ? "(from model)" : "(none — using raw prompt)");

    httplib::Server svr;
    svr.Get("/v1",                      handle_health);
    svr.Get("/health",                  handle_health);
    svr.Post("/v1/chat/completions",    handle_chat_completions);

    fprintf(stderr,
        "\nPTI server on http://0.0.0.0:%d\n"
        "  PTI streams : %d\n"
        "  Temperature : %.2f (overridable per request)\n"
        "  Context     : %d tokens (effective ~%d per sequence)\n"
        "  KV cache    : Q8_0\n\n",
        args.port, args.pti_streams, args.temperature,
        args.n_ctx, args.n_ctx / 4);

    svr.listen("0.0.0.0", args.port);

    llama_model_free(G.model);
    llama_backend_free();
    return 0;
}
