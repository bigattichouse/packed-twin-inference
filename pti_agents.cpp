/*
 * pti_agents.cpp — PA.0: packed-agents plumbing demo
 *
 * N independent prompt streams (default 4, -s up to 16) decode in ONE
 * llama_context with one batched llama_decode per step (one token per live
 * stream). Measures aggregate throughput vs the same prompts run sequentially,
 * and asserts each packed lane is byte-identical to its solo run.
 *
 * Expected from the M5.1 measurement (4-seq batch = 1.86× one stream):
 *   aggregate ≈ 4 / 1.86 ≈ 2.15× the sequential token rate.
 *
 * This is the substrate for PACKED_AGENTS.md: PA.1 turns the four streams
 * into coordinator + 3 workers with plan / fan-out / parallel / gather.
 *
 * Build:  make agents
 * Run:    bin/pti_agents -m ../gguf/Qwen3.6-27B-UD-Q6_K_XL.gguf [-s 8]
 *         (runs sequential then packed on N built-in prompts; prints ratio + gate)
 */

#include "../llama.cpp/include/llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#define MAX_STREAMS   16
#define PREFILL_CHUNK 1024
#define MAX_TOKENS    32768

static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// quiet llama/ggml logging: only WARN+ passes (--verbose restores)
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

// deterministic tie-break argmax (same rule as the other pti binaries)
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

struct Stream {
    std::vector<llama_token> prompt_toks;
    std::string  text;        // generated output
    std::vector<llama_token> gen_toks;  // generated tokens (solo-vs-packed identity gate)
    llama_token  tok_last = 0;
    llama_pos    pos      = 0;
    int          n_gen    = 0;
    bool         live     = false;
};

struct Globals {
    llama_model       *model = nullptr;
    const llama_vocab *vocab = nullptr;
    llama_context     *ctx   = nullptr;
    llama_memory_t     mem   = nullptr;
    int32_t            n_vocab = 0;
} G;

static std::string tok_str(llama_token t) {
    char buf[256];
    int len = llama_token_to_piece(G.vocab, t, buf, (int32_t)sizeof(buf), 0, true);
    return len > 0 ? std::string(buf, len) : std::string{};
}

// prefill one stream's prompt into its sequence (chunked); returns last logits idx
static bool prefill_stream(Stream &st, llama_seq_id seq, llama_batch &batch, int *last_idx) {
    int n = (int)st.prompt_toks.size();
    for (int i0 = 0; i0 < n; i0 += PREFILL_CHUNK) {
        int nb = n - i0 < PREFILL_CHUNK ? n - i0 : PREFILL_CHUNK;
        batch_clear(&batch);
        for (int j = 0; j < nb; j++)
            batch_add(&batch, st.prompt_toks[i0 + j], (llama_pos)(i0 + j), seq, i0 + j == n - 1);
        if (llama_decode(G.ctx, batch) != 0) return false;
        *last_idx = nb - 1;
    }
    return true;
}

// run all streams: packed=false → one at a time in seq 0 (solo reference);
// packed=true → one batched decode per step. Returns total generated tokens;
// *wall_out = decode-loop seconds (prefill excluded).
static int run_streams(std::vector<Stream> &streams, int max_new, bool packed,
                       double *wall_out, double *prefill_out) {
    llama_batch batch = llama_batch_init(PREFILL_CHUNK + 8, 0, (int)streams.size());
    int total = 0;
    double wall = 0.0, prefill = 0.0;

    if (packed) {
        // all streams prefilled into their own seqs, then one batch per step
        double t0 = now_sec();
        for (int s = 0; s < (int)streams.size(); s++) {
            llama_memory_seq_rm(G.mem, s, 0, -1);
            int last_idx = 0;
            if (!prefill_stream(streams[s], s, batch, &last_idx)) {
                fprintf(stderr, "prefill failed (stream %d)\n", s);
                llama_batch_free(batch);
                return 0;
            }
            streams[s].tok_last = (llama_token)argmax_f(llama_get_logits_ith(G.ctx, last_idx), G.n_vocab);
            streams[s].pos      = (llama_pos)streams[s].prompt_toks.size();
            streams[s].text     = tok_str(streams[s].tok_last);
            streams[s].gen_toks.push_back(streams[s].tok_last);
            streams[s].n_gen    = 1;
            streams[s].live     = !llama_vocab_is_eog(G.vocab, streams[s].tok_last);
            total++;
        }
        prefill = now_sec() - t0;

        double t1 = now_sec();
        for (;;) {
            // one token per live stream, one decode for all of them
            batch_clear(&batch);
            std::vector<int> order(streams.size()); int n_live = 0;
            for (int s = 0; s < (int)streams.size(); s++) {
                Stream &st = streams[s];
                if (!st.live || st.n_gen >= max_new) { st.live = false; continue; }
                batch_add(&batch, st.tok_last, st.pos, s, true);
                order[n_live++] = s;
            }
            if (n_live == 0) break;
            if (llama_decode(G.ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); break; }
            for (int i = 0; i < n_live; i++) {
                Stream &st = streams[order[i]];
                llama_token nxt = (llama_token)argmax_f(llama_get_logits_ith(G.ctx, i), G.n_vocab);
                st.pos++;
                st.tok_last = nxt;
                st.n_gen++;
                total++;
                if (llama_vocab_is_eog(G.vocab, nxt)) st.live = false;
                else { st.text += tok_str(nxt); st.gen_toks.push_back(nxt); }
            }
        }
        wall = now_sec() - t1;
    } else {
        // same prompts, one stream at a time in seq 0
        for (auto &st : streams) {
            llama_memory_seq_rm(G.mem, 0, 0, -1);
            double t0 = now_sec();
            int last_idx = 0;
            if (!prefill_stream(st, 0, batch, &last_idx)) {
                fprintf(stderr, "prefill failed\n");
                break;
            }
            prefill += now_sec() - t0;
            st.tok_last = (llama_token)argmax_f(llama_get_logits_ith(G.ctx, last_idx), G.n_vocab);
            st.pos      = (llama_pos)st.prompt_toks.size();
            st.text     = tok_str(st.tok_last);
            st.gen_toks.push_back(st.tok_last);
            st.n_gen    = 1;
            total++;

            double t1 = now_sec();
            while (st.n_gen < max_new && !llama_vocab_is_eog(G.vocab, st.tok_last)) {
                batch_clear(&batch);
                batch_add(&batch, st.tok_last, st.pos, 0, true);
                if (llama_decode(G.ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); break; }
                st.tok_last = (llama_token)argmax_f(llama_get_logits_ith(G.ctx, 0), G.n_vocab);
                st.pos++;
                st.n_gen++;
                total++;
                if (!llama_vocab_is_eog(G.vocab, st.tok_last)) { st.text += tok_str(st.tok_last); st.gen_toks.push_back(st.tok_last); }
            }
            wall += now_sec() - t1;
        }
    }

    llama_batch_free(batch);
    *wall_out    = wall;
    *prefill_out = prefill;
    return total;
}

int main(int argc, char **argv) {
    char  model_path[512] = {};
    int   max_new   = 96;
    int   n_ctx     = 16384;
    int   n_streams = 4;
    bool  show_text = false;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-m")   && i+1 < argc) strncpy(model_path, argv[++i], sizeof(model_path)-1);
        else if (!strcmp(argv[i], "-n")   && i+1 < argc) max_new   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c")   && i+1 < argc) n_ctx     = atoi(argv[++i]);
        else if ((!strcmp(argv[i], "-s") || !strcmp(argv[i], "--streams")) && i+1 < argc) n_streams = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--text"))     show_text = true;
        else if (!strcmp(argv[i], "--verbose"))  g_verbose_logs = true;
        else { fprintf(stderr, "Usage: %s -m <model> [-s streams(1-%d)] [-n max] [-c ctx] [--text] [--verbose]\n", argv[0], MAX_STREAMS); return 1; }
    }
    if (!model_path[0]) { fprintf(stderr, "Error: -m required\n"); return 1; }
    if (n_streams < 1) n_streams = 1;
    if (n_streams > MAX_STREAMS) { fprintf(stderr, "note: clamping -s to MAX_STREAMS=%d\n", MAX_STREAMS); n_streams = MAX_STREAMS; }
    {   // non-unified KV gives n_ctx/n_seq_max per lane; ensure each lane fits its prompt + gen
        int need = (max_new + 320) * n_streams;
        if (n_ctx < need) n_ctx = need;
    }

    llama_log_set(pti_log_cb, nullptr);
    llama_backend_init();

    fprintf(stderr, "Loading %s ...\n", model_path);
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    G.model = llama_model_load_from_file(model_path, mp);
    if (!G.model) { fprintf(stderr, "load failed\n"); return 1; }
    G.vocab   = llama_model_get_vocab(G.model);
    G.n_vocab = llama_vocab_n_tokens(G.vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)n_ctx;
    cp.n_batch = PREFILL_CHUNK;
    cp.n_seq_max = (uint32_t)n_streams;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.kv_unified      = false;   // M7.2 exactness: unified KV flips fp near-ties by batch shape
    G.ctx = llama_init_from_model(G.model, cp);
    if (!G.ctx) { fprintf(stderr, "context failed\n"); return 1; }
    G.mem = llama_get_memory(G.ctx);

    // independent worker-shaped prompts (PA.1 will generate these from a plan).
    // Pool of MAX_STREAMS; the first n_streams are used.
    const char *prompts[MAX_STREAMS] = {
        "Write a javascript function gravityStep(bird, dt) that applies gravity and velocity to a flappy-bird player object. Output only code.",
        "Write a javascript function spawnPipe(world) that appends a new pipe pair with a random gap to world.pipes. Output only code.",
        "Write a javascript function checkCollision(bird, pipes) that returns true when the bird hits a pipe or the ground. Output only code.",
        "Write a javascript function updateScore(world) that increments world.score when the bird passes a pipe. Output only code.",
        "Write a javascript function debounce(fn, ms) that returns a debounced version of fn. Output only code.",
        "Write a javascript function deepClone(obj) that returns a structural deep copy of a JSON-like object. Output only code.",
        "Write a javascript function formatBytes(n) that formats a byte count as a human-readable string. Output only code.",
        "Write a javascript function parseQuery(url) that returns an object of the URL query parameters. Output only code.",
        "Write a javascript function shuffle(arr) that returns a new array with the elements randomly permuted. Output only code.",
        "Write a javascript function clamp(x, lo, hi) that constrains x to the inclusive range [lo, hi]. Output only code.",
        "Write a javascript function groupBy(arr, keyFn) that groups array items into a Map by a key function. Output only code.",
        "Write a javascript function retryAsync(fn, times) that retries an async fn up to N times. Output only code.",
        "Write a javascript function rgbToHex(r, g, b) that converts an RGB triple to a hex color string. Output only code.",
        "Write a javascript function memoize(fn) that caches results of a pure single-argument function. Output only code.",
        "Write a javascript function flatten(arr) that fully flattens an arbitrarily nested array. Output only code.",
        "Write a javascript function slugify(text) that converts a title string into a URL slug. Output only code.",
    };

    auto make_streams = [&]() {
        std::vector<Stream> v(n_streams);
        for (int s = 0; s < n_streams; s++) {
            v[s].prompt_toks.resize(MAX_TOKENS);
            int n = llama_tokenize(G.vocab, prompts[s], (int32_t)strlen(prompts[s]),
                                   v[s].prompt_toks.data(), MAX_TOKENS, true, true);
            v[s].prompt_toks.resize(n > 0 ? n : 0);
        }
        return v;
    };

    fprintf(stderr, "\n══ PA.0 — packed agents plumbing demo (%d streams, -n %d) ══\n\n", n_streams, max_new);

    // ── sequential baseline ──────────────────────────────────────────────────
    auto seq_streams = make_streams();
    double seq_wall = 0, seq_pf = 0;
    fprintf(stderr, "[1/2] sequential: %d prompts one at a time...\n", n_streams);
    int seq_total = run_streams(seq_streams, max_new, /*packed=*/false, &seq_wall, &seq_pf);
    fprintf(stderr, "      %d tok in %.1fs decode (+%.1fs prefill) = %.1f tok/s\n",
            seq_total, seq_wall, seq_pf, seq_total / seq_wall);

    // clear everything between modes
    for (int s = 0; s < n_streams; s++) llama_memory_seq_rm(G.mem, s, 0, -1);

    // ── packed ───────────────────────────────────────────────────────────────
    auto par_streams = make_streams();
    double par_wall = 0, par_pf = 0;
    fprintf(stderr, "[2/2] packed: %d streams, one batched decode per step...\n", n_streams);
    int par_total = run_streams(par_streams, max_new, /*packed=*/true, &par_wall, &par_pf);
    fprintf(stderr, "      %d tok in %.1fs decode (+%.1fs prefill) = %.1f tok/s\n",
            par_total, par_wall, par_pf, par_total / par_wall);

    fprintf(stderr, "\n════════════════════════════════════════════════\n");
    fprintf(stderr, "  PA.0 result\n");
    fprintf(stderr, "  sequential : %5.1f tok/s  (%d tok, %.1fs)\n", seq_total / seq_wall, seq_total, seq_wall);
    fprintf(stderr, "  packed     : %5.1f tok/s  (%d tok, %.1fs)\n", par_total / par_wall, par_total, par_wall);
    fprintf(stderr, "  aggregate  : %.2fx   (%d streams packed vs sequential)\n",
            (par_total / par_wall) / (seq_total / seq_wall), n_streams);
    fprintf(stderr, "  wall-clock : %.2fx   (decode loops, equal token caps)\n",
            seq_wall / par_wall);
    fprintf(stderr, "════════════════════════════════════════════════\n");

    // ── byte-identity gate: each packed lane's tokens must equal its solo (sequential) run ──
    // The cooperative design requires a stream's output to be invariant to co-residence in the
    // batch. Exact config: f16 KV (default) + kv_unified=false + ε=0.05 tie-break argmax.
    fprintf(stderr, "\n── byte-identity gate (packed lane == solo) ──\n");
    int gate_fail = 0;
    for (int s = 0; s < n_streams; s++) {
        const std::vector<llama_token> &solo = seq_streams[s].gen_toks;
        const std::vector<llama_token> &pack = par_streams[s].gen_toks;
        size_t n = solo.size() < pack.size() ? solo.size() : pack.size();
        long diverge = -1;
        for (size_t i = 0; i < n; i++) if (solo[i] != pack[i]) { diverge = (long)i; break; }
        if (diverge < 0 && solo.size() == pack.size()) {
            fprintf(stderr, "  stream %d: IDENTICAL (%zu tok)\n", s, solo.size());
        } else {
            gate_fail++;
            long at = diverge >= 0 ? diverge : (long)n;
            fprintf(stderr, "  stream %d: DIVERGED at tok %ld (solo %zu, packed %zu)\n",
                    s, at, solo.size(), pack.size());
            if (diverge >= 0)
                fprintf(stderr, "            solo[%ld]=%d '%s'  vs  packed[%ld]=%d '%s'\n",
                        at, solo[at], tok_str(solo[at]).c_str(),
                        at, pack[at], tok_str(pack[at]).c_str());
        }
    }
    if (gate_fail == 0)
        fprintf(stderr, "  GATE PASS — packed output byte-identical to solo on all %d lanes\n", n_streams);
    else
        fprintf(stderr, "  GATE FAIL — %d/%d lanes diverged (see above)\n", gate_fail, n_streams);
    fprintf(stderr, "════════════════════════════════════════════════\n");

    if (show_text) {
        for (int s = 0; s < n_streams; s++) {
            printf("\n───── stream %d ─────\n%s\n", s, par_streams[s].text.c_str());
        }
    }

    llama_free(G.ctx);
    llama_model_free(G.model);
    llama_backend_free();
    return gate_fail ? 2 : 0;
}
