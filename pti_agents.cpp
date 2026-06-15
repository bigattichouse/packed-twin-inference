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
#include <filesystem>
#include <regex>
#include <string>
#include <sys/wait.h>
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
static bool g_no_think     = false;   // --no-think: force a closed empty <think></think> after the
                                      // generation prompt so boss+workers skip reasoning (faster, esp. the plan)
static char g_out_path[512] = {};     // --out: write final artifact to this file
static bool g_no_gather    = false;   // --no-gather: skip gather phase, print pieces separately
static bool g_tools        = false;   // --tools: let workers emit <function=...> tool calls (write_file/run)
static bool g_allow_run    = false;   // --allow-run: permit the `run` verb (sandboxed shell exec); implies --tools
static char g_work_dir[512] = "pti_work";  // --work-dir: sandbox dir for write_file / run
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
    const char        *tmpl  = nullptr;   // chat template (PA.1 boss prompting)
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

// ───────────────────────── PA.2: work-pool ───────────────────────────────────
// M items over n_lanes; a lane that finishes (EOG or cap) is refilled from the
// queue (seq_rm + prefill the next item), keeping the batch full while backlog
// lasts. The pooled fix for the PA.1b straggler.
static void run_pool(const std::vector<std::string> &items, int n_lanes, int max_new,
                     std::vector<std::string> *out_opt = nullptr, const std::string &prefix = "") {
    int M = (int)items.size();
    if (n_lanes > M) n_lanes = M;
    llama_batch batch = llama_batch_init(PREFILL_CHUNK + 8, 0, n_lanes + 1);
    std::vector<Stream> lanes(n_lanes);
    std::vector<int>    lane_item(n_lanes, -1);
    std::vector<std::string> out(M);
    int next = 0, total = 0, refills = 0;

    // ── PA.2.1: cache the common prefix (preamble+shared) once in a base seq; each lane is a
    //    seq_cp clone of it + a small delta-prefill of just the item (the M7.5b base cache).
    const llama_seq_id BASE = n_lanes;                 // base seq (needs n_seq_max >= n_lanes+1)
    std::vector<llama_token> prefix_toks; int prefix_len = 0;
    bool use_cache = !prefix.empty();
    if (use_cache) {
        prefix_toks.assign(prefix.size() + 16, 0);
        int pn = llama_tokenize(G.vocab, prefix.c_str(), (int32_t)prefix.size(),
                                prefix_toks.data(), (int32_t)prefix_toks.size(), true, true);
        if (pn <= 0) use_cache = false;
        else {
            prefix_toks.resize(pn); prefix_len = pn;
            llama_memory_seq_rm(G.mem, BASE, 0, -1);
            Stream base_st; base_st.prompt_toks = prefix_toks; int li = 0;
            if (!prefill_stream(base_st, BASE, batch, &li)) use_cache = false;   // cache the starter once
        }
    }

    auto start_lane = [&](int L, int item) -> bool {
        std::vector<llama_token> full(items[item].size() + 16, 0);
        int fn = llama_tokenize(G.vocab, items[item].c_str(), (int32_t)items[item].size(),
                                full.data(), (int32_t)full.size(), true, true);
        if (fn <= 0) return false;
        full.resize(fn);
        bool cached = use_cache && fn > prefix_len && (fn - prefix_len) <= PREFILL_CHUNK;
        for (int k = 0; cached && k < prefix_len; k++) if (full[k] != prefix_toks[k]) cached = false;
        llama_memory_seq_rm(G.mem, L, 0, -1);
        int last_idx = 0;
        if (cached) {
            llama_memory_seq_cp(G.mem, BASE, L, 0, -1);        // clone the starter (= roll back to base)
            batch_clear(&batch);
            int dn = fn - prefix_len;                          // delta-prefill only the item part
            for (int j = 0; j < dn; j++)
                batch_add(&batch, full[prefix_len + j], (llama_pos)(prefix_len + j), L, j == dn - 1);
            if (llama_decode(G.ctx, batch) != 0) return false;
            last_idx = dn - 1;
        } else {                                               // fallback: full prefill (no cache / unstable prefix)
            lanes[L].prompt_toks = full;
            if (!prefill_stream(lanes[L], L, batch, &last_idx)) return false;
        }
        lanes[L].tok_last = (llama_token)argmax_f(llama_get_logits_ith(G.ctx, last_idx), G.n_vocab);
        lanes[L].pos      = (llama_pos)fn;
        lanes[L].text     = tok_str(lanes[L].tok_last);
        lanes[L].n_gen    = 1;
        lanes[L].live     = !llama_vocab_is_eog(G.vocab, lanes[L].tok_last);
        lane_item[L]      = item;
        total++;
        fprintf(stderr, "  → item %d → lane %d (%s)\n", item, L, cached ? "cloned+delta" : "full prefill");
        return true;
    };

    double t0 = now_sec();
    for (int L = 0; L < n_lanes && next < M; L++) start_lane(L, next++);
    double prefill0 = now_sec() - t0;

    double t1 = now_sec();
    for (;;) {
        batch_clear(&batch);
        std::vector<int> ord;
        for (int L = 0; L < n_lanes; L++)
            if (lanes[L].live && lanes[L].n_gen < max_new) {
                batch_add(&batch, lanes[L].tok_last, lanes[L].pos, L, true);
                ord.push_back(L);
            }
        if (ord.empty()) break;
        if (llama_decode(G.ctx, batch) != 0) { fprintf(stderr, "pool decode failed\n"); break; }
        std::vector<int> finished;
        for (size_t i = 0; i < ord.size(); i++) {
            int L = ord[i];
            llama_token nxt = (llama_token)argmax_f(llama_get_logits_ith(G.ctx, (int)i), G.n_vocab);
            lanes[L].pos++; lanes[L].tok_last = nxt; lanes[L].n_gen++; total++;
            if (llama_vocab_is_eog(G.vocab, nxt) || lanes[L].n_gen >= max_new) {
                out[lane_item[L]] = lanes[L].text;
                fprintf(stderr, "  ✓ item %d done (%d tok)\n", lane_item[L], lanes[L].n_gen);
                lanes[L].live = false;
                finished.push_back(L);
            } else {
                lanes[L].text += tok_str(nxt);
            }
        }
        for (int L : finished) {                    // refill from the queue, else idle
            if (next < M) { start_lane(L, next++); refills++; }
            else lane_item[L] = -1;
        }
    }
    double wall = now_sec() - t1;
    llama_batch_free(batch);

    double agg = wall > 0 ? total / wall : 0.0;
    fprintf(stderr, "\n════════════════════════════════════════════════\n");
    fprintf(stderr, "  PA.2 work-pool result\n");
    fprintf(stderr, "  %d items, %d lanes, cap %d/item, %d refills\n", M, n_lanes, max_new, refills);
    if (use_cache) fprintf(stderr, "  prefix cache: %d tok cloned per lane (delta-prefill only the item)\n", prefix_len);
    fprintf(stderr, "  %d tok in %.1fs decode (+%.1fs initial prefill)\n", total, wall, prefill0);
    fprintf(stderr, "  aggregate  : %.1f tok/s = %.2fx vs 19.3 baseline\n", agg, agg / 19.3);
    fprintf(stderr, "  sequential ≈ %.0fs for the same %d tok; pool did it in %.1fs\n",
            total / 19.3, total, wall);
    fprintf(stderr, "════════════════════════════════════════════════\n");
    if (out_opt) *out_opt = out;
}

// ───────────────────────── PA.1: work-order envelope ─────────────────────────
// The boss emits ONE planning block; the harness parses only the thin marker
// envelope (passing BluePrint bodies through verbatim). Format — design §4:
//   <<<PLAN strategy=function lang=js>>>
//   shared:
//   <blueprint> ...frozen interface + deps... </blueprint>
//   <<<PIECE id=w1 exports=foo,bar>>>
//   instruction: implement this function ...
//   <blueprint> ...spec... </blueprint>
//   <<<SELF id=boss exports=integrate,smokeTest>>>  ...
//   <<<END>>>
struct Piece {
    std::string id;
    std::vector<std::string> exports;
    std::string language;
    std::string instruction;
    std::string blueprint;
    bool        is_boss = false;
};
struct WorkOrder {
    std::string strategy;
    std::string shared;             // frozen interface, prepended to every lane
    std::vector<Piece> pieces;      // workers, then the boss SELF piece
    bool        ok = false;
    std::string error;
};

static std::string trim(const std::string &s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static std::vector<std::string> split_csv(const std::string &s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t c = s.find(',', i);
        if (c == std::string::npos) c = s.size();
        std::string t = trim(s.substr(i, c - i));
        if (!t.empty()) out.push_back(t);
        i = c + 1;
    }
    return out;
}

// "<<<TAG k=v k=v>>>" → tag + attrs; false if the line is not a marker
static bool parse_marker(const std::string &line, std::string &tag,
                         std::vector<std::pair<std::string,std::string>> &attrs) {
    std::string t = trim(line);
    if (t.rfind("<<<", 0) != 0) return false;
    size_t end = t.find(">>>");
    if (end == std::string::npos) return false;
    std::string inner = trim(t.substr(3, end - 3));
    size_t sp = inner.find_first_of(" \t");
    tag = inner.substr(0, sp);
    attrs.clear();
    if (sp == std::string::npos) return true;
    std::string rest = inner.substr(sp + 1);
    size_t i = 0;
    while (i < rest.size()) {
        size_t nsp = rest.find_first_of(" \t", i);
        if (nsp == std::string::npos) nsp = rest.size();
        std::string kv = rest.substr(i, nsp - i);
        size_t eq = kv.find('=');
        if (eq != std::string::npos)
            attrs.push_back({ trim(kv.substr(0, eq)), trim(kv.substr(eq + 1)) });
        else if (!attrs.empty())
            attrs.back().second += kv;   // value continued past a space, e.g. "exports=a, b"
        i = nsp + 1;
    }
    return true;
}

static std::string attr_get(const std::vector<std::pair<std::string,std::string>> &a,
                            const std::string &k, const std::string &dflt = "") {
    for (auto &p : a) if (p.first == k) return p.second;
    return dflt;
}

// <blueprint>...</blueprint> if present, else the trimmed body
static std::string extract_blueprint(const std::string &body) {
    size_t a = body.find("<blueprint>");
    size_t b = body.find("</blueprint>");
    if (a != std::string::npos && b != std::string::npos && b > a)
        return trim(body.substr(a + 11, b - (a + 11)));
    return trim(body);
}

// text after the first "instruction:" line
static std::string extract_instruction(const std::string &body) {
    size_t p = body.find("instruction:");
    if (p == std::string::npos) return "";
    size_t e = body.find('\n', p);
    size_t s = p + 12;
    return trim(body.substr(s, (e == std::string::npos ? body.size() : e) - s));
}

static WorkOrder parse_work_order(const std::string &text) {
    WorkOrder wo;
    std::vector<std::string> lines;
    { size_t i = 0; while (true) {
        size_t nl = text.find('\n', i);
        if (nl == std::string::npos) { lines.push_back(text.substr(i)); break; }
        lines.push_back(text.substr(i, nl - i)); i = nl + 1; } }

    std::string default_lang;
    enum { NONE, SHARED, INPIECE } sect = NONE;
    std::string buf;
    Piece cur; bool have_cur = false;
    auto flush_piece = [&]() {
        if (!have_cur) return;
        cur.instruction = extract_instruction(buf);
        cur.blueprint   = extract_blueprint(buf);
        wo.pieces.push_back(cur);
        have_cur = false; cur = Piece();
    };
    for (auto &ln : lines) {
        std::string tag; std::vector<std::pair<std::string,std::string>> at;
        if (parse_marker(ln, tag, at)) {
            if (tag == "PLAN") {
                wo.strategy  = attr_get(at, "strategy");
                default_lang = attr_get(at, "lang");
                sect = SHARED; buf.clear();
            } else if (tag == "PIECE" || tag == "SELF") {
                if (sect == SHARED) wo.shared = extract_blueprint(buf);
                flush_piece();
                cur.id       = attr_get(at, "id");
                cur.exports  = split_csv(attr_get(at, "exports"));
                cur.language = attr_get(at, "lang", default_lang);
                cur.is_boss  = (tag == "SELF");
                have_cur = true; sect = INPIECE; buf.clear();
            } else if (tag == "/PIECE" || tag == "/SELF") {   // explicit piece close (clean delineation)
                flush_piece();
                sect = NONE; buf.clear();
            } else if (tag == "END") {
                if (sect == SHARED) wo.shared = extract_blueprint(buf);
                flush_piece();
                break;
            }
        } else {
            buf += ln; buf += '\n';
        }
    }
    if (have_cur) flush_piece();
    if (sect == SHARED && wo.shared.empty()) wo.shared = extract_blueprint(buf);
    // Robustness: if the PLAN marker was malformed (model drops the leading <<< on PLAN),
    // the shared block never got captured — recover it from the blueprint before piece 1.
    if (wo.shared.empty()) {
        size_t fp = text.find("<<<PIECE"), fs = text.find("<<<SELF");
        size_t f = std::min(fp == std::string::npos ? text.size() : fp,
                            fs == std::string::npos ? text.size() : fs);
        wo.shared = extract_blueprint(text.substr(0, f));
    }

    if (wo.pieces.empty()) { wo.error = "no pieces parsed"; return wo; }
    for (auto &p : wo.pieces)
        if (p.id.empty()) { wo.error = "a piece is missing id="; return wo; }
    std::vector<std::string> seen;
    for (auto &p : wo.pieces) for (auto &e : p.exports) {
        for (auto &s : seen) if (s == e) { wo.error = "export collision: " + e; return wo; }
        seen.push_back(e);
    }
    wo.ok = true;
    return wo;
}

static const char *SAMPLE_ENVELOPE =
    "<<<PLAN strategy=function lang=js>>>\n"
    "shared:\n"
    "<blueprint>\n"
    "  World { pipes: Pipe[], bird: {x,y,vy}, score: int, gravity: float }\n"
    "  Pipe  { x: float, gapY: float, gapH: float, passed: bool }\n"
    "  deps: []\n"
    "</blueprint>\n"
    "<<<PIECE id=w1 exports=gravityStep>>>\n"
    "instruction: implement this function given the public params of the current class, in js\n"
    "<blueprint> gravityStep(bird, dt) -> void { apply gravity then velocity to bird } </blueprint>\n"
    "<<</PIECE>>>\n"
    "<<<PIECE id=w2 exports=spawnPipe>>>\n"
    "instruction: implement this function given the public params of the current class, in js\n"
    "<blueprint> spawnPipe(world) -> void { append a Pipe with a random gap to world.pipes } </blueprint>\n"
    "<<</PIECE>>>\n"
    "<<<PIECE id=w3 exports=checkCollision>>>\n"
    "instruction: implement this function given the public params of the current class, in js\n"
    "<blueprint> checkCollision(bird, pipes) -> bool { true if bird hits a pipe or ground } </blueprint>\n"
    "<<</PIECE>>>\n"
    "<<<SELF id=boss exports=stepWorld, smokeTest>>>\n"
    "instruction: implement this object in js - wire the three functions and a smoke test\n"
    "<blueprint> Integration { stepWorld(world, dt) -> void, smokeTest() -> bool } </blueprint>\n"
    "<<</SELF>>>\n"
    "<<<END>>>\n";

static void print_work_order(const WorkOrder &wo) {
    fprintf(stderr, "strategy : %s\n", wo.strategy.c_str());
    fprintf(stderr, "shared   : %zu chars — \"%.50s...\"\n", wo.shared.size(), trim(wo.shared).c_str());
    for (auto &p : wo.pieces) {
        fprintf(stderr, "  %-5s boss=%d lang=%-3s exports=[", p.id.c_str(), p.is_boss, p.language.c_str());
        for (size_t i = 0; i < p.exports.size(); i++) fprintf(stderr, "%s%s", i ? "," : "", p.exports[i].c_str());
        fprintf(stderr, "]\n");
        fprintf(stderr, "        instr: %.66s\n", p.instruction.c_str());
        fprintf(stderr, "        bp   : %.66s\n", p.blueprint.c_str());
    }
    if (wo.ok) fprintf(stderr, "PARSE OK — %zu pieces, no export collisions\n", wo.pieces.size());
    else       fprintf(stderr, "PARSE FAIL — %s\n", wo.error.c_str());
}

static int parse_self_test() {
    fprintf(stderr, "── PA.1a work-order envelope parse self-test ──\n");
    WorkOrder wo = parse_work_order(SAMPLE_ENVELOPE);
    print_work_order(wo);
    return wo.ok ? 0 : 2;
}

// ── boss prompting (PA.1 PLAN phase) ─────────────────────────────────────────
struct Msg { std::string role, content; };

static std::string apply_chat_template(const std::vector<Msg> &msgs, bool add_ass = true) {
    std::vector<llama_chat_message> chat;
    for (auto &m : msgs) chat.push_back({ m.role.c_str(), m.content.c_str() });
    int32_t need = llama_chat_apply_template(G.tmpl, chat.data(), chat.size(), add_ass, nullptr, 0);
    if (need < 0) { std::string raw; for (auto &m : msgs) raw += m.content + "\n"; return raw; }
    std::vector<char> buf(need + 1);
    llama_chat_apply_template(G.tmpl, chat.data(), chat.size(), add_ass, buf.data(), (int32_t)buf.size());
    return std::string(buf.data(), need);
}

// greedy single-stream decode in seq 0 from a formatted prompt; returns the text
static std::string boss_generate(const std::string &prompt, int max_tok) {
    std::vector<llama_token> toks(prompt.size() + 16);
    int n = llama_tokenize(G.vocab, prompt.c_str(), (int32_t)prompt.size(),
                           toks.data(), (int32_t)toks.size(), true, true);
    if (n < 0) { toks.resize(-n);
        n = llama_tokenize(G.vocab, prompt.c_str(), (int32_t)prompt.size(),
                           toks.data(), (int32_t)toks.size(), true, true); }
    toks.resize(n > 0 ? n : 0);
    llama_memory_seq_rm(G.mem, 0, 0, -1);
    llama_batch batch = llama_batch_init(PREFILL_CHUNK + 8, 0, 1);
    int last_idx = 0;
    for (int i0 = 0; i0 < n; i0 += PREFILL_CHUNK) {
        int nb = n - i0 < PREFILL_CHUNK ? n - i0 : PREFILL_CHUNK;
        batch_clear(&batch);
        for (int j = 0; j < nb; j++)
            batch_add(&batch, toks[i0 + j], (llama_pos)(i0 + j), 0, i0 + j == n - 1);
        if (llama_decode(G.ctx, batch) != 0) { llama_batch_free(batch); return "[prefill failed]"; }
        last_idx = nb - 1;
    }
    llama_token tok = (llama_token)argmax_f(llama_get_logits_ith(G.ctx, last_idx), G.n_vocab);
    llama_pos pos = n;
    std::string out;
    for (int gen = 0; gen < max_tok && !llama_vocab_is_eog(G.vocab, tok); gen++) {
        { std::string piece = tok_str(tok); out += piece; fputs(piece.c_str(), stderr); }   // stream live
        batch_clear(&batch);
        batch_add(&batch, tok, pos, 0, true);
        if (llama_decode(G.ctx, batch) != 0) break;
        tok = (llama_token)argmax_f(llama_get_logits_ith(G.ctx, 0), G.n_vocab);
        pos++;
    }
    llama_batch_free(batch);
    return out;
}

static const char *BOSS_PROMPT =
    "You are the COORDINATOR of a packed-agent coding team. Workers run in parallel and in\n"
    "ISOLATION — a worker sees only the shared interface, its own instruction, and its spec,\n"
    "never another worker or you. Workers are FULL, CAPABLE models: give each a meaty,\n"
    "clearly-specified chunk and it will handle it.\n\n"
    "Decompose the user's coding task into a HANDFUL of SUBSTANTIAL, independent, balanced\n"
    "items — each one a <<<PIECE>>>. SIZE FLOOR: every item is AT LEAST a full class or module —\n"
    "a cohesive component with several methods/functions AND its own tests. NEVER a single\n"
    "function and never a one-liner (tiny items waste time on per-item overhead). When unsure,\n"
    "make pieces BIGGER: a whole class, a whole module, or a whole subsystem per item — err\n"
    "toward fewer, meatier pieces. Avoid one giant item that stragglers the batch, but a class\n"
    "is the minimum unit. Pick a split strategy: file (one module/class per item — PREFER THIS),\n"
    "function (several related functions grouped per item over one shared interface), or role\n"
    "(impl/tests/docs over one target).\n\n"
    "Output EXACTLY this envelope, one <<<PIECE>>> per work item, nothing after <<<END>>>\n"
    "(UPPERCASE = placeholders):\n\n"
    "<<<PLAN strategy=STRATEGY lang=LANG>>>\n"
    "shared:\n"
    "<blueprint>\n"
    "  the frozen interface every item relies on: types and signatures,\n"
    "  and a line  deps: [ allowed libraries, or [] for none ]\n"
    "</blueprint>\n"
    "<<<PIECE id=w1 exports=SYM1,SYM2>>>\n"
    "instruction: what this item must build (a sentence)\n"
    "<blueprint>\n"
    "  a COMPLETE spec for this item — as many lines as needed: signatures, behavior,\n"
    "  edge cases, and tests. Be thorough; the worker sees only this.\n"
    "</blueprint>\n"
    "<<</PIECE>>>\n"
    "<<<PIECE id=w2 exports=SYM>>>\n"
    "instruction: ...\n"
    "<blueprint> ... </blueprint>\n"
    "<<</PIECE>>>\n"
    "(... a few substantial PIECEs, each closed with <<</PIECE>>> so it can dispatch immediately ...)\n"
    "<<<END>>>\n\n"
    "Rules: items must be independent (no shared mutable state mid-flight); exported symbols\n"
    "must not collide; keep items SUBSTANTIAL and similar-sized. ORDER THEM LARGEST-FIRST — put\n"
    "the piece you expect to be the most code (the most involved component) first, so the longest\n"
    "job starts earliest and the lanes finish together (less idle tail). Emit the envelope only.";

// lean worker preamble — the framework lives in the boss, NOT here (design §3/§5.2)
static const char *WORKER_PREAMBLE =
    "You implement ONE piece of a larger program. You are given a frozen shared interface and\n"
    "a spec for YOUR PIECE ONLY. Produce just that piece: the implementation (plus a quick\n"
    "inline test only if natural) for your declared exports, in the given language, and nothing\n"
    "else — no prose, no other functions, no re-declaring the shared interface. Match the\n"
    "declared signatures exactly. Output only code.";

// Tool-call instructions appended to the worker preamble when --tools is set. Format and verb
// names match nanocoder's XML convention (../nanocoder, source/app/prompts + source/tools):
// the tool name is the tag, parameters are nested tags.
static std::string worker_preamble_text() {
    std::string p = WORKER_PREAMBLE;
    if (g_tools) {
        p += "\n\nYou MAY call tools to act on the filesystem. Emit calls in EXACTLY this XML "
             "format (the tool NAME is the tag; each parameter is a nested tag):\n"
             "<create_file>\n<path>relative/path.ext</path>\n<content>\n"
             "...the full file contents...\n</content>\n</create_file>\n"
             "- create_file — write your implementation (and its tests) to a file. RELATIVE paths only.";
        if (g_allow_run)
            p += "\n<execute_bash>\n<command>npm test</command>\n</execute_bash>\n"
                 "- execute_bash — run a quick check or your tests; the output is captured and shown to the coordinator.";
        p += "\nUse the tools to write your files, then also output the code inline as usual.";
    }
    return p;
}

// the common system turn (preamble + shared interface) — identical for every item in a job,
// so its rendered+tokenized form is the cacheable prefix (PA.2.1).
static std::string build_worker_system(const WorkOrder &wo) {
    return worker_preamble_text() + "\n\nShared interface (rely on these):\n" + wo.shared;
}
static std::string build_prefix(const WorkOrder &wo) {
    return apply_chat_template({ {"system", build_worker_system(wo)} }, /*add_ass=*/false);
}

// a lane's full prompt: the common system turn + this item's user turn (the per-item delta)
static std::string build_lane_prompt(const WorkOrder &wo, const Piece &p) {
    std::string user;
    if (p.is_boss) {                       // boss SELF lane: knows the workers' declared exports
        user += "The workers are separately producing these exports: ";
        bool first = true;
        for (auto &q : wo.pieces) {
            if (q.is_boss) continue;
            for (auto &e : q.exports) { user += (first ? "" : ", ") + e; first = false; }
        }
        user += ".\n\n";
    }
    user += "Your piece";
    if (!p.language.empty()) user += " (" + p.language + ")";
    user += ": " + p.instruction + "\n\nSpec:\n" + p.blueprint;
    std::string s = apply_chat_template({ {"system", build_worker_system(wo)}, {"user", user} }, /*add_ass=*/true);
    if (g_no_think) s += "<think>\n\n</think>\n\n";   // skip worker reasoning
    return s;
}

// ───────────────────────── PA.1c: gather phase ───────────────────────────────
// After all workers finish, inject their outputs into the boss and let it
// produce a single merged artifact.

// Extract content from ```lang ... ``` fenced code blocks.
// If no fences found, return the full text unchanged (graceful fallback).
// If multiple fences, return the first one.
static std::string extract_code_fence(const std::string &text) {
    size_t open = text.find("```");
    if (open == std::string::npos) return trim(text);   // no fence → raw code, pass through
    // skip the fence marker + optional lang tag to find newline
    size_t nl = text.find('\n', open + 3);
    if (nl == std::string::npos) nl = open + 3;
    else nl++;  // include the newline
    size_t close = text.find("```", nl);
    // Unclosed fence (e.g. truncated worker output): treat EOF as the close, per
    // markdown convention. Returning the content after the opening fence still
    // strips the ``` marker and any prose before it — more useful for the merge
    // than passing the raw fence through.
    if (close == std::string::npos) return trim(text.substr(nl));
    return trim(text.substr(nl, close - nl));
}

// Build the gather injection text: each worker's output wrapped in markers,
// followed by the gather instruction for the boss.
static std::string build_gather_prompt(
    const WorkOrder &wo,
    const std::vector<std::pair<std::string, std::string>> &worker_results,  // id -> output
    const std::string &boss_self_output)
{
    std::string p;
    p += "Below are the actual outputs from your workers. Use these implementations AS-IS.\n\n";

    for (auto &wr : worker_results) {
        p += "<<<WORKER_DONE id=" + wr.first + ">>>";
        // extract clean code from fenced blocks if present
        p += extract_code_fence(wr.second);
        p += "\n<<<END_WORKER>>>\n\n";
    }

    if (!boss_self_output.empty()) {
        p += "<<<SELF_DONE id=boss>>>\n";
        p += extract_code_fence(boss_self_output);
        p += "\n<<<END_SELF>>>\n\n";
    }

    // declared exports for drift detection
    p += "<<<GATHER_INSTRUCTION>>>\n";
    p += "You have received the outputs from all workers. Your job is to produce a SINGLE, ";
    p += "COMPLETE, RUNNABLE artifact.\n\n";
    p += "For each worker above, you see their actual implementation. Use these implementations ";
    p += "AS-IS — do not rewrite them. Deduplicate overlapping helpers (keep one copy).\n\n";
    p += "Declared exports per worker for reference:\n";
    for (auto &pc : wo.pieces) {
        if (pc.is_boss) continue;
        p += "  " + pc.id + " -> [";
        for (size_t i = 0; i < pc.exports.size(); i++)
            p += (i ? "," : "") + pc.exports[i];
        p += "]\n";
    }

    p += "\nYour output must:\n";
    p += "1. Include the shared interface (types, signatures)\n";
    p += "2. Include every worker's implementation (dedup overlapping helpers — keep one copy)\n";
    p += "3. Include your integration/glue code, UPDATED to match actual worker signatures\n";
    p += "4. Add a dependency manifest if needed\n";
    p += "5. Be wrapped in a single code fence: ```<lang> ... ```\n\n";
    p += "Output ONLY the code fence, nothing else.\n";
    p += "<<<END_GATHER>>>";
    return p;
}

// Delta-prefill tokens into seq at position start_pos (the inject mechanism,
// same as PA.2.1 delta prefill). Returns the last logits index.
static bool inject(llama_seq_id seq, llama_pos start_pos,
                   const std::vector<llama_token> &toks, llama_batch &batch) {
    int n = (int)toks.size();
    for (int i0 = 0; i0 < n; i0 += PREFILL_CHUNK) {
        int nb = n - i0 < PREFILL_CHUNK ? n - i0 : PREFILL_CHUNK;
        batch_clear(&batch);
        for (int j = 0; j < nb; j++)
            batch_add(&batch, toks[i0 + j], (llama_pos)(start_pos + i0 + j), seq,
                      i0 + j == n - 1);
        if (llama_decode(G.ctx, batch) != 0) return false;
    }
    return true;
}

// System prompt for the gather pass — sets the boss's role for the merge. The
// user turn (built by build_gather_prompt) carries the actual instruction + bodies.
static const char *GATHER_SYSTEM =
    "You are the COORDINATOR of a packed-agent coding team. Your workers have finished "
    "their pieces in isolation; integrate them into one cohesive, runnable artifact.";

// Run the gather phase: clear the boss seq, prefill the gather turn, decode the
// merged artifact until EOG or max_gather. Returns the boss's gather output.
static std::string run_gather_phase(llama_seq_id boss_seq,
                                     const std::string &gather_content, int max_gather)
{
    // Fresh boss context for the merge. The seq's KV still holds leftovers — the
    // boss's own plan (non-stream) or the last pool item that ran on this lane
    // (streaming). Prefilling the gather turn on top would write KV cells at
    // positions that already have cells in this seq, duplicating them and
    // corrupting attention. Clear the seq and start the gather turn at pos 0.
    llama_memory_seq_rm(G.mem, boss_seq, 0, -1);

    // Wrap as a chat turn so the instruct model *answers* the merge request rather
    // than continuing raw text (matches boss_generate / the worker prompts).
    std::string prompt = apply_chat_template(
        { {"system", GATHER_SYSTEM}, {"user", gather_content} }, /*add_ass=*/true);
    if (g_no_think) prompt += "<think>\n\n</think>\n\n";

    std::vector<llama_token> gt(prompt.size() + 16, 0);
    int gn = llama_tokenize(G.vocab, prompt.c_str(), (int32_t)prompt.size(),
                            gt.data(), (int32_t)gt.size(), true, true);
    if (gn <= 0) return "[gather tokenize failed]";
    gt.resize(gn);

    llama_batch batch = llama_batch_init(PREFILL_CHUNK + 8, 0, 1);
    if (!inject(boss_seq, 0, gt, batch)) {
        llama_batch_free(batch);
        return "[gather inject failed]";
    }

    llama_token tok = (llama_token)argmax_f(llama_get_logits_ith(G.ctx, gn - 1), G.n_vocab);
    llama_pos pos = (llama_pos)gn;
    std::string out;

    fprintf(stderr, "\n── PA.1c GATHER — boss merging artifact ──\n");
    for (int gen = 0; gen < max_gather && !llama_vocab_is_eog(G.vocab, tok); gen++) {
        std::string piece = tok_str(tok);
        out += piece;
        fputs(piece.c_str(), stderr);
        batch_clear(&batch);
        batch_add(&batch, tok, pos, boss_seq, true);
        if (llama_decode(G.ctx, batch) != 0) break;
        tok = (llama_token)argmax_f(llama_get_logits_ith(G.ctx, 0), G.n_vocab);
        pos++;
    }
    llama_batch_free(batch);
    return out;
}

// Write the final artifact to --out (if set) or stdout.
static void write_artifact(const std::string &artifact) {
    if (g_out_path[0]) {
        FILE *fp = fopen(g_out_path, "w");
        if (fp) { fputs(artifact.c_str(), fp); fclose(fp);
            fprintf(stderr, "\n── artifact written to %s (%zu chars) ──\n", g_out_path, artifact.size());
        } else { perror("fopen --out"); }
    } else {
        printf("\n════════════════════════════════════════════════\n");
        printf("── FINAL ARTIFACT (%zu chars) ──\n%s\n════════════════════════════════════════════════\n",
               artifact.size(), artifact.c_str());
    }
}

// ───────────────────────── PA.5: worker tool-calls (§8.3) ────────────────────
// Workers may emit nanocoder-style XML tool calls (see ../nanocoder): the tool name
// is the tag, parameters are nested tags. The harness executes a small allowlist —
// create_file (sandboxed to --work-dir) and execute_bash (gated behind --allow-run,
// run inside --work-dir with a timeout + a destructive-command guard).
struct ToolCall {
    std::string name;
    std::vector<std::pair<std::string,std::string>> params;
};
static std::string tc_param(const ToolCall &c, const std::string &k) {
    for (auto &p : c.params) if (p.first == k) return p.second;
    return "";
}

// Parse <tool>...<param>value</param>...</tool> for the KNOWN tool names only (so a
// stray <div> etc. in worker code is never mistaken for a tool call).
static std::vector<ToolCall> parse_tool_calls(const std::string &text) {
    static const char *TOOLS[] = { "create_file", "execute_bash" };
    std::vector<ToolCall> calls;
    for (const char *tool : TOOLS) {
        std::string open = std::string("<") + tool + ">", close = std::string("</") + tool + ">";
        size_t pos = 0;
        while (true) {
            size_t o = text.find(open, pos);
            if (o == std::string::npos) break;
            size_t c = text.find(close, o + open.size());
            if (c == std::string::npos) break;
            std::string inner = text.substr(o + open.size(), c - (o + open.size()));
            ToolCall call; call.name = tool;
            size_t pp = 0;
            while (true) {                                   // nested <key>value</key> params
                size_t k0 = inner.find('<', pp);
                if (k0 == std::string::npos) break;
                size_t k1 = inner.find('>', k0);
                if (k1 == std::string::npos) break;
                std::string key = trim(inner.substr(k0 + 1, k1 - (k0 + 1)));
                if (key.empty() || key[0] == '/') { pp = k1 + 1; continue; }   // skip close tags
                std::string kclose = "</" + key + ">";
                size_t v1 = inner.find(kclose, k1 + 1);
                if (v1 == std::string::npos) { pp = k1 + 1; continue; }
                std::string val = inner.substr(k1 + 1, v1 - (k1 + 1));
                if (!val.empty() && val.front() == '\n') val.erase(0, 1);
                if (!val.empty() && val.back()  == '\n') val.pop_back();
                call.params.push_back({ key, val });
                pp = v1 + kclose.size();
            }
            calls.push_back(call);
            pos = c + close.size();
        }
    }
    return calls;
}

// Destructive-command guard for execute_bash — mirrors nanocoder's blocklist.
static bool is_dangerous_cmd(const std::string &cmd) {
    static const std::regex pats[] = {
        std::regex(R"(rm\s+-rf\s+/(?!\w))", std::regex::icase),  // rm -rf / (but allow /path)
        std::regex("mkfs", std::regex::icase),
        std::regex(R"(dd\s+if=)", std::regex::icase),
        std::regex(R"(:\(\)\{:\|:&\};:)"),                       // fork bomb
        std::regex(R"(>\s*/dev/sd[a-z])", std::regex::icase),
        std::regex(R"(chmod\s+-R\s+000)", std::regex::icase),
    };
    for (auto &p : pats) if (std::regex_search(cmd, p)) return true;
    return false;
}

// Resolve a worker-supplied path inside the work dir; reject absolute / traversal.
static bool safe_join(const std::string &rel, std::filesystem::path &out) {
    if (rel.empty() || rel.front() == '/') return false;
    if (rel.find("..") != std::string::npos) return false;
    out = std::filesystem::path(g_work_dir) / rel;
    return true;
}

// Execute the tool calls in each worker's output; return a textual report (files
// written + command output) that gets folded into the gather context.
static std::string run_worker_tools(
    const std::vector<std::pair<std::string,std::string>> &worker_results) {
    namespace fs = std::filesystem;
    std::error_code ec; fs::create_directories(g_work_dir, ec);
    std::string report;
    for (auto &wr : worker_results) {
        for (auto &c : parse_tool_calls(wr.second)) {
            if (c.name == "create_file") {
                std::string rel = tc_param(c, "path");
                if (rel.empty()) rel = tc_param(c, "file_path");
                std::string content = tc_param(c, "content");
                fs::path p;
                if (!safe_join(rel, p)) {
                    fprintf(stderr, "  ⚠ %s: create_file rejected unsafe path '%s'\n", wr.first.c_str(), rel.c_str());
                    report += "  [" + wr.first + "] create_file REJECTED (unsafe path: " + rel + ")\n";
                    continue;
                }
                fs::create_directories(p.parent_path(), ec);
                FILE *fp = fopen(p.string().c_str(), "w");
                if (fp) { fwrite(content.data(), 1, content.size(), fp); fclose(fp);
                    fprintf(stderr, "  ✎ %s wrote %s (%zu bytes)\n", wr.first.c_str(), p.string().c_str(), content.size());
                    report += "  [" + wr.first + "] wrote " + p.string() + " (" + std::to_string(content.size()) + " bytes)\n";
                } else {
                    report += "  [" + wr.first + "] create_file FAILED: " + p.string() + "\n";
                }
            } else if (c.name == "execute_bash") {
                std::string cmd = trim(tc_param(c, "command"));
                if (cmd.empty()) continue;
                if (!g_allow_run) {
                    fprintf(stderr, "  ⏭ %s: execute_bash skipped (--allow-run off): %s\n", wr.first.c_str(), cmd.c_str());
                    report += "  [" + wr.first + "] execute_bash SKIPPED (--allow-run off): " + cmd + "\n";
                    continue;
                }
                if (is_dangerous_cmd(cmd)) {
                    fprintf(stderr, "  ⛔ %s: blocked destructive command: %s\n", wr.first.c_str(), cmd.c_str());
                    report += "  [" + wr.first + "] execute_bash BLOCKED (destructive): " + cmd + "\n";
                    continue;
                }
                std::string full = "cd " + std::string(g_work_dir) + " && timeout 60 " + cmd + " 2>&1";
                fprintf(stderr, "  ▶ %s: execute_bash `%s`\n", wr.first.c_str(), cmd.c_str());
                FILE *pp = popen(full.c_str(), "r");
                if (!pp) { report += "  [" + wr.first + "] execute_bash FAILED to start: " + cmd + "\n"; continue; }
                std::string out; char buf[4096]; size_t n;
                while ((n = fread(buf, 1, sizeof(buf), pp)) > 0) out.append(buf, n);
                int rc = pclose(pp); int code = WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
                fputs(out.c_str(), stderr);
                if (out.size() > 2000) out = out.substr(0, 2000) + "\n...[truncated]...";
                report += "  [" + wr.first + "] ran `" + cmd + "` -> exit " + std::to_string(code) + ":\n" + out + "\n";
            }
        }
    }
    return report;
}

// Drive the gather→write step shared by both pipelines: build the gather prompt,
// run the boss merge, strip the fence, emit the artifact. n_lanes is the worker
// lane count (n_seq_max = n_lanes + 1, the +1 being the prefix-cache seq).
static void finish_gather(const WorkOrder &wo,
                          const std::vector<std::pair<std::string,std::string>> &worker_results,
                          const std::string &boss_self_output, int n_lanes) {
    std::string gprompt    = build_gather_prompt(wo, worker_results, boss_self_output);
    if (g_tools) {
        fprintf(stderr, "\n── PA.5 TOOLS — executing worker tool calls (work-dir: %s, run %s) ──\n",
                g_work_dir, g_allow_run ? "ENABLED" : "disabled");
        std::string report = run_worker_tools(worker_results);
        if (!report.empty())
            gprompt += "\n\n<<<TOOL_RESULTS>>>\nFiles written / commands run by the workers "
                       "(sandboxed in '" + std::string(g_work_dir) + "'):\n" + report + "<<<END_TOOL_RESULTS>>>\n";
    }
    int total_ctx  = (int)llama_n_ctx(G.ctx);
    int max_gather = (total_ctx / (n_lanes + 1)) / 2;   // generous cap for the merge
    if (max_gather < 512) max_gather = 512;
    std::string gather_out = run_gather_phase(0, gprompt, max_gather);
    std::string artifact   = extract_code_fence(gather_out);
    if (artifact.empty()) artifact = trim(gather_out);
    write_artifact(artifact);
}

// Self-test for gather text utilities (GPU-free, like --parse-test)
static int gather_self_test() {
    fprintf(stderr, "── PA.1c gather self-test ──\n");
    int fail = 0;

    // U1: fenced code
    {
        std::string input = "```js\nfunction foo() { return 1; }\n```";
        std::string result = extract_code_fence(input);
        bool ok = result == "function foo() { return 1; }";
        fprintf(stderr, "  U1 fenced js: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) { fprintf(stderr, "    got: \"%s\"\n", result.c_str()); fail++; }
    }

    // U2: fenced with lang tag
    {
        std::string input = "```javascript\nconst x = 42;\n```";
        std::string result = extract_code_fence(input);
        bool ok = result == "const x = 42;";
        fprintf(stderr, "  U2 fenced javascript: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) { fprintf(stderr, "    got: \"%s\"\n", result.c_str()); fail++; }
    }

    // U3: no fences → fallback to full text
    {
        std::string input = "function bar() { return 2; }";
        std::string result = extract_code_fence(input);
        bool ok = result == "function bar() { return 2; }";
        fprintf(stderr, "  U3 no fences: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) { fprintf(stderr, "    got: \"%s\"\n", result.c_str()); fail++; }
    }

    // U4: multiple fences → first only
    {
        std::string input = "```js\nfirst()\n```\n```js\nsecond()\n```";
        std::string result = extract_code_fence(input);
        bool ok = result == "first()";
        fprintf(stderr, "  U4 multiple fences: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) { fprintf(stderr, "    got: \"%s\"\n", result.c_str()); fail++; }
    }

    // U5: fence with prose before/after
    {
        std::string input = "Here is the code:\n```python\ndef hello(): pass\n```\nDone.";
        std::string result = extract_code_fence(input);
        bool ok = result == "def hello(): pass";
        fprintf(stderr, "  U5 fence with prose: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) { fprintf(stderr, "    got: \"%s\"\n", result.c_str()); fail++; }
    }

    // U6: empty string
    {
        std::string input = "";
        std::string result = extract_code_fence(input);
        bool ok = result.empty();
        fprintf(stderr, "  U6 empty: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) { fprintf(stderr, "    got: \"%s\"\n", result.c_str()); fail++; }
    }

    // U7: unclosed fence → fallback
    {
        std::string input = "```js\nfunction unclosed() {";
        std::string result = extract_code_fence(input);
        bool ok = result == "function unclosed() {";
        fprintf(stderr, "  U7 unclosed fence: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) { fprintf(stderr, "    got: \"%s\"\n", result.c_str()); fail++; }
    }

    // U8: build_gather_prompt contains markers and instruction
    {
        std::vector<std::pair<std::string, std::string>> workers = {
            {"w1", "function gravityStep() {}"},
            {"w2", "```js\nfunction spawnPipe() {}\n```"}
        };
        WorkOrder wo; wo.ok = false;
        Piece p1; p1.id = "w1"; p1.exports = {"gravityStep"};
        Piece p2; p2.id = "w2"; p2.exports = {"spawnPipe"};
        wo.pieces.push_back(p1); wo.pieces.push_back(p2);

        std::string prompt = build_gather_prompt(wo, workers, "");
        bool ok = (prompt.find("<<<WORKER_DONE id=w1>>>") != std::string::npos) &&
                  (prompt.find("<<<END_WORKER>>>") != std::string::npos) &&
                  (prompt.find("<<<GATHER_INSTRUCTION>>>") != std::string::npos) &&
                  (prompt.find("<<<END_GATHER>>>") != std::string::npos) &&
                  (prompt.find("w2") != std::string::npos);
        fprintf(stderr, "  U8 gather prompt markers: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) { fprintf(stderr, "    prompt length: %zu\n", prompt.size()); fail++; }
    }

    // U9: build_gather_prompt with 0 workers (edge case)
    {
        WorkOrder wo; wo.ok = false;
        std::string prompt = build_gather_prompt(wo, {}, "");
        bool ok = (prompt.find("<<<GATHER_INSTRUCTION>>>") != std::string::npos) &&
                  (prompt.find("<<<WORKER_DONE") == std::string::npos);
        fprintf(stderr, "  U9 empty gather prompt: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) fail++;
    }

    // U10: worker output contains <<< markers — verify no false matches
    {
        std::vector<std::pair<std::string, std::string>> workers = {
            {"w1", "function test() { if (a <<< b) return a; }"}
        };
        WorkOrder wo; wo.ok = false;
        Piece p1; p1.id = "w1"; p1.exports = {"test"};
        wo.pieces.push_back(p1);

        std::string prompt = build_gather_prompt(wo, workers, "");
        // The <<<WORKER_DONE marker should be present; the <<< inside the code is just text
        bool ok = (prompt.find("<<<WORKER_DONE id=w1>>>") != std::string::npos);
        fprintf(stderr, "  U10 worker with markers: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) fail++;
    }

    // ── PA.5 tool-call parser + safety guard (nanocoder-style <tool><param>…) ──
    // T1: create_file with path + content
    {
        std::string in = "<create_file>\n<path>src/a.js</path>\n<content>\nconst a=1;\n</content>\n</create_file>";
        auto calls = parse_tool_calls(in);
        bool ok = calls.size() == 1 && calls[0].name == "create_file" &&
                  tc_param(calls[0], "path") == "src/a.js" && tc_param(calls[0], "content") == "const a=1;";
        fprintf(stderr, "  T1 parse create_file: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) fail++;
    }
    // T2: execute_bash with command
    {
        auto calls = parse_tool_calls("<execute_bash><command>node test.js</command></execute_bash>");
        bool ok = calls.size() == 1 && calls[0].name == "execute_bash" &&
                  tc_param(calls[0], "command") == "node test.js";
        fprintf(stderr, "  T2 parse execute_bash: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) fail++;
    }
    // T3: stray <div> etc. is NOT a tool call (known-tools-only)
    {
        bool ok = parse_tool_calls("render() { return <div>hi</div>; }").empty();
        fprintf(stderr, "  T3 ignores unknown tags: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) fail++;
    }
    // T4: destructive-command guard (mirrors nanocoder's blocklist)
    {
        bool ok = is_dangerous_cmd("rm -rf /") && is_dangerous_cmd("mkfs.ext4 /dev/sda") &&
                  !is_dangerous_cmd("npm test") && !is_dangerous_cmd("rm -rf /tmp/build");
        fprintf(stderr, "  T4 dangerous-cmd guard: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) fail++;
    }

    fprintf(stderr, "  %s (%d/14 passed)\n", fail == 0 ? "ALL PASS" : "SOME FAILED", 14 - fail);
    return fail > 0 ? 3 : 0;
}

// ───────────────────────── PA.3: streaming decomposition ─────────────────────
// Boss (lane nW) plans WHILE workers (lanes 0..nW-1) execute. Each <<<PIECE>>>…
// <<</PIECE>>> the boss closes is parsed from its live stream and enqueued; a free
// worker clones the cached shared prefix + delta-prefills it and starts immediately.
// After <<<END>>> the boss lane joins the worker pool. Kills the plan-tax idle.
struct QItem { std::string id, instruction, blueprint, language; std::vector<std::string> exports; };

static void run_pipeline(const std::string &task, int max_new, int n_lanes) {
    const int nW   = n_lanes > 1 ? n_lanes - 1 : 1;   // worker lanes 0..nW-1
    const int BOSS = nW;                              // boss lane/seq (joins pool after END)
    const int BASE = nW + 1;                          // cached shared-prefix seq
    llama_batch batch = llama_batch_init(PREFILL_CHUNK + 8, 0, n_lanes + 1);
    double t0 = now_sec();

    struct Lane { int seq; int item; bool live; bool is_boss; std::string text;
                  llama_token tok; llama_pos pos; int n_gen; };
    std::vector<Lane> lanes(nW + 1);
    for (int i = 0; i <= nW; i++) { lanes[i] = Lane{ i, -1, false, false, "", 0, 0, 0 }; }
    lanes[BOSS].is_boss = true;

    std::vector<QItem> queue;
    std::vector<std::string> outv;
    int next_assign = 0, total = 0;

    std::string shared;
    bool prefix_cached = false;
    std::vector<llama_token> prefix_toks; int prefix_len = 0;

    auto cache_prefix = [&](const std::string &sh) -> bool {
        std::string pfx = apply_chat_template({ {"system",
            worker_preamble_text() + "\n\nShared interface (rely on these):\n" + sh} }, false);
        prefix_toks.assign(pfx.size() + 16, 0);
        int pn = llama_tokenize(G.vocab, pfx.c_str(), (int32_t)pfx.size(),
                                prefix_toks.data(), (int32_t)prefix_toks.size(), true, true);
        if (pn <= 0) return false;
        prefix_toks.resize(pn); prefix_len = pn;
        llama_memory_seq_rm(G.mem, BASE, 0, -1);
        Stream tmp; tmp.prompt_toks = prefix_toks; int li = 0;
        return prefill_stream(tmp, BASE, batch, &li);
    };

    // prefill boss prompt into seq BOSS
    G.tmpl = llama_model_chat_template(G.model, nullptr);
    std::string bprompt = apply_chat_template({ {"system", BOSS_PROMPT}, {"user", task} });
    if (g_no_think) bprompt += "<think>\n\n</think>\n\n";
    {
        std::vector<llama_token> bt(bprompt.size() + 16);
        int bn = llama_tokenize(G.vocab, bprompt.c_str(), (int32_t)bprompt.size(), bt.data(), (int32_t)bt.size(), true, true);
        if (bn < 0) { bt.resize(-bn); bn = llama_tokenize(G.vocab, bprompt.c_str(), (int32_t)bprompt.size(), bt.data(), (int32_t)bt.size(), true, true); }
        bt.resize(bn > 0 ? bn : 0);
        llama_memory_seq_rm(G.mem, BOSS, 0, -1);
        Stream tmp; tmp.prompt_toks = bt; int li = 0;
        if (!prefill_stream(tmp, BOSS, batch, &li)) { fprintf(stderr, "boss prefill failed\n"); llama_batch_free(batch); return; }
        lanes[BOSS].tok = (llama_token)argmax_f(llama_get_logits_ith(G.ctx, li), G.n_vocab);
        lanes[BOSS].pos = (llama_pos)bt.size();
        lanes[BOSS].live = true;
    }
    fprintf(stderr, "\n══ PA.3 streaming pipeline — boss(lane %d) + %d workers (live) ══\n", BOSS, nW);

    size_t parse_cur = 0; bool boss_done = false;
    auto pump_boss = [&]() {
        std::string &bp = lanes[BOSS].text;
        if (!prefix_cached) {
            // Shared block = the <blueprint> before the first <<<PIECE. Do NOT require a
            // literal <<<PLAN marker: the model sometimes drops the leading <<< on the PLAN
            // line (while emitting <<<PIECE / <<<END correctly), which used to stall the whole
            // run — no shared cached → prefix 0 tok → no worker ever starts.
            size_t fp = bp.find("<<<PIECE");
            if (fp != std::string::npos) {
                shared = extract_blueprint(bp.substr(0, fp));
                if (cache_prefix(shared)) { prefix_cached = true;
                    fprintf(stderr, "\n  [shared cached: %d tok — workers can start]\n", prefix_len); }
            }
        }
        for (;;) {
            size_t open = bp.find("<<<PIECE", parse_cur);
            if (open == std::string::npos) break;
            size_t oend = bp.find(">>>", open);
            if (oend == std::string::npos) break;
            size_t close = bp.find("<<</PIECE>>>", oend);
            if (close == std::string::npos) break;
            std::string omark = bp.substr(open, oend + 3 - open), tag;
            std::vector<std::pair<std::string,std::string>> at;
            parse_marker(omark, tag, at);
            std::string body = bp.substr(oend + 3, close - (oend + 3));
            QItem q; q.id = attr_get(at, "id"); q.exports = split_csv(attr_get(at, "exports"));
            q.language = attr_get(at, "lang"); q.instruction = extract_instruction(body); q.blueprint = extract_blueprint(body);
            queue.push_back(q); outv.push_back("");
            fprintf(stderr, "\n  [enqueued %s — %d queued]\n", q.id.c_str(), (int)queue.size());
            parse_cur = close + 12;
        }
        if (bp.find("<<<END>>>", parse_cur) != std::string::npos) boss_done = true;
    };

    auto start_worker = [&](int L, int qi) -> bool {
        const QItem &q = queue[qi];
        std::string user = "Your piece";
        if (!q.language.empty()) user += " (" + q.language + ")";
        user += ": " + q.instruction + "\n\nSpec:\n" + q.blueprint;
        std::string full = apply_chat_template({ {"system",
            worker_preamble_text() + "\n\nShared interface (rely on these):\n" + shared}, {"user", user} }, true);
        if (g_no_think) full += "<think>\n\n</think>\n\n";
        std::vector<llama_token> ft(full.size() + 16);
        int fn = llama_tokenize(G.vocab, full.c_str(), (int32_t)full.size(), ft.data(), (int32_t)ft.size(), true, true);
        if (fn <= 0) return false;
        ft.resize(fn);
        bool cached = prefix_cached && fn > prefix_len && (fn - prefix_len) <= PREFILL_CHUNK;
        for (int k = 0; cached && k < prefix_len; k++) if (ft[k] != prefix_toks[k]) cached = false;
        llama_memory_seq_rm(G.mem, lanes[L].seq, 0, -1);
        int last_idx = 0;
        if (cached) {
            llama_memory_seq_cp(G.mem, BASE, lanes[L].seq, 0, -1);
            batch_clear(&batch);
            int dn = fn - prefix_len;
            for (int j = 0; j < dn; j++) batch_add(&batch, ft[prefix_len + j], (llama_pos)(prefix_len + j), lanes[L].seq, j == dn - 1);
            if (llama_decode(G.ctx, batch) != 0) return false;
            last_idx = dn - 1;
        } else {
            Stream tmp; tmp.prompt_toks = ft; if (!prefill_stream(tmp, lanes[L].seq, batch, &last_idx)) return false;
        }
        lanes[L].tok = (llama_token)argmax_f(llama_get_logits_ith(G.ctx, last_idx), G.n_vocab);
        lanes[L].pos = (llama_pos)fn; lanes[L].text = tok_str(lanes[L].tok);
        lanes[L].n_gen = 1; lanes[L].live = true; lanes[L].item = qi; total++;
        fprintf(stderr, "  → %s → lane %d (%s)\n", q.id.c_str(), L, cached ? "cloned+delta" : "full");
        return true;
    };

    double tg = now_sec();
    for (;;) {
        if (prefix_cached)
            for (int L = 0; L <= nW && next_assign < (int)queue.size(); L++)
                if (!lanes[L].live && lanes[L].item < 0 && (!lanes[L].is_boss || boss_done))
                    start_worker(L, next_assign++);

        batch_clear(&batch);
        std::vector<int> ord;
        for (int L = 0; L <= nW; L++) if (lanes[L].live) { batch_add(&batch, lanes[L].tok, lanes[L].pos, lanes[L].seq, true); ord.push_back(L); }
        if (ord.empty()) break;
        if (llama_decode(G.ctx, batch) != 0) { fprintf(stderr, "pipeline decode failed\n"); break; }
        for (size_t i = 0; i < ord.size(); i++) {
            int L = ord[i];
            llama_token nxt = (llama_token)argmax_f(llama_get_logits_ith(G.ctx, (int)i), G.n_vocab);
            lanes[L].pos++; lanes[L].tok = nxt; lanes[L].n_gen++; total++;
            if (lanes[L].is_boss && !boss_done) {
                std::string pc = tok_str(nxt); lanes[L].text += pc; fputs(pc.c_str(), stderr);
                pump_boss();
                if (boss_done || llama_vocab_is_eog(G.vocab, nxt) || lanes[L].n_gen >= max_new * 3) {
                    boss_done = true; lanes[L].live = false; lanes[L].item = -1;
                    fprintf(stderr, "\n  [boss done planning (%d tok) — joins the worker pool]\n", lanes[L].n_gen);
                }
            } else {
                if (llama_vocab_is_eog(G.vocab, nxt) || lanes[L].n_gen >= max_new) {
                    outv[lanes[L].item] = lanes[L].text;
                    fprintf(stderr, "  ✓ %s done (%d tok)\n", queue[lanes[L].item].id.c_str(), lanes[L].n_gen);
                    lanes[L].live = false; lanes[L].item = -1;
                } else lanes[L].text += tok_str(nxt);
            }
        }
    }
    double wall = now_sec() - tg;
    llama_batch_free(batch);

    fprintf(stderr, "\n════════════════════════════════════════════════\n");
    fprintf(stderr, "  PA.3 streaming pipeline result\n");
    fprintf(stderr, "  %d items, %d worker lanes + boss, prefix %d tok\n", (int)queue.size(), nW, prefix_len);
    fprintf(stderr, "  %d tok in %.1fs concurrent (plan overlapped), %.1fs total\n", total, wall, now_sec() - t0);
    fprintf(stderr, "  aggregate  : %.1f tok/s = %.2fx vs 19.3 baseline\n", wall > 0 ? total / wall : 0.0, wall > 0 ? total / wall / 19.3 : 0.0);
    fprintf(stderr, "════════════════════════════════════════════════\n");
    // ── PA.1c GATHER (streaming path) — boss merges the pieces into one artifact ──
    // The boss joined the worker pool after planning, so every outv[i] is a worker
    // piece (no separate SELF lane here). Build a WorkOrder view of the queue so the
    // gather prompt can list each piece's declared exports for drift detection.
    if (!g_no_gather && !queue.empty()) {
        WorkOrder wo; wo.shared = shared; wo.ok = true;
        std::vector<std::pair<std::string, std::string>> worker_results;
        for (size_t i = 0; i < queue.size() && i < outv.size(); i++) {
            Piece p; p.id = queue[i].id; p.exports = queue[i].exports; p.language = queue[i].language;
            wo.pieces.push_back(p);
            std::string body = outv[i];
            size_t th = body.find("</think>"); if (th != std::string::npos) body = body.substr(th + 8);
            worker_results.push_back({ queue[i].id, trim(body) });
        }
        finish_gather(wo, worker_results, "", n_lanes);
    } else {
        for (size_t i = 0; i < queue.size(); i++) {
            std::string body = outv[i]; size_t th = body.find("</think>"); if (th != std::string::npos) body = body.substr(th + 8);
            printf("\n───── item %s ─────\n%s\n", queue[i].id.c_str(), trim(body).c_str());
        }
    }
}

int main(int argc, char **argv) {
    char  model_path[512] = {};
    char  task[2048] = {};
    int   max_new   = 96;
    int   n_ctx     = 16384;
    int   n_streams = 4;
    bool  show_text = false;
    bool  parse_test = false;
    bool  gather_test = false;
    int   pool_items = 0;
    bool  plan_only  = false;
    bool  kv_q8      = true;    // Q8 KV default: byte-exactness is a spec-dec artifact, not needed for agents (~2x context)
    bool  no_stream  = false;   // --no-stream: old sequential plan->pool path (for A/B vs streaming)

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-m")   && i+1 < argc) strncpy(model_path, argv[++i], sizeof(model_path)-1);
        else if (!strcmp(argv[i], "-p")   && i+1 < argc) strncpy(task, argv[++i], sizeof(task)-1);
        else if (!strcmp(argv[i], "-n")   && i+1 < argc) max_new   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c")   && i+1 < argc) n_ctx     = atoi(argv[++i]);
        else if ((!strcmp(argv[i], "-s") || !strcmp(argv[i], "--streams")) && i+1 < argc) n_streams = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--parse-test")) parse_test = true;
        else if (!strcmp(argv[i], "--pool")  && i+1 < argc) pool_items = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--plan-only")) plan_only = true;
        else if (!strcmp(argv[i], "--kv-q8"))    kv_q8 = true;
        else if (!strcmp(argv[i], "--kv-f16"))   kv_q8 = false;   // opt into f16: reproducible (byte-identical) runs
        else if (!strcmp(argv[i], "--no-think")) g_no_think = true;
        else if (!strcmp(argv[i], "--no-stream")) no_stream = true;
        else if (!strcmp(argv[i], "--text"))     show_text = true;
        else if (!strcmp(argv[i], "--out")     && i+1 < argc) strncpy(g_out_path, argv[++i], sizeof(g_out_path)-1);
        else if (!strcmp(argv[i], "--no-gather")) g_no_gather = true;
        else if (!strcmp(argv[i], "--gather-test")) gather_test = true;
        else if (!strcmp(argv[i], "--tools")) g_tools = true;
        else if (!strcmp(argv[i], "--allow-run")) { g_allow_run = true; g_tools = true; }  // run implies tools
        else if (!strcmp(argv[i], "--work-dir") && i+1 < argc) strncpy(g_work_dir, argv[++i], sizeof(g_work_dir)-1);
        else if (!strcmp(argv[i], "--verbose"))  g_verbose_logs = true;
        else { fprintf(stderr, "Usage: %s -m <model> [-p \"task\"] [-s streams(1-%d)] [-n max] [-c ctx] [--text] [--parse-test] [--pool M] [--plan-only] [--kv-q8|--kv-f16] [--no-think] [--no-stream] [--out FILE] [--no-gather] [--tools] [--allow-run] [--work-dir DIR] [--verbose]\n", argv[0], MAX_STREAMS); return 1; }
    }
    if (parse_test)  return parse_self_test();   // PA.1a: GPU-free envelope parser check
    if (gather_test) return gather_self_test();  // PA.1c: GPU-free gather self-test
    if (!model_path[0]) { fprintf(stderr, "Error: -m required\n"); return 1; }
    if (n_streams < 1) n_streams = 1;
    if (n_streams > MAX_STREAMS) { fprintf(stderr, "note: clamping -s to MAX_STREAMS=%d\n", MAX_STREAMS); n_streams = MAX_STREAMS; }
    if (task[0] && max_new == 96) max_new = 768;   // small pooled work items: think + a function
    if (task[0] && n_ctx == 16384) n_ctx = 131072; // 128k pipeline context — Q8 KV fits it (≈ f16 @ 64k bytes)
    if (pool_items > 0 && max_new == 96) max_new = 256;   // PA.2 pool items: think + a short function
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
    cp.n_seq_max = (uint32_t)n_streams + (task[0] ? 1u : 0u);  // pipeline reserves a base seq for the prefix cache (PA.2.1)
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.kv_unified      = false;   // M7.2 exactness: unified KV flips fp near-ties by batch shape
    if (kv_q8) {                  // ~2x context; trades byte-determinism (M7.2: Q8 KV is batch-shape-dependent under FA)
        cp.type_k = GGML_TYPE_Q8_0;
        cp.type_v = GGML_TYPE_Q8_0;
    }
    G.ctx = llama_init_from_model(G.model, cp);
    if (!G.ctx) { fprintf(stderr, "context failed\n"); return 1; }
    G.mem = llama_get_memory(G.ctx);

    // ── PA.1 pipeline (PA.1a = PLAN phase): boss decomposes the task ──────────
    if (task[0]) {
        if (!no_stream) {                 // PA.3: streaming decomposition — default for -p
            run_pipeline(task, max_new, n_streams);
            llama_free(G.ctx); llama_model_free(G.model); llama_backend_free();
            return 0;
        }
        G.tmpl = llama_model_chat_template(G.model, nullptr);
        std::vector<Msg> msgs = {{"system", BOSS_PROMPT}, {"user", task}};
        std::string prompt = apply_chat_template(msgs);
        if (g_no_think) prompt += "<think>\n\n</think>\n\n";   // skip the boss's plan reasoning (kills the plan tax)
        fprintf(stderr, "\n══ PA.1 PLAN — boss decomposing ══\n  task: %s\n\n", task);
        double t0 = now_sec();
        int plan_cap = n_ctx / (n_streams + 1) - 768;  // full seq budget (n_seq_max=n_streams+1 w/ base) — it needs
        if (plan_cap < 1024) plan_cap = 1024;      // room to break down many items / work a blueprint
        std::string raw = boss_generate(prompt, plan_cap);
        double el = now_sec() - t0;
        std::string plan = raw;                         // drop <think> preamble if present
        size_t th = plan.find("</think>");
        if (th != std::string::npos) plan = plan.substr(th + 8);
        fprintf(stderr, "\n── boss plan done (%.1fs, %zu chars; streamed above) ──\n── parsed ──\n",
                el, trim(plan).size());
        WorkOrder wo = parse_work_order(plan);
        print_work_order(wo);
        if (!wo.ok) { llama_free(G.ctx); llama_model_free(G.model); llama_backend_free(); return 2; }
        if (plan_only) { llama_free(G.ctx); llama_model_free(G.model); llama_backend_free(); return 0; }

        // ── PA.2: pool the work items over n_streams lanes (refill keeps the batch full) ──
        std::vector<std::string> items;
        for (auto &p : wo.pieces) items.push_back(build_lane_prompt(wo, p));
        fprintf(stderr, "\n══ PA.2 POOL — %d work items, %d lanes ══\n", (int)items.size(), n_streams);
        std::vector<std::string> outputs;
        run_pool(items, n_streams, max_new, &outputs, build_prefix(wo));   // PA.2.1: cache preamble+shared
        // ── PA.1c GATHER ─────────────────────────────────────────────────────
        if (!g_no_gather) {
            // build worker_results vector: id -> output (strip thinking tags)
            std::vector<std::pair<std::string, std::string>> worker_results;
            std::string boss_self_output;
            for (size_t i = 0; i < wo.pieces.size() && i < outputs.size(); i++) {
                std::string body = outputs[i];
                size_t bth = body.find("</think>");
                if (bth != std::string::npos) body = body.substr(bth + 8);
                body = trim(body);
                if (wo.pieces[i].is_boss) {
                    boss_self_output = body;
                } else {
                    worker_results.push_back({ wo.pieces[i].id, body });
                }
            }
            finish_gather(wo, worker_results, boss_self_output, n_streams);
        } else {
            for (size_t i = 0; i < wo.pieces.size() && i < outputs.size(); i++) {
                std::string body = outputs[i];
                size_t bth = body.find("</think>");
                if (bth != std::string::npos) body = body.substr(bth + 8);
                printf("\n───── item %s ─────\n%s\n", wo.pieces[i].id.c_str(), trim(body).c_str());
            }
        }
        llama_free(G.ctx);
        llama_model_free(G.model);
        llama_backend_free();
        return 0;
    }

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

    // ── PA.2: work-pool test — first M built-in prompts as a queue over n_streams lanes ──
    if (pool_items > 0) {
        int M = pool_items < MAX_STREAMS ? pool_items : MAX_STREAMS;
        std::vector<std::string> items;
        for (int i = 0; i < M; i++) items.push_back(prompts[i]);
        fprintf(stderr, "\n══ PA.2 work-pool — %d items, %d lanes, cap %d ══\n", M, n_streams, max_new);
        run_pool(items, n_streams, max_new);
        llama_free(G.ctx); llama_model_free(G.model); llama_backend_free();
        return 0;
    }

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
    fprintf(stderr, "\n── byte-identity diagnostic (packed vs solo — informational, not pass/fail) ──\n");
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
        fprintf(stderr, "  all %d lanes identical packed-vs-solo (reproducible run)\n", n_streams);
    else
        fprintf(stderr, "  %d/%d lanes diverged — near-tie variance (expected under Q8 / higher N; valid output, not an error)\n", gate_fail, n_streams);
    fprintf(stderr, "════════════════════════════════════════════════\n");

    if (show_text) {
        for (int s = 0; s < n_streams; s++) {
            printf("\n───── stream %d ─────\n%s\n", s, par_streams[s].text.c_str());
        }
    }

    llama_free(G.ctx);
    llama_model_free(G.model);
    llama_backend_free();
    return 0;   // byte-identity is a diagnostic for agents, not pass/fail (Q6_K weights → no absolute reference)
}
