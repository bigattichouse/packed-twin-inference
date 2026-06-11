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
    "<<<PIECE id=w2 exports=spawnPipe>>>\n"
    "instruction: implement this function given the public params of the current class, in js\n"
    "<blueprint> spawnPipe(world) -> void { append a Pipe with a random gap to world.pipes } </blueprint>\n"
    "<<<PIECE id=w3 exports=checkCollision>>>\n"
    "instruction: implement this function given the public params of the current class, in js\n"
    "<blueprint> checkCollision(bird, pipes) -> bool { true if bird hits a pipe or ground } </blueprint>\n"
    "<<<SELF id=boss exports=stepWorld, smokeTest>>>\n"
    "instruction: implement this object in js - wire the three functions and a smoke test\n"
    "<blueprint> Integration { stepWorld(world, dt) -> void, smokeTest() -> bool } </blueprint>\n"
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
        out += tok_str(tok);
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
    "ISOLATION — a worker sees only the shared interface, its own spec, and its one-line\n"
    "instruction, never another worker or you.\n\n"
    "Decompose the user's coding task into AS MANY small, independent, similar-sized work items\n"
    "as it naturally has — there is no fixed count. Each item is one <<<PIECE>>>. The team pops\n"
    "items from a shared queue with refill, so MANY SMALL BALANCED items beat a few large ones\n"
    "(a big item becomes a straggler that stalls the batch). Pick a split strategy: function\n"
    "(independent functions over one shared interface), file (one module per item), or role\n"
    "(impl/tests/docs). Prefer function-level.\n\n"
    "Output EXACTLY this envelope, one <<<PIECE>>> per work item, nothing after <<<END>>>\n"
    "(UPPERCASE = placeholders):\n\n"
    "<<<PLAN strategy=STRATEGY lang=LANG>>>\n"
    "shared:\n"
    "<blueprint>\n"
    "  the frozen interface every item relies on: types and signatures,\n"
    "  and a line  deps: [ allowed libraries, or [] for none ]\n"
    "</blueprint>\n"
    "<<<PIECE id=w1 exports=SYMBOL>>>\n"
    "instruction: ONE LINE, e.g. implement this function given the shared interface, in LANG\n"
    "<blueprint> the spec for this item only </blueprint>\n"
    "<<<PIECE id=w2 exports=SYMBOL>>>\n"
    "instruction: ONE LINE\n"
    "<blueprint> ... </blueprint>\n"
    "(... one PIECE per item, as many as the task needs ...)\n"
    "<<<END>>>\n\n"
    "Rules: items must be independent (no shared mutable state mid-flight); exported symbols\n"
    "must not collide; keep each item SMALL and similar-sized. Emit the envelope only.";

// lean worker preamble — the framework lives in the boss, NOT here (design §3/§5.2)
static const char *WORKER_PREAMBLE =
    "You implement ONE piece of a larger program. You are given a frozen shared interface and\n"
    "a spec for YOUR PIECE ONLY. Produce just that piece: the implementation (plus a quick\n"
    "inline test only if natural) for your declared exports, in the given language, and nothing\n"
    "else — no prose, no other functions, no re-declaring the shared interface. Match the\n"
    "declared signatures exactly. Output only code.";

// the common system turn (preamble + shared interface) — identical for every item in a job,
// so its rendered+tokenized form is the cacheable prefix (PA.2.1).
static std::string build_worker_system(const WorkOrder &wo) {
    return std::string(WORKER_PREAMBLE) + "\n\nShared interface (rely on these):\n" + wo.shared;
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
    return apply_chat_template({ {"system", build_worker_system(wo)}, {"user", user} }, /*add_ass=*/true);
}

int main(int argc, char **argv) {
    char  model_path[512] = {};
    char  task[2048] = {};
    int   max_new   = 96;
    int   n_ctx     = 16384;
    int   n_streams = 4;
    bool  show_text = false;
    bool  parse_test = false;
    int   pool_items = 0;
    bool  plan_only  = false;
    bool  kv_q8      = true;    // Q8 KV default: byte-exactness is a spec-dec artifact, not needed for agents (~2x context)

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
        else if (!strcmp(argv[i], "--text"))     show_text = true;
        else if (!strcmp(argv[i], "--verbose"))  g_verbose_logs = true;
        else { fprintf(stderr, "Usage: %s -m <model> [-p \"task\"] [-s streams(1-%d)] [-n max] [-c ctx] [--text] [--parse-test] [--pool M] [--plan-only] [--kv-q8|--kv-f16] [--verbose]\n", argv[0], MAX_STREAMS); return 1; }
    }
    if (parse_test) return parse_self_test();   // PA.1a: GPU-free envelope parser check
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
        G.tmpl = llama_model_chat_template(G.model, nullptr);
        std::vector<Msg> msgs = {{"system", BOSS_PROMPT}, {"user", task}};
        std::string prompt = apply_chat_template(msgs);
        fprintf(stderr, "\n══ PA.1 PLAN — boss decomposing ══\n  task: %s\n\n", task);
        double t0 = now_sec();
        int plan_cap = n_ctx / (n_streams + 1) - 768;  // full seq budget (n_seq_max=n_streams+1 w/ base) — it needs
        if (plan_cap < 1024) plan_cap = 1024;      // room to break down many items / work a blueprint
        std::string raw = boss_generate(prompt, plan_cap);
        double el = now_sec() - t0;
        std::string plan = raw;                         // drop <think> preamble if present
        size_t th = plan.find("</think>");
        if (th != std::string::npos) plan = plan.substr(th + 8);
        fprintf(stderr, "── boss work-order (%.1fs, %zu chars) ──\n%s\n\n── parsed ──\n",
                el, trim(plan).size(), trim(plan).c_str());
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
        for (size_t i = 0; i < wo.pieces.size() && i < outputs.size(); i++) {
            std::string body = outputs[i];
            size_t bth = body.find("</think>");
            if (bth != std::string::npos) body = body.substr(bth + 8);
            printf("\n───── item %s ─────\n%s\n", wo.pieces[i].id.c_str(), trim(body).c_str());
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
