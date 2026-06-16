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
#include "../llama.cpp/src/llama-ext.h"   // pre-norm hidden access for MTP (PA.3)

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
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
static char g_out_path[512] = {};     // --out: write final artifact to this file
static bool g_no_gather    = false;   // --no-gather: skip gather phase, print pieces separately
static bool g_tools        = false;   // --tools: let workers emit <function=...> tool calls (write_file/run)
static bool g_allow_run    = false;   // --allow-run: permit the `run` verb (sandboxed shell exec); implies --tools
static char g_work_dir[512] = "pti_work";  // --work-dir: sandbox dir for write_file / run
static bool g_mtp          = false;   // --mtp: per-lane MTP drafting (PA.3) — doubles n_seq_max
// Qwen3.6 recommended sampling (model card), PER ROLE. Default: boss reasons (thinking), workers
// implement a clear spec without reasoning (faster). NEVER greedy in thinking mode (Qwen:
// greedy → repetition/degradation). --mtp is the one greedy path. Resolved in main().
struct SParams { float temp; float top_p; float min_p; float presence; int top_k; };
static SParams  g_boss_sp     = {0.6f, 0.95f, 0.0f, 0.0f, 20};   // boss: thinking/coding
static SParams  g_worker_sp   = {0.7f, 0.80f, 0.0f, 1.5f, 20};   // workers: instruct (no-think)
static bool     g_boss_think   = true;    // boss plans/gathers with reasoning
static bool     g_worker_think = false;   // workers implement from the spec without reasoning
static float    g_temp     = -1.0f;   // -t: override temperature for both roles
static uint32_t g_seed     = 42;      // base seed; lane L uses g_seed + L (per-stream streams)
static bool     g_general  = false;   // --general: boss uses thinking-general temps (1.0) vs coding (0.6)
static bool     g_greedy   = false;   // resolved: true under --mtp (greedy speculative decode)
static int      g_repair_budget = 2;  // --repair-budget: max verify→repair rounds (PA.4c)
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
    llama_model       *model   = nullptr;
    const llama_vocab *vocab   = nullptr;
    llama_context     *ctx     = nullptr;
    llama_memory_t     mem     = nullptr;
    int32_t            n_vocab = 0;
    const char        *tmpl    = nullptr;   // chat template (PA.1 boss prompting)
    llama_context     *ctx_mtp = nullptr;   // PA.3: nextn-head draft context (--mtp)
    int32_t            n_embd  = 0;         // model hidden size (MTP embd feed)
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

// ───────────────────────── PA.3: per-lane MTP drafting ───────────────────────
// Pure bookkeeping (model-free, unit-tested via --mtp-test). See spec/PA3_MTP_DESIGN.md.

// Per-lane sequence layout when --mtp is on (else: lanes 0..n-1, base n).
static int mtp_ckpt_seq(int lane, int n_lanes) { return n_lanes + lane; }  // checkpoint of lane L
static int mtp_base_seq(int n_lanes)           { return 2 * n_lanes; }     // prefix-cache seq
static int mtp_seqmax(int n_lanes, bool mtp_on){ return mtp_on ? 2 * n_lanes + 1 : n_lanes + 1; }

// Verdict of verifying one packed MTP step for a lane: how many tokens were emitted
// (e), how many drafts accepted (acc), whether it was a full accept, whether to stop.
struct MtpVerdict { int e; int acc; bool full; bool stop; };

// a0 = real next token (argmax at tok_last); a1 = next-next (argmax at draft slot,
// only meaningful on a full accept). EOG short-circuits before consuming the draft.
static MtpVerdict mtp_verify(bool has_draft, llama_token draft,
                             llama_token a0, bool a0_eog,
                             llama_token a1, bool a1_eog) {
    (void)a1;
    MtpVerdict v{1, 0, false, false};
    if (a0_eog) { v.stop = true; return v; }      // emitted a0 (EOG) — never draft past it
    if (has_draft && a0 == draft) {               // draft matched the real next token
        v.e = 2; v.acc = 1; v.full = true;
        if (a1_eog) v.stop = true;
    }
    return v;
}

// ───────────────────────── PA.3: MTP draft context + feed ────────────────────
// Create the nextn-head draft context (one MTP seq per lane). Returns false if the
// model has no nextn head (caller runs greedy). Mirrors pti_lookup M7.1.
static bool setup_mtp(int n_lanes) {
    llama_set_embeddings_pre_norm(G.ctx, true, false);    // expose pre-norm hidden on main ctx
    llama_context_params mp = llama_context_default_params();
    mp.n_ctx     = llama_n_ctx(G.ctx);
    mp.n_batch   = PREFILL_CHUNK;
    mp.n_seq_max = n_lanes;
    mp.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
    mp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    G.ctx_mtp = llama_init_from_model(G.model, mp);
    if (G.ctx_mtp) {
        llama_set_embeddings_pre_norm(G.ctx_mtp, true, true);
        fprintf(stderr, "MTP draft context: active (%d lanes)\n", n_lanes);
        return true;
    }
    fprintf(stderr, "MTP draft context: UNAVAILABLE (no nextn head?) — running greedy\n");
    return false;
}

// Feed one (emitted token, its pre-norm hidden from the MAIN ctx) pair into the MTP
// context at (seq, pos); argmax of the nextn logits = draft for pos+1. Last-pair-only
// (n=1), matching pti_lookup. Returns -1 on failure. `mb` must be embd-capable.
static llama_token mtp_feed1(llama_batch *mb, llama_seq_id seq,
                             llama_token tok, int h_idx, llama_pos pos) {
    if (!G.ctx_mtp) return -1;
    const float *h = llama_get_embeddings_pre_norm_ith(G.ctx, h_idx);
    if (!h) return -1;
    mb->n_tokens     = 1;
    mb->token[0]     = tok;
    mb->pos[0]       = pos;
    mb->n_seq_id[0]  = 1;
    mb->seq_id[0][0] = seq;
    mb->logits[0]    = 1;
    memcpy(mb->embd, h, (size_t)G.n_embd * sizeof(float));
    if (llama_decode(G.ctx_mtp, *mb) != 0) return -1;
    const float *logits = llama_get_logits_ith(G.ctx_mtp, 0);
    if (!logits) return -1;
    return (llama_token)argmax_f(logits, G.n_vocab);
}

// ───────────────────────── Qwen sampling (model-card defaults) ───────────────
// Qwen3.6 model-card params for a role given (thinking?, general?).
static SParams qwen_params(bool think, bool general) {
    if (!think)  return { 0.7f, 0.80f, 0.0f, 1.5f, 20 };   // instruct / non-thinking
    if (general) return { 1.0f, 0.95f, 0.0f, 0.0f, 20 };   // thinking — general
    return             { 0.6f, 0.95f, 0.0f, 0.0f, 20 };    // thinking — precise coding
}
// One chain per lane (per-lane seed → independent reproducible streams; penalties
// keep per-lane history). Order: penalties → top_k → top_p → min_p → temp → dist.
static llama_sampler *make_sampler(uint32_t seed, const SParams &sp) {
    llama_sampler *s = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (sp.presence > 0.0f)  // instruct mode / quantized-repetition guard (Qwen: presence 1.5)
        llama_sampler_chain_add(s, llama_sampler_init_penalties(1024, 1.0f, 0.0f, sp.presence));
    llama_sampler_chain_add(s, llama_sampler_init_top_k(sp.top_k));
    llama_sampler_chain_add(s, llama_sampler_init_top_p(sp.top_p, 1));
    llama_sampler_chain_add(s, llama_sampler_init_min_p(sp.min_p, 1));
    llama_sampler_chain_add(s, llama_sampler_init_temp(sp.temp));
    llama_sampler_chain_add(s, llama_sampler_init_dist(seed));
    return s;
}
// Pick a token from the logits at batch index `idx`: sampler if present, else greedy
// (greedy only on the --mtp path and prefill seeds; thinking decode always samples).
static llama_token pick(llama_sampler *s, int idx) {
    if (!s) return (llama_token)argmax_f(llama_get_logits_ith(G.ctx, idx), G.n_vocab);
    return llama_sampler_sample(s, G.ctx, idx);
}

// ───────────────────────── PA.2: work-pool ───────────────────────────────────
// M items over n_lanes; a lane that finishes (EOG or cap) is refilled from the
// queue (seq_rm + prefill the next item), keeping the batch full while backlog
// lasts. The pooled fix for the PA.1b straggler. With --mtp (PA.3) each lane also
// carries a nextn draft, verified in the same batch, with per-lane SSM rollback.
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
    // seq layout: working lanes 0..n_lanes-1; with --mtp, checkpoints n_lanes..2n-1,
    // base 2n (mtp_seqmax). Without --mtp, base = n_lanes (the original layout).
    const llama_seq_id BASE = g_mtp ? (llama_seq_id)mtp_base_seq(n_lanes) : (llama_seq_id)n_lanes;
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

    // ── PA.3 MTP per-lane state ──────────────────────────────────────────────
    bool mtp_on = g_mtp && G.ctx_mtp;
    std::vector<llama_token> cand(n_lanes, -1);        // current draft per lane (-1 = none)
    llama_batch mtp_batch = {};
    llama_memory_t mem_mtp = nullptr;
    long n_fire = 0, n_acc = 0;                        // MTP stats
    if (mtp_on) {
        mtp_batch = llama_batch_init(8, G.n_embd, 1);  // embd-capable, 1 pair/feed
        mtp_batch.token = (llama_token *)malloc(8 * sizeof(llama_token));
        mem_mtp = llama_get_memory(G.ctx_mtp);
    }
    // per-lane Qwen samplers (null when greedy / --mtp); seed g_seed+L → independent streams
    std::vector<llama_sampler *> smpl(n_lanes, nullptr);
    if (!g_greedy) for (int L = 0; L < n_lanes; L++) smpl[L] = make_sampler(g_seed + (uint32_t)L, g_worker_sp);

    auto start_lane = [&](int L, int item) -> bool {
        std::vector<llama_token> full(items[item].size() + 16, 0);
        int fn = llama_tokenize(G.vocab, items[item].c_str(), (int32_t)items[item].size(),
                                full.data(), (int32_t)full.size(), true, true);
        if (fn <= 0) return false;
        full.resize(fn);
        bool cached = use_cache && fn > prefix_len && (fn - prefix_len) <= PREFILL_CHUNK;
        for (int k = 0; cached && k < prefix_len; k++) if (full[k] != prefix_toks[k]) cached = false;
        llama_memory_seq_rm(G.mem, L, 0, -1);
        if (smpl[L]) llama_sampler_reset(smpl[L]);             // fresh sampler history for the new item
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
        lanes[L].tok_last = pick(smpl[L], last_idx);          // sample (Qwen) or greedy (--mtp)
        lanes[L].pos      = (llama_pos)fn;
        lanes[L].text     = tok_str(lanes[L].tok_last);
        lanes[L].n_gen    = 1;
        lanes[L].live     = !llama_vocab_is_eog(G.vocab, lanes[L].tok_last);
        lane_item[L]      = item;
        total++;
        if (mtp_on) {                                   // seed this lane's draft from the prefill hidden
            llama_memory_seq_rm(mem_mtp, L, 0, -1);     // fresh MTP cache for the (re)started lane
            cand[L] = lanes[L].live
                    ? mtp_feed1(&mtp_batch, L, lanes[L].tok_last, last_idx, lanes[L].pos)
                    : -1;
        }
        fprintf(stderr, "  → item %d → lane %d (%s)\n", item, L, cached ? "cloned+delta" : "full prefill");
        return true;
    };

    double t0 = now_sec();
    for (int L = 0; L < n_lanes && next < M; L++) start_lane(L, next++);
    double prefill0 = now_sec() - t0;

    double t1 = now_sec();
    if (mtp_on) {
      // ── PA.3 packed MTP loop: per-lane draft + batched verify + SSM rollback ──
      struct Slot { int lane, tok_idx, draft_idx; bool draft; llama_token tok_old; llama_pos p0; };
      for (;;) {
        std::vector<int> ord;
        for (int L = 0; L < n_lanes; L++)
            if (lanes[L].live && lanes[L].n_gen < max_new) ord.push_back(L);
        if (ord.empty()) break;

        // refresh each lane's checkpoint S(pos) and build the [tok_last, draft?] batch
        batch_clear(&batch);
        std::vector<Slot> slots;
        for (int L : ord) {
            llama_seq_id cseq = (llama_seq_id)mtp_ckpt_seq(L, n_lanes);
            llama_memory_seq_rm(G.mem, cseq, 0, -1);
            llama_memory_seq_cp(G.mem, L, cseq, 0, -1);          // checkpoint = S(pos)
            Slot s; s.lane = L; s.tok_old = lanes[L].tok_last; s.p0 = lanes[L].pos;
            s.draft = (cand[L] != -1);
            s.tok_idx = batch.n_tokens;
            batch_add(&batch, lanes[L].tok_last, lanes[L].pos, L, true);
            if (s.draft) { s.draft_idx = batch.n_tokens;
                batch_add(&batch, cand[L], lanes[L].pos + 1, L, true); }
            else s.draft_idx = -1;
            slots.push_back(s);
        }
        if (llama_decode(G.ctx, batch) != 0) { fprintf(stderr, "pool(mtp) decode failed\n"); break; }

        // verify + emit; queue MTP feeds (read pre-norm BEFORE any rebuild decode)
        struct Feed { llama_seq_id seq; llama_token tok; int h_idx; llama_pos pos; int lane; };
        std::vector<Feed>  feeds;
        std::vector<int>   finished;
        std::vector<Slot>  rebuild;
        for (auto &s : slots) {
            int L = s.lane;
            llama_token a0 = (llama_token)argmax_f(llama_get_logits_ith(G.ctx, s.tok_idx), G.n_vocab);
            bool a0_eog = llama_vocab_is_eog(G.vocab, a0);
            llama_token a1 = 0; bool a1_eog = false;
            if (s.draft && a0 == cand[L]) {
                a1 = (llama_token)argmax_f(llama_get_logits_ith(G.ctx, s.draft_idx), G.n_vocab);
                a1_eog = llama_vocab_is_eog(G.vocab, a1);
            }
            MtpVerdict v = mtp_verify(s.draft, cand[L], a0, a0_eog, a1, a1_eog);
            if (s.draft) { n_fire++; n_acc += v.acc; }

            if (!a0_eog) lanes[L].text += tok_str(a0);
            lanes[L].n_gen++; total++;
            llama_token last_emit = a0; int last_h = s.tok_idx;
            if (v.full) {
                if (!a1_eog) lanes[L].text += tok_str(a1);
                lanes[L].n_gen++; total++;
                last_emit = a1; last_h = s.draft_idx;
            }
            lanes[L].pos += v.e; lanes[L].tok_last = last_emit;

            bool done = v.stop || lanes[L].n_gen >= max_new;
            if (!done) feeds.push_back({ (llama_seq_id)L, last_emit, last_h, lanes[L].pos, L });
            if (done) {
                out[lane_item[L]] = lanes[L].text;
                fprintf(stderr, "  ✓ item %d done (%d tok)\n", lane_item[L], lanes[L].n_gen);
                lanes[L].live = false; finished.push_back(L);
            } else if (!v.full) {
                rebuild.push_back(s);                              // missed → needs rollback
            }
        }

        // next drafts (uses the main decode's pre-norm hidden — must precede rebuild)
        for (auto &f : feeds) cand[f.lane] = mtp_feed1(&mtp_batch, f.seq, f.tok, f.h_idx, f.pos);

        // rollback missed lanes to S(p0) and re-consume tok_old → S(p0+1)
        if (!rebuild.empty()) {
            batch_clear(&batch);
            for (auto &s : rebuild) {
                llama_seq_id cseq = (llama_seq_id)mtp_ckpt_seq(s.lane, n_lanes);
                llama_memory_seq_rm(G.mem, s.lane, 0, -1);
                llama_memory_seq_cp(G.mem, cseq, s.lane, 0, -1);  // restore S(p0)
                batch_add(&batch, s.tok_old, s.p0, s.lane, false);
            }
            if (llama_decode(G.ctx, batch) != 0) { fprintf(stderr, "pool(mtp) rebuild failed\n"); break; }
        }

        for (int L : finished) {                                  // refill from the queue, else idle
            if (next < M) { start_lane(L, next++); refills++; }
            else { lane_item[L] = -1; cand[L] = -1; }
        }
      }
    } else {
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
            llama_token nxt = pick(smpl[L], (int)i);            // Qwen sampling per lane
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
    }
    double wall = now_sec() - t1;
    for (auto *s : smpl) if (s) llama_sampler_free(s);
    if (mtp_on) { free(mtp_batch.token); mtp_batch.token = nullptr; llama_batch_free(mtp_batch); }
    llama_batch_free(batch);

    double agg = wall > 0 ? total / wall : 0.0;
    fprintf(stderr, "\n════════════════════════════════════════════════\n");
    fprintf(stderr, "  PA.2 work-pool result\n");
    fprintf(stderr, "  %d items, %d lanes, cap %d/item, %d refills\n", M, n_lanes, max_new, refills);
    if (use_cache) fprintf(stderr, "  prefix cache: %d tok cloned per lane (delta-prefill only the item)\n", prefix_len);
    fprintf(stderr, "  %d tok in %.1fs decode (+%.1fs initial prefill)\n", total, wall, prefill0);
    fprintf(stderr, "  aggregate  : %.1f tok/s = %.2fx vs 19.3 baseline\n", agg, agg / 19.3);
    if (mtp_on) fprintf(stderr, "  MTP drafts : %ld fired, %ld accepted (%.0f%%)\n",
                        n_fire, n_acc, n_fire ? 100.0 * n_acc / n_fire : 0.0);
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
        std::string el = e; for (auto &c : el) c = (char)tolower((unsigned char)c);
        if (el.empty() || el == "none") continue;   // sentinel: piece exports nothing (index.html, *.test.js)
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
    // Regression: the "none" sentinel (index.html, *.test.js) must NOT count as an export collision.
    const char *NONE_ENV =
        "<<<PLAN strategy=file lang=js>>>\nshared:\n<blueprint>\nx\n</blueprint>\n"
        "<<<PIECE id=bird exports=Bird>>>\ninstruction: src/bird.js\n<blueprint>b</blueprint>\n<<</PIECE>>>\n"
        "<<<PIECE id=index exports=none>>>\ninstruction: index.html\n<blueprint>i</blueprint>\n<<</PIECE>>>\n"
        "<<<PIECE id=style exports=none>>>\ninstruction: style.css\n<blueprint>s</blueprint>\n<<</PIECE>>>\n"
        "<<<END>>>\n";
    WorkOrder w2 = parse_work_order(NONE_ENV);
    fprintf(stderr, "  none-sentinel: %s (%s)\n", w2.ok ? "PASS" : "FAIL",
            w2.ok ? "no false collision" : w2.error.c_str());
    return (wo.ok && w2.ok) ? 0 : 2;
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
    llama_sampler *s = g_greedy ? nullptr : make_sampler(g_seed, g_boss_sp);   // boss plan sampler (Qwen)
    llama_token tok = pick(s, last_idx);
    llama_pos pos = n;
    std::string out;
    for (int gen = 0; gen < max_tok && !llama_vocab_is_eog(G.vocab, tok); gen++) {
        { std::string piece = tok_str(tok); out += piece; fputs(piece.c_str(), stderr); }   // stream live
        batch_clear(&batch);
        batch_add(&batch, tok, pos, 0, true);
        if (llama_decode(G.ctx, batch) != 0) break;
        tok = pick(s, 0);
        pos++;
    }
    if (s) llama_sampler_free(s);
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

// Boss system prompt + a tools-mode addendum (files on disk, dir structure, per-module tests,
// the final verifier). Used wherever the boss is prompted.
static std::string boss_system_text() {
    std::string p = BOSS_PROMPT;
    if (g_tools)
        p += "\n\nTOOLS / FILES MODE: workers write REAL files to disk, exactly ONE file per piece. "
             "Plan a clean DIRECTORY STRUCTURE (state it in the shared block, e.g. src/ for modules). "
             "Emit one piece per MODULE (e.g. src/bird.js) plus a piece for the entry point "
             "(index.html). Do NOT write tests yourself: after the modules are built, the harness "
             "auto-generates a unit test for each module (given your file + the shared design) and a "
             "VERIFIER runs them — the job is DONE only when the tests pass. So make modules small, "
             "focused, and TESTABLE, with clear exports declared in the shared interface.";
    return p;
}

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
        p += "\n\nYou MUST save your work to files with this tool (the tool NAME is the tag; each "
             "parameter is a nested tag):\n"
             "<create_file>\n<path>relative/path.ext</path>\n<content>\n...full file...\n</content>\n</create_file>\n"
             "- Write your module to its file, using the exact path/folders the shared interface specifies.\n"
             "- ALSO write a matching test file (e.g. <name>.test.js) with console.assert checks that call "
             "process.exit(1) on ANY failure — it must be runnable as `node <name>.test.js` and exit "
             "non-zero when it fails. A VERIFIER runs every test file at the end; your code must pass.\n"
             "Emit one create_file per file (module, then its test). RELATIVE paths only.";
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
    if (!g_worker_think) s += "<think>\n\n</think>\n\n";   // workers implement from spec, no reasoning
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
    if (!g_boss_think) prompt += "<think>\n\n</think>\n\n";   // gather is a boss-role turn

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

    llama_sampler *s = g_greedy ? nullptr : make_sampler(g_seed, g_boss_sp);   // gather sampler (Qwen)
    // inject() prefills in PREFILL_CHUNK batches → the last token's logits live at the end of the
    // final batch, NOT at global index gn-1. Use -1 (last output); gn-1 overflows for gn>chunk.
    llama_token tok = pick(s, -1);
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
        tok = pick(s, 0);
        pos++;
    }
    if (s) llama_sampler_free(s);
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

// PA.4: build a "write a test for this module" task — given file X (the written module) and
// design Y. The test-maker is a fresh session, so it gets the SAME context an implementer had:
// the project goal/contract ("we're building THIS") + this component's blueprint ("here is your
// part + the spec around it") + the actual code. Tests target the spec, not just the code as-written.
static std::string build_test_task(const std::string &relpath, const std::string &content,
                                   const std::string &goal, const std::string &blueprint) {
    std::filesystem::path p(relpath);
    std::string base = p.filename().string();
    size_t dot = base.rfind(".js"); if (dot != std::string::npos) base = base.substr(0, dot);
    std::string testpath = "test/" + base + ".test.js";
    std::string user =
        "We are building THIS project — goal + interface contract:\n"
        + (goal.empty() ? "(no goal supplied)" : goal) +
        "\n\nYour part: write a Node.js unit test for the '" + base + "' module. Test it against its "
        "SPEC (below), not merely whatever the current code happens to do.\n\nSpec / blueprint for '"
        + base + "':\n" + (blueprint.empty() ? "(none on disk)" : blueprint) +
        "\n\nModule file (" + relpath + "):\n```js\n" + content +
        "\n```\n\nThe test MUST: require the module (e.g. const m = require('../" + relpath + "')), exercise "
        "its exported behavior with console.assert, print a line per check, and call process.exit(1) on "
        "ANY failure so `node` exits non-zero. Save it with create_file at path: " + testpath +
        "\nOutput only the create_file tool call.";
    std::string s = apply_chat_template({ {"system", worker_preamble_text()}, {"user", user} }, true);
    if (!g_worker_think) s += "<think>\n\n</think>\n\n";   // test-writers are instruct lanes
    return s;
}

// ───────────────────────── PA.4c: verify→repair helpers ──────────────────────
// Pure, unit-tested via --coord-test. The loop that uses them is wired into the verifier.
// Which module a failing test targets: "bird.test.js" → the module whose base is "bird".
static std::string module_for_test(const std::string &test_filename,
                                   const std::vector<std::string> &modules) {
    std::string base = test_filename;
    size_t s = base.find(".test.js"); if (s != std::string::npos) base = base.substr(0, s);
    for (auto &m : modules)
        if (std::filesystem::path(m).filename().string() == base + ".js") return m;
    return "";
}
// Amend instruction (user turn) for a failing module — fix the code so its test passes.
// Gives the worker the full context (it's a fresh session): SPEC + module + test + error.
static std::string build_amend_user(const std::string &base, const std::string &modpath,
                                    const std::string &modcontent, const std::string &testcontent,
                                    const std::string &error, const std::string &spec) {
    return "Component '" + base + "' FAILED its test. Fix the MODULE so the test passes (do not change "
           "the test).\n\nSpec / blueprint for '" + base + "':\n" + (spec.empty() ? "(none)" : spec) +
           "\n\nModule (" + modpath + "):\n```\n" + modcontent + "\n```\n\nTest:\n```\n" + testcontent +
           "\n```\n\nTest output / error:\n" + error +
           "\n\nRe-emit the corrected module via create_file at " + modpath + ". Output only the tool call.";
}
// Repair scheduler verdict given the round, the budget, and how many tests still fail.
enum RepairAction { RA_DONE, RA_REPAIR, RA_GIVEUP };
static RepairAction repair_verdict(int round, int budget, int n_failed) {
    if (n_failed == 0) return RA_DONE;
    if (round >= budget) return RA_GIVEUP;
    return RA_REPAIR;
}
// PA.4d: parse the boss arbiter's rework requests — <<<REWORK file=PATH>>> guidance <<<END>>>
static std::vector<std::pair<std::string,std::string>> parse_rework(const std::string &text) {
    std::vector<std::pair<std::string,std::string>> out;
    const std::string open = "<<<REWORK file=";
    size_t pos = 0;
    while (true) {
        size_t o = text.find(open, pos); if (o == std::string::npos) break;
        size_t fe = text.find(">>>", o); if (fe == std::string::npos) break;
        std::string file = trim(text.substr(o + open.size(), fe - (o + open.size())));
        size_t end = text.find("<<<END>>>", fe); if (end == std::string::npos) break;
        std::string guide = trim(text.substr(fe + 3, end - (fe + 3)));
        if (!file.empty() && file.find("..") == std::string::npos) out.push_back({ file, guide });
        pos = end + 9;
    }
    return out;
}
// PA.4d arbiter prompt (boss): decide rework on L1-exhausted failures — the boss DELEGATES via REWORK.
static std::string build_arbiter_user(const std::string &contract, const std::string &failblock) {
    return "Worker-level repair exhausted its budget on the failures below and gave up. As the "
           "COORDINATOR, decide the rework. IMPORTANT: the MODULE may be correct and the TEST may be "
           "BUGGY — judge which file is actually wrong. You DELEGATE (parallel workers do the fix): emit "
           "one rework request per file to rewrite —\n<<<REWORK file=relative/path>>>\nwhat is wrong and "
           "how to fix it\n<<<END>>>\n(module OR test). Emit none to accept the failures as-is.\n\n"
           "Interface contract:\n" + contract + "\n\nFailing pieces (module, test, error):\n" + failblock;
}
// PA.4d rework task (worker): rewrite ONE file, given the full context (fresh session needs it all):
// spec + module + test + error + the boss's guidance.
static std::string build_rework_user(const std::string &target, const std::string &spec,
                                     const std::string &modcontent, const std::string &testcontent,
                                     const std::string &error, const std::string &guidance) {
    return "A test is failing. Study the FULL context below, then rewrite ONLY '" + target + "' to fix it.\n\n"
           "Spec / blueprint:\n" + (spec.empty() ? "(none)" : spec) +
           "\n\nModule:\n```\n" + (modcontent.empty() ? "(none)" : modcontent) +
           "\n```\n\nTest:\n```\n" + (testcontent.empty() ? "(none)" : testcontent) +
           "\n```\n\nTest output / error:\n" + error +
           "\n\nCoordinator's guidance:\n" + guidance +
           "\n\nRe-emit the COMPLETE corrected '" + target + "' via create_file at " + target + ". Output only the tool call.";
}

// GPU-free self-test for the repair bookkeeping (like --gather-test / --mtp-test).
static int coord_self_test() {
    fprintf(stderr, "── PA.4c coord self-test ──\n");
    int fail = 0;
    auto chk = [&](const char *n, bool ok){ fprintf(stderr, "  %s: %s\n", n, ok?"PASS":"FAIL"); if(!ok) fail++; };
    { std::vector<std::string> m={"src/bird.js","src/pipes.js"};
      chk("R1 module_for_test",  module_for_test("bird.test.js", m) == "src/bird.js"); }
    { std::vector<std::string> m={"src/bird.js"};
      chk("R2 no-match empty",   module_for_test("zzz.test.js", m).empty()); }
    { std::string u = build_amend_user("bird","src/bird.js","CODE","TEST","ERR","SPEC");
      chk("R3 amend has parts",  u.find("src/bird.js")!=std::string::npos && u.find("CODE")!=std::string::npos
                              && u.find("TEST")!=std::string::npos && u.find("ERR")!=std::string::npos
                              && u.find("SPEC")!=std::string::npos); }
    chk("R4 verdict done",   repair_verdict(0,3,0) == RA_DONE);
    chk("R5 verdict repair", repair_verdict(0,3,2) == RA_REPAIR);
    chk("R6 verdict giveup", repair_verdict(3,3,2) == RA_GIVEUP);
    { auto rw = parse_rework("noise <<<REWORK file=test/a.test.js>>>\nfix the arc spy\n<<<END>>> tail");
      chk("R7 parse_rework", rw.size()==1 && rw[0].first=="test/a.test.js" && rw[0].second=="fix the arc spy"); }
    { auto rw = parse_rework("no markers here"); chk("R8 parse_rework empty", rw.empty()); }
    { std::string u = build_rework_user("src/bird.js","SPEC","MODC","TESTC","ERRC","GUIDE");   // fresh worker: full triad
      chk("R9 rework full ctx", u.find("src/bird.js")!=std::string::npos && u.find("SPEC")!=std::string::npos
                             && u.find("MODC")!=std::string::npos && u.find("TESTC")!=std::string::npos
                             && u.find("ERRC")!=std::string::npos && u.find("GUIDE")!=std::string::npos); }
    { std::string u = build_test_task("src/bird.js","CODEX","GOALX","BPX");   // test-maker: goal + spec + code
      chk("R10 testgen ctx", u.find("GOALX")!=std::string::npos && u.find("BPX")!=std::string::npos
                          && u.find("CODEX")!=std::string::npos); }
    fprintf(stderr, "  %s (%d/10 passed)\n", fail==0 ? "ALL PASS" : "SOME FAILED", 10-fail);
    return fail > 0 ? 5 : 0;
}

static std::string read_file_str(const std::string &path);   // defined in the PA.6 block below
static std::string strip_think(const std::string &in);        // defined in the PA.6 block below

// PA.4: files-on-disk finalize + the "test verifier" (replaces merge-gather under --tools).
//   1) STORE modules to disk; 2) TEST-GEN: a test task per module (given its file + the design),
//   run as a second pool; 3) STORE tests; 4) VERIFY: run every *.test.js (done only when green).
static void finalize_verify(const WorkOrder &wo,
                            const std::vector<std::pair<std::string,std::string>> &worker_results,
                            int n_lanes, int max_new) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fprintf(stderr, "\n══ PA.4 STORE — writing module files to %s ══\n", g_work_dir);
    run_worker_tools(worker_results);                    // modules → disk

    // ── TEST-GEN: one test task per module (*.js, not *.test.js) ──
    std::vector<std::string> mods;
    for (auto it = fs::recursive_directory_iterator(g_work_dir, ec);
         it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) break;
        if (!it->is_regular_file(ec)) continue;
        std::string n = it->path().filename().string();
        if (n.size() >= 3 && n.substr(n.size()-3) == ".js" &&
            !(n.size() >= 8 && n.substr(n.size()-8) == ".test.js")) mods.push_back(it->path().string());
    }
    std::sort(mods.begin(), mods.end());
    if (!mods.empty()) {
        fprintf(stderr, "\n══ PA.4 TEST-GEN — writing a test per module (given goal + spec + file): %d module(s) ══\n", (int)mods.size());
        // The project goal/contract: the reconciled INTERFACE.md if present, else the triage shared block.
        std::string contract = read_file_str(std::string(g_work_dir) + "/design/INTERFACE.md");
        std::string goal_ctx = contract.empty() ? wo.shared : contract;
        std::vector<std::string> items, ids;
        for (auto &m : mods) {
            std::string rel = fs::relative(m, g_work_dir, ec).string();
            std::string content; FILE *f = fopen(m.c_str(), "r");
            if (f) { char b[8192]; size_t k; while ((k = fread(b,1,sizeof(b),f)) > 0) content.append(b,k); fclose(f); }
            std::string comp = fs::path(rel).filename().string();
            { size_t j = comp.rfind(".js"); if (j != std::string::npos) comp = comp.substr(0, j); }
            std::string bp = read_file_str(std::string(g_work_dir) + "/design/" + comp + ".blueprint");
            items.push_back(build_test_task(rel, content, goal_ctx, bp));
            ids.push_back(rel);
        }
        std::vector<std::string> touts;
        run_pool(items, n_lanes, max_new, &touts, "");   // test-writer lanes
        std::vector<std::pair<std::string,std::string>> test_results;
        for (size_t i = 0; i < ids.size() && i < touts.size(); i++) {
            std::string body = touts[i]; size_t th = body.find("</think>");
            if (th != std::string::npos) body = body.substr(th + 8);
            test_results.push_back({ ids[i], trim(body) });
        }
        fprintf(stderr, "\n══ PA.4 STORE — writing test files ══\n");
        run_worker_tools(test_results);
    }

    // ── VERIFY + REPAIR (PA.4c): run all tests; on failure amend the module & re-verify ──
    struct TestRes { std::string rel; bool ok; std::string out; };
    auto run_all_tests = [&]() {
        std::vector<TestRes> res; std::vector<std::string> tests;
        for (auto it = fs::recursive_directory_iterator(g_work_dir, ec);
             it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (ec) break;
            if (it->is_regular_file(ec)) { std::string n = it->path().filename().string();
                if (n.size() >= 8 && n.substr(n.size()-8) == ".test.js") tests.push_back(it->path().string()); }
        }
        std::sort(tests.begin(), tests.end());
        for (auto &t : tests) {
            std::string rel = fs::relative(t, g_work_dir, ec).string();
            std::string cmd = "cd '" + std::string(g_work_dir) + "' && timeout 30 node '" + rel + "' 2>&1";
            FILE *pp = popen(cmd.c_str(), "r");
            std::string out; if (pp){ char b[4096]; size_t k; while((k=fread(b,1,sizeof(b),pp))>0) out.append(b,k); }
            int rc = pp ? pclose(pp) : -1; int code = (pp && WIFEXITED(rc)) ? WEXITSTATUS(rc) : -1;
            res.push_back({ rel, code == 0, out });
        }
        return res;
    };
    auto list_modules = [&]() {
        std::vector<std::string> mods;
        for (auto it = fs::recursive_directory_iterator(g_work_dir, ec);
             it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (ec) break;
            if (!it->is_regular_file(ec)) continue;
            std::string n = it->path().filename().string();
            if (n.size() >= 3 && n.substr(n.size()-3) == ".js" && !(n.size()>=8 && n.substr(n.size()-8)==".test.js"))
                mods.push_back(it->path().string());
        }
        return mods;
    };

    // Repair/rework is precise debugging — let those lanes REASON (coding 0.6), like designers.
    // Items above are built against the live g_worker_think, so set it before the loop runs them.
    bool sv_wt = g_worker_think; SParams sv_sp = g_worker_sp;
    if (!g_greedy) { g_worker_think = true; g_worker_sp = qwen_params(true, false); }

    bool arbiter_done = false;   // PA.4d: boss arbiter escalates once on L1 budget exhaustion
    for (int round = 0; ; round++) {
        std::vector<TestRes> res = run_all_tests();
        fprintf(stderr, "\n══ PA.4 VERIFY%s — %d test file(s) ══\n", round ? " (re-check)" : "", (int)res.size());
        if (res.empty()) { fprintf(stderr, "  ⚠ NO *.test.js produced — cannot verify\n"); break; }
        int passed = 0; std::vector<TestRes> fails;
        for (auto &r : res) {
            if (r.ok) { passed++; fprintf(stderr, "  [PASS] %s\n", r.rel.c_str()); }
            else { fails.push_back(r); fprintf(stderr, "  [FAIL] %s\n", r.rel.c_str());
                   std::string o = r.out; if (o.size() > 300) o = o.substr(0,300) + "..."; if (!o.empty()) fprintf(stderr, "      %s\n", o.c_str()); }
        }
        RepairAction act = repair_verdict(round, g_repair_budget, (int)fails.size());
        if (act == RA_DONE)   { fprintf(stderr, "══ VERIFY RESULT: %d/%d passed — DONE (all green) ══\n", passed, (int)res.size()); break; }
        if (act == RA_GIVEUP) {
            if (!arbiter_done && !fails.empty()) {   // PA.4d: escalate to the boss once; it re-queues rework
                arbiter_done = true;
                fprintf(stderr, "\n══ PA.4d ARBITER — escalating %d failure(s) to the boss ══\n", (int)fails.size());
                std::string contract = read_file_str(std::string(g_work_dir) + "/design/INTERFACE.md");
                std::vector<std::string> mods = list_modules(); std::string fb;
                for (auto &r : fails) {
                    std::string testfn = std::filesystem::path(r.rel).filename().string();
                    std::string modpath = module_for_test(testfn, mods);
                    std::string modrel = modpath.empty() ? std::string("(none)") : fs::relative(modpath, g_work_dir, ec).string();
                    fb += "\n--- test " + r.rel + "  (module " + modrel + ") ---\n";
                    if (!modpath.empty()) fb += "MODULE " + modrel + ":\n" + read_file_str(modpath) + "\n";
                    fb += "TEST " + r.rel + ":\n" + read_file_str(std::string(g_work_dir) + "/" + r.rel) + "\nERROR:\n" + r.out + "\n";
                }
                std::string aprompt = apply_chat_template({ {"system", boss_system_text()}, {"user", build_arbiter_user(contract, fb)} });
                auto reworks = parse_rework(strip_think(boss_generate(aprompt, 2048)));   // boss judges (thinks)
                fprintf(stderr, "  boss requested %d rework item(s)\n", (int)reworks.size());
                if (!reworks.empty()) {
                    std::vector<std::string> ritems, rids;
                    for (auto &rw : reworks) {
                        std::string tgt = rw.first;
                        std::string fn = std::filesystem::path(tgt).filename().string(), comp = fn;
                        { size_t s = comp.find(".test.js");
                          if (s != std::string::npos) comp = comp.substr(0, s);
                          else { size_t j = comp.rfind(".js"); if (j != std::string::npos) comp = comp.substr(0, j); } }
                        std::string modpath = module_for_test(comp + ".test.js", mods);
                        std::string spec = read_file_str(std::string(g_work_dir) + "/design/" + comp + ".blueprint");
                        std::string modc = modpath.empty() ? "" : read_file_str(modpath);
                        std::string testc, err;                 // pull the comp's test + error from fails
                        for (auto &r : fails) {
                            std::string tb = std::filesystem::path(r.rel).filename().string();
                            size_t s = tb.find(".test.js"); if (s != std::string::npos) tb = tb.substr(0, s);
                            if (tb == comp) { testc = read_file_str(std::string(g_work_dir) + "/" + r.rel); err = r.out; break; }
                        }
                        std::string u = build_rework_user(tgt, spec, modc, testc, err, rw.second);
                        std::string sp = apply_chat_template({ {"system", worker_preamble_text()}, {"user", u} }, true);
                        if (!g_worker_think) sp += "<think>\n\n</think>\n\n";
                        ritems.push_back(sp); rids.push_back(tgt);
                    }
                    std::vector<std::string> routs; run_pool(ritems, n_lanes, max_new, &routs, "");
                    std::vector<std::pair<std::string,std::string>> rres;
                    for (size_t i = 0; i < rids.size() && i < routs.size(); i++) rres.push_back({ rids[i], strip_think(routs[i]) });
                    run_worker_tools(rres);
                    continue;   // re-verify after the boss-directed rework
                }
            }
            fprintf(stderr, "══ VERIFY RESULT: %d/%d passed — GIVEN UP after %d round(s)%s ══\n",
                    passed, (int)res.size(), g_repair_budget, arbiter_done ? " + boss arbiter" : "");
            break;
        }

        // RA_REPAIR: amend each failing module (given its file + the test + the error), then re-verify
        fprintf(stderr, "\n══ PA.4c REPAIR round %d/%d — amending %d failing module(s) ══\n", round+1, g_repair_budget, (int)fails.size());
        std::vector<std::string> mods = list_modules(), aitems, aids;
        for (auto &r : fails) {
            std::string testfn = std::filesystem::path(r.rel).filename().string();
            std::string modpath = module_for_test(testfn, mods);
            if (modpath.empty()) { fprintf(stderr, "  (no module maps to %s — skip)\n", testfn.c_str()); continue; }
            std::string base = testfn; size_t s = base.find(".test.js"); if (s != std::string::npos) base = base.substr(0, s);
            std::string modrel = fs::relative(modpath, g_work_dir, ec).string();
            std::string spec = read_file_str(std::string(g_work_dir) + "/design/" + base + ".blueprint");
            std::string user = build_amend_user(base, modrel, read_file_str(modpath),
                                                read_file_str(std::string(g_work_dir) + "/" + r.rel), r.out, spec);
            std::string sp = apply_chat_template({ {"system", worker_preamble_text()}, {"user", user} }, true);
            if (!g_worker_think) sp += "<think>\n\n</think>\n\n";
            aitems.push_back(sp); aids.push_back(modrel);
        }
        if (aitems.empty()) { fprintf(stderr, "  nothing repairable; stopping\n"); break; }
        std::vector<std::string> aouts; run_pool(aitems, n_lanes, max_new, &aouts, "");
        std::vector<std::pair<std::string,std::string>> ares;
        for (size_t i = 0; i < aids.size() && i < aouts.size(); i++) {
            std::string b = aouts[i]; size_t t = b.find("</think>"); if (t != std::string::npos) b = b.substr(t + 8);
            ares.push_back({ aids[i], trim(b) });
        }
        run_worker_tools(ares);   // overwrite the fixed modules; loop re-verifies
    }
    g_worker_think = sv_wt; g_worker_sp = sv_sp;   // restore (implementers = instruct)
}

// ───────────────────────── PA.6: staged design→build pipeline ─────────────────
// Light triage → goal blueprint → parallel DESIGN pool (blueprints) → parallel
// IMPLEMENT pool (modules, reading blueprints) → test-gen + verify. Parallelizes
// the design thinking (the serial plan tax). See spec/PA6_PIPELINE_DESIGN.md.
static const char *TRIAGE_PROMPT =
    "You are the COORDINATOR. You ASSIGN work to other agents — you do NOT build code or tests "
    "yourself. Separate DESIGNER agents write each component's blueprint, IMPLEMENTER agents write the "
    "code, and the harness generates the tests. Your only job here: a BRIEF component MAP for a "
    "multi-file project — names, responsibilities, exports, paths. Emit ONLY this envelope:\n\n"
    "<<<PLAN strategy=file lang=LANG>>>\n"
    "shared:\n<blueprint>\n  one or two lines: how the components connect (the goal); deps: []\n</blueprint>\n"
    "<<<PIECE id=NAME exports=SYM,...>>>\n"
    "instruction: one line — responsibility + target file path (e.g. src/bird.js)\n"
    "<blueprint> one line — key responsibility, NOT a spec </blueprint>\n"
    "<<</PIECE>>>\n"
    "(one PIECE per component; short lowercase ids like bird, pipes, renderer, engine; include an "
    "index.html piece with id=index)\n"
    "<<<END>>>\n\n"
    "Keep it to names + one-line responsibilities + exports + paths. Do NOT write specs or code.\n"
    "Do NOT create test pieces (no *.test ids, no NAME.test.js paths) EVEN IF the task asks for tests — "
    "the harness writes one unit test per module automatically after the modules are built. List ONLY "
    "the real source modules + the index. Use exports=none for pieces that export nothing (e.g. index).";

static std::string read_file_str(const std::string &path) {
    std::string s; FILE *f = fopen(path.c_str(), "r");
    if (f) { char b[8192]; size_t k; while ((k = fread(b,1,sizeof(b),f)) > 0) s.append(b,k); fclose(f); }
    return s;
}

// the goal blueprint: the component map every lane carries (global context, §3)
static std::string build_goal_blueprint(const WorkOrder &wo) {
    std::string g = "GOAL BLUEPRINT — the project's components and how they connect:\n" + wo.shared + "\nComponents:\n";
    for (auto &p : wo.pieces) {
        g += "  - " + p.id + " — " + p.instruction + "  [exports: ";
        for (size_t i = 0; i < p.exports.size(); i++) g += (i ? "," : "") + p.exports[i];
        g += "]\n";
    }
    return g;
}

// DESIGN task: one designer per component → design/<id>.blueprint (run with g_worker_think=true)
static std::string build_design_task(const Piece &p, const std::string &goal) {
    std::string exp; for (size_t i = 0; i < p.exports.size(); i++) exp += (i ? "," : "") + p.exports[i];
    std::string user =
        "You are the DESIGNER for component '" + p.id + "'.\n\n" + goal +
        "\nYour component '" + p.id + "': " + p.instruction + "\nExports: " + exp + "\n\n"
        "Write a concise BLUEPRINT for '" + p.id + "': its exported API (names, params, types), behavior, "
        "key edge cases, and how it interacts with the other components above. Precise prose/pseudocode, "
        "NOT full code. Save it with create_file at path design/" + p.id + ".blueprint";
    std::string s = apply_chat_template({ {"system", worker_preamble_text()}, {"user", user} }, true);
    if (!g_worker_think) s += "<think>\n\n</think>\n\n";
    return s;
}

// IMPLEMENT task: one implementer per component → its module, given its blueprint + siblings (read-file→inject)
static std::string build_impl_task(const Piece &p, const std::string &goal) {
    namespace fs = std::filesystem; std::error_code ec;
    std::string own = read_file_str(std::string(g_work_dir) + "/design/" + p.id + ".blueprint");
    std::string sibs, ddir = std::string(g_work_dir) + "/design";
    for (auto it = fs::directory_iterator(ddir, ec); !ec && it != fs::directory_iterator(); it.increment(ec)) {
        if (!it->is_regular_file(ec)) continue;
        std::string n = it->path().filename().string();
        if (n == p.id + ".blueprint") continue;
        sibs += "--- " + n + " ---\n" + read_file_str(it->path().string()) + "\n";
    }
    std::string user =
        "You are the IMPLEMENTER for component '" + p.id + "'.\n\n" + goal +
        "\nYour component's blueprint:\n" + (own.empty() ? ("(design missing) " + p.instruction) : own) +
        "\n\nOther components' interfaces (integrate via these):\n" + (sibs.empty() ? "(none)" : sibs) +
        "\nImplement '" + p.id + "' exactly per its blueprint, integrating through the interfaces above. "
        "Save it with create_file at the file path its blueprint/assignment specifies.";
    std::string s = apply_chat_template({ {"system", worker_preamble_text()}, {"user", user} }, true);
    if (!g_worker_think) s += "<think>\n\n</think>\n\n";
    return s;
}

// RECONCILE: the boss unifies the parallel (possibly-conflicting) blueprints into ONE
// authoritative interface contract that all implementers obey (fixes cross-component drift).
static std::string build_reconcile_user(const std::string &goal, const std::string &blueprints) {
    return "The DESIGNERS independently proposed the component blueprints below. They likely CONFLICT "
           "on shared signatures (constructor shapes, method names/args, who owns input/score, "
           "canvas vs context, module system). Produce ONE AUTHORITATIVE INTERFACE CONTRACT that every "
           "implementer MUST follow.\n\n" + goal + "\nProposed blueprints:\n" + blueprints +
           "\n\nResolve EVERY cross-component conflict decisively. State, per component, the exact "
           "exported name + method signatures (names, params, returns). Pin the GLOBAL decisions all "
           "files share: (1) MODULE SYSTEM = CommonJS (module.exports / require) so `node <file>.test.js` "
           "works; (2) how components are instantiated/wired; (3) who handles input and who owns score; "
           "(4) the canvas/context convention. Output the contract as concise markdown — no preamble, no code.";
}

// strip a leading <think>…</think> from a worker output
static std::string strip_think(const std::string &in) {
    size_t t = in.find("</think>");
    return trim(t != std::string::npos ? in.substr(t + 8) : in);
}

// The PA.6 staged pipeline (non-stream, --tools): triage → design → implement → test → verify.
static void run_pipeline_staged(const std::string &task, int n_lanes, int max_new) {
    G.tmpl = llama_model_chat_template(G.model, nullptr);

    // ── TRIAGE (light/fast): boss emits the component map; the design pool does the thinking ──
    std::string tprompt = apply_chat_template({ {"system", TRIAGE_PROMPT}, {"user", task} });
    tprompt += "<think>\n\n</think>\n\n";   // light triage — no deep thinking here
    fprintf(stderr, "\n══ PA.6 TRIAGE — boss mapping components (light) ══\n");
    std::string plan = strip_think(boss_generate(tprompt, 2048));
    fprintf(stderr, "\n── triage parsed ──\n");
    WorkOrder wo = parse_work_order(plan);
    print_work_order(wo);
    if (!wo.ok || wo.pieces.empty()) { fprintf(stderr, "PA.6 triage parse failed: %s\n", wo.error.c_str()); return; }

    // The harness owns test generation (the test-gen stage). Drop any test pieces the triage emitted
    // anyway, so implementers never write *.test.js (which would duplicate/clash with test-gen).
    {
        size_t before = wo.pieces.size();
        auto is_test = [](const Piece &p) {
            std::string id = p.id; for (auto &c : id) c = (char)tolower((unsigned char)c);
            if (id.size() >= 5 && id.substr(id.size()-5) == ".test") return true;
            std::string in = p.instruction; for (auto &c : in) c = (char)tolower((unsigned char)c);
            return in.find(".test.js") != std::string::npos;
        };
        wo.pieces.erase(std::remove_if(wo.pieces.begin(), wo.pieces.end(), is_test), wo.pieces.end());
        if (wo.pieces.size() != before)
            fprintf(stderr, "  (dropped %zu test piece(s) — harness generates tests after build)\n", before - wo.pieces.size());
        if (wo.pieces.empty()) { fprintf(stderr, "PA.6 triage produced only test pieces — nothing to build\n"); return; }
    }
    std::string goal = build_goal_blueprint(wo);

    // ── DESIGN pool (parallel, THINKING) → design/<id>.blueprint ──
    bool sv_wt = g_worker_think; SParams sv_sp = g_worker_sp;
    g_worker_think = true; g_worker_sp = qwen_params(true, false);   // designers reason (coding 0.6)
    std::vector<std::string> ditems, dids;
    for (auto &p : wo.pieces) { ditems.push_back(build_design_task(p, goal)); dids.push_back(p.id); }
    fprintf(stderr, "\n══ PA.6 DESIGN — %d designers (parallel, thinking) ══\n", (int)ditems.size());
    std::vector<std::string> douts; run_pool(ditems, n_lanes, max_new, &douts, "");
    std::vector<std::pair<std::string,std::string>> dres;
    for (size_t i = 0; i < dids.size() && i < douts.size(); i++) dres.push_back({ dids[i], strip_think(douts[i]) });
    fprintf(stderr, "\n══ PA.6 STORE — design blueprints ══\n");
    run_worker_tools(dres);
    g_worker_think = sv_wt; g_worker_sp = sv_sp;   // restore (implementers = instruct)

    // ── RECONCILE (boss, serial but bounded): unify the parallel blueprints into ONE
    //    authoritative interface contract → design/INTERFACE.md, appended to the goal so every
    //    implementer obeys the SAME signatures (fixes the cross-component drift). ──
    {
        namespace fs = std::filesystem; std::error_code ec;
        std::string bps, ddir = std::string(g_work_dir) + "/design";
        for (auto it = fs::directory_iterator(ddir, ec); !ec && it != fs::directory_iterator(); it.increment(ec)) {
            if (!it->is_regular_file(ec)) continue;
            std::string n = it->path().filename().string();
            if (n == "INTERFACE.md") continue;
            bps += "=== " + n + " ===\n" + read_file_str(it->path().string()) + "\n";
        }
        if (!bps.empty()) {
            std::string rprompt = apply_chat_template({ {"system", boss_system_text()}, {"user", build_reconcile_user(goal, bps)} });
            fprintf(stderr, "\n══ PA.6 RECONCILE — boss unifying the interface contract ══\n");
            std::string contract = strip_think(boss_generate(rprompt, 3072));   // boss thinks (resolve conflicts), bounded
            FILE *cf = fopen((ddir + "/INTERFACE.md").c_str(), "w");
            if (cf) { fputs(contract.c_str(), cf); fclose(cf);
                fprintf(stderr, "\n── wrote design/INTERFACE.md (%zu chars) ──\n", contract.size()); }
            goal += "\n\n=== AUTHORITATIVE INTERFACE CONTRACT (obey EXACTLY; overrides individual blueprints) ===\n"
                  + contract + "\n";
        }
    }

    // ── IMPLEMENT pool (parallel, instruct) → modules, reading the contract + blueprints ──
    std::vector<std::string> iitems, iids;
    for (auto &p : wo.pieces) { iitems.push_back(build_impl_task(p, goal)); iids.push_back(p.id); }
    fprintf(stderr, "\n══ PA.6 IMPLEMENT — %d implementers (parallel) ══\n", (int)iitems.size());
    std::vector<std::string> iouts; run_pool(iitems, n_lanes, max_new, &iouts, "");
    std::vector<std::pair<std::string,std::string>> ires;
    for (size_t i = 0; i < iids.size() && i < iouts.size(); i++) ires.push_back({ iids[i], strip_think(iouts[i]) });

    // ── TEST-GEN + VERIFY (PA.4b): store modules → a test per module → run all tests ──
    finalize_verify(wo, ires, n_lanes, max_new);
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

// GPU-free self-test for the MTP bookkeeping (helpers defined above run_pool).
static int mtp_self_test() {
    fprintf(stderr, "── PA.3 MTP self-test ──\n");
    int fail = 0;
    auto chk = [&](const char *name, bool ok){
        fprintf(stderr, "  %s: %s\n", name, ok ? "PASS" : "FAIL"); if (!ok) fail++; };
    { MtpVerdict v = mtp_verify(true,  7, 7, false, 9, false); chk("M1 full accept", v.e==2 && v.acc==1 && v.full && !v.stop); }
    { MtpVerdict v = mtp_verify(true,  7, 5, false, 0, false); chk("M2 miss",        v.e==1 && v.acc==0 && !v.full && !v.stop); }
    { MtpVerdict v = mtp_verify(false,-1, 5, false, 0, false); chk("M3 no draft",    v.e==1 && v.acc==0 && !v.full); }
    { MtpVerdict v = mtp_verify(true,  7, 7, true,  9, false); chk("M4 a0 eog",      v.e==1 && v.stop); }
    { MtpVerdict v = mtp_verify(true,  7, 7, false, 9, true ); chk("M5 a1 eog",      v.e==2 && v.full && v.stop); }
    chk("M6 ckpt seq", mtp_ckpt_seq(0,4)==4 && mtp_ckpt_seq(3,4)==7);
    chk("M7 base seq", mtp_base_seq(4)==8);
    chk("M8 seqmax",   mtp_seqmax(4,true)==9 && mtp_seqmax(4,false)==5);
    fprintf(stderr, "  %s (%d/8 passed)\n", fail==0 ? "ALL PASS" : "SOME FAILED", 8 - fail);
    return fail > 0 ? 4 : 0;
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
    // per-lane Qwen samplers; boss lane uses boss params, workers use worker params
    std::vector<llama_sampler *> smpl(nW + 1, nullptr);
    if (!g_greedy) for (int i = 0; i <= nW; i++)
        smpl[i] = make_sampler(g_seed + (uint32_t)i, i == BOSS ? g_boss_sp : g_worker_sp);

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
    std::string bprompt = apply_chat_template({ {"system", boss_system_text()}, {"user", task} });
    if (!g_boss_think) bprompt += "<think>\n\n</think>\n\n";   // boss plan (streaming) — boss role
    {
        std::vector<llama_token> bt(bprompt.size() + 16);
        int bn = llama_tokenize(G.vocab, bprompt.c_str(), (int32_t)bprompt.size(), bt.data(), (int32_t)bt.size(), true, true);
        if (bn < 0) { bt.resize(-bn); bn = llama_tokenize(G.vocab, bprompt.c_str(), (int32_t)bprompt.size(), bt.data(), (int32_t)bt.size(), true, true); }
        bt.resize(bn > 0 ? bn : 0);
        llama_memory_seq_rm(G.mem, BOSS, 0, -1);
        Stream tmp; tmp.prompt_toks = bt; int li = 0;
        if (!prefill_stream(tmp, BOSS, batch, &li)) { fprintf(stderr, "boss prefill failed\n"); llama_batch_free(batch); return; }
        lanes[BOSS].tok = pick(smpl[BOSS], li);
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
        if (!g_worker_think) full += "<think>\n\n</think>\n\n";   // streaming worker — no reasoning
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
        if (smpl[L]) llama_sampler_reset(smpl[L]);            // fresh sampler for the new piece
        lanes[L].tok = pick(smpl[L], last_idx);
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
            llama_token nxt = pick(smpl[L], (int)i);            // Qwen sampling per lane (boss + workers)
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
    for (auto *s : smpl) if (s) llama_sampler_free(s);
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
    bool  mtp_test = false;
    bool  coord_test = false;
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
        else if (!strcmp(argv[i], "--no-think")) { g_boss_think = false; g_worker_think = false; }  // all instruct
        else if (!strcmp(argv[i], "--all-think")) g_worker_think = true;   // workers reason too (slower)
        else if (!strcmp(argv[i], "--no-stream")) no_stream = true;
        else if (!strcmp(argv[i], "--text"))     show_text = true;
        else if (!strcmp(argv[i], "--out")     && i+1 < argc) strncpy(g_out_path, argv[++i], sizeof(g_out_path)-1);
        else if (!strcmp(argv[i], "--no-gather")) g_no_gather = true;
        else if (!strcmp(argv[i], "--gather-test")) gather_test = true;
        else if (!strcmp(argv[i], "--tools")) g_tools = true;
        else if (!strcmp(argv[i], "--allow-run")) { g_allow_run = true; g_tools = true; }  // run implies tools
        else if (!strcmp(argv[i], "--work-dir") && i+1 < argc) strncpy(g_work_dir, argv[++i], sizeof(g_work_dir)-1);
        else if (!strcmp(argv[i], "--mtp"))      g_mtp = true;            // PA.3 per-lane MTP drafting
        else if (!strcmp(argv[i], "--mtp-test")) mtp_test = true;
        else if (!strcmp(argv[i], "--coord-test")) coord_test = true;
        else if ((!strcmp(argv[i], "-t") || !strcmp(argv[i], "--temp")) && i+1 < argc) g_temp = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--general")) g_general = true;         // thinking-general temps (1.0)
        else if (!strcmp(argv[i], "--greedy")) g_greedy = true;           // diagnostic: force greedy (no sampling)
        else if (!strcmp(argv[i], "--repair-budget") && i+1 < argc) g_repair_budget = atoi(argv[++i]);  // PA.4c
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) g_seed = (uint32_t)strtoul(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--verbose"))  g_verbose_logs = true;
        else { fprintf(stderr, "Usage: %s -m <model> [-p \"task\"] [-s streams(1-%d)] [-n max] [-c ctx] [--text] [--parse-test] [--pool M] [--plan-only] [--kv-q8|--kv-f16] [--no-think] [--all-think] [--no-stream] [--out FILE] [--no-gather] [--tools] [--allow-run] [--work-dir DIR] [--mtp] [-t temp] [--general] [--seed N] [--repair-budget N] [--verbose]\n", argv[0], MAX_STREAMS); return 1; }
    }
    if (parse_test)  return parse_self_test();   // PA.1a: GPU-free envelope parser check
    if (gather_test) return gather_self_test();  // PA.1c: GPU-free gather self-test
    if (mtp_test)    return mtp_self_test();     // PA.3: GPU-free MTP bookkeeping self-test
    if (coord_test)  return coord_self_test();   // PA.4c: GPU-free repair-loop bookkeeping self-test
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

    // ── Qwen3.6 recommended sampling (model card), keyed by mode; -t overrides temp ──
    // Qwen: NEVER greedy in thinking mode (repetition/degradation). --mtp is the lone greedy path.
    if (g_mtp || g_greedy) {
        g_greedy = true;
        fprintf(stderr, "[sampling] greedy (%s) — diagnostic/reference, not the product\n",
                g_mtp ? "--mtp speculative" : "--greedy");
    } else {
        g_boss_sp   = qwen_params(g_boss_think,   g_general);   // boss: think (coding/general) or instruct
        g_worker_sp = qwen_params(g_worker_think, false);       // workers: think/coding or instruct
        if (g_temp >= 0) { g_boss_sp.temp = g_temp; g_worker_sp.temp = g_temp; }   // -t overrides both
        fprintf(stderr, "[sampling] boss %s temp %.2f top_p %.2f presence %.1f | workers %s temp %.2f top_p %.2f presence %.1f (seed %u)\n",
                g_boss_think ? (g_general ? "think/general" : "think/coding") : "instruct",
                g_boss_sp.temp, g_boss_sp.top_p, g_boss_sp.presence,
                g_worker_think ? "think/coding" : "instruct",
                g_worker_sp.temp, g_worker_sp.top_p, g_worker_sp.presence, g_seed);
    }

    if (g_mtp && !no_stream) {    // PA.3 MTP v1 lives in the pool path; streaming MTP is a follow-up
        fprintf(stderr, "[note] --mtp is pool-path only (v1); forcing --no-stream\n");
        no_stream = true;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)n_ctx;
    cp.n_batch = PREFILL_CHUNK;
    // --mtp reserves a checkpoint seq per lane (2N) + base (mtp_seqmax); else N + base.
    cp.n_seq_max = (uint32_t)(g_mtp ? mtp_seqmax(n_streams, true)
                                    : n_streams + (task[0] ? 1 : 0));
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.kv_unified      = false;   // M7.2 exactness: unified KV flips fp near-ties by batch shape
    if (kv_q8) {                  // ~2x context; trades byte-determinism (M7.2: Q8 KV is batch-shape-dependent under FA)
        cp.type_k = GGML_TYPE_Q8_0;
        cp.type_v = GGML_TYPE_Q8_0;
    }
    G.ctx = llama_init_from_model(G.model, cp);
    if (!G.ctx) { fprintf(stderr, "context failed\n"); return 1; }
    G.mem = llama_get_memory(G.ctx);
    if (g_mtp) {                  // PA.3: spin up the nextn-head draft context (falls back if absent)
        G.n_embd = llama_model_n_embd(G.model);
        if (!setup_mtp(n_streams)) g_mtp = false;
    }

    // ── PA.1 pipeline (PA.1a = PLAN phase): boss decomposes the task ──────────
    if (task[0]) {
        if (!no_stream) {                 // PA.3: streaming decomposition — default for -p
            run_pipeline(task, max_new, n_streams);
            llama_free(G.ctx); llama_model_free(G.model); llama_backend_free();
            return 0;
        }
        if (g_tools) {                    // PA.6: staged design→build pipeline (triage → design → implement → test → verify)
            run_pipeline_staged(task, n_streams, max_new);
            if (G.ctx_mtp) llama_free(G.ctx_mtp);
            llama_free(G.ctx); llama_model_free(G.model); llama_backend_free();
            return 0;
        }
        G.tmpl = llama_model_chat_template(G.model, nullptr);
        std::vector<Msg> msgs = {{"system", boss_system_text()}, {"user", task}};
        std::string prompt = apply_chat_template(msgs);
        if (!g_boss_think) prompt += "<think>\n\n</think>\n\n";   // boss plan (non-stream) — boss role
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
            if (g_tools) finalize_verify(wo, worker_results, n_streams, max_new);  // PA.4: store modules → gen+run tests → verify
            else         finish_gather(wo, worker_results, boss_self_output, n_streams);  // legacy single-blob merge
        } else {
            for (size_t i = 0; i < wo.pieces.size() && i < outputs.size(); i++) {
                std::string body = outputs[i];
                size_t bth = body.find("</think>");
                if (bth != std::string::npos) body = body.substr(bth + 8);
                printf("\n───── item %s ─────\n%s\n", wo.pieces[i].id.c_str(), trim(body).c_str());
            }
        }
        if (G.ctx_mtp) llama_free(G.ctx_mtp);
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
