#include "pti_agents.h"

bool setup_mtp(int n_lanes) {
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

// Qwen3.6 model-card params for a role given (thinking?, general?).
SParams qwen_params(bool think, bool general) {
    if (!think)  return { 0.7f, 0.80f, 0.0f, 1.5f, 20 };   // instruct / non-thinking
    if (general) return { 1.0f, 0.95f, 0.0f, 0.0f, 20 };   // thinking — general
    return             { 0.6f, 0.95f, 0.0f, 0.0f, 20 };    // thinking — precise coding
}

// One chain per lane (per-lane seed → independent reproducible streams; penalties
// keep per-lane history). Order: penalties → top_k → top_p → min_p → temp → dist.
llama_sampler *make_sampler(uint32_t seed, const SParams &sp) {
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
llama_token pick(llama_sampler *s, int idx) {
    if (!s) return (llama_token)argmax_f(llama_get_logits_ith(G.ctx, idx), G.n_vocab);
    return llama_sampler_sample(s, G.ctx, idx);
}

// ───────────────────────── PA.2: work-pool ───────────────────────────────────
void run_pool(const std::vector<std::string> &items, int n_lanes, int max_new,
              std::vector<std::string> *out_opt, const std::string &prefix) {
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
