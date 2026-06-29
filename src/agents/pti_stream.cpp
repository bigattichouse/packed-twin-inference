#include "pti_agents.h"

// prefill one stream's prompt into its sequence (chunked); returns last logits idx
bool prefill_stream(Stream &st, llama_seq_id seq, llama_batch &batch, int *last_idx) {
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
int run_streams(std::vector<Stream> &streams, int max_new, bool packed,
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

// qwen_params (the Qwen3.6 model-card role defaults) lives in pti_mtp.cpp.

// Sampler chain: seed per lane → per-stream independence → reproducible packed-vs-solo.
void init_sampler(std::vector<llama_sampler *> &smpl, int n_lanes) {
    for (int L = 0; L < n_lanes; L++) {
        uint32_t lane_seed = g_seed + (uint32_t)L;
        auto *chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
        if (g_greedy || g_mtp) {
            llama_sampler_chain_add(chain, llama_sampler_init_greedy());
        } else {
            bool think = (L == n_lanes - 1) ? g_boss_think : g_worker_think;
            SParams sp = qwen_params(think, g_general);
            if (sp.top_k > 0) llama_sampler_chain_add(chain, llama_sampler_init_top_k(sp.top_k));
            if (sp.top_p > 0) llama_sampler_chain_add(chain, llama_sampler_init_top_p(sp.top_p, 1));
            if (sp.min_p > 0) llama_sampler_chain_add(chain, llama_sampler_init_min_p(sp.min_p, 1));
            llama_sampler_chain_add(chain, llama_sampler_init_temp(sp.temp));
            llama_sampler_chain_add(chain, llama_sampler_init_dist(lane_seed));
        }
        smpl.push_back(chain);
    }
}

void free_sampler(std::vector<llama_sampler *> &smpl) {
    for (auto *s : smpl) if (s) llama_sampler_free(s);
    smpl.clear();
}

// ───────────────────────── PA.3: per-lane MTP drafting ───────────────────────
// Pure bookkeeping (model-free, unit-tested via --mtp-test). See spec/PA3_MTP_DESIGN.md.

// Per-lane sequence layout when --mtp is on (else: lanes 0..n-1, base n).
int mtp_ckpt_seq(int lane, int n_lanes) { return n_lanes + lane; }  // checkpoint of lane L
int  mtp_base_seq(int n_lanes)           { return 2 * n_lanes; }     // prefix-cache seq
int  mtp_seqmax(int n_lanes, bool mtp_on){ return mtp_on ? 2 * n_lanes + 1 : n_lanes + 1; }

// Verdict of verifying one packed MTP step for a lane: how many tokens were emitted
// (e), how many drafts accepted (acc), whether it was a full accept, whether to stop.
MtpVerdict mtp_verify(bool has_draft, llama_token draft,
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
