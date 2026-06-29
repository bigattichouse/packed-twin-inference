/*
 * pti_common.cpp — globals, utilities, log callback, batch helpers
 *
 * PA.0: packed-agents plumbing demo
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

#include "pti_agents.h"

#include <ctime>

// quiet llama/ggml logging: only WARN+ passes (--verbose restores)
bool     g_verbose_logs = false;
char     g_out_path[512] = {};     // --out: write final artifact to this file
bool     g_no_gather    = false;   // --no-gather: skip gather phase, print pieces separately
bool     g_tools        = false;   // --tools: let workers emit <function=...> tool calls (write_file/run)
bool     g_allow_run    = false;   // --allow-run: permit the `run` verb (sandboxed shell exec); implies --tools
char     g_work_dir[512] = "pti_work";  // --work-dir: sandbox dir for write_file / run
bool     g_mtp          = false;   // --mtp: per-lane MTP drafting (PA.3) — doubles n_seq_max
// Qwen3.6 recommended sampling (model card), PER ROLE. Default: boss reasons (thinking), workers
// implement a clear spec without reasoning (faster). NEVER greedy in thinking mode (Qwen:
// greedy → repetition/degradation). --mtp is the one greedy path. Resolved in main().
SParams  g_boss_sp     = {0.6f, 0.95f, 0.0f, 0.0f, 20};   // boss: thinking/coding
SParams  g_worker_sp   = {0.7f, 0.80f, 0.0f, 1.5f, 20};   // workers: instruct (no-think)
bool     g_boss_think   = true;    // boss plans/gathers with reasoning
bool     g_worker_think = false;   // workers implement from the spec without reasoning
float    g_temp     = -1.0f;   // -t: override temperature for both roles
uint32_t g_seed     = 42;      // base seed; lane L uses g_seed + L (per-stream streams)
bool     g_general  = false;   // --general: boss uses thinking-general temps (1.0) vs coding (0.6)
bool     g_greedy   = false;   // resolved: true under --mtp (greedy speculative decode)
int      g_repair_budget = 2;  // --repair-budget: max verify→repair rounds (PA.4c)
Globals  G;

static enum ggml_log_level g_last_lvl = GGML_LOG_LEVEL_NONE;
void pti_log_cb(enum ggml_log_level level, const char *text, void *) {
    if (g_verbose_logs) { fputs(text, stderr); return; }
    if (level == GGML_LOG_LEVEL_CONT) {
        if (g_last_lvl >= GGML_LOG_LEVEL_WARN && g_last_lvl != GGML_LOG_LEVEL_CONT)
            fputs(text, stderr);
        return;
    }
    g_last_lvl = level;
    if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// deterministic tie-break argmax (same rule as the other pti binaries)
static constexpr float ARGMAX_EPS = 0.05f;
int32_t argmax_f(const float *v, int32_t n) {
    int32_t best = 0;
    for (int32_t i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    float cut = v[best] - ARGMAX_EPS;
    for (int32_t i = 0; i < best; i++)
        if (v[i] >= cut) return i;
    return best;
}

void batch_add(struct llama_batch *b, llama_token tok,
               llama_pos pos, llama_seq_id seq, bool want_logits) {
    b->token   [b->n_tokens] = tok;
    b->pos     [b->n_tokens] = pos;
    b->n_seq_id[b->n_tokens] = 1;
    b->seq_id  [b->n_tokens][0] = seq;
    b->logits  [b->n_tokens] = want_logits ? 1 : 0;
    b->n_tokens++;
}

void batch_clear(struct llama_batch *b) { b->n_tokens = 0; }

std::string tok_str(llama_token t) {
    char buf[256];
    int len = llama_token_to_piece(G.vocab, t, buf, (int32_t)sizeof(buf), 0, true);
    return len > 0 ? std::string(buf, len) : std::string{};
}
