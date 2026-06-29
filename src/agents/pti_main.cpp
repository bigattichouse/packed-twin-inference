/*
 * pti_main.cpp — CLI entry point, model loading, pipeline dispatch
 *
 * Contains: main() — argument parsing, model/context initialization,
 *           pipeline selection (streaming vs staged vs pool vs PA.0 demo),
 *           and the built-in prompt pool for pool-mode testing.
 *
 * Dependencies: all other pti_*.cpp modules
 * Exposes:  main()
 */

#include "pti_agents.h"

// Effective context window: set by main()'s arg parsing, read by the pipeline
// helpers below (carved out of main()). File-scope to pti_main.cpp — not a global.
static int n_ctx = 16384;

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

// ─── helpers (file-local) ────────────────────────────────────────────────────
// ─── PA.0: packed agents plumbing demo (sequential vs packed) ─────────────────
static void run_packed_demo(int n_streams, int max_new) {
    fprintf(stderr, "\n══ PA.0 — packed agents plumbing demo (%d streams, -n %d) ══\n\n", n_streams, max_new);

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

    // ── byte-identity gate ───────────────────────────────────────────────────
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
}

// ─── PA.2: work-pool test ────────────────────────────────────────────────────
static void run_pool_test(int pool_items, int n_streams, int max_new) {
    int M = pool_items < MAX_STREAMS ? pool_items : MAX_STREAMS;
    std::vector<std::string> items;
    for (int i = 0; i < M; i++) items.push_back(prompts[i]);
    fprintf(stderr, "\n══ PA.2 work-pool — %d items, %d lanes, cap %d ══\n", M, n_streams, max_new);
    run_pool(items, n_streams, max_new);
}

// ─── PA.1: non-streaming pipeline (plan → pool → gather) ─────────────────────
// Returns the process exit code. Does NOT free G.ctx/model/backend — main() owns
// that cleanup and does it once after this returns (freeing here → double-free crash).
static int run_nonstream_pipeline(const char *task, int max_new, int n_streams, bool plan_only) {
    G.tmpl = llama_model_chat_template(G.model, nullptr);
    std::vector<Msg> msgs = {{"system", boss_system_text()}, {"user", task}};
    std::string prompt = apply_chat_template(msgs);
    if (!g_boss_think) prompt += "<think>\n\n</think>\n\n";
    fprintf(stderr, "\n══ PA.1 PLAN — boss decomposing ══\n  task: %s\n\n", task);
    double t0 = now_sec();
    int plan_cap = n_ctx / (n_streams + 1) - 768;
    if (plan_cap < 1024) plan_cap = 1024;
    std::string raw = boss_generate(prompt, plan_cap);
    double el = now_sec() - t0;
    std::string plan = raw;
    size_t th = plan.find("</think>");
    if (th != std::string::npos) plan = plan.substr(th + 8);
    fprintf(stderr, "\n── boss plan done (%.1fs, %zu chars; streamed above) ──\n── parsed ──\n",
            el, trim(plan).size());
    WorkOrder wo = parse_work_order(plan);
    print_work_order(wo);
    if (!wo.ok) return 2;
    if (plan_only) return 0;

    // ── PA.2: pool ───────────────────────────────────────────────────────────
    std::vector<std::string> items;
    for (auto &p : wo.pieces) items.push_back(build_lane_prompt(wo, p));
    fprintf(stderr, "\n══ PA.2 POOL — %d work items, %d lanes ══\n", (int)items.size(), n_streams);
    std::vector<std::string> outputs;
    run_pool(items, n_streams, max_new, &outputs, build_prefix(wo));

    // ── PA.1c GATHER ─────────────────────────────────────────────────────────
    if (!g_no_gather) {
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
        if (g_tools) finalize_verify(wo, worker_results, n_streams, max_new);
        else         finish_gather(wo, worker_results, boss_self_output, n_streams);
    } else {
        for (size_t i = 0; i < wo.pieces.size() && i < outputs.size(); i++) {
            std::string body = outputs[i];
            size_t bth = body.find("</think>");
            if (bth != std::string::npos) body = body.substr(bth + 8);
            printf("\n───── item %s ─────\n%s\n", wo.pieces[i].id.c_str(), trim(body).c_str());
        }
    }
    return 0;
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
    char  model_path[512] = {};
    char  task[2048] = {};
    int   max_new   = 96;
    n_ctx = 16384;
    int   n_streams = 4;
    bool  show_text = false;
    bool  parse_test = false;
    bool  gather_test = false;
    bool  mtp_test = false;
    bool  coord_test = false;
    bool  eager_test = false;
    int   pool_items = 0;
    bool  plan_only  = false;
    bool  kv_q8      = true;
    bool  no_stream  = false;
    bool  g_lang_explicit = false;

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
        else if (!strcmp(argv[i], "--kv-f16"))   kv_q8 = false;
        else if (!strcmp(argv[i], "--no-think")) { g_boss_think = false; g_worker_think = false; }
        else if (!strcmp(argv[i], "--all-think")) g_worker_think = true;
        else if (!strcmp(argv[i], "--no-stream")) no_stream = true;
        else if (!strcmp(argv[i], "--text"))     show_text = true;
        else if (!strcmp(argv[i], "--out")     && i+1 < argc) strncpy(g_out_path, argv[++i], sizeof(g_out_path)-1);
        else if (!strcmp(argv[i], "--no-gather")) g_no_gather = true;
        else if (!strcmp(argv[i], "--gather-test")) gather_test = true;
        else if (!strcmp(argv[i], "--tools")) g_tools = true;
        else if (!strcmp(argv[i], "--allow-run")) { g_allow_run = true; g_tools = true; }
        else if (!strcmp(argv[i], "--work-dir") && i+1 < argc) strncpy(g_work_dir, argv[++i], sizeof(g_work_dir)-1);
        else if (!strcmp(argv[i], "--mtp"))      g_mtp = true;
        else if (!strcmp(argv[i], "--mtp-test")) mtp_test = true;
        else if (!strcmp(argv[i], "--coord-test")) coord_test = true;
        else if (!strcmp(argv[i], "--eager-test")) eager_test = true;
        else if ((!strcmp(argv[i], "-t") || !strcmp(argv[i], "--temp")) && i+1 < argc) g_temp = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--general")) g_general = true;
        else if (!strcmp(argv[i], "--greedy")) g_greedy = true;
        else if (!strcmp(argv[i], "--repair-budget") && i+1 < argc) g_repair_budget = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lang") && i+1 < argc) { g_stack = stack_profile(argv[++i]); g_lang_explicit = true; }
        else if (!strcmp(argv[i], "--blackboard") && i+1 < argc) strncpy(g_blackboard, argv[++i], sizeof(g_blackboard)-1);
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) g_seed = (uint32_t)strtoul(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--verbose"))  g_verbose_logs = true;
        else { fprintf(stderr, "Usage: %s -m <model> [-p \"task\"] [-s streams(1-%d)] [-n max] [-c ctx] [--text] [--parse-test] [--pool M] [--plan-only] [--kv-q8|--kv-f16] [--no-think] [--all-think] [--no-stream] [--out FILE] [--no-gather] [--tools] [--allow-run] [--work-dir DIR] [--mtp] [-t temp] [--general] [--seed N] [--repair-budget N] [--verbose]\n", argv[0], MAX_STREAMS); return 1; }
    }

    // ── GPU-free self-tests ──────────────────────────────────────────────────
    if (parse_test)  return parse_self_test();
    if (gather_test) return gather_self_test();
    if (mtp_test)    return mtp_self_test();
    if (coord_test)  return coord_self_test();
    if (eager_test)  return eager_self_test();
    if (!model_path[0]) { fprintf(stderr, "Error: -m required\n"); return 1; }

    // ── PA.8: stack resolution + variable store ──────────────────────────────
    if (g_tools) {
        if (!g_lang_explicit) {
            std::string detected = detect_lang_in_dir(g_work_dir);
            std::string stated   = lang_from_task(task);
            std::string chosen   = !stated.empty() ? stated : detected;
            if (!stated.empty() && !detected.empty() && stated != detected)
                fprintf(stderr, "── PA.8 §9.2: task says '%s' but project looks '%s' — honoring the request, '%s' ──\n",
                        stated.c_str(), detected.c_str(), stated.c_str());
            if (!chosen.empty()) { g_stack = stack_profile(chosen);
                fprintf(stderr, "── PA.8 §9.2 resolved stack: %s (%s) ──\n", chosen.c_str(),
                        !stated.empty() ? "stated in task" : "detected from project"); }
        }
        vars_seed_from_stack(); vars_load();
        if (!g_vars.empty()) fprintf(stderr, "── PA.8 vars (%s): %zu loaded ──\n", vars_path().c_str(), g_vars.size());
    }

    // ── defaults ─────────────────────────────────────────────────────────────
    if (n_streams < 1) n_streams = 1;
    if (n_streams > MAX_STREAMS) { fprintf(stderr, "note: clamping -s to MAX_STREAMS=%d\n", MAX_STREAMS); n_streams = MAX_STREAMS; }
    if (task[0] && max_new == 96) max_new = 768;
    if (task[0] && n_ctx == 16384) n_ctx = 131072;
    if (pool_items > 0 && max_new == 96) max_new = 256;
    { int need = (max_new + 320) * n_streams; if (n_ctx < need) n_ctx = need; }

    // ── model loading ────────────────────────────────────────────────────────
    llama_log_set(pti_log_cb, nullptr);
    llama_backend_init();

    fprintf(stderr, "Loading %s ...\n", model_path);
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    G.model = llama_model_load_from_file(model_path, mp);
    if (!G.model) { fprintf(stderr, "load failed\n"); return 1; }
    G.vocab   = llama_model_get_vocab(G.model);
    G.n_vocab = llama_vocab_n_tokens(G.vocab);

    // ── Qwen3.6 recommended sampling ─────────────────────────────────────────
    if (g_mtp || g_greedy) {
        g_greedy = true;
        fprintf(stderr, "[sampling] greedy (%s) — diagnostic/reference, not the product\n",
                g_mtp ? "--mtp speculative" : "--greedy");
    } else {
        g_boss_sp   = qwen_params(g_boss_think,   g_general);
        g_worker_sp = qwen_params(g_worker_think, false);
        if (g_temp >= 0) { g_boss_sp.temp = g_temp; g_worker_sp.temp = g_temp; }
        fprintf(stderr, "[sampling] boss %s temp %.2f top_p %.2f presence %.1f | workers %s temp %.2f top_p %.2f presence %.1f (seed %u)\n",
                g_boss_think ? (g_general ? "think/general" : "think/coding") : "instruct",
                g_boss_sp.temp, g_boss_sp.top_p, g_boss_sp.presence,
                g_worker_think ? "think/coding" : "instruct",
                g_worker_sp.temp, g_worker_sp.top_p, g_worker_sp.presence, g_seed);
    }

    if (g_mtp && !no_stream) {
        fprintf(stderr, "[note] --mtp is pool-path only (v1); forcing --no-stream\n");
        no_stream = true;
    }

    // ── context ───────────────────────────────────────────────────────────────
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)n_ctx;
    cp.n_batch = PREFILL_CHUNK;
    cp.n_seq_max = (uint32_t)(g_mtp ? mtp_seqmax(n_streams, true)
                                    : n_streams + (task[0] ? 1 : 0));
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.kv_unified      = false;
    if (kv_q8) { cp.type_k = GGML_TYPE_Q8_0; cp.type_v = GGML_TYPE_Q8_0; }
    G.ctx = llama_init_from_model(G.model, cp);
    if (!G.ctx) { fprintf(stderr, "context failed\n"); return 1; }
    G.mem = llama_get_memory(G.ctx);
    if (g_mtp) {
        G.n_embd = llama_model_n_embd(G.model);
        if (!setup_mtp(n_streams)) g_mtp = false;
    }

    // ── pipeline dispatch ────────────────────────────────────────────────────
    if (task[0]) {
        if (!no_stream) {
            run_pipeline(task, max_new, n_streams);
            llama_free(G.ctx); llama_model_free(G.model); llama_backend_free();
            return 0;
        }
        if (g_tools) {
            run_pipeline_staged(task, n_streams, max_new);
            if (G.ctx_mtp) llama_free(G.ctx_mtp);
            llama_free(G.ctx); llama_model_free(G.model); llama_backend_free();
            return 0;
        }
        int rc = run_nonstream_pipeline(task, max_new, n_streams, plan_only);
        if (G.ctx_mtp) llama_free(G.ctx_mtp);
        llama_free(G.ctx); llama_model_free(G.model); llama_backend_free();
        return rc;
    }

    if (pool_items > 0) {
        run_pool_test(pool_items, n_streams, max_new);
        llama_free(G.ctx); llama_model_free(G.model); llama_backend_free();
        return 0;
    }

    run_packed_demo(n_streams, max_new);

    if (show_text) {
        for (int s = 0; s < n_streams; s++) {
            printf("\n───── stream %d ─────\n%s\n", s, prompts[s]);
        }
    }

    llama_free(G.ctx);
    llama_model_free(G.model);
    llama_backend_free();
    return 0;
}
