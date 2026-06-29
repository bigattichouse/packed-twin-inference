/*
 * pti_pipeline.cpp — PA.3 streaming pipeline + PA.6 staged design→build
 *
 * Contains: run_pipeline() (streaming: boss plans mid-stream while workers
 * execute), run_pipeline_staged() (PA.6: triage → design → implement →
 * test → verify), and all the pipeline-stage prompt builders.
 *
 * Dependencies: pti_common.cpp, pti_stream.cpp, pti_pool.cpp,
 *               pti_parser.cpp, pti_prompt.cpp, pti_tools.cpp,
 *               pti_memory.cpp, pti_eager.cpp, pti_verify.cpp
 * Exposes:  run_pipeline(), run_pipeline_staged()
 */

#include "pti_agents.h"

// ─── helpers (file-local) ────────────────────────────────────────────────────
// ─── PA.6 prompts ────────────────────────────────────────────────────────────
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

static std::string build_goal_blueprint(const WorkOrder &wo) {
    std::string g = "GOAL BLUEPRINT — the project's components and how they connect:\n" + wo.shared + "\nComponents:\n";
    for (auto &p : wo.pieces) {
        g += "  - " + p.id + " — " + p.instruction + "  [exports: ";
        for (size_t i = 0; i < p.exports.size(); i++) g += (i ? "," : "") + p.exports[i];
        g += "]\n";
    }
    return g;
}

static std::string design_user(const Piece &p) {
    std::string exp; for (size_t i = 0; i < p.exports.size(); i++) exp += (i ? "," : "") + p.exports[i];
    return "You are the DESIGNER for component '" + p.id + "' (the project goal + components are above).\n"
           "Your component '" + p.id + "': " + p.instruction + "\nExports: " + exp + "\n\n"
           "Write a concise BLUEPRINT for '" + p.id + "': its exported API (names, params, types), behavior, "
           "key edge cases, and how it interacts with the other components above. Precise prose/pseudocode, "
           "NOT full code. Save it with create_file at path design/" + p.id + ".blueprint";
}

static std::string impl_shared(const std::string &goal) {
    namespace fs = std::filesystem; std::error_code ec;
    std::string bps, ddir = std::string(g_work_dir) + "/design";
    for (auto it = fs::directory_iterator(ddir, ec); !ec && it != fs::directory_iterator(); it.increment(ec)) {
        if (!it->is_regular_file(ec)) continue;
        std::string n = it->path().filename().string();
        if (n == "INTERFACE.md") continue;
        bps += "--- " + n + " ---\n" + read_file_str(it->path().string()) + "\n";
    }
    return goal + "\n\nALL COMPONENT BLUEPRINTS (integrate via these):\n" + (bps.empty() ? "(none)" : bps);
}

static std::string impl_user(const Piece &p) {
    return "You are the IMPLEMENTER for component '" + p.id + "'. Its blueprint is '" + p.id +
           ".blueprint' in the blueprints above; implement it EXACTLY, integrating through the other "
           "components' interfaces above and obeying the CONTRACT. Save the MODULE with create_file at the "
           "path its blueprint/assignment specifies (e.g. src/" + p.id + ".js). If you also write a test, "
           "put it in the PROPER place: create_file at test/" + p.id + ".test.js (require the module via "
           "require('../src/" + p.id + "')) — NEVER a *.test.js inside src/.";
}

static std::string build_reconcile_user(const std::string &goal, const std::string &blueprints) {
    return "The DESIGNERS independently proposed the component blueprints below. They likely CONFLICT "
           "on shared signatures (constructor shapes, method names/args, who owns input/score, "
           "canvas vs context, module system). Produce ONE AUTHORITATIVE INTERFACE CONTRACT that every "
           "implementer MUST follow.\n\n" + goal + "\nProposed blueprints:\n" + blueprints +
           "\n\nResolve EVERY cross-component conflict decisively. State, per component, the exact "
           "exported name + method signatures (names, params, returns). Pin the GLOBAL decisions all "
           "files share: (1) MODULE SYSTEM = CommonJS (module.exports / require) so `node <file>.test.js` "
           "works; (2) how components are instantiated/wired; (3) who handles input and who owns score; "
           "(4) the canvas/context convention. (5) DEPENDENCY POLICY — **you, the designer, DICTATE the "
           "libraries** if the task did not name them: default to ZERO external packages. Tests run with "
           "plain `node` and use ONLY Node built-ins + `console.assert` — NO jsdom, jest, or any npm "
           "package; collaborators are stubbed directly (e.g. `engine.renderer = { clear(){} }`). Put this "
           "as a TECH DECISIONS section at the TOP of the contract so every implementer and test follows "
           "it. Output the contract as concise markdown — no preamble, no code.";
}

static std::string build_partial_reconcile_user(const std::string &goal, const std::string &group_bps) {
    return "Reconcile this SUBSET of component blueprints into a PARTIAL interface contract: resolve "
           "conflicts AMONG these components (exact exported names + method signatures), and note any "
           "interface they expose that OTHER (not-shown) components will depend on.\n\n" + goal +
           "\nBlueprints in this group:\n" + group_bps +
           "\n\nOutput the partial contract as concise markdown (per-component exact signatures). NO code, "
           "NO tool calls — just the markdown.";
}

static std::string build_merge_user(const std::string &goal, const std::string &partials) {
    return "Merge these PARTIAL interface contracts (each reconciled a subset of components in parallel) "
           "into ONE AUTHORITATIVE INTERFACE CONTRACT. Resolve any CROSS-group conflict decisively and pin "
           "the GLOBAL decisions: (1) MODULE SYSTEM = CommonJS; (2) wiring/instantiation; (3) input/score "
           "ownership; (4) canvas/context convention; (5) DEPENDENCY POLICY — default ZERO external "
           "packages; tests use Node built-ins + console.assert, NO jsdom/jest (stub collaborators). Put a "
           "TECH DECISIONS section at the TOP.\n\n" + goal + "\nPartial contracts:\n" + partials +
           "\n\nOutput the unified contract as concise markdown — no preamble, no code.";
}

// ─── PA.6: staged design→build pipeline ──────────────────────────────────────
void run_pipeline_staged(const std::string &task, int n_lanes, int max_new) {
    G.tmpl = llama_model_chat_template(G.model, nullptr);

    // ── TRIAGE ──
    std::string tprompt = apply_chat_template({ {"system", TRIAGE_PROMPT}, {"user", task} });
    tprompt += "<think>\n\n</think>\n\n";
    fprintf(stderr, "\n══ PA.6 TRIAGE — boss mapping components (light) ══\n");
    std::string plan = strip_think(boss_generate(tprompt, 2048));
    fprintf(stderr, "\n── triage parsed ──\n");
    WorkOrder wo = parse_work_order(plan);
    print_work_order(wo);
    if (!wo.ok || wo.pieces.empty()) { fprintf(stderr, "PA.6 triage parse failed: %s\n", wo.error.c_str()); return; }

    // Drop test pieces
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
            fprintf(stderr, "  (dropped %zu test piece(s) — harness generates tests)\n", before - wo.pieces.size());
        if (wo.pieces.empty()) { fprintf(stderr, "PA.6 triage produced only test pieces\n"); return; }
    }
    std::string goal = build_goal_blueprint(wo);
    { std::string vr = vars_render(); if (!vr.empty()) goal = vr + "\n" + goal; }

    // ── DESIGN pool (parallel, THINKING) ──
    bool sv_wt = g_worker_think; SParams sv_sp = g_worker_sp;
    g_worker_think = true; g_worker_sp = qwen_params(true, false);
    std::vector<std::string> ditems, dids;
    for (auto &p : wo.pieces) { ditems.push_back(stage_item(goal, design_user(p))); dids.push_back(p.id); }
    fprintf(stderr, "\n══ PA.6 DESIGN — %d designers (parallel, thinking) ══\n", (int)ditems.size());
    std::vector<std::string> douts; run_pool(ditems, n_lanes, max_new, &douts, stage_prefix(goal));
    std::vector<std::pair<std::string,std::string>> dres;
    for (size_t i = 0; i < dids.size() && i < douts.size(); i++) dres.push_back({ dids[i], strip_think(douts[i]) });
    fprintf(stderr, "\n══ PA.6 STORE — design blueprints ══\n");
    run_worker_tools(dres);
    g_worker_think = sv_wt; g_worker_sp = sv_sp;

    // ── RECONCILE ──
    {
        namespace fs = std::filesystem; std::error_code ec;
        std::vector<std::pair<std::string,std::string>> bps;
        std::string ddir = std::string(g_work_dir) + "/design";
        for (auto it = fs::directory_iterator(ddir, ec); !ec && it != fs::directory_iterator(); it.increment(ec)) {
            if (!it->is_regular_file(ec)) continue;
            std::string n = it->path().filename().string();
            if (n == "INTERFACE.md") continue;
            bps.push_back({ n, read_file_str(it->path().string()) });
        }
        std::sort(bps.begin(), bps.end());
        if (!bps.empty()) {
            int N = (int)bps.size();
            auto groups = reconcile_groups(N, reconcile_parallel_g(N, n_lanes));
            std::string contract;
            if (groups.size() <= 1) {
                std::string all; for (auto &b : bps) all += "=== " + b.first + " ===\n" + b.second + "\n";
                fprintf(stderr, "\n══ PA.6 RECONCILE — 1 pass (%d blueprints) ══\n", N);
                contract = strip_think(boss_generate(apply_chat_template(
                    { {"system", boss_system_text()}, {"user", build_reconcile_user(goal, all)} }), 3072));
            } else {
                fprintf(stderr, "\n══ PA.6 RECONCILE — %d parallel group passes + merge ══\n", (int)groups.size());
                bool sv2 = g_worker_think; SParams sv2sp = g_worker_sp;
                if (!g_greedy) { g_worker_think = true; g_worker_sp = qwen_params(true, false); }
                std::vector<std::string> gitems;
                for (auto &grp : groups) {
                    std::string gb; for (int i = grp.first; i < grp.first + grp.second; i++)
                        gb += "=== " + bps[i].first + " ===\n" + bps[i].second + "\n";
                    gitems.push_back(apply_chat_template({ {"system", boss_system_text()},
                                                           {"user", build_partial_reconcile_user(goal, gb)} }, true));
                }
                std::vector<std::string> gouts; run_pool(gitems, n_lanes, 2048, &gouts, "");
                g_worker_think = sv2; g_worker_sp = sv2sp;
                std::string partials;
                for (size_t i = 0; i < gouts.size(); i++)
                    partials += "=== PARTIAL " + std::to_string(i+1) + " ===\n" + strip_think(gouts[i]) + "\n\n";
                fprintf(stderr, "── merging %d partial contracts ──\n", (int)gouts.size());
                contract = strip_think(boss_generate(apply_chat_template(
                    { {"system", boss_system_text()}, {"user", build_merge_user(goal, partials)} }), 2560));
            }
            FILE *cf = fopen((ddir + "/INTERFACE.md").c_str(), "w");
            if (cf) { fputs(contract.c_str(), cf); fclose(cf);
                fprintf(stderr, "\n── wrote design/INTERFACE.md (%zu chars) ──\n", contract.size()); }
            goal += "\n\n=== AUTHORITATIVE INTERFACE CONTRACT (obey EXACTLY; overrides individual blueprints) ===\n"
                  + contract + "\n";
        }
    }

    // ── IMPLEMENT pool (parallel, instruct) ──
    std::vector<std::string> iitems, iids;
    std::string ishared = impl_shared(goal);
    for (auto &p : wo.pieces) { iitems.push_back(stage_item(ishared, impl_user(p))); iids.push_back(p.id); }
    fprintf(stderr, "\n══ PA.6 IMPLEMENT — %d implementers (parallel) ══\n", (int)iitems.size());
    std::vector<std::string> iouts; run_pool(iitems, n_lanes, max_new, &iouts, stage_prefix(ishared));
    std::vector<std::pair<std::string,std::string>> ires;
    for (size_t i = 0; i < iids.size() && i < iouts.size(); i++) ires.push_back({ iids[i], strip_think(iouts[i]) });

    // ── TEST-GEN + VERIFY ──
    finalize_verify(wo, ires, n_lanes, max_new);
}

// ─── PA.3: streaming pipeline (boss plans while workers execute) ─────────────
struct QItem { std::string id, instruction, blueprint, language; std::vector<std::string> exports; };
void run_pipeline(const std::string &task, int max_new, int n_lanes) {
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
