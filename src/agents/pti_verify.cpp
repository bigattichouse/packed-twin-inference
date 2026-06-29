/*
 * pti_verify.cpp — PA.4: finalize, verify→repair loop, boss arbiter
 *
 * Contains: finalize_verify() (PA.4 files-on-disk verify→repair loop),
 *           gather_self_test(), mtp_self_test().
 *
 * Dependencies: pti_common.cpp, pti_stream.cpp, pti_pool.cpp,
 *               pti_parser.cpp, pti_prompt.cpp, pti_tools.cpp,
 *               pti_memory.cpp, pti_eager.cpp
 * Exposes:  finalize_verify(), gather_self_test(), mtp_self_test()
 */

#include "pti_agents.h"
#include <sys/wait.h>

// ─── helpers (file-local) ────────────────────────────────────────────────────
std::string read_file_str(const std::string &path) {
    std::string s; FILE *f = fopen(path.c_str(), "r");
    if (f) { char b[8192]; size_t k; while ((k = fread(b,1,sizeof(b),f)) > 0) s.append(b,k); fclose(f); }
    return s;
}

std::string strip_think(const std::string &in) {
    size_t t = in.find("</think>");
    return trim(t != std::string::npos ? in.substr(t + 8) : in);
}

std::string stage_prefix(const std::string &shared) {
    return apply_chat_template({ {"system", worker_preamble_text() + std::string("\n\n") + shared} }, false);
}

std::string stage_item(const std::string &shared, const std::string &user) {
    std::string s = apply_chat_template({ {"system", worker_preamble_text() + std::string("\n\n") + shared},
                                          {"user", user} }, true);
    if (!g_worker_think) s += "<think>\n\n</think>\n\n";
    return s;
}

// RepairAction enum
// ─── PA.4: finalize + verify→repair loop ─────────────────────────────────────
void finalize_verify(const WorkOrder &wo,
                     const std::vector<std::pair<std::string,std::string>> &worker_results,
                     int n_lanes, int max_new) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fprintf(stderr, "\n══ PA.4 STORE — writing module files to %s ══\n", g_work_dir);
    run_worker_tools(worker_results);

    // PA.7a: collaborator code
    auto collab_for = [&](const std::string &own_content, const std::string &own_path,
                          const std::vector<std::string> &allmods) {
        std::string lc = own_content; std::transform(lc.begin(), lc.end(), lc.begin(), ::tolower);
        std::string out;
        for (auto &other : allmods) {
            if (other == own_path) continue;
            std::string obase = fs::path(other).filename().string();
            size_t j = obase.rfind(".js"); if (j != std::string::npos) obase = obase.substr(0, j);
            if (obase.empty()) continue;
            std::string lb = obase; std::transform(lb.begin(), lb.end(), lb.begin(), ::tolower);
            if (lc.find(lb) != std::string::npos)
                out += "// " + fs::relative(other, g_work_dir, ec).string() + "\n```js\n" + read_file_str(other) + "\n```\n";
        }
        return out;
    };

    // ── TEST-GEN: one test task per module ──
    std::vector<std::string> mods;
    for (auto it = fs::recursive_directory_iterator(g_work_dir, ec);
         it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) break;
        if (!it->is_regular_file(ec)) continue;
        std::string n = it->path().filename().string();
        if (is_src_file(n)) mods.push_back(it->path().string());
    }
    std::sort(mods.begin(), mods.end());
    std::string contract0 = read_file_str(std::string(g_work_dir) + "/design/INTERFACE.md");
    std::string goal_ctx = contract0.empty() ? wo.shared : contract0;
    { std::string vr = vars_render(); if (!vr.empty()) goal_ctx = vr + "\n" + goal_ctx; }

    auto gen_tests_for = [&](const std::vector<std::string> &targets, const std::vector<std::string> &allmods) {
        if (targets.empty()) return;
        std::vector<std::string> items, ids;
        for (auto &m : targets) {
            std::string rel = fs::relative(m, g_work_dir, ec).string();
            std::string content = read_file_str(m);
            std::string comp = fs::path(rel).filename().string();
            { size_t j = comp.rfind(".js"); if (j != std::string::npos) comp = comp.substr(0, j); }
            std::string bp = read_file_str(std::string(g_work_dir) + "/design/" + comp + ".blueprint");
            items.push_back(stage_item(goal_ctx, test_user(rel, content, bp, collab_for(content, m, allmods))));
            ids.push_back(rel);
        }
        std::vector<std::string> touts;
        run_pool(items, n_lanes, max_new, &touts, stage_prefix(goal_ctx));
        std::vector<std::pair<std::string,std::string>> trs;
        for (size_t i = 0; i < ids.size() && i < touts.size(); i++) trs.push_back({ ids[i], strip_think(touts[i]) });
        run_worker_tools(trs);
    };
    if (!mods.empty()) {
        std::vector<std::string> untested0;
        for (auto &m : mods) {
            std::string comp = fs::path(m).filename().string();
            { size_t j = comp.rfind(".js"); if (j != std::string::npos) comp = comp.substr(0, j); }
            if (!fs::exists(std::string(g_work_dir) + "/test/" + test_file_name(comp), ec)) untested0.push_back(m);
        }
        fprintf(stderr, "\n══ PA.4 TEST-GEN — %d module(s); %d already tested, generating %d ══\n",
                (int)mods.size(), (int)(mods.size() - untested0.size()), (int)untested0.size());
        gen_tests_for(untested0, mods);
    }

    // ── PA4 §4.7: INTEGRATION tests for internal nodes ──
    if (!mods.empty()) {
        std::vector<std::string> bases;
        for (auto &m : mods) bases.push_back(comp_key(m));
        std::vector<std::string> iitems, iids;
        for (auto &m : mods) {
            std::string base = comp_key(m), content = read_file_str(m);
            auto refs = module_refs(content, base, bases);
            if (refs.empty()) continue;
            if (fs::exists(std::string(g_work_dir) + "/test/" + base + ".integration" + g_stack.test_suffix, ec)) continue;
            std::string sib;
            for (auto &rb : refs) for (auto &mm : mods) if (comp_key(mm) == rb) {
                sib += "// src/" + rb + ".js\n```js\n" + read_file_str(mm) + "\n```\n"; break; }
            iitems.push_back(stage_item(goal_ctx, build_integration_task(goal_ctx, base, content, sib)));
            iids.push_back("test/" + base + ".integration" + g_stack.test_suffix);
        }
        if (!iitems.empty()) {
            fprintf(stderr, "\n══ PA4 §4.7 INTEGRATION TEST-GEN — %d internal node(s) ══\n", (int)iitems.size());
            std::vector<std::string> iouts; run_pool(iitems, n_lanes, max_new, &iouts, stage_prefix(goal_ctx));
            std::vector<std::pair<std::string,std::string>> itrs;
            for (size_t i = 0; i < iids.size() && i < iouts.size(); i++) itrs.push_back({ iids[i], strip_think(iouts[i]) });
            run_worker_tools(itrs);
        }
    }

    // ── VERIFY + REPAIR (PA.4c) ──
    struct TestRes { std::string rel; bool ok; std::string out; };
    auto run_all_tests = [&]() {
        std::vector<TestRes> res; std::vector<std::string> tests;
        for (auto it = fs::recursive_directory_iterator(g_work_dir, ec);
             it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (ec) break;
            if (it->is_regular_file(ec)) { std::string n = it->path().filename().string();
                if (is_test_file(n)) tests.push_back(it->path().string()); }
        }
        std::sort(tests.begin(), tests.end());
        for (auto &t : tests) {
            std::string rel = fs::relative(t, g_work_dir, ec).string();
            std::string cmd = "cd '" + std::string(g_work_dir) + "' && timeout 30 " + run_test_cmd(rel) + " 2>&1";
            FILE *pp = popen(cmd.c_str(), "r");
            std::string out; if (pp){ char b[4096]; size_t k; while((k=fread(b,1,sizeof(b),pp))>0) out.append(b,k); }
            int rc = pp ? pclose(pp) : -1; int code = (pp && WIFEXITED(rc)) ? WEXITSTATUS(rc) : -1;
            res.push_back({ rel, code == 0, out });
        }
        return res;
    };
    auto list_modules = [&]() {
        std::vector<std::string> ms;
        for (auto it = fs::recursive_directory_iterator(g_work_dir, ec);
             it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (ec) break;
            if (!it->is_regular_file(ec)) continue;
            std::string n = it->path().filename().string();
            if (is_src_file(n)) ms.push_back(it->path().string());
        }
        return ms;
    };

    bool sv_wt = g_worker_think; SParams sv_sp = g_worker_sp;
    if (!g_greedy) { g_worker_think = true; g_worker_sp = qwen_params(true, false); }

    int arbiter_rounds = 0;
    const int ARBITER_BUDGET = 2;
    std::string arbiter_diag;
    int prev_failed = -1;
    std::map<std::string,std::string> journal;
    auto j_append = [&](const std::string &comp, const std::string &lvl, int rnd, const std::string &out) {
        std::string a = extract_attempt(out);
        journal[comp] += "[round " + std::to_string(rnd) + " " + lvl + "] " + (a.empty() ? "(no note)" : a) + "\n";
    };

    for (int round = 0; ; round++) {
        std::vector<TestRes> res = run_all_tests();
        std::vector<std::string> curmods = list_modules(), untested;
        for (auto &m : curmods) {
            std::string comp = fs::path(m).filename().string();
            { size_t j = comp.rfind(".js"); if (j != std::string::npos) comp = comp.substr(0, j); }
            if (!fs::exists(std::string(g_work_dir) + "/test/" + test_file_name(comp), ec)) untested.push_back(m);
        }
        fprintf(stderr, "\n══ PA.4 VERIFY%s — %d test file(s), %d untested module(s) ══\n",
                round ? " (re-check)" : "", (int)res.size(), (int)untested.size());
        if (res.empty() && untested.empty()) { fprintf(stderr, "  ⚠ NO modules/tests — cannot verify\n"); break; }
        int passed = 0; std::vector<TestRes> fails;
        for (auto &r : res) {
            if (r.ok) { passed++; fprintf(stderr, "  [PASS] %s\n", r.rel.c_str()); }
            else { fails.push_back(r); fprintf(stderr, "  [FAIL] %s\n", r.rel.c_str());
                   std::string o = r.out; if (o.size() > 300) o = o.substr(0,300) + "..."; if (!o.empty()) fprintf(stderr, "      %s\n", o.c_str()); }
        }
        for (auto &m : untested) fprintf(stderr, "  [UNTESTED] %s\n", fs::relative(m, g_work_dir, ec).string().c_str());
        int issues = (int)fails.size() + (int)untested.size();
        RepairAction act = repair_verdict(round, g_repair_budget, issues);
        bool no_progress = (prev_failed >= 0 && issues >= prev_failed && untested.empty());
        if (act == RA_REPAIR && no_progress) {
            fprintf(stderr, "  (L1 made no progress (%d→%d issues) — escalating to the boss early)\n",
                    prev_failed, issues);
            act = RA_GIVEUP;
        }
        if (act == RA_REPAIR) {
            for (auto &r : fails) {
                if (module_for_test(std::filesystem::path(r.rel).filename().string(), curmods).empty()) {
                    fprintf(stderr, "  (test %s maps to no single module → only L2 can attribute it)\n", r.rel.c_str());
                    act = RA_GIVEUP; break;
                }
                auto bad = contradictory_asserts(read_file_str(std::string(g_work_dir) + "/" + r.rel));
                if (!bad.empty()) {
                    fprintf(stderr, "  (contradictory test %s asserts %s to multiple values → arbiter)\n",
                            r.rel.c_str(), bad[0].c_str());
                    act = RA_GIVEUP; break;
                }
            }
        }
        prev_failed = issues;
        if (act == RA_DONE)   { fprintf(stderr, "══ VERIFY RESULT: %d/%d passed — DONE (all green) ══\n", passed, (int)res.size()); break; }
        if (act == RA_GIVEUP) {
            if (arbiter_rounds < ARBITER_BUDGET && !fails.empty()) {
                arbiter_rounds++;
                fprintf(stderr, "\n══ PA.4d ARBITER (escalation %d/%d) — %d failure(s) to the boss ══\n",
                        arbiter_rounds, ARBITER_BUDGET, (int)fails.size());
                std::string contract = read_file_str(std::string(g_work_dir) + "/design/INTERFACE.md");
                std::vector<std::string> mods = list_modules(); std::string fb;
                if (!arbiter_diag.empty())
                    fb += "PRIOR ARBITER REASONING:\n" + arbiter_diag + "\n";
                for (auto &r : fails) {
                    std::string testfn = std::filesystem::path(r.rel).filename().string();
                    std::string modpath = module_for_test(testfn, mods);
                    std::string modrel = modpath.empty() ? std::string("(none)") : fs::relative(modpath, g_work_dir, ec).string();
                    std::string comp = comp_key(testfn);
                    fb += "\n--- test " + r.rel + "  (module " + modrel + ") ---\n";
                    if (!journal[comp].empty()) fb += "PRIOR ATTEMPTS:\n" + journal[comp];
                    if (!modpath.empty()) fb += "MODULE " + modrel + ":\n" + read_file_str(modpath) + "\n";
                    { auto bad = contradictory_asserts(read_file_str(std::string(g_work_dir) + "/" + r.rel));
                      if (!bad.empty()) fb += "⚠ CONTRADICTORY TEST — " + bad[0] + " to MULTIPLE values: the TEST is the bug.\n"; }
                    fb += frame_executed_truth(r.out);
                    fb += "TEST " + r.rel + ":\n" + read_file_str(std::string(g_work_dir) + "/" + r.rel) + "\nERROR:\n" + r.out + "\n";
                }
                std::string aprompt = apply_chat_template({ {"system", boss_system_text()}, {"user", build_arbiter_user(contract, fb)} });
                std::string araw = strip_think(boss_generate(aprompt, 2048));
                arbiter_diag += "── escalation " + std::to_string(arbiter_rounds) + " ──\n" + araw + "\n\n";
                auto reworks = reworks_from_plan(araw);
                fprintf(stderr, "  boss requested %d rework item(s)\n", (int)reworks.size());
                if (!reworks.empty()) {
                    std::vector<std::string> ritems, rids;
                    for (auto &rw : reworks) {
                        std::string tgt = rw.first, comp = comp_key(tgt), rbase = integration_base(tgt);
                        std::string modpath = module_for_test(test_file_name(rbase), mods);
                        std::string spec =
                            (goal_ctx.empty() ? "" : "=== INTERFACE CONTRACT ===\n" + goal_ctx + "\n\n")
                            + "=== blueprint: " + rbase + " ===\n" + read_file_str(std::string(g_work_dir) + "/design/" + rbase + ".blueprint");
                        std::string modc = modpath.empty() ? "" : read_file_str(modpath);
                        std::string testc, err;
                        for (auto &r : fails)
                            if (comp_key(r.rel) == comp) { testc = read_file_str(std::string(g_work_dir) + "/" + r.rel); err = r.out; break; }
                        std::string u = build_rework_user(tgt, spec, modc, testc, frame_executed_truth(err) + err, rw.second,
                                                          collab_for(modc, modpath, mods), journal[comp]);
                        std::string sp = apply_chat_template({ {"system", worker_preamble_text()}, {"user", u} }, true);
                        if (!g_worker_think) sp += "<think>\n\n</think>\n\n";
                        ritems.push_back(sp); rids.push_back(tgt);
                    }
                    std::vector<std::string> routs; run_pool(ritems, n_lanes, max_new, &routs, "");
                    std::vector<std::pair<std::string,std::string>> rres;
                    for (size_t i = 0; i < rids.size() && i < routs.size(); i++) {
                        rres.push_back({ rids[i], strip_think(routs[i]) });
                        std::string c = comp_key(rids[i]);
                        j_append(c, "L2", round, routs[i]);
                    }
                    run_worker_tools(rres);
                    continue;
                }
            }
            fprintf(stderr, "══ VERIFY RESULT: %d/%d passed — GIVEN UP after %d round(s) + %d arbiter escalation(s) ══\n",
                    passed, (int)res.size(), g_repair_budget, arbiter_rounds);
            break;
        }
        // RA_REPAIR: fill missing tests, then amend
        if (!untested.empty()) {
            fprintf(stderr, "\n══ PA.4 TEST-GEN (repair) round %d/%d — %d untested module(s) ══\n",
                    round+1, g_repair_budget, (int)untested.size());
            gen_tests_for(untested, curmods);
            continue;
        }
        fprintf(stderr, "\n══ PA.4c REPAIR round %d/%d — amending %d failing module(s) ══\n", round+1, g_repair_budget, (int)fails.size());
        std::vector<std::string> mods = list_modules(), aitems, aids;
        for (auto &r : fails) {
            std::string testfn = std::filesystem::path(r.rel).filename().string();
            std::string modpath = module_for_test(testfn, mods);
            if (modpath.empty()) { fprintf(stderr, "  (no module maps to %s — skip)\n", testfn.c_str()); continue; }
            std::string base = comp_key(testfn);
            std::string modrel = fs::relative(modpath, g_work_dir, ec).string();
            std::string spec =
                (goal_ctx.empty() ? "" : "=== INTERFACE CONTRACT ===\n" + goal_ctx + "\n\n")
                + "=== blueprint: " + base + " ===\n" + read_file_str(std::string(g_work_dir) + "/design/" + base + ".blueprint");
            std::string modcontent = read_file_str(modpath);
            std::string user = build_amend_user(base, modrel, modcontent,
                                                read_file_str(std::string(g_work_dir) + "/" + r.rel),
                                                frame_executed_truth(r.out) + r.out, spec,
                                                collab_for(modcontent, modpath, mods), journal[base]);
            std::string sp = apply_chat_template({ {"system", worker_preamble_text()}, {"user", user} }, true);
            if (!g_worker_think) sp += "<think>\n\n</think>\n\n";
            aitems.push_back(sp); aids.push_back(modrel);
        }
        if (aitems.empty()) { fprintf(stderr, "  nothing repairable; stopping\n"); break; }
        std::vector<std::string> aouts; run_pool(aitems, n_lanes, max_new, &aouts, "");
        std::vector<std::pair<std::string,std::string>> ares;
        for (size_t i = 0; i < aids.size() && i < aouts.size(); i++) {
            ares.push_back({ aids[i], strip_think(aouts[i]) });
            std::string c = comp_key(aids[i]);
            j_append(c, "L1", round, aouts[i]);
        }
        run_worker_tools(ares);
    }
    g_worker_think = sv_wt; g_worker_sp = sv_sp;
}

// ─── self-tests ──────────────────────────────────────────────────────────────
int gather_self_test() {
    fprintf(stderr, "── PA.1c gather self-test ──\n");
    int fail = 0;
    auto chk = [&](const char *n, bool ok){ fprintf(stderr, "  %s: %s\n", n, ok ? "PASS" : "FAIL"); if (!ok) fail++; };
    { std::string r = extract_code_fence("```js\nfunction foo() { return 1; }\n```");
      chk("U1 fenced js", r == "function foo() { return 1; }"); }
    { std::string r = extract_code_fence("```javascript\nconst x = 42;\n```");
      chk("U2 fenced javascript", r == "const x = 42;"); }
    { std::string r = extract_code_fence("function bar() { return 2; }");
      chk("U3 no fences", r == "function bar() { return 2; }"); }
    { std::string r = extract_code_fence("```js\nfirst()\n```\n```js\nsecond()\n```");
      chk("U4 multiple fences", r == "first()"); }
    { std::string r = extract_code_fence("Here is the code:\n```python\ndef hello(): pass\n```\nDone.");
      chk("U5 fence with prose", r == "def hello(): pass"); }
    { std::string r = extract_code_fence("");
      chk("U6 empty", r.empty()); }
    { std::string r = extract_code_fence("```js\nfunction unclosed() {");
      chk("U7 unclosed fence", r == "function unclosed() {"); }
    { std::vector<std::pair<std::string, std::string>> workers = {
        {"w1", "function gravityStep() {}"}, {"w2", "```js\nfunction spawnPipe() {}\n```"} };
      WorkOrder wo; wo.ok = false;
      Piece p1; p1.id = "w1"; p1.exports = {"gravityStep"};
      Piece p2; p2.id = "w2"; p2.exports = {"spawnPipe"};
      wo.pieces.push_back(p1); wo.pieces.push_back(p2);
      std::string prompt = build_gather_prompt(wo, workers, "");
      bool ok = prompt.find("<<<WORKER_DONE id=w1>>>") != std::string::npos
             && prompt.find("<<<END_WORKER>>>") != std::string::npos
             && prompt.find("<<<GATHER_INSTRUCTION>>>") != std::string::npos
             && prompt.find("<<<END_GATHER>>>") != std::string::npos;
      chk("U8 gather prompt markers", ok); }
    { WorkOrder wo; wo.ok = false;
      std::string prompt = build_gather_prompt(wo, {}, "");
      bool ok = prompt.find("<<<GATHER_INSTRUCTION>>>") != std::string::npos
             && prompt.find("<<<WORKER_DONE") == std::string::npos;
      chk("U9 empty gather prompt", ok); }
    { std::vector<std::pair<std::string, std::string>> workers = {
        {"w1", "function test() { if (a <<< b) return a; }"} };
      WorkOrder wo; wo.ok = false;
      Piece p1; p1.id = "w1"; p1.exports = {"test"};
      wo.pieces.push_back(p1);
      std::string prompt = build_gather_prompt(wo, workers, "");
      chk("U10 worker with markers", prompt.find("<<<WORKER_DONE id=w1>>>") != std::string::npos); }
    // PA.5 tool-call parser
    { auto calls = parse_tool_calls("<create_file>\n<path>src/a.js</path>\n<content>\nconst a=1;\n</content>\n</create_file>");
      chk("T1 parse create_file", calls.size()==1 && calls[0].name=="create_file"
                                  && tc_param(calls[0],"path")=="src/a.js" && tc_param(calls[0],"content")=="const a=1;"); }
    { auto calls = parse_tool_calls("<execute_bash><command>node test.js</command></execute_bash>");
      chk("T2 parse execute_bash", calls.size()==1 && calls[0].name=="execute_bash"
                                  && tc_param(calls[0],"command")=="node test.js"); }
    { chk("T3 ignores unknown tags", parse_tool_calls("render() { return <div>hi</div>; }").empty()); }
    { chk("T4 dangerous-cmd guard", is_dangerous_cmd("rm -rf /") && is_dangerous_cmd("mkfs.ext4 /dev/sda")
                                  && !is_dangerous_cmd("npm test") && !is_dangerous_cmd("rm -rf /tmp/build")); }
    // PA.7b active-retrieval
    { auto calls = parse_tool_calls("<read_file><path>src/renderer.js</path></read_file>");
      chk("T5 parse read_file", calls.size()==1 && calls[0].name=="read_file" && tc_param(calls[0],"path")=="src/renderer.js"); }
    { auto calls = parse_tool_calls("<abandon><reason>waiting on src/renderer.js</reason></abandon>");
      chk("T6 parse abandon", calls.size()==1 && calls[0].name=="abandon" && tc_param(calls[0],"reason")=="waiting on src/renderer.js"); }
    { std::string p = "/tmp/_pa7_read_test.js"; FILE *f = fopen(p.c_str(), "w");
      if (f) { fputs("module.exports={};", f); fclose(f); }
      bool ok = resolve_read("/tmp", "_pa7_read_test.js") == "module.exports={};"
             && pa7_is_notfound(resolve_read("/tmp", "_pa7_nope.js"))
             && pa7_is_notfound(resolve_read("/tmp", "../etc/passwd"));
      remove(p.c_str());
      chk("T7 resolve_read", ok); }
    { std::string q1 = build_blocked_requeue("ORIGTASK", "need renderer.js", "");
      std::string q2 = build_blocked_requeue("ORIGTASK", "need renderer.js", "CTXCODE");
      bool ok = q1.find("ORIGTASK")!=std::string::npos && q1.find("need renderer.js")!=std::string::npos
             && q1.find("CTXCODE")==std::string::npos && q2.find("CTXCODE")!=std::string::npos;
      chk("T8 build_blocked_requeue", ok); }
    fprintf(stderr, "  %s (%d/18 passed)\n", fail == 0 ? "ALL PASS" : "SOME FAILED", 18 - fail);
    return fail > 0 ? 3 : 0;
}

int mtp_self_test() {
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
