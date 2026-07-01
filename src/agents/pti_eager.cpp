/*
 * pti_eager.cpp — PA.7: eager-scheduling core + PA.4 repair helpers
 *
 * Dependency-keyed dataflow replacing PA.6's bulk barriers.  A free lane
 * pulls the highest-priority item whose inputs already exist.  Includes
 * the makespan simulator that proves the dispatcher is work-conserving.
 * Also contains the repair bookkeeping (build_amend_user, repair_verdict,
 * build_arbiter_user, build_rework_user, parse_rework, reworks_from_plan).
 *
 * Dependencies: pti_common.cpp, pti_parser.cpp, pti_memory.cpp
 * Exposes:  build_dag(), eager_ready(), eager_simulate(), staged_simulate(),
 *           build_amend_user(), build_arbiter_user(), build_rework_user(),
 *           parse_rework(), reworks_from_plan(), coord_self_test()
 */

#include "pti_agents.h"

// ─── repair bookkeeping helpers ──────────────────────────────────────────────
std::string extract_attempt(const std::string &out) {
    size_t a = out.find("ATTEMPT:");
    if (a == std::string::npos) return "";
    size_t e = out.find('\n', a);
    return trim(out.substr(a + 8, (e == std::string::npos ? out.size() : e) - (a + 8)));
}

RepairAction repair_verdict(int round, int budget, int n_failed) {
    if (n_failed == 0) return RA_DONE;
    if (round >= budget) return RA_GIVEUP;
    return RA_REPAIR;
}

// PA4 §4.6 self-contradiction lint helpers
static size_t find_top_eqeqeq(const std::string &s) {
    int depth = 0; bool inq = false; char q = 0;
    for (size_t i = 0; i + 2 < s.size(); i++) { char c = s[i];
        if (inq) { if (c == '\\') i++; else if (c == q) inq = false; continue; }
        if (c == '\'' || c == '"' || c == '`') { inq = true; q = c; }
        else if (c == '(' || c == '[' || c == '{') depth++;
        else if (c == ')' || c == ']' || c == '}') depth--;
        else if (depth == 0 && c == '=' && s[i+1] == '=' && s[i+2] == '=') return i;
    }
    return std::string::npos;
}

std::vector<std::string> contradictory_asserts(const std::string &src) {
    std::map<std::string, std::vector<std::string>> seen;
    auto nows = [](const std::string &s){ std::string o; for (char c : s) if (!isspace((unsigned char)c)) o += c; return o; };
    for (size_t pos = 0; ; ) {
        pos = src.find("assert", pos);
        if (pos == std::string::npos) break;
        size_t after = pos + 6; char nx = after < src.size() ? src[after] : 0;
        if (nx != '(' && nx != '.') { pos = after; continue; }
        size_t op = src.find('(', after); if (op == std::string::npos) break;
        size_t cl = match_paren(src, op);
        pos = (cl == std::string::npos) ? op + 1 : cl + 1;
        if (cl == std::string::npos) continue;
        std::string inside = src.substr(op + 1, cl - op - 1), call, exp;
        size_t eq = find_top_eqeqeq(inside);
        if (eq != std::string::npos) { call = inside.substr(0, eq);
            auto a = split_top_commas(inside.substr(eq + 3)); exp = a.empty() ? "" : a[0]; }
        else { auto a = split_top_commas(inside); if (a.size() < 2) continue; call = a[0]; exp = a[1]; }
        call = nows(call); exp = nows(exp);
        if (call.empty() || exp.empty()) continue;
        auto &v = seen[call]; if (std::find(v.begin(), v.end(), exp) == v.end()) v.push_back(exp);
    }
    std::vector<std::string> out;
    for (auto &kv : seen) if (kv.second.size() > 1) out.push_back(kv.first);
    return out;
}

std::string frame_executed_truth(const std::string &err) {
    std::string a, e; std::smatch m;
    static const std::regex rne(R"(([^\n]{1,80}?)\s*!==\s*([^\n]{1,80}?)\s*(?:\n|$))");
    static const std::regex ract(R"([Aa]ctual[^\n:]*:\s*([^\n]{1,80}))");
    static const std::regex rexp(R"([Ee]xpected[^\n:]*:\s*([^\n]{1,80}))");
    if (std::regex_search(err, m, rne)) { a = m[1].str(); e = m[2].str(); }
    else { std::smatch ma, me; if (std::regex_search(err, ma, ract) && std::regex_search(err, me, rexp)) { a = ma[1].str(); e = me[1].str(); } }
    auto t = [](std::string s){ size_t b = s.find_first_not_of(" \t+-"); if (b == std::string::npos) return std::string();
                               size_t z = s.find_last_not_of(" \t,"); return s.substr(b, z - b + 1); };
    a = t(a); e = t(e);
    if (a.empty() || e.empty()) return "";
    return "EXECUTED FACT (trust the run over any re-derivation): the code actually produced `" + a +
           "`, the test asserted `" + e + "`. Decide which the SPEC endorses — if the run matches the spec, "
           "the TEST is wrong; otherwise the module.\n";
}

// PA4 §4.7 integration-test layer
std::vector<std::string> module_refs(const std::string &content, const std::string &self_base,
                                            const std::vector<std::string> &all_bases) {
    auto idchar = [](char c){ return isalnum((unsigned char)c) || c == '_'; };
    std::vector<std::string> out;
    for (auto &b : all_bases) {
        if (b.empty() || b == self_base) continue;
        for (size_t p = 0; (p = content.find(b, p)) != std::string::npos; p += b.size()) {
            char bef = p ? content[p-1] : ' ', aft = p + b.size() < content.size() ? content[p + b.size()] : ' ';
            if (!idchar(bef) && !idchar(aft)) { out.push_back(b); break; }
        }
    }
    return out;
}

std::string build_integration_task(const std::string &goal, const std::string &target,
                                          const std::string &target_code, const std::string &sibling_code) {
    std::string ipath = "test/" + target + ".integration" + g_stack.test_suffix;
    return "Write an INTEGRATION test for `" + target + "` that composes it with its REAL "
           "collaborators — import the ACTUAL sibling modules below, do NOT mock them. Stub ONLY the "
           "external boundary (DOM/canvas/timers/RNG/network/fs/DB), nothing internal. Assert that the wiring "
           "the contract specifies works end-to-end (the methods each calls on the other exist and behave). "
           "Write it " + test_directive() + ".\n\n"
           "Project goal/contract:\n" + goal + "\n\nTarget (src/" + target + g_stack.src_ext + "):\n```\n" + target_code +
           "\n```\n\nREAL collaborators (import these, do NOT mock):\n" + sibling_code +
           "\n\nWrite ONLY the test via create_file at " + ipath + ". No prose.";
}

std::string integration_base(const std::string &path) {
    std::string c = comp_key(path);
    size_t i = c.rfind(".integration"); if (i != std::string::npos) c = c.substr(0, i);
    return c;
}

// Amend instruction (user turn) for a failing module
std::string build_amend_user(const std::string &base, const std::string &modpath,
                             const std::string &modcontent, const std::string &testcontent,
                             const std::string &error, const std::string &spec,
                             const std::string &collaborators, const std::string &journal) {
    return "Component '" + base + "' FAILED its test. Fix the MODULE so the test passes (do not change "
           "the test).\n\n" +
           (journal.empty() ? "" : "PRIOR REPAIR ATTEMPTS on this component (read FIRST — do NOT repeat "
                                   "these; if they all failed, the bug is elsewhere):\n" + journal + "\n\n") +
           "Spec / blueprint for '" + base + "':\n" + (spec.empty() ? "(none)" : spec) +
           "\n\nModule (" + modpath + "):\n```\n" + modcontent + "\n```\n\nTest:\n```\n" + testcontent +
           "\n```\n\nTest output / error:\n" + error +
           (collaborators.empty() ? "" : "\n\nCollaborator modules it interacts with:\n" + collaborators) +
           "\n\nFirst write ONE line `ATTEMPT: <what you think is wrong + what you changed>`, then re-emit "
           "the corrected module via create_file at " + modpath + ".";
}

// PA.4d: parse the boss arbiter's rework requests
std::vector<std::pair<std::string,std::string>> parse_rework(const std::string &text) {
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

std::vector<std::pair<std::string,std::string>> reworks_from_plan(const std::string &raw) {
    std::vector<std::pair<std::string,std::string>> out;
    WorkOrder wo = parse_work_order(raw);
    static const std::regex pathre(R"([A-Za-z0-9_./-]+\.(?:js|mjs|cjs|ts|html|css|json|blueprint))");
    auto add = [&](const std::string &f, const std::string &g) {
        if (f.empty() || f.find("..") != std::string::npos) return;
        for (auto &e : out) if (e.first == f) return;
        out.push_back({ f, g });
    };
    for (auto &p : wo.pieces) {
        std::string guide = p.instruction;
        if (!p.blueprint.empty()) guide += (guide.empty() ? "" : "\n") + p.blueprint;
        for (auto &e : p.exports)
            if (std::regex_match(e, pathre)) add(e, guide);
        for (std::sregex_iterator it(p.instruction.begin(), p.instruction.end(), pathre), end; it != end; ++it)
            add(it->str(), guide);
    }
    return out;
}

// Dedicated REPAIR-ARBITER system prompt (PA.4d). The arbiter must emit a rework WORK-ORDER
// envelope, NOT restate the contract — feeding it the generic triage prompt (boss_system_text)
// biases the model to re-emit the shared interface and forget the fix-pieces (→ 0 reworks → give up).
std::string build_boss_arbiter_prompt() {
    return
        "You are the REPAIR ARBITER of a coding team. Failing tests that worker-level repair could "
        "not fix are escalated to you — each with its EXECUTED error, the test, the module, and the "
        "frozen interface contract.\n\n"
        "Do ALL of your analysis inside <think>...</think>. For EACH failure, judge which FILE is "
        "wrong: the MODULE or the TEST. Tests are frequently the bug — calling a getter as a method "
        "(`x.isExpired()` when the contract declares `get isExpired()`); asserting a value the spec "
        "does not require; or being FLAKY — depending on wall-clock timing / `Date.now()` resolution "
        "/ sleeps so two operations collide on the same millisecond. Trust the EXECUTED error and "
        "the contract over any re-derivation.\n\n"
        "REWORK AND ADAPT — make progress WITHOUT thrashing: while ANY test fails you MUST delegate "
        "at least ONE fix, but emit the MINIMAL, most-targeted set — usually ONE piece, only the "
        "file(s) DIRECTLY responsible for a failing test. NEVER rework a file whose tests already "
        "pass, and never just diagnose. Adapt the fix to the cause: rewrite a wrong or flaky TEST to "
        "be DETERMINISTIC (control time explicitly / use distinct timestamps — never rely on real "
        "elapsed ms); fix a wrong MODULE; or harden a fragile DESIGN (e.g. an LRU that ties on equal "
        "timestamps → use a monotonic access counter, not the wall clock).\n\n"
        "After </think>, your FINAL answer must be ONLY this WORK-ORDER envelope — nothing before or "
        "after it, no prose, no contract restatement. ONE <<<PIECE>>> per file to rewrite; the file "
        "path in exports=, the concrete bug + fix in the instruction line:\n\n"
        "<<<PLAN strategy=file lang=LANG>>>\n"
        "shared:\n<blueprint>the one fact the fixes must share (one line)</blueprint>\n"
        "<<<PIECE id=fix1 exports=PATH/TO/FILE>>>\n"
        "instruction: what is wrong + exactly how to fix it (one line)\n"
        "<blueprint>one line</blueprint>\n"
        "<<</PIECE>>>\n"
        "<<<END>>>";
}

std::string build_arbiter_user(const std::string &contract, const std::string &failblock) {
    return "Worker-level repair exhausted its budget on the failures below and gave up. As the "
           "COORDINATOR, decide the rework. IMPORTANT: the MODULE may be correct and the TEST may be "
           "BUGGY — judge which file is actually wrong. You MAY target: a module (src/X.js), a test "
           "(test/X.test.js), or **a spec (design/X.blueprint) to RE-SPEC the component** — the blueprint "
           "is a LIVING doc; rewriting it updates the spec and a later round re-implements against it. "
           "You DELEGATE: parallel workers do the fixes. Emit a WORK-ORDER envelope — ONE PIECE per file "
           "to rewrite, "
           "the file path in exports=, and what is wrong + how to fix it in the instruction line:\n\n"
           "<<<PLAN strategy=file lang=LANG>>>\n"
           "shared:\n<blueprint>\none line\n</blueprint>\n"
           "<<<PIECE id=r1 exports=test/foo.test.js>>>\n"
           "instruction: what is wrong with this file and how to fix it\n"
           "<blueprint>one line</blueprint>\n"
           "<<</PIECE>>>\n"
           "<<<END>>>\n\n"
           "Emit a PLAN with zero PIECEs to accept the failures as-is.\n\n"
           "Interface contract:\n" + contract + "\n\nFailing pieces (module, test, error):\n" + failblock;
}

std::string build_rework_user(const std::string &target, const std::string &spec,
                                     const std::string &modcontent, const std::string &testcontent,
                                     const std::string &error, const std::string &guidance,
                                     const std::string &collaborators, const std::string &journal) {
    return "A test is failing. Study the FULL context below, then rewrite ONLY '" + target + "' to fix it.\n\n" +
           (journal.empty() ? "" : "PRIOR REPAIR ATTEMPTS (read FIRST — do NOT repeat these):\n" + journal + "\n\n") +
           "Spec / blueprint:\n" + (spec.empty() ? "(none)" : spec) +
           "\n\nModule:\n```\n" + (modcontent.empty() ? "(none)" : modcontent) +
           "\n```\n\nTest:\n```\n" + (testcontent.empty() ? "(none)" : testcontent) +
           "\n```\n\nTest output / error:\n" + error +
           (collaborators.empty() ? "" :
               "\n\nCollaborator modules it interacts with — match their ACTUAL calls (e.g. a fake canvas "
               "must implement every ctx.* method the real code uses):\n" + collaborators) +
           "\n\nCoordinator's guidance:\n" + guidance +
           "\n\nFirst write ONE line `ATTEMPT: <what you think is wrong + what you changed>`, then re-emit "
           "the COMPLETE corrected '" + target + "' via create_file at " + target + ".";
}

// ─── PA.6 §6.3: parallel reconcile ───────────────────────────────────────────
std::vector<std::pair<int,int>> reconcile_groups(int n, int g) {
    std::vector<std::pair<int,int>> out;
    if (n <= 0) return out;                       // guard: avoids n/g divide-by-zero when n==0
    if (g < 1) g = 1; if (g > n) g = n;
    int base = n / g, rem = n % g, start = 0;
    for (int i = 0; i < g; i++) { int c = base + (i < rem ? 1 : 0); out.push_back({ start, c }); start += c; }
    return out;
}
int reconcile_parallel_g(int n, int lanes) { return (lanes >= 1 && n >= 2 * lanes) ? std::min(lanes, n) : 1; }

// ─── PA.7: eager-scheduling core (PURE) ──────────────────────────────────────
// EKind and EItem are defined in pti_agents.h — use the header definitions

bool eager_thinks(EKind k) { return k == E_DESIGN || k == E_REWORK; }

int eager_prio(EKind k) {
    switch (k) { case E_DESIGN: return 0; case E_REWORK: return 1; case E_IMPL: return 2;
                 case E_TESTGEN: return 3; default: return 4; }
}

bool eager_dep_met(const std::vector<EItem> &items, size_t i) {
    auto done_of = [&](EKind k, const std::string &c) {
        for (auto &x : items) if (x.kind == k && x.comp == c) return x.done;
        return false;
    };
    switch (items[i].kind) {
        case E_DESIGN:  return true;
        case E_IMPL:    return done_of(E_DESIGN,  items[i].comp);
        case E_TESTGEN: return done_of(E_IMPL,    items[i].comp);
        case E_VERIFY:  return done_of(E_TESTGEN, items[i].comp);
        case E_REWORK:  return true;
    }
    return false;
}

std::vector<int> eager_ready(const std::vector<EItem> &items) {
    std::vector<int> r;
    for (size_t i = 0; i < items.size(); i++)
        if (!items[i].done && !items[i].dispatched && eager_dep_met(items, i)) r.push_back((int)i);
    std::stable_sort(r.begin(), r.end(),
                     [&](int a, int b){ return eager_prio(items[a].kind) < eager_prio(items[b].kind); });
    return r;
}

int eager_simulate(std::vector<EItem> items, int n_lanes, bool *idle) {
    if (idle) *idle = false;
    if (n_lanes < 1) n_lanes = 1;
    std::vector<int> rem(n_lanes, 0), job(n_lanes, -1);
    auto all_done = [&]{ for (auto &x : items) if (!x.done) return false; return true; };
    int t = 0;
    while (!all_done()) {
        for (int L = 0; L < n_lanes; L++) {
            if (rem[L] > 0) continue;
            auto ready = eager_ready(items);
            if (ready.empty()) break;
            int idx = ready.front(); items[idx].dispatched = true;
            job[L] = idx; rem[L] = std::max(1, items[idx].cost);
        }
        if (idle) for (int L = 0; L < n_lanes; L++)
            if (rem[L] == 0 && !eager_ready(items).empty()) *idle = true;
        int step = -1;
        for (int L = 0; L < n_lanes; L++) if (rem[L] > 0 && (step < 0 || rem[L] < step)) step = rem[L];
        if (step < 0) break;
        t += step;
        for (int L = 0; L < n_lanes; L++) if (rem[L] > 0) { rem[L] -= step;
            if (rem[L] == 0) { items[job[L]].done = true; job[L] = -1; } }
    }
    return t;
}

int staged_simulate(std::vector<EItem> items, int n_lanes) {
    if (n_lanes < 1) n_lanes = 1;
    int t = 0;
    for (int k = E_DESIGN; k <= E_VERIFY; k++) {
        std::vector<int> costs;
        for (auto &x : items) if ((int)x.kind == k) costs.push_back(std::max(1, x.cost));
        if (costs.empty()) continue;
        std::sort(costs.rbegin(), costs.rend());
        std::vector<int> lane(n_lanes, 0);
        for (int c : costs) {
            int L = 0; for (int i = 1; i < n_lanes; i++) if (lane[i] < lane[L]) L = i;
            lane[L] += c;
        }
        int mx = 0; for (int v : lane) mx = std::max(mx, v);
        t += mx;
    }
    return t;
}

// Build the DAG: for each unique component, create [DESIGN, IMPL, TESTGEN, VERIFY] items.
void build_dag(const std::vector<std::string> &components, int default_cost,
               std::vector<EItem> &dag) {
    dag.clear();
    for (auto &c : components) {
        EItem d; d.kind = E_DESIGN;  d.comp = c; d.cost = default_cost; dag.push_back(d);
        EItem i; i.kind = E_IMPL;    i.comp = c; i.cost = default_cost; dag.push_back(i);
        EItem t; t.kind = E_TESTGEN; t.comp = c; t.cost = default_cost / 2; if (t.cost < 1) t.cost = 1; dag.push_back(t);
        EItem v; v.kind = E_VERIFY;  v.comp = c; v.cost = 1; dag.push_back(v);
    }
}

// ─── PA.7 eager-scheduling self-test ─────────────────────────────────────────
int eager_self_test() {
    fprintf(stderr, "── PA.7 eager-scheduling self-test ──\n");
    int fail = 0;
    auto chk = [&](const char *n, bool ok){ fprintf(stderr, "  %s: %s\n", n, ok?"PASS":"FAIL"); if(!ok) fail++; };

    { std::vector<EItem> it = { {E_DESIGN,"bird"}, {E_IMPL,"bird"} };   // E1 readiness gating
      auto r = eager_ready(it);
      chk("E1 design ready, impl gated", r.size()==1 && it[r[0]].kind==E_DESIGN);
      it[0].done = true; auto r2 = eager_ready(it);
      chk("E2 impl ready after design", r2.size()==1 && it[r2[0]].kind==E_IMPL); }

    { std::vector<EItem> it = {                                         // E3 priority order
        {E_DESIGN,"c"}, {E_DESIGN,"b"}, {E_IMPL,"b"},
        {E_DESIGN,"a"}, {E_IMPL,"a"}, {E_TESTGEN,"a"} };
      it[1].done=true; it[3].done=true; it[4].done=true;               // b:design done, a:design+impl done
      auto r = eager_ready(it);
      chk("E3 design<impl<testgen", r.size()==3 && it[r[0]].kind==E_DESIGN
                                 && it[r[1]].kind==E_IMPL && it[r[2]].kind==E_TESTGEN); }

    chk("E4 per-item mode", eager_thinks(E_DESIGN) && eager_thinks(E_REWORK)
                         && !eager_thinks(E_IMPL) && !eager_thinks(E_TESTGEN));

    // E5/E6/E7: uneven, dependency-coupled work (≈ the measured 624–1385 design / 340–1671 impl spread)
    const char *comps[] = {"bird","pipes","renderer","engine"};
    int d[] = {7,9,12,6}, im[] = {8,8,10,17};
    std::vector<EItem> work;
    for (int i = 0; i < 4; i++) { work.push_back({E_DESIGN,comps[i],d[i]}); work.push_back({E_IMPL,comps[i],im[i]});
                                  work.push_back({E_TESTGEN,comps[i],3}); work.push_back({E_VERIFY,comps[i],1}); }
    bool idle = false; int e = eager_simulate(work, 4, &idle); int s = staged_simulate(work, 4);
    chk("E5 dispatcher work-conserving (no idle on ready work)", !idle);
    fprintf(stderr, "     makespan: eager=%d  staged(barriers)=%d  → %.0f%% faster\n",
            e, s, s ? 100.0*(s-e)/s : 0.0);
    chk("E6 eager <= staged", e <= s);
    chk("E7 eager < staged on uneven work", e < s);

    fprintf(stderr, "  %s (%d/8 passed)\n", fail==0 ? "ALL PASS" : "SOME FAILED", 8-fail);
    return fail > 0 ? 5 : 0;
}

// ─── self-test for all coordination helpers ──────────────────────────────────
// GPU-free self-test for the repair bookkeeping (like --gather-test / --mtp-test).
int coord_self_test() {
    fprintf(stderr, "── PA.4c coord self-test ──\n");
    int fail = 0;
    StackProfile _sv_stack = g_stack; g_stack = stack_profile("javascript");
    auto chk = [&](const char *n, bool ok){ fprintf(stderr, "  %s: %s\n", n, ok?"PASS":"FAIL"); if(!ok) fail++; };
    { std::vector<std::string> m={"src/bird.js","src/pipes.js"};
      chk("R1 module_for_test",  module_for_test("bird.test.js", m) == "src/bird.js"); }
    { std::vector<std::string> m={"src/bird.js"};
      chk("R2 no-match empty",   module_for_test("zzz.test.js", m).empty()); }
    { std::string u = build_amend_user("bird","src/bird.js","CODE","TEST","ERR","SPEC","COLLAB","JRNL");
      chk("R3 amend has parts",  u.find("src/bird.js")!=std::string::npos && u.find("CODE")!=std::string::npos
                              && u.find("TEST")!=std::string::npos && u.find("ERR")!=std::string::npos
                              && u.find("SPEC")!=std::string::npos && u.find("COLLAB")!=std::string::npos
                              && u.find("JRNL")!=std::string::npos && u.find("ATTEMPT:")!=std::string::npos); }
    chk("R4 verdict done",   repair_verdict(0,3,0) == RA_DONE);
    chk("R5 verdict repair", repair_verdict(0,3,2) == RA_REPAIR);
    chk("R6 verdict giveup", repair_verdict(3,3,2) == RA_GIVEUP);
    { auto rw = parse_rework("noise <<<REWORK file=test/a.test.js>>>\nfix the arc spy\n<<<END>>> tail");
      chk("R7 parse_rework", rw.size()==1 && rw[0].first=="test/a.test.js" && rw[0].second=="fix the arc spy"); }
    { auto rw = parse_rework("no markers here"); chk("R8 parse_rework empty", rw.empty()); }
    { std::string u = build_rework_user("src/bird.js","SPEC","MODC","TESTC","ERRC","GUIDE","COLLAB","JRNL");
      chk("R9 rework full ctx", u.find("src/bird.js")!=std::string::npos && u.find("SPEC")!=std::string::npos
                             && u.find("MODC")!=std::string::npos && u.find("TESTC")!=std::string::npos
                             && u.find("ERRC")!=std::string::npos && u.find("GUIDE")!=std::string::npos
                             && u.find("COLLAB")!=std::string::npos && u.find("JRNL")!=std::string::npos); }
    { std::string u = test_user("src/bird.js","CODEX","BPX","COLLABZ");
      chk("R10 testgen ctx", u.find("BPX")!=std::string::npos && u.find("COLLABZ")!=std::string::npos
                          && u.find("CODEX")!=std::string::npos); }
    { std::string plan =
        "<think>\n\n</think>\n<<<PLAN strategy=file lang=js>>>\nshared:\n<blueprint>x</blueprint>\n"
        "<<<PIECE id=r1 exports=test/pipes.test.js>>>\ninstruction: the assertion p.x===400 is wrong\n<blueprint>b</blueprint>\n<<</PIECE>>>\n"
        "<<<END>>>\n";
      auto rw = reworks_from_plan(plan);
      chk("R11 workorder→rework", rw.size()==1 && rw[0].first=="test/pipes.test.js"
                               && rw[0].second.find("p.x===400")!=std::string::npos); }
    { std::string empty = "<<<PLAN strategy=file lang=js>>>\nshared:\n<blueprint>x</blueprint>\n<<<END>>>\n";
      chk("R12 empty plan→none", reworks_from_plan(empty).empty()); }
    { std::string plan =
        "<<<PLAN strategy=file lang=js>>>\nshared:\n<blueprint>x</blueprint>\n"
        "<<<PIECE id=r1 exports=design/pipes.blueprint>>>\ninstruction: respec gapSize default to 120\n<blueprint>b</blueprint>\n<<</PIECE>>>\n"
        "<<<END>>>\n";
      auto rw = reworks_from_plan(plan);
      chk("R13 respec blueprint target", rw.size()==1 && rw[0].first=="design/pipes.blueprint"); }
    { chk("R14 extract_attempt", extract_attempt("blah\nATTEMPT: gapSize was off by one; fixed it\n<create_file>")
                                   == "gapSize was off by one; fixed it"
                              && extract_attempt("no marker here").empty()); }
    { auto g = reconcile_groups(13, 4);
      int tot = 0, mn = 99, mx = 0; for (auto &p : g) { tot += p.second; mn = std::min(mn, p.second); mx = std::max(mx, p.second); }
      chk("R15 reconcile_groups", g.size()==4 && tot==13 && (mx-mn)<=1
                               && reconcile_groups(3,8).size()==3 && reconcile_groups(0,4).empty()); }
    { chk("R16 comp_key strips .blueprint/.test.js/.js",
          comp_key("design/stringUtils.blueprint")=="stringUtils" && comp_key("test/bird.test.js")=="bird"
       && comp_key("src/pipes.js")=="pipes" && comp_key("engine")=="engine"); }
    { std::string t =
        "assert.strictEqual(truncate('hello', 6), 'hello.', 'msg');\n"
        "assert.strictEqual(truncate('hello', 6), 'hel...', 'msg2');\n"
        "assert.strictEqual(truncate('hi', 5), 'hi', 'ok');\n"
        "assert(m.capitalize('') === '');\n";
      auto c = contradictory_asserts(t);
      chk("R17 contradictory_asserts", c.size()==1 && c[0]=="truncate('hello',6)"); }
    { std::string f = frame_executed_truth("AssertionError: msg\n    'hello' !== 'hello.'\n");
      chk("R18 frame_executed_truth", f.find("`'hello'`")!=std::string::npos
                                   && f.find("`'hello.'`")!=std::string::npos
                                   && frame_executed_truth("no comparison here").empty()); }
    { chk("R19 reconcile gate (N>=2*lanes)",
          reconcile_parallel_g(6,4)==1 && reconcile_parallel_g(13,4)==4 && reconcile_parallel_g(8,4)==4
       && reconcile_parallel_g(3,2)==1); }
    { std::vector<std::string> bases = {"bird","pipes","engine","renderer"};
      auto r = module_refs("const bird=require('./bird'); renderer.draw(bird);", "engine", bases);
      bool ok = std::find(r.begin(),r.end(),"bird")!=r.end() && std::find(r.begin(),r.end(),"renderer")!=r.end()
             && std::find(r.begin(),r.end(),"pipes")==r.end() && std::find(r.begin(),r.end(),"engine")==r.end();
      auto leaf = module_refs("function add(a,b){return a+b;}", "math", bases);
      auto noword = module_refs("blackbird flies", "x", {"bird"});
      chk("R20 module_refs (internal/leaf/whole-word)", ok && leaf.empty() && noword.empty()); }
    { std::string t = build_integration_task("GOAL", "engine", "CODEX", "// bird.js\nstub");
      chk("R21 build_integration_task", t.find("engine")!=std::string::npos && t.find("do NOT mock")!=std::string::npos
                                     && t.find("integration.test.js")!=std::string::npos && t.find("CODEX")!=std::string::npos); }
    { chk("R22 integration_base",
          integration_base("test/engine.integration.test.js")=="engine"
       && integration_base("test/bird.test.js")=="bird" && integration_base("src/pipes.js")=="pipes"); }
    { chk("R23 tool_call_truncated",
          tool_call_truncated("blah <create_file><path>x</path><content>partial...")
       && !tool_call_truncated("<create_file><path>x</path><content>full</content></create_file>")
       && !tool_call_truncated("no tool call here")); }
    { auto js = stack_profile("javascript"); auto py = stack_profile("python");
      chk("R24 stack_profile", js.src_ext==".js" && js.test_suffix==".test.js" && js.runner=="node"
                            && py.src_ext==".py" && py.test_suffix=="_test.py"
                            && has_suffix("a.test.js",".test.js") && !has_suffix("a.js",".test.js")); }
    { bool ok_js = is_test_file("bird.test.js") && !is_src_file("bird.test.js") && is_src_file("bird.js")
                && test_file_name("bird")=="bird.test.js" && src_file_name("bird")=="bird.js";
      StackProfile sv = g_stack; g_stack = stack_profile("python");
      bool ok_py = is_test_file("bird_test.py") && is_src_file("bird.py") && !is_test_file("bird.py")
                && test_file_name("bird")=="bird_test.py";
      g_stack = sv;
      chk("R25 file helpers (js+python)", ok_js && ok_py); }
    { auto m = vars_parse("# a comment\nlang = javascript\ndb_url = postgres://h/db?x=1\n\nnote = a\\nb\n");
      chk("R26 vars_parse (comments, '=' in value, escape)",
          m["lang"]=="javascript" && m["db_url"]=="postgres://h/db?x=1" && m["note"]=="a\nb" && m.count("# a comment")==0); }
    { std::map<std::string,std::string> sv = g_vars; g_vars.clear();
      g_vars["lang"]="javascript"; g_vars["note"]="line1\nline2";
      std::string ser; for (auto &kv : g_vars) ser += kv.first + " = " + line_escape(kv.second) + "\n";
      auto back = vars_parse(ser); g_vars = sv;
      chk("R27 vars round-trip (multiline escaped)", back["lang"]=="javascript" && back["note"]=="line1\nline2"
                                                  && ser.find("line1\\nline2")!=std::string::npos); }
    { std::map<std::string,std::string> sv = g_vars; g_vars.clear();
      int n = vars_absorb("...reasoning...\nSET_VAR db_engine=mysql\nSET_VAR storage = MyISAM\nmore text\n");
      bool ok = n==2 && g_vars["db_engine"]=="mysql" && g_vars["storage"]=="MyISAM";
      g_vars = sv;
      chk("R28 vars_absorb (SET_VAR discovery)", ok); }
    { StackProfile svp = g_stack; std::map<std::string,std::string> svv = g_vars;
      g_vars.clear();
      std::string js = test_directive(); g_stack = stack_profile("python"); std::string py = test_directive();
      g_vars["test_framework"] = "pytest"; std::string pyfw = test_directive();
      g_stack = svp; g_vars = svv;
      chk("R29 test_directive retargets",
          js.find("javascript")!=std::string::npos && js.find("console.assert")!=std::string::npos
       && py.find("python")!=std::string::npos && py.find("console.assert")==std::string::npos
       && pyfw.find("pytest")!=std::string::npos); }
    { chk("R30 lang_from_task",
          lang_from_task("Build a CLI tool in python")=="python"
       && lang_from_task("a web game in JavaScript")=="javascript"
       && lang_from_task("design a postgres schema")=="sql"
       && lang_from_task("build a game").empty()
       && lang_from_task("a jszip-like thing").empty()); }
    { chk("R31 lang_from_marker",
          lang_from_marker("package.json")=="javascript" && lang_from_marker("requirements.txt")=="python"
       && lang_from_marker("pyproject.toml")=="python" && lang_from_marker("README.md").empty()); }
    { chk("R32 test_critic_note",
          !test_critic_note("## TEST DEFECTS\n- CONTRACT-MISMATCH — foo(1) — spec says 2").empty()
       && test_critic_note("TESTS OK").empty()
       && test_critic_note("## TEST DEFECTS").empty()         // bare heading, no bullet → no note
       && test_critic_note("no verdict at all").empty()); }
    { chk("R33 code_critic_note",
          !code_critic_note("## CONTRACT VIOLATIONS\n- CALL-SYNTAX — e.isExpired() — contract declares get isExpired").empty()
       && code_critic_note("CODE OK").empty()
       && code_critic_note("## CONTRACT VIOLATIONS").empty()      // bare heading, no bullet → no note
       && code_critic_note("looks fine to me").empty()); }
    { std::string sy = build_critic_system("ROLEX", "CONTRACTY");   // contract must survive into the cached system turn
      chk("R34 build_critic_system carries role+contract",
          sy.find("ROLEX")!=std::string::npos && sy.find("CONTRACTY")!=std::string::npos); }
    fprintf(stderr, "  %s (%d/34 passed)\n", fail==0 ? "ALL PASS" : "SOME FAILED", 34-fail);
    g_stack = _sv_stack;
    return fail > 0 ? 5 : 0;
}
