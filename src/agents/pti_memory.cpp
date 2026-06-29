/*
 * pti_memory.cpp — PA.8: blackboard, variable store, stack profile,
 *             tool execution harness, test generation helpers
 *
 * Contains: StackProfile, variable store (g_vars), stack resolution,
 *           run_worker_tools, test_user, comp_key, match_paren, split_top_commas
 *
 * Dependencies: pti_common.cpp, pti_parser.cpp, pti_tools.cpp
 * Exposes:  g_stack, stack_profile(), vars_load/save/absorb/render(),
 *           var_or(), test_directive(), module_for_test(), comp_key(),
 *           run_worker_tools(), test_user()
 */

#include "pti_agents.h"
#include <regex>

// ── PA.8 §9: STACK PROFILE — centralize the (currently JS-baked) language assumptions ──────────
// (Layer 1a: the 4 core mechanics — detection, run cmd, module_for_test, comp_key. Path-strings +
// prompts are still JS-literal = Layer 1b/2.)

StackProfile g_stack = { "javascript", ".js", ".test.js", "node",
    "CommonJS (`module.exports` / `require`, NOT `import`)",
    "use ONLY built-ins + `console.assert` (NO external packages); print a line per check; call `process.exit(1)` on ANY failure" };

StackProfile stack_profile(std::string name) {
    for (auto &c : name) c = (char)tolower((unsigned char)c);
    if (name == "python" || name == "py") return { "python", ".py",  "_test.py",  "python3",
        "standard modules (`def` functions; `import`)",
        "use plain `assert` statements; print a line per check; exit non-zero (`raise SystemExit(1)`) on ANY failure" };
    if (name == "sql")                    return { "sql",    ".sql", "_test.sql", "psql -f",
        "plain SQL scripts (no module system)",
        "assert via queries returning expected rows; signal failure with a non-zero exit / RAISE" };
    return { "javascript", ".js", ".test.js", "node",
        "CommonJS (`module.exports` / `require`, NOT `import`)",
        "use ONLY built-ins + `console.assert` (NO external packages); print a line per check; call `process.exit(1)` on ANY failure" };
}

bool has_suffix(const std::string &s, const std::string &suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}
std::string stk_test_suffix() { return g_stack.test_suffix; }
std::string stk_module_hint() { return g_stack.module_hint; }
bool is_test_file(const std::string &fn) { return has_suffix(fn, g_stack.test_suffix); }
bool is_src_file (const std::string &fn) { return has_suffix(fn, g_stack.src_ext) && !is_test_file(fn); }
std::string test_file_name(const std::string &comp) { return comp + g_stack.test_suffix; }
std::string src_file_name (const std::string &comp) { return comp + g_stack.src_ext; }
std::string run_test_cmd  (const std::string &rel)  { return g_stack.runner + " '" + rel + "'"; }

// ── PA.8 §9.2: STACK RESOLUTION ──────────────────────────────────────────────
static bool word_at(const std::string &h, const std::string &n, size_t p) {
    auto idc = [](char c){ return isalnum((unsigned char)c) || c == '_' || c == '+'; };
    char b = p ? h[p-1] : ' ', a = (p + n.size() < h.size()) ? h[p + n.size()] : ' ';
    return !idc(b) && !idc(a);
}
static bool has_word(const std::string &haystack_lc, const std::string &w) {
    for (size_t p = 0; (p = haystack_lc.find(w, p)) != std::string::npos; p += w.size())
        if (word_at(haystack_lc, w, p)) return true;
    return false;
}
std::string lang_from_task(const std::string &task) {
    std::string t = task; for (auto &c : t) c = (char)tolower((unsigned char)c);
    if (has_word(t,"python")||has_word(t,"pytest")||has_word(t,"django")||has_word(t,"flask")) return "python";
    if (has_word(t,"javascript")||has_word(t,"node")||has_word(t,"node.js")||has_word(t,"js")) return "javascript";
    if (has_word(t,"sql")||has_word(t,"postgres")||has_word(t,"postgresql")||has_word(t,"sqlite")) return "sql";
    return "";
}
std::string lang_from_marker(const std::string &filename) {
    if (filename == "package.json") return "javascript";
    if (filename == "requirements.txt" || filename == "pyproject.toml" || filename == "setup.py") return "python";
    return "";
}
std::string detect_lang_in_dir(const std::string &dir) {
    std::error_code ec;
    for (auto it = std::filesystem::directory_iterator(dir, ec);
         !ec && it != std::filesystem::directory_iterator(); it.increment(ec)) {
        if (!it->is_regular_file(ec)) continue;
        std::string l = lang_from_marker(it->path().filename().string());
        if (!l.empty()) return l;
    }
    return "";
}

// ── PA.8 §9.1: the session VARIABLE STORE ────────────────────────────────────
// Persisted ONE LINE PER ITEM (`key = value`) in <project>/.blackboard/memory
char g_blackboard[512] = "";
std::map<std::string,std::string> g_vars;
std::string blackboard_dir() { return g_blackboard[0] ? std::string(g_blackboard)
                                                       : std::string(g_work_dir) + "/.blackboard"; }
std::string vars_path()      { return blackboard_dir() + "/memory"; }

std::string line_escape(const std::string &s) {
    std::string o; o.reserve(s.size());
    for (char c : s) { if (c == '\\') o += "\\\\"; else if (c == '\n') o += "\\n"; else if (c == '\r') {} else o += c; }
    return o;
}
static std::string line_unescape(const std::string &s) {
    std::string o; o.reserve(s.size());
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '\\' && i+1 < s.size()) { char e = s[++i]; o += (e == 'n') ? '\n' : e; }
        else o += s[i];
    }
    return o;
}
std::map<std::string,std::string> vars_parse(const std::string &s) {
    std::map<std::string,std::string> out; size_t i = 0;
    while (i <= s.size()) {
        size_t nl = s.find('\n', i); std::string line = s.substr(i, (nl == std::string::npos ? s.size() : nl) - i);
        i = (nl == std::string::npos) ? s.size() + 1 : nl + 1;
        std::string t = trim(line);
        if (t.empty() || t[0] == '#') continue;
        size_t eq = t.find('=');  if (eq == std::string::npos) continue;
        std::string k = trim(t.substr(0, eq)), v = trim(t.substr(eq + 1));
        if (!k.empty()) out[k] = line_unescape(v);
    }
    return out;
}
void vars_save() {
    std::error_code ec; std::filesystem::create_directories(blackboard_dir(), ec);
    FILE *f = fopen(vars_path().c_str(), "w"); if (!f) return;
    for (auto &kv : g_vars) fprintf(f, "%s = %s\n", kv.first.c_str(), line_escape(kv.second).c_str());
    fclose(f);
}
void vars_load() {
    FILE *f = fopen(vars_path().c_str(), "r"); if (!f) return;
    std::string s; char b[4096]; size_t k; while ((k = fread(b,1,sizeof(b),f)) > 0) s.append(b,k); fclose(f);
    for (auto &kv : vars_parse(s)) g_vars[kv.first] = kv.second;
}
void vars_seed_from_stack() { g_vars.emplace("lang", g_stack.lang); g_vars.emplace("src_ext", g_stack.src_ext);
                               g_vars.emplace("test_suffix", g_stack.test_suffix); g_vars.emplace("runner", g_stack.runner); }
std::string vars_render() {
    if (g_vars.empty()) return "";
    std::string o = "PROJECT VARIABLES (resolved decisions — conform to ALL of these):\n";
    for (auto &kv : g_vars) o += "  " + kv.first + " = " + kv.second + "\n";
    return o;
}
int vars_absorb(const std::string &worker_out) {
    static const std::regex re(R"(SET_VAR\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^\n]+))");
    int n = 0; for (std::sregex_iterator it(worker_out.begin(), worker_out.end(), re), e; it != e; ++it) {
        g_vars[(*it)[1].str()] = trim((*it)[2].str()); n++; }
    return n;
}
std::string var_or(const std::string &key, const std::string &fallback) {
    auto it = g_vars.find(key); return (it != g_vars.end() && !it->second.empty()) ? it->second : fallback;
}

// PA.8 Layer 2: the stack-neutral "how to write + run a test here" directive
std::string test_directive() {
    std::string lang = var_or("lang", g_stack.lang);
    std::string fw   = var_or("test_framework", "");
    std::string th   = var_or("test_hint", g_stack.test_hint);
    std::string d = "in " + lang + ", " + (fw.empty() ? "" : "using the " + fw + " framework, ") + th
                  + "; runnable with `" + var_or("runner", g_stack.runner) + " <file" + var_or("test_suffix", g_stack.test_suffix) + ">`";
    return d;
}

// ── Destructive-command guard (regex-based, for run_worker_tools) ────────────
static bool safe_join(const std::string &rel, std::filesystem::path &out) {
    if (rel.empty() || rel.front() == '/') return false;
    if (rel.find("..") != std::string::npos) return false;
    out = std::filesystem::path(g_work_dir) / rel;
    return true;
}

// Detect truncated tool calls (cap truncation). Pure; R23.
bool tool_call_truncated(const std::string &out) {
    auto open_no_close = [&](const char *o, const char *c){
        size_t p = out.rfind(o); return p != std::string::npos && out.find(c, p) == std::string::npos; };
    return open_no_close("<create_file>", "</create_file>") || open_no_close("<content>", "</content>");
}

// Execute the tool calls in each worker's output; return a textual report.
std::string run_worker_tools(const std::vector<std::pair<std::string,std::string>> &worker_results) {
    namespace fs = std::filesystem;
    std::error_code ec; fs::create_directories(g_work_dir, ec);
    std::string report;
    for (auto &wr : worker_results) {
        auto calls = parse_tool_calls(wr.second);
        if (calls.empty() && tool_call_truncated(wr.second)) {
            fprintf(stderr, "  ⚠ %s: tool call TRUNCATED by the -n cap — no file written; raise -n\n", wr.first.c_str());
            report += "  [" + wr.first + "] TRUNCATED tool call (hit -n cap) — no file written\n";
        }
        for (auto &c : calls) {
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
    // PA.8 §9.1: absorb any `SET_VAR key=value` facts
    int nv = 0; for (auto &wr : worker_results) nv += vars_absorb(wr.second);
    if (nv) { vars_save(); fprintf(stderr, "  ⊕ %d discovered variable(s) → %s\n", nv, vars_path().c_str()); }
    return report;
}

// PA.4: the per-item USER turn of a "write a test for this module" task
std::string test_user(const std::string &relpath, const std::string &content,
                      const std::string &blueprint, const std::string &collaborators) {
    std::filesystem::path p(relpath);
    std::string base = p.filename().string();
    base = comp_key(base);
    std::string testpath = "test/" + base + stk_test_suffix();
    return "Write a unit test for the '" + base + "' module — test it against its SPEC (below), "
        "not merely whatever the current code happens to do. (The project goal + interface contract are "
        "above.)\n\nSpec / blueprint for '" + base + "':\n" + (blueprint.empty() ? "(none on disk)" : blueprint) +
        "\n\nModule file (" + relpath + "):\n```\n" + content + "\n```\n" +
        (collaborators.empty() ? "" :
            "\nCollaborator modules this one interacts with — so your stubs/mocks match the methods "
            "they ACTUALLY call (e.g. a fake canvas must implement every ctx.* method the real code uses):\n"
            + collaborators) +
        "\nThe test MUST: be written " + test_directive() + "; import the module via " +
        var_or("module_hint", stk_module_hint()) + " (e.g. from '../" + relpath + "'); **stub collaborators "
        "directly** (NO external packages — do NOT pull in jsdom/jest/etc.); follow any TECH DECISIONS in the "
        "contract above. Save it with create_file at path: " + testpath + "\nOutput only the create_file tool call.";
}

// ── PA4 §4.6 repair-quality helpers ──────────────────────────────────────────
std::string comp_key(const std::string &path) {
    std::string b = std::filesystem::path(path).filename().string();
    for (const std::string &suf : { g_stack.test_suffix, std::string(".blueprint"), g_stack.src_ext,
                                    std::string(".mjs"), std::string(".cjs"), std::string(".ts"), std::string(".py") })
        if (has_suffix(b, suf)) return b.substr(0, b.size() - suf.size());
    return b;
}

size_t match_paren(const std::string &s, size_t open) {
    int depth = 0; bool inq = false; char q = 0;
    for (size_t i = open; i < s.size(); i++) { char c = s[i];
        if (inq) { if (c == '\\') i++; else if (c == q) inq = false; continue; }
        if (c == '\'' || c == '"' || c == '`') { inq = true; q = c; }
        else if (c == '(') depth++;
        else if (c == ')' && --depth == 0) return i;
    }
    return std::string::npos;
}

std::vector<std::string> split_top_commas(const std::string &s) {
    std::vector<std::string> out; int depth = 0; bool inq = false; char q = 0; size_t start = 0;
    for (size_t i = 0; i < s.size(); i++) { char c = s[i];
        if (inq) { if (c == '\\') i++; else if (c == q) inq = false; continue; }
        if (c == '\'' || c == '"' || c == '`') { inq = true; q = c; }
        else if (c == '(' || c == '[' || c == '{') depth++;
        else if (c == ')' || c == ']' || c == '}') depth--;
        else if (c == ',' && depth == 0) { out.push_back(s.substr(start, i - start)); start = i + 1; }
    }
    out.push_back(s.substr(start));
    return out;
}

// Which module a failing test targets: "bird.test.js" → the module whose base is "bird".
std::string module_for_test(const std::string &test_filename,
                            const std::vector<std::string> &modules) {
    std::string base = test_filename;
    if (has_suffix(base, g_stack.test_suffix)) base = base.substr(0, base.size() - g_stack.test_suffix.size());
    for (auto &m : modules)
        if (std::filesystem::path(m).filename().string() == src_file_name(base)) return m;
    return "";
}
