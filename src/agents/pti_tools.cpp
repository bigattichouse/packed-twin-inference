/*
 * pti_tools.cpp — PA.5: worker tool-calls + PA.7b active-retrieval
 *
 * Workers may emit nanocoder-style XML tool calls.  The harness executes a
 * small allowlist — create_file (sandboxed to --work-dir) and execute_bash
 * (gated behind --allow-run with a destructive-command guard).
 *
 * PA.7b active-retrieval primitives (read_file / abandon) are also parsed
 * here; the mid-stream loop wires them later.
 *
 * Exposes:  parse_tool_calls(), run_sandbox(), PA.7b helpers
 */

#include "pti_agents.h"
#include <sys/wait.h>
#include <sys/stat.h>
#include <unistd.h>
#include <csignal>

std::string tc_param(const ToolCall &c, const std::string &k) {
    for (auto &p : c.params) if (p.first == k) return p.second;
    return "";
}

// Parse <tool>...<param>value</param>...</tool> for the KNOWN tool names only (so a
// stray <div> etc. in worker code is never mistaken for a tool call).
std::vector<ToolCall> parse_tool_calls(const std::string &text) {
    // create_file/execute_bash (PA.5) + read_file/abandon (PA.7b active retrieval — parsed now,
    // handled by the mid-stream loop when that lands; harmless until a prompt offers them).
    static const char *TOOLS[] = { "create_file", "execute_bash", "read_file", "abandon" };
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

// ── PA.7b active-retrieval primitives (pure; the mid-stream loop wires them later) ────────────────
// A worker's read_file → the file's content, or a NOT_FOUND sentinel when it isn't built yet (eager
// mode: its producer hasn't run). NOT_FOUND is the signal to abandon + re-queue gated on that file.
static const char *PA7_NOTFOUND = "NOT_FOUND";
std::string resolve_read(const std::string &work_dir, const std::string &rel) {
    if (rel.empty() || rel.find("..") != std::string::npos)
        return std::string(PA7_NOTFOUND) + ": invalid path";
    std::string full = work_dir + "/" + rel;
    FILE *f = fopen(full.c_str(), "r");
    if (!f) return std::string(PA7_NOTFOUND) + ": " + rel + " (not built yet)";
    std::string s; char b[8192]; size_t k; while ((k = fread(b,1,sizeof(b),f)) > 0) s.append(b,k); fclose(f);
    return s;
}
bool pa7_is_notfound(const std::string &r) { return r.rfind(PA7_NOTFOUND, 0) == 0; }

// Re-queue a worker that called abandon(reason): the harness holds the original dispatch prompt and
// re-enqueues it with the reason appended (+ the awaited file's content once it exists). The model
// reconstructs nothing — the harness owns continuity. (require → abandon → re-queue, never wait.)
std::string build_blocked_requeue(const std::string &orig_prompt, const std::string &reason,
                                  const std::string &resolved) {
    std::string out = orig_prompt + "\n\n[Earlier you abandoned this task: " + trim(reason) + "]";
    if (!resolved.empty() && !pa7_is_notfound(resolved))
        out += "\nThe file you were waiting on is now available:\n```\n" + resolved + "\n```\n"
               "Complete the task now.";
    return out;
}

// Destructive-command guard for execute_bash (regex-based; matches the monolith / the T4 test).
bool is_dangerous_cmd(const std::string &cmd) {
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

// ── sandbox execution (PA.5) ────────────────────────────────────────────────
static std::string run_sandbox(const ToolCall &call, const std::string &work_dir) {
    // Ensure work_dir exists.
    mkdir(work_dir.c_str(), 0755);

    if (call.name == "create_file") {
        std::string path = tc_param(call, "path");
        std::string content = tc_param(call, "content");
        if (path.empty()) return "ERROR: create_file missing path";
        // Sanitize — refuse to escape work_dir.
        if (path.find("..") != std::string::npos) return "ERROR: create_file escapes work-dir";
        std::string full = work_dir + "/" + path;
        // Create parent dirs if needed.
        { size_t p = 0; while ((p = full.find('/', p+1)) != std::string::npos)
              mkdir(full.substr(0, p).c_str(), 0755); }
        FILE *f = fopen(full.c_str(), "w");
        if (!f) return "ERROR: create_file failed to open";
        fputs(content.c_str(), f); fclose(f);
        return "OK: created " + path + " (" + std::to_string(content.size()) + " bytes)";
    }

    if (call.name == "execute_bash") {
        if (!g_allow_run) return "ERROR: execute_bash requires --allow-run";
        std::string cmd = tc_param(call, "cmd");
        if (cmd.empty()) return "ERROR: execute_bash missing cmd";
        if (is_dangerous_cmd(cmd)) return "ERROR: blocked dangerous command";
        char cmd_full[4096];
        snprintf(cmd_full, sizeof(cmd_full), "cd '%s' && timeout 30 bash -c '%s' 2>&1",
                 work_dir.c_str(), cmd.c_str());
        std::string result;
        FILE *pf = popen(cmd_full, "r");
        if (!pf) return "ERROR: popen failed";
        char buf[4096]; size_t n;
        while ((n = fread(buf, 1, sizeof(buf), pf)) > 0) result.append(buf, n);
        pclose(pf);
        if (result.size() > 2048) result = result.substr(0, 2040) + "\n... [truncated]";
        return result;
    }

    return "UNKNOWN tool: " + call.name;
}
