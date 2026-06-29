/*
 * pti_parser.cpp — work-order envelope parser
 *
 * Parses the <<<PLAN/PIECE/SELF/END>>> envelope the boss emits after
 * decomposing a task.  Pure text processing — no GPU, no globals
 * beyond the shared header types.
 *
 * Exposes:  trim(), split_csv(), parse_marker(), parse_work_order()
 */

#include "pti_agents.h"

// ─── helpers (file-local) ────────────────────────────────────────────────────

std::string attr_get(const std::vector<std::pair<std::string,std::string>> &a,
                     const std::string &k, const std::string &dflt) {
    for (auto &p : a) if (p.first == k) return p.second;
    return dflt;
}

// <blueprint>...</blueprint> if present, else the trimmed body
std::string extract_blueprint(const std::string &body) {
    size_t a = body.find("<blueprint>");
    size_t b = body.find("</blueprint>");
    if (a != std::string::npos && b != std::string::npos && b > a)
        return trim(body.substr(a + 11, b - (a + 11)));
    return trim(body);
}

// text after the first "instruction:" line
std::string extract_instruction(const std::string &body) {
    size_t p = body.find("instruction:");
    if (p == std::string::npos) return "";
    size_t e = body.find('\n', p);
    size_t s = p + 12;
    return trim(body.substr(s, (e == std::string::npos ? body.size() : e) - s));
}

// ─── public API ──────────────────────────────────────────────────────────────

std::string trim(const std::string &s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

std::vector<std::string> split_csv(const std::string &s) {
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
bool parse_marker(const std::string &line, std::string &tag,
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

// Parse the boss's work-order envelope into a WorkOrder struct.
// Returns ok=true on success with all pieces populated.
WorkOrder parse_work_order(const std::string &text) {
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
            } else if (tag == "/PIECE" || tag == "/SELF") {
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
    // Robustness: if the PLAN marker was malformed, recover the shared block.
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
        if (el.empty() || el == "none") continue;
        for (auto &s : seen) if (s == e) { wo.error = "export collision: " + e; return wo; }
        seen.push_back(e);
    }
    wo.ok = true;
    return wo;
}
