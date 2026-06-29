/*
 * pti_prompt.cpp — PA.1: boss prompting, chat template, gather phase
 *
 * Contains: boss system prompts, chat template application, boss_generate(),
 * print_work_order(), extract_code_fence(), build_gather_prompt(), run_gather_phase(),
 * and the worker preamble + lane prompt builders.
 *
 * Dependencies: pti_common.cpp (llama primitives, pick/tok_str),
 *               pti_stream.cpp (inject, batch helpers),
 *               pti_parser.cpp (parse_work_order, trim),
 *               pti_memory.cpp (test_directive, var_or, stk_*)
 * Exposes:  boss_system_text(), apply_chat_template(), boss_generate(),
 *           print_work_order(), build_lane_prompt(), build_prefix(),
 *           run_gather_phase(), write_artifact(), finish_gather()
 */

#include "pti_agents.h"

// ─── sample envelope (self-test only) ────────────────────────────────────────
static const char *SAMPLE_ENVELOPE =
    "<<<PLAN strategy=function lang=js>>>\n"
    "shared:\n"
    "<blueprint>\n"
    "  Bird  { x: float, y: float, vy: float }\n"
    "  Pipe  { x: float, gapY: float, gapH: float, passed: bool }\n"
    "  deps: []\n"
    "</blueprint>\n"
    "<<<PIECE id=w1 exports=gravityStep>>>\n"
    "instruction: implement this function given the public params of the current class, in js\n"
    "<blueprint> gravityStep(bird, dt) -> void { apply gravity then velocity to bird } </blueprint>\n"
    "<<</PIECE>>>\n"
    "<<<PIECE id=w2 exports=spawnPipe>>>\n"
    "instruction: implement this function given the public params of the current class, in js\n"
    "<blueprint> spawnPipe(world) -> void { append a Pipe with a random gap to world.pipes } </blueprint>\n"
    "<<</PIECE>>>\n"
    "<<<PIECE id=w3 exports=checkCollision>>>\n"
    "instruction: implement this function given the public params of the current class, in js\n"
    "<blueprint> checkCollision(bird, pipes) -> bool { true if bird hits a pipe or ground } </blueprint>\n"
    "<<</PIECE>>>\n"
    "<<<SELF id=boss exports=stepWorld, smokeTest>>>\n"
    "instruction: implement this object in js - wire the three functions and a smoke test\n"
    "<blueprint> Integration { stepWorld(world, dt) -> void, smokeTest() -> bool } </blueprint>\n"
    "<<</SELF>>>\n"
    "<<<END>>>\n";

// ─── work order display + self-test ──────────────────────────────────────────
void print_work_order(const WorkOrder &wo) {
    fprintf(stderr, "strategy : %s\n", wo.strategy.c_str());
    fprintf(stderr, "shared   : %zu chars — \"%.50s...\"\n", wo.shared.size(), trim(wo.shared).c_str());
    for (auto &p : wo.pieces) {
        fprintf(stderr, "  %-5s boss=%d lang=%-3s exports=[", p.id.c_str(), p.is_boss, p.language.c_str());
        for (size_t i = 0; i < p.exports.size(); i++) fprintf(stderr, "%s%s", i ? "," : "", p.exports[i].c_str());
        fprintf(stderr, "]\n");
        fprintf(stderr, "        instr: %.66s\n", p.instruction.c_str());
        fprintf(stderr, "        bp   : %.66s\n", p.blueprint.c_str());
    }
    if (wo.ok) fprintf(stderr, "PARSE OK — %zu pieces, no export collisions\n", wo.pieces.size());
    else       fprintf(stderr, "PARSE FAIL — %s\n", wo.error.c_str());
}

int parse_self_test() {
    fprintf(stderr, "── PA.1a work-order envelope parse self-test ──\n");
    WorkOrder wo = parse_work_order(SAMPLE_ENVELOPE);
    print_work_order(wo);
    // Regression: the "none" sentinel (index.html, *.test.js) must NOT count as an export collision.
    const char *NONE_ENV =
        "<<<PLAN strategy=file lang=js>>>\nshared:\n<blueprint>\nx\n</blueprint>\n"
        "<<<PIECE id=bird exports=Bird>>>\ninstruction: src/bird.js\n<blueprint>b</blueprint>\n<<</PIECE>>>\n"
        "<<<PIECE id=index exports=none>>>\ninstruction: index.html\n<blueprint>i</blueprint>\n<<</PIECE>>>\n"
        "<<<PIECE id=style exports=none>>>\ninstruction: style.css\n<blueprint>s</blueprint>\n<<</PIECE>>>\n"
        "<<<END>>>\n";
    WorkOrder w2 = parse_work_order(NONE_ENV);
    fprintf(stderr, "  none-sentinel: %s (%s)\n", w2.ok ? "PASS" : "FAIL",
            w2.ok ? "no false collision" : w2.error.c_str());
    return (wo.ok && w2.ok) ? 0 : 2;
}

// ─── boss prompting (PA.1 PLAN phase) ────────────────────────────────────────

std::string apply_chat_template(const std::vector<Msg> &msgs, bool add_ass) {
    std::vector<llama_chat_message> chat;
    for (auto &m : msgs) chat.push_back({ m.role.c_str(), m.content.c_str() });
    int32_t need = llama_chat_apply_template(G.tmpl, chat.data(), chat.size(), add_ass, nullptr, 0);
    if (need < 0) { std::string raw; for (auto &m : msgs) raw += m.content + "\n"; return raw; }
    std::vector<char> buf(need + 1);
    llama_chat_apply_template(G.tmpl, chat.data(), chat.size(), add_ass, buf.data(), (int32_t)buf.size());
    return std::string(buf.data(), need);
}

// greedy single-stream decode in seq 0 from a formatted prompt; returns the text
std::string boss_generate(const std::string &prompt, int max_tok) {
    std::vector<llama_token> toks(prompt.size() + 16);
    int n = llama_tokenize(G.vocab, prompt.c_str(), (int32_t)prompt.size(),
                           toks.data(), (int32_t)toks.size(), true, true);
    if (n < 0) { toks.resize(-n);
        n = llama_tokenize(G.vocab, prompt.c_str(), (int32_t)prompt.size(),
                           toks.data(), (int32_t)toks.size(), true, true); }
    toks.resize(n > 0 ? n : 0);
    llama_memory_seq_rm(G.mem, 0, 0, -1);
    llama_batch batch = llama_batch_init(PREFILL_CHUNK + 8, 0, 1);
    int last_idx = 0;
    for (int i0 = 0; i0 < n; i0 += PREFILL_CHUNK) {
        int nb = n - i0 < PREFILL_CHUNK ? n - i0 : PREFILL_CHUNK;
        batch_clear(&batch);
        for (int j = 0; j < nb; j++)
            batch_add(&batch, toks[i0 + j], (llama_pos)(i0 + j), 0, i0 + j == n - 1);
        if (llama_decode(G.ctx, batch) != 0) { llama_batch_free(batch); return "[prefill failed]"; }
        last_idx = nb - 1;
    }
    llama_sampler *s = g_greedy ? nullptr : make_sampler(g_seed, g_boss_sp);   // boss plan sampler (Qwen)
    llama_token tok = pick(s, last_idx);
    llama_pos pos = n;
    std::string out;
    for (int gen = 0; gen < max_tok && !llama_vocab_is_eog(G.vocab, tok); gen++) {
        { std::string piece = tok_str(tok); out += piece; fputs(piece.c_str(), stderr); }   // stream live
        batch_clear(&batch);
        batch_add(&batch, tok, pos, 0, true);
        if (llama_decode(G.ctx, batch) != 0) break;
        tok = pick(s, 0);
        pos++;
    }
    if (s) llama_sampler_free(s);
    llama_batch_free(batch);
    return out;
}

static const char *BOSS_PROMPT =
    "You are the COORDINATOR of a packed-agent coding team. Workers run in parallel and in\n"
    "ISOLATION — a worker sees only the shared interface, its own instruction, and its spec,\n"
    "never another worker or you. Workers are FULL, CAPABLE models: give each a meaty,\n"
    "clearly-specified chunk and it will handle it.\n\n"
    "Decompose the user's coding task into a HANDFUL of SUBSTANTIAL, independent, balanced\n"
    "items — each one a <<<PIECE>>>. SIZE FLOOR: every item is AT LEAST a full class or module —\n"
    "a cohesive component with several methods/functions AND its own tests. NEVER a single\n"
    "function and never a one-liner (tiny items waste time on per-item overhead). When unsure,\n"
    "make pieces BIGGER: a whole class, a whole module, or a whole subsystem per item — err\n"
    "toward fewer, meatier pieces. Avoid one giant item that stragglers the batch, but a class\n"
    "is the minimum unit. Pick a split strategy: file (one module/class per item — PREFER THIS),\n"
    "function (several related functions grouped per item over one shared interface), or role\n"
    "(impl/tests/docs over one target).\n\n"
    "Output EXACTLY this envelope, one <<<PIECE>>> per work item, nothing after <<<END>>>\n"
    "(UPPERCASE = placeholders):\n\n"
    "<<<PLAN strategy=STRATEGY lang=LANG>>>\n"
    "shared:\n"
    "<blueprint>\n"
    "  the frozen interface every item relies on: types and signatures,\n"
    "  and a line  deps: [ allowed libraries, or [] for none ]\n"
    "</blueprint>\n"
    "<<<PIECE id=w1 exports=SYM1,SYM2>>>\n"
    "instruction: what this item must build (a sentence)\n"
    "<blueprint>\n"
    "  a COMPLETE spec for this item — as many lines as needed: signatures, behavior,\n"
    "  edge cases, and tests. Be thorough; the worker sees only this.\n"
    "</blueprint>\n"
    "<<</PIECE>>>\n"
    "<<<PIECE id=w2 exports=SYM>>>\n"
    "instruction: ...\n"
    "<blueprint> ... </blueprint>\n"
    "<<</PIECE>>>\n"
    "(... a few substantial PIECEs, each closed with <<</PIECE>>> so it can dispatch immediately ...)\n"
    "<<<END>>>\n\n"
    "Rules: items must be independent (no shared mutable state mid-flight); exported symbols\n"
    "must not collide; keep items SUBSTANTIAL and similar-sized. ORDER THEM LARGEST-FIRST — put\n"
    "the piece you expect to be the most code (the most involved component) first, so the longest\n"
    "job starts earliest and the lanes finish together (less idle tail). Emit the envelope only.";

// Boss system prompt + a tools-mode addendum (files on disk, dir structure, per-module tests,
// the final verifier). Used wherever the boss is prompted.
std::string boss_system_text() {
    std::string p = BOSS_PROMPT;
    if (g_tools)
        p += "\n\nTOOLS / FILES MODE: workers write REAL files to disk, exactly ONE file per piece. "
             "Plan a clean DIRECTORY STRUCTURE (state it in the shared block, e.g. src/ for modules). "
             "Emit one piece per MODULE (e.g. src/bird.js) plus a piece for the entry point "
             "(index.html). Do NOT write tests yourself: after the modules are built, the harness "
             "auto-generates a unit test for each module (given your file + the shared design) and a "
             "VERIFIER runs them — the job is DONE only when the tests pass. So make modules small, "
             "focused, and TESTABLE, with clear exports declared in the shared interface.";
    return p;
}

// lean worker preamble — the framework lives in the boss, NOT here (design §3/§5.2)
static const char *WORKER_PREAMBLE =
    "You implement ONE piece of a larger program. You are given a frozen shared interface and\n"
    "a spec for YOUR PIECE ONLY. Produce just that piece: the implementation (plus a quick\n"
    "inline test only if natural) for your declared exports, in the given language, and nothing\n"
    "else — no prose, no other functions, no re-declaring the shared interface. Match the\n"
    "declared signatures exactly. Output only code.";

// Tool-call instructions appended to the worker preamble when --tools is set. Format and verb
// names match nanocoder's XML convention (../nanocoder, source/app/prompts + source/tools):
// the tool name is the tag, parameters are nested tags.
std::string worker_preamble_text() {
    std::string p = WORKER_PREAMBLE;
    if (g_tools) {
        p += "\n\nYou MUST save your work to files with this tool (the tool NAME is the tag; each "
             "parameter is a nested tag):\n"
             "<create_file>\n<path>relative/path.ext</path>\n<content>\n...full file...\n</content>\n</create_file>\n"
             "- Write your module to its file, using the exact path/folders the shared interface specifies.\n"
             "- ALSO write a matching test file — " + test_directive() + ". A VERIFIER runs every test "
             "file at the end; your code must pass.\n"
             "Emit one create_file per file (module, then its test). RELATIVE paths only.";
        p += "\n\nIf you DISCOVER a concrete project fact others must know (e.g. the database engine, a "
             "runtime version, a required dependency), record it on its own line: `SET_VAR key=value` "
             "(e.g. `SET_VAR db_engine=mysql`). It is saved and shared with every other agent.";
    }
    return p;
}

// the common system turn (preamble + shared interface) — identical for every item in a job,
// so its rendered+tokenized form is the cacheable prefix (PA.2.1).
static std::string build_worker_system(const WorkOrder &wo) {
    return worker_preamble_text() + "\n\nShared interface (rely on these):\n" + wo.shared;
}
std::string build_prefix(const WorkOrder &wo) {
    return apply_chat_template({ {"system", build_worker_system(wo)} }, /*add_ass=*/false);
}

// a lane's full prompt: the common system turn + this item's user turn (the per-item delta)
std::string build_lane_prompt(const WorkOrder &wo, const Piece &p) {
    std::string user;
    if (p.is_boss) {                       // boss SELF lane: knows the workers' declared exports
        user += "The workers are separately producing these exports: ";
        bool first = true;
        for (auto &q : wo.pieces) {
            if (q.is_boss) continue;
            for (auto &e : q.exports) { user += (first ? "" : ", ") + e; first = false; }
        }
        user += ".\n\n";
    }
    user += "Your piece";
    if (!p.language.empty()) user += " (" + p.language + ")";
    user += ": " + p.instruction + "\n\nSpec:\n" + p.blueprint;
    std::string s = apply_chat_template({ {"system", build_worker_system(wo)}, {"user", user} }, /*add_ass=*/true);
    if (!g_worker_think) s += "<think>\n\n</think>\n\n";   // workers implement from spec, no reasoning
    return s;
}

// ───────────────────────── PA.1c: gather phase ───────────────────────────────
// After all workers finish, inject their outputs into the boss and let it
// produce a single merged artifact.

// Extract content from ```lang ... ``` fenced code blocks.
// If no fences found, return the full text unchanged (graceful fallback).
// If multiple fences, return the first one.
std::string extract_code_fence(const std::string &text) {
    size_t open = text.find("```");
    if (open == std::string::npos) return trim(text);   // no fence → raw code, pass through
    // skip the fence marker + optional lang tag to find newline
    size_t nl = text.find('\n', open + 3);
    if (nl == std::string::npos) nl = open + 3;
    else nl++;  // include the newline
    size_t close = text.find("```", nl);
    // Unclosed fence (e.g. truncated worker output): treat EOF as the close, per
    // markdown convention. Returning the content after the opening fence still
    // strips the ``` marker and any prose before it — more useful for the merge
    // than passing the raw fence through.
    if (close == std::string::npos) return trim(text.substr(nl));
    return trim(text.substr(nl, close - nl));
}

// Build the gather injection text: each worker's output wrapped in markers,
// followed by the gather instruction for the boss.
std::string build_gather_prompt(
    const WorkOrder &wo,
    const std::vector<std::pair<std::string, std::string>> &worker_results,  // id -> output
    const std::string &boss_self_output)
{
    std::string p;
    p += "Below are the actual outputs from your workers. Use these implementations AS-IS.\n\n";

    for (auto &wr : worker_results) {
        p += "<<<WORKER_DONE id=" + wr.first + ">>>";
        // extract clean code from fenced blocks if present
        p += extract_code_fence(wr.second);
        p += "\n<<<END_WORKER>>>\n\n";
    }

    if (!boss_self_output.empty()) {
        p += "<<<SELF_DONE id=boss>>>\n";
        p += extract_code_fence(boss_self_output);
        p += "\n<<<END_SELF>>>\n\n";
    }

    // declared exports for drift detection
    p += "<<<GATHER_INSTRUCTION>>>\n";
    p += "You have received the outputs from all workers. Your job is to produce a SINGLE, ";
    p += "COMPLETE, RUNNABLE artifact.\n\n";
    p += "For each worker above, you see their actual implementation. Use these implementations ";
    p += "AS-IS — do not rewrite them. Deduplicate overlapping helpers (keep one copy).\n\n";
    p += "Declared exports per worker for reference:\n";
    for (auto &pc : wo.pieces) {
        if (pc.is_boss) continue;
        p += "  " + pc.id + " -> [";
        for (size_t i = 0; i < pc.exports.size(); i++)
            p += (i ? "," : "") + pc.exports[i];
        p += "]\n";
    }

    p += "\nYour output must:\n";
    p += "1. Include the shared interface (types, signatures)\n";
    p += "2. Include every worker's implementation (dedup overlapping helpers — keep one copy)\n";
    p += "3. Include your integration/glue code, UPDATED to match actual worker signatures\n";
    p += "4. Add a dependency manifest if needed\n";
    p += "5. Be wrapped in a single code fence: ```<lang> ... ```\n\n";
    p += "Output ONLY the code fence, nothing else.\n";
    p += "<<<END_GATHER>>>";
    return p;
}

// Delta-prefill tokens into seq at position start_pos (the inject mechanism,
// same as PA.2.1 delta prefill). Returns the last logits index.
static bool inject(llama_seq_id seq, llama_pos start_pos,
                   const std::vector<llama_token> &toks, llama_batch &batch) {
    int n = (int)toks.size();
    for (int i0 = 0; i0 < n; i0 += PREFILL_CHUNK) {
        int nb = n - i0 < PREFILL_CHUNK ? n - i0 : PREFILL_CHUNK;
        batch_clear(&batch);
        for (int j = 0; j < nb; j++)
            batch_add(&batch, toks[i0 + j], (llama_pos)(start_pos + i0 + j), seq,
                      i0 + j == n - 1);
        if (llama_decode(G.ctx, batch) != 0) return false;
    }
    return true;
}

// System prompt for the gather pass — sets the boss's role for the merge. The
// user turn (built by build_gather_prompt) carries the actual instruction + bodies.
static const char *GATHER_SYSTEM =
    "You are the COORDINATOR of a packed-agent coding team. Your workers have finished "
    "their pieces in isolation; integrate them into one cohesive, runnable artifact.";

// Run the gather phase: clear the boss seq, prefill the gather turn, decode the
// merged artifact until EOG or max_gather. Returns the boss's gather output.
std::string run_gather_phase(llama_seq_id boss_seq,
                             const std::string &gather_content, int max_gather)
{
    // Fresh boss context for the merge. The seq's KV still holds leftovers — the
    // boss's own plan (non-stream) or the last pool item that ran on this lane
    // (streaming). Prefilling the gather turn on top would write KV cells at
    // positions that already have cells in this seq, duplicating them and
    // corrupting attention. Clear the seq and start the gather turn at pos 0.
    llama_memory_seq_rm(G.mem, boss_seq, 0, -1);

    // Wrap as a chat turn so the instruct model *answers* the merge request rather
    // than continuing raw text (matches boss_generate / the worker prompts).
    std::string prompt = apply_chat_template(
        { {"system", GATHER_SYSTEM}, {"user", gather_content} }, /*add_ass=*/true);
    if (!g_boss_think) prompt += "<think>\n\n</think>\n\n";   // gather is a boss-role turn

    std::vector<llama_token> gt(prompt.size() + 16, 0);
    int gn = llama_tokenize(G.vocab, prompt.c_str(), (int32_t)prompt.size(),
                            gt.data(), (int32_t)gt.size(), true, true);
    if (gn <= 0) return "[gather tokenize failed]";
    gt.resize(gn);

    llama_batch batch = llama_batch_init(PREFILL_CHUNK + 8, 0, 1);
    if (!inject(boss_seq, 0, gt, batch)) {
        llama_batch_free(batch);
        return "[gather inject failed]";
    }

    llama_sampler *s = g_greedy ? nullptr : make_sampler(g_seed, g_boss_sp);   // gather sampler (Qwen)
    // inject() prefills in PREFILL_CHUNK batches → the last token's logits live at the end of the
    // final batch, NOT at global index gn-1. Use -1 (last output); gn-1 overflows for gn>chunk.
    llama_token tok = pick(s, -1);
    llama_pos pos = (llama_pos)gn;
    std::string out;

    fprintf(stderr, "\n── PA.1c GATHER — boss merging artifact ──\n");
    for (int gen = 0; gen < max_gather && !llama_vocab_is_eog(G.vocab, tok); gen++) {
        std::string piece = tok_str(tok);
        out += piece;
        fputs(piece.c_str(), stderr);
        batch_clear(&batch);
        batch_add(&batch, tok, pos, boss_seq, true);
        if (llama_decode(G.ctx, batch) != 0) break;
        tok = pick(s, 0);
        pos++;
    }
    if (s) llama_sampler_free(s);
    llama_batch_free(batch);
    return out;
}

// Write the final artifact to --out (if set) or stdout.
static void write_artifact(const std::string &artifact) {
    if (g_out_path[0]) {
        FILE *fp = fopen(g_out_path, "w");
        if (fp) { fputs(artifact.c_str(), fp); fclose(fp);
            fprintf(stderr, "\n── artifact written to %s (%zu chars) ──\n", g_out_path, artifact.size());
        } else { perror("fopen --out"); }
    } else {
        printf("\n════════════════════════════════════════════════\n");
        printf("── FINAL ARTIFACT (%zu chars) ──\n%s\n════════════════════════════════════════════════\n",
               artifact.size(), artifact.c_str());
    }
}

// Drive the gather→write step shared by both pipelines: build the gather prompt,
// run the boss merge, strip the fence, emit the artifact. n_lanes is the worker
// lane count (n_seq_max = n_lanes + 1, the +1 being the prefix-cache seq).
void finish_gather(const WorkOrder &wo,
                   const std::vector<std::pair<std::string,std::string>> &worker_results,
                   const std::string &boss_self_output, int n_lanes) {
    std::string gprompt    = build_gather_prompt(wo, worker_results, boss_self_output);
    if (g_tools) {
        fprintf(stderr, "\n── PA.5 TOOLS — executing worker tool calls (work-dir: %s, run %s) ──\n",
                g_work_dir, g_allow_run ? "ENABLED" : "disabled");
        std::string report = run_worker_tools(worker_results);
        if (!report.empty())
            gprompt += "\n\n<<<TOOL_RESULTS>>>\nFiles written / commands run by the workers "
                       "(sandboxed in '" + std::string(g_work_dir) + "'):\n" + report + "<<<END_TOOL_RESULTS>>>\n";
    }
    int total_ctx  = (int)llama_n_ctx(G.ctx);
    int max_gather = (total_ctx / (n_lanes + 1)) / 2;   // generous cap for the merge
    if (max_gather < 512) max_gather = 512;
    std::string gather_out = run_gather_phase(0, gprompt, max_gather);
    std::string artifact   = extract_code_fence(gather_out);
    if (artifact.empty()) artifact = trim(gather_out);
    write_artifact(artifact);
}
