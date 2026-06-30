/*
 * pti_agents.h — public header for the pti_agents split
 *
 * All struct definitions, extern global declarations, and cross-file
 * function declarations. Each .cpp in src/agents/ includes ONLY this file
 * (plus the system headers below).
 *
 * Contract rule: a function is declared here iff it is called across file
 * boundaries. Signatures must match the definitions verbatim (which are in
 * turn faithful to the original pti_agents.cpp monolith). File-internal
 * helpers stay `static` in their .cpp and do NOT appear here.
 */

#include "../../llama.cpp/include/llama.h"
#include "../../llama.cpp/src/llama-ext.h"   // pre-norm hidden access for MTP (PA.3)

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <filesystem>
#include <map>
#include <regex>
#include <string>
#include <sys/wait.h>
#include <vector>

#define MAX_STREAMS   16
#define PREFILL_CHUNK 1024
#define MAX_TOKENS    32768

// ═══════════════════════════════════════════════════════════════════════════════
// Structs
// ═══════════════════════════════════════════════════════════════════════════════

struct SParams { float temp; float top_p; float min_p; float presence; int top_k; };

struct Stream {
    std::vector<llama_token> prompt_toks;
    std::string  text;        // generated output
    std::vector<llama_token> gen_toks;  // generated tokens (solo-vs-packed identity gate)
    llama_token  tok_last = 0;
    llama_pos    pos      = 0;
    int          n_gen    = 0;
    bool         live     = false;
};

struct Globals {
    llama_model       *model   = nullptr;
    const llama_vocab *vocab   = nullptr;
    llama_context     *ctx     = nullptr;
    llama_memory_t     mem     = nullptr;
    int32_t            n_vocab = 0;
    const char        *tmpl    = nullptr;   // chat template (PA.1 boss prompting)
    llama_context     *ctx_mtp = nullptr;   // PA.3: nextn-head draft context (--mtp)
    int32_t            n_embd  = 0;         // model hidden size (MTP embd feed)
} extern G;

struct MtpVerdict { int e; int acc; bool full; bool stop; };

struct Piece {
    std::string id;
    std::vector<std::string> exports;
    std::string language;
    std::string instruction;
    std::string blueprint;
    bool        is_boss = false;
};

struct WorkOrder {
    std::string strategy;
    std::string shared;             // frozen interface, prepended to every lane
    std::vector<Piece> pieces;      // workers, then the boss SELF piece
    bool        ok = false;
    std::string error;
};

struct Msg { std::string role, content; };

struct ToolCall {
    std::string name;
    std::vector<std::pair<std::string,std::string>> params;
};

struct StackProfile {
    std::string lang;         // "javascript"
    std::string src_ext;      // ".js"       — a module file is <comp> + src_ext
    std::string test_suffix;  // ".test.js"  — a test file is <comp> + test_suffix (the harness controls naming)
    std::string runner;       // "node"      — a test runs as: runner '<relpath>'
    std::string module_hint;  // PA.8 Layer 2: the export/module-system convention (prompt-templated)
    std::string test_hint;    // PA.8 Layer 2: the assertion-style + pass/fail convention (prompt-templated)
};

// PA.7 eager scheduling — EKind must precede EItem (EItem.kind is typed EKind).
// Order matches the monolith: priority/mode logic switches on the names, but the
// numeric values feed the makespan sim, so the order is part of the contract.
enum EKind { E_DESIGN = 0, E_IMPL, E_TESTGEN, E_VERIFY, E_REWORK };

struct EItem {
    EKind       kind;
    std::string comp;          // component id (bird, pipes, …)
    int         cost = 1;      // work units (≈ tokens) — for the makespan sim
    bool        done = false;
    bool        dispatched = false;
};

struct EagerDAG {
    std::vector<EItem> items;
    std::vector<std::string> components;
};

// PA.4 repair verdict — defined in pti_eager.cpp (repair_verdict), read in pti_verify.cpp.
enum RepairAction { RA_DONE, RA_REPAIR, RA_GIVEUP };

// ═══════════════════════════════════════════════════════════════════════════════
// Global variables (defined in pti_common.cpp)
// ═══════════════════════════════════════════════════════════════════════════════

extern bool     g_verbose_logs;
extern char     g_out_path[512];
extern bool     g_no_gather;
extern bool     g_tools;
extern bool     g_allow_run;
extern char     g_work_dir[512];
extern bool     g_mtp;
extern SParams  g_boss_sp;
extern SParams  g_worker_sp;
extern bool     g_boss_think;
extern bool     g_worker_think;
extern float    g_temp;
extern uint32_t g_seed;
extern bool     g_general;
extern bool     g_greedy;
extern int      g_repair_budget;
extern StackProfile g_stack;          // defined in pti_memory.cpp
extern char     g_blackboard[512];    // defined in pti_memory.cpp
extern std::map<std::string,std::string> g_vars;   // defined in pti_memory.cpp
// NOTE: n_ctx and g_lang_explicit are main()-local (pti_main.cpp), not globals.

// ═══════════════════════════════════════════════════════════════════════════════
// Function declarations (cross-file only)
// ═══════════════════════════════════════════════════════════════════════════════

// ── pti_common.cpp ──────────────────────────────────────────────────────────
double  now_sec();
std::string tok_str(llama_token t);
void    batch_add(struct llama_batch *b, llama_token tok, llama_pos pos, llama_seq_id seq, bool want_logits);
void    batch_clear(struct llama_batch *b);
int32_t argmax_f(const float *v, int32_t n);
void    pti_log_cb(enum ggml_log_level level, const char *text, void *);

// ── pti_stream.cpp ──────────────────────────────────────────────────────────
bool prefill_stream(Stream &st, llama_seq_id seq, llama_batch &batch, int *last_idx);
int  run_streams(std::vector<Stream> &streams, int max_new, bool packed,
                 double *wall_out, double *pf_out);
void init_sampler(std::vector<llama_sampler *> &smpl, int n_lanes);
void free_sampler(std::vector<llama_sampler *> &smpl);
MtpVerdict mtp_verify(bool has_draft, llama_token draft,
                      llama_token a0, bool a0_eog,
                      llama_token a1, bool a1_eog);
int  mtp_ckpt_seq(int lane, int n_lanes);
int  mtp_base_seq(int n_lanes);
int  mtp_seqmax(int n_lanes, bool mtp_on);

// ── pti_mtp.cpp — MTP drafting + the PA.2 work-pool ──────────────────────────
bool setup_mtp(int n_lanes);
llama_sampler *make_sampler(uint32_t seed, const SParams &sp);
llama_token pick(llama_sampler *s, int idx);
SParams qwen_params(bool think, bool general);
void run_pool(const std::vector<std::string> &items, int n_lanes, int max_new,
              std::vector<std::string> *out_opt = nullptr, const std::string &prefix = "");

// ── pti_parser.cpp ──────────────────────────────────────────────────────────
std::string trim(const std::string &s);
std::vector<std::string> split_csv(const std::string &s);
bool parse_marker(const std::string &line, std::string &tag,
                  std::vector<std::pair<std::string,std::string>> &attrs);
WorkOrder parse_work_order(const std::string &text);
std::string attr_get(const std::vector<std::pair<std::string,std::string>> &a,
                     const std::string &k, const std::string &dflt = "");
size_t match_paren(const std::string &s, size_t open);
std::vector<std::string> split_top_commas(const std::string &s);
std::string extract_instruction(const std::string &text);
std::string extract_blueprint(const std::string &text);

// ── pti_prompt.cpp ──────────────────────────────────────────────────────────
std::string boss_system_text();
std::string worker_system_text();
std::string worker_preamble_text();
std::string build_prefix(const WorkOrder &wo);
std::string build_lane_prompt(const WorkOrder &wo, const Piece &p);
std::string build_gather_prompt(const WorkOrder &wo,
                                const std::vector<std::pair<std::string,std::string>> &worker_results,
                                const std::string &boss_self_output);
std::string apply_chat_template(const std::vector<Msg> &msgs, bool add_ass = true);
std::string boss_generate(const std::string &prompt, int max_tok);
std::string extract_code_fence(const std::string &text);
void print_work_order(const WorkOrder &wo);
std::string var_or(const std::string &key, const std::string &fallback);
std::string test_directive();
std::string test_user(const std::string &module, const std::string &code,
                      const std::string &blueprint, const std::string &collaborators);

// ── pti_tools.cpp ───────────────────────────────────────────────────────────
std::vector<ToolCall> parse_tool_calls(const std::string &text);
bool execute_tool_call(const ToolCall &tc);
std::string apply_tool_calls(const std::string &text);
std::string tc_param(const ToolCall &c, const std::string &k);
bool is_dangerous_cmd(const std::string &cmd);
std::string resolve_read(const std::string &work_dir, const std::string &rel);
bool pa7_is_notfound(const std::string &r);
std::string build_blocked_requeue(const std::string &orig_prompt, const std::string &reason,
                                  const std::string &resolved);

// ── pti_memory.cpp ──────────────────────────────────────────────────────────
void    ensure_blackboard();
std::string read_blackboard();
void    write_blackboard(const std::string &data);
void    set_var(const std::string &key, const std::string &value);
std::vector<std::string> parse_set_var(const std::string &text);
StackProfile stack_profile(std::string name);
bool has_suffix(const std::string &s, const std::string &suf);
std::string module_for_test(const std::string &test_filename, const std::vector<std::string> &modules);
bool is_integration_test(const std::string &test_file, const std::vector<std::string> &all_modules);
std::string integration_hint(const std::string &module, const std::vector<std::string> &all_modules, const StackProfile &sp);
void finish_gather(const WorkOrder &wo,
                   const std::vector<std::pair<std::string,std::string>> &worker_results,
                   const std::string &boss_self_output, int n_streams);

// ── pti_memory.cpp — file helpers (PA.8 stack-aware naming) ──────────────────
bool is_test_file(const std::string &name);
bool is_src_file(const std::string &name);
std::string test_file_name(const std::string &comp);
std::string src_file_name(const std::string &comp);
std::string comp_key(const std::string &path);
std::string lang_from_task(const std::string &task);
std::string lang_from_marker(const std::string &marker);
std::string run_test_cmd(const std::string &test_rel);
std::string blackboard_dir();
std::string vars_path();
void vars_save();
void vars_load();
void vars_seed_from_stack();
std::map<std::string,std::string> vars_parse(const std::string &text);
std::string vars_render();
int vars_absorb(const std::string &text);
std::string line_escape(const std::string &s);
bool tool_call_truncated(const std::string &text);
std::string run_worker_tools(const std::vector<std::pair<std::string,std::string>> &worker_results);
std::string detect_lang_in_dir(const std::string &dir);

// ── pti_verify.cpp ──────────────────────────────────────────────────────────
std::string build_test_prompt(const std::string &module, const std::string &code,
                              const StackProfile &sp);
std::string build_repair_prompt(const std::string &module, const std::string &code,
                                const std::string &test, const std::string &error,
                                const std::string &journal, const StackProfile &sp);
std::string build_boss_arbiter_prompt();
std::vector<std::string> finalize_test_results(const std::string &output);
bool test_is_passing(const std::string &output);
std::string test_summary(const std::vector<std::string> &results);
bool has_contradictory_tests(const std::vector<std::string> &tests);
std::string build_integration_test_prompt(const std::string &module,
                                          const std::vector<std::string> &all_modules,
                                          const StackProfile &sp);
void finalize_verify(const WorkOrder &wo,
                     const std::vector<std::pair<std::string,std::string>> &worker_results,
                     int n_lanes, int max_new);

// ── pti_eager.cpp — PA.7 scheduling internals are file-local; these PA.4 repair
//    / integration helpers + reconcile sizing are shared with verify/pipeline ─
std::vector<std::string> contradictory_asserts(const std::string &src);
std::string frame_executed_truth(const std::string &err);
std::string extract_attempt(const std::string &out);
RepairAction repair_verdict(int round, int budget, int n_failed);
std::vector<std::string> module_refs(const std::string &content, const std::string &self_base,
                                     const std::vector<std::string> &all_bases);
std::string build_integration_task(const std::string &goal, const std::string &target,
                                   const std::string &target_code, const std::string &sibling_code);
std::string integration_base(const std::string &path);
std::vector<std::pair<int,int>> reconcile_groups(int n, int g);
int reconcile_parallel_g(int n, int lanes);
std::string build_amend_user(const std::string &base, const std::string &modpath,
                             const std::string &modcontent, const std::string &testcontent,
                             const std::string &error, const std::string &spec,
                             const std::string &collaborators, const std::string &journal);
std::string build_arbiter_user(const std::string &contract, const std::string &failblock);
std::string build_rework_user(const std::string &target, const std::string &spec,
                              const std::string &modcontent, const std::string &testcontent,
                              const std::string &error, const std::string &guidance,
                              const std::string &collaborators, const std::string &journal);
std::vector<std::pair<std::string,std::string>> reworks_from_plan(const std::string &raw);

// ── pti_verify.cpp — PA.6 shared helpers (read/strip/stage) ──────────────────
std::string read_file_str(const std::string &path);
std::string strip_think(const std::string &in);
std::string stage_prefix(const std::string &shared);
std::string stage_item(const std::string &shared, const std::string &user);

// ── pti_pipeline.cpp ────────────────────────────────────────────────────────
void run_pipeline(const std::string &task, int max_new, int n_lanes);
void run_pipeline_staged(const std::string &task, int n_lanes, int max_new);

// ── self-tests (defined in their domain files, dispatched from pti_main.cpp) ─
int parse_self_test();    // pti_prompt.cpp
int gather_self_test();   // pti_verify.cpp
int mtp_self_test();      // pti_verify.cpp
int coord_self_test();    // pti_eager.cpp
int eager_self_test();    // pti_eager.cpp
