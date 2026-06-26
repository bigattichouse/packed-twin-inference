# SEPARATION OF CONCERNS — Full Repo Reorganization Plan

> **Input**: 20 source files at repo root, `pti_agents.cpp` is a 3,337-line monolith
> **Output**: 13 domain files in `src/agents/`, all other sources in `src/<module>/`, clean Makefile
> **Goal**: Every source file in `src/`, logically grouped, each ≤700 lines. All existing `make` targets work identically.
> **Constraint**: After merge, every `make` target and run shortcut produces the same binary with identical CLI and behavior.

---

## Target Directory Layout

```
packed-twin-inference/
├── Makefile                          ← rewritten, all paths updated
├── bin/                              ← binaries (unchanged output location)
│   ├── pti_test
│   ├── pti_test_cuda
│   ├── libpti.so
│   ├── pti_llama
│   ├── pti_mtp
│   ├── pti_2seq
│   ├── pti_4seq
│   ├── pti_agents
│   ├── pti_debug
│   ├── pti_gemv_bench
│   ├── pti_kbatch_bench
│   ├── pti_q6k_bench
│   ├── pti_lookup
│   ├── pti_chat
│   ├── pti_mtp_probe
│   └── pti_server
├── src/
│   ├── agents/                       ← pti_agents.cpp split into 13 files
│   │   ├── pti_agents.h              ← public header (structs, extern, declarations)
│   │   ├── pti_common.cpp            ← globals, utilities (lines 1–122)
│   │   ├── pti_stream.cpp            ← prefill, decode, sampling (130–306)
│   │   ├── pti_mtp.cpp               ← MTP drafting (307–419)
│   │   ├── pti_pool.cpp              ← work-queue refill (420–582)
│   │   ├── pti_parser.cpp            ← work-order envelope parser (583–739)
│   │   ├── pti_prompt.cpp            ← system prompts, chat template (740–1079)
│   │   ├── pti_tools.cpp             ← tool calls, sandbox (1080–1200)
│   │   ├── pti_memory.cpp            ← blackboard, vars, stack profile (1201–1550)
│   │   ├── pti_eager.cpp             ← eager scheduling PA.7 (1551–2000)
│   │   ├── pti_verify.cpp            ← verify→repair PA.4 (2001–2349)
│   │   ├── pti_pipeline.cpp          ← pipeline orchestration (2350–3050)
│   │   └── pti_main.cpp              ← CLI, main(), PA.0 demo (3051–3337)
│   ├── pti/                          ← speculative decoding binaries
│   │   ├── pti_llama.c               ← 2-seq PTI (C)
│   │   ├── pti_2seq.cpp              ← 2-seq PTI (C++)
│   │   ├── pti_mtp.cpp               ← 3-seq + MTP re-init
│   │   ├── pti_4seq.cpp              ← 4-seq PTI (headline result)
│   │   ├── pti_lookup.cpp            ← n-gram lookup speculative decode
│   │   └── pti_mtp_probe.cpp         ← MTP head semantics probe
│   ├── server/                       ← HTTP server, chat CLI
│   │   ├── pti_server.cpp            ← OpenAI-compatible HTTP server
│   │   └── pti_chat.cpp              ← interactive chat CLI
│   ├── bench/                        ← benchmarks, debug tools
│   │   ├── pti_debug.cpp             ← single-sequence debug
│   │   ├── pti_gemv_bench.cpp        ← GEMV fusion benchmark
│   │   └── pti_kbatch_bench.cpp      ← k-token batch cost curve
│   ├── kernel/                       ← GPU kernels
│   │   ├── pti_kernel.hip            ← HIP kernel (bandwidth)
│   │   └── pti_q6k_bench.hip         ← Q6_K MMVQ microbench
│   ├── scripts/                      ← shell scripts
│   │   ├── pti-cli.sh                ← CLI wrapper
│   │   ├── llama-server-pti.sh       ← server launcher
│   │   ├── audit.sh                  ← correctness + performance audit
│   │   ├── flappy_validate.sh        ← validation script
│   │   ├── scale_validate.sh         ← scale validation
│   │   └── seq_vs_packed_ab.sh       ← A/B comparison
│   └── python/                       ← Python scripts
│       ├── pti_hip.py                ← HIP kernel Python bridge
│       ├── infer.py                  ← inference script
│       ├── pack.py                   ← packing script
│       ├── benchmark.py              ← benchmark runner
│       ├── benchmark_comparison.py   ← comparison analysis
│       └── svg_test.py               ← SVG test
├── llama.cpp/                        ← unchanged (external dependency)
├── llama.cpp-files/                  ← unchanged
├── llama.cpp-patch/                  ← unchanged
├── spec/                             ← unchanged
├── validate/                         ← unchanged
├── .clangd                           ← may need path update
├── .gitignore                        ← add src/ exclusion patterns if needed
└── [docs, diagrams, .md files]       ← unchanged at root
```

---

## File-by-File Move Map

### `src/agents/` — pti_agents.cpp split (13 files)

| File | Source Lines | ~Lines | Domain |
|------|-------------|--------|--------|
| `pti_agents.h` | (new) | ~200 | Public header: structs, extern globals, function declarations |
| `pti_common.cpp` | 1–122 | ~120 | Globals, utilities, log callback, batch helpers |
| `pti_stream.cpp` | 130–306 | ~180 | Prefill, decode loop, sampler init/cleanup |
| `pti_mtp.cpp` | 307–419 | ~110 | MTP drafting, checkpoint/rollback |
| `pti_pool.cpp` | 420–582 | ~160 | Work-queue refill, prefix cache |
| `pti_parser.cpp` | 583–739 | ~160 | `<<<` envelope parser, WorkOrder |
| `pti_prompt.cpp` | 740–1079 | ~340 | System prompts, chat template, boss_generate |
| `pti_tools.cpp` | 1080–1200 | ~120 | Tool calls, sandbox execution |
| `pti_memory.cpp` | 1201–1550 | ~350 | Blackboard, vars, stack profile |
| `pti_eager.cpp` | 1551–2000 | ~450 | Eager scheduling DAG, PA.7 |
| `pti_verify.cpp` | 2001–2349 | ~350 | Verify→repair loop, L1/L2 repair |
| `pti_pipeline.cpp` | 2350–3050 | ~700 | Streaming + staged pipeline orchestration |
| `pti_main.cpp` | 3051–3337 | ~290 | CLI, main(), self-tests, PA.0 demo |

### `src/pti/` — speculative decoding (move as-is)

| File | Move From | Lines | Notes |
|------|-----------|-------|-------|
| `pti_llama.c` | `pti_llama.c` | ~500 | C file, simplest PTI prototype |
| `pti_2seq.cpp` | `pti_2seq.cpp` | ~350 | 2-seq, best measured throughput |
| `pti_mtp.cpp` | `pti_mtp.cpp` | ~650 | 3-seq + MTP re-init |
| `pti_4seq.cpp` | `pti_4seq.cpp` | ~700 | 4-seq headline result |
| `pti_lookup.cpp` | `pti_lookup.cpp` | ~900 | n-gram lookup speculative decode |
| `pti_mtp_probe.cpp` | `pti_mtp_probe.cpp` | ~280 | MTP head semantics probe |

### `src/server/` — HTTP server + chat CLI (move as-is)

| File | Move From | Lines | Notes |
|------|-----------|-------|-------|
| `pti_server.cpp` | `pti_server.cpp` | ~1,600 | OpenAI-compatible HTTP server. Has vendor includes (`cpp-httplib`, `nlohmann/json`, `chat.h`) |
| `pti_chat.cpp` | `pti_chat.cpp` | ~600 | Interactive chat CLI, mode-switchable |

### `src/bench/` — benchmarks + debug (move as-is)

| File | Move From | Lines | Notes |
|------|-----------|-------|-------|
| `pti_debug.cpp` | `pti_debug.cpp` | ~250 | Single-sequence debug |
| `pti_gemv_bench.cpp` | `pti_gemv_bench.cpp` | ~380 | GEMV fusion benchmark |
| `pti_kbatch_bench.cpp` | `pti_kbatch_bench.cpp` | ~290 | k-token batch cost curve |

### `src/kernel/` — GPU kernels (move as-is)

| File | Move From | Lines | Compiler |
|------|-----------|-------|----------|
| `pti_kernel.hip` | `pti_kernel.hip` | ~1,700 | `hipcc` (AMD ROCm) |
| `pti_q6k_bench.hip` | `pti_q6k_bench.hip` | ~1,050 | `hipcc` (AMD ROCm) |

### `src/scripts/` — shell scripts (move as-is)

| File | Move From |
|------|-----------|
| `pti-cli.sh` | `pti-cli.sh` |
| `llama-server-pti.sh` | `llama-server-pti.sh` |
| `audit.sh` | `audit.sh` |
| `flappy_validate.sh` | `flappy_validate.sh` |
| `scale_validate.sh` | `scale_validate.sh` |
| `seq_vs_packed_ab.sh` | `seq_vs_packed_ab.sh` |

### `src/python/` — Python scripts (move as-is)

| File | Move From |
|------|-----------|
| `pti_hip.py` | `pti_hip.py` |
| `infer.py` | `infer.py` |
| `pack.py` | `pack.py` |
| `benchmark.py` | `benchmark.py` |
| `benchmark_comparison.py` | `benchmark_comparison.py` |
| `svg_test.py` | `svg_test.py` |

---

## Section-by-Section Breakdown (pti_agents.cpp → src/agents/)

### 0. `src/agents/pti_agents.h` — single public header (new)

Everything the other `.cpp` files need to see. No implementation.

**Contents (in order):**
- `#include` guards for llama, ggml, stdlib headers (lines 20–34 of original)
- `#define MAX_STREAMS / PREFILL_CHUNK / MAX_TOKENS` (lines 36–38)
- `struct SParams` (line 57)
- `struct Stream` (lines 103–111)
- `struct Globals` (lines 113–122)
- `struct WorkOrder` and `struct Piece` (lines 584–593)
- `struct RepairJournal` (from repair bookkeeping section)
- `struct RepairResult`
- `struct AgentStack` (from stack profile §9)
- `struct ToolCall` (from PA.5)
- `struct Blueprint` (from design pipeline)
- `struct EagerDAG / EagerNode` (from PA.7)
- **External declarations** for all globals currently at file scope (lines 47–66): `g_verbose_logs`, `g_out_path`, `g_no_gather`, `g_tools`, `g_allow_run`, `g_work_dir`, `g_mtp`, `g_boss_sp`, `g_worker_sp`, `g_boss_think`, `g_worker_think`, `g_temp`, `g_seed`, `g_general`, `g_greedy`, `g_repair_budget`, `G` (singleton)
- **Function declarations** — every function that crosses file boundaries

---

### 1. `src/agents/pti_common.cpp` — low-level utilities and globals

**Source lines**: 1–122

**Moves here:**
- File header comment (lines 1–18)
- All global variables (lines 47–66)
- `static void pti_log_cb()` (lines 68–78)
- `static constexpr float ARGMAX_EPS` (line 80)
- `static int32_t argmax_f()` (lines 81–89)
- `static void batch_add()` (lines 91–99)
- `static void batch_clear()` (line 101)
- `static std::string tok_str()` (lines 124–129)

**Exposes**: `now_sec()`, `tok_str()`, `batch_add()`, `batch_clear()`, `argmax_f()`, `G`, all globals

---

### 2. `src/agents/pti_stream.cpp` — stream lifecycle: prefill, decode, sampling

**Source lines**: 130–306

**Moves here:**
- `static bool prefill_stream()` (line 131)
- `static void init_sampler()` (from sampler initialization block)
- `static void free_sampler()` (from sampler cleanup block)
- `static int run_streams()` — packed and sequential decode loop
- `static int32_t mtp_ckpt_seq()` (checkpoint seq ID helper)

**Dependencies**: `pti_common.cpp` (Globals, Stream, batch_add, batch_clear, tok_str)
**Exposes**: `prefill_stream()`, `init_sampler()`, `free_sampler()`, `run_streams()`

---

### 3. `src/agents/pti_mtp.cpp` — MTP speculative drafting (PA.3)

**Source lines**: 307–419

**Moves here:**
- MTP context setup (`G.ctx_mtp` initialization)
- `static int32_t mtp_draft()` — single-lane MTP draft
- `static std::vector<std::pair<int,llama_token>> mtp_draft_lane()` — per-lane draft with rollback
- Checkpoint management: `mtp_checkpoint()`, `mtp_rollback()`
- MTP sample extraction from pre-norm hidden states

**Dependencies**: `pti_common.cpp`, `pti_stream.cpp`
**Exposes**: MTP context init, `mtp_draft_lane()`, checkpoint/rollback

---

### 4. `src/agents/pti_pool.cpp` — work-queue refill and pool decode

**Source lines**: 420–582

**Moves here:**
- `static bool run_pool()` — M items over N lanes with refill (plain path)
- `static bool run_pool_mtp()` — M items over N lanes with refill (MTP speculative path)
- Prefix cache implementation (PA.2.1): base seq clone, delta prefill
- Pool worker start/finish tracking (`start_worker()`, lane assignment loop)

**Dependencies**: `pti_common.cpp`, `pti_stream.cpp`, `pti_mtp.cpp`
**Exposes**: `run_pool()`, `run_pool_mtp()`

---

### 5. `src/agents/pti_parser.cpp` — work-order envelope parsing

**Source lines**: 583–739

**Moves here:**
- `static std::vector<std::string> split_sections()` — `<<<` delimited parser
- `static WorkOrder parse_work_order()` — main parser with fallback recovery
- `static std::string extract_code_fence()` — code-fence extraction
- Export collision detection
- `static void print_work_order()` — stderr display

**Dependencies**: `pti_common.cpp` (tok_str)
**Exposes**: `parse_work_order()`, `extract_code_fence()`, `print_work_order()`

---

### 6. `src/agents/pti_prompt.cpp` — system prompts and chat template

**Source lines**: 740–1079

**Moves here:**
- `static std::string boss_system_text()` / `worker_system_text()`
- `static std::string build_prefix()` / `build_lane_prompt()` / `build_gather_prompt()`
- `static std::string apply_chat_template()` — chat template application
- All `static const char*` prompt strings (boss plan, worker implement, gather merge, etc.)
- `static std::string boss_generate()` — boss generation wrapper

**Dependencies**: `pti_common.cpp`, `pti_parser.cpp`
**Exposes**: All build_* and *_system_text functions, `apply_chat_template()`, `boss_generate()`

---

### 7. `src/agents/pti_tools.cpp` — tool call parsing and execution (PA.5)

**Source lines**: 1080–1200

**Moves here:**
- `static std::vector<ToolCall> parse_tool_calls()` — XML tool-call extraction
- `static bool execute_tool_call()` — dispatch: create_file, execute_bash, read_file, abandon
- Destructive-command guard (blocklist)
- Sandbox enforcement (`--work-dir` boundary), 60s timeout
- `static std::string apply_tool_calls()` — apply parsed calls, return results

**Dependencies**: `pti_common.cpp` (g_work_dir, g_allow_run, g_tools)
**Exposes**: `parse_tool_calls()`, `execute_tool_call()`, `apply_tool_calls()`

---

### 8. `src/agents/pti_memory.cpp` — blackboard / variable store / stack profile (PA.8 §9)

**Source lines**: 1201–1550

**Moves here:**
- Blackboard: `ensure_blackboard()`, `read_blackboard()`, `write_blackboard()`
- Variable store: `set_var()`, `var_or()`, `parse_set_var()`
- Stack profile (§9): `lang_from_task()`, `lang_from_marker()`
- Integration test helpers: `module_for_test()`, `is_integration_test()`, `integration_hint()`

**Dependencies**: `pti_common.cpp` (g_work_dir)
**Exposes**: All blackboard/variable functions, stack profile functions, integration test helpers

---

### 9. `src/agents/pti_eager.cpp` — eager scheduling (PA.7) — core built, not yet integrated

**Source lines**: 1551–2000

**Moves here:**
- `eager_build_dag()` — construct dependency graph from WorkOrder
- `eager_ready()` — dependency gating per component
- `eager_next_batch()` — compute next batch of ready lanes
- `eager_simulate()` — makespan simulation (proves 18% faster vs barriers)
- `eager_all_done()` — completion check
- Self-test harness for PA.7 primitives (8 tests)

**Dependencies**: `pti_parser.cpp` (WorkOrder, Piece)
**Exposes**: All eager_* functions, `EagerDAG`, `EagerNode`

**Note**: Currently not wired into `run_pool` or `run_pipeline_staged`. Exposed for future integration.

---

### 10. `src/agents/pti_verify.cpp` — verify→repair loop (PA.4)

**Source lines**: 2001–2349

**Moves here:**
- `build_test_prompt()` / `build_repair_prompt()` / `build_boss_arbiter_prompt()`
- `finalize_test_results()` / `test_is_passing()` / `test_summary()`
- `escalate_to_boss()` — L1→L2 escalation
- `has_contradictory_tests()` — §4.6 contradiction lint
- `build_integration_test_prompt()` — §4.7 integration test gen
- `finalize_verify()` — main verify→repair orchestrator:
  1. Store modules to disk
  2. Generate tests per module
  3. Run tests
  4. L1 repair loop (up to `g_repair_budget` rounds)
  5. L2 boss arbiter escalation (multi-round, RESPEC-on-stuck)
  6. Write repair journal

**Dependencies**: `pti_common.cpp`, `pti_parser.cpp`, `pti_tools.cpp`, `pti_memory.cpp`, `pti_pool.cpp`
**Exposes**: `finalize_verify()`, `build_test_prompt()`, `build_repair_prompt()`, test parsing, escalation

---

### 11. `src/agents/pti_pipeline.cpp` — pipeline orchestration (streaming + staged)

**Source lines**: 2350–3050

**Moves here:**

**Streaming pipeline** (`run_pipeline`):
- Boss plans in a dedicated lane
- Workers execute from a pool, refilled as pieces close in the boss's live stream
- After `<<<END>>>`, boss joins the worker pool
- Gather phase merges outputs via `finish_gather()`

**Staged pipeline** (`run_pipeline_staged`):
- **Triage** → **Design** → **Reconcile** → **Implement** → **Test-gen + Verify**

**Dependencies**: `pti_common.cpp`, `pti_parser.cpp`, `pti_prompt.cpp`, `pti_tools.cpp`, `pti_memory.cpp`, `pti_pool.cpp`, `pti_verify.cpp`
**Exposes**: `run_pipeline()`, `run_pipeline_staged()`, `finish_gather()`

---

### 12. `src/agents/pti_main.cpp` — entrypoint, CLI, PA.0 demo, main()

**Source lines**: 3051–3337

**Moves here:**
- `static void usage()` — help text
- Self-test entry points: `--parse-test`, `--gather-test`, `--mtp-test`, `--coord-test`, `--eager-test`
- `main()` — CLI argument parsing, model loading, MTP context setup, dispatch

**Dependencies**: Everything (top-level orchestrator)
**Exposes**: `main()`

---

## Cross-Reference: Agent File Dependency DAG

```
pti_agents.h
  ↑
pti_common.cpp          ← pti_agents.h
  ↑
pti_stream.cpp          ← pti_common.cpp, pti_agents.h
pti_mtp.cpp             ← pti_common.cpp, pti_stream.cpp, pti_agents.h
pti_pool.cpp            ← pti_common.cpp, pti_stream.cpp, pti_mtp.cpp, pti_agents.h
pti_parser.cpp          ← pti_agents.h
  ↑
pti_prompt.cpp          ← pti_common.cpp, pti_parser.cpp, pti_agents.h
pti_tools.cpp           ← pti_common.cpp, pti_agents.h
pti_memory.cpp          ← pti_common.cpp, pti_agents.h
pti_eager.cpp           ← pti_parser.cpp, pti_agents.h
  ↑
pti_verify.cpp          ← pti_common.cpp, pti_parser.cpp, pti_tools.cpp,
                          pti_memory.cpp, pti_pool.cpp, pti_agents.h
  ↑
pti_pipeline.cpp        ← pti_common.cpp, pti_parser.cpp, pti_prompt.cpp,
                          pti_tools.cpp, pti_memory.cpp, pti_pool.cpp,
                          pti_verify.cpp, pti_agents.h
  ↑
pti_main.cpp            ← all of the above
```

No circular dependencies. The graph is a clean DAG.

---

## Makefile Rewrite

The Makefile stays at repo root. All source references change from root paths to `src/<module>/` paths. Every existing target name (`make agents`, `make 4seq`, `make server`, etc.) is preserved.

### Source path variables (new section at top of Makefile)

```makefile
# ── Source directories ────────────────────────────────────────────────────────
SRC_DIR    := src
SRC_AGENTS := $(SRC_DIR)/agents
SRC_PTI    := $(SRC_DIR)/pti
SRC_SERVER := $(SRC_DIR)/server
SRC_BENCH  := $(SRC_DIR)/bench
SRC_KERNEL := $(SRC_DIR)/kernel
```

### Include path strategy (Option A recommended)

All source files use `#include "../llama.cpp/..."` which is relative to repo root. After moving sources into `src/`, add `-I..` to compiler flags so every existing `#include` path still resolves without editing any source file.

```makefile
LLAMA_CFLAGS   := -O2 -I.. $(addprefix -I,$(LLAMA_INC))
LLAMA_CXXFLAGS := -O2 -std=c++17 -I.. $(addprefix -I,$(LLAMA_INC)) -I$(LLAMA_DIR)/src
```

The only files that need include edits are the `src/agents/` split files, which get `#include "pti_agents.h"` instead.

### Target-by-target rewrite

```makefile
# ── HIP kernel (AMD) ──────────────────────────────────────────────────────────
$(TARGET_BIN): $(SRC_KERNEL)/pti_kernel.hip | $(BINDIR)
	$(HIPCC) $(HIP_FLAGS) -o $@ $<

$(TARGET_SO): $(SRC_KERNEL)/pti_kernel.hip | $(BINDIR)
	$(HIPCC) $(HIP_FLAGS) $(SHARED) -o $@ $<

# ── HIP kernel (CUDA) ────────────────────────────────────────────────────────
$(TARGET_BIN_CUDA): $(SRC_KERNEL)/pti_kernel.hip | $(BINDIR)
	$(NVCC) $(CUDA_FLAGS) -o $@ $<

$(TARGET_SO_CUDA): $(SRC_KERNEL)/pti_kernel.hip | $(BINDIR)
	$(NVCC) $(CUDA_FLAGS) $(SHARED) -Xcompiler -fPIC -o $@ $<

# ── pti_llama: 2-seq PTI (C) ─────────────────────────────────────────────────
$(PTI_LLAMA): $(SRC_PTI)/pti_llama.c | $(BINDIR)
	gcc $(LLAMA_CFLAGS) -o $@ $< $(LLAMA_LDFLAGS) -lstdc++

# ── pti_2seq: 2-seq PTI (C++) ────────────────────────────────────────────────
$(PTI_2SEQ): $(SRC_PTI)/pti_2seq.cpp | $(BINDIR)
	g++ $(LLAMA_CXXFLAGS) -o $@ $< $(LLAMA_LDFLAGS)

# ── pti_mtp: 3-seq + MTP re-init (C++) ───────────────────────────────────────
$(PTI_MTP): $(SRC_PTI)/pti_mtp.cpp | $(BINDIR)
	g++ $(LLAMA_CXXFLAGS) -o $@ $< $(LLAMA_LDFLAGS)

# ── pti_4seq: 4-seq PTI (C++) ────────────────────────────────────────────────
$(PTI_4SEQ): $(SRC_PTI)/pti_4seq.cpp | $(BINDIR)
	g++ $(LLAMA_CXXFLAGS) -o $@ $< $(LLAMA_LDFLAGS)

# ── pti_lookup: n-gram lookup speculative decoding ───────────────────────────
$(PTI_LOOKUP): $(SRC_PTI)/pti_lookup.cpp | $(BINDIR)
	g++ $(LLAMA_CXXFLAGS) -o $@ $< $(LLAMA_LDFLAGS)

# ── pti_mtp_probe: MTP head semantics probe ──────────────────────────────────
$(PTI_MTPPROBE): $(SRC_PTI)/pti_mtp_probe.cpp | $(BINDIR)
	g++ $(LLAMA_CXXFLAGS) -o $@ $< $(LLAMA_LDFLAGS)

# ── pti_agents: packed-agents (multi-file split) ─────────────────────────────
PTI_AGENTS_SRCS := $(SRC_AGENTS)/pti_common.cpp \
                   $(SRC_AGENTS)/pti_stream.cpp \
                   $(SRC_AGENTS)/pti_mtp.cpp \
                   $(SRC_AGENTS)/pti_pool.cpp \
                   $(SRC_AGENTS)/pti_parser.cpp \
                   $(SRC_AGENTS)/pti_prompt.cpp \
                   $(SRC_AGENTS)/pti_tools.cpp \
                   $(SRC_AGENTS)/pti_memory.cpp \
                   $(SRC_AGENTS)/pti_eager.cpp \
                   $(SRC_AGENTS)/pti_verify.cpp \
                   $(SRC_AGENTS)/pti_pipeline.cpp \
                   $(SRC_AGENTS)/pti_main.cpp

$(PTI_AGENTS): $(PTI_AGENTS_SRCS) $(SRC_AGENTS)/pti_agents.h | $(BINDIR)
	g++ $(LLAMA_CXXFLAGS) -I$(SRC_AGENTS) -o $@ $(PTI_AGENTS_SRCS) $(LLAMA_LDFLAGS)

# ── pti_server: OpenAI-compatible HTTP server ────────────────────────────────
$(PTI_SERVER): $(SRC_SERVER)/pti_server.cpp | $(BINDIR)
	g++ $(LLAMA_CXXFLAGS) \
	    -I$(LLAMA_DIR)/vendor \
	    -I$(LLAMA_DIR)/vendor/cpp-httplib \
	    -I$(LLAMA_DIR)/common \
	    -o $@ $< \
	    $(LLAMA_DIR)/build/vendor/cpp-httplib/libcpp-httplib.a \
	    -lllama-common \
	    $(LLAMA_LDFLAGS) -lpthread -lssl -lcrypto

# ── pti_chat: interactive chat CLI ───────────────────────────────────────────
$(PTI_CHAT): $(SRC_SERVER)/pti_chat.cpp | $(BINDIR)
	g++ $(LLAMA_CXXFLAGS) -o $@ $< $(LLAMA_LDFLAGS)

# ── pti_debug: single-sequence debug ────────────────────────────────────────
$(PTI_DEBUG): $(SRC_BENCH)/pti_debug.cpp | $(BINDIR)
	g++ $(LLAMA_CXXFLAGS) -o $@ $< $(LLAMA_LDFLAGS)

# ── pti_gemv_bench: GEMV fusion benchmark ────────────────────────────────────
$(PTI_BENCH): $(SRC_BENCH)/pti_gemv_bench.cpp | $(BINDIR)
	g++ $(LLAMA_CXXFLAGS) -o $@ $< $(LLAMA_LDFLAGS)

# ── pti_q6k_bench: Q6_K MMVQ microbench (HIP) ───────────────────────────────
$(PTI_Q6K): $(SRC_KERNEL)/pti_q6k_bench.hip | $(BINDIR)
	$(HIPCC) $(HIP_FLAGS) -o $@ $<

# ── pti_kbatch_bench: k-token batch cost curve ──────────────────────────────
$(PTI_KBENCH): $(SRC_BENCH)/pti_kbatch_bench.cpp | $(BINDIR)
	g++ $(LLAMA_CXXFLAGS) -o $@ $< $(LLAMA_LDFLAGS)
```

### Updated phony targets, audit, and run shortcuts

```makefile
.PHONY: all cuda test cuda-test shared cuda-shared clean help \
        llama mtp 4seq server debug bench kbench audit all-llama \
        llama-run-pti llama-run-base \
        mtp-run mtp-run-base \
        4seq-run 4seq-run-base kbench-run agents agents-run \
        chat lookup mtpprobe q6kbench q6kbench-run \
        agents-clean src-clean

# ── Audit (uses src/scripts/audit.sh) ────────────────────────────────────────
audit: debug 4seq bench
	./src/scripts/audit.sh

audit-quick: debug 4seq
	./src/scripts/audit.sh --quick --no-bench

# ── Clean targets ────────────────────────────────────────────────────────────
clean:
	rm -rf $(BINDIR) *.o

agents-clean:
	rm -rf $(SRC_AGENTS)/

src-clean:
	rm -rf $(SRC_DIR)/
```

The `help` target echo output should be updated to show the new directory structure.

---

## Migration Procedure (for the new session)

Execute in 4 phases. Each phase ends with a build + test gate.

### Phase 0: Directory structure

```bash
mkdir -p src/agents src/pti src/server src/bench src/kernel src/scripts src/python
```

### Phase 1: Move all non-agents sources (simple file moves)

Move everything that is NOT `pti_agents.cpp` into its target subdirectory. No content changes needed.

```bash
# src/pti/ — speculative decoding
mv pti_llama.c pti_2seq.cpp pti_mtp.cpp pti_4seq.cpp pti_lookup.cpp pti_mtp_probe.cpp src/pti/

# src/server/
mv pti_server.cpp pti_chat.cpp src/server/

# src/bench/
mv pti_debug.cpp pti_gemv_bench.cpp pti_kbatch_bench.cpp src/bench/

# src/kernel/
mv pti_kernel.hip pti_q6k_bench.hip src/kernel/

# src/scripts/
mv pti-cli.sh llama-server-pti.sh audit.sh flappy_validate.sh scale_validate.sh seq_vs_packed_ab.sh src/scripts/

# src/python/
mv pti_hip.py infer.py pack.py benchmark.py benchmark_comparison.py svg_test.py src/python/
```

### Phase 1 gate: Update Makefile paths for moved files, verify all non-agents targets

1. Rewrite Makefile with new source paths and `-I..` flag
2. Build and verify every non-agents target:
```bash
make llama        # src/pti/pti_llama.c
make 2seq-2       # src/pti/pti_2seq.cpp
make mtp          # src/pti/pti_mtp.cpp
make 4seq         # src/pti/pti_4seq.cpp
make lookup       # src/pti/pti_lookup.cpp
make mtpprobe     # src/pti/pti_mtp_probe.cpp
make chat         # src/server/pti_chat.cpp
make server       # src/server/pti_server.cpp (vendor deps)
make debug        # src/bench/pti_debug.cpp
make bench        # src/bench/pti_gemv_bench.cpp
make kbench       # src/bench/pti_kbatch_bench.cpp
make              # src/kernel/pti_kernel.hip
make q6kbench     # src/kernel/pti_q6k_bench.hip
```

All must build and produce identical binaries to the pre-move versions.

### Phase 2: Split pti_agents.cpp (13 files in src/agents/)

Follow the bottom-up extraction order from the dependency DAG. For each file:
1. Cut the source lines from `pti_agents.cpp`
2. Write to `src/agents/<name>.cpp` with `#include "pti_agents.h"` at the top
3. Remove `static` from functions that need to be visible externally (match header declarations)
4. Keep `static` on functions only called within the same file
5. Preserve all PA.x milestone markers and § references

**Extraction order** (bottom-up, dependency-safe):

| Step | File | Source Lines | Depends On |
|------|------|-------------|------------|
| 2a | `pti_agents.h` | (new) | — |
| 2b | `pti_common.cpp` | 1–122 | header only |
| 2c | `pti_parser.cpp` | 583–739 | header only |
| 2d | `pti_tools.cpp` | 1080–1200 | common |
| 2e | `pti_memory.cpp` | 1201–1550 | common |
| 2f | `pti_eager.cpp` | 1551–2000 | parser |
| 2g | `pti_stream.cpp` | 130–306 | common |
| 2h | `pti_mtp.cpp` | 307–419 | common, stream |
| 2i | `pti_pool.cpp` | 420–582 | common, stream, mtp |
| 2j | `pti_prompt.cpp` | 740–1079 | common, parser |
| 2k | `pti_verify.cpp` | 2001–2349 | common, parser, tools, memory, pool |
| 2l | `pti_pipeline.cpp` | 2350–3050 | everything except main |
| 2m | `pti_main.cpp` | 3051–3337 | everything |

### Phase 2 gate: Build agents + run all self-tests

```bash
make agents
bin/pti_agents --parse-test    # work-order parser
bin/pti_agents --gather-test   # gather + tool-call + PA.7 primitives
bin/pti_agents --coord-test    # repair bookkeeping + stack profile + vars
bin/pti_agents --mtp-test      # MTP bookkeeping
bin/pti_agents --eager-test    # eager scheduling
```

### Phase 3: Delete originals + final cleanup

```bash
rm pti_agents.cpp              # deleted after split verified
```

Update `.clangd` to point to `src/` paths.
Update `.gitignore` if needed (add `src/**/*.o`).
Update any doc references to old file paths (`PLAN.md`, `KERNEL_PLAN.md`, `PTI_AGENT_GAPS.md`).

### Phase 3 gate: Full clean build

```bash
make clean
make agents                    # multi-file split
make llama 2seq-2 mtp 4seq     # src/pti/
make server chat               # src/server/
make debug bench kbench        # src/bench/
make q6kbench                  # src/kernel/
make lookup mtpprobe           # src/pti/
```

All targets build. All self-tests pass. No source files at repo root except Makefile, docs, and configs.

---

## Key Rules

1. **No header guards for `pti_agents.h` needed** — single internal header, not a library
2. **No `.cpp` includes another `.cpp`** — only `#include "pti_agents.h"` in each agents `.cpp`
3. **Globals live in one place** — `pti_common.cpp` owns all global variable definitions; other files only see `extern` declarations from the header
4. **Self-tests stay where their code lives** — parser tests in `pti_parser.cpp`, eager tests in `pti_eager.cpp`, etc. `pti_main.cpp` only dispatches
5. **Preserve all comments** — PA.x milestone markers and § references are the historical record
6. **No behavior changes** — pure refactor. Self-tests passing = correct
7. **Preserve all existing Makefile target names** — `make agents`, `make 4seq`, `make server` etc. all work identically
8. **Use `-I..` to preserve include paths** — no need to edit `#include "../llama.cpp/..."` in moved files
9. **Shell scripts and Python scripts are leaf moves** — no content changes, just directory placement
10. **`bin/` output location unchanged** — all binaries still go to `bin/`, no script paths break

---

## Expected File Sizes After Split

### src/agents/ (split from pti_agents.cpp)

| File | Lines | Complexity |
|------|-------|------------|
| `pti_agents.h` | ~200 | low (declarations only) |
| `pti_common.cpp` | ~120 | low (utilities) |
| `pti_stream.cpp` | ~180 | medium (decode loop) |
| `pti_mtp.cpp` | ~110 | medium (MTP drafting) |
| `pti_pool.cpp` | ~160 | medium (work queue) |
| `pti_parser.cpp` | ~160 | medium (envelope parser) |
| `pti_prompt.cpp` | ~340 | low (string builders) |
| `pti_tools.cpp` | ~120 | medium (sandbox exec) |
| `pti_memory.cpp` | ~350 | medium (blackboard + stack profile) |
| `pti_eager.cpp` | ~450 | high (DAG scheduling) |
| `pti_verify.cpp` | ~350 | high (repair orchestrator) |
| `pti_pipeline.cpp` | ~700 | high (pipeline orchestration) |
| `pti_main.cpp` | ~290 | low (CLI + dispatch) |
| **Total** | **~3,330** | (same as original ± overhead) |

### Root directory after refactor

| Category | Files |
|----------|-------|
| Configs | `Makefile`, `.clangd`, `.gitignore` |
| Docs | `README.md`, `DESIGN.md`, `PLAN.md`, `PACKED_AGENTS.md`, `PTI_AGENT_GAPS.md`, `KERNEL_PLAN.md`, `FAILED_EXPERIMENTS.md`, `POSITIVE_RESULTS.md`, `Overlapped-model-idea.md`, `SEPARATION_OF_CONCERNS.md` |
| Diagrams | `diagram-*.svg` |
| Prompts | `demo_edit_prompt.txt` |
| External deps | `llama.cpp/`, `llama.cpp-files/`, `llama.cpp-patch/` |
| Spec/Validate | `spec/`, `validate/` |
| **Source files** | **none** (all in `src/`) |

### Largest files after refactor

| File | Lines |
|------|-------|
| `src/server/pti_server.cpp` | ~1,600 (not split, single domain) |
| `src/pti/pti_lookup.cpp` | ~900 (not split, single domain) |
| `src/agents/pti_pipeline.cpp` | ~700 |
| `src/agents/pti_eager.cpp` | ~450 |
| `src/agents/pti_memory.cpp` | ~350 |
| `src/agents/pti_verify.cpp` | ~350 |
| `src/agents/pti_prompt.cpp` | ~340 |

The two largest unsplit files (`pti_server.cpp` at ~1,600 and `pti_lookup.cpp` at ~900) are each a single coherent domain. If future splits are desired, they'd be separate refactor efforts.
