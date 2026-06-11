# Packed Twin Inference — build
#
# ── AMD / ROCm ────────────────────────────────────────────────────────────────
#   make                    MI50  (gfx906, default)
#   make ARCH=gfx908        MI100
#   make ARCH=gfx90a        MI200 / MI210 / MI250
#   make ARCH=gfx1100       RX 7900 XTX
#   make ARCH=gfx1030       RX 6900 XT
#
# ── NVIDIA / CUDA ─────────────────────────────────────────────────────────────
#   make cuda               auto-detect GPU, or:
#   make cuda SM=80         A100
#   make cuda SM=89         RTX 4090
#   make cuda SM=86         RTX 3090 / 3080
#   make cuda SM=75         RTX 2080 Ti / T4
#
# ── llama.cpp PTI binaries ────────────────────────────────────────────────────
#   make 2seq-2             pti_2seq   (2-seq, C++) ← best measured: 16.3 tok/s
#   make llama              pti_llama  (2-seq, C, earlier prototype)
#   make mtp                pti_mtp    (3-seq + MTP re-init, C++)
#   make 4seq               pti_4seq   (4-seq, C++) ← 14.6 tok/s (0.77× baseline)
#   make all-llama          all three
#
# ── Run shortcuts ─────────────────────────────────────────────────────────────
#   make 4seq-run           4-seq PTI on UD-Q6_K_XL
#   make 4seq-run-base      baseline on UD-Q6_K_XL
#   make mtp-run / mtp-run-base
#   make llama-run-pti / llama-run-base
#
# ── Other ─────────────────────────────────────────────────────────────────────
#   make test               build HIP kernel + run bandwidth benchmarks
#   make cuda-test          build CUDA kernel + run benchmarks
#   make shared             libpti.so for pti_hip.py (AMD)
#   make clean

# ── Output directory ──────────────────────────────────────────────────────────
BINDIR := bin

# ── AMD / ROCm ────────────────────────────────────────────────────────────────
ARCH      ?= gfx906
HIPCC     ?= hipcc
HIP_FLAGS := -O3 --offload-arch=$(ARCH) -std=c++14 -lm

# ── NVIDIA / CUDA ─────────────────────────────────────────────────────────────
NVCC      ?= nvcc
SM        ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
               | head -1 | tr -d '.' || echo 80)
CUDA_FLAGS := -O3 -arch=sm_$(SM) -x cu -lm

SHARED    := -shared -fPIC

# ── Targets ───────────────────────────────────────────────────────────────────
TARGET_BIN      := $(BINDIR)/pti_test
TARGET_BIN_CUDA := $(BINDIR)/pti_test_cuda
TARGET_SO       := $(BINDIR)/libpti.so
TARGET_SO_CUDA  := $(BINDIR)/libpti_cuda.so
SRC             := pti_kernel.hip

PTI_LLAMA  := $(BINDIR)/pti_llama
PTI_MTP    := $(BINDIR)/pti_mtp
PTI_2SEQ   := $(BINDIR)/pti_2seq
PTI_4SEQ   := $(BINDIR)/pti_4seq
PTI_SERVER := $(BINDIR)/pti_server
PTI_DEBUG  := $(BINDIR)/pti_debug
PTI_BENCH  := $(BINDIR)/pti_gemv_bench
PTI_KBENCH := $(BINDIR)/pti_kbatch_bench

# ── llama.cpp paths ───────────────────────────────────────────────────────────
LLAMA_DIR    ?= ../llama.cpp
LLAMA_INC    := $(LLAMA_DIR)/include $(LLAMA_DIR)/ggml/include
LLAMA_LIBDIR := $(LLAMA_DIR)/build/bin
LLAMA_CFLAGS   := -O2 $(addprefix -I,$(LLAMA_INC))
LLAMA_CXXFLAGS := -O2 -std=c++17 $(addprefix -I,$(LLAMA_INC)) -I$(LLAMA_DIR)/src
LLAMA_LDFLAGS  := -L$(LLAMA_LIBDIR) -lllama -Wl,-rpath,$(abspath $(LLAMA_LIBDIR)) -lm

# ── Model / run defaults ──────────────────────────────────────────────────────
TEST_MODEL  ?= ../gguf/Qwen3.6-27B-Q5_K_M.gguf
MTP_MODEL   ?= ../gguf/Qwen3.6-27B-UD-Q6_K_XL.gguf
TEST_PROMPT ?= The key to faster LLM inference is
TEST_TOKENS ?= 80
NGL         ?= 99

# ── Phony targets ─────────────────────────────────────────────────────────────
.PHONY: all cuda test cuda-test shared cuda-shared clean help \
        llama mtp 4seq server debug bench kbench audit all-llama \
        llama-run-pti llama-run-base \
        mtp-run mtp-run-base \
        4seq-run 4seq-run-base kbench-run


# ── bin/ directory ────────────────────────────────────────────────────────────
$(BINDIR):
	mkdir -p $(BINDIR)

# ── HIP kernel (AMD) ──────────────────────────────────────────────────────────
all: $(TARGET_BIN)

$(TARGET_BIN): $(SRC) | $(BINDIR)
	$(HIPCC) $(HIP_FLAGS) -o $@ $<
	@echo "Built $@ for $(ARCH)"

shared: $(TARGET_SO)
$(TARGET_SO): $(SRC) | $(BINDIR)
	$(HIPCC) $(HIP_FLAGS) $(SHARED) -o $@ $<
	@echo "Built $@ for $(ARCH)"

test: $(TARGET_BIN)
	$(TARGET_BIN)

# ── HIP kernel (CUDA) ─────────────────────────────────────────────────────────
cuda: $(TARGET_BIN_CUDA)

$(TARGET_BIN_CUDA): $(SRC) | $(BINDIR)
	$(NVCC) $(CUDA_FLAGS) -o $@ $<
	@echo "Built $@ for sm_$(SM)"

cuda-shared: $(TARGET_SO_CUDA)
$(TARGET_SO_CUDA): $(SRC) | $(BINDIR)
	$(NVCC) $(CUDA_FLAGS) $(SHARED) -Xcompiler -fPIC -o $@ $<
	@echo "Built $@ for sm_$(SM)"

cuda-test: $(TARGET_BIN_CUDA)
	$(TARGET_BIN_CUDA)

# ── pti_llama: 2-sequence PTI (C) ─────────────────────────────────────────────
llama: $(PTI_LLAMA)

$(PTI_LLAMA): pti_llama.c | $(BINDIR)
	gcc $(LLAMA_CFLAGS) -o $@ $< $(LLAMA_LDFLAGS) -lstdc++
	@echo "Built $@"

llama-run-pti: $(PTI_LLAMA)
	$(PTI_LLAMA) -m $(TEST_MODEL) -p "$(TEST_PROMPT)" -n $(TEST_TOKENS) -ngl $(NGL)

llama-run-base: $(PTI_LLAMA)
	$(PTI_LLAMA) -m $(TEST_MODEL) -p "$(TEST_PROMPT)" -n $(TEST_TOKENS) -ngl $(NGL) --baseline

# ── pti_2seq: 2-sequence PTI (C++) ───────────────────────────────────────────
2seq-2: $(PTI_2SEQ)

$(PTI_2SEQ): pti_2seq.cpp | $(BINDIR)
	g++ $(LLAMA_CXXFLAGS) -o $@ $< $(LLAMA_LDFLAGS)
	@echo "Built $@"

# ── pti_mtp: 3-sequence PTI + MTP re-init (C++) ───────────────────────────────
mtp: $(PTI_MTP)

$(PTI_MTP): pti_mtp.cpp | $(BINDIR)
	g++ $(LLAMA_CXXFLAGS) -o $@ $< $(LLAMA_LDFLAGS)
	@echo "Built $@"

mtp-run: $(PTI_MTP)
	$(PTI_MTP) -m $(MTP_MODEL) -p "$(TEST_PROMPT)" -n $(TEST_TOKENS) -ngl $(NGL)

mtp-run-base: $(PTI_MTP)
	$(PTI_MTP) -m $(MTP_MODEL) -p "$(TEST_PROMPT)" -n $(TEST_TOKENS) -ngl $(NGL) --baseline

# ── pti_4seq: 4-sequence PTI (C++) — headline result ─────────────────────────
4seq: $(PTI_4SEQ)

$(PTI_4SEQ): pti_4seq.cpp | $(BINDIR)
	g++ $(LLAMA_CXXFLAGS) -o $@ $< $(LLAMA_LDFLAGS)
	@echo "Built $@"

4seq-run: $(PTI_4SEQ)
	$(PTI_4SEQ) -m $(MTP_MODEL) -p "$(TEST_PROMPT)" -n $(TEST_TOKENS) -ngl $(NGL)

4seq-run-base: $(PTI_4SEQ)
	$(PTI_4SEQ) -m $(MTP_MODEL) -p "$(TEST_PROMPT)" -n $(TEST_TOKENS) -ngl $(NGL) --baseline

# ── pti_debug: single-sequence debug script ──────────────────────────────────
debug: $(PTI_DEBUG)

$(PTI_DEBUG): pti_debug.cpp | $(BINDIR)
	g++ $(LLAMA_CXXFLAGS) -o $@ $< $(LLAMA_LDFLAGS)
	@echo "Built $@"

# ── pti_gemv_bench: GEMV fusion benchmark ────────────────────────────────────
bench: $(PTI_BENCH)

$(PTI_BENCH): pti_gemv_bench.cpp | $(BINDIR)
	g++ $(LLAMA_CXXFLAGS) -o $@ $< $(LLAMA_LDFLAGS)
	@echo "Built $@"

# ── pti_q6k_bench: M6.1 Q6_K MMVQ replica microbench (HIP) ───────────────────
PTI_Q6K := $(BINDIR)/pti_q6k_bench

q6kbench: $(PTI_Q6K)

$(PTI_Q6K): pti_q6k_bench.hip | $(BINDIR)
	$(HIPCC) $(HIP_FLAGS) -o $@ $<
	@echo "Built $@"

q6kbench-run: $(PTI_Q6K)
	$(PTI_Q6K)

# ── pti_lookup: M6.4 n-gram lookup speculative decoding ──────────────────────
PTI_LOOKUP := $(BINDIR)/pti_lookup

lookup: $(PTI_LOOKUP)

$(PTI_LOOKUP): pti_lookup.cpp | $(BINDIR)
	g++ $(LLAMA_CXXFLAGS) -o $@ $< $(LLAMA_LDFLAGS)
	@echo "Built $@"

# ── pti_chat: interactive chat CLI (llama-cli equivalent, mode-switchable) ───
PTI_CHAT := $(BINDIR)/pti_chat

chat: $(PTI_CHAT)

$(PTI_CHAT): pti_chat.cpp | $(BINDIR)
	g++ $(LLAMA_CXXFLAGS) -o $@ $< $(LLAMA_LDFLAGS)
	@echo "Built $@"

# ── pti_mtp_probe: M7.0 MTP head semantics probe ─────────────────────────────
PTI_MTPPROBE := $(BINDIR)/pti_mtp_probe

mtpprobe: $(PTI_MTPPROBE)

$(PTI_MTPPROBE): pti_mtp_probe.cpp | $(BINDIR)
	g++ $(LLAMA_CXXFLAGS) -o $@ $< $(LLAMA_LDFLAGS)
	@echo "Built $@"

# ── pti_kbatch_bench: M6.0 k-token batch cost curve ──────────────────────────
kbench: $(PTI_KBENCH)

$(PTI_KBENCH): pti_kbatch_bench.cpp | $(BINDIR)
	g++ $(LLAMA_CXXFLAGS) -o $@ $< $(LLAMA_LDFLAGS)
	@echo "Built $@"

kbench-run: $(PTI_KBENCH)
	$(PTI_KBENCH) -m $(MTP_MODEL) -p "$(TEST_PROMPT)" -ngl $(NGL)

# ── pti_server: PTI HTTP server (OpenAI-compatible) ───────────────────────────
server: $(PTI_SERVER)

$(PTI_SERVER): pti_server.cpp | $(BINDIR)
	g++ $(LLAMA_CXXFLAGS) \
	    -I$(LLAMA_DIR)/vendor \
	    -I$(LLAMA_DIR)/vendor/cpp-httplib \
	    -I$(LLAMA_DIR)/common \
	    -o $@ $< \
	    $(LLAMA_DIR)/build/vendor/cpp-httplib/libcpp-httplib.a \
	    -lllama-common \
	    $(LLAMA_LDFLAGS) -lpthread -lssl -lcrypto
	@echo "Built $@"

# ── Audit: correctness + performance for each PTI step ───────────────────────
audit: debug 4seq bench
	@echo "Running PTI audit (8-10 min on MI50)..."
	./audit.sh

audit-quick: debug 4seq
	@echo "Running quick PTI audit (no bench)..."
	./audit.sh --quick --no-bench

# ── Build all llama.cpp binaries ──────────────────────────────────────────────
all-llama: $(PTI_LLAMA) $(PTI_MTP) $(PTI_4SEQ)

# ── Utility ───────────────────────────────────────────────────────────────────
clean:
	rm -rf $(BINDIR) *.o

help:
	@echo "HIP kernel (AMD):"
	@echo "  make [ARCH=gfx906|gfx908|gfx90a|gfx1100]   build bin/pti_test"
	@echo "  make test                                    build + run benchmarks"
	@echo "  make shared                                  build bin/libpti.so"
	@echo ""
	@echo "HIP kernel (CUDA):"
	@echo "  make cuda [SM=80|86|89]                      build bin/pti_test_cuda"
	@echo "  make cuda-test                               build + run benchmarks"
	@echo ""
	@echo "llama.cpp PTI binaries (all output to bin/):"
	@echo "  make llama                                   bin/pti_llama  (2-seq)"
	@echo "  make mtp                                     bin/pti_mtp    (3-seq + MTP)"
	@echo "  make 2seq-2                                  bin/pti_2seq   (2-seq, best 16.3 tok/s)"
	@echo "  make 4seq                                    bin/pti_4seq   (4-seq, 14.6 tok/s)"
	@echo "  make all-llama                               all three"
	@echo ""
	@echo "Run shortcuts:"
	@echo "  make 4seq-run        4-seq PTI on UD-Q6_K_XL (14.6 tok/s)"
	@echo "  make 4seq-run-base   baseline for comparison"
	@echo "  make mtp-run / mtp-run-base"
	@echo "  make llama-run-pti / llama-run-base"
	@echo ""
	@echo "Variables:"
	@echo "  ARCH=gfx906          GPU arch (default: gfx906 = MI50)"
	@echo "  MTP_MODEL=path       model for 4seq/mtp runs (default: UD-Q6_K_XL)"
	@echo "  TEST_MODEL=path      model for llama runs (default: Q5_K_M)"
	@echo "  TEST_TOKENS=80       tokens to generate"
	@echo "  NGL=99               GPU layers"
