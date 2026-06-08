# Packed Twin Inference — GPU kernel build
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
# ── Other ─────────────────────────────────────────────────────────────────────
#   make test               build (AMD) + run self-test
#   make cuda-test          build (CUDA) + run self-test
#   make shared             libpti.so for pti_hip.py (AMD)
#   make clean

# ── AMD defaults ──────────────────────────────────────────────────────────────
ARCH     ?= gfx906
HIPCC    ?= hipcc
HIP_FLAGS := -O3 --offload-arch=$(ARCH) -std=c++14 -lm

# ── CUDA defaults ──────────────────────────────────────────────────────────────
NVCC     ?= nvcc
SM       ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
              | head -1 | tr -d '.' || echo 80)
CUDA_FLAGS := -O3 -arch=sm_$(SM) -x cu -lm

SHARED   := -shared -fPIC

TARGET_BIN      := pti_test
TARGET_BIN_CUDA := pti_test_cuda
TARGET_SO       := libpti.so
TARGET_SO_CUDA  := libpti_cuda.so
SRC             := pti_kernel.hip

.PHONY: all cuda test cuda-test shared cuda-shared clean help

# ── AMD targets ───────────────────────────────────────────────────────────────
all: $(TARGET_BIN)

$(TARGET_BIN): $(SRC)
	$(HIPCC) $(HIP_FLAGS) -o $@ $<
	@echo "Built $(TARGET_BIN) for $(ARCH)"

shared: $(TARGET_SO)
$(TARGET_SO): $(SRC)
	$(HIPCC) $(HIP_FLAGS) $(SHARED) -o $@ $<
	@echo "Built $(TARGET_SO) for $(ARCH)"

test: $(TARGET_BIN)
	./$(TARGET_BIN)

# ── CUDA targets ──────────────────────────────────────────────────────────────
cuda: $(TARGET_BIN_CUDA)

$(TARGET_BIN_CUDA): $(SRC)
	$(NVCC) $(CUDA_FLAGS) -o $@ $<
	@echo "Built $(TARGET_BIN_CUDA) for sm_$(SM)"

cuda-shared: $(TARGET_SO_CUDA)
$(TARGET_SO_CUDA): $(SRC)
	$(NVCC) $(CUDA_FLAGS) $(SHARED) -Xcompiler -fPIC -o $@ $<
	@echo "Built $(TARGET_SO_CUDA) for sm_$(SM)"

cuda-test: $(TARGET_BIN_CUDA)
	./$(TARGET_BIN_CUDA)

# ── llama.cpp PTI binary ─────────────────────────────────────────────────────
#   make llama           build pti_llama (GGUF inference, no GPU kernel needed)
#   make llama-run-pti   run PTI on Qwen3.6-27B-Q5_K_M
#   make llama-run-base  run baseline on Qwen3.6-27B-Q5_K_M

LLAMA_DIR    ?= ../llama.cpp
LLAMA_INC    := $(LLAMA_DIR)/include $(LLAMA_DIR)/ggml/include
LLAMA_LIBDIR := $(LLAMA_DIR)/build/bin
LLAMA_CFLAGS := -O2 $(addprefix -I,$(LLAMA_INC))
LLAMA_CXXFLAGS := -O2 -std=c++17 $(addprefix -I,$(LLAMA_INC)) -I$(LLAMA_DIR)/src
LLAMA_LDFLAGS := -L$(LLAMA_LIBDIR) -lllama -Wl,-rpath,$(abspath $(LLAMA_LIBDIR)) -lm

TEST_MODEL   ?= ../gguf/Qwen3.6-27B-Q5_K_M.gguf
MTP_MODEL    ?= ../gguf/Qwen3.6-27B-UD-Q6_K_XL.gguf
TEST_PROMPT  ?= The key to faster LLM inference is
TEST_TOKENS  ?= 80
NGL          ?= 99

.PHONY: llama llama-run-pti llama-run-base mtp mtp-run mtp-run-base

llama: pti_llama

pti_llama: pti_llama.c
	gcc $(LLAMA_CFLAGS) -o $@ $< $(LLAMA_LDFLAGS) -lstdc++
	@echo "Built pti_llama"

llama-run-pti: pti_llama
	./pti_llama -m $(TEST_MODEL) -p "$(TEST_PROMPT)" -n $(TEST_TOKENS) -ngl $(NGL)

llama-run-base: pti_llama
	./pti_llama -m $(TEST_MODEL) -p "$(TEST_PROMPT)" -n $(TEST_TOKENS) -ngl $(NGL) --baseline

# ── pti_mtp: 3-sequence PTI + MTP re-init ────────────────────────────────────
#   make mtp                   build pti_mtp (C++, requires UD-Q6_K_XL for MTP)
#   make mtp-run               run 3-seq PTI on UD-Q6_K_XL
#   make mtp-run-base          run baseline on UD-Q6_K_XL

mtp: pti_mtp

pti_mtp: pti_mtp.cpp
	g++ $(LLAMA_CXXFLAGS) -o $@ $< $(LLAMA_LDFLAGS)
	@echo "Built pti_mtp"

mtp-run: pti_mtp
	./pti_mtp -m $(MTP_MODEL) -p "$(TEST_PROMPT)" -n $(TEST_TOKENS) -ngl $(NGL)

mtp-run-base: pti_mtp
	./pti_mtp -m $(MTP_MODEL) -p "$(TEST_PROMPT)" -n $(TEST_TOKENS) -ngl $(NGL) --baseline

# ── pti_4seq: 4-sequence PTI ──────────────────────────────────────────────────
#   make 4seq              build pti_4seq
#   make 4seq-run          run 4-seq PTI on UD-Q6_K_XL (target: ~2×)
#   make 4seq-run-base     run baseline on UD-Q6_K_XL

4seq: pti_4seq

pti_4seq: pti_4seq.cpp
	g++ $(LLAMA_CFLAGS) -o $@ $< $(LLAMA_LDFLAGS)
	@echo "Built pti_4seq"

4seq-run: pti_4seq
	./pti_4seq -m $(MTP_MODEL) -p "$(TEST_PROMPT)" -n $(TEST_TOKENS) -ngl $(NGL)

4seq-run-base: pti_4seq
	./pti_4seq -m $(MTP_MODEL) -p "$(TEST_PROMPT)" -n $(TEST_TOKENS) -ngl $(NGL) --baseline

# ── Utility ───────────────────────────────────────────────────────────────────
clean:
	rm -f $(TARGET_BIN) $(TARGET_BIN_CUDA) $(TARGET_SO) $(TARGET_SO_CUDA) pti_llama *.o

help:
	@echo "AMD:     make [ARCH=gfx906|gfx1100|...]  shared  test"
	@echo "CUDA:    make cuda [SM=80|89|86|75]       cuda-shared  cuda-test"
	@echo "llama:   make llama                        build pti_llama (GGUF PTI)"
	@echo "         make llama-run-pti  TEST_MODEL=.. NGL=99"
	@echo "         make llama-run-base TEST_MODEL=.."
	@echo "SM is auto-detected from nvidia-smi if not set"
