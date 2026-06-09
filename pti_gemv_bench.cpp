/*
 * pti_gemv_bench.cpp — does ggml fuse N-sequence GEMV or load weights N times?
 *
 * Theory: autoregressive decode is memory-bandwidth bound on weight reads.
 * Batching N sequences through one llama_decode should share the weight load.
 * If it does, ms/step stays flat as N grows. If not, ms/step scales with N.
 *
 * Interpretation:
 *   scaling(N) ≈ N   → weights loaded N× per step (no fusion, PTI adds overhead)
 *   scaling(N) ≈ 1   → weights shared (PTI amortises weight cost correctly)
 *   scaling(N) ≈ 1-2 → partial fusion
 *
 * Uses a short context (64 tok) to keep KV/SSM overhead small relative to GEMV.
 *
 * Build: make bench
 * Run:   bin/pti_gemv_bench -m ../gguf/Qwen3.6-27B-UD-Q6_K_XL.gguf
 */

#include "../llama.cpp/include/llama.h"

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void batch_add(struct llama_batch *b, llama_token tok,
                      llama_pos pos, llama_seq_id seq, bool want_logits) {
    b->token[b->n_tokens]       = tok;
    b->pos[b->n_tokens]         = pos;
    b->n_seq_id[b->n_tokens]    = 1;
    b->seq_id[b->n_tokens][0]   = seq;
    b->logits[b->n_tokens]      = want_logits ? 1 : 0;
    b->n_tokens++;
}

static void batch_clear(struct llama_batch *b) { b->n_tokens = 0; }

static llama_token argmax(const float *v, int32_t n) {
    llama_token best = 0;
    for (int32_t i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    return best;
}

struct Result {
    int    n_seq;
    double ms_per_step;
    double scaling;        // relative to n_seq=1
    double bw_gb_s;        // effective bandwidth if weights loaded once
    double bw_per_seq;     // bw_gb_s / n_seq (what each seq costs)
};

static Result run(llama_model *model, const llama_vocab *vocab,
                  int n_seq, float model_gb, int n_ctx,
                  int n_warmup, int n_steps, double base_ms) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx         = (uint32_t)n_ctx;
    cp.n_batch       = (uint32_t)(n_seq + 4);
    cp.n_ubatch      = (uint32_t)(n_seq + 4);
    cp.n_seq_max     = (uint32_t)n_seq;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k        = GGML_TYPE_Q8_0;
    cp.type_v        = GGML_TYPE_Q8_0;

    llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context init failed for n_seq=%d\n", n_seq); return {}; }

    llama_memory_t mem  = llama_get_memory(ctx);
    int32_t n_vocab     = llama_vocab_n_tokens(vocab);
    llama_token bos     = llama_vocab_bos(vocab);
    if (bos < 0) bos    = 1;

    // Minimal prefill: single BOS token on seq 0
    llama_batch batch = llama_batch_init(n_seq + 8, 0, n_seq);
    batch_clear(&batch);
    batch_add(&batch, bos, 0, 0, true);

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "prefill failed\n");
        llama_batch_free(batch); llama_free(ctx);
        return {};
    }

    // Copy seq 0 state to all others so they start identically
    for (int s = 1; s < n_seq; s++)
        llama_memory_seq_cp(mem, 0, s, 0, -1);

    // Starting token: greedy pick from prefill logits
    llama_token cur = argmax(llama_get_logits_ith(ctx, 0), n_vocab);
    llama_pos pos   = 1;

    auto step = [&]() {
        batch_clear(&batch);
        for (int s = 0; s < n_seq; s++)
            batch_add(&batch, cur, pos, s, s == 0);
        if (llama_decode(ctx, batch) != 0) return false;
        cur = argmax(llama_get_logits_ith(ctx, 0), n_vocab);
        pos++;
        return true;
    };

    for (int i = 0; i < n_warmup; i++) if (!step()) break;

    double t0 = now_sec();
    for (int i = 0; i < n_steps; i++) if (!step()) break;
    double elapsed = now_sec() - t0;

    llama_batch_free(batch);
    llama_free(ctx);

    Result r;
    r.n_seq       = n_seq;
    r.ms_per_step = (elapsed / n_steps) * 1000.0;
    r.scaling     = (base_ms > 0) ? r.ms_per_step / base_ms : 1.0;
    r.bw_gb_s     = model_gb / (elapsed / n_steps);
    r.bw_per_seq  = r.bw_gb_s / n_seq;
    return r;
}

int main(int argc, char **argv) {
    const char *model_path = nullptr;
    int   ngl      = 99;
    int   n_warmup = 5;
    int   n_steps  = 20;
    int   n_ctx    = 128;     // short: minimises KV/SSM noise vs GEMV signal

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-m")   && i+1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "-ngl") && i+1 < argc) ngl        = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-w")   && i+1 < argc) n_warmup   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s")   && i+1 < argc) n_steps    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-ctx") && i+1 < argc) n_ctx      = atoi(argv[++i]);
    }

    if (!model_path) {
        fprintf(stderr,
            "Usage: %s -m <model.gguf> [-ngl N] [-w warmup] [-s steps] [-ctx N]\n"
            "\n"
            "Measures whether ggml shares weight loads across N decode sequences.\n"
            "Uses short context (default 128) to isolate GEMV from KV/SSM overhead.\n",
            argv[0]);
        return 1;
    }

    llama_backend_init();

    fprintf(stderr, "Loading %s ...\n", model_path);
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    const llama_vocab *vocab = llama_model_get_vocab(model);

    // Derive model weight size from file size (printed in load_tensors output).
    // print_info reports GPU buffer size; we use that as the effective weight
    // data read per step in the fully-unfused case.
    // Hard-coded from our measured run: 23.51 GB on GPU + 1.29 GB CPU = 24.22 GB
    // Adjust if using a different model.
    const float MODEL_GB = 24.22f;

    fprintf(stderr,
        "\nModel weight data: %.2f GB (GPU portion: 23.51 GB)\n"
        "Warmup: %d steps    Measured: %d steps    Context: %d tokens\n\n",
        MODEL_GB, n_warmup, n_steps, n_ctx);

    // ── theory ───────────────────────────────────────────────────────────────
    fprintf(stderr,
        "MI50 HBM bandwidth: ~1024 GB/s\n"
        "Minimum ms/step if weights loaded once:  %.1f ms  (%.0f GB / 1024 GB/s)\n"
        "Minimum ms/step if loaded per-seq (N=4): %.1f ms\n\n",
        MODEL_GB / 1024.0 * 1000.0,
        MODEL_GB,
        MODEL_GB * 4 / 1024.0 * 1000.0);

    // ── run ───────────────────────────────────────────────────────────────────
    fprintf(stderr, "Running benchmarks...\n\n");

    const int counts[] = {1, 2, 4};
    Result results[3];
    double base_ms = 0;

    for (int i = 0; i < 3; i++) {
        int n = counts[i];
        fprintf(stderr, "  N=%d ... ", n); fflush(stderr);
        results[i] = run(model, vocab, n, MODEL_GB, n_ctx,
                         n_warmup, n_steps, base_ms);
        if (n == 1) base_ms = results[i].ms_per_step;
        fprintf(stderr, "%.1f ms/step\n", results[i].ms_per_step);
    }

    // ── report ────────────────────────────────────────────────────────────────
    fprintf(stderr, "\n");
    fprintf(stderr,
        "═══════════════════════════════════════════════════════════════════\n"
        "  N   ms/step   ms/seq   scaling   eff_BW(GB/s)   BW_per_seq\n"
        "  ─   ───────   ──────   ───────   ────────────   ──────────\n");

    for (int i = 0; i < 3; i++) {
        Result &r = results[i];
        fprintf(stderr,
            "  %d   %6.1f    %6.1f   %5.2f×    %7.1f        %7.1f\n",
            r.n_seq, r.ms_per_step, r.ms_per_step / r.n_seq,
            r.scaling, r.bw_gb_s, r.bw_per_seq);
    }

    fprintf(stderr,
        "═══════════════════════════════════════════════════════════════════\n\n");

    // Interpret scaling
    double s4 = results[2].ms_per_step / results[0].ms_per_step;
    fprintf(stderr, "N=4 scaling: %.2f×\n\n", s4);
    if (s4 < 1.5)
        fprintf(stderr, "→ Weights SHARED:   one load serves all N seqs. PTI should win.\n");
    else if (s4 < 3.0)
        fprintf(stderr, "→ PARTIAL fusion:   some sharing, not full. M5 kernel would help.\n");
    else
        fprintf(stderr, "→ Weights NOT shared: loaded ~N× per step. M5 kernel required to unlock PTI.\n");

    fprintf(stderr,
        "\nTheoretical PTI throughput once weights are shared:\n"
        "  N=1 baseline: %.1f tok/s\n"
        "  N=4 ideal:    %.1f tok/s  (weights shared, KV/SSM overhead small)\n",
        1000.0 / results[0].ms_per_step,
        1000.0 / results[0].ms_per_step * 4.0 *
            (results[0].ms_per_step / results[2].ms_per_step));

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
