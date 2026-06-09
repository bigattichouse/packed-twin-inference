// pti_mmvq_patch.cu — Phase 2 kernel patch for llama.cpp
//
// Target file: ggml/src/ggml-cuda/mmvq.cu
//
// Problem:  mul_mat_vec_q<type, ncols_dst=4> on GCN (MI50) runs at 2.04x the
//           cost of ncols_dst=1, giving only 1.96x throughput for 4 sequences.
//           Root cause: for each weight block at position kby, the kernel reads
//           activation blocks from 4 widely-separated memory addresses:
//
//             j=0: y[0 * stride_col_y + kby]   base + 0 * 5760 + kby*36
//             j=1: y[1 * stride_col_y + kby]   base + 1 * 5760 + kby*36
//             j=2: y[2 * stride_col_y + kby]   base + 2 * 5760 + kby*36
//             j=3: y[3 * stride_col_y + kby]   base + 3 * 5760 + kby*36
//
//           (stride_col_y = K/QK8_1 = 5120/32 = 160 blocks = 5760 bytes)
//
//           These four addresses are 5760 bytes apart → 4 separate L2 cache
//           lines per kby iteration × 160 kby iterations = 640 L2 transactions
//           per sequence row, vs 160 for N=1.  L2 transaction count is 4x, which
//           saturates the L2 and reduces effective HBM BW from 393 GB/s to 177 GB/s.
//
// Fix:     Pre-pack the 4 activation columns into an interleaved layout:
//
//             y_intlv[kby * 4 + j]   (all four j blocks adjacent per kby)
//
//           Each kby iteration fetches 4 × 36 = 144 bytes in 3 consecutive cache
//           lines instead of 4 scattered cache lines.  L2 transaction count drops
//           from 4× back toward 1× per kby, pushing effective BW back toward
//           393 GB/s and the step cost back toward 1.0×.
//
// Expected: step overhead 2.04× → ~1.1×, tok/s 38.1 → 60–75.
//
// Apply this patch as a PR to llama.cpp.  The existing mul_mat_vec_q kernel is
// unchanged; the new template param defaults to false so no existing code path
// is affected.

// ─── Prerequisites ───────────────────────────────────────────────────────────
// In ggml/src/ggml-cuda/mmvq.cu at line 486:
//   const block_q8_1 * y = ((const block_q8_1 *) vy) + offset;
// The activation tensor vy is already Q8_1 quantized by this point (the
// quantize_row_q8_1_cuda step runs just before the GEMV kernel in ggml-cuda.cu).
// The interleaving step also operates on block_q8_1 units, so it is format-safe.

#pragma once

#include "mmvq.cuh"          // block_q8_1, MMVQ_MAX_BATCH_SIZE, QK8_1
#include "ggml-cuda.cuh"     // ggml_cuda_error_check

// ─────────────────────────────────────────────────────────────────────────────
// Part 1 — Interleave kernel
// ─────────────────────────────────────────────────────────────────────────────
//
// Converts the standard Q8_1 activation layout used by llama_decode:
//
//   [col0_blk0 .. col0_blk(K-1) | col1_blk0 .. col1_blk(K-1) | ...]
//                   ^stride_col_y blocks between columns^
//
// into the PTI interleaved layout expected by mul_mat_vec_q_pti:
//
//   [col0_blk0, col1_blk0, col2_blk0, col3_blk0,
//    col0_blk1, col1_blk1, col2_blk1, col3_blk1, ...]
//
// Launch with: <<<(K_blocks + 127) / 128, 128, 0, stream>>>
//   one thread per K_block, each thread writes 4 adjacent output blocks.
//
// Buffer requirement: K_blocks × N × sizeof(block_q8_1) bytes.
// For Qwen3.6-27B FFN (K=5120, N=4): 160 × 4 × 36 = 23040 bytes = 22.5 KB.

template <int N_COLS>
static __global__ void pti_interleave_q8_1_kernel(
        const block_q8_1 * __restrict__ y_col,   // standard layout: [N_COLS × K_blocks]
        block_q8_1       * __restrict__ y_intlv,  // interleaved output: [K_blocks × N_COLS]
        int K_blocks,                              // K / QK8_1
        int stride_col)                            // blocks between columns (= K_blocks)
{
    const int kby = blockIdx.x * blockDim.x + threadIdx.x;
    if (kby >= K_blocks) return;

#pragma unroll
    for (int j = 0; j < N_COLS; ++j) {
        y_intlv[kby * N_COLS + j] = y_col[j * stride_col + kby];
    }
}

// Host wrapper: interleave N_COLS Q8_1 activation columns in-place into buf.
// buf must hold K_blocks * N_COLS * sizeof(block_q8_1) bytes.
// Called once per layer per decode step in PTI mode; cost is negligible vs GEMV.
template <int N_COLS>
static void pti_interleave_activations(
        const block_q8_1 * y,       // Q8_1 activations [N_COLS × K_blocks]
        block_q8_1       * buf,     // pre-allocated interleaved buffer
        int K_blocks,
        cudaStream_t stream)
{
    const int threads = 128;
    const int blocks  = (K_blocks + threads - 1) / threads;
    pti_interleave_q8_1_kernel<N_COLS><<<blocks, threads, 0, stream>>>(
        y, buf, K_blocks, K_blocks);
    GGML_CUDA_CHECK(cudaGetLastError());
}

// ─────────────────────────────────────────────────────────────────────────────
// Part 2 — Modified mul_mat_vec_q inner loop (diff format)
// ─────────────────────────────────────────────────────────────────────────────
//
// In mmvq.cu, add template param `bool pti_interleaved = false` to the kernel:
//
//   // BEFORE (line 396):
//   template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false>
//   static __global__ void mul_mat_vec_q(...)
//
//   // AFTER:
//   template <ggml_type type, int ncols_dst, bool has_fusion,
//             bool small_k = false, bool pti_interleaved = false>
//   static __global__ void mul_mat_vec_q(...)
//
// Then replace the inner loop (lines 489-509) with the version below.
// The `if constexpr` branch is compiled away at zero cost for the default path.
//
// IMPORTANT: stride_col_y is no longer used in the pti_interleaved path, so
// the compiler will eliminate the corresponding load.

// ── New inner loop (replaces lines 489-509 in mmvq.cu) ─────────────────────
//
// for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
//     const int kby = kbx * (qk/QK8_1);
//     const int kqs = vdr * (tid % (qi/vdr));
//
//     if constexpr (pti_interleaved) {
//         // All ncols_dst activation blocks for this kby position are adjacent:
//         //   y_intlv[kby*ncols_dst + 0], [kby*ncols_dst + 1], ...
//         // This fetches all 4 from 3 consecutive 64-byte cache lines instead of
//         // 4 scattered cache lines 5760 bytes apart.
//         const block_q8_1 * ybase = &y[kby * ncols_dst];
//
//         #pragma unroll
//         for (int j = 0; j < ncols_dst; ++j) {
//             #pragma unroll
//             for (int i = 0; i < rows_per_cuda_block; ++i) {
//                 tmp[j][i] += vec_dot_q_cuda(
//                     vx, &ybase[j], kbx_offset + i*stride_row_x + kbx, kqs);
//             }
//         }
//     } else {
//         // Original code — unchanged for all non-PTI paths
//         #pragma unroll
//         for (int j = 0; j < ncols_dst; ++j) {
//             #pragma unroll
//             for (int i = 0; i < rows_per_cuda_block; ++i) {
//                 tmp[j][i] += vec_dot_q_cuda(
//                     vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
//                 if constexpr (has_fusion) {
//                     if (use_gate) {
//                         tmp_gate[j][i] += vec_dot_q_cuda(
//                             vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
//                     }
//                 }
//             }
//         }
//     }
// }
//
// Note: has_fusion and pti_interleaved are mutually exclusive in practice
// (PTI decode never uses gated FFN fusions simultaneously).  If needed, the
// interleaved path can be extended to support fusion by also interleaving the
// gate activations into a second buffer.

// ─────────────────────────────────────────────────────────────────────────────
// Part 3 — PTI dispatch function
// ─────────────────────────────────────────────────────────────────────────────
//
// Add alongside ggml_cuda_mul_mat_vec_q in ggml-cuda.cu.
// Called when ctx->pti_params.n_streams == 4 and ncols_dst == 4.
//
// The interleaved activation buffer is allocated once in llama_context_pti_init
// and reused across decode steps.

// Pseudo-code for integration into ggml-cuda.cu:
//
// void ggml_cuda_mul_mat_vec_q_pti(ggml_backend_cuda_context & ctx,
//                                   ggml_tensor * dst) {
//     // ... existing setup code from ggml_cuda_mul_mat_vec_q ...
//
//     const int ncols_dst = src1->ne[1];  // = 4 for PTI
//
//     if (ncols_dst == 4 && ctx.pti_mode) {
//         // Interleave activation columns into pre-allocated buffer
//         const block_q8_1 * y_q8 = (const block_q8_1 *) src1_as_f16->data;
//         block_q8_1 * y_intlv    = ctx.pti_activation_buf;   // 22.5 KB for Qwen3.6
//         const int K_blocks       = src0->ne[0] / QK8_1;
//
//         pti_interleave_activations<4>(y_q8, y_intlv, K_blocks, stream);
//
//         // Launch kernel with pti_interleaved=true, passing y_intlv as vy
//         mul_mat_vec_q_switch_ncols_dst<type, /*pti_interleaved=*/true>(
//             src0->data, y_intlv, nullptr, fusion, dst->data,
//             ncols_x, nrows_x, ncols_dst,
//             stride_row_x, /*stride_col_y unused*/ 0, stride_col_dst,
//             ...stream);
//     } else {
//         // Original path
//         ggml_cuda_mul_mat_vec_q(ctx, dst);
//     }
// }

// ─────────────────────────────────────────────────────────────────────────────
// Part 4 — llama_context PTI state
// ─────────────────────────────────────────────────────────────────────────────
//
// In src/llama.cpp, add to llama_context:
//
//   struct llama_context {
//       ...
//       // PTI mode — set by llama_context_set_pti()
//       int  pti_n_streams    = 0;     // 0 = off, 2/3/4 = PTI active
//       bool pti_interleaved  = false; // true once Phase 2 kernel is merged
//       void * pti_act_buf    = nullptr; // K_blocks * 4 * sizeof(block_q8_1)
//       size_t pti_act_buf_sz = 0;
//   };
//
//   void llama_context_set_pti(llama_context * ctx, int n_streams) {
//       ctx->pti_n_streams   = n_streams;
//       ctx->pti_interleaved = (n_streams == 4);   // enable Phase 2 path
//       // Allocate interleaved buffer (max FFN size: 17408/32 * 4 * 36 = 78 KB)
//       const size_t max_K_blocks = ctx->model.hparams.n_ff / QK8_1 + 1;
//       ctx->pti_act_buf_sz  = max_K_blocks * n_streams * sizeof(block_q8_1);
//       CUDA_CHECK(cudaMalloc(&ctx->pti_act_buf, ctx->pti_act_buf_sz));
//   }

// ─────────────────────────────────────────────────────────────────────────────
// Part 5 — Expected performance after patch
// ─────────────────────────────────────────────────────────────────────────────
//
// Measured (before patch, pti_4seq.cpp):
//   N=4, 105 ms/step, 177 GB/s effective, 38.1 tok/s, 2.04× overhead
//
// Expected (after patch, interleaved float4 path):
//   L2 activation transactions: 640 → ~180 per sequence row  (3.6× reduction)
//   Effective BW target: 177 → ~330–380 GB/s  (approaching N=1 efficiency)
//   Step cost target:    105 → ~52–57 ms  (1.05–1.15× overhead vs 47 ms)
//   tok/s target:        38.1 → 60–75 tok/s
//
// This matches the pti_linear4_q8_intlv benchmark in pti_kernel.hip (130 GB/s
// effective for Q8 custom kernel).  The Q5_K_M advantage (60% more data per
// cache line) should push the interleaved MMVQ path higher, toward 300+ GB/s.
//
// Phase 3 (MTP head) then adds ~1.5 free tokens per step on top of this,
// targeting 80–100 tok/s total.

// ─────────────────────────────────────────────────────────────────────────────
// Quick validation test
// ─────────────────────────────────────────────────────────────────────────────
//
// To verify correctness of the interleaved path before benchmarking:
//
//   1. Run pti_4seq with GGML_CUDA_FORCE_MMVQ_PTI=1 (new env var in ggml-cuda.cu)
//   2. Compare output tokens against baseline pti_4seq without the flag
//   3. Output must be bit-identical (same argmax on the same logits)
//   4. Measure step time: should drop from ~105 ms toward ~52 ms
//
// A correctness mismatch means the interleaved buffer is mis-indexed.
// The most common error: stride_col_y used in the interleaved path — check
// that the `if constexpr (pti_interleaved)` branch does NOT use stride_col_y.
