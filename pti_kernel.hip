/*
 * pti_kernel.hip — Packed Twin Inference GPU kernel
 *
 * One weight load from HBM serves two independent inference streams:
 *   Twin A: confirms token at sequence position n
 *   Twin B: speculatively computes token at position n+1
 *
 * ── AMD / ROCm ────────────────────────────────────────────────────────────────
 *   hipcc -O3 --offload-arch=gfx906  -o pti_test pti_kernel.hip   # MI50
 *   hipcc -O3 --offload-arch=gfx908  -o pti_test pti_kernel.hip   # MI100
 *   hipcc -O3 --offload-arch=gfx1100 -o pti_test pti_kernel.hip   # RX 7900
 *
 * ── NVIDIA / CUDA ─────────────────────────────────────────────────────────────
 *   nvcc  -O3 -arch=sm_80  -o pti_test pti_kernel.hip  -x cu      # A100
 *   nvcc  -O3 -arch=sm_89  -o pti_test pti_kernel.hip  -x cu      # RTX 4090
 *   nvcc  -O3 -arch=sm_86  -o pti_test pti_kernel.hip  -x cu      # RTX 3090
 *
 * The GPU kernels are identical for both.  The portability shim below maps
 * HIP host-API calls to their CUDA equivalents when compiled with nvcc.
 *
 * Weight block layout (68 bytes per 32-weight block):
 *   bytes  0-1  : scale_a  (float16)
 *   bytes  2-3  : scale_b  (float16)
 *   bytes  4-67 : 32 × uint16  (hi byte = Twin A int8, lo byte = Twin B int8)
 */

// ── Portability shim: HIP ↔ CUDA ─────────────────────────────────────────────
//
// nvcc defines __CUDACC__; hipcc defines __HIP_PLATFORM_AMD__ (or _HCC_).
// When __CUDACC__ is set we remap every hip* symbol to its cuda* equivalent
// and provide hipLaunchKernelGGL as a variadic macro wrapper around <<<>>>.

#ifdef __CUDACC__
  #include <cuda_runtime.h>
  #include <cuda_fp16.h>

  // Types
  #define hipError_t          cudaError_t
  #define hipSuccess          cudaSuccess
  #define hipEvent_t          cudaEvent_t

  // Memory
  #define hipMalloc           cudaMalloc
  #define hipFree             cudaFree
  #define hipMemcpy           cudaMemcpy
  #define hipMemcpyHostToDevice  cudaMemcpyHostToDevice
  #define hipMemcpyDeviceToHost  cudaMemcpyDeviceToHost

  // Synchronisation
  #define hipDeviceSynchronize   cudaDeviceSynchronize
  #define hipGetErrorString      cudaGetErrorString

  // Events
  #define hipEventCreate         cudaEventCreate
  #define hipEventDestroy        cudaEventDestroy
  #define hipEventRecord         cudaEventRecord
  #define hipEventSynchronize    cudaEventSynchronize
  #define hipEventElapsedTime    cudaEventElapsedTime

  // Kernel launch — hipLaunchKernelGGL(kernel, grid, block, smem, stream, ...)
  //                 becomes   kernel<<<grid, block, smem, stream>>>(...)
  #define hipLaunchKernelGGL(kernel, grid, block, smem, stream, ...) \
      kernel<<<grid, block, smem, stream>>>(__VA_ARGS__)

  // NVIDIA warp = 32 threads; AMD wavefront = 64.
  // The kernels use __syncthreads() throughout so correctness is unaffected,
  // but the tiling constants are tuned for MI50.  Override for CUDA:
  #undef  WARP_SIZE
  #define WARP_SIZE 32

#else
  #include <hip/hip_runtime.h>
  #include <hip/hip_fp16.h>
#endif
// ─────────────────────────────────────────────────────────────────────────────

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>

// ── Constants ────────────────────────────────────────────────────────────────

#define BLOCK_SIZE    32          // weights per quantization block
#define BLOCK_BYTES   68          // 2 scales (f16) + 32 weights (uint16)
#define WARP_SIZE     64          // MI50 wavefront size
#define TILE_K        32          // input feature tile size (= BLOCK_SIZE)
#define MAX_THREADS   256

// ── PTI Packed Linear Layer ──────────────────────────────────────────────────
//
// Computes:
//   y_a[m] = sum_k  scale_a[block(m,k)] * int8(packed[m,k] >> 8)  * x_a[k]
//   y_b[m] = sum_k  scale_b[block(m,k)] * int8(packed[m,k] & 0xFF) * x_b[k]
//
// Grid:  (M,)          — one block per output feature
// Block: (TILE_K,)     — threads cover one input tile

__global__ void pti_linear(
    const uint8_t* __restrict__ W,       // packed weight tensor [M × blocks × 68 bytes]
    const float*   __restrict__ x_a,     // Twin A input [K]
    const float*   __restrict__ x_b,     // Twin B input [K]
    float*         __restrict__ y_a,     // Twin A output [M]
    float*         __restrict__ y_b,     // Twin B output [M]
    int M, int K)
{
    int row         = blockIdx.x;
    if (row >= M) return;

    int tid         = threadIdx.x;
    int n_blocks    = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocks_step = blockDim.x / BLOCK_SIZE;   // blocks processed per iteration

    // LDS: tile of packed weight blocks + both input vectors
    __shared__ uint8_t  s_w[TILE_K * BLOCK_BYTES / BLOCK_SIZE];  // one block row
    __shared__ float    s_xa[TILE_K];
    __shared__ float    s_xb[TILE_K];

    float acc_a = 0.f, acc_b = 0.f;

    const uint8_t* row_ptr = W + (size_t)row * n_blocks * BLOCK_BYTES;

    for (int bl = 0; bl < n_blocks; ++bl) {
        int k0 = bl * BLOCK_SIZE;

        // Cooperatively load one weight block + input tile into LDS
        if (tid < BLOCK_BYTES)
            s_w[tid] = row_ptr[bl * BLOCK_BYTES + tid];

        if (k0 + tid < K) {
            s_xa[tid] = x_a[k0 + tid];
            s_xb[tid] = x_b[k0 + tid];
        } else {
            s_xa[tid] = 0.f;
            s_xb[tid] = 0.f;
        }
        __syncthreads();

        // Only first BLOCK_SIZE threads do the accumulation for this block
        if (tid < BLOCK_SIZE) {
            // Unpack scales (bytes 0-3 of the block)
            uint16_t sa_bits = ((uint16_t)s_w[0]) | ((uint16_t)s_w[1] << 8);
            uint16_t sb_bits = ((uint16_t)s_w[2]) | ((uint16_t)s_w[3] << 8);
            float s_a = __half2float(__ushort_as_half(sa_bits));
            float s_b = __half2float(__ushort_as_half(sb_bits));

            // Unpack weight pair at position tid
            uint16_t pw = ((uint16_t)s_w[4 + tid*2]) |
                          ((uint16_t)s_w[4 + tid*2 + 1] << 8);
            int8_t   wa = (int8_t)((int16_t)pw >> 8);   // signed hi byte
            int8_t   wb = (int8_t)(pw & 0xFF);           // signed lo byte

            acc_a += s_a * (float)wa * s_xa[tid];
            acc_b += s_b * (float)wb * s_xb[tid];
        }
        __syncthreads();
    }

    // Reduce across the BLOCK_SIZE threads contributing to this row
    // (all other threads have acc = 0)
    __shared__ float red_a[MAX_THREADS];
    __shared__ float red_b[MAX_THREADS];
    red_a[tid] = (tid < BLOCK_SIZE) ? acc_a : 0.f;
    red_b[tid] = (tid < BLOCK_SIZE) ? acc_b : 0.f;
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red_a[tid] += red_a[tid + s];
            red_b[tid] += red_b[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        y_a[row] = red_a[0];
        y_b[row] = red_b[0];
    }
}

// ── Tiled version for large K (better LDS utilisation) ───────────────────────
//
// Each thread handles one weight block column and accumulates across its tile.
// Better for larger models where K >> BLOCK_SIZE.

__global__ void pti_linear_tiled(
    const uint8_t* __restrict__ W,
    const float*   __restrict__ x_a,
    const float*   __restrict__ x_b,
    float*         __restrict__ y_a,
    float*         __restrict__ y_b,
    int M, int K)
{
    int row      = blockIdx.x;
    int tid      = threadIdx.x;     // each thread owns one block column
    int n_blocks = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float acc_a = 0.f, acc_b = 0.f;

    const uint8_t* row_ptr = W + (size_t)row * n_blocks * BLOCK_BYTES;

    // Each thread processes one block at a time, strided by blockDim.x
    for (int bl = tid; bl < n_blocks; bl += blockDim.x) {
        const uint8_t* blk = row_ptr + bl * BLOCK_BYTES;
        int k0 = bl * BLOCK_SIZE;

        uint16_t sa_bits = ((uint16_t)blk[0]) | ((uint16_t)blk[1] << 8);
        uint16_t sb_bits = ((uint16_t)blk[2]) | ((uint16_t)blk[3] << 8);
        float s_a = __half2float(__ushort_as_half(sa_bits));
        float s_b = __half2float(__ushort_as_half(sb_bits));

        for (int i = 0; i < BLOCK_SIZE; ++i) {
            int k = k0 + i;
            if (k >= K) break;

            uint16_t pw = ((uint16_t)blk[4 + i*2]) |
                          ((uint16_t)blk[4 + i*2 + 1] << 8);
            int8_t wa = (int8_t)((int16_t)pw >> 8);
            int8_t wb = (int8_t)(pw & 0xFF);

            acc_a += s_a * (float)wa * x_a[k];
            acc_b += s_b * (float)wb * x_b[k];
        }
    }

    // Warp reduction via LDS
    __shared__ float red_a[MAX_THREADS];
    __shared__ float red_b[MAX_THREADS];
    red_a[tid] = acc_a;
    red_b[tid] = acc_b;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red_a[tid] += red_a[tid + s];
            red_b[tid] += red_b[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        y_a[row] = red_a[0];
        y_b[row] = red_b[0];
    }
}

// ── PTI Attention Score Kernel ───────────────────────────────────────────────
//
// Computes scaled dot-product attention scores for both twins.
// Twin A attends over KV cache[0..n],   Twin B attends over KV cache[0..n+1].
//
// scores_a[h, t] = Q_a[h] · K_cache[t] / sqrt(head_dim)   for t in [0..n]
// scores_b[h, t] = Q_b[h] · K_cache[t] / sqrt(head_dim)   for t in [0..n+1]
//
// Grid:  (n_heads, seq_len_b)   — one block per (head, key position)
// Block: (head_dim,)

__global__ void pti_attn_scores(
    const float* __restrict__ Q_a,        // [n_heads, head_dim]
    const float* __restrict__ Q_b,        // [n_heads, head_dim]
    const float* __restrict__ K_cache,    // [seq_len_b, n_heads, head_dim]
    float*       __restrict__ scores_a,   // [n_heads, seq_len_a]  (seq_len_a = seq_len_b - 1)
    float*       __restrict__ scores_b,   // [n_heads, seq_len_b]
    int n_heads, int head_dim,
    int seq_len_a, int seq_len_b,
    float scale)                          // 1/sqrt(head_dim)
{
    int head = blockIdx.x;
    int t    = blockIdx.y;    // key position
    int d    = threadIdx.x;   // head dimension index

    if (head >= n_heads || d >= head_dim) return;

    __shared__ float s_qa[256];
    __shared__ float s_qb[256];
    __shared__ float s_k[256];

    s_qa[d] = Q_a[head * head_dim + d];
    s_qb[d] = Q_b[head * head_dim + d];
    s_k[d]  = K_cache[t * n_heads * head_dim + head * head_dim + d];
    __syncthreads();

    // Parallel dot product reduction
    __shared__ float dot_a[256];
    __shared__ float dot_b[256];
    dot_a[d] = s_qa[d] * s_k[d];
    dot_b[d] = s_qb[d] * s_k[d];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (d < s) {
            dot_a[d] += dot_a[d + s];
            dot_b[d] += dot_b[d + s];
        }
        __syncthreads();
    }

    if (d == 0) {
        if (t < seq_len_a)
            scores_a[head * seq_len_a + t] = dot_a[0] * scale;
        scores_b[head * seq_len_b + t]     = dot_b[0] * scale;
    }
}

// ── PTI Verification ─────────────────────────────────────────────────────────
//
// Checks whether Twin B's speculation matched Twin A's actual output.
// Returns 1 (accept) or 0 (reject) per batch item.
//
// speculation[i] = the token Twin B assumed Twin A would produce
// logits_a[i, vocab] = Twin A's logit distribution this step
//
// accept[i] = (argmax(logits_a[i]) == speculation[i]) ? 1 : 0

__global__ void pti_verify(
    const float* __restrict__ logits_a,   // [batch, vocab_size]
    const int*   __restrict__ speculation, // [batch]
    int*         __restrict__ accept,      // [batch]  out: 0 or 1
    int vocab_size)
{
    int b = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float s_max_val[MAX_THREADS];
    __shared__ int   s_max_idx[MAX_THREADS];

    const float* row = logits_a + b * vocab_size;

    // Each thread finds max over its stride
    float thread_max = -FLT_MAX;
    int   thread_idx = -1;
    for (int v = tid; v < vocab_size; v += blockDim.x) {
        if (row[v] > thread_max) {
            thread_max = row[v];
            thread_idx = v;
        }
    }
    s_max_val[tid] = thread_max;
    s_max_idx[tid] = thread_idx;
    __syncthreads();

    // Reduction: keep argmax
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_max_val[tid + s] > s_max_val[tid]) {
                s_max_val[tid] = s_max_val[tid + s];
                s_max_idx[tid] = s_max_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        accept[b] = (s_max_idx[0] == speculation[b]) ? 1 : 0;
    }
}

// ── Softmax (in-place, single row) ───────────────────────────────────────────

__global__ void pti_softmax(float* logits, int vocab_size) {
    int tid = threadIdx.x;
    __shared__ float s_max[MAX_THREADS];
    __shared__ float s_sum[MAX_THREADS];

    // Find max
    float tmax = -FLT_MAX;
    for (int v = tid; v < vocab_size; v += blockDim.x)
        tmax = fmaxf(tmax, logits[v]);
    s_max[tid] = tmax;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) s_max[tid] = fmaxf(s_max[tid], s_max[tid+s]);
        __syncthreads();
    }
    float gmax = s_max[0];

    // Exp and partial sum
    float tsum = 0.f;
    for (int v = tid; v < vocab_size; v += blockDim.x) {
        float e = expf(logits[v] - gmax);
        logits[v] = e;
        tsum += e;
    }
    s_sum[tid] = tsum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid+s];
        __syncthreads();
    }
    float gsum = s_sum[0];

    for (int v = tid; v < vocab_size; v += blockDim.x)
        logits[v] /= gsum;
}

// ── Host-side packing helper ──────────────────────────────────────────────────
//
// Packs two Q8 weight tensors into PTI uint16 format.
// Input:  scales_a[n_blocks], weights_a[n_blocks][32]  (float16 + int8)
//         scales_b[n_blocks], weights_b[n_blocks][32]
// Output: packed[n_blocks * BLOCK_BYTES]

void pti_pack_weights(
    const uint16_t* scales_a, const int8_t* weights_a,
    const uint16_t* scales_b, const int8_t* weights_b,
    uint8_t* packed, int n_blocks)
{
    for (int bl = 0; bl < n_blocks; ++bl) {
        uint8_t* dst = packed + bl * BLOCK_BYTES;
        // Scale A (f16 LE)
        dst[0] = scales_a[bl] & 0xFF;
        dst[1] = (scales_a[bl] >> 8) & 0xFF;
        // Scale B (f16 LE)
        dst[2] = scales_b[bl] & 0xFF;
        dst[3] = (scales_b[bl] >> 8) & 0xFF;
        // Weight pairs
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            uint8_t wa = (uint8_t)weights_a[bl * BLOCK_SIZE + i];
            uint8_t wb = (uint8_t)weights_b[bl * BLOCK_SIZE + i];
            dst[4 + i*2]     = wb;          // lo byte = Twin B
            dst[4 + i*2 + 1] = wa;          // hi byte = Twin A
        }
    }
}

// ── Host wrapper: single PTI forward pass ────────────────────────────────────

struct PTILinearConfig {
    int M;          // output features
    int K;          // input features
    int n_blocks;   // K / BLOCK_SIZE, rounded up
};

void pti_linear_launch(
    const uint8_t* d_W,       // packed weights on device [M × n_blocks × BLOCK_BYTES]
    const float*   d_xa,      // Twin A input on device [K]
    const float*   d_xb,      // Twin B input on device [K]
    float*         d_ya,      // Twin A output on device [M]
    float*         d_yb,      // Twin B output on device [M]
    PTILinearConfig cfg)
{
    // Choose kernel based on K
    if (cfg.K <= BLOCK_SIZE) {
        // Small K: one block per output row, threads handle the one block
        dim3 grid(cfg.M);
        dim3 block(BLOCK_SIZE);
        hipLaunchKernelGGL(pti_linear, grid, block, 0, 0,
            d_W, d_xa, d_xb, d_ya, d_yb, cfg.M, cfg.K);
    } else {
        // Large K: tiled kernel, 256 threads per row
        dim3 grid(cfg.M);
        dim3 block(256);
        hipLaunchKernelGGL(pti_linear_tiled, grid, block, 0, 0,
            d_W, d_xa, d_xb, d_ya, d_yb, cfg.M, cfg.K);
    }
}

// ── Minimal test ─────────────────────────────────────────────────────────────

#define CHECK_HIP(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error: %s at %s:%d\n", \
                hipGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

static float randf() { return (float)rand() / RAND_MAX * 2.f - 1.f; }

int main(void) {
    srand(42);

    int M = 128, K = 256;
    int n_blocks = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t w_bytes  = (size_t)M * n_blocks * BLOCK_BYTES;
    size_t x_bytes  = K * sizeof(float);
    size_t y_bytes  = M * sizeof(float);

    // ── Allocate host buffers ─────────────────────────────────────────────
    uint8_t* h_W  = (uint8_t*)malloc(w_bytes);
    float*   h_xa = (float*)  malloc(x_bytes);
    float*   h_xb = (float*)  malloc(x_bytes);
    float*   h_ya = (float*)  malloc(y_bytes);
    float*   h_yb = (float*)  malloc(y_bytes);

    // Fill with random Q8 weights and inputs
    for (size_t i = 0; i < w_bytes; ++i)  h_W[i]  = (uint8_t)(rand() & 0xFF);
    for (int i = 0; i < K; ++i)          h_xa[i] = randf();
    for (int i = 0; i < K; ++i)          h_xb[i] = randf();

    // Fix up scale bytes to be valid f16 (avoid NaN/inf)
    for (int row = 0; row < M; ++row) {
        for (int bl = 0; bl < n_blocks; ++bl) {
            uint8_t* blk = h_W + (row * n_blocks + bl) * BLOCK_BYTES;
            // Scale = 1.0 in f16 = 0x3C00
            blk[0] = 0x00; blk[1] = 0x3C;  // scale_a = 1.0
            blk[2] = 0x00; blk[3] = 0x3C;  // scale_b = 1.0
        }
    }

    // ── Allocate device buffers ───────────────────────────────────────────
    uint8_t *d_W;
    float   *d_xa, *d_xb, *d_ya, *d_yb;

    CHECK_HIP(hipMalloc(&d_W,  w_bytes));
    CHECK_HIP(hipMalloc(&d_xa, x_bytes));
    CHECK_HIP(hipMalloc(&d_xb, x_bytes));
    CHECK_HIP(hipMalloc(&d_ya, y_bytes));
    CHECK_HIP(hipMalloc(&d_yb, y_bytes));

    CHECK_HIP(hipMemcpy(d_W,  h_W,  w_bytes, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_xa, h_xa, x_bytes, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_xb, h_xb, x_bytes, hipMemcpyHostToDevice));

    // ── Launch PTI matmul ─────────────────────────────────────────────────
    PTILinearConfig cfg = {M, K, n_blocks};
    pti_linear_launch(d_W, d_xa, d_xb, d_ya, d_yb, cfg);
    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipMemcpy(h_ya, d_ya, y_bytes, hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(h_yb, d_yb, y_bytes, hipMemcpyDeviceToHost));

    // ── Verify against CPU reference ──────────────────────────────────────
    float max_err_a = 0.f, max_err_b = 0.f;
    for (int m = 0; m < M; ++m) {
        float ref_a = 0.f, ref_b = 0.f;
        for (int bl = 0; bl < n_blocks; ++bl) {
            const uint8_t* blk = h_W + (m * n_blocks + bl) * BLOCK_BYTES;
            uint16_t sa_bits = blk[0] | (blk[1] << 8);
            uint16_t sb_bits = blk[2] | (blk[3] << 8);
            float s_a = __half2float(__ushort_as_half(sa_bits));
            float s_b = __half2float(__ushort_as_half(sb_bits));
            for (int i = 0; i < BLOCK_SIZE; ++i) {
                int k = bl * BLOCK_SIZE + i;
                if (k >= K) break;
                uint8_t lo = blk[4 + i*2];
                uint8_t hi = blk[4 + i*2 + 1];
                int8_t  wa = (int8_t)hi;
                int8_t  wb = (int8_t)lo;
                ref_a += s_a * (float)wa * h_xa[k];
                ref_b += s_b * (float)wb * h_xb[k];
            }
        }
        max_err_a = fmaxf(max_err_a, fabsf(h_ya[m] - ref_a));
        max_err_b = fmaxf(max_err_b, fabsf(h_yb[m] - ref_b));
    }

    printf("PTI linear M=%d K=%d\n", M, K);
    printf("  Twin A max error vs CPU: %.6f\n", max_err_a);
    printf("  Twin B max error vs CPU: %.6f\n", max_err_b);
    printf("  %s\n", (max_err_a < 1e-2f && max_err_b < 1e-2f) ? "PASS" : "FAIL");

    // ── Timing ───────────────────────────────────────────────────────────
    hipEvent_t t0, t1;
    CHECK_HIP(hipEventCreate(&t0));
    CHECK_HIP(hipEventCreate(&t1));

    int N_ITER = 1000;
    CHECK_HIP(hipEventRecord(t0));
    for (int i = 0; i < N_ITER; ++i)
        pti_linear_launch(d_W, d_xa, d_xb, d_ya, d_yb, cfg);
    CHECK_HIP(hipEventRecord(t1));
    CHECK_HIP(hipEventSynchronize(t1));

    float ms;
    CHECK_HIP(hipEventElapsedTime(&ms, t0, t1));
    printf("\nThroughput: %.2f µs/call  (%d iterations)\n",
           ms * 1000.f / N_ITER, N_ITER);
    printf("Effective BW: %.1f GB/s  (both twins, %zu bytes weights)\n",
           (float)w_bytes * N_ITER / (ms * 1e-3f) / 1e9f, w_bytes);

    // Cleanup
    hipFree(d_W); hipFree(d_xa); hipFree(d_xb); hipFree(d_ya); hipFree(d_yb);
    free(h_W); free(h_xa); free(h_xb); free(h_ya); free(h_yb);
    hipEventDestroy(t0); hipEventDestroy(t1);

    return 0;
}
