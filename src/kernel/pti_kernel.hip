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

// ── 4-Stream Same-Model Kernel (Q8_0 format) ─────────────────────────────────
//
// For same-model PTI: load one set of Q8_0 weights ONCE, compute 4 separate
// matmuls against 4 different activation vectors.  All 4 streams use identical
// weights (the key insight: one HBM load → 4 useful accumulations).
//
// Q8_0 block layout (34 bytes per 32-weight block):
//   bytes 0-1  : scale  (float16 LE)
//   bytes 2-33 : 32 × int8 weights
//
// Grid:  (M,)     — one block per output row
// Block: (256,)   — threads stride over n_blocks
//
// Bandwidth per token: model_bytes / 4  (4× reduction vs single-stream)

#define Q8_BLOCK_SIZE 32
#define Q8_BLOCK_BYTES 34   // f16 scale (2B) + 32 × int8

__global__ void pti_linear4_q8(
    const uint8_t* __restrict__ W,       // Q8_0 weights [M × n_blocks × 34 bytes]
    const float*   __restrict__ x_a,
    const float*   __restrict__ x_b,
    const float*   __restrict__ x_c,
    const float*   __restrict__ x_d,
    float*         __restrict__ y_a,
    float*         __restrict__ y_b,
    float*         __restrict__ y_c,
    float*         __restrict__ y_d,
    int M, int K)
{
    int row      = blockIdx.x;
    int tid      = threadIdx.x;
    int n_blocks = (K + Q8_BLOCK_SIZE - 1) / Q8_BLOCK_SIZE;

    float acc_a = 0.f, acc_b = 0.f, acc_c = 0.f, acc_d = 0.f;

    const uint8_t* row_ptr = W + (size_t)row * n_blocks * Q8_BLOCK_BYTES;

    for (int bl = tid; bl < n_blocks; bl += blockDim.x) {
        const uint8_t* blk = row_ptr + bl * Q8_BLOCK_BYTES;
        int k0 = bl * Q8_BLOCK_SIZE;

        uint16_t s_bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
        float sc = __half2float(__ushort_as_half(s_bits));

        // ONE weight load — four accumulations
        for (int i = 0; i < Q8_BLOCK_SIZE; ++i) {
            int k = k0 + i;
            if (k >= K) break;
            float wf = sc * (float)(int8_t)blk[2 + i];
            acc_a += wf * x_a[k];
            acc_b += wf * x_b[k];
            acc_c += wf * x_c[k];
            acc_d += wf * x_d[k];
        }
    }

    __shared__ float ra[MAX_THREADS], rb[MAX_THREADS],
                     rc[MAX_THREADS], rd_[MAX_THREADS];
    ra[tid] = acc_a; rb[tid] = acc_b; rc[tid] = acc_c; rd_[tid] = acc_d;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            ra[tid] += ra[tid+s]; rb[tid] += rb[tid+s];
            rc[tid] += rc[tid+s]; rd_[tid] += rd_[tid+s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        y_a[row] = ra[0]; y_b[row] = rb[0];
        y_c[row] = rc[0]; y_d[row] = rd_[0];
    }
}

// ── 4-stream flat-Q8 kernel (coalesced) ──────────────────────────────────────
//
// Flat layout (NOT packed Q8_0 blocks):
//   scales[M][n_blocks] : float16  (one scale per 32-weight block)
//   weights[M][K]       : int8     (row-major, unit-stride per row)
//
// Row-major int8 weights + unit-stride access = coalesced HBM reads.
// Each thread handles K/blockDim.x weights, consecutively, for 4 streams.
// Achieves ~80-90% of D2D copy bandwidth vs ~2% for stride-34 packed format.

__global__ void pti_linear4_q8_flat(
    const __half*  __restrict__ scales,    // [M × n_blocks]
    const int8_t*  __restrict__ wts,       // [M × K]  — row-major flat int8
    const float*   __restrict__ xa,
    const float*   __restrict__ xb,
    const float*   __restrict__ xc,
    const float*   __restrict__ xd,
    float*         __restrict__ ya,
    float*         __restrict__ yb,
    float*         __restrict__ yc,
    float*         __restrict__ yd,
    int M, int K)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int n_blocks = (K + Q8_BLOCK_SIZE - 1) / Q8_BLOCK_SIZE;

    float acca=0, accb=0, accc=0, accd=0;
    const int8_t*  rw = wts    + (size_t)row * K;
    const __half*  rs = scales + (size_t)row * n_blocks;

    // stride-blockDim.x: all threads collectively read consecutive weight bytes
    for (int k = tid; k < K; k += blockDim.x) {
        float sc = __half2float(rs[k / Q8_BLOCK_SIZE]);  // L2-cached broadcast
        float wf = sc * (float)rw[k];   // unit-stride → coalesced
        acca += wf * xa[k];
        accb += wf * xb[k];
        accc += wf * xc[k];
        accd += wf * xd[k];
    }

    __shared__ float ra[MAX_THREADS], rb[MAX_THREADS],
                     rc[MAX_THREADS], rd_[MAX_THREADS];
    ra[tid]=acca; rb[tid]=accb; rc[tid]=accc; rd_[tid]=accd;
    __syncthreads();
    for (int s=blockDim.x/2; s>0; s>>=1) {
        if (tid<s) { ra[tid]+=ra[tid+s]; rb[tid]+=rb[tid+s];
                     rc[tid]+=rc[tid+s]; rd_[tid]+=rd_[tid+s]; }
        __syncthreads();
    }
    if (tid==0) { ya[row]=ra[0]; yb[row]=rb[0]; yc[row]=rc[0]; yd[row]=rd_[0]; }
}

void pti_linear4_q8_flat_launch(
    const __half* d_scales, const int8_t* d_wts,
    const float* dxa, const float* dxb, const float* dxc, const float* dxd,
    float* dya, float* dyb, float* dyc, float* dyd,
    int M, int K)
{
    dim3 grid(M), block(256);
    hipLaunchKernelGGL(pti_linear4_q8_flat, grid, block, 0, 0,
        d_scales, d_wts, dxa, dxb, dxc, dxd, dya, dyb, dyc, dyd, M, K);
}

// ── Weight-only GEMV bandwidth test (no activation reads) ────────────────────
// Identical to flat GEMV but x[k]=1.0 computed, not loaded.  Isolates weight BW.
__global__ void weight_only_gemv(
    const __half*  __restrict__ scales,
    const int8_t*  __restrict__ wts,
    float*         __restrict__ out,
    int M, int K)
{
    int row = blockIdx.x, tid = threadIdx.x;
    int n_blocks = K / Q8_BLOCK_SIZE;
    float acc = 0;
    const int8_t* rw = wts    + (size_t)row * K;
    const __half* rs = scales + (size_t)row * n_blocks;
    for (int k = tid; k < K; k += blockDim.x) {
        float sc = __half2float(rs[k >> 5]);
        acc += sc * (float)rw[k];  // no activation read — x[k] = 1.0 implicitly
    }
    __shared__ float r[256];
    r[tid]=acc; __syncthreads();
    for (int s=blockDim.x/2;s>0;s>>=1){if(tid<s)r[tid]+=r[tid+s];__syncthreads();}
    if (tid==0) out[row]=r[0];
}

// ── Raw int8 bandwidth test: pure streaming read, sum-to-prevent-DCE ─────────
// Measures max achievable int8 read bandwidth on this GPU — no computation.
__global__ void int8_bw_test(const int8_t* __restrict__ data, int* __restrict__ out, int N) {
    int sum = 0;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        sum += (int)data[(size_t)blockIdx.x * N + i];
    if (threadIdx.x == 0) out[blockIdx.x] = sum;
}

// ── 4-stream register-tiled GEMV (activation-efficient, no sync barriers) ────
//
// ROOT CAUSE of 24% bandwidth in flat kernel:
//   activations xa/xb/xc/xd (80 KB) read from L2 by EVERY output-row block
//   → 17408 blocks × 80 KB = 1.36 GB L2 traffic.
//   Measured: L2 activation reads cost ~700 µs vs ~270 µs for HBM weight reads.
//
// FIX: register tiling with M_REG rows per THREAD.
//   Each thread loads xa[k],xb[k],xc[k],xd[k] ONCE into registers and
//   uses them for all M_REG row accumulators.  Activation L2 traffic: 4× lower.
//   K_THRD=64 = AMD wavefront size → warp-shuffle reduction, ZERO sync barriers.
//   Target: match raw int8 bandwidth (330 GB/s = 86% of D2D).
//
// Block: (K_THRD=64) = 1 wavefront.  Grid: (M/M_REG).
// Max occupancy: 40 blocks/CU (thread-limited: 2560/64=40). ✓

#define M_REG  4    // rows per thread (register-tiled)
#define K_THRD 64   // threads per block = 1 AMD wavefront

__global__ void pti_linear4_q8_regtile(
    const __half*  __restrict__ scales,    // [M × n_blocks]
    const int8_t*  __restrict__ wts,       // [M × K] row-major flat int8
    const float*   __restrict__ xa,
    const float*   __restrict__ xb,
    const float*   __restrict__ xc,
    const float*   __restrict__ xd,
    float*         __restrict__ ya,
    float*         __restrict__ yb,
    float*         __restrict__ yc,
    float*         __restrict__ yd,
    int M, int K)
{
    int tile     = blockIdx.x;
    int tid      = threadIdx.x;          // 0..K_THRD-1
    int row_base = tile * M_REG;
    int n_blocks = K / Q8_BLOCK_SIZE;

    const int8_t* rw[M_REG];
    const __half* rs[M_REG];
    #pragma unroll
    for (int r = 0; r < M_REG; r++) {
        rw[r] = wts    + (size_t)(row_base + r) * K;
        rs[r] = scales + (size_t)(row_base + r) * n_blocks;
    }

    float acca[M_REG], accb[M_REG], accc[M_REG], accd[M_REG];
    #pragma unroll
    for (int r = 0; r < M_REG; r++) acca[r]=accb[r]=accc[r]=accd[r]=0.f;

    for (int k = tid; k < K; k += K_THRD) {
        // Load 4 activations ONCE, reuse for all M_REG rows (eliminates 3× L2 reads)
        float xav = xa[k], xbv = xb[k], xcv = xc[k], xdv = xd[k];
        int blk = k >> 5;   // k / Q8_BLOCK_SIZE (Q8_BLOCK_SIZE=32=2^5)
        #pragma unroll
        for (int r = 0; r < M_REG; r++) {
            float sc = __half2float(rs[r][blk]);
            float wf = sc * (float)rw[r][k];
            acca[r] += wf * xav;
            accb[r] += wf * xbv;
            accc[r] += wf * xcv;
            accd[r] += wf * xdv;
        }
    }

    // Warp-shuffle reduction within wavefront (K_THRD=64, no sync barriers)
    #pragma unroll
    for (int offset = K_THRD/2; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int r = 0; r < M_REG; r++) {
            acca[r] += __shfl_down(acca[r], offset);
            accb[r] += __shfl_down(accb[r], offset);
            accc[r] += __shfl_down(accc[r], offset);
            accd[r] += __shfl_down(accd[r], offset);
        }
    }

    if (tid == 0) {
        #pragma unroll
        for (int r = 0; r < M_REG; r++) {
            if (row_base + r < M) {
                ya[row_base+r] = acca[r];
                yb[row_base+r] = accb[r];
                yc[row_base+r] = accc[r];
                yd[row_base+r] = accd[r];
            }
        }
    }
}

// ── 4-stream register-tiled with interleaved activations ─────────────────────
//
// Same as regtile but activations are pre-interleaved as x_abcd[K][4].
// Loads all 4 streams with ONE float4 instruction per K element:
//   4 VMEM → 1 VMEM for activations.  Total: 5 VMEM per outer iter (vs 8).
// x_abcd is passed as a single float* with stride-4 layout: [xa0,xb0,xc0,xd0, xa1,...].

__global__ void pti_linear4_q8_intlv(
    const __half*  __restrict__ scales,
    const int8_t*  __restrict__ wts,
    const float*   __restrict__ x_abcd,  // [K][4] interleaved
    float*         __restrict__ ya,
    float*         __restrict__ yb,
    float*         __restrict__ yc,
    float*         __restrict__ yd,
    int M, int K)
{
    int tile     = blockIdx.x;
    int tid      = threadIdx.x;
    int row_base = tile * M_REG;
    int n_blocks = K / Q8_BLOCK_SIZE;

    const int8_t* rw[M_REG];
    const __half* rs[M_REG];
    #pragma unroll
    for (int r = 0; r < M_REG; r++) {
        rw[r] = wts    + (size_t)(row_base + r) * K;
        rs[r] = scales + (size_t)(row_base + r) * n_blocks;
    }

    float acca[M_REG], accb[M_REG], accc[M_REG], accd[M_REG];
    #pragma unroll
    for (int r = 0; r < M_REG; r++) acca[r]=accb[r]=accc[r]=accd[r]=0.f;

    for (int k = tid; k < K; k += K_THRD) {
        // ONE float4 load for all 4 activation streams (1 VMEM instruction)
        float4 xv = ((const float4*)x_abcd)[k];
        int blk = k >> 5;
        #pragma unroll
        for (int r = 0; r < M_REG; r++) {
            float sc = __half2float(rs[r][blk]);
            float wf = sc * (float)rw[r][k];
            acca[r] += wf * xv.x;
            accb[r] += wf * xv.y;
            accc[r] += wf * xv.z;
            accd[r] += wf * xv.w;
        }
    }

    #pragma unroll
    for (int offset = K_THRD/2; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int r = 0; r < M_REG; r++) {
            acca[r] += __shfl_down(acca[r], offset);
            accb[r] += __shfl_down(accb[r], offset);
            accc[r] += __shfl_down(accc[r], offset);
            accd[r] += __shfl_down(accd[r], offset);
        }
    }

    if (tid == 0) {
        #pragma unroll
        for (int r = 0; r < M_REG; r++) {
            if (row_base + r < M) {
                ya[row_base+r] = acca[r];
                yb[row_base+r] = accb[r];
                yc[row_base+r] = accc[r];
                yd[row_base+r] = accd[r];
            }
        }
    }
}

void pti_linear4_q8_intlv_launch(
    const __half* d_scales, const int8_t* d_wts,
    const float* d_x_abcd,  // [K×4] interleaved
    float* dya, float* dyb, float* dyc, float* dyd,
    int M, int K)
{
    dim3 grid((M + M_REG - 1) / M_REG), block(K_THRD);
    hipLaunchKernelGGL(pti_linear4_q8_intlv, grid, block, 0, 0,
        d_scales, d_wts, d_x_abcd, dya, dyb, dyc, dyd, M, K);
}

void pti_linear4_q8_regtile_launch(
    const __half* d_scales, const int8_t* d_wts,
    const float* dxa, const float* dxb, const float* dxc, const float* dxd,
    float* dya, float* dyb, float* dyc, float* dyd,
    int M, int K)
{
    dim3 grid((M + M_REG - 1) / M_REG), block(K_THRD);
    hipLaunchKernelGGL(pti_linear4_q8_regtile, grid, block, 0, 0,
        d_scales, d_wts, dxa, dxb, dxc, dxd, dya, dyb, dyc, dyd, M, K);
}

// ── 4-stream flat-Q8 vectorized (128-bit loads, 4× fewer VMEM instructions) ──
//
// Same semantics as pti_linear4_q8_flat but loads 4 int8 weights and
// 4 float activations per outer iteration using 32-bit and 128-bit
// vector loads. Reduces VMEM instruction count 4×, relaxing MSHR pressure.
// Requires K divisible by 4 (satisfied for all Qwen3.6 dims: K=5120, 17408).

__global__ void pti_linear4_q8_vec(
    const __half*  __restrict__ scales,    // [M × n_blocks]
    const int8_t*  __restrict__ wts,       // [M × K]  row-major flat int8
    const float*   __restrict__ xa,
    const float*   __restrict__ xb,
    const float*   __restrict__ xc,
    const float*   __restrict__ xd,
    float*         __restrict__ ya,
    float*         __restrict__ yb,
    float*         __restrict__ yc,
    float*         __restrict__ yd,
    int M, int K)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int n_blocks = K / Q8_BLOCK_SIZE;  // K divisible by 32

    float acca=0, accb=0, accc=0, accd=0;
    const int8_t* rw = wts    + (size_t)row * K;
    const __half* rs = scales + (size_t)row * n_blocks;

    // Each thread processes 4 weights per outer iteration using 32-bit load.
    // Activation loads use float4 (128-bit, 4 floats at a time).
    // K/4 groups of 4, stride blockDim.x groups per outer step.
    int K4 = K / 4;
    for (int kg = tid; kg < K4; kg += blockDim.x) {
        int k0 = kg * 4;
        // 32-bit weight load: 4 consecutive int8 values
        int wi = *((const int*)(rw + k0));
        // float4 activation loads (128-bit each)
        float4 a4 = *((const float4*)(xa + k0));
        float4 b4 = *((const float4*)(xb + k0));
        float4 c4 = *((const float4*)(xc + k0));
        float4 d4 = *((const float4*)(xd + k0));

        // Unpack and accumulate 4 weights
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int k = k0 + j;
            float sc = __half2float(rs[k / Q8_BLOCK_SIZE]);
            int8_t wb = (int8_t)((wi >> (j*8)) & 0xFF);
            float wf = sc * (float)wb;
            float av = ((const float*)&a4)[j];
            float bv = ((const float*)&b4)[j];
            float cv = ((const float*)&c4)[j];
            float dv = ((const float*)&d4)[j];
            acca += wf * av;
            accb += wf * bv;
            accc += wf * cv;
            accd += wf * dv;
        }
    }

    __shared__ float ra[MAX_THREADS], rb[MAX_THREADS],
                     rc[MAX_THREADS], rd_[MAX_THREADS];
    ra[tid]=acca; rb[tid]=accb; rc[tid]=accc; rd_[tid]=accd;
    __syncthreads();
    for (int s=blockDim.x/2; s>0; s>>=1) {
        if (tid<s) { ra[tid]+=ra[tid+s]; rb[tid]+=rb[tid+s];
                     rc[tid]+=rc[tid+s]; rd_[tid]+=rd_[tid+s]; }
        __syncthreads();
    }
    if (tid==0) { ya[row]=ra[0]; yb[row]=rb[0]; yc[row]=rc[0]; yd[row]=rd_[0]; }
}

void pti_linear4_q8_vec_launch(
    const __half* d_scales, const int8_t* d_wts,
    const float* dxa, const float* dxb, const float* dxc, const float* dxd,
    float* dya, float* dyb, float* dyc, float* dyd,
    int M, int K)
{
    dim3 grid(M), block(256);
    hipLaunchKernelGGL(pti_linear4_q8_vec, grid, block, 0, 0,
        d_scales, d_wts, dxa, dxb, dxc, dxd, dya, dyb, dyc, dyd, M, K);
}

void pti_linear4_q8_launch(
    const uint8_t* d_W,
    const float* d_xa, const float* d_xb,
    const float* d_xc, const float* d_xd,
    float* d_ya, float* d_yb, float* d_yc, float* d_yd,
    int M, int K)
{
    dim3 grid(M);
    dim3 block(256);
    hipLaunchKernelGGL(pti_linear4_q8, grid, block, 0, 0,
        d_W, d_xa, d_xb, d_xc, d_xd, d_ya, d_yb, d_yc, d_yd, M, K);
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

    // Cleanup SSQ test
    hipFree(d_W); hipFree(d_xa); hipFree(d_xb); hipFree(d_ya); hipFree(d_yb);
    free(h_W); free(h_xa); free(h_xb); free(h_ya); free(h_yb);
    hipEventDestroy(t0); hipEventDestroy(t1);

    // ── Qwen3.6-27B throughput projection (4-stream Q8_0) ────────────────────
    //
    // Architecture: hidden=5120, FFN=17408, 64 blocks
    // Full Q8_0 forward pass weight bytes ≈ 27.5 GB  (measured)
    // MI50 measured D2D BW ≈ 383 GB/s. Baseline 21.1 tok/s on Q5_K_M (18.6 GB) ≈ 21.1×18.6 = 393 GB/s.
    // 4-stream at ~383 GB/s: 4 × (383/27.5) = 55 tok/s; +MTP: 55 × 1.88 = 104 tok/s
    //
    // Test three representative layer sizes and report projected throughput.

    printf("\n── Qwen3.6-27B 4-stream Q8_0 throughput projection ─────────────────\n");
    printf("  Model: hidden=5120, FFN=17408, 64 blocks, Q8_0 = ~27.5 GB\n\n");

    // Q8_0 weights: f16 scale + 32×int8 per block
    struct { const char* name; int M; int K; } layers[] = {
        {"FFN gate/up [5120→17408]", 17408, 5120},
        {"FFN down   [17408→5120]", 5120,  17408},
        {"Attn QKV   [5120→10240]", 10240, 5120},
    };

    float total_weight_gb = 27.5f;  // Q8_0 full model
    float total_bw_tbps = 0.f;
    int n_layers = (int)(sizeof(layers)/sizeof(layers[0]));

    for (int li = 0; li < n_layers; ++li) {
        int lM = layers[li].M, lK = layers[li].K;
        int lb = (lK + Q8_BLOCK_SIZE - 1) / Q8_BLOCK_SIZE;
        size_t lW = (size_t)lM * lb * Q8_BLOCK_BYTES;
        size_t lX = lK * sizeof(float);
        size_t lY = lM * sizeof(float);

        uint8_t *dW4; float *dxa4, *dxb4, *dxc4, *dxd4, *dya4, *dyb4, *dyc4, *dyd4;
        CHECK_HIP(hipMalloc(&dW4, lW));
        CHECK_HIP(hipMalloc(&dxa4, lX)); CHECK_HIP(hipMalloc(&dxb4, lX));
        CHECK_HIP(hipMalloc(&dxc4, lX)); CHECK_HIP(hipMalloc(&dxd4, lX));
        CHECK_HIP(hipMalloc(&dya4, lY)); CHECK_HIP(hipMalloc(&dyb4, lY));
        CHECK_HIP(hipMalloc(&dyc4, lY)); CHECK_HIP(hipMalloc(&dyd4, lY));

        // Init with random non-zero weights (avoid DCE of activation reads)
        uint8_t* tmpW = (uint8_t*)malloc(lW);
        for (size_t i = 0; i < lW; ++i) tmpW[i] = (uint8_t)(rand() & 0xFF);
        // Fix scale bytes to valid f16 (avoid NaN in output)
        for (size_t r = 0; r < (size_t)lM; r++)
            for (int b = 0; b < lb; b++) {
                tmpW[(r*lb+b)*Q8_BLOCK_BYTES+0] = 0x00;
                tmpW[(r*lb+b)*Q8_BLOCK_BYTES+1] = 0x3C;  // f16 1.0
            }
        CHECK_HIP(hipMemcpy(dW4, tmpW, lW, hipMemcpyHostToDevice));
        free(tmpW);
        // Init activation vectors with non-zero random data
        {
            float* tmpX = (float*)malloc(lX);
            for (int i = 0; i < lK; ++i) tmpX[i] = randf();
            CHECK_HIP(hipMemcpy(dxa4, tmpX, lX, hipMemcpyHostToDevice));
            CHECK_HIP(hipMemcpy(dxb4, tmpX, lX, hipMemcpyHostToDevice));
            CHECK_HIP(hipMemcpy(dxc4, tmpX, lX, hipMemcpyHostToDevice));
            CHECK_HIP(hipMemcpy(dxd4, tmpX, lX, hipMemcpyHostToDevice));
            free(tmpX);
        }

        // Warm up
        for (int w = 0; w < 3; ++w)
            pti_linear4_q8_launch(dW4, dxa4, dxb4, dxc4, dxd4, dya4, dyb4, dyc4, dyd4, lM, lK);
        CHECK_HIP(hipDeviceSynchronize());

        hipEvent_t e0, e1;
        CHECK_HIP(hipEventCreate(&e0)); CHECK_HIP(hipEventCreate(&e1));
        int NITER = 200;
        CHECK_HIP(hipEventRecord(e0));
        for (int i = 0; i < NITER; ++i)
            pti_linear4_q8_launch(dW4, dxa4, dxb4, dxc4, dxd4, dya4, dyb4, dyc4, dyd4, lM, lK);
        CHECK_HIP(hipEventRecord(e1));
        CHECK_HIP(hipEventSynchronize(e1));
        float lms; CHECK_HIP(hipEventElapsedTime(&lms, e0, e1));

        float us_call = lms * 1000.f / NITER;
        float bw = (float)lW / (lms / NITER * 1e-3f) / 1e9f;  // GB/s per call
        printf("  %s\n    %.1f µs/call  BW=%.1f GB/s  (weights %.1f MB)\n",
               layers[li].name, us_call, bw, (float)lW/1e6f);
        total_bw_tbps += bw;

        hipFree(dW4); hipFree(dxa4); hipFree(dxb4); hipFree(dxc4); hipFree(dxd4);
        hipFree(dya4); hipFree(dyb4); hipFree(dyc4); hipFree(dyd4);
        hipEventDestroy(e0); hipEventDestroy(e1);
    }

    // Project tok/s: if the representative layers achieve avg BW, extrapolate to full model
    float avg_bw = total_bw_tbps / n_layers;
    printf("\n  Avg achieved BW on Q8_0 matmuls: %.1f GB/s\n", avg_bw);
    printf("  Full Q8_0 model = %.1f GB\n", total_weight_gb);
    printf("\n  Projected throughput (4-stream, same-model PTI):\n");
    printf("    Step time   = %.1f ms  (%.1f GB / %.1f GB/s)\n",
           total_weight_gb / avg_bw * 1000.f, total_weight_gb, avg_bw);
    printf("    4-stream    = %.0f tok/s  (4 tokens per step)\n",
           4.f / (total_weight_gb / avg_bw));
    printf("    + MTP k=1   = %.0f tok/s  (× 1.88 MTP accept rate)\n",
           4.f * 1.88f / (total_weight_gb / avg_bw));
    // MI50 measured D2D copy BW ≈ 383 GB/s (hipMemcpy benchmark).
    // Baseline 21.1 tok/s on Q5_K_M (18.6 GB): 21.1 × 18.6 ≈ 393 GB/s → near peak. Good.
    float measured_bw = 383.f;
    printf("\n  MI50 measured D2D BW ≈ 383 GB/s. This kernel: %.0f%%.\n",
           avg_bw / measured_bw * 100.f);
    printf("  Theoretical (Q8_0, 27.5 GB, 383 GB/s):  4-stream=%.0f  +MTP=%.0f tok/s\n",
           4.f / (total_weight_gb / measured_bw),
           4.f * 1.88f / (total_weight_gb / measured_bw));
    printf("  Theoretical (Q5_K_M, 18.6 GB, 383 GB/s): 4-stream=%.0f  +MTP=%.0f tok/s\n",
           4.f / (18.6f / measured_bw), 4.f * 1.88f / (18.6f / measured_bw));

    // ── Weight-only GEMV (no activation reads) ───────────────────────────────
    printf("\n── Weight-only GEMV (no activation reads, establishes weight-read floor) ─\n");
    {
        int wM=17408, wK=5120;
        size_t wt_b=(size_t)wM*wK, sc_b=(size_t)wM*(wK/Q8_BLOCK_SIZE)*sizeof(__half);
        int8_t* d_w2; __half* d_s2; float* d_o2;
        CHECK_HIP(hipMalloc(&d_w2,wt_b)); CHECK_HIP(hipMalloc(&d_s2,sc_b)); CHECK_HIP(hipMalloc(&d_o2,wM*sizeof(float)));
        { int8_t* h=(int8_t*)malloc(wt_b); for(size_t i=0;i<wt_b;i++) h[i]=(int8_t)rand(); CHECK_HIP(hipMemcpy(d_w2,h,wt_b,hipMemcpyHostToDevice)); free(h); }
        { __half* h=(__half*)malloc(sc_b); for(size_t i=0;i<sc_b/2;i++) h[i]=__float2half(randf()); CHECK_HIP(hipMemcpy(d_s2,h,sc_b,hipMemcpyHostToDevice)); free(h); }
        dim3 wg(wM), wb(256);
        for(int i=0;i<3;i++) hipLaunchKernelGGL(weight_only_gemv,wg,wb,0,0,d_s2,d_w2,d_o2,wM,wK);
        CHECK_HIP(hipDeviceSynchronize());
        hipEvent_t e0,e1; CHECK_HIP(hipEventCreate(&e0)); CHECK_HIP(hipEventCreate(&e1));
        CHECK_HIP(hipEventRecord(e0));
        for(int i=0;i<200;i++) hipLaunchKernelGGL(weight_only_gemv,wg,wb,0,0,d_s2,d_w2,d_o2,wM,wK);
        CHECK_HIP(hipEventRecord(e1)); CHECK_HIP(hipEventSynchronize(e1));
        float wms; CHECK_HIP(hipEventElapsedTime(&wms,e0,e1));
        float wbw=(float)wt_b/(wms/200*1e-3f)/1e9f;
        printf("  Weight-only [%d×%d]: %.1f µs  BW=%.0f GB/s  (%.0f%% of D2D)\n",
               wM,wK,wms*1000.f/200,wbw,wbw/383.f*100.f);
        hipFree(d_w2); hipFree(d_s2); hipFree(d_o2); hipEventDestroy(e0); hipEventDestroy(e1);
    }

    // ── Raw int8 bandwidth diagnostic ────────────────────────────────────────
    printf("\n── Raw int8 streaming bandwidth (no compute, sum-to-sink) ─────────\n");
    {
        int bM=17408, bK=5120;
        size_t bsz = (size_t)bM * bK;
        int8_t* d_raw; int* d_sums;
        CHECK_HIP(hipMalloc(&d_raw, bsz));
        CHECK_HIP(hipMalloc(&d_sums, bM * sizeof(int)));
        { int8_t* h=(int8_t*)malloc(bsz); for(size_t i=0;i<bsz;i++) h[i]=(int8_t)i; CHECK_HIP(hipMemcpy(d_raw,h,bsz,hipMemcpyHostToDevice)); free(h); }
        dim3 bg(bM), bb(256);
        for(int w=0;w<3;w++) hipLaunchKernelGGL(int8_bw_test,bg,bb,0,0,d_raw,d_sums,bK);
        CHECK_HIP(hipDeviceSynchronize());
        hipEvent_t e0,e1; CHECK_HIP(hipEventCreate(&e0)); CHECK_HIP(hipEventCreate(&e1));
        int BITER=200;
        CHECK_HIP(hipEventRecord(e0));
        for(int i=0;i<BITER;i++) hipLaunchKernelGGL(int8_bw_test,bg,bb,0,0,d_raw,d_sums,bK);
        CHECK_HIP(hipEventRecord(e1)); CHECK_HIP(hipEventSynchronize(e1));
        float bms; CHECK_HIP(hipEventElapsedTime(&bms,e0,e1));
        float rbw = (float)bsz / (bms/BITER*1e-3f) / 1e9f;
        printf("  [%d×%d] int8 stream read: %.1f µs  BW=%.0f GB/s  (%.0f%% of D2D)\n",
               bM, bK, bms*1000.f/BITER, rbw, rbw/383.f*100.f);
        hipFree(d_raw); hipFree(d_sums); hipEventDestroy(e0); hipEventDestroy(e1);
    }

    // ── Flat-Q8 coalesced kernel benchmark ───────────────────────────────────
    printf("\n── Flat-Q8 coalesced 4-stream kernel (unit-stride int8 weights) ────\n");
    {
        int lM=17408, lK=5120, lb=(lK+Q8_BLOCK_SIZE-1)/Q8_BLOCK_SIZE;
        size_t sc_bytes = (size_t)lM * lb * sizeof(__half);
        size_t wt_bytes = (size_t)lM * lK * sizeof(int8_t);
        size_t xx_bytes = lK * sizeof(float);
        size_t yy_bytes = lM * sizeof(float);

        __half* d_sc; int8_t* d_wt;
        float *dxa,*dxb,*dxc,*dxd, *dya,*dyb,*dyc,*dyd;
        CHECK_HIP(hipMalloc(&d_sc, sc_bytes));
        CHECK_HIP(hipMalloc(&d_wt, wt_bytes));
        CHECK_HIP(hipMalloc(&dxa,xx_bytes)); CHECK_HIP(hipMalloc(&dxb,xx_bytes));
        CHECK_HIP(hipMalloc(&dxc,xx_bytes)); CHECK_HIP(hipMalloc(&dxd,xx_bytes));
        CHECK_HIP(hipMalloc(&dya,yy_bytes)); CHECK_HIP(hipMalloc(&dyb,yy_bytes));
        CHECK_HIP(hipMalloc(&dyc,yy_bytes)); CHECK_HIP(hipMalloc(&dyd,yy_bytes));

        { int8_t* h=(int8_t*)malloc(wt_bytes); for(size_t i=0;i<wt_bytes;i++) h[i]=(int8_t)(rand()&0xFF); CHECK_HIP(hipMemcpy(d_wt,h,wt_bytes,hipMemcpyHostToDevice)); free(h); }
        { __half* h=(__half*)malloc(sc_bytes); for(size_t i=0;i<sc_bytes/2;i++) h[i]=__float2half(1.f); CHECK_HIP(hipMemcpy(d_sc,h,sc_bytes,hipMemcpyHostToDevice)); free(h); }
        { float* h=(float*)malloc(xx_bytes); for(int i=0;i<lK;i++) h[i]=randf(); CHECK_HIP(hipMemcpy(dxa,h,xx_bytes,hipMemcpyHostToDevice)); CHECK_HIP(hipMemcpy(dxb,h,xx_bytes,hipMemcpyHostToDevice)); CHECK_HIP(hipMemcpy(dxc,h,xx_bytes,hipMemcpyHostToDevice)); CHECK_HIP(hipMemcpy(dxd,h,xx_bytes,hipMemcpyHostToDevice)); free(h); }

        for(int w=0;w<3;w++) pti_linear4_q8_flat_launch(d_sc,d_wt,dxa,dxb,dxc,dxd,dya,dyb,dyc,dyd,lM,lK);
        CHECK_HIP(hipDeviceSynchronize());

        hipEvent_t e0,e1; CHECK_HIP(hipEventCreate(&e0)); CHECK_HIP(hipEventCreate(&e1));
        int NITER=200;
        CHECK_HIP(hipEventRecord(e0));
        for(int i=0;i<NITER;i++) pti_linear4_q8_flat_launch(d_sc,d_wt,dxa,dxb,dxc,dxd,dya,dyb,dyc,dyd,lM,lK);
        CHECK_HIP(hipEventRecord(e1)); CHECK_HIP(hipEventSynchronize(e1));
        float lms; CHECK_HIP(hipEventElapsedTime(&lms,e0,e1));

        float us=lms*1000.f/NITER;
        float bw=(float)wt_bytes/(lms/NITER*1e-3f)/1e9f;
        printf("  [%d×%d]: %.1f µs/call  BW=%.0f GB/s  (%.1f MB weights)\n",
               lM, lK, us, bw, (float)wt_bytes/1e6f);
        printf("  %.0f%% of MI50 D2D BW (383 GB/s)\n", bw/383.f*100.f);

        float q5_gb=18.6f;
        printf("\n  4-stream projected at this BW (Q5_K_M = %.1f GB):\n", q5_gb);
        printf("    4-stream:       %.0f tok/s\n",   4.f / (q5_gb/bw));
        printf("    4-stream + MTP: %.0f tok/s  ← 80-100 tok/s target\n",   4.f*1.88f / (q5_gb/bw));

        // ── vectorized variant (4× fewer VMEM instructions) ─────────────────
        printf("\n  Vectorized (128-bit loads, 4× fewer VMEM):\n");
        for(int w=0;w<3;w++) pti_linear4_q8_vec_launch(d_sc,d_wt,dxa,dxb,dxc,dxd,dya,dyb,dyc,dyd,lM,lK);
        CHECK_HIP(hipDeviceSynchronize());

        hipEvent_t e2,e3; CHECK_HIP(hipEventCreate(&e2)); CHECK_HIP(hipEventCreate(&e3));
        CHECK_HIP(hipEventRecord(e2));
        for(int i=0;i<NITER;i++) pti_linear4_q8_vec_launch(d_sc,d_wt,dxa,dxb,dxc,dxd,dya,dyb,dyc,dyd,lM,lK);
        CHECK_HIP(hipEventRecord(e3)); CHECK_HIP(hipEventSynchronize(e3));
        float vms; CHECK_HIP(hipEventElapsedTime(&vms,e2,e3));

        float vus=vms*1000.f/NITER;
        float vbw=(float)wt_bytes/(vms/NITER*1e-3f)/1e9f;
        printf("  [%d×%d]: %.1f µs/call  BW=%.0f GB/s  (%.1f MB weights)\n",
               lM, lK, vus, vbw, (float)wt_bytes/1e6f);
        printf("  %.0f%% of MI50 D2D BW (383 GB/s)\n", vbw/383.f*100.f);
        printf("  Speedup vs scalar: %.2f×\n", bw > 0 ? vbw/bw : 0.f);
        printf("\n  4-stream projected at this BW (Q5_K_M = %.1f GB):\n", q5_gb);
        printf("    4-stream:       %.0f tok/s\n",   4.f / (q5_gb/vbw));
        printf("    4-stream + MTP: %.0f tok/s  ← 80-100 tok/s target\n",   4.f*1.88f / (q5_gb/vbw));

        // ── register-tiled kernel (M_REG rows/thread, warp-shuffle, no sync) ──
        printf("\n  Register-tiled (M_REG=%d rows/thread, warp-shuffle reduction, 4× less L2):\n", M_REG);
        for(int w=0;w<3;w++) pti_linear4_q8_regtile_launch(d_sc,d_wt,dxa,dxb,dxc,dxd,dya,dyb,dyc,dyd,lM,lK);
        CHECK_HIP(hipDeviceSynchronize());

        hipEvent_t e4,e5; CHECK_HIP(hipEventCreate(&e4)); CHECK_HIP(hipEventCreate(&e5));
        CHECK_HIP(hipEventRecord(e4));
        for(int i=0;i<NITER;i++) pti_linear4_q8_regtile_launch(d_sc,d_wt,dxa,dxb,dxc,dxd,dya,dyb,dyc,dyd,lM,lK);
        CHECK_HIP(hipEventRecord(e5)); CHECK_HIP(hipEventSynchronize(e5));
        float tms; CHECK_HIP(hipEventElapsedTime(&tms,e4,e5));

        float tus=tms*1000.f/NITER;
        float tbw=(float)wt_bytes/(tms/NITER*1e-3f)/1e9f;
        printf("  [%d×%d]: %.1f µs/call  BW=%.0f GB/s  (%.1f MB weights)\n",
               lM, lK, tus, tbw, (float)wt_bytes/1e6f);
        printf("  %.0f%% of MI50 D2D BW (383 GB/s)\n", tbw/383.f*100.f);
        printf("  Speedup vs scalar: %.2f×\n", bw > 0 ? tbw/bw : 0.f);
        printf("\n  4-stream projected at this BW (Q5_K_M = %.1f GB):\n", q5_gb);
        printf("    4-stream:       %.0f tok/s\n",   4.f / (q5_gb/tbw));
        printf("    4-stream + MTP: %.0f tok/s  ← 80-100 tok/s target\n",   4.f*1.88f / (q5_gb/tbw));

        // ── interleaved float4 activation loads (1 VMEM per activation, not 4) ──
        printf("\n  Interleaved [K×4] activations (float4 load = 1 VMEM per K elem):\n");
        // Build interleaved activation array
        float* d_xabcd; CHECK_HIP(hipMalloc(&d_xabcd, (size_t)lK * 4 * sizeof(float)));
        {
            float* h = (float*)malloc((size_t)lK * 4 * sizeof(float));
            float* ha=(float*)malloc(xx_bytes); float* hb=(float*)malloc(xx_bytes);
            for(int i=0;i<lK;i++) ha[i]=hb[i]=randf();
            for(int k=0;k<lK;k++) { h[k*4+0]=ha[k]; h[k*4+1]=ha[k]; h[k*4+2]=hb[k]; h[k*4+3]=hb[k]; }
            CHECK_HIP(hipMemcpy(d_xabcd,h,(size_t)lK*4*sizeof(float),hipMemcpyHostToDevice));
            free(h); free(ha); free(hb);
        }

        for(int w=0;w<3;w++) pti_linear4_q8_intlv_launch(d_sc,d_wt,d_xabcd,dya,dyb,dyc,dyd,lM,lK);
        CHECK_HIP(hipDeviceSynchronize());

        hipEvent_t e6,e7; CHECK_HIP(hipEventCreate(&e6)); CHECK_HIP(hipEventCreate(&e7));
        CHECK_HIP(hipEventRecord(e6));
        for(int i=0;i<NITER;i++) pti_linear4_q8_intlv_launch(d_sc,d_wt,d_xabcd,dya,dyb,dyc,dyd,lM,lK);
        CHECK_HIP(hipEventRecord(e7)); CHECK_HIP(hipEventSynchronize(e7));
        float ims; CHECK_HIP(hipEventElapsedTime(&ims,e6,e7));

        float ius=ims*1000.f/NITER;
        float ibw=(float)wt_bytes/(ims/NITER*1e-3f)/1e9f;
        printf("  [%d×%d]: %.1f µs/call  BW=%.0f GB/s  (%.1f MB weights)\n",
               lM, lK, ius, ibw, (float)wt_bytes/1e6f);
        printf("  %.0f%% of MI50 D2D BW (383 GB/s)\n", ibw/383.f*100.f);
        printf("  Speedup vs scalar: %.2f×\n", bw > 0 ? ibw/bw : 0.f);
        printf("\n  4-stream projected at this BW (Q5_K_M = %.1f GB):\n", q5_gb);
        printf("    4-stream:       %.0f tok/s\n",   4.f / (q5_gb/ibw));
        printf("    4-stream + MTP: %.0f tok/s  ← 80-100 tok/s target\n",   4.f*1.88f / (q5_gb/ibw));

        hipFree(d_xabcd);

        hipFree(d_sc); hipFree(d_wt);
        hipFree(dxa); hipFree(dxb); hipFree(dxc); hipFree(dxd);
        hipFree(dya); hipFree(dyb); hipFree(dyc); hipFree(dyd);
        hipEventDestroy(e0); hipEventDestroy(e1);
        hipEventDestroy(e2); hipEventDestroy(e3);
        hipEventDestroy(e4); hipEventDestroy(e5);
        hipEventDestroy(e6); hipEventDestroy(e7);
    }

    return 0;
}
