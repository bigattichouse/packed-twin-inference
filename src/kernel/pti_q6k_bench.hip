/*
 * pti_q6k_bench.hip — M6.1: standalone replica of ggml's mul_mat_vec_q<Q6_K>
 *
 * Faithful extraction of the MMVQ Q6_K path (GCN parameter table, gfx906)
 * from ggml-cuda/mmvq.cu + vecdotq.cuh, stripped of channels/samples/ids/
 * fusion (all collapse to 1/identity for our decode shapes).
 *
 * Purpose: iterate on kernel optimizations in seconds instead of rebuilding
 * llama.cpp. Valid proxy if it reproduces the in-tree ncols scaling ratios
 * (1.00× / ~1.2× / ~1.7-1.9×) within ±10%.
 *
 * Reports per ncols_dst ∈ {1,2,4,8}:
 *   - correctness vs CPU integer reference (same dp4a math)
 *   - µs/call, weight GB/s, ratio vs ncols=1
 *   - VGPR count + waves/SIMD (via hipFuncGetAttributes)
 *
 * Build:  make q6kbench       (hipcc -O3 --offload-arch=gfx906)
 * Run:    bin/pti_q6k_bench
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define CHECK_HIP(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error: %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ── Q6_K / Q8_1 formats (ggml-common.h) ──────────────────────────────────────

#define QK_K   256
#define QR6_K  2
#define QI6_K  (QK_K / (4*QR6_K))   // 32 ints of ql per block
#define QK8_1  32
#define QI8_1  (QK8_1 / 4)          // 8 ints per q8_1 block

typedef struct {
    uint8_t ql[QK_K/2];      // 128 B — lower 4 bits
    uint8_t qh[QK_K/4];      //  64 B — upper 2 bits
    int8_t  scales[QK_K/16]; //  16 B — 8-bit block scales
    __half  d;               //   2 B — super-block scale
} block_q6_K;                // 210 B per 256 weights

typedef struct {
    __half  d;               // delta
    __half  s;               // d * sum(qs)
    int8_t  qs[QK8_1];       // 32 quants
} block_q8_1;                // 36 B per 32 values

static_assert(sizeof(block_q6_K) == 210, "q6_K size");
static_assert(sizeof(block_q8_1) ==  36, "q8_1 size");

// ── Device helpers (verbatim semantics from ggml) ────────────────────────────

static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x; // 2-byte alignment
    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;
    return x32;
}

static __device__ __forceinline__ int get_int_b4(const void * x, const int & i32) {
    return ((const int *) x)[i32];
}

static __device__ __forceinline__ int dp4a(const int a, const int b, int c) {
#if defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__)
    return __builtin_amdgcn_sdot4(a, b, c, false);
#else
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif
}

static __device__ __forceinline__ int vsubss4(const int a, const int b) {
#if defined(__HIP_PLATFORM_AMD__)
    // unpack-subtract-pack with signed saturation; values here never saturate
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    int r;
    int8_t * r8 = (int8_t *) &r;
    #pragma unroll
    for (int i = 0; i < 4; i++) r8[i] = (int8_t)(a8[i] - b8[i]);
    return r;
#else
    return __vsubss4(a, b);
#endif
}

template<int width>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
    // XOR butterfly (as upstream): leaves the sum in ALL lanes — the
    // rows_per_block>1 epilogue has lane tx write tmp[j][tx], so a
    // shfl_down reduction (sum only in lane 0) corrupts odd rows.
    #pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1)
        x += __shfl_xor(x, offset, width);
    return x;
}

// ── vec_dot_q6_K_q8_1 (verbatim from vecdotq.cuh) ────────────────────────────

static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmvq(
    const int & vl, const int & vh, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d, const float * __restrict__ d8) {

    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc  = scales[4*i];
        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;
        const int vi  = vsubss4((vil | vih), 0x20202020); // (vil|vih) - 32
        sumf += d8[i] * (dp4a(vi, u[i], 0) * sc);
    }
    return d*sumf;
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q6_K * bq6_K = (const block_q6_K *) vbq + kbx;

    const int bq8_offset   = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift     = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    const int vl = get_int_b2(bq6_K->ql, iqs);
    const int vh = get_int_b2(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;

    const int8_t * scales = bq6_K->scales + scale_offset;

    int   u [QR6_K];
    float d8[QR6_K];
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        u[i]  = get_int_b4(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1);
        d8[i] = __half2float(bq8_1[bq8_offset + 2*i].d);
    }

    return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, __half2float(bq6_K->d), d8);
}

// ── Replica MMVQ kernel — GCN parameter table (gfx906) ──────────────────────
//
// GCN table (mmvq.cu): nwarps = ncols≤4 ? 2 : 1;  rows_per_block = ncols==1 ? 1 : 2
// warp_size 64;  vdr(Q6_K)=1, qi=QI6_K=32, qk=QK_K=256
// blocks_per_iter = vdr*nwarps*warp_size/qi

template <int ncols_dst>
static constexpr int calc_nwarps()        { return ncols_dst <= 4 ? 2 : 1; }
template <int ncols_dst>
static constexpr int calc_rows_per_blk()  { return ncols_dst == 1 ? 1 : 2; }

template <int ncols_dst>
__launch_bounds__(calc_nwarps<ncols_dst>()*64, 1)
__global__ void mmvq_q6k(
        const void * __restrict__ vx,   // block_q6_K [nrows × blocks_per_row]
        const void * __restrict__ vy,   // block_q8_1 [ncols_dst × stride_col_y]
        float      * __restrict__ dst,  // [ncols_dst × stride_col_dst]
        const int ncols_x,              // K
        const int stride_row_x,         // q6_K blocks per row  (K/256)
        const int stride_col_y,         // q8_1 blocks per col  (K/32)
        const int stride_col_dst,       // nrows
        const int nrows) {

    constexpr int qk  = QK_K;
    constexpr int qi  = QI6_K;
    constexpr int vdr = 1;                                  // VDR_Q6_K_Q8_1_MMVQ
    constexpr int nwarps              = calc_nwarps<ncols_dst>();
    constexpr int rows_per_cuda_block = calc_rows_per_blk<ncols_dst>();
    constexpr int warp_size           = 64;

    const     int tid  = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter  = vdr * nwarps*warp_size / qi;

    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = (const block_q8_1 *) vy;
    const int kbx_offset = row0*stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (tid % (qi/vdr));

        // M5.1 loop order: i outer, j inner (weight block address j-invariant)
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            const int cur_kbx = kbx_offset + i*stride_row_x + kbx;
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                tmp[j][i] += vec_dot_q6_K_q8_1(vx, &y[j*stride_col_y + kby], cur_kbx, kqs);
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
        }
        if (threadIdx.x < rows_per_cuda_block && (rows_per_cuda_block == 1 || row0 + threadIdx.x < (int)stride_col_dst)) {
            dst[j*stride_col_dst + row0 + threadIdx.x] = tmp[j][threadIdx.x];
        }
    }
}

// ── M6.2-V1: fused-unpack kernel ─────────────────────────────────────────────
//
// Key insight: vec_dot_q6_K_q8_1 re-does the FULL weight unpack (vl/vh loads,
// shift/mask/vsubss4 ≈ 10 VALU) for every column j, but the unpacked weights
// depend only on the block — not on j. M5.1 hoisted only the block ADDRESS.
// This kernel restructures the loop body:
//   per kbx:  load u[j][qr], d8[j][qr] once   (y data: j-dependent, i-invariant)
//   per i:    unpack vi[qr], load scales/d once (weight data: i-dependent, j-invariant)
//   per j:    tmp[j][i] += d * (d8[j][0]*(dp4a(vi0,u)*sc0) + d8[j][1]*(dp4a(vi1,u)*sc1))
// FP accumulation order per (i,j) is IDENTICAL to upstream → bit-identical results.
// VALU per (kbx,thread): ~(2 rows × 10 unpack) + (4j × 2 rows × ~3) ≈ 68
// vs upstream ~(4j × 2 rows × 13 × 2qr) ≈ 208.

template <int ncols_dst>
__launch_bounds__(calc_nwarps<ncols_dst>()*64, 1)
__global__ void mmvq_q6k_fused(
        const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
        const int ncols_x, const int stride_row_x, const int stride_col_y,
        const int stride_col_dst, const int nrows) {

    constexpr int qk  = QK_K;
    constexpr int qi  = QI6_K;
    constexpr int vdr = 1;
    constexpr int nwarps              = calc_nwarps<ncols_dst>();
    constexpr int rows_per_cuda_block = calc_rows_per_blk<ncols_dst>();
    constexpr int warp_size           = 64;

    const     int tid  = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter  = vdr * nwarps*warp_size / qi;

    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    const block_q8_1 * y = (const block_q8_1 *) vy;
    const int kbx_offset = row0*stride_row_x;

    const int kqs = vdr * (tid % (qi/vdr));    // iqs, fixed per thread
    const int bq8_offset   = 2 * QR6_K * (kqs / (QI6_K/2)) + (kqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (kqs / (QI6_K/2)) + (kqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift     = 2 * ((kqs % (QI6_K/2)) / (QI6_K/4));
    const int qh_idx       = (QI6_K/4) * (kqs / (QI6_K/2)) + kqs % (QI6_K/4);

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);

        // y loads: depend on (j, qr) only — load once, reuse for all rows
        int   u [ncols_dst][QR6_K];
        float d8[ncols_dst][QR6_K];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int qr = 0; qr < QR6_K; ++qr) {
                const block_q8_1 *b8 = &y[j*stride_col_y + kby + bq8_offset + 2*qr];
                u [j][qr] = get_int_b4(b8->qs, kqs % QI8_1);
                d8[j][qr] = __half2float(b8->d);
            }
        }

#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            const block_q6_K *bq6 = (const block_q6_K *) vx + kbx_offset + i*stride_row_x + kbx;

            // weight unpack: depends on (i, qr) only — once per row, all columns
            const int vl = get_int_b2(bq6->ql, kqs);
            const int vh = get_int_b2(bq6->qh, qh_idx) >> vh_shift;
            const float d = __half2float(bq6->d);

            int   vi[QR6_K];
            int   sc[QR6_K];
#pragma unroll
            for (int qr = 0; qr < QR6_K; ++qr) {
                const int vil = (vl >> (4*qr)) & 0x0F0F0F0F;
                const int vih = ((vh >> (4*qr)) << 4) & 0x30303030;
                vi[qr] = vsubss4((vil | vih), 0x20202020);
                sc[qr] = bq6->scales[scale_offset + 4*qr];
            }

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                float sumf = 0.0f;
#pragma unroll
                for (int qr = 0; qr < QR6_K; ++qr) {
                    sumf += d8[j][qr] * (dp4a(vi[qr], u[j][qr], 0) * sc[qr]);
                }
                tmp[j][i] += d*sumf;
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j)
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i)
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
        }
        if (threadIdx.x < rows_per_cuda_block && (rows_per_cuda_block == 1 || row0 + threadIdx.x < stride_col_dst))
            dst[j*stride_col_dst + row0 + threadIdx.x] = tmp[j][threadIdx.x];
    }
}

// ── M6.2-V2 "colwarp": column-per-block, weights shared via L2 ────────────────
//
// Inverse decomposition: each block computes ONE column at one row (the exact
// n=1 kernel body → 24 VGPRs, 10 waves/SIMD), grid = (ncols, nrows). Grid-x
// fastest → the ncols blocks for the same row are scheduled adjacently, so
// col 0 pulls each weight block from HBM and cols 1..n-1 hit it in L2.
// Trades 4× register/VALU concentration for n× L2 read amplification.
// FP order per column is IDENTICAL to the n=1 kernel → bit-identical outputs.

__launch_bounds__(2*64, 1)
__global__ void mmvq_q6k_colwarp(
        const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
        const int ncols_x, const int stride_row_x, const int stride_col_y,
        const int stride_col_dst, const int nrows) {

    constexpr int qk  = QK_K;
    constexpr int qi  = QI6_K;
    constexpr int vdr = 1;
    constexpr int nwarps    = 2;
    constexpr int warp_size = 64;

    const     int tid  = warp_size*threadIdx.y + threadIdx.x;
    const     int col  = blockIdx.x;            // column index (x = fastest)
    const     int row  = blockIdx.y;            // one row per block
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter  = vdr * nwarps*warp_size / qi;

    float tmp = 0.0f;
    const block_q8_1 * y = ((const block_q8_1 *) vy) + col*stride_col_y;
    const int kbx_offset = row*stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (tid % (qi/vdr));
        tmp += vec_dot_q6_K_q8_1(vx, &y[kby], kbx_offset + kbx, kqs);
    }

    __shared__ float tmp_shared[nwarps-1][warp_size];
    if (threadIdx.y > 0) {
        tmp_shared[threadIdx.y-1][threadIdx.x] = tmp;
    }
    __syncthreads();
    if (threadIdx.y > 0) return;

    tmp += tmp_shared[0][threadIdx.x];
    tmp  = warp_reduce_sum<warp_size>(tmp);

    if (threadIdx.x == 0) {
        dst[col*stride_col_dst + row] = tmp;
    }
}

// ── M6.2-V3 "warpcol": warp-per-column within ONE block (guaranteed L1 share) ─
//
// Each block handles one row; warp w computes column w over the FULL kbx
// range (the n=1 inner loop, 24 VGPRs). All column-warps are in the same
// block → same CU → weight reads are L1-temporal-shared by construction.
// No LDS, no __syncthreads, per-warp xor-reduce, lane 0 writes.

__global__ void mmvq_q6k_warpcol(
        const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
        const int ncols_x, const int stride_row_x, const int stride_col_y,
        const int stride_col_dst, const int nrows) {

    constexpr int qk  = QK_K;
    constexpr int qi  = QI6_K;
    constexpr int vdr = 1;
    constexpr int warp_size = 64;

    const int col  = threadIdx.y;               // warp index = column
    const int row  = blockIdx.x;
    const int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * warp_size / qi;   // 2 per warp

    float tmp = 0.0f;
    const block_q8_1 * y = ((const block_q8_1 *) vy) + col*stride_col_y;
    const int kbx_offset = row*stride_row_x;

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));
        tmp += vec_dot_q6_K_q8_1(vx, &y[kby], kbx_offset + kbx, kqs);
    }

    tmp = warp_reduce_sum<warp_size>(tmp);
    if (threadIdx.x == 0) {
        dst[col*stride_col_dst + row] = tmp;
    }
}

// ── Probe B: occupancy-forced clone (compiler must fit ≥6 waves/EU ≤42 VGPRs) ─
//
// Same body as mmvq_q6k<4>. If this is FASTER despite any spills → the n=4
// penalty is occupancy. If SLOWER → registers aren't the limiter.

__attribute__((amdgpu_waves_per_eu(6)))
__launch_bounds__(2*64, 1)
__global__ void mmvq_q6k_w6(
        const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
        const int ncols_x, const int stride_row_x, const int stride_col_y,
        const int stride_col_dst, const int nrows) {

    constexpr int ncols_dst = 4;
    constexpr int qk  = QK_K;
    constexpr int qi  = QI6_K;
    constexpr int vdr = 1;
    constexpr int nwarps              = 2;
    constexpr int rows_per_cuda_block = 2;
    constexpr int warp_size           = 64;

    const     int tid  = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter  = vdr * nwarps*warp_size / qi;

    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    const block_q8_1 * y = (const block_q8_1 *) vy;
    const int kbx_offset = row0*stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (tid % (qi/vdr));
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            const int cur_kbx = kbx_offset + i*stride_row_x + kbx;
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                tmp[j][i] += vec_dot_q6_K_q8_1(vx, &y[j*stride_col_y + kby], cur_kbx, kqs);
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1][ncols_dst][rows_per_cuda_block][warp_size];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j)
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i)
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
        }
        if (threadIdx.x < rows_per_cuda_block && row0 + threadIdx.x < stride_col_dst)
            dst[j*stride_col_dst + row0 + threadIdx.x] = tmp[j][threadIdx.x];
    }
}

// ── CPU integer reference (same math, scalar) ───────────────────────────────

static float cpu_vec_dot_q6k(const block_q6_K *bq, const block_q8_1 *y8, int iqs) {
    const int bq8_offset   = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift     = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    const uint16_t *ql16 = (const uint16_t *) bq->ql;
    const uint16_t *qh16 = (const uint16_t *) bq->qh;
    int vl = ql16[2*iqs] | (ql16[2*iqs+1] << 16);
    int qh_i = (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4);
    int vh = (int)(qh16[2*qh_i] | (qh16[2*qh_i+1] << 16)) >> vh_shift;

    float sumf = 0.f;
    for (int i = 0; i < QR6_K; ++i) {
        const int sc  = bq->scales[scale_offset + 4*i];
        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;
        int vi = vil | vih;
        const block_q8_1 *b8 = &y8[bq8_offset + 2*i];
        const int8_t *u8 = b8->qs + 4*(iqs % QI8_1);
        int dot = 0;
        for (int b = 0; b < 4; b++) {
            int8_t w = (int8_t)(((vi >> (8*b)) & 0xFF)) - 32;
            dot += (int)w * (int)u8[b];
        }
        sumf += __half2float(b8->d) * (dot * sc);
    }
    return __half2float(bq->d) * sumf;
}

// ── Benchmark driver ─────────────────────────────────────────────────────────

static float randf() { return (float)rand() / RAND_MAX * 2.f - 1.f; }

struct Shape { const char *name; int M; int K; };

template <int ncols_dst, bool FUSED = false>
static void run_one(const Shape &sh, const block_q6_K *d_x, const block_q8_1 *d_y,
                    float *d_dst, const block_q6_K *h_x, const block_q8_1 *h_y,
                    double base_us[], int do_ref) {
    const int M = sh.M, K = sh.K;
    const int blocks_per_row = K / QK_K;
    const int q8_per_col     = K / QK8_1;
    constexpr int rows_per_blk = calc_rows_per_blk<ncols_dst>();
    constexpr int nwarps       = calc_nwarps<ncols_dst>();

    dim3 grid((M + rows_per_blk - 1) / rows_per_blk);
    dim3 block(64, nwarps);

    auto launch = [&]() {
        if constexpr (FUSED) {
            hipLaunchKernelGGL((mmvq_q6k_fused<ncols_dst>), grid, block, 0, 0,
                               d_x, d_y, d_dst, K, blocks_per_row, q8_per_col, M, M);
        } else {
            hipLaunchKernelGGL((mmvq_q6k<ncols_dst>), grid, block, 0, 0,
                               d_x, d_y, d_dst, K, blocks_per_row, q8_per_col, M, M);
        }
    };

    // correctness
    if (do_ref) {
        launch();
        CHECK_HIP(hipDeviceSynchronize());
        float *h_dst = (float*)malloc((size_t)ncols_dst * M * sizeof(float));
        CHECK_HIP(hipMemcpy(h_dst, d_dst, (size_t)ncols_dst * M * sizeof(float), hipMemcpyDeviceToHost));

        double max_rel = 0.0;
        const int check_rows = 64;   // spot-check
        for (int j = 0; j < ncols_dst; j++) {
            for (int r = 0; r < check_rows; r++) {
                int row = (r * 271) % M;
                float ref = 0.f;
                for (int kbx = 0; kbx < blocks_per_row; kbx++)
                    for (int iqs = 0; iqs < QI6_K; iqs++)
                        ref += cpu_vec_dot_q6k(h_x + (size_t)row*blocks_per_row + kbx,
                                               h_y + (size_t)j*q8_per_col + kbx*(QK_K/QK8_1), iqs);
                float got = h_dst[(size_t)j*M + row];
                double rel = fabs(got - ref) / fmax(1.0, fmax(fabs(got), fabs(ref)));
                if (rel > max_rel) max_rel = rel;
            }
        }
        free(h_dst);
        printf("    %s ncols=%d correctness: max_rel=%.2e %s\n",
               FUSED ? "fused" : "base ", ncols_dst, max_rel, max_rel < 5e-3 ? "PASS" : "FAIL");
    }

    // VGPR / occupancy
    hipFuncAttributes attr;
    const void *kfn = FUSED ? reinterpret_cast<const void*>(&mmvq_q6k_fused<ncols_dst>)
                            : reinterpret_cast<const void*>(&mmvq_q6k<ncols_dst>);
    CHECK_HIP(hipFuncGetAttributes(&attr, kfn));
    int waves = 256 / (attr.numRegs > 0 ? attr.numRegs : 256);

    // timing
    const int NITER = 200;
    for (int w = 0; w < 5; w++) launch();
    CHECK_HIP(hipDeviceSynchronize());

    hipEvent_t e0, e1;
    CHECK_HIP(hipEventCreate(&e0)); CHECK_HIP(hipEventCreate(&e1));
    CHECK_HIP(hipEventRecord(e0));
    for (int i = 0; i < NITER; i++) launch();
    CHECK_HIP(hipEventRecord(e1));
    CHECK_HIP(hipEventSynchronize(e1));
    float ms; CHECK_HIP(hipEventElapsedTime(&ms, e0, e1));
    hipEventDestroy(e0); hipEventDestroy(e1);

    double us = ms * 1000.0 / NITER;
    size_t w_bytes = (size_t)M * blocks_per_row * sizeof(block_q6_K);
    double bw = (double)w_bytes / (us * 1e-6) / 1e9;

    if (ncols_dst == 1 && !FUSED) base_us[0] = us;
    printf("    %s ncols=%d: %8.1f µs   %6.1f GB/s   %.2f×    VGPRs=%-3d waves/SIMD=%d\n",
           FUSED ? "fused" : "base ", ncols_dst, us, bw, us / base_us[0], attr.numRegs, waves);
}

static void run_colwarp(int ncols, const Shape &sh, const block_q6_K *d_x, const block_q8_1 *d_y,
                        float *d_dst, const block_q6_K *h_x, const block_q8_1 *h_y,
                        double base_us[], bool warpcol = false) {
    const int M = sh.M, K = sh.K;
    const int blocks_per_row = K / QK_K;
    const int q8_per_col     = K / QK8_1;

    dim3 grid  = warpcol ? dim3(M)         : dim3(ncols, M);
    dim3 block = warpcol ? dim3(64, ncols) : dim3(64, 2);
    const char *tag = warpcol ? "warpc" : "colwp";

    auto launch = [&]() {
        if (warpcol) {
            hipLaunchKernelGGL(mmvq_q6k_warpcol, grid, block, 0, 0,
                               d_x, d_y, d_dst, K, blocks_per_row, q8_per_col, M, M);
        } else {
            hipLaunchKernelGGL(mmvq_q6k_colwarp, grid, block, 0, 0,
                               d_x, d_y, d_dst, K, blocks_per_row, q8_per_col, M, M);
        }
    };

    // correctness (spot-check, all columns)
    launch();
    CHECK_HIP(hipDeviceSynchronize());
    float *h_dst = (float*)malloc((size_t)ncols * M * sizeof(float));
    CHECK_HIP(hipMemcpy(h_dst, d_dst, (size_t)ncols * M * sizeof(float), hipMemcpyDeviceToHost));
    double max_rel = 0.0;
    for (int j = 0; j < ncols; j++) {
        for (int r = 0; r < 64; r++) {
            int row = (r * 271) % M;
            float ref = 0.f;
            for (int kbx = 0; kbx < blocks_per_row; kbx++)
                for (int iqs = 0; iqs < QI6_K; iqs++)
                    ref += cpu_vec_dot_q6k(h_x + (size_t)row*blocks_per_row + kbx,
                                           h_y + (size_t)j*q8_per_col + kbx*(QK_K/QK8_1), iqs);
            float got = h_dst[(size_t)j*M + row];
            double rel = fabs(got - ref) / fmax(1.0, fmax(fabs(got), fabs(ref)));
            if (rel > max_rel) max_rel = rel;
        }
    }
    free(h_dst);
    printf("    %s ncols=%d correctness: max_rel=%.2e %s\n",
           tag, ncols, max_rel, max_rel < 5e-3 ? "PASS" : "FAIL");

    hipFuncAttributes attr;
    const void *kfn = warpcol ? reinterpret_cast<const void*>(&mmvq_q6k_warpcol)
                              : reinterpret_cast<const void*>(&mmvq_q6k_colwarp);
    CHECK_HIP(hipFuncGetAttributes(&attr, kfn));

    const int NITER = 200;
    for (int w = 0; w < 5; w++) launch();
    CHECK_HIP(hipDeviceSynchronize());
    hipEvent_t e0, e1;
    CHECK_HIP(hipEventCreate(&e0)); CHECK_HIP(hipEventCreate(&e1));
    CHECK_HIP(hipEventRecord(e0));
    for (int i = 0; i < NITER; i++) launch();
    CHECK_HIP(hipEventRecord(e1)); CHECK_HIP(hipEventSynchronize(e1));
    float ms; CHECK_HIP(hipEventElapsedTime(&ms, e0, e1));
    hipEventDestroy(e0); hipEventDestroy(e1);

    double us = ms * 1000.0 / NITER;
    size_t w_bytes = (size_t)M * blocks_per_row * sizeof(block_q6_K);
    double bw = (double)w_bytes / (us * 1e-6) / 1e9;
    int waves = 256 / (attr.numRegs > 0 ? attr.numRegs : 256);
    printf("    %s ncols=%d: %8.1f µs   %6.1f GB/s   %.2f×    VGPRs=%-3d waves/SIMD=%d\n",
           tag, ncols, us, bw, us / base_us[0], attr.numRegs, waves);
}

int main(void) {
    srand(42);

    Shape shapes[] = {
        {"FFN gate/up [17408 × 5120]", 17408, 5120},
        {"FFN down    [5120 × 17408]", 5120,  17408},
        {"Attn QKV    [10240 × 5120]", 10240, 5120},
    };

    printf("M6.1 — Q6_K MMVQ replica (GCN table, gfx906)\n");
    printf("════════════════════════════════════════════════════════════════════\n");

    for (size_t si = 0; si < sizeof(shapes)/sizeof(shapes[0]); si++) {
        const Shape &sh = shapes[si];
        const int M = sh.M, K = sh.K;
        const int blocks_per_row = K / QK_K;
        const int q8_per_col     = K / QK8_1;
        const int max_cols       = 8;

        size_t x_bytes = (size_t)M * blocks_per_row * sizeof(block_q6_K);
        size_t y_bytes = (size_t)max_cols * q8_per_col * sizeof(block_q8_1);
        size_t d_bytes = (size_t)max_cols * M * sizeof(float);

        block_q6_K *h_x = (block_q6_K*)malloc(x_bytes);
        block_q8_1 *h_y = (block_q8_1*)malloc(y_bytes);

        // random Q6_K blocks
        for (size_t b = 0; b < (size_t)M * blocks_per_row; b++) {
            for (int i = 0; i < QK_K/2; i++)  h_x[b].ql[i] = (uint8_t)(rand() & 0xFF);
            for (int i = 0; i < QK_K/4; i++)  h_x[b].qh[i] = (uint8_t)(rand() & 0xFF);
            for (int i = 0; i < QK_K/16; i++) h_x[b].scales[i] = (int8_t)((rand() % 64) - 32);
            h_x[b].d = __float2half(0.001f + 0.01f * (float)rand() / RAND_MAX);
        }
        // random q8_1 activations
        for (size_t b = 0; b < (size_t)max_cols * q8_per_col; b++) {
            int sum = 0;
            for (int i = 0; i < QK8_1; i++) { h_y[b].qs[i] = (int8_t)((rand() % 255) - 127); sum += h_y[b].qs[i]; }
            float d = 0.01f + 0.02f * (float)rand() / RAND_MAX;
            h_y[b].d = __float2half(d);
            h_y[b].s = __float2half(d * sum);
        }

        block_q6_K *d_x; block_q8_1 *d_y; float *d_dst;
        CHECK_HIP(hipMalloc(&d_x, x_bytes));
        CHECK_HIP(hipMalloc(&d_y, y_bytes));
        CHECK_HIP(hipMalloc(&d_dst, d_bytes));
        CHECK_HIP(hipMemcpy(d_x, h_x, x_bytes, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_y, h_y, y_bytes, hipMemcpyHostToDevice));

        printf("\n  %s  (weights %.1f MB)\n", sh.name, x_bytes / 1e6);

        double base_us[1] = {1.0};
        int do_ref = 1;   // CPU spot-check on every shape
        run_one<1>(sh, d_x, d_y, d_dst, h_x, h_y, base_us, do_ref);
        run_one<2>(sh, d_x, d_y, d_dst, h_x, h_y, base_us, do_ref);
        run_one<4>(sh, d_x, d_y, d_dst, h_x, h_y, base_us, do_ref);
        run_one<8>(sh, d_x, d_y, d_dst, h_x, h_y, base_us, do_ref);
        run_one<2, true>(sh, d_x, d_y, d_dst, h_x, h_y, base_us, do_ref);
        run_one<4, true>(sh, d_x, d_y, d_dst, h_x, h_y, base_us, do_ref);
        run_colwarp(2, sh, d_x, d_y, d_dst, h_x, h_y, base_us, true);
        run_colwarp(4, sh, d_x, d_y, d_dst, h_x, h_y, base_us, true);
        run_colwarp(8, sh, d_x, d_y, d_dst, h_x, h_y, base_us, true);

        // ── Diagnostic probes at ncols=4 ────────────────────────────────────
        {
            const int NITER = 200;
            dim3 grid((M + 1) / 2), block(64, 2);
            hipEvent_t e0, e1;
            CHECK_HIP(hipEventCreate(&e0)); CHECK_HIP(hipEventCreate(&e1));

            // Probe A: all 4 columns share one y (stride_col_y=0) — isolates
            // activation L2 traffic. Timing-only (output is 4 identical cols).
            for (int w = 0; w < 5; w++)
                hipLaunchKernelGGL((mmvq_q6k<4>), grid, block, 0, 0,
                                   d_x, d_y, d_dst, K, blocks_per_row, 0, M, M);
            CHECK_HIP(hipDeviceSynchronize());
            CHECK_HIP(hipEventRecord(e0));
            for (int i = 0; i < NITER; i++)
                hipLaunchKernelGGL((mmvq_q6k<4>), grid, block, 0, 0,
                                   d_x, d_y, d_dst, K, blocks_per_row, 0, M, M);
            CHECK_HIP(hipEventRecord(e1)); CHECK_HIP(hipEventSynchronize(e1));
            float ms_a; CHECK_HIP(hipEventElapsedTime(&ms_a, e0, e1));

            // Probe B: occupancy-forced clone (≥6 waves/EU)
            hipFuncAttributes attr;
            CHECK_HIP(hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(&mmvq_q6k_w6)));
            for (int w = 0; w < 5; w++)
                hipLaunchKernelGGL(mmvq_q6k_w6, grid, block, 0, 0,
                                   d_x, d_y, d_dst, K, blocks_per_row, q8_per_col, M, M);
            CHECK_HIP(hipDeviceSynchronize());
            CHECK_HIP(hipEventRecord(e0));
            for (int i = 0; i < NITER; i++)
                hipLaunchKernelGGL(mmvq_q6k_w6, grid, block, 0, 0,
                                   d_x, d_y, d_dst, K, blocks_per_row, q8_per_col, M, M);
            CHECK_HIP(hipEventRecord(e1)); CHECK_HIP(hipEventSynchronize(e1));
            float ms_b; CHECK_HIP(hipEventElapsedTime(&ms_b, e0, e1));

            printf("    probe A (shared-y, L2 test):   %8.1f µs   %.2f× vs n=1\n",
                   ms_a*1000.0/NITER, ms_a*1000.0/NITER / base_us[0]);
            printf("    probe B (waves_per_eu=6):      %8.1f µs   %.2f× vs n=1   VGPRs=%d spills=%s\n",
                   ms_b*1000.0/NITER, ms_b*1000.0/NITER / base_us[0],
                   attr.numRegs, attr.localSizeBytes > 0 ? "YES" : "no");
            hipEventDestroy(e0); hipEventDestroy(e1);
        }

        hipFree(d_x); hipFree(d_y); hipFree(d_dst);
        free(h_x); free(h_y);
    }

    printf("\n════════════════════════════════════════════════════════════════════\n");
    printf("Gate (M6.1): ncols=2 ≈ 1.2×, ncols=4 ≈ 1.7-1.9× (within ±10%% of\n");
    printf("in-tree ratios) → replica is a valid proxy for M6.2 iteration.\n");
    return 0;
}
