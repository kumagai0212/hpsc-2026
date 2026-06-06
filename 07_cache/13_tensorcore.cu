#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
using namespace std;

static inline void check_cuda(cudaError_t status, const char* where) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s failed: %s\n", where, cudaGetErrorString(status));
        exit(1);
    }
}

#define M_TILE   128
#define N_TILE   128
#define K_TILE   32
#define STAGES   2

#define WARPS_M  2
#define WARPS_N  8
#define WARPS    (WARPS_M * WARPS_N)
#define THREADS  (WARPS * 32)

#define M_WARP   (M_TILE / WARPS_M)         // 64
#define N_WARP   (N_TILE / WARPS_N)         // 32

#define MMA_M    16
#define MMA_N    8
#define MMA_K    16
#define WTM      (M_WARP / MMA_M)           // 4
#define WTN      (N_WARP / MMA_N)           // 4

// wrkA[k][m], wrkB[k][n]; +8 half pad to break bank patterns
// (proper XOR swizzle comes in 18).
#define WRK_A_LD (M_TILE + 8)
#define WRK_B_LD (N_TILE + 8)

#define A_STEPS  ((K_TILE * M_TILE) / (THREADS * 8))   // 1 half8 chunk/thread
#define B_STEPS  ((N_TILE * K_TILE) / (THREADS * 8))   // 2 half8 chunks/thread

// ---- cp.async helpers ----
__device__ __forceinline__
void cp_async16(uint32_t smem_int_ptr, const void* gmem_ptr) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_int_ptr), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
__device__ __forceinline__ void cp_async_wait_lt1() {
    asm volatile("cp.async.wait_group 1;\n" ::);
}
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}

// ---- ldmatrix helpers ----
// Loads a 16x16 (4 x 8x8 sub-tiles) of fp16 from shared memory with transpose.
// Returns 4 .b32 regs per thread (= 8 halfs = 4 half2).
__device__ __forceinline__
void ldmatrix_trans_x4(uint32_t (&r)[4], uint32_t smem_int_ptr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(smem_int_ptr));
}

// ---- mma helpers ----
// D = A * B + C   (m16 n8 k16, fp32 acc, fp16 inputs)
//   A: 4 x .b32  (= 8 fp16)
//   B: 2 x .b32  (= 4 fp16)
//   C/D: 4 x .f32
__device__ __forceinline__
void mma_m16n8k16(float (&d)[4],
                  const uint32_t (&a)[4],
                  const uint32_t (&b)[2],
                  const float (&c)[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__global__ __launch_bounds__(THREADS, 1)
void sgemm_v17(int M, int N, int K,
               const half* __restrict__ A,
               const half* __restrict__ B,
               float* __restrict__ C) {
    const int bm = blockIdx.x * M_TILE;
    const int bn = blockIdx.y * N_TILE;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int wm = warp_id / WARPS_N;        // 0..1
    const int wn = warp_id % WARPS_N;        // 0..7

    extern __shared__ unsigned char smem[];
    half*  wrkA = reinterpret_cast<half*>(smem);
    half*  wrkB = wrkA + STAGES * K_TILE * WRK_A_LD;

    auto issue_load = [&](int stage, int kbase) {
        const int wrkA_base = stage * (K_TILE * WRK_A_LD);
        #pragma unroll
        for (int s = 0; s < A_STEPS; s++) {
            int idx8   = s * THREADS + tid;
            int m_off8 = idx8 & ((M_TILE / 8) - 1);
            int k_off  = idx8 / (M_TILE / 8);
            int m_off  = m_off8 * 8;
            const half* gptr = &A[(kbase + k_off) * M + bm + m_off];
            uint32_t sptr = __cvta_generic_to_shared(
                &wrkA[wrkA_base + k_off * WRK_A_LD + m_off]);
            cp_async16(sptr, gptr);
        }
        const int wrkB_base = stage * (K_TILE * WRK_B_LD);
        #pragma unroll
        for (int s = 0; s < B_STEPS; s++) {
            int idx8   = s * THREADS + tid;
            int n_off8 = idx8 & ((N_TILE / 8) - 1);
            int k_off  = idx8 / (N_TILE / 8);
            int n_off  = n_off8 * 8;
            const half* gptr = &B[(kbase + k_off) * N + bn + n_off];
            uint32_t sptr = __cvta_generic_to_shared(
                &wrkB[wrkB_base + k_off * WRK_B_LD + n_off]);
            cp_async16(sptr, gptr);
        }
    };

    // Accumulators: WTM x WTN tiles of 16x8, each held as 4 floats per thread.
    float acc[WTM][WTN][4];
    #pragma unroll
    for (int i = 0; i < WTM; i++)
        #pragma unroll
        for (int j = 0; j < WTN; j++)
            #pragma unroll
            for (int x = 0; x < 4; x++)
                acc[i][j][x] = 0.0f;

    // ---- Prologue: load tile 0 ----
    issue_load(0, 0);
    cp_async_commit();

    const int num_k_iters = K / K_TILE;
    for (int it = 0; it < num_k_iters; it++) {
        const int cur = it % STAGES;

        if (it + 1 < num_k_iters) {
            int nxt = (it + 1) % STAGES;
            issue_load(nxt, (it + 1) * K_TILE);
            cp_async_commit();
            cp_async_wait_lt1();
        } else {
            cp_async_wait_all();
        }

        __syncthreads();

        // ---- MMA over K_TILE in steps of MMA_K=16 ----
        const int wrkA_base = cur * (K_TILE * WRK_A_LD);
        const int wrkB_base = cur * (K_TILE * WRK_B_LD);

        // Compute lane-local row address used by ldmatrix.
        // ldmatrix.x4 takes 32 thread addresses (one row each, 8 halfs).
        // For matrix_a 16x16 col-major-in-shmem, with .trans we feed:
        //   lane T (0..7)  -> row offset (kk + T)   in K, col offset 0       in M
        //   lane T (8..15) -> row offset (kk + T-8) in K, col offset 8       in M
        //   lane T (16..23)-> row offset (kk + T-16 + 8) in K, col offset 0  in M
        //   lane T (24..31)-> row offset (kk + T-24 + 8) in K, col offset 8  in M
        // For matrix_b (16x16 = 2 N-tiles of 16x8 stacked horizontally in N),
        // similar pattern but col offset spans 16 N halfs.
        //
        // A is loaded once per M-tile (WTM = 4 calls per K-step).
        // B is loaded once per pair of N-tiles (WTN/2 = 2 calls per K-step).
        const int lane_row_a = (lane & 7) + ((lane >> 4) << 3);      // 0..7 or 8..15
        const int lane_col_a = ((lane >> 3) & 1) << 3;               // 0 or 8
        const int lane_row_b = (lane & 7) + ((lane >> 4) << 3);
        const int lane_col_b = ((lane >> 3) & 1) << 3;

        uint32_t a_regs[WTM][4];
        uint32_t b_regs[WTN][2];

        #pragma unroll
        for (int kk = 0; kk < K_TILE; kk += MMA_K) {
            // Load A: 4 ldmatrix.trans.x4, one per M tile.
            #pragma unroll
            for (int i = 0; i < WTM; i++) {
                int m_base = wm * M_WARP + i * MMA_M;
                int row = kk + lane_row_a;
                int col = m_base + lane_col_a;
                uint32_t sptr = __cvta_generic_to_shared(
                    &wrkA[wrkA_base + row * WRK_A_LD + col]);
                ldmatrix_trans_x4(a_regs[i], sptr);
            }

            // Load B: 2 ldmatrix.trans.x4 covering 32 N (= 4 N tiles of 8).
            //         Each x4 gives 16x16 -> we extract 2 N tiles (16x8 each).
            uint32_t b_x4[2][4];
            #pragma unroll
            for (int g = 0; g < 2; g++) {
                int n_base = wn * N_WARP + g * 16;
                int row = kk + lane_row_b;
                int col = n_base + lane_col_b;
                uint32_t sptr = __cvta_generic_to_shared(
                    &wrkB[wrkB_base + row * WRK_B_LD + col]);
                ldmatrix_trans_x4(b_x4[g], sptr);
            }
            // Re-pack into per-N-tile registers (each tile uses 2 .b32).
            // ldmatrix.x4 result for matrix_b 16x16 stored as col-major view:
            //   sub-tile 0 (regs[0]): K=0..7, N=0..7  -> N-tile 0 lower half
            //   sub-tile 1 (regs[1]): K=0..7, N=8..15 -> N-tile 1 lower half
            //   sub-tile 2 (regs[2]): K=8..15, N=0..7 -> N-tile 0 upper half
            //   sub-tile 3 (regs[3]): K=8..15, N=8..15-> N-tile 1 upper half
            // So N-tile 0 = {regs[0], regs[2]}, N-tile 1 = {regs[1], regs[3]}.
            b_regs[0][0] = b_x4[0][0]; b_regs[0][1] = b_x4[0][2];
            b_regs[1][0] = b_x4[0][1]; b_regs[1][1] = b_x4[0][3];
            b_regs[2][0] = b_x4[1][0]; b_regs[2][1] = b_x4[1][2];
            b_regs[3][0] = b_x4[1][1]; b_regs[3][1] = b_x4[1][3];

            // 16 mma calls.
            #pragma unroll
            for (int i = 0; i < WTM; i++) {
                #pragma unroll
                for (int j = 0; j < WTN; j++) {
                    mma_m16n8k16(acc[i][j], a_regs[i], b_regs[j], acc[i][j]);
                }
            }
        }
    }

    // ---- Store C ----
    // mma.m16n8k16 result layout (per thread T):
    //   d0 = D[T/4,     (T%4)*2 + 0]
    //   d1 = D[T/4,     (T%4)*2 + 1]
    //   d2 = D[T/4 + 8, (T%4)*2 + 0]
    //   d3 = D[T/4 + 8, (T%4)*2 + 1]
    // For col-major C with ld=M, C[m_row, n_col] = C[n_col * M + m_row].
    const int t_row = lane / 4;
    const int t_col = (lane & 3) * 2;
    #pragma unroll
    for (int i = 0; i < WTM; i++) {
        int m_base = bm + wm * M_WARP + i * MMA_M;
        #pragma unroll
        for (int j = 0; j < WTN; j++) {
            int n_base = bn + wn * N_WARP + j * MMA_N;
            int r0 = m_base + t_row + 0;
            int r1 = m_base + t_row + 8;
            int c0 = n_base + t_col + 0;
            int c1 = n_base + t_col + 1;
            C[(size_t)c0 * M + r0] = acc[i][j][0];
            C[(size_t)c1 * M + r0] = acc[i][j][1];
            C[(size_t)c0 * M + r1] = acc[i][j][2];
            C[(size_t)c1 * M + r1] = acc[i][j][3];
        }
    }
}

int main(int argc, const char **argv) {
    int m = 10240, k = 4096, n = 8192;
    float alpha = 1.0, beta = 0.0;
    int Nt = 10;
    float *A, *B, *C, *C2;
    half *A16, *B16_T;
    cudaMallocManaged(&A, m * k * sizeof(float));
    cudaMallocManaged(&B, k * n * sizeof(float));
    cudaMallocManaged(&C,  m * n * sizeof(float));
    cudaMallocManaged(&C2, m * n * sizeof(float));
    cudaMallocManaged(&A16, m * k * sizeof(half));
    cudaMallocManaged(&B16_T, k * n * sizeof(half));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++)
            A[k*i+j] = drand48();
    for (int i = 0; i < k; i++)
        for (int j = 0; j < n; j++)
            B[n*i+j] = drand48();
    for (int ki = 0; ki < k; ki++)
        for (int mi = 0; mi < m; mi++)
            A16[ki * m + mi] = __float2half(A[ki * m + mi]);
    for (int ki = 0; ki < k; ki++)
        for (int ni = 0; ni < n; ni++)
            B16_T[ki * n + ni] = __float2half(B[ni * k + ki]);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            C[m*i+j] = C2[m*i+j] = 0;

    cublasHandle_t handle;
    cublasCreate(&handle);

    auto tic = chrono::steady_clock::now();
    for (int i = 0; i < Nt + 2; i++) {
        if (i == 2) tic = chrono::steady_clock::now();
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                     A, CUDA_R_32F, m, B, CUDA_R_32F, k, &beta,
                     C, CUDA_R_32F, m,
                     CUBLAS_COMPUTE_32F_FAST_16F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaDeviceSynchronize();
    }
    auto toc = chrono::steady_clock::now();
    int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
    double tcublas = chrono::duration<double>(toc - tic).count() / Nt;
    double cublas_flops = double(num_flops) / tcublas / 1.0e9;

    dim3 block(THREADS);
    dim3 grid((m + M_TILE - 1) / M_TILE, (n + N_TILE - 1) / N_TILE);
    size_t smem_bytes =
          (STAGES * K_TILE * WRK_A_LD) * sizeof(half)
        + (STAGES * K_TILE * WRK_B_LD) * sizeof(half);
    int dev = 0;
    int max_smem_optin = 0;
    check_cuda(cudaGetDevice(&dev), "cudaGetDevice");
    check_cuda(cudaDeviceGetAttribute(&max_smem_optin,
                                      cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                      dev),
               "cudaDeviceGetAttribute(MaxSharedMemoryPerBlockOptin)");
    if (smem_bytes > (size_t)max_smem_optin) {
        fprintf(stderr,
                "sgemm_v17 needs %zu bytes dynamic shared memory, "
                "but this device allows %d bytes per block\n",
                smem_bytes, max_smem_optin);
        exit(1);
    }
    check_cuda(cudaFuncSetAttribute(sgemm_v17,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)smem_bytes),
               "cudaFuncSetAttribute(MaxDynamicSharedMemorySize)");
    check_cuda(cudaFuncSetAttribute(sgemm_v17,
                                    cudaFuncAttributePreferredSharedMemoryCarveout,
                                    cudaSharedmemCarveoutMaxShared),
               "cudaFuncSetAttribute(PreferredSharedMemoryCarveout)");
    for (int i = 0; i < Nt + 2; i++) {
        if (i == 2) tic = chrono::steady_clock::now();
        sgemm_v17<<<grid, block, smem_bytes>>>(m, n, k, A16, B16_T, C2);
        check_cuda(cudaGetLastError(), "sgemm_v17 launch");
        check_cuda(cudaDeviceSynchronize(), "sgemm_v17 synchronize");
    }
    toc = chrono::steady_clock::now();
    double tours = chrono::duration<double>(toc - tic).count() / Nt;
    double ours_flops = double(num_flops) / tours / 1.0e9;
    printf("CUBLAS: %.2f Gflops, CUTLASS: %.2f Gflops\n", cublas_flops, ours_flops);

    double err = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            err += fabs(C[m*i+j] - C2[m*i+j]);
    printf("error: %lf\n", err / n / m);

    cudaFree(A); cudaFree(B); cudaFree(A16); cudaFree(B16_T); cudaFree(C); cudaFree(C2);
    cublasDestroy(handle);
}
