// 13_tensorcore_v2.cu
// 13_tensorcore.cu を ldmatrix + mma.sync PTX 直叩き + XOR-swizzled shared layout に
// 書き直した版。 目的は wmma intrinsics の上にある LDS-MMA 発行ストールを抜けて
// cuBLAS により近づくこと。
//
// 主な構造:
//  - タイル: BM=128, BN=256, BK=32, STAGES=3
//  - 16 warps (4x4) / 512 threads / 1 block per SM
//  - 各 warp は WM=32, WN=64 を担当
//  - mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 を per-warp FM*FN=2*8=16 回 / kk
//  - FK = BK/16 = 2
//  - A: 共有レイアウト a_smem[STAGES][BK][BM] (k-major, m-minor) row-major
//       XOR swizzle: 物理 m チャンク = 論理 m チャンク ^ (k & 7)
//       ldmatrix.x4.trans で 1 frag = 16M x 16K
//  - B: 共有レイアウト b_smem[STAGES][BK][BN] (k-major, n-minor) row-major
//       XOR swizzle: 物理 n チャンク = 論理 n チャンク ^ (k & 7)
//       ldmatrix.x2.trans で 1 frag = 16K x 8N
//  - cp.async.cg 3 段パイプライン (current/next/next-next)
//  - inner-K register double-buffer (2 段ピンポン)
//  - ブロックスウィズル (GROUP_M=8)

#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <chrono>
using namespace std;

// ---------- Tile / warp configuration ----------
constexpr int BM = 128;
constexpr int BN = 256;
constexpr int BK = 32;
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 4;
constexpr int THREADS = 32 * WARPS_M * WARPS_N;   // 512
constexpr int WM = BM / WARPS_M;                  // 32
constexpr int WN = BN / WARPS_N;                  // 64

// mma.sync.m16n8k16 タイル形状
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;
constexpr int FM = WM / MMA_M;                    // 2
constexpr int FN = WN / MMA_N;                    // 8
constexpr int FK = BK / MMA_K;                    // 2

constexpr int STAGES = 3;

// 共有メモリの row stride。XOR swizzle で bank conflict を消すので PAD は不要。
// LDA = BM = 128 halfs = 256 bytes (16 chunks of 8 halfs)
// LDB = BN = 256 halfs = 512 bytes (32 chunks of 8 halfs)
constexpr int LDA = BM;
constexpr int LDB = BN;
constexpr int A_CHUNKS_PER_ROW = BM / 8;  // 16
constexpr int B_CHUNKS_PER_ROW = BN / 8;  // 32

// ---------- PTX helpers ----------
__device__ __forceinline__ uint32_t cvta_smem(const void *ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async_16(uint32_t smem_addr, const void *src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :: "r"(smem_addr), "l"(src));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// ldmatrix.x4.trans: 4 つの 8x8 half タイルを共有メモリから読み、転置して
// mma.sync.m16n8k16 の A オペランド (16M x 16K row-major) として配布する。
__device__ __forceinline__ void ldmatrix_x4_trans(
    uint32_t (&d)[4], uint32_t smem_addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(smem_addr));
}

// ldmatrix.x2.trans: 2 つの 8x8 half タイルを読み、 mma.sync.m16n8k16 の
// B オペランド (16K x 8N col-major) として配布する。
__device__ __forceinline__ void ldmatrix_x2_trans(
    uint32_t (&d)[2], uint32_t smem_addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
      "{%0, %1}, [%2];\n"
      : "=r"(d[0]), "=r"(d[1])
      : "r"(smem_addr));
}

// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
//   A: 16M x 16K row-major, 4 u32 regs per thread
//   B: 16K x 8N col-major, 2 u32 regs per thread
//   D = A*B + C, C/D: 16M x 8N fp32, 4 fp32 regs per thread
__device__ __forceinline__ void mma_m16n8k16(
    float (&d)[4],
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

// XOR swizzle: 行 row 内の 8-half チャンク chunk_c の物理位置 = chunk_c ^ (row & 7)
// 8 連続行で全 8 ローテーションを使い切り、 32 banks に均等分散される。
__device__ __forceinline__ int swizzle_chunk(int chunk_c, int row) {
  return chunk_c ^ (row & 7);
}

// ---------- Stage load via cp.async (HBM -> shared) ----------
__device__ __forceinline__ void stage_load(
    int tid,
    int tile_k,
    int stage,
    int dim_m,
    int dim_n,
    int offset_a_m,
    int offset_b_n,
    const half *__restrict__ d_a,    // A は M x K col-major: A[k*dim_m + m]
    const half *__restrict__ d_b,    // B_T は K x N row-major: B_T[k*dim_n + n]
    half a_smem[][BK][LDA],
    half b_smem[][BK][LDB]) {
  // A: (BK * BM) halfs を 8 halfs ずつ転送 → (BK*BM)/8 チャンク
  constexpr int A_TOTAL = (BK * BM) / 8;          // 32*128/8 = 512
  #pragma unroll
  for (int idx = tid; idx < A_TOTAL; idx += THREADS) {
    int row = idx / A_CHUNKS_PER_ROW;             // k 方向
    int chunk_c = idx % A_CHUNKS_PER_ROW;         // m 方向のチャンク (0..15)
    int phys_chunk = swizzle_chunk(chunk_c, row);
    uint32_t smem_addr = cvta_smem(&a_smem[stage][row][phys_chunk * 8]);
    cp_async_16(smem_addr,
                &d_a[(tile_k + row) * dim_m + offset_a_m + chunk_c * 8]);
  }
  // B: (BK * BN) halfs
  constexpr int B_TOTAL = (BK * BN) / 8;          // 32*256/8 = 1024
  #pragma unroll
  for (int idx = tid; idx < B_TOTAL; idx += THREADS) {
    int row = idx / B_CHUNKS_PER_ROW;             // k 方向
    int chunk_c = idx % B_CHUNKS_PER_ROW;         // n 方向のチャンク (0..31)
    int phys_chunk = swizzle_chunk(chunk_c, row);
    uint32_t smem_addr = cvta_smem(&b_smem[stage][row][phys_chunk * 8]);
    cp_async_16(smem_addr,
                &d_b[(tile_k + row) * dim_n + offset_b_n + chunk_c * 8]);
  }
  cp_async_commit();
}

// ---------- Inner-K LDS: 1 個の mma 分の A/B フラグメントを ldmatrix で読む ----------
// A: 1 frag = 16M x 16K → ldmatrix.x4.trans
// Lane t の役割:
//   matrix = t >> 3 (0..3)
//   row_in_matrix = t & 7 (0..7)
//   matrix 0: (M tile=0, K tile=0) → (M=0..7,  K=0..7)
//   matrix 1: (M tile=0, K tile=1) → (M=0..7,  K=8..15)
//   matrix 2: (M tile=1, K tile=0) → (M=8..15, K=0..7)
//   matrix 3: (M tile=1, K tile=1) → (M=8..15, K=8..15)
// → K tile = matrix & 1, M tile = matrix >> 1
__device__ __forceinline__ void load_a_frag(
    uint32_t (&dst)[4],
    half a_smem[][BK][LDA],
    int stage,
    int k_base_kk,                  // 0 or 16 within the stage
    int m_base_frag,                // warp の M ベース + r * MMA_M
    int lane) {
  int matrix     = lane >> 3;
  int row_in_mat = lane & 7;
  int k_tile     = matrix & 1;
  int m_tile     = matrix >> 1;
  int k_off      = k_base_kk + k_tile * 8 + row_in_mat;
  int m_logical  = m_base_frag + m_tile * 8;
  int chunk_log  = m_logical / 8;
  int chunk_phys = swizzle_chunk(chunk_log, k_off);
  uint32_t smem_addr = cvta_smem(&a_smem[stage][k_off][chunk_phys * 8]);
  ldmatrix_x4_trans(dst, smem_addr);
}

// B: 1 frag = 16K x 8N → ldmatrix.x2.trans
// Lane t の役割 (lane 0..15 のみアドレス有効、 16..31 のアドレスは ldmatrix.x2 で無視されるが
// 安全のため有効な共有アドレスを与える):
//   matrix 0: (K=0..7,  N=0..7)
//   matrix 1: (K=8..15, N=0..7)
// → K = (matrix & 1) * 8 + row_in_matrix
__device__ __forceinline__ void load_b_frag(
    uint32_t (&dst)[2],
    half b_smem[][BK][LDB],
    int stage,
    int k_base_kk,
    int n_base_frag,                // warp の N ベース + c * MMA_N
    int lane) {
  int lane16     = lane & 15;       // 16..31 は 0..15 として扱う (アドレス無視)
  int matrix     = lane16 >> 3;     // 0..1
  int row_in_mat = lane16 & 7;
  int k_off      = k_base_kk + matrix * 8 + row_in_mat;
  int n_logical  = n_base_frag;     // chunk 始点 (n_base_frag は 8 の倍数 = chunk-aligned)
  int chunk_log  = n_logical / 8;
  int chunk_phys = swizzle_chunk(chunk_log, k_off);
  uint32_t smem_addr = cvta_smem(&b_smem[stage][k_off][chunk_phys * 8]);
  ldmatrix_x2_trans(dst, smem_addr);
}

// ---------- メインカーネル ----------
__global__ __launch_bounds__(THREADS, 1) void kernel(
    int dim_m, int dim_n, int dim_k,
    const half *__restrict__ d_a,
    const half *__restrict__ d_b,
    float *__restrict__ d_c) {
  // ---- block swizzle (GROUP_M=8) ----
  constexpr int GROUP_M = 8;
  int num_pid_m = gridDim.x;
  int num_pid_n = gridDim.y;
  int pid = blockIdx.x + blockIdx.y * num_pid_m;
  int num_pid_in_group = GROUP_M * num_pid_n;
  int group_id = pid / num_pid_in_group;
  int first_pid_m = group_id * GROUP_M;
  int group_size_m = min(num_pid_m - first_pid_m, GROUP_M);
  int pid_m = first_pid_m + (pid % group_size_m);
  int pid_n = (pid % num_pid_in_group) / group_size_m;
  int offset_a_m = BM * pid_m;
  int offset_b_n = BN * pid_n;

  int tid = threadIdx.x;
  int warp_id = tid >> 5;
  int lane = tid & 31;
  int warp_row = warp_id / WARPS_N;   // 0..WARPS_M-1
  int warp_col = warp_id % WARPS_N;   // 0..WARPS_N-1
  int m_base_warp = warp_row * WM;
  int n_base_warp = warp_col * WN;

  __shared__ __align__(16) half a_smem[STAGES][BK][LDA];
  __shared__ __align__(16) half b_smem[STAGES][BK][LDB];

  // 累算 (per warp FM*FN tiles, 各 4 fp32/thread)
  float acc[FM][FN][4];
  #pragma unroll
  for (int r = 0; r < FM; r++)
    #pragma unroll
    for (int c = 0; c < FN; c++)
      #pragma unroll
      for (int e = 0; e < 4; e++) acc[r][c][e] = 0.0f;

  int num_tiles = dim_k / BK;

  // ---- prologue: STAGES-1 個のステージを先行投入 ----
  int prefetch = 0;
  #pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (prefetch < num_tiles) {
      stage_load(tid, prefetch * BK, s, dim_m, dim_n,
                 offset_a_m, offset_b_n, d_a, d_b, a_smem, b_smem);
      prefetch++;
    } else {
      cp_async_commit();
    }
  }

  // フラグメントのピンポンバッファ (kk と kk+1 をオーバーラップ)
  uint32_t a_frag[2][FM][4];
  uint32_t b_frag[2][FN][2];

  for (int tile = 0; tile < num_tiles; ++tile) {
    int cur = tile % STAGES;

    // 次ステージを発行
    int nxt = prefetch % STAGES;
    if (prefetch < num_tiles) {
      stage_load(tid, prefetch * BK, nxt, dim_m, dim_n,
                 offset_a_m, offset_b_n, d_a, d_b, a_smem, b_smem);
      prefetch++;
    } else {
      cp_async_commit();
    }

    cp_async_wait<STAGES - 1>();
    __syncthreads();

    // ---- inner-K register double-buffer ----
    // kk=0 のフラグメントを先読み
    #pragma unroll
    for (int r = 0; r < FM; r++) {
      load_a_frag(a_frag[0][r], a_smem, cur, 0,
                  m_base_warp + r * MMA_M, lane);
    }
    #pragma unroll
    for (int c = 0; c < FN; c++) {
      load_b_frag(b_frag[0][c], b_smem, cur, 0,
                  n_base_warp + c * MMA_N, lane);
    }

    #pragma unroll
    for (int kk = 0; kk < FK; kk++) {
      int cur_buf = kk & 1;
      int nxt_buf = cur_buf ^ 1;
      // 次の kk のフラグメントをプリロード
      if (kk + 1 < FK) {
        int kn = (kk + 1) * MMA_K;
        #pragma unroll
        for (int r = 0; r < FM; r++) {
          load_a_frag(a_frag[nxt_buf][r], a_smem, cur, kn,
                      m_base_warp + r * MMA_M, lane);
        }
        #pragma unroll
        for (int c = 0; c < FN; c++) {
          load_b_frag(b_frag[nxt_buf][c], b_smem, cur, kn,
                      n_base_warp + c * MMA_N, lane);
        }
      }
      // 現バッファで mma 発行 (FM*FN = 16 mma per kk)
      #pragma unroll
      for (int r = 0; r < FM; r++) {
        #pragma unroll
        for (int c = 0; c < FN; c++) {
          mma_m16n8k16(acc[r][c],
                       a_frag[cur_buf][r],
                       b_frag[cur_buf][c],
                       acc[r][c]);
        }
      }
    }
  }

  // ---- C への書き戻し ----
  // mma.sync.m16n8k16.f32 の結果配布:
  //   lane t の 4 fp32 = (row, col) で
  //     r[0]: (t>>2,   2*(t&3)+0)
  //     r[1]: (t>>2,   2*(t&3)+1)
  //     r[2]: (t>>2+8, 2*(t&3)+0)
  //     r[3]: (t>>2+8, 2*(t&3)+1)
  // C は col-major: d_c[c_n * dim_m + c_m]
  int lane_row = lane >> 2;
  int lane_col = (lane & 3) * 2;
  #pragma unroll
  for (int r = 0; r < FM; r++) {
    int m_base = offset_a_m + m_base_warp + r * MMA_M;
    #pragma unroll
    for (int c = 0; c < FN; c++) {
      int n_base = offset_b_n + n_base_warp + c * MMA_N;
      int m0 = m_base + lane_row;
      int m1 = m_base + lane_row + 8;
      int n0 = n_base + lane_col + 0;
      int n1 = n_base + lane_col + 1;
      if (m1 < dim_m && n1 < dim_n) {
        d_c[n0 * dim_m + m0] = acc[r][c][0];
        d_c[n1 * dim_m + m0] = acc[r][c][1];
        d_c[n0 * dim_m + m1] = acc[r][c][2];
        d_c[n1 * dim_m + m1] = acc[r][c][3];
      }
    }
  }
}

int main(int argc, const char **argv) {
  int m = 10240;
  int k = 4096;
  int n = 8192;
  float alpha = 1.0;
  float beta = 0.0;
  int Nt = 10;
  float *A, *B, *C, *C2;
  half *A16, *B16_T;
  cudaMallocManaged(&A, m * k * sizeof(float));
  cudaMallocManaged(&B, k * n * sizeof(float));
  cudaMallocManaged(&C, m * n * sizeof(float));
  cudaMallocManaged(&C2, m * n * sizeof(float));
  cudaMallocManaged(&A16, m * k * sizeof(half));
  cudaMallocManaged(&B16_T, k * n * sizeof(half));
  for (int i=0; i<m; i++)
    for (int j=0; j<k; j++)
      A[k*i+j] = drand48();
  for (int i=0; i<k; i++)
    for (int j=0; j<n; j++)
      B[n*i+j] = drand48();
  for (int i=0; i<m; i++)
    for (int j=0; j<k; j++)
      A16[k*i+j] = __float2half(A[k*i+j]);
  for (int ki=0; ki<k; ki++)
    for (int ni=0; ni<n; ni++)
      B16_T[ki*n + ni] = __float2half(B[ni*k + ki]);
  for (int i=0; i<n; i++)
    for (int j=0; j<m; j++)
      C[m*i+j] = C2[m*i+j] = 0;
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  cudaEvent_t start_event, stop_event;
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);
  float cublas_ms = 0.0f;
  cudaEventRecord(start_event);
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) cudaEventRecord(start_event);
    cublasGemmEx(cublas_handle,
                 CUBLAS_OP_N, CUBLAS_OP_N,
                 m, n, k, &alpha,
                 A, CUDA_R_32F, m,
                 B, CUDA_R_32F, k,
                 &beta,
                 C, CUDA_R_32F, m,
                 CUBLAS_COMPUTE_32F_FAST_16F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
  }
  int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
  cudaEventRecord(stop_event);
  cudaEventSynchronize(stop_event);
  cudaEventElapsedTime(&cublas_ms, start_event, stop_event);
  double tcublas = double(cublas_ms) / 1.0e3 / Nt;
  double cublas_flops = double(num_flops) / tcublas / 1.0e9;

  dim3 block = dim3(THREADS);
  dim3 grid = dim3((m + BM - 1) / BM, (n + BN - 1) / BN);
  float kernel_ms = 0.0f;
  cudaEventRecord(start_event);
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) cudaEventRecord(start_event);
    kernel<<< grid, block >>>(m, n, k, A16, B16_T, C2);
    cudaDeviceSynchronize();
  }
  cudaEventRecord(stop_event);
  cudaEventSynchronize(stop_event);
  cudaEventElapsedTime(&kernel_ms, start_event, stop_event);
  double tcutlass = double(kernel_ms) / 1.0e3 / Nt;
  double cutlass_flops = double(num_flops) / tcutlass / 1.0e9;
  printf("CUBLAS: %.2f Gflops, CUTLASS: %.2f Gflops\n", cublas_flops, cutlass_flops);
  double err = 0;
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++) {
      err += fabs(C[m*i+j] - C2[m*i+j]);
    }
  }
  printf("error: %lf\n", err/n/m);
  cudaFree(A);
  cudaFree(B);
  cudaFree(A16);
  cudaFree(B16_T);
  cudaFree(C);
  cudaFree(C2);
  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);
  cublasDestroy(cublas_handle);
}
