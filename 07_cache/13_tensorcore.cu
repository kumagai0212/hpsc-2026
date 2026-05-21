#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <chrono>
using namespace std;
using namespace nvcuda;

// ---------- Tile / warp configuration ----------
// 1ブロックで 128x256 の C を担当 (M x N)。
// ブロックは 16 warps = 512 threads。 warp grid は 2 (M) x 8 (N)。
// 各 warp は 64x32 の C を担当 = 4x2 個の 16x16 WMMA フラグメント。
// 算術強度向上(64→85 FLOPs/B)とワープ並列度(8→16)で発行ストールを隠す。
constexpr int BM = 128;
constexpr int BN = 256;
constexpr int BK = 32;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 8;
constexpr int THREADS = 32 * WARPS_M * WARPS_N;   // 512
constexpr int WM = BM / WARPS_M;                  // 64
constexpr int WN = BN / WARPS_N;                  // 32
constexpr int FM = WM / 16;                       // 4
constexpr int FN = WN / 16;                       // 2
constexpr int FK = BK / 16;                       // 2
// 共有メモリ: stage A = 32*(128+8)*2 = 8704, stage B = 32*(256+8)*2 = 16896
// 3 stages 合計 ≈ 76800 bytes (≈75KB) で静的 __shared__ の範囲内に収まる。
constexpr int STAGES = 3;
// shared memory のバンク競合を避けるためのパディング (8 halfs = 16 bytes)
constexpr int PAD = 8;
constexpr int LDA = BM + PAD;
constexpr int LDB = BN + PAD;

__device__ __forceinline__ void cp_async_16(void *dst, const void *src) {
  unsigned int smem_addr = static_cast<unsigned int>(__cvta_generic_to_shared(dst));
  // cg (cache global only): L1 を経由せず L2 から直接 shared に流すので
  // streaming な GEMM 用途では一般的に ca より高速。
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :: "r"(smem_addr), "l"(src));
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;");
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void stage_load_cp_async(
    int tid,
    int tile_k,
    int stage,
    int dim_m,
    int dim_n,
    int offset_a_m,
    int offset_b_n,
    const half *__restrict__ d_a,
    const half *__restrict__ d_b,
    half block_a[][BK][LDA],
    half block_b[][BK][LDB]) {
  // A: M x K 列優先 (k 番目の列が連続) なので、shared[ki][mi] = d_a[(tile_k+ki)*dim_m + offset_a_m + mi]
  // 1 thread = 16 bytes (8 halfs) を 1 命令でロード。
  // BK*BM/8 = 32*128/8 = 512 chunks / 256 threads = 2 chunks per thread。
  constexpr int A_CHUNKS = (BK * BM) / 8;
  constexpr int A_COLS = BM / 8;            // 1 行あたりの chunk 数
  #pragma unroll
  for (int idx = tid; idx < A_CHUNKS; idx += THREADS) {
    int row = idx / A_COLS;
    int col = (idx % A_COLS) * 8;
    cp_async_16(&block_a[stage][row][col],
                &d_a[(tile_k + row) * dim_m + offset_a_m + col]);
  }
  // B (転置済み): K x N 行優先 (n 番目の行が連続)
  constexpr int B_CHUNKS = (BK * BN) / 8;
  constexpr int B_COLS = BN / 8;
  #pragma unroll
  for (int idx = tid; idx < B_CHUNKS; idx += THREADS) {
    int row = idx / B_COLS;
    int col = (idx % B_COLS) * 8;
    cp_async_16(&block_b[stage][row][col],
                &d_b[(tile_k + row) * dim_n + offset_b_n + col]);
  }
  cp_async_commit_group();
}

__global__ __launch_bounds__(THREADS, 1) void kernel(
    int dim_m, int dim_n, int dim_k,
    const half *__restrict__ d_a,
    const half *__restrict__ d_b,
    float *__restrict__ d_c) {
  int offset_a_m = BM * blockIdx.x;
  int offset_b_n = BN * blockIdx.y;
  int tid = threadIdx.x;
  int warp_id = tid >> 5;
  int warp_row = warp_id / WARPS_N;   // 0..WARPS_M-1
  int warp_col = warp_id % WARPS_N;   // 0..WARPS_N-1

  __shared__ __align__(16) half block_a[STAGES][BK][LDA];
  __shared__ __align__(16) half block_b[STAGES][BK][LDB];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FM][FN];
  #pragma unroll
  for (int r = 0; r < FM; r++)
    #pragma unroll
    for (int c = 0; c < FN; c++)
      wmma::fill_fragment(acc[r][c], 0.0f);

  int num_tiles = dim_k / BK;

  // ---- prologue: STAGES-1 個のステージを先行投入 ----
  int prefetch = 0;
  #pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (prefetch < num_tiles) {
      stage_load_cp_async(tid, prefetch * BK, s, dim_m, dim_n,
                          offset_a_m, offset_b_n, d_a, d_b, block_a, block_b);
      prefetch++;
    } else {
      cp_async_commit_group();
    }
  }

  for (int tile = 0; tile < num_tiles; ++tile) {
    int cur = tile % STAGES;

    // 次のステージを発行 (パイプラインを満たし続ける)
    int nxt = prefetch % STAGES;
    if (prefetch < num_tiles) {
      stage_load_cp_async(tid, prefetch * BK, nxt, dim_m, dim_n,
                          offset_a_m, offset_b_n, d_a, d_b, block_a, block_b);
      prefetch++;
    } else {
      cp_async_commit_group();
    }

    // 現在のステージが書き込み終わるのを待つ。
    // STAGES-1 個のグループが in-flight として残っている状態にする。
    cp_async_wait_group<STAGES - 1>();
    __syncthreads();

    int base_a_col = warp_row * WM;   // この warp の M オフセット
    int base_b_col = warp_col * WN;   // この warp の N オフセット

    // ---- inner-K register double-buffer ----
    // FK=2 でも、フラグメント LDS と mma_sync をオーバーラップさせるために
    // ピンポンバッファで kk と kk+1 を重ねて発行する。
    // テンソルコア発行ストールが短縮され、 Compute Throughput が上がる狙い。
    using FragA = wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major>;
    using FragB = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>;
    FragA a_frag[2][FM];
    FragB b_frag[2][FN];

    // kk = 0 のフラグメントを先読み
    #pragma unroll
    for (int c = 0; c < FN; c++) {
      wmma::load_matrix_sync(b_frag[0][c],
          &block_b[cur][0 * 16][base_b_col + c * 16], LDB);
    }
    #pragma unroll
    for (int r = 0; r < FM; r++) {
      wmma::load_matrix_sync(a_frag[0][r],
          &block_a[cur][0 * 16][base_a_col + r * 16], LDA);
    }

    #pragma unroll
    for (int kk = 0; kk < FK; kk++) {
      int cur_buf = kk & 1;
      int nxt_buf = cur_buf ^ 1;
      // 次の kk のフラグメントをプリロード（最後のイテレーションでは省略）
      if (kk + 1 < FK) {
        int kn = (kk + 1) * 16;
        #pragma unroll
        for (int c = 0; c < FN; c++) {
          wmma::load_matrix_sync(b_frag[nxt_buf][c],
              &block_b[cur][kn][base_b_col + c * 16], LDB);
        }
        #pragma unroll
        for (int r = 0; r < FM; r++) {
          wmma::load_matrix_sync(a_frag[nxt_buf][r],
              &block_a[cur][kn][base_a_col + r * 16], LDA);
        }
      }
      // 現バッファで mma 発行（プリロード LDS と発行をオーバーラップ）
      #pragma unroll
      for (int r = 0; r < FM; r++) {
        #pragma unroll
        for (int c = 0; c < FN; c++) {
          wmma::mma_sync(acc[r][c], a_frag[cur_buf][r], b_frag[cur_buf][c], acc[r][c]);
        }
      }
    }
  }

  // ---- C への書き戻し ----
  #pragma unroll
  for (int r = 0; r < FM; r++) {
    #pragma unroll
    for (int c = 0; c < FN; c++) {
      int c_m = offset_a_m + warp_row * WM + r * 16;
      int c_n = offset_b_n + warp_col * WN + c * 16;
      if (c_n < dim_n && c_m < dim_m)
        wmma::store_matrix_sync(&d_c[c_n * dim_m + c_m],
                                acc[r][c], dim_m, wmma::mem_col_major);
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
  half *A16, *B16_T;  // B16_T: K×N row-majorに転置したhalf行列
  cudaMallocManaged(&A, m * k * sizeof(float));
  cudaMallocManaged(&B, k * n * sizeof(float));
  cudaMallocManaged(&C, m * n * sizeof(float));
  cudaMallocManaged(&C2, m * n * sizeof(float));
  cudaMallocManaged(&A16, m * k * sizeof(half));
  cudaMallocManaged(&B16_T, k * n * sizeof(half));
  auto init_tic = chrono::steady_clock::now();
  for (int i=0; i<m; i++)
    for (int j=0; j<k; j++)
      A[k*i+j] = drand48();
  for (int i=0; i<k; i++)
    for (int j=0; j<n; j++)
      B[n*i+j] = drand48();
  auto init_toc = chrono::steady_clock::now();

  auto convert_a_tic = chrono::steady_clock::now();
  // 変更点: half入力を一度だけ作成し、計測中の型変換コストを排除
  for (int i=0; i<m; i++)
    for (int j=0; j<k; j++)
      A16[k*i+j] = __float2half(A[k*i+j]);
  auto convert_a_toc = chrono::steady_clock::now();

  auto convert_b_tic = chrono::steady_clock::now();
  // 変更点: Bをまとめてfloat->half変換しつつK×N row-majorに転置。
  // 元のBはcuBLASの列優先レイアウトでB[n][k]=B_ptr[n*k+k_idx]だったが、
  // 転置後はB16_T[k][n]=B16_T[ki*n+ni]とし、カーネル内でcoalescedに読める。
  for (int ki=0; ki<k; ki++)
    for (int ni=0; ni<n; ni++)
      B16_T[ki*n + ni] = __float2half(B[ni*k + ki]);
  auto convert_b_toc = chrono::steady_clock::now();
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
    if (i == 2) {
      cudaEventRecord(start_event);
    }
    cublasGemmEx(cublas_handle,
		 CUBLAS_OP_N,
		 CUBLAS_OP_N,
		 m,
		 n,
		 k,
		 &alpha,
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
  // 1 ブロックで 128x128 タイルを処理。 256 threads = 8 warps。
  dim3 block = dim3(THREADS);
  dim3 grid = dim3((m + BM - 1) / BM, (n + BN - 1) / BN);
  float kernel_ms = 0.0f;
  cudaEventRecord(start_event);
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) {
      cudaEventRecord(start_event);
    }
    kernel<<< grid, block >>>(m,
			      n,
			      k,
            A16,
            B16_T,  // 転置済みK×N half行列を渡す
			      C2);
    cudaDeviceSynchronize();
  }
  cudaEventRecord(stop_event);
  cudaEventSynchronize(stop_event);
  cudaEventElapsedTime(&kernel_ms, start_event, stop_event);
  double tcutlass = double(kernel_ms) / 1.0e3 / Nt;
  double cutlass_flops = double(num_flops) / tcutlass / 1.0e9;
  double init_seconds = chrono::duration<double>(init_toc - init_tic).count();
  double convert_a_seconds = chrono::duration<double>(convert_a_toc - convert_a_tic).count();
  double convert_b_seconds = chrono::duration<double>(convert_b_toc - convert_b_tic).count();
  printf("Init A/B: %.6f s\n", init_seconds);
  printf("Convert A -> half: %.6f s\n", convert_a_seconds);
  printf("Transpose + convert B -> half: %.6f s\n", convert_b_seconds);
  printf("cuBLAS avg: %.6f s\n", tcublas);
  printf("Kernel avg: %.6f s\n", tcutlass);
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
