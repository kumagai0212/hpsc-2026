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

__device__ __forceinline__ void cp_async_16(void *dst, const void *src) {
  unsigned int smem_addr = static_cast<unsigned int>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(src));
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void cp_async_wait_group_0() {
  asm volatile("cp.async.wait_group 0;");
}

__device__ __forceinline__ void stage_load_cp_async(
    int thread_id,
    int tile_k,
    int stage,
    int dim_m,
    int dim_n,
    int offset_a_m,
    int offset_b_n,
    const half *__restrict__ d_a,
    const half *__restrict__ d_b,
    half block_a[][16][64],
    half block_b[][16][64]) {
  // 変更点: 1回のtile読み込みを64スレッドで分担し、A/Bのcp.asyncを同時に発行する
#pragma unroll
  for (int chunk = thread_id; chunk < 128; chunk += 64) {
    int row = chunk >> 3;
    int col = (chunk & 7) << 3;
    cp_async_16(&block_a[stage][row][col], &d_a[(tile_k + row) * dim_m + offset_a_m + col]);
    cp_async_16(&block_b[stage][row][col], &d_b[(tile_k + row) * dim_n + offset_b_n + col]);
  }
  cp_async_commit_group();
}

__global__ void kernel(int dim_m, int dim_n, int dim_k,
           const half *__restrict__ d_a,
           const half *__restrict__ d_b,
           float *__restrict__ d_c) {
  int offset_a_m = 64 * blockIdx.x;
  int offset_b_n = 64 * blockIdx.y;
  int warp_id = threadIdx.x / 32;

  __shared__ __align__(16) half block_a[3][16][64];
  __shared__ __align__(16) half block_b[3][16][64];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4]; //16*16*16の箱のイメージ

  //initialize output to zero
  for (int r = 0; r < 2; r++)
    for (int c = 0; c < 4; c++)
      wmma::fill_fragment(acc[r][c], 0.0f);

  int num_tiles = dim_k / 16;

  // 変更点: 最初に3タイル分を先読みし、以後は3タイル先を仕込むことで
  // 今の計算より十分前に共有メモリへ載せる。
  stage_load_cp_async(threadIdx.x, 0, 0, dim_m, dim_n, offset_a_m, offset_b_n, d_a, d_b, block_a, block_b);
  if (num_tiles > 1) {
    stage_load_cp_async(threadIdx.x, 16, 1, dim_m, dim_n, offset_a_m, offset_b_n, d_a, d_b, block_a, block_b);
  }
  if (num_tiles > 2) {
    stage_load_cp_async(threadIdx.x, 32, 2, dim_m, dim_n, offset_a_m, offset_b_n, d_a, d_b, block_a, block_b);
  }
  cp_async_wait_group_0();
  __syncthreads();

  for (int tile = 0; tile < num_tiles; ++tile) {
    int cur = tile % 3;
    int nxt = (tile + 3) % 3;

    if (tile + 3 < num_tiles) {
      // 変更点: 3タイル先を仕込むことで、cp.async の待ちを今の計算でより長く隠す
      stage_load_cp_async(threadIdx.x, (tile + 3) * 16, nxt, dim_m, dim_n, offset_a_m, offset_b_n, d_a, d_b, block_a, block_b);
    }

    // 変更点: Bフラグメントはrに依存しないため、先に4個まとめてロードして再利用
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4];
    for (int c = 0; c < 4; c++) {
      wmma::load_matrix_sync(b_frag[c], &block_b[cur][0][c * 16], 64);
    }
    for (int r = 0; r < 2; r++) {
      int row_tile = warp_id * 2 + r;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
      wmma::load_matrix_sync(a_frag, &block_a[cur][0][row_tile * 16], 64);
      for (int c = 0; c < 4; c++) {
        wmma::mma_sync(acc[r][c], a_frag, b_frag[c], acc[r][c]);
      }
    }

    if (tile + 1 < num_tiles) {
      cp_async_wait_group_0();
      __syncthreads();
    }
  }

  for (int r = 0; r < 2; r++) {
    for (int c = 0; c < 4; c++) {
      int c_m = offset_a_m + (warp_id * 2 + r) * 16;
      int c_n = offset_b_n + c * 16;
      if (c_n < dim_n && c_m < dim_m)
        wmma::store_matrix_sync(&d_c[c_n * dim_m + c_m], acc[r][c], dim_m, wmma::mem_col_major);
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
  for (int i=0; i<m; i++)
    for (int j=0; j<k; j++)
      A[k*i+j] = drand48();
  for (int i=0; i<k; i++)
    for (int j=0; j<n; j++)
      B[n*i+j] = drand48();
  // 変更点: half入力を一度だけ作成し、計測中の型変換コストを排除
  for (int i=0; i<m; i++)
    for (int j=0; j<k; j++)
      A16[k*i+j] = __float2half(A[k*i+j]);
  // 変更点: Bをまとめてfloat->half変換しつつK×N row-majorに転置。
  // 元のBはcuBLASの列優先レイアウトでB[n][k]=B_ptr[n*k+k_idx]だったが、
  // 転置後はB16_T[k][n]=B16_T[ki*n+ni]とし、カーネル内でcoalescedに読める。
  for (int ki=0; ki<k; ki++)
    for (int ni=0; ni<n; ni++)
      B16_T[ki*n + ni] = __float2half(B[ni*k + ki]);
  for (int i=0; i<n; i++)
    for (int j=0; j<m; j++)
      C[m*i+j] = C2[m*i+j] = 0;
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  auto tic = chrono::steady_clock::now();
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
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
  auto toc = chrono::steady_clock::now();
  int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
  double tcublas = chrono::duration<double>(toc - tic).count() / Nt;
  double cublas_flops = double(num_flops) / tcublas / 1.0e9;
  int tile = 64;
  dim3 block = dim3(tile);
  dim3 grid = dim3((m+tile-1)/tile, (n+tile-1)/tile);
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    kernel<<< grid, block >>>(m,
			      n,
			      k,
            A16,
            B16_T,  // 転置済みK×N half行列を渡す
			      C2);
    cudaDeviceSynchronize();
  }
  toc = chrono::steady_clock::now();
  double tcutlass = chrono::duration<double>(toc - tic).count() / Nt;
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
  cublasDestroy(cublas_handle);
}
