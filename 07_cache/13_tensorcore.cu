#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <mma.h>
#include <chrono>
using namespace std;
using namespace nvcuda;

__global__ void kernel(int dim_m, int dim_n, int dim_k,
		       float *d_a, float *d_b, float *d_c) {
  int offset_a_m = 64 * blockIdx.x;
  int offset_b_n = 64 * blockIdx.y;
  int i = threadIdx.x;
  int warp_id = threadIdx.x / 32;

  __shared__ half block_a[16][64];
  __shared__ half block_b[16][64];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
  for (int r = 0; r < 2; r++)
    for (int c = 0; c < 4; c++)
      wmma::fill_fragment(acc[r][c], 0.0f);

  for (int k = 0; k < dim_k; k += 16) {
    __syncthreads();
    for (int j = 0; j < 16; ++j) {
      block_a[j][i] = __float2half(d_a[(k + j) * dim_m + offset_a_m + i]);
      block_b[j][i] = __float2half(d_b[(offset_b_n + i) * dim_k + k + j]);
    }
    __syncthreads();
    for (int r = 0; r < 2; r++) {
      int row_tile = warp_id * 2 + r;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
      wmma::load_matrix_sync(a_frag, &block_a[0][row_tile * 16], 64);
      for (int c = 0; c < 4; c++) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
        wmma::load_matrix_sync(b_frag, &block_b[0][c * 16], 64);
        wmma::mma_sync(acc[r][c], a_frag, b_frag, acc[r][c]);
      }
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
  cudaMallocManaged(&A, m * k * sizeof(float));
  cudaMallocManaged(&B, k * n * sizeof(float));
  cudaMallocManaged(&C, m * n * sizeof(float));
  cudaMallocManaged(&C2, m * n * sizeof(float));
  for (int i=0; i<m; i++)
    for (int j=0; j<k; j++)
      A[k*i+j] = drand48();
  for (int i=0; i<k; i++)
    for (int j=0; j<n; j++)
      B[n*i+j] = drand48();
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
			      A,
			      B,
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
  cudaFree(C);
  cudaFree(C2);
  cublasDestroy(cublas_handle);
}
