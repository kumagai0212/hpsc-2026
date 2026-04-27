#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void bucket_init(int* bucket, int range) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < range) {
    bucket[idx] = 0;
  }
}
__global__ void bucket_add(int* key, int* bucket, int n, int range) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    atomicAdd(&bucket[key[idx]], 1);
  }
}

__global__ void bucket_sort(int* key, int* bucket, int n, int range) {
  int idx = threadIdx.x;
  __shared__ int offset[256];
  __shared__ int temp[256];

  offset[idx] = bucket[idx];
  __syncthreads();

  for (int j = 1; j < range; j <<= 1) {
    // copy bucket to temp
    temp[idx] = offset[idx]; 
    __syncthreads();       
    
    // add temp[idx-j] to offset[idx]
    if (idx >= j) {
      offset[idx] += temp[idx - j]; 
    }
    __syncthreads();        
  }

  int start_point = 0;
  if (idx > 0) {
    start_point = offset[idx - 1];
  }

  for (int i=start_point; i<start_point+bucket[idx]; i++) {
    key[i] = idx;
  }

}

int main() {
  int n = 50;
  int range = 5;
  int *key;
  cudaMallocManaged(&key, n*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  //std::vector<int> bucket(range); 
  int *bucket;
  cudaMallocManaged(&bucket, range*sizeof(int));
  bucket_init<<<(range+255)/256, 256>>>(bucket, range);
  cudaDeviceSynchronize(); //wait to finish bucket_init kernel
  bucket_add<<<(n+255)/256, 256>>>(key, bucket, n, range);
  cudaDeviceSynchronize(); //wait to finish bucket_add kernel
  bucket_sort<<<1, range>>>(key, bucket, n, range);
  cudaDeviceSynchronize(); //wait to finish bucket_sort kernel

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
  cudaFree(bucket);
  cudaFree(key);
  return 0;
}
