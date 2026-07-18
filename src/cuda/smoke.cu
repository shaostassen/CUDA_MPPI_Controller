#include "mppi/cuda/check.cuh"
#include "mppi/cuda/smoke.cuh"

namespace mppi::cuda_smoke {
namespace {

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

}  // namespace

void vector_add(const float* a, const float* b, float* c, int n) {
  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice));

  constexpr int kThreads = 256;
  int blocks = (n + kThreads - 1) / kThreads;
  add_kernel<<<blocks, kThreads>>>(d_a, d_b, d_c, n);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
}

}  // namespace mppi::cuda_smoke
