#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "mppi/cuda/check.cuh"
#include "mppi/philox.hpp"

// S1.2 AC: identical sequences host vs device for the same counters. Raw
// blocks and u01 must match bit-for-bit (integer-only / exact float paths);
// normals go through device transcendentals, so those get a tolerance.
// Only built with ENABLE_CUDA=ON — runs on the Orin via scripts/gpu-test.sh.

namespace {

using mppi::rng::Philox4x32_10;

constexpr std::uint32_t kSeed = 2026;
constexpr std::uint32_t kIter = 7;
constexpr int kK = 256;
constexpr int kT = 32;

__global__ void fill_blocks(std::uint32_t* words, float* uniforms, float* normals) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= kK) return;
  for (int t = 0; t < kT; ++t) {
    auto b = Philox4x32_10::block(kSeed, kIter, k, t);
    int base = (k * kT + t) * 4;
    for (int i = 0; i < 4; ++i) {
      words[base + i] = b.w[i];
      uniforms[base + i] = mppi::rng::u01(b.w[i]);
    }
    mppi::rng::normal_pair(b.w[0], b.w[1], normals[base + 0], normals[base + 1]);
    mppi::rng::normal_pair(b.w[2], b.w[3], normals[base + 2], normals[base + 3]);
  }
}

}  // namespace

TEST(PhiloxParity, DeviceMatchesHost) {
  constexpr int n = kK * kT * 4;
  std::uint32_t* d_words;
  float *d_uniforms, *d_normals;
  CUDA_CHECK(cudaMalloc(&d_words, n * sizeof(std::uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_uniforms, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_normals, n * sizeof(float)));

  fill_blocks<<<(kK + 127) / 128, 128>>>(d_words, d_uniforms, d_normals);
  CUDA_CHECK(cudaGetLastError());

  std::vector<std::uint32_t> words(n);
  std::vector<float> uniforms(n), normals(n);
  CUDA_CHECK(cudaMemcpy(words.data(), d_words, n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(uniforms.data(), d_uniforms, n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(normals.data(), d_normals, n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_words));
  CUDA_CHECK(cudaFree(d_uniforms));
  CUDA_CHECK(cudaFree(d_normals));

  for (int k = 0; k < kK; ++k) {
    for (int t = 0; t < kT; ++t) {
      auto b = Philox4x32_10::block(kSeed, kIter, k, t);
      int base = (k * kT + t) * 4;
      for (int i = 0; i < 4; ++i) {
        ASSERT_EQ(words[base + i], b.w[i]) << "raw block mismatch at k=" << k << " t=" << t;
        ASSERT_EQ(uniforms[base + i], mppi::rng::u01(b.w[i]))
            << "u01 mismatch at k=" << k << " t=" << t;
      }
      float z[4];
      mppi::rng::normal_pair(b.w[0], b.w[1], z[0], z[1]);
      mppi::rng::normal_pair(b.w[2], b.w[3], z[2], z[3]);
      for (int i = 0; i < 4; ++i) {
        ASSERT_NEAR(normals[base + i], z[i], 1e-4f) << "normal mismatch at k=" << k << " t=" << t;
      }
    }
  }
}
