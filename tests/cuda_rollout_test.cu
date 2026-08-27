#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

#include "cartpole_task.hpp"
#include "mppi/cuda/device_buffers.cuh"
#include "mppi/cuda/rollout.cuh"

// S2.2 AC: per-rollout costs match the CPU reference to <= 1e-4 relative at
// K in {1k, 16k, 100k}. Only built with ENABLE_CUDA=ON, so this runs on the
// Orin via scripts/gpu-test.sh.
//
// Note what this test can and cannot fail on. Both sides call the same
// rollout_and_cost(), and tests/layout_test.cpp already proves the two
// noise layouts agree bit-for-bit on the host. So a failure here means a
// device-specific problem — floating-point evaluation of the
// transcendentals, an indexing or launch-bounds error, or a race — not two
// diverging implementations of MPPI.

namespace {

using Plant = cartpole_task::Plant;
using Cost = cartpole_task::SwingUpCost;

struct ParityResult {
  double max_rel = 0.0;
  int worst_k = -1;
  int exact = 0;
};

ParityResult compare_costs(int samples) {
  Plant plant;
  Cost cost;
  auto cfg = cartpole_task::config(17);
  cfg.samples = samples;

  cartpole_task::Controller ctrl(plant, cost, cfg);
  cartpole_task::State x{0.05f, 0.0f, static_cast<float>(M_PI) - 0.2f, 0.0f};

  // One step first so the nominal sequence is non-trivial.
  std::array<float, 1> u_out;
  ctrl.step(x, u_out);

  const std::vector<float> u_nominal = ctrl.nominal_sequence();
  auto cpu_params = ctrl.rollout_params();

  // CPU reference costs for this exact state and nominal sequence.
  const std::size_t noise_n = static_cast<std::size_t>(samples) * cfg.horizon;
  std::vector<float> cpu_noise(noise_n, 0.0f);
  std::vector<float> cpu_costs(samples);
  for (int k = 0; k < samples; ++k) {
    cpu_costs[k] = mppi::rollout_and_cost<Plant, Cost, 1>(plant, cost, x, u_nominal.data(),
                                                          cpu_params, k, cpu_noise.data());
  }

  // Device run with the layout the GPU controller actually uses, so this
  // test exercises production indexing rather than a variant of it.
  auto gpu_params = cpu_params;
  gpu_params.layout = mppi::layout_gemv_major(samples, 1);

  mppi::cuda::DeviceBuffer<float> d_u(u_nominal.size());
  mppi::cuda::DeviceBuffer<float> d_noise(noise_n);
  mppi::cuda::DeviceBuffer<float> d_costs(samples);
  d_u.upload(u_nominal.data(), u_nominal.size());
  d_noise.zero();

  mppi::cuda::launch_rollout_cost<Plant, Cost, 1>(plant, cost, x, d_u.data(), gpu_params, samples,
                                                  d_noise.data(), d_costs.data());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> gpu_costs(samples);
  d_costs.download(gpu_costs.data(), samples);

  ParityResult r;
  for (int k = 0; k < samples; ++k) {
    double denom = std::max(1.0, std::fabs(static_cast<double>(cpu_costs[k])));
    double rel = std::fabs(static_cast<double>(gpu_costs[k]) - cpu_costs[k]) / denom;
    if (rel > r.max_rel) {
      r.max_rel = rel;
      r.worst_k = k;
    }
    if (gpu_costs[k] == cpu_costs[k]) {
      ++r.exact;
    }
  }

  // The noise the device wrote must land in the corresponding time-major
  // slots — a coalescing-motivated layout is worthless if it is wrong.
  std::vector<float> gpu_noise(noise_n);
  d_noise.download(gpu_noise.data(), noise_n);
  for (int k = 0; k < samples; k += std::max(1, samples / 64)) {
    for (int t = 0; t < cfg.horizon; ++t) {
      float a = cpu_noise[cpu_params.layout.index(k, t, 0)];
      float b = gpu_noise[gpu_params.layout.index(k, t, 0)];
      EXPECT_NEAR(a, b, 1e-4f) << "noise mismatch k=" << k << " t=" << t;
    }
  }
  return r;
}

}  // namespace

TEST(CudaRollout, MatchesCpuAt1k) {
  auto r = compare_costs(1024);
  RecordProperty("max_rel", std::to_string(r.max_rel));
  RecordProperty("bit_exact", r.exact);
  EXPECT_LT(r.max_rel, 1e-4) << "worst k=" << r.worst_k;
}

TEST(CudaRollout, MatchesCpuAt16k) {
  auto r = compare_costs(16384);
  RecordProperty("max_rel", std::to_string(r.max_rel));
  RecordProperty("bit_exact", r.exact);
  EXPECT_LT(r.max_rel, 1e-4) << "worst k=" << r.worst_k;
}

TEST(CudaRollout, MatchesCpuAt100k) {
  auto r = compare_costs(100000);
  RecordProperty("max_rel", std::to_string(r.max_rel));
  RecordProperty("bit_exact", r.exact);
  EXPECT_LT(r.max_rel, 1e-4) << "worst k=" << r.worst_k;
}

// Grid sizing must cover a sample count that is not a multiple of the block
// size, and must not run past the end of the buffers.
TEST(CudaRollout, HandlesRaggedSampleCount) {
  auto r = compare_costs(1000);  // 1000 = 3.9 blocks of 256
  EXPECT_LT(r.max_rel, 1e-4) << "worst k=" << r.worst_k;
}
