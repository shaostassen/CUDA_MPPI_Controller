#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

#include "cartpole_task.hpp"
#include "mppi/rollout.hpp"

// The CUDA path stores noise time-major ([T][K][M]) so that at fixed t
// adjacent threads read adjacent memory, while the CPU reference stores it
// rollout-major ([K][T][M]). That re-indexing is the only genuinely new
// logic in the GPU port, and it is testable without a GPU: run the shared
// rollout core under both layouts and require identical costs and identical
// noise values in the corresponding slots.
//
// If this passes, a GPU/CPU cost mismatch cannot be blamed on the layout.

namespace {

constexpr int kSamples = 256;

}  // namespace

TEST(NoiseLayout, TimeMajorMatchesRolloutMajor) {
  cartpole_task::Plant plant;
  cartpole_task::SwingUpCost cost;
  auto cfg = cartpole_task::config(5);
  cfg.samples = kSamples;

  cartpole_task::Controller ctrl(plant, cost, cfg);
  cartpole_task::State x{0.1f, 0.0f, static_cast<float>(M_PI) - 0.3f, 0.0f};

  // Give the nominal sequence some non-zero structure first, so the test
  // is not just exercising the u == 0 case.
  std::array<float, 1> u_out;
  ctrl.step(x, u_out);
  ctrl.step(x, u_out);

  const std::vector<float> u_nominal = ctrl.nominal_sequence();
  auto params_rm = ctrl.rollout_params();  // rollout-major, as the CPU uses

  auto params_tm = params_rm;
  params_tm.layout = mppi::layout_time_major(cfg.samples, 1);

  const std::size_t n = static_cast<std::size_t>(cfg.samples) * cfg.horizon;
  std::vector<float> noise_rm(n, 0.0f), noise_tm(n, 0.0f);
  std::vector<float> costs_rm(cfg.samples), costs_tm(cfg.samples);

  for (int k = 0; k < cfg.samples; ++k) {
    costs_rm[k] = mppi::rollout_and_cost<cartpole_task::Plant, cartpole_task::SwingUpCost, 1>(
        plant, cost, x, u_nominal.data(), params_rm, k, noise_rm.data());
    costs_tm[k] = mppi::rollout_and_cost<cartpole_task::Plant, cartpole_task::SwingUpCost, 1>(
        plant, cost, x, u_nominal.data(), params_tm, k, noise_tm.data());
  }

  for (int k = 0; k < cfg.samples; ++k) {
    ASSERT_EQ(costs_rm[k], costs_tm[k]) << "cost differs at k=" << k;
  }

  for (int k = 0; k < cfg.samples; ++k) {
    for (int t = 0; t < cfg.horizon; ++t) {
      float a = noise_rm[params_rm.layout.index(k, t, 0)];
      float b = noise_tm[params_tm.layout.index(k, t, 0)];
      ASSERT_EQ(a, b) << "noise differs at k=" << k << " t=" << t;
    }
  }
}

// Both layouts must address each (k, t, i) slot exactly once — an aliasing
// bug here would silently corrupt the weighted-noise sum on device.
TEST(NoiseLayout, BothLayoutsAreBijections) {
  const int K = 7, T = 5, M = 2;
  for (auto layout : {mppi::layout_rollout_major(T, M), mppi::layout_time_major(K, M)}) {
    std::vector<int> hits(static_cast<std::size_t>(K) * T * M, 0);
    for (int k = 0; k < K; ++k) {
      for (int t = 0; t < T; ++t) {
        for (int i = 0; i < M; ++i) {
          int idx = layout.index(k, t, i);
          ASSERT_GE(idx, 0);
          ASSERT_LT(idx, static_cast<int>(hits.size()));
          ++hits[idx];
        }
      }
    }
    for (std::size_t i = 0; i < hits.size(); ++i) {
      EXPECT_EQ(hits[i], 1) << "slot " << i << " addressed " << hits[i] << " times";
    }
  }
}

// The rollout core must reproduce the controller's own per-rollout costs
// when handed the controller's parameters — this is the CPU half of S2.2's
// parity AC, with the GPU half in tests/cuda_rollout_test.cu.
TEST(RolloutCore, MatchesControllerRolloutCosts) {
  cartpole_task::Plant plant;
  cartpole_task::SwingUpCost cost;
  auto cfg = cartpole_task::config(9);
  cfg.samples = kSamples;

  cartpole_task::Controller ctrl(plant, cost, cfg);
  cartpole_task::State x{0.0f, 0.0f, static_cast<float>(M_PI), 0.0f};

  std::vector<float> u_before = ctrl.nominal_sequence();
  auto params = ctrl.rollout_params();

  std::array<float, 1> u_out;
  ctrl.step(x, u_out);

  std::vector<float> noise(static_cast<std::size_t>(cfg.samples) * cfg.horizon, 0.0f);
  for (int k = 0; k < cfg.samples; ++k) {
    float mine = mppi::rollout_and_cost<cartpole_task::Plant, cartpole_task::SwingUpCost, 1>(
        plant, cost, x, u_before.data(), params, k, noise.data());
    ASSERT_EQ(mine, ctrl.rollout_costs()[k]) << "k=" << k;
  }
}
