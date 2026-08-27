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

// Every layout must address each (k, t, i) slot exactly once — an aliasing
// bug here would silently corrupt the weighted-noise sum on device.
TEST(NoiseLayout, AllLayoutsAreBijections) {
  const int K = 7, T = 5, M = 2;
  for (auto layout : {mppi::layout_rollout_major(T, M), mppi::layout_time_major(K, M),
                      mppi::layout_gemv_major(K, M)}) {
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

// For m == 1 the literal decision-9 layout and the gemv layout must be the
// same mapping — that is what makes adopting the latter free for the
// cartpole work rather than a behavioral change.
TEST(NoiseLayout, GemvAndTimeMajorAgreeWhenSingleControlDim) {
  const int K = 9, T = 4;
  auto tm = mppi::layout_time_major(K, 1);
  auto gm = mppi::layout_gemv_major(K, 1);
  for (int k = 0; k < K; ++k) {
    for (int t = 0; t < T; ++t) {
      ASSERT_EQ(tm.index(k, t, 0), gm.index(k, t, 0)) << "k=" << k << " t=" << t;
    }
  }
}

// The weighted noise sum is a cuBLAS Sgemv on device (PLAN.md §1 decision 8):
// with the gemv layout the noise buffer IS a column-major K x (T*M) matrix,
// so du = A^T w. Getting those call parameters wrong is an easy and quiet
// mistake, so emulate the exact call on the host — same transpose, same lda,
// same indexing cuBLAS would use — and require it to reproduce the
// straightforward reference sum.
TEST(NoiseLayout, SgemvFormulationMatchesReferenceWeightedSum) {
  const int K = 32, T = 6, M = 3;
  auto layout = mppi::layout_gemv_major(K, M);

  std::vector<float> noise(static_cast<std::size_t>(K) * T * M);
  for (int k = 0; k < K; ++k) {
    for (int t = 0; t < T; ++t) {
      for (int i = 0; i < M; ++i) {
        noise[layout.index(k, t, i)] = 0.01f * static_cast<float>(k) +
                                        0.1f * static_cast<float>(t) - static_cast<float>(i);
      }
    }
  }
  std::vector<float> w(K);
  for (int k = 0; k < K; ++k) {
    w[k] = 1.0f / static_cast<float>(K) + 0.001f * static_cast<float>(k % 5);
  }

  // Reference: the plain nested sum the CPU controller performs.
  std::vector<float> du_ref(static_cast<std::size_t>(T) * M, 0.0f);
  for (int t = 0; t < T; ++t) {
    for (int i = 0; i < M; ++i) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc += w[k] * noise[layout.index(k, t, i)];
      }
      du_ref[static_cast<std::size_t>(t) * M + i] = acc;
    }
  }

  // Emulated cublasSgemv(handle, CUBLAS_OP_T, m=K, n=T*M, &one, A, lda=K,
  //                      w, 1, &zero, y, 1):
  //   y[j] = sum_k A[k + j*lda] * w[k]
  const int lda = K;
  const int n = T * M;
  std::vector<float> y(n, 0.0f);
  for (int j = 0; j < n; ++j) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += noise[k + j * lda] * w[k];
    }
    y[j] = acc;
  }

  // y is indexed by j = t*M + i, which is exactly du's own ordering.
  for (int t = 0; t < T; ++t) {
    for (int i = 0; i < M; ++i) {
      int j = t * M + i;
      ASSERT_NEAR(y[j], du_ref[j], 1e-6f) << "t=" << t << " i=" << i;
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
