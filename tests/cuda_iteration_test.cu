#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "cartpole_task.hpp"
#include "mppi/cuda/controller.cuh"
#include "mppi/integrators.hpp"

// S2.3 AC: a full GPU iteration matches the CPU reference to <= 1e-4
// relative over 100 seeded iterations, and the cartpole swings up when the
// closed loop is driven by the GPU controller.
//
// Only built with ENABLE_CUDA=ON — runs on the Orin via scripts/gpu-test.sh.

namespace {

using Plant = cartpole_task::Plant;
using Cost = cartpole_task::SwingUpCost;
using GpuCtrl = mppi::cuda::GpuController<Plant, Cost, 1>;

float rel_diff(float a, float b) {
  return std::fabs(a - b) / std::max(1.0f, std::fabs(b));
}

}  // namespace

// Both controllers are driven along the SAME state trajectory (the CPU's),
// so every iteration compares like with like instead of letting the two
// loops drift apart and then blaming the reduction path.
TEST(CudaIteration, MatchesCpuOverHundredIterations) {
  Plant plant;
  Cost cost;
  auto cfg = cartpole_task::config(4);
  cfg.samples = 4096;

  cartpole_task::Controller cpu(plant, cost, cfg);
  GpuCtrl gpu(plant, cost, cfg);

  cartpole_task::State x{0.0f, 0.0f, static_cast<float>(M_PI), 0.0f};

  double worst_u = 0.0, worst_seq = 0.0;
  int worst_iter = -1;

  for (int i = 0; i < 100; ++i) {
    std::array<float, 1> u_cpu, u_gpu;
    cpu.step(x, u_cpu);
    gpu.step(x, u_gpu);

    double du = rel_diff(u_gpu[0], u_cpu[0]);
    if (du > worst_u) {
      worst_u = du;
      worst_iter = i;
    }

    // The emitted control alone is a weak check — the whole nominal
    // sequence has to track, or errors hide until they compound.
    auto seq_cpu = cpu.nominal_sequence();
    auto seq_gpu = gpu.nominal_sequence();
    ASSERT_EQ(seq_cpu.size(), seq_gpu.size());
    for (std::size_t j = 0; j < seq_cpu.size(); ++j) {
      worst_seq = std::max<double>(worst_seq, rel_diff(seq_gpu[j], seq_cpu[j]));
    }

    x = mppi::integrators::rk4_step(plant, x, u_cpu[0], cartpole_task::kDt);
  }

  RecordProperty("worst_control_rel", std::to_string(worst_u));
  RecordProperty("worst_sequence_rel", std::to_string(worst_seq));
  EXPECT_LT(worst_u, 1e-4) << "worst at iteration " << worst_iter;
  EXPECT_LT(worst_seq, 1e-4);
}

// The softmax weights themselves must match: this is what isolates a
// reduction-path bug (CUB min, the exp pass, the normalizer) from a rollout
// bug, which cuda_rollout_test.cu already covers.
TEST(CudaIteration, WeightsAndBetaMatchCpu) {
  Plant plant;
  Cost cost;
  auto cfg = cartpole_task::config(21);
  cfg.samples = 2048;

  cartpole_task::Controller cpu(plant, cost, cfg);
  GpuCtrl gpu(plant, cost, cfg);
  cartpole_task::State x{0.1f, 0.0f, static_cast<float>(M_PI) - 0.4f, 0.0f};

  std::array<float, 1> u_cpu, u_gpu;
  cpu.step(x, u_cpu);
  gpu.step(x, u_gpu);

  const auto& w_cpu = cpu.rollout_weights();
  auto w_gpu = gpu.rollout_weights();
  ASSERT_EQ(w_cpu.size(), w_gpu.size());

  double sum_gpu = 0.0, worst = 0.0;
  for (std::size_t k = 0; k < w_cpu.size(); ++k) {
    sum_gpu += w_gpu[k];
    worst = std::max<double>(worst, std::fabs(w_gpu[k] - w_cpu[k]));
  }
  RecordProperty("worst_weight_abs", std::to_string(worst));

  // Weights are a probability distribution; the normalizer must make them
  // sum to one however the reduction was performed.
  EXPECT_NEAR(sum_gpu, 1.0, 1e-3);
  EXPECT_LT(worst, 1e-6) << "softmax weights diverge";

  float beta_cpu = *std::min_element(cpu.rollout_costs().begin(), cpu.rollout_costs().end());
  EXPECT_LT(rel_diff(gpu.beta(), beta_cpu), 1e-4) << "CUB min disagrees with CPU min";
}

// S2.4: the hand-rolled warp-shuffle path must agree with the library path.
// Summation order differs (a shuffle tree versus whatever CUB and cuBLAS do
// internally), so this is a tolerance check, not bit-equality — but it is a
// tight one, because both reduce the same values.
TEST(CudaReductionV2, HandRolledMatchesLibraryPath) {
  Plant plant;
  Cost cost;
  auto cfg = cartpole_task::config(8);
  cfg.samples = 8192;

  GpuCtrl v1(plant, cost, cfg);
  GpuCtrl v2(plant, cost, cfg);
  v2.set_reduction_path(mppi::cuda::ReductionPath::HandRolled);

  cartpole_task::State x{0.0f, 0.0f, static_cast<float>(M_PI), 0.0f};
  double worst = 0.0;

  for (int i = 0; i < 25; ++i) {
    std::array<float, 1> u1, u2;
    v1.step(x, u1);
    v2.step(x, u2);
    worst = std::max<double>(worst, rel_diff(u2[0], u1[0]));

    auto s1 = v1.nominal_sequence();
    auto s2 = v2.nominal_sequence();
    for (std::size_t j = 0; j < s1.size(); ++j) {
      worst = std::max<double>(worst, rel_diff(s2[j], s1[j]));
    }
    x = mppi::integrators::rk4_step(plant, x, u1[0], cartpole_task::kDt);
  }
  RecordProperty("worst_v1_v2_rel", std::to_string(worst));
  EXPECT_LT(worst, 1e-4);

  // The hand-rolled min is exact regardless of order — no excuse for drift.
  EXPECT_FLOAT_EQ(v1.beta(), v2.beta());
}

// S2.5 AC: regenerating epsilon from Philox counters must give results
// bit-identical to storing it. Same reduction path on both sides, so the
// accumulation order is identical and only the origin of each epsilon
// differs — anything but exact equality is a bug, not rounding.
TEST(CudaNoiseRegeneration, BitIdenticalToStoredPath) {
  Plant plant;
  Cost cost;
  auto cfg = cartpole_task::config(12);
  cfg.samples = 4096;

  GpuCtrl stored(plant, cost, cfg);
  stored.set_reduction_path(mppi::cuda::ReductionPath::HandRolled);

  GpuCtrl regen(plant, cost, cfg);
  regen.set_noise_mode(mppi::cuda::NoiseMode::Regenerated);

  cartpole_task::State x{0.0f, 0.0f, static_cast<float>(M_PI), 0.0f};
  for (int i = 0; i < 25; ++i) {
    std::array<float, 1> u_s, u_r;
    stored.step(x, u_s);
    regen.step(x, u_r);
    ASSERT_EQ(u_s[0], u_r[0]) << "iteration " << i;

    auto seq_s = stored.nominal_sequence();
    auto seq_r = regen.nominal_sequence();
    for (std::size_t j = 0; j < seq_s.size(); ++j) {
      ASSERT_EQ(seq_s[j], seq_r[j]) << "iteration " << i << " element " << j;
    }
    x = mppi::integrators::rk4_step(plant, x, u_s[0], cartpole_task::kDt);
  }

  // The saving that motivates the variant: K*T*M floats never allocated.
  RecordProperty("stored_noise_bytes", std::to_string(stored.noise_bytes()));
  RecordProperty("regen_noise_bytes", std::to_string(regen.noise_bytes()));
  EXPECT_GT(stored.noise_bytes(), 0u);
  EXPECT_EQ(regen.noise_bytes(), 0u);
}

// S2.3 AC: swing-up driven entirely by the GPU controller.
TEST(CudaIteration, SwingUpSucceedsOnGpuLoop) {
  Plant plant;
  Cost cost;
  auto cfg = cartpole_task::config(0);

  GpuCtrl gpu(plant, cost, cfg);
  cartpole_task::State x{0.0f, 0.0f, static_cast<float>(M_PI), 0.0f};

  float t = 0.0f, held_since = -1.0f;
  bool ok = false;
  while (t < 6.5f) {
    std::array<float, 1> u;
    gpu.step(x, u);
    x = mppi::integrators::rk4_step(plant, x, u[0], cartpole_task::kDt);
    t += cartpole_task::kDt;
    bool in_goal = std::fabs(x.theta) < 0.1f && std::fabs(x.x) < 0.5f;
    if (in_goal) {
      if (held_since < 0.0f) held_since = t;
      if (t - held_since >= 1.0f && held_since <= 5.0f) {
        ok = true;
        break;
      }
    } else {
      held_since = -1.0f;
    }
  }
  EXPECT_TRUE(ok) << "GPU-driven swing-up failed";
}
