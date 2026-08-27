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
