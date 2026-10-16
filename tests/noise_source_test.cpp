#include "mppi/noise_source.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

#include "cartpole_task.hpp"

// S2.5 asks for results bit-identical to the stored-noise path when epsilon
// is regenerated from Philox counters instead of kept in memory. That claim
// is testable here, without a GPU: storing simply wrote whatever
// sample_control() returned, so regenerating must reproduce it exactly.
//
// The device run has to hold this too, and will for the same reason — both
// paths call the same code on the same hardware, so whatever its sinf/cosf
// do, they do identically for both. (Host and device may still differ from
// each other by a few ULP; that is the separate 1e-4 CPU<->GPU question.)

namespace {

using Stored = mppi::StoredNoise<float, 1>;
using Regenerated = mppi::RegeneratedNoise<float, 1>;

constexpr int kSamples = 512;

}  // namespace

TEST(NoiseSource, RegeneratedEqualsStoredBitExactly) {
  cartpole_task::Plant plant;
  cartpole_task::SwingUpCost cost;
  auto cfg = cartpole_task::config(13);
  cfg.samples = kSamples;

  cartpole_task::Controller ctrl(plant, cost, cfg);
  cartpole_task::State x{0.0f, 0.0f, static_cast<float>(M_PI), 0.0f};

  // Two steps so the nominal sequence is non-trivial and the iteration
  // counter (part of the Philox counter) is non-zero.
  std::array<float, 1> u_out;
  ctrl.step(x, u_out);
  ctrl.step(x, u_out);

  const std::vector<float> u_nominal = ctrl.nominal_sequence();
  auto params = ctrl.rollout_params();
  params.layout = mppi::layout_gemv_major(cfg.samples, 1);

  // Fill the buffer the way the rollout kernel would.
  std::vector<float> noise(static_cast<std::size_t>(cfg.samples) * cfg.horizon, 0.0f);
  for (int k = 0; k < cfg.samples; ++k) {
    mppi::rollout_and_cost<cartpole_task::Plant, cartpole_task::SwingUpCost, 1>(
        plant, cost, x, u_nominal.data(), params, k, noise.data());
  }

  Stored stored{noise.data(), params.layout};
  Regenerated regen{params};

  int compared = 0;
  for (int k = 0; k < cfg.samples; ++k) {
    for (int t = 0; t < cfg.horizon; ++t) {
      float a = stored.eps(k, t, 0, u_nominal.data());
      float b = regen.eps(k, t, 0, u_nominal.data());
      ASSERT_EQ(a, b) << "k=" << k << " t=" << t;
      ++compared;
    }
  }
  RecordProperty("elements_compared", compared);
}

// The claim that matters downstream is not per-element equality but that the
// weighted sum — the thing that actually moves the control sequence — comes
// out identical.
TEST(NoiseSource, WeightedSumIdenticalFromEitherSource) {
  cartpole_task::Plant plant;
  cartpole_task::SwingUpCost cost;
  auto cfg = cartpole_task::config(29);
  cfg.samples = kSamples;

  cartpole_task::Controller ctrl(plant, cost, cfg);
  cartpole_task::State x{0.2f, -0.1f, static_cast<float>(M_PI) - 0.5f, 0.0f};

  std::array<float, 1> u_out;
  ctrl.step(x, u_out);

  const std::vector<float> u_nominal = ctrl.nominal_sequence();
  auto params = ctrl.rollout_params();
  params.layout = mppi::layout_gemv_major(cfg.samples, 1);

  std::vector<float> noise(static_cast<std::size_t>(cfg.samples) * cfg.horizon, 0.0f);
  for (int k = 0; k < cfg.samples; ++k) {
    mppi::rollout_and_cost<cartpole_task::Plant, cartpole_task::SwingUpCost, 1>(
        plant, cost, x, u_nominal.data(), params, k, noise.data());
  }

  const auto& w = ctrl.rollout_weights();
  Stored stored{noise.data(), params.layout};
  Regenerated regen{params};

  for (int t = 0; t < cfg.horizon; ++t) {
    float sum_stored = 0.0f, sum_regen = 0.0f;
    for (int k = 0; k < cfg.samples; ++k) {
      sum_stored += w[k] * stored.eps(k, t, 0, u_nominal.data());
      sum_regen += w[k] * regen.eps(k, t, 0, u_nominal.data());
    }
    ASSERT_EQ(sum_stored, sum_regen) << "t=" << t;
  }
}

// Skipping the noise writes must not perturb the cost: the rollout is the
// same simulation either way.
TEST(NoiseSource, SkippingNoiseStoresLeavesCostsUnchanged) {
  cartpole_task::Plant plant;
  cartpole_task::SwingUpCost cost;
  auto cfg = cartpole_task::config(31);
  cfg.samples = 128;

  cartpole_task::Controller ctrl(plant, cost, cfg);
  cartpole_task::State x{0.0f, 0.0f, static_cast<float>(M_PI), 0.0f};
  const std::vector<float> u_nominal = ctrl.nominal_sequence();
  auto params = ctrl.rollout_params();

  std::vector<float> noise(static_cast<std::size_t>(cfg.samples) * cfg.horizon, 0.0f);
  for (int k = 0; k < cfg.samples; ++k) {
    float with_store =
        mppi::rollout_and_cost<cartpole_task::Plant, cartpole_task::SwingUpCost, 1, true>(
            plant, cost, x, u_nominal.data(), params, k, noise.data());
    float without =
        mppi::rollout_and_cost<cartpole_task::Plant, cartpole_task::SwingUpCost, 1, false>(
            plant, cost, x, u_nominal.data(), params, k, nullptr);
    ASSERT_EQ(with_store, without) << "k=" << k;
  }
}
