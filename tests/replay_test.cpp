#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

#include "cartpole_task.hpp"
#include "mppi/integrators.hpp"

// replay_rollout() is what the visualization draws, so it has to reproduce
// the trajectory step() actually costed — not merely a plausible one. The
// check: replay a rollout, recompute its trajectory cost here in step()'s
// own accumulation order, and compare against the controller's stored cost
// for that k. Agreement to the last float bit means the ghost trails are
// literally the sampled futures, not a lookalike re-simulation.
TEST(ReplayRollout, ReproducesTheTrajectoryStepCosted) {
  cartpole_task::Plant plant;
  auto cfg = cartpole_task::config(3);
  cfg.samples = 64;  // small K: this test is about fidelity, not throughput
  cartpole_task::Controller ctrl(plant, {}, cfg);
  cartpole_task::SwingUpCost cost;

  cartpole_task::State x{0.0f, 0.0f, static_cast<float>(M_PI), 0.0f};
  const float gamma = mppi::costs::coupling_gamma(cfg.lambda, cfg.alpha);
  const float sigma_sq = cfg.sigma[0] * cfg.sigma[0];
  int exact = 0, compared = 0;

  for (int iter = 0; iter < 3; ++iter) {
    // Snapshot the nominal sequence as it stands BEFORE the step: that is
    // what this iteration's rollouts are perturbations around.
    std::vector<float> u_before = ctrl.nominal_sequence();
    std::uint32_t iteration = ctrl.iteration();

    std::array<float, 1> u_out;
    ctrl.step(x, u_out);

    for (int k : {0, 7, 31, 63}) {
      auto trace = ctrl.replay_rollout(x, u_before, iteration, k);
      ASSERT_EQ(trace.states.size(), static_cast<std::size_t>(cfg.horizon) + 1);

      float recomputed = 0.0f;
      for (int t = 0; t < cfg.horizon; ++t) {
        recomputed += mppi::costs::control_coupling(gamma, u_before[t], trace.eps[t], sigma_sq);
        recomputed += cost.running(trace.states[t + 1], t);
      }
      recomputed += cost.terminal(trace.states[cfg.horizon]);

      float expected = ctrl.rollout_costs()[k];
      ++compared;
      if (recomputed == expected) {
        ++exact;
      }
      EXPECT_NEAR(recomputed, expected, 1e-3f * std::max(1.0f, std::fabs(expected)))
          << "iteration " << iter << " rollout " << k;
    }

    x = mppi::integrators::rk4_step(plant, x, u_out[0], cfg.dt);
  }

  RecordProperty("bit_exact_rollouts", exact);
  RecordProperty("compared_rollouts", compared);
}
