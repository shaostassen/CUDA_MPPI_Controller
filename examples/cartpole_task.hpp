#pragma once

#include <array>

#include "mppi/controller.hpp"
#include "mppi/dynamics/cartpole.hpp"

// Shared cartpole swing-up task definition: the empirically tuned S1.4
// config and its cost, used by the tests, the examples, and the S1.5 LQR
// cross-check. Tuning provenance: grid sweep over (lambda, sigma, w_x) at
// 20 seeds, finalists on the full 100-seed AC set — this config scored
// 100/100, sigma = 6 scored 96/100, lambda = 8 collapsed to ~50%.
namespace cartpole_task {

using Plant = mppi::dynamics::CartPole<float>;
using State = Plant::State;

inline constexpr float kRunWeights[4] = {3.0f, 0.2f, 10.0f, 0.2f};   // x, x_dot, theta, theta_dot
inline constexpr float kTermWeights[4] = {15.0f, 1.0f, 100.0f, 2.0f};
inline constexpr float kLambda = 2.0f;
inline constexpr float kSigma = 3.0f;
inline constexpr float kForceLimit = 15.0f;
inline constexpr float kDt = 0.02f;

struct SwingUpCost {
  float running(const State& s, int) const {
    float e[4] = {s.x, s.x_dot, s.theta, s.theta_dot};
    float w[4] = {kRunWeights[0], kRunWeights[1], kRunWeights[2], kRunWeights[3]};
    return mppi::costs::quadratic(e, w);
  }
  float terminal(const State& s) const {
    float e[4] = {s.x, s.x_dot, s.theta, s.theta_dot};
    float w[4] = {kTermWeights[0], kTermWeights[1], kTermWeights[2], kTermWeights[3]};
    return mppi::costs::quadratic(e, w);
  }
};

using Controller = mppi::Controller<Plant, SwingUpCost>;

inline Controller::Config config(unsigned seed) {
  Controller::Config cfg;
  cfg.samples = 512;  // K per the S1.4 AC
  cfg.horizon = 50;   // T per the S1.4 AC
  cfg.dt = kDt;
  cfg.lambda = kLambda;
  cfg.alpha = 0.0f;
  cfg.sigma = {kSigma};
  cfg.u_min = {-kForceLimit};
  cfg.u_max = {kForceLimit};
  cfg.seed = seed;
  return cfg;
}

// Mean first control MPPI emits when parked at a fixed state: warm the plan
// in, then average the emitted control across iterations and seeds. Used by
// the LQR cross-check, where single iterates are too noisy to read.
inline double mean_control_at(const State& x0, int n_seeds, int window) {
  Plant plant;
  double sum = 0.0;
  for (unsigned seed = 0; seed < static_cast<unsigned>(n_seeds); ++seed) {
    Controller ctrl(plant, SwingUpCost{}, config(seed * 7919u + 1u));
    std::array<float, 1> u{};
    double seed_sum = 0.0;
    for (int i = 0; i < 10 + window; ++i) {
      ctrl.step(x0, u);
      if (i >= 10) {
        seed_sum += u[0];
      }
    }
    sum += seed_sum / window;
  }
  return sum / n_seeds;
}

}  // namespace cartpole_task
