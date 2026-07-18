#include <gtest/gtest.h>

#include <cmath>

#include "cartpole_task.hpp"
#include "mppi/lqr.hpp"

namespace {

mppi::lqr::LinearSystem cartpole_linear() {
  return mppi::lqr::linearize_cartpole_euler(1.0, 0.1, 0.5, 9.81, cartpole_task::kDt);
}

mppi::lqr::Vec4 q_run() {
  return {cartpole_task::kRunWeights[0], cartpole_task::kRunWeights[1],
          cartpole_task::kRunWeights[2], cartpole_task::kRunWeights[3]};
}

mppi::lqr::Vec4 q_term() {
  return {cartpole_task::kTermWeights[0], cartpole_task::kTermWeights[1],
          cartpole_task::kTermWeights[2], cartpole_task::kTermWeights[3]};
}

// Implicit MPPI control cost, R = lambda * Sigma^-1 / 2.
double r_implicit() {
  return cartpole_task::kLambda / (2.0 * cartpole_task::kSigma * cartpole_task::kSigma);
}

}  // namespace

// The DARE solution must stabilize the linearized plant: u = -Kx from a
// perturbed start decays to ~zero. Pins the Riccati solver independently of
// anything MPPI does.
TEST(LqrReference, ClosedLoopStabilizesLinearizedCartpole) {
  auto sys = cartpole_linear();
  auto k = mppi::lqr::dare_gain(sys, q_run(), r_implicit());

  double x[4] = {0.2, 0.0, 0.05, 0.0};
  for (int step = 0; step < 600; ++step) {
    double u = 0.0;
    for (int j = 0; j < 4; ++j) {
      u -= k[j] * x[j];
    }
    double xn[4];
    for (int i = 0; i < 4; ++i) {
      xn[i] = sys.B[i] * u;
      for (int j = 0; j < 4; ++j) {
        xn[i] += sys.A[i][j] * x[j];
      }
    }
    for (int i = 0; i < 4; ++i) {
      x[i] = xn[i];
    }
  }
  double norm = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]);
  EXPECT_LT(norm, 1e-6);
}

// The finite-horizon gain (light w_term terminal) must sit below the
// infinite-horizon gain in every dimension — this ordering is the
// quantitative explanation for why MPPI reads "softer" than textbook LQR.
TEST(LqrReference, FiniteHorizonGainIsLighterThanInfinite) {
  auto sys = cartpole_linear();
  auto k_inf = mppi::lqr::dare_gain(sys, q_run(), r_implicit());
  auto k_fin = mppi::lqr::finite_horizon_gain(sys, q_run(), q_term(), r_implicit(), 50);
  for (int j = 0; j < 4; ++j) {
    EXPECT_LT(std::fabs(k_fin[j]), std::fabs(k_inf[j])) << "dim " << j;
  }
}

// S1.5 AC: near upright, MPPI's mean control tracks the LQ gain response.
//
// The fair target is the FINITE-horizon gain with P_T = diag(w_term) —
// that is the problem MPPI (horizon 50, terminal w_term) actually
// approximates. Per state dimension we compare effective gains via the
// antisymmetric difference (u(+v) - u(-v)) / (u_lqr(+v) - u_lqr(-v)),
// which cancels even-order noise.
//
// Tolerance stated: ratio within [0.75, 1.25]. Measured locally: x 0.91,
// x_dot 0.82, theta 0.99, theta_dot 1.02. Expected deviations, in order of
// size: entropy shrinkage from finite lambda = 2 (softmax-weighted noise
// average is biased toward the zero-mean prior), finite-K sampling noise,
// and the linearization itself. See docs/plots/lqr_cross_check.png for the
// full sweep.
TEST(MppiLqrCrossCheck, EffectiveGainTracksFiniteHorizonLqr) {
  auto sys = cartpole_linear();
  auto k_fin = mppi::lqr::finite_horizon_gain(sys, q_run(), q_term(), r_implicit(), 50);

  const double ranges[4] = {0.3, 0.4, 0.06, 0.2};
  const int kSeeds = 30;
  const int kWindow = 30;

  for (int dim = 0; dim < 4; ++dim) {
    double xv[4] = {0, 0, 0, 0};

    xv[dim] = ranges[dim];
    cartpole_task::State plus{static_cast<float>(xv[0]), static_cast<float>(xv[1]),
                               static_cast<float>(xv[2]), static_cast<float>(xv[3])};
    xv[dim] = -ranges[dim];
    cartpole_task::State minus{static_cast<float>(xv[0]), static_cast<float>(xv[1]),
                                static_cast<float>(xv[2]), static_cast<float>(xv[3])};

    double u_plus = cartpole_task::mean_control_at(plus, kSeeds, kWindow);
    double u_minus = cartpole_task::mean_control_at(minus, kSeeds, kWindow);
    double lqr_span = 2.0 * (-k_fin[dim]) * ranges[dim];  // u(+v) - u(-v) for u = -Kx
    double ratio = (u_plus - u_minus) / lqr_span;

    EXPECT_GE(ratio, 0.75) << "dim " << dim;
    EXPECT_LE(ratio, 1.25) << "dim " << dim;
  }
}
