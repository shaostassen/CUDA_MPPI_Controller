#include <gtest/gtest.h>

#include <cmath>

#include "mppi/costs/quadratic.hpp"
#include "mppi/weights.hpp"

using mppi::costs::control_coupling;
using mppi::costs::coupling_gamma;
using mppi::costs::quadratic;
using mppi::weights::softmax_weights;

// All expected values below are hand-computed with inputs chosen so every
// intermediate is exactly representable in binary floating point (powers of
// two, exact divisions) — "exactly" in the S1.3 AC sense, not tautologically
// re-running the library math.

// 2 * 1.5^2 + 0.5 * (-2)^2 = 2 * 2.25 + 0.5 * 4 = 4.5 + 2 = 6.5, all exact.
TEST(QuadraticCost, HandComputedExample) {
  double e[2] = {1.5, -2.0};
  double w[2] = {2.0, 0.5};
  EXPECT_DOUBLE_EQ(quadratic(e, w), 6.5);
}

// gamma = lambda * (1 - alpha) = 2 * (1 - 0.5) = 1, exact.
TEST(ControlCoupling, GammaFromLambdaAlpha) {
  EXPECT_DOUBLE_EQ(coupling_gamma(2.0, 0.5), 1.0);
  EXPECT_DOUBLE_EQ(coupling_gamma(3.0, 1.0), 0.0);  // alpha = 1 switches it off
}

// Scalar form: gamma * u * eps / sigma^2 = 1 * 0.5 * 0.25 / 4 = 0.03125.
TEST(ControlCoupling, HandComputedScalarExample) {
  EXPECT_DOUBLE_EQ(control_coupling(1.0, 0.5, 0.25, 4.0), 0.03125);
}

// Two dims: 0.5*0.25/4 + (-1)*0.5/0.25 = 0.03125 - 2 = -1.96875, all exact.
TEST(ControlCoupling, HandComputedVectorExample) {
  double u[2] = {0.5, -1.0};
  double eps[2] = {0.25, 0.5};
  double sigma_sq[2] = {4.0, 0.25};
  EXPECT_DOUBLE_EQ(control_coupling(1.0, u, eps, sigma_sq), -1.96875);
}

// S1.3 AC: the 3-sample weight example.
//
// Hand derivation: lambda = 2, S = {5, 7, 9}.
//   beta = min S = 5
//   (S - beta) / lambda = {0/2, 2/2, 4/2} = {0, 1, 2}   (each exact in FP)
//   unnormalized = {e^0, e^-1, e^-2} = {1, 0.36787944117144233, 0.13533528323661269}
//   Z = 1.5032147244080550...
//   w = {0.66524..., 0.24473..., 0.09003...}
// (Independently recognizable as softmax(2, 1, 0) — a standard worked
// example — which is a second anchor for the expected values.)
//
// The e^-1 / e^-2 literals below are correctly-rounded doubles written from
// the decimal expansions, and the expected weights are derived from them by
// plain arithmetic — never by calling the code under test.
TEST(SoftmaxWeights, HandComputedThreeSampleExample) {
  const double kExpNeg1 = 0.36787944117144233;
  const double kExpNeg2 = 0.13533528323661269;
  const double z = 1.0 + kExpNeg1 + kExpNeg2;

  double costs[3] = {5.0, 7.0, 9.0};
  double w[3];
  double beta = softmax_weights(costs, 3, 2.0, w);

  EXPECT_DOUBLE_EQ(beta, 5.0);
  EXPECT_NEAR(w[0], 1.0 / z, 1e-15);
  EXPECT_NEAR(w[1], kExpNeg1 / z, 1e-15);
  EXPECT_NEAR(w[2], kExpNeg2 / z, 1e-15);
  EXPECT_NEAR(w[0] + w[1] + w[2], 1.0, 1e-15);
}

// The baseline subtraction must make weights invariant to a uniform cost
// shift. Shift by 1000: (S - beta) stays {0, 2, 4} exactly, so the weights
// must come out bit-identical, not merely close.
TEST(SoftmaxWeights, InvariantToUniformCostShift) {
  double costs[3] = {5.0, 7.0, 9.0};
  double shifted[3] = {1005.0, 1007.0, 1009.0};
  double w[3], w_shifted[3];
  softmax_weights(costs, 3, 2.0, w);
  double beta = softmax_weights(shifted, 3, 2.0, w_shifted);

  EXPECT_DOUBLE_EQ(beta, 1005.0);
  for (int k = 0; k < 3; ++k) {
    EXPECT_EQ(w[k], w_shifted[k]);
  }
}
