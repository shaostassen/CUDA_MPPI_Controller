#pragma once

#include <cstddef>

#include "mppi/macros.hpp"

namespace mppi::costs {

// Diagonal weighted quadratic: sum_i weight[i] * error[i]^2. Running and
// terminal costs are both this with different weights, so there is one
// function, not two.
template <typename Scalar, std::size_t N>
MPPI_HOST_DEVICE Scalar quadratic(const Scalar (&error)[N], const Scalar (&weight)[N]) {
  Scalar c = Scalar(0);
  for (std::size_t i = 0; i < N; ++i) {
    c += weight[i] * error[i] * error[i];
  }
  return c;
}

// gamma = lambda * (1 - alpha), PLAN.md §1 decision 6. alpha = 0 is the
// standard information-theoretic case; alpha = 1 switches the coupling off.
template <typename Scalar>
MPPI_HOST_DEVICE Scalar coupling_gamma(Scalar lambda, Scalar alpha) {
  return lambda * (Scalar(1) - alpha);
}

// Per-timestep control-coupling term with diagonal Sigma:
// gamma * u^T Sigma^-1 eps, where sigma_sq[i] is the sampling variance of
// control dim i. Added to the running cost of every rollout timestep.
template <typename Scalar, std::size_t M>
MPPI_HOST_DEVICE Scalar control_coupling(Scalar gamma, const Scalar (&u)[M], const Scalar (&eps)[M],
                                         const Scalar (&sigma_sq)[M]) {
  Scalar c = Scalar(0);
  for (std::size_t i = 0; i < M; ++i) {
    c += u[i] * eps[i] / sigma_sq[i];
  }
  return gamma * c;
}

// m = 1 convenience so scalar-control systems (pendulum, cartpole) don't
// have to wrap their force in a one-element array at every call site.
template <typename Scalar>
MPPI_HOST_DEVICE Scalar control_coupling(Scalar gamma, Scalar u, Scalar eps, Scalar sigma_sq) {
  return gamma * u * eps / sigma_sq;
}

}  // namespace mppi::costs
