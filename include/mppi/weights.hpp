#pragma once

#include <cmath>

#include "mppi/macros.hpp"

namespace mppi::weights {

// Baseline-subtracted softmax over trajectory costs (PLAN.md §1 decision 6):
// beta = min_k S_k, w_k = exp(-(S_k - beta) / lambda) / Z. The baseline
// subtraction keeps exp() from underflowing when costs are large; the
// weights are mathematically unchanged by it (the shift cancels in Z).
// Host-side on purpose — the GPU version arrives with S2.3's reduction
// kernels and must match this within 1e-4 rel, per that story's AC.
//
// Returns beta (the min cost), which the caller wants for logging anyway.
template <typename Scalar>
Scalar softmax_weights(const Scalar* costs, int K, Scalar lambda, Scalar* out_weights) {
  Scalar beta = costs[0];
  for (int k = 1; k < K; ++k) {
    beta = costs[k] < beta ? costs[k] : beta;
  }
  Scalar z = Scalar(0);
  for (int k = 0; k < K; ++k) {
    out_weights[k] = std::exp(-(costs[k] - beta) / lambda);
    z += out_weights[k];
  }
  Scalar inv_z = Scalar(1) / z;
  for (int k = 0; k < K; ++k) {
    out_weights[k] *= inv_z;
  }
  return beta;
}

}  // namespace mppi::weights
