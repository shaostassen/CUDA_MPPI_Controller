#pragma once

#include "mppi/macros.hpp"
#include "mppi/rollout.hpp"

namespace mppi {

// Where the weighted-noise sum gets its epsilon values from. The two
// sources are the S2.5 experiment: keep K*T*M floats in global memory and
// read them back, or keep nothing and recompute each one from its Philox
// counter.
//
// Both expose the same eps(k, t, i, u_nominal), so the reduction kernel is
// written once and instantiated twice. That is also what makes "bit-
// identical" a real claim rather than an aspiration: the accumulation order
// is the same code, so the only question is whether a regenerated epsilon
// equals the stored one — and since storing simply wrote what
// sample_control() returned, it does, exactly.

template <typename Scalar, int M>
struct StoredNoise {
  const Scalar* noise;
  NoiseLayout layout;

  MPPI_HOST_DEVICE Scalar eps(int k, int t, int i, const Scalar*) const {
    return noise[layout.index(k, t, i)];
  }
};

template <typename Scalar, int M>
struct RegeneratedNoise {
  RolloutParams<Scalar, M> params;

  // Recomputes the Philox block for (k, t) and returns dim i's post-clamp
  // effective noise. Costs one Philox block per (k, t) instead of a global
  // load — the trade S2.5 exists to measure.
  MPPI_HOST_DEVICE Scalar eps(int k, int t, int i, const Scalar* u_nominal) const {
    Scalar u_pert[M];
    Scalar e[M];
    sample_control(params, k, t, u_nominal + static_cast<std::size_t>(t) * M, u_pert, e);
    return e[i];
  }
};

}  // namespace mppi
