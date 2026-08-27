#pragma once

#include <cstdint>

#include "mppi/costs/quadratic.hpp"
#include "mppi/integrators.hpp"
#include "mppi/macros.hpp"
#include "mppi/philox.hpp"

namespace mppi {

// Device-safe clamp: std::clamp is not annotated for device code, and
// relying on --expt-relaxed-constexpr just to borrow it is not worth it.
template <typename Scalar>
MPPI_HOST_DEVICE inline Scalar clamp_to(Scalar v, Scalar lo, Scalar hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Noise buffer addressing. The CPU reference stores [K][T][M]; the CUDA
// path stores [T][K][M] so that at fixed t adjacent threads (k) touch
// adjacent memory (PLAN.md §1 decision 9). Expressing both as a pair of
// strides lets one rollout function serve both layouts, which is what makes
// CPU<->GPU parity structural rather than a coincidence to be debugged.
struct NoiseLayout {
  int rollout_stride;  // elements between consecutive k
  int time_stride;     // elements between consecutive t

  MPPI_HOST_DEVICE int index(int k, int t, int i) const {
    return k * rollout_stride + t * time_stride + i;
  }
};

MPPI_HOST_DEVICE inline NoiseLayout layout_rollout_major(int horizon, int m) {  // [K][T][M]
  return NoiseLayout{horizon * m, m};
}

MPPI_HOST_DEVICE inline NoiseLayout layout_time_major(int samples, int m) {  // [T][K][M]
  return NoiseLayout{m, samples * m};
}

// Everything a single rollout needs, as a POD so it can be passed by value
// into a kernel launch.
template <typename Scalar, int M>
struct RolloutParams {
  int horizon;
  Scalar dt;
  Scalar gamma;  // lambda * (1 - alpha)
  Scalar sigma[M];
  Scalar u_min[M];
  Scalar u_max[M];
  std::uint32_t seed;
  std::uint32_t iteration;
  NoiseLayout layout;
};

// Draws rollout k's control at timestep t from the Philox counter
// (seed, iteration, k, t), clamps to the actuator limits, and reports the
// post-clamp effective noise: what the rollout actually applied is also what
// the update must average, or clamped samples would pull the nominal toward
// controls nobody simulated.
template <typename Scalar, int M>
MPPI_HOST_DEVICE void sample_control(const RolloutParams<Scalar, M>& p, int k, int t,
                                     const Scalar* u_nom, Scalar* u_pert, Scalar* eps) {
  auto block = rng::Philox4x32_10::block(p.seed, p.iteration, static_cast<std::uint32_t>(k),
                                         static_cast<std::uint32_t>(t));
  float z[4];
  rng::normal_pair(block.w[0], block.w[1], z[0], z[1]);
  rng::normal_pair(block.w[2], block.w[3], z[2], z[3]);
  for (int i = 0; i < M; ++i) {
    Scalar raw = u_nom[i] + p.sigma[i] * static_cast<Scalar>(z[i]);
    u_pert[i] = clamp_to(raw, p.u_min[i], p.u_max[i]);
    eps[i] = u_pert[i] - u_nom[i];
  }
}

// Simulates rollout k over the horizon and returns its trajectory cost,
// writing the post-clamp effective noise into `noise` at the layout's slots.
//
// This is THE definition of a rollout: the CPU reference (controller.hpp)
// and the CUDA kernel (src/cuda/rollout.cu) both call it, so they cannot
// disagree about the algorithm — only about floating-point evaluation of
// the transcendentals inside, which is what the 1e-4 parity AC covers.
template <typename Dynamics, typename Cost, int M>
MPPI_HOST_DEVICE typename Dynamics::ScalarType rollout_and_cost(
    const Dynamics& dyn, const Cost& cost, const typename Dynamics::State& x0,
    const typename Dynamics::ScalarType* u_nominal,
    const RolloutParams<typename Dynamics::ScalarType, M>& p, int k,
    typename Dynamics::ScalarType* noise) {
  using Scalar = typename Dynamics::ScalarType;

  typename Dynamics::State x = x0;
  Scalar s_cost = Scalar(0);

  for (int t = 0; t < p.horizon; ++t) {
    Scalar u_pert[M];
    Scalar eps[M];
    const Scalar* u_nom = u_nominal + static_cast<std::size_t>(t) * M;
    sample_control(p, k, t, u_nom, u_pert, eps);

    for (int i = 0; i < M; ++i) {
      noise[p.layout.index(k, t, i)] = eps[i];
      s_cost += costs::control_coupling(p.gamma, u_nom[i], eps[i], p.sigma[i] * p.sigma[i]);
    }

    if constexpr (M == 1) {
      x = integrators::euler_step(dyn, x, u_pert[0], p.dt);
    } else {
      x = integrators::euler_step(dyn, x, u_pert, p.dt);
    }
    s_cost += cost.running(x, t);
  }

  return s_cost + cost.terminal(x);
}

}  // namespace mppi
