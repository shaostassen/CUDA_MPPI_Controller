#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "mppi/costs/quadratic.hpp"
#include "mppi/execution.hpp"
#include "mppi/integrators.hpp"
#include "mppi/philox.hpp"
#include "mppi/rollout.hpp"
#include "mppi/weights.hpp"

namespace mppi {

// Standalone (not nested) so controllers that differ only in execution
// policy share one Config type — a nested Config would make the Serial and
// OpenMp instantiations mutually unconvertible for no reason.
template <typename Scalar, int M>
struct ControllerConfig {
  int samples = 512;              // K
  int horizon = 50;               // T
  Scalar dt = Scalar(0.02);
  Scalar lambda = Scalar(1);
  Scalar alpha = Scalar(0);       // gamma = lambda * (1 - alpha)
  std::array<Scalar, M> sigma{};  // sampling std dev per dim; must be > 0
  std::array<Scalar, M> u_min{};
  std::array<Scalar, M> u_max{};
  std::uint32_t seed = 0;
};

// Single-thread CPU reference of information-theoretic MPPI (PLAN.md §1
// decisions 5-6): Euler rollouts, gamma-coupled running cost, baseline-
// subtracted softmax update. This is the semantic oracle every later
// backend (OpenMP S1.6, CUDA E2) gets tested against, so clarity beats
// speed throughout.
//
// Cost must provide running(state, t) and terminal(state). Dynamics is any
// model exposing State/ScalarType/derivative. M is the control dimension
// (one Philox block = 4 words covers M <= 4 noise draws per (k, t)).
// Exec parallelizes the rollout and update loops (see execution.hpp); the
// structure keeps parallel results bit-identical to serial, so S1.6's
// 1e-4-rel parity AC holds with margin to spare.
template <typename Dynamics, typename Cost, int M = 1, typename Exec = exec::Serial>
class Controller {
  static_assert(M >= 1 && M <= 4, "one Philox block per (k,t) covers at most 4 control dims");

 public:
  using Scalar = typename Dynamics::ScalarType;
  using State = typename Dynamics::State;

  using Config = ControllerConfig<Scalar, M>;

  Controller(Dynamics dynamics, Cost cost, Config config)
      : dyn_(dynamics),
        cost_(cost),
        cfg_(config),
        u_(static_cast<std::size_t>(cfg_.horizon) * M, Scalar(0)),
        noise_(static_cast<std::size_t>(cfg_.samples) * cfg_.horizon * M),
        costs_(cfg_.samples),
        weights_(cfg_.samples) {}

  // One MPPI iteration against current state x0: sample K noisy rollouts,
  // cost them, softmax-weight, update the nominal sequence, emit u_0, shift.
  // Returns beta (min trajectory cost) for logging.
  Scalar step(const State& x0, std::array<Scalar, M>& u_out) {
    const Scalar gamma = costs::coupling_gamma(cfg_.lambda, cfg_.alpha);
    const int K = cfg_.samples;
    const int T = cfg_.horizon;

    const auto params = rollout_params();
    Exec::for_range(K, [&](int k) {
      costs_[k] = rollout_and_cost<Dynamics, Cost, M>(dyn_, cost_, x0, u_.data(), params, k,
                                                      noise_.data());
    });

    // Serial on purpose: O(K) and it keeps the weight normalization order
    // identical regardless of thread count.
    Scalar beta = weights::softmax_weights(costs_.data(), K, cfg_.lambda, weights_.data());

    // Parallel over t: each (t, i) sum over k runs serially inside one
    // thread, so summation order — and therefore the result — never depends
    // on the thread count.
    Exec::for_range(T, [&](int t) {
      for (int i = 0; i < M; ++i) {
        Scalar du = Scalar(0);
        for (int k = 0; k < K; ++k) {
          du += weights_[k] * noise_slot(k, t)[i];
        }
        std::size_t idx = static_cast<std::size_t>(t) * M + i;
        u_[idx] = std::clamp(u_[idx] + du, cfg_.u_min[i], cfg_.u_max[i]);
      }
    });

    for (int i = 0; i < M; ++i) {
      u_out[i] = u_[i];
    }

    // Receding horizon: shift one step left, repeat the last entry.
    for (int t = 0; t + 1 < T; ++t) {
      for (int i = 0; i < M; ++i) {
        u_[static_cast<std::size_t>(t) * M + i] = u_[static_cast<std::size_t>(t + 1) * M + i];
      }
    }
    for (int i = 0; i < M; ++i) {
      u_[static_cast<std::size_t>(T - 1) * M + i] = u_[static_cast<std::size_t>(T - 2) * M + i];
    }

    ++iteration_;
    return beta;
  }

  const std::vector<Scalar>& nominal_sequence() const { return u_; }
  const std::vector<Scalar>& rollout_costs() const { return costs_; }
  const std::vector<Scalar>& rollout_weights() const { return weights_; }
  std::uint32_t iteration() const { return iteration_; }

  struct RolloutTrace {
    std::vector<State> states;     // horizon + 1, starting at x0
    std::vector<Scalar> controls;  // horizon * M, post-clamp applied control
    std::vector<Scalar> eps;       // horizon * M, post-clamp effective noise
  };

  // Replays rollout k of `iteration` exactly as step() simulated it, given
  // the nominal sequence as it stood before that step(). Noise is
  // regenerated from Philox counters rather than stored trajectories, so
  // the hot loop pays nothing for visualization. Shares sample_control()
  // with step(), so ghost trails cannot drift from what was really rolled
  // out (S1.7; S4.6's hero render uses the same path).
  RolloutTrace replay_rollout(const State& x0, const std::vector<Scalar>& u_snapshot,
                              std::uint32_t iteration, int k) const {
    RolloutTrace trace;
    trace.states.reserve(static_cast<std::size_t>(cfg_.horizon) + 1);
    trace.controls.resize(static_cast<std::size_t>(cfg_.horizon) * M);
    trace.eps.resize(static_cast<std::size_t>(cfg_.horizon) * M);
    trace.states.push_back(x0);

    auto params = rollout_params();
    params.iteration = iteration;

    State x = x0;
    for (int t = 0; t < cfg_.horizon; ++t) {
      std::size_t off = static_cast<std::size_t>(t) * M;
      mppi::sample_control<Scalar, M>(params, k, t, &u_snapshot[off], &trace.controls[off],
                                      &trace.eps[off]);
      if constexpr (M == 1) {
        x = integrators::euler_step(dyn_, x, trace.controls[off], cfg_.dt);
      } else {
        Scalar u_pert[M];
        for (int i = 0; i < M; ++i) {
          u_pert[i] = trace.controls[off + i];
        }
        x = integrators::euler_step(dyn_, x, u_pert, cfg_.dt);
      }
      trace.states.push_back(x);
    }
    return trace;
  }

  // Packs the config into the POD the shared rollout core consumes. The CPU
  // reference stores noise rollout-major ([K][T][M]); the CUDA path swaps in
  // the time-major layout for coalescing (PLAN.md §1 decision 9). Public so
  // the CUDA backend can build launch parameters from the same config.
  RolloutParams<Scalar, M> rollout_params() const {
    RolloutParams<Scalar, M> p{};
    p.horizon = cfg_.horizon;
    p.dt = cfg_.dt;
    p.gamma = costs::coupling_gamma(cfg_.lambda, cfg_.alpha);
    for (int i = 0; i < M; ++i) {
      p.sigma[i] = cfg_.sigma[i];
      p.u_min[i] = cfg_.u_min[i];
      p.u_max[i] = cfg_.u_max[i];
    }
    p.seed = cfg_.seed;
    p.iteration = iteration_;
    p.layout = layout_rollout_major(cfg_.horizon, M);
    return p;
  }

 private:
  Scalar* noise_slot(int k, int t) {
    return noise_.data() + (static_cast<std::size_t>(k) * cfg_.horizon + t) * M;
  }

  Dynamics dyn_;
  Cost cost_;
  Config cfg_;
  std::uint32_t iteration_ = 0;
  std::vector<Scalar> u_;       // nominal control sequence, [T][M]
  std::vector<Scalar> noise_;   // effective noise per rollout, [K][T][M]
  std::vector<Scalar> costs_;   // S_k
  std::vector<Scalar> weights_;
};

}  // namespace mppi
