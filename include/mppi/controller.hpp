#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "mppi/costs/quadratic.hpp"
#include "mppi/integrators.hpp"
#include "mppi/philox.hpp"
#include "mppi/weights.hpp"

namespace mppi {

// Single-thread CPU reference of information-theoretic MPPI (PLAN.md §1
// decisions 5-6): Euler rollouts, gamma-coupled running cost, baseline-
// subtracted softmax update. This is the semantic oracle every later
// backend (OpenMP S1.6, CUDA E2) gets tested against, so clarity beats
// speed throughout.
//
// Cost must provide running(state, t) and terminal(state). Dynamics is any
// model exposing State/ScalarType/derivative. M is the control dimension
// (one Philox block = 4 words covers M <= 4 noise draws per (k, t)).
template <typename Dynamics, typename Cost, int M = 1>
class Controller {
  static_assert(M >= 1 && M <= 4, "one Philox block per (k,t) covers at most 4 control dims");

 public:
  using Scalar = typename Dynamics::ScalarType;
  using State = typename Dynamics::State;

  struct Config {
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

    for (int k = 0; k < K; ++k) {
      State x = x0;
      Scalar s_cost = Scalar(0);
      for (int t = 0; t < T; ++t) {
        auto block = rng::Philox4x32_10::block(cfg_.seed, iteration_, static_cast<std::uint32_t>(k),
                                               static_cast<std::uint32_t>(t));
        float z[4];
        rng::normal_pair(block.w[0], block.w[1], z[0], z[1]);
        rng::normal_pair(block.w[2], block.w[3], z[2], z[3]);

        Scalar u_pert[M];
        Scalar* eps = noise_slot(k, t);
        for (int i = 0; i < M; ++i) {
          Scalar u_nom = u_[static_cast<std::size_t>(t) * M + i];
          Scalar raw = u_nom + cfg_.sigma[i] * static_cast<Scalar>(z[i]);
          u_pert[i] = std::clamp(raw, cfg_.u_min[i], cfg_.u_max[i]);
          // Post-clamp effective noise: what the rollout actually applied is
          // also what the update must average, or clamped samples would pull
          // the nominal toward controls nobody simulated.
          eps[i] = u_pert[i] - u_nom;
          s_cost += costs::control_coupling(gamma, u_nom, eps[i], cfg_.sigma[i] * cfg_.sigma[i]);
        }
        if constexpr (M == 1) {
          x = integrators::euler_step(dyn_, x, u_pert[0], cfg_.dt);
        } else {
          x = integrators::euler_step(dyn_, x, u_pert, cfg_.dt);
        }
        s_cost += cost_.running(x, t);
      }
      s_cost += cost_.terminal(x);
      costs_[k] = s_cost;
    }

    Scalar beta = weights::softmax_weights(costs_.data(), K, cfg_.lambda, weights_.data());

    for (int t = 0; t < T; ++t) {
      for (int i = 0; i < M; ++i) {
        Scalar du = Scalar(0);
        for (int k = 0; k < K; ++k) {
          du += weights_[k] * noise_slot(k, t)[i];
        }
        std::size_t idx = static_cast<std::size_t>(t) * M + i;
        u_[idx] = std::clamp(u_[idx] + du, cfg_.u_min[i], cfg_.u_max[i]);
      }
    }

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
  std::uint32_t iteration() const { return iteration_; }

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
