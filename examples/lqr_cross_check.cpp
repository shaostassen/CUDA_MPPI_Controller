// S1.5 data generator: sweeps single-dimension state perturbations about
// upright and prints CSV comparing MPPI's mean control against the
// infinite-horizon and finite-horizon (T = 50, w_term terminal) LQ gains.
//
// Regenerate the committed artifact with:
//   ./build/cpu/examples/lqr_cross_check > docs/plots/lqr_cross_check.csv
//   python3 scripts/plot_lqr_check.py
#include <cmath>
#include <cstdio>
#include <vector>

#include "cartpole_task.hpp"
#include "mppi/lqr.hpp"

int main() {
  auto sys = mppi::lqr::linearize_cartpole_euler(1.0, 0.1, 0.5, 9.81, cartpole_task::kDt);
  mppi::lqr::Vec4 q{cartpole_task::kRunWeights[0], cartpole_task::kRunWeights[1],
                    cartpole_task::kRunWeights[2], cartpole_task::kRunWeights[3]};
  mppi::lqr::Vec4 q_term{cartpole_task::kTermWeights[0], cartpole_task::kTermWeights[1],
                         cartpole_task::kTermWeights[2], cartpole_task::kTermWeights[3]};
  // Implicit MPPI control cost: R = lambda * Sigma^-1 / 2 (KL of N(u, Sigma)
  // from N(0, Sigma), Williams-style information-theoretic derivation).
  const double r = cartpole_task::kLambda / (2.0 * cartpole_task::kSigma * cartpole_task::kSigma);

  auto k_inf = mppi::lqr::dare_gain(sys, q, r);
  auto k_fin = mppi::lqr::finite_horizon_gain(sys, q, q_term, r, 50);

  const char* names[4] = {"x", "x_dot", "theta", "theta_dot"};
  const double ranges[4] = {0.3, 0.4, 0.06, 0.2};
  const int kSeeds = 20;
  const int kWindow = 20;

  std::printf("dim,value,u_lqr_inf,u_lqr_fin,u_mppi_mean,u_mppi_sem\n");
  for (int dim = 0; dim < 4; ++dim) {
    for (int step = -4; step <= 4; ++step) {
      if (step == 0) {
        continue;
      }
      double v = ranges[dim] * step / 4.0;
      double xv[4] = {0, 0, 0, 0};
      xv[dim] = v;
      cartpole_task::State s{static_cast<float>(xv[0]), static_cast<float>(xv[1]),
                              static_cast<float>(xv[2]), static_cast<float>(xv[3])};
      double u_inf = 0.0, u_fin = 0.0;
      for (int j = 0; j < 4; ++j) {
        u_inf -= k_inf[j] * xv[j];
        u_fin -= k_fin[j] * xv[j];
      }
      // Per-seed means for a standard error alongside the grand mean.
      std::vector<double> per_seed;
      for (unsigned seed = 0; seed < kSeeds; ++seed) {
        cartpole_task::Plant plant;
        cartpole_task::Controller ctrl(plant, cartpole_task::SwingUpCost{},
                                       cartpole_task::config(seed * 7919u + 1u));
        std::array<float, 1> u{};
        double acc = 0.0;
        for (int i = 0; i < 10 + kWindow; ++i) {
          ctrl.step(s, u);
          if (i >= 10) {
            acc += u[0];
          }
        }
        per_seed.push_back(acc / kWindow);
      }
      double mean = 0.0;
      for (double m : per_seed) {
        mean += m;
      }
      mean /= per_seed.size();
      double var = 0.0;
      for (double m : per_seed) {
        var += (m - mean) * (m - mean);
      }
      double sem = std::sqrt(var / (per_seed.size() - 1) / per_seed.size());
      std::printf("%s,%.4f,%.6f,%.6f,%.6f,%.6f\n", names[dim], v, u_inf, u_fin, mean, sem);
      std::fflush(stdout);
    }
  }
  return 0;
}
