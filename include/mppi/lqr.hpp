#pragma once

#include <array>
#include <cmath>

namespace mppi::lqr {

// Discrete-time LQR reference for the S1.5 cross-check. Validation-grade on
// purpose: fixed 4-state / 1-input shapes, FP64 (oracle-test precision per
// PLAN.md §1 decision 3), plain loops. Not used by the controller itself.

using Mat4 = std::array<std::array<double, 4>, 4>;
using Vec4 = std::array<double, 4>;

struct LinearSystem {
  Mat4 A;
  Vec4 B;
};

// Linearization of the cartpole about upright (x = 0, theta = 0), state
// order (x, x_dot, theta, theta_dot), zero damping, discretized with the
// SAME integrator MPPI plans with (explicit Euler, decision 5) — comparing
// against the RK4 truth instead would fold integrator mismatch into the
// gain comparison.
//
// Continuous-time, from dynamics/cartpole.hpp with sin t -> t, cos t -> 1,
// theta_dot^2 -> 0:
//   x_ddot     = (F - m g theta) / M
//   theta_ddot = (g theta - x_ddot) / l
inline LinearSystem linearize_cartpole_euler(double cart_mass, double pole_mass, double pole_length,
                                             double gravity, double dt) {
  const double M = cart_mass, m = pole_mass, l = pole_length, g = gravity;

  Mat4 a_c{{{0, 1, 0, 0}, {0, 0, -m * g / M, 0}, {0, 0, 0, 1}, {0, 0, (M + m) * g / (M * l), 0}}};
  Vec4 b_c{0, 1.0 / M, 0, -1.0 / (M * l)};

  LinearSystem sys;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      sys.A[i][j] = (i == j ? 1.0 : 0.0) + a_c[i][j] * dt;
    }
    sys.B[i] = b_c[i] * dt;
  }
  return sys;
}

namespace detail {

// One backward Riccati step: P <- Q + A'PA - A'PB (r + B'PB)^-1 B'PA,
// re-symmetrized. Without the symmetrization the fixed-point iteration
// actually diverges here: rounding introduces a tiny antisymmetric
// component, the Riccati map only contracts on the symmetric cone, and the
// antisymmetric error grows like rho(A)^2 per iteration through the
// unstable open-loop mode — observed empirically as convergence by ~iter
// 200, then blow-up with s < 0 by ~iter 600.
//
// Returns the max element-wise change.
inline double riccati_step(Mat4& p, const LinearSystem& sys, const Vec4& q_diag, double r) {
  Mat4 pa{};
  Vec4 pb{};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        pa[i][j] += p[i][k] * sys.A[k][j];
      }
      pb[i] += p[i][j] * sys.B[j];
    }
  }
  double s = r;
  Vec4 bpa{};
  for (int i = 0; i < 4; ++i) {
    s += sys.B[i] * pb[i];
    for (int j = 0; j < 4; ++j) {
      bpa[j] += sys.B[i] * pa[i][j];
    }
  }
  Mat4 next{};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      double atpa = 0.0;
      for (int k = 0; k < 4; ++k) {
        atpa += sys.A[k][i] * pa[k][j];
      }
      next[i][j] = (i == j ? q_diag[i] : 0.0) + atpa - bpa[i] * bpa[j] / s;
    }
  }
  double max_delta = 0.0;
  for (int i = 0; i < 4; ++i) {
    for (int j = i; j < 4; ++j) {
      double sym = 0.5 * (next[i][j] + next[j][i]);
      max_delta = std::max(max_delta, std::fabs(sym - p[i][j]));
      p[i][j] = sym;
      p[j][i] = sym;
    }
  }
  return max_delta;
}

// K = (r + B'PB)^-1 B'PA for u = -K x.
inline Vec4 gain_from_p(const Mat4& p, const LinearSystem& sys, double r) {
  Mat4 pa{};
  Vec4 pb{};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        pa[i][j] += p[i][k] * sys.A[k][j];
      }
      pb[i] += p[i][j] * sys.B[j];
    }
  }
  double s = r;
  for (int i = 0; i < 4; ++i) {
    s += sys.B[i] * pb[i];
  }
  Vec4 k{};
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 4; ++i) {
      k[j] += sys.B[i] * pa[i][j];
    }
    k[j] /= s;
  }
  return k;
}

}  // namespace detail

// Infinite-horizon gain: iterate the Riccati step to a fixed point.
inline Vec4 dare_gain(const LinearSystem& sys, const Vec4& q_diag, double r, int max_iter = 100000,
                      double tol = 1e-11) {
  Mat4 p{};
  for (int i = 0; i < 4; ++i) {
    p[i][i] = q_diag[i];
  }
  for (int iter = 0; iter < max_iter; ++iter) {
    if (detail::riccati_step(p, sys, q_diag, r) < tol) {
      break;
    }
  }
  return detail::gain_from_p(p, sys, r);
}

// First-step gain of the finite-horizon LQ problem with terminal weight
// P_T = diag(q_terminal): T backward Riccati steps from the terminal cost.
// This is the problem MPPI with horizon T and terminal cost w_term is
// actually approximating, so it is the fair cross-check target; dare_gain
// differs from it exactly by the light-terminal/finite-horizon effect.
inline Vec4 finite_horizon_gain(const LinearSystem& sys, const Vec4& q_diag, const Vec4& q_terminal,
                                double r, int horizon) {
  Mat4 p{};
  for (int i = 0; i < 4; ++i) {
    p[i][i] = q_terminal[i];
  }
  for (int t = 0; t < horizon; ++t) {
    detail::riccati_step(p, sys, q_diag, r);
  }
  return detail::gain_from_p(p, sys, r);
}

}  // namespace mppi::lqr
