#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "mppi/dynamics/cartpole.hpp"
#include "mppi/dynamics/pendulum.hpp"
#include "mppi/integrators.hpp"

using mppi::dynamics::CartPole;
using mppi::dynamics::Pendulum;
using mppi::integrators::rk4_step;

// S1.1 AC: unforced pendulum energy drift < 0.1% over 10 s (RK4, dt 0.01, FP64).
TEST(PendulumDynamics, UnforcedEnergyConservationRK4) {
  Pendulum<double> pendulum;
  Pendulum<double>::State state{/*theta=*/1.0, /*omega=*/0.0};

  double dt = 0.01;
  int steps = static_cast<int>(10.0 / dt);

  double e0 = pendulum.energy(state);
  for (int i = 0; i < steps; ++i) {
    state = rk4_step(pendulum, state, /*torque=*/0.0, dt);
  }
  double e1 = pendulum.energy(state);

  double rel_drift = std::abs(e1 - e0) / std::abs(e0);
  EXPECT_LT(rel_drift, 0.001);
}

// S1.1 AC: small-angle period within 1% of the analytic SHM period.
TEST(PendulumDynamics, SmallAnglePeriodMatchesAnalytic) {
  Pendulum<double> pendulum;
  Pendulum<double>::State state{/*theta=*/0.05, /*omega=*/0.0};

  double dt = 0.001;
  double t = 0.0;

  // A turning point (omega crossing zero) occurs twice per period; the time
  // between the 1st and 3rd crossing is one full period.
  std::vector<double> crossing_times;
  double prev_omega = state.omega;
  double prev_t = t;

  while (crossing_times.size() < 3) {
    Pendulum<double>::State next = rk4_step(pendulum, state, 0.0, dt);
    double next_t = t + dt;

    if ((prev_omega <= 0.0 && next.omega > 0.0) || (prev_omega >= 0.0 && next.omega < 0.0)) {
      double frac = prev_omega / (prev_omega - next.omega);
      crossing_times.push_back(prev_t + frac * dt);
    }

    prev_omega = next.omega;
    prev_t = next_t;
    state = next;
    t = next_t;
  }

  double measured_period = crossing_times[2] - crossing_times[0];
  double analytic_period = 2.0 * M_PI * std::sqrt(pendulum.length / pendulum.gravity);

  double rel_error = std::abs(measured_period - analytic_period) / analytic_period;
  EXPECT_LT(rel_error, 0.01);
}

// Not part of S1.1's AC, but a hand-derived cart-pole model is easy to get a
// sign wrong in — both equilibria should be exactly stationary at rest.
TEST(CartPoleDynamics, UprightEquilibriumIsStationary) {
  CartPole<double> cartpole;
  CartPole<double>::State state{0.0, 0.0, /*theta=*/0.0, 0.0};
  auto ds = cartpole.derivative(state, /*force=*/0.0);
  EXPECT_NEAR(ds.x_dot, 0.0, 1e-12);
  EXPECT_NEAR(ds.theta_dot, 0.0, 1e-12);
}

TEST(CartPoleDynamics, HangingEquilibriumIsStationary) {
  CartPole<double> cartpole;
  CartPole<double>::State state{0.0, 0.0, /*theta=*/M_PI, 0.0};
  auto ds = cartpole.derivative(state, /*force=*/0.0);
  EXPECT_NEAR(ds.x_dot, 0.0, 1e-12);
  EXPECT_NEAR(ds.theta_dot, 0.0, 1e-12);
}
