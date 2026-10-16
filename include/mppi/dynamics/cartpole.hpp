#pragma once

#include <cmath>

#include "mppi/macros.hpp"

namespace mppi::dynamics {

// Cart with a point-mass pole pivoted on top, point mass at distance
// `pole_length` from the pivot. theta = 0 is upright (the swing-up goal);
// theta = pi is hanging straight down (the swing-up start state).
//
// Derived via Lagrangian mechanics with generalized coordinates (x, theta);
// upright/hanging equilibria and their stability were checked by hand
// against the closed form before this was written down.
template <typename Scalar>
struct CartPole {
  using ScalarType = Scalar;

  Scalar cart_mass = Scalar(1);
  Scalar pole_mass = Scalar(0.1);
  Scalar pole_length = Scalar(0.5);  // pivot to point mass
  Scalar gravity = Scalar(9.81);
  Scalar cart_damping = Scalar(0);  // viscous damping on the cart
  Scalar pole_damping = Scalar(0);  // viscous damping at the pole pivot

  struct State {
    Scalar x;
    Scalar x_dot;
    Scalar theta;
    Scalar theta_dot;

    MPPI_HOST_DEVICE State operator+(const State& o) const {
      return State{x + o.x, x_dot + o.x_dot, theta + o.theta, theta_dot + o.theta_dot};
    }
    MPPI_HOST_DEVICE State operator*(Scalar s) const {
      return State{x * s, x_dot * s, theta * s, theta_dot * s};
    }
  };

  // Returns d(State)/dt, reusing State's layout: `.x` holds x_dot, `.x_dot`
  // holds x_ddot, `.theta` holds theta_dot, `.theta_dot` holds theta_ddot.
  MPPI_HOST_DEVICE State derivative(const State& s, Scalar force) const {
    Scalar sin_t = std::sin(s.theta);
    Scalar cos_t = std::cos(s.theta);
    Scalar M = cart_mass;
    Scalar m = pole_mass;
    Scalar l = pole_length;
    Scalar g = gravity;

    Scalar denom = M + m * sin_t * sin_t;

    Scalar x_ddot = (force - cart_damping * s.x_dot + m * l * sin_t * s.theta_dot * s.theta_dot -
                     m * g * sin_t * cos_t + (pole_damping * s.theta_dot * cos_t) / l) /
                    denom;

    Scalar theta_ddot = (g * sin_t - cos_t * x_ddot - pole_damping * s.theta_dot / (m * l)) / l;

    return State{s.x_dot, x_ddot, s.theta_dot, theta_ddot};
  }
};

}  // namespace mppi::dynamics
