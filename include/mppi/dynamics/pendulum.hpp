#pragma once

#include <cmath>

#include "mppi/macros.hpp"

namespace mppi::dynamics {

// Point-mass pendulum on a massless rigid rod, pivoted at the origin.
// theta = 0 is the downward (hanging, stable) equilibrium.
template <typename Scalar>
struct Pendulum {
  using ScalarType = Scalar;

  Scalar mass = Scalar(1);
  Scalar length = Scalar(1);
  Scalar gravity = Scalar(9.81);
  Scalar damping = Scalar(0);  // viscous damping at the pivot

  struct State {
    Scalar theta;
    Scalar omega;

    MPPI_HOST_DEVICE State operator+(const State& o) const {
      return State{theta + o.theta, omega + o.omega};
    }
    MPPI_HOST_DEVICE State operator*(Scalar s) const {
      return State{theta * s, omega * s};
    }
  };

  // Returns d(State)/dt, reusing State's layout: the result's `.theta` field
  // holds theta_dot and `.omega` holds omega_dot (this is what lets the
  // integrator combine states and derivatives with the same +/* operators).
  MPPI_HOST_DEVICE State derivative(const State& s, Scalar torque) const {
    Scalar inertia = mass * length * length;
    Scalar omega_dot =
        (-gravity / length) * std::sin(s.theta) - (damping * s.omega) / inertia + torque / inertia;
    return State{s.omega, omega_dot};
  }

  // Total mechanical energy, zeroed at the hanging equilibrium (theta=0).
  MPPI_HOST_DEVICE Scalar energy(const State& s) const {
    Scalar kinetic = Scalar(0.5) * mass * length * length * s.omega * s.omega;
    Scalar potential = mass * gravity * length * (Scalar(1) - std::cos(s.theta));
    return kinetic + potential;
  }
};

}  // namespace mppi::dynamics
