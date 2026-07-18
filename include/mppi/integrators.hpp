#pragma once

#include "mppi/macros.hpp"

namespace mppi::integrators {

template <typename System, typename Control>
MPPI_HOST_DEVICE typename System::State euler_step(const System& sys,
                                                     const typename System::State& s,
                                                     const Control& u,
                                                     typename System::ScalarType dt) {
  typename System::State ds = sys.derivative(s, u);
  return s + ds * dt;
}

template <typename System, typename Control>
MPPI_HOST_DEVICE typename System::State rk4_step(const System& sys,
                                                   const typename System::State& s,
                                                   const Control& u,
                                                   typename System::ScalarType dt) {
  using Scalar = typename System::ScalarType;
  typename System::State k1 = sys.derivative(s, u);
  typename System::State k2 = sys.derivative(s + k1 * (dt * Scalar(0.5)), u);
  typename System::State k3 = sys.derivative(s + k2 * (dt * Scalar(0.5)), u);
  typename System::State k4 = sys.derivative(s + k3 * dt, u);
  return s + (k1 + k2 * Scalar(2) + k3 * Scalar(2) + k4) * (dt / Scalar(6));
}

}  // namespace mppi::integrators
