#include <gtest/gtest.h>
#include <omp.h>

#include <array>
#include <cmath>

#include "cartpole_task.hpp"
#include "mppi/integrators.hpp"

// Only compiled when OpenMP is available (see tests/CMakeLists.txt).

namespace {

using OmpController =
    mppi::Controller<cartpole_task::Plant, cartpole_task::SwingUpCost, 1, mppi::exec::OpenMp>;

}  // namespace

// S1.6 AC: the OpenMP control update matches single-thread within 1e-4 rel
// (summation-order tolerance). Our structure — static schedule, disjoint
// writes, per-(t) serial reductions — should make it bit-identical; the
// AC tolerance stays as the assertion so a compiler that vectorizes the
// outlined loop body differently doesn't fail the build.
TEST(OpenMpPath, ControlUpdateMatchesSerialAcrossIterations) {
  cartpole_task::Plant plant;
  cartpole_task::Controller serial(plant, {}, cartpole_task::config(11));
  OmpController parallel(plant, {}, cartpole_task::config(11));

  // Drive both along the same trajectory (serial's closed loop) so every
  // iteration compares the full update pipeline at an identical state.
  cartpole_task::State x{0.0f, 0.0f, static_cast<float>(M_PI), 0.0f};
  int exact = 0;
  const int kIters = 60;
  for (int i = 0; i < kIters; ++i) {
    std::array<float, 1> u_serial, u_parallel;
    serial.step(x, u_serial);
    parallel.step(x, u_parallel);

    float tol = 1e-4f * std::max(1.0f, std::fabs(u_serial[0]));
    ASSERT_NEAR(u_parallel[0], u_serial[0], tol) << "iteration " << i;
    if (u_parallel[0] == u_serial[0]) {
      ++exact;
    }
    x = mppi::integrators::rk4_step(plant, x, u_serial[0], cartpole_task::kDt);
  }
  RecordProperty("bit_exact_iterations", exact);
  RecordProperty("omp_max_threads", omp_get_max_threads());
}

// The parallel controller must also work closed loop on its own outputs.
TEST(OpenMpPath, SwingUpSucceedsClosedLoop) {
  for (unsigned seed : {0u, 1u}) {
    cartpole_task::Plant plant;
    OmpController ctrl(plant, {}, cartpole_task::config(seed));
    cartpole_task::State x{0.0f, 0.0f, static_cast<float>(M_PI), 0.0f};
    float t = 0.0f, held_since = -1.0f;
    bool ok = false;
    while (t < 6.5f) {
      std::array<float, 1> u;
      ctrl.step(x, u);
      x = mppi::integrators::rk4_step(plant, x, u[0], cartpole_task::kDt);
      t += cartpole_task::kDt;
      bool in_goal = std::fabs(x.theta) < 0.1f && std::fabs(x.x) < 0.5f;
      if (in_goal) {
        if (held_since < 0.0f) held_since = t;
        if (t - held_since >= 1.0f && held_since <= 5.0f) {
          ok = true;
          break;
        }
      } else {
        held_since = -1.0f;
      }
    }
    EXPECT_TRUE(ok) << "seed " << seed;
  }
}
