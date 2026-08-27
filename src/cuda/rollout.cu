#include "mppi/cuda/rollout.cuh"
#include "mppi/dynamics/pendulum.hpp"

// rollout.cuh is a template header, so none of its device code is compiled
// until something instantiates it. This translation unit instantiates the
// kernel against a library-owned dynamics model, which means a device-side
// compile error shows up when the library itself is built rather than in
// whichever test first happens to use the kernel.
//
// The tests instantiate the same kernel with the real cartpole task.

namespace mppi::cuda {
namespace {

struct PendulumCanaryCost {
  MPPI_HOST_DEVICE float running(const dynamics::Pendulum<float>::State& s, int) const {
    return s.theta * s.theta + 0.1f * s.omega * s.omega;
  }
  MPPI_HOST_DEVICE float terminal(const dynamics::Pendulum<float>::State& s) const {
    return 10.0f * s.theta * s.theta;
  }
};

}  // namespace

// Exported so the instantiation cannot be discarded. Not used by the
// controller — the pendulum is the simplest model that exercises the whole
// rollout path on device.
void launch_pendulum_canary(const dynamics::Pendulum<float>& dyn,
                            const dynamics::Pendulum<float>::State& x0, const float* u_nominal,
                            const RolloutParams<float, 1>& params, int samples, float* noise,
                            float* costs) {
  launch_rollout_cost<dynamics::Pendulum<float>, PendulumCanaryCost, 1>(
      dyn, PendulumCanaryCost{}, x0, u_nominal, params, samples, noise, costs);
}

}  // namespace mppi::cuda
