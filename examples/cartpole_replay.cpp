// S1.7 data generator: runs the 10-second cartpole swing-up and dumps both
// the executed trajectory and a subsample of the futures MPPI considered at
// each control step (the "ghost trails").
//
// The ghosts come from Controller::replay_rollout(), which regenerates each
// rollout from its Philox counters — so what gets drawn is bit-identical to
// what the controller actually costed, not a lookalike re-simulation.
//
// Usage:
//   ./build/cpu/examples/cartpole_replay <out_dir>
// writes <out_dir>/trajectory.csv and <out_dir>/ghosts.csv.
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "cartpole_task.hpp"
#include "mppi/integrators.hpp"

namespace {

constexpr float kDuration = 10.0f;
constexpr int kGhosts = 10;       // rollouts drawn per frame
constexpr int kGhostStride = 5;   // subsample along the horizon

}  // namespace

int main(int argc, char** argv) {
  std::string out_dir = argc > 1 ? argv[1] : ".";
  std::FILE* traj = std::fopen((out_dir + "/trajectory.csv").c_str(), "w");
  std::FILE* ghosts = std::fopen((out_dir + "/ghosts.csv").c_str(), "w");
  if (!traj || !ghosts) {
    std::fprintf(stderr, "cannot open output files in %s\n", out_dir.c_str());
    return 1;
  }

  cartpole_task::Plant plant;
  auto cfg = cartpole_task::config(0);
  cartpole_task::Controller ctrl(plant, {}, cfg);

  // Hanging start, the swing-up task's initial condition.
  cartpole_task::State x{0.0f, 0.0f, static_cast<float>(M_PI), 0.0f};

  std::fprintf(traj, "frame,t,x,x_dot,theta,theta_dot,u,beta\n");
  std::fprintf(ghosts, "frame,k,step,x,theta\n");

  const int steps = static_cast<int>(kDuration / cartpole_task::kDt);
  const int k_stride = cfg.samples / kGhosts;

  for (int frame = 0; frame < steps; ++frame) {
    float t = frame * cartpole_task::kDt;

    // Snapshot before stepping: this iteration's rollouts perturb around it.
    std::vector<float> u_before = ctrl.nominal_sequence();
    std::uint32_t iteration = ctrl.iteration();

    std::array<float, 1> u;
    float beta = ctrl.step(x, u);

    std::fprintf(traj, "%d,%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", frame, t, x.x, x.x_dot, x.theta,
                 x.theta_dot, u[0], beta);

    for (int g = 0; g < kGhosts; ++g) {
      int k = g * k_stride;
      auto trace = ctrl.replay_rollout(x, u_before, iteration, k);
      for (std::size_t s = 0; s < trace.states.size(); s += kGhostStride) {
        std::fprintf(ghosts, "%d,%d,%zu,%.5f,%.5f\n", frame, k, s, trace.states[s].x,
                     trace.states[s].theta);
      }
    }

    // Ground-truth environment integrates with RK4 (PLAN.md §1 decision 5).
    x = mppi::integrators::rk4_step(plant, x, u[0], cartpole_task::kDt);
  }

  std::fclose(traj);
  std::fclose(ghosts);
  std::fprintf(stderr, "wrote %d frames to %s\n", steps, out_dir.c_str());
  return 0;
}
