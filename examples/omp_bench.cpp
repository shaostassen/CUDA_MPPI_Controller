// S1.6 speedup measurement: serial vs OpenMP MPPI iteration at K = 16k,
// T = 50 (the AC's benchmark point). The recorded AC number comes from the
// Orin's 6 A78AE cores:
//
//   OMP_NUM_THREADS=6 OMP_PROC_BIND=close OMP_PLACES=cores \
//     ./build/cpu/examples/omp_bench
//
// Runs elsewhere (e.g. a dev Mac) are engineering verification only — see
// docs/openmp.md.
#include <omp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <vector>

#include "cartpole_task.hpp"

namespace {

template <typename Controller>
double median_step_ms(int reps) {
  cartpole_task::Plant plant;
  auto cfg = cartpole_task::config(0);
  cfg.samples = 16384;
  cfg.horizon = 50;
  Controller ctrl(plant, {}, cfg);
  cartpole_task::State x{0.0f, 0.0f, 3.14159f, 0.0f};

  std::array<float, 1> u;
  for (int i = 0; i < 3; ++i) {
    ctrl.step(x, u);  // warmup
  }
  std::vector<double> ms;
  for (int i = 0; i < reps; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    ctrl.step(x, u);
    auto t1 = std::chrono::steady_clock::now();
    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  std::sort(ms.begin(), ms.end());
  return ms[ms.size() / 2];
}

}  // namespace

int main() {
  using Serial = cartpole_task::Controller;
  using Omp =
      mppi::Controller<cartpole_task::Plant, cartpole_task::SwingUpCost, 1, mppi::exec::OpenMp>;

  const int reps = 15;
  double serial_ms = median_step_ms<Serial>(reps);
  double omp_ms = median_step_ms<Omp>(reps);

  std::printf("K=16384 T=50, median of %d iterations\n", reps);
  std::printf("threads      : %d\n", omp_get_max_threads());
  std::printf("serial       : %8.3f ms\n", serial_ms);
  std::printf("openmp       : %8.3f ms\n", omp_ms);
  std::printf("speedup      : %8.2fx\n", serial_ms / omp_ms);
  return 0;
}
