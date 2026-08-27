// S2.4 / S2.5 micro-benchmark: reduction path v1 (CUB + cuBLAS) against v2
// (warp-shuffle + custom weighted sum), and stored noise against regenerated.
//
// Produces the rows that decide which path becomes the default and what goes
// into Table 3. Run on the Orin with clocks locked:
//
//   jetson_clocks
//   ./build/release/examples/cuda_reduction_bench > docs/plots/reduction_bench.csv
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <string>
#include <vector>

#include "cartpole_task.hpp"
#include "mppi/cuda/controller.cuh"

namespace {

using Plant = cartpole_task::Plant;
using Cost = cartpole_task::SwingUpCost;
using GpuCtrl = mppi::cuda::GpuController<Plant, Cost, 1>;

struct Timing {
  double mean_ms;
  double p50_ms;
  double p99_ms;
  std::size_t noise_bytes;
};

// Timed with CUDA events on the device timeline. Warmup 50 / measure 500 per
// the §4 protocol; the profiler's own numbers are never used for the table.
Timing time_path(int samples, mppi::cuda::ReductionPath path, mppi::cuda::NoiseMode mode) {
  Plant plant;
  Cost cost;
  auto cfg = cartpole_task::config(0);
  cfg.samples = samples;

  GpuCtrl ctrl(plant, cost, cfg);
  ctrl.set_reduction_path(path);
  ctrl.set_noise_mode(mode);  // may force HandRolled; that is intended

  cartpole_task::State x{0.0f, 0.0f, 3.14159265f, 0.0f};

  for (int i = 0; i < 50; ++i) {
    ctrl.step_device(x);
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::vector<double> ms;
  ms.reserve(500);
  for (int i = 0; i < 500; ++i) {
    cudaEventRecord(start);
    ctrl.step_device(x);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float el = 0.0f;
    cudaEventElapsedTime(&el, start, stop);
    ms.push_back(el);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  std::sort(ms.begin(), ms.end());
  double sum = 0.0;
  for (double v : ms) {
    sum += v;
  }
  Timing t;
  t.mean_ms = sum / ms.size();
  t.p50_ms = ms[ms.size() / 2];
  t.p99_ms = ms[static_cast<std::size_t>(ms.size() * 0.99)];
  t.noise_bytes = ctrl.noise_bytes();
  return t;
}

}  // namespace

int main() {
  std::printf("K,T,path,noise_mode,mean_ms,p50_ms,p99_ms,noise_bytes\n");

  const int T = 50;
  for (int K : {1024, 4096, 16384, 65536}) {
    struct Case {
      const char* path_name;
      const char* mode_name;
      mppi::cuda::ReductionPath path;
      mppi::cuda::NoiseMode mode;
    };
    const Case cases[] = {
        {"v1_library", "stored", mppi::cuda::ReductionPath::Library,
         mppi::cuda::NoiseMode::Stored},
        {"v2_handrolled", "stored", mppi::cuda::ReductionPath::HandRolled,
         mppi::cuda::NoiseMode::Stored},
        {"v2_handrolled", "regenerated", mppi::cuda::ReductionPath::HandRolled,
         mppi::cuda::NoiseMode::Regenerated},
    };
    for (const auto& c : cases) {
      Timing t = time_path(K, c.path, c.mode);
      std::printf("%d,%d,%s,%s,%.4f,%.4f,%.4f,%zu\n", K, T, c.path_name, c.mode_name, t.mean_ms,
                  t.p50_ms, t.p99_ms, t.noise_bytes);
      std::fflush(stdout);
    }
  }
  return 0;
}
