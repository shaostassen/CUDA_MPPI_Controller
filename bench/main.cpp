// S3.1 benchmark harness: times one MPPI iteration across backends at a
// given (K, T) and emits a CSV row per backend.
//
//   ./mppi_bench --config bench/configs/16k_50.json
//   ./mppi_bench --config ... --no-header >> results.csv
//
// scripts/run_table1.sh drives it over every config to regenerate Table 1.
//
// All timings are one full MPPI iteration — sample, roll out, reduce,
// update — with state resident wherever the backend allows, per §4.
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "bench/config.hpp"
#include "bench/power.hpp"
#include "bench/stats.hpp"
#include "cartpole_task.hpp"

#if defined(MPPI_BENCH_WITH_CUDA)
#include "mppi/cuda/controller.cuh"
#endif

namespace {

using Plant = cartpole_task::Plant;
using Cost = cartpole_task::SwingUpCost;

cartpole_task::Controller::Config make_config(const bench::BenchConfig& b) {
  auto cfg = cartpole_task::config(b.seed);
  cfg.samples = b.samples;
  cfg.horizon = b.horizon;
  cfg.dt = static_cast<float>(b.dt);
  cfg.lambda = static_cast<float>(b.lambda);
  cfg.sigma = {static_cast<float>(b.sigma)};
  cfg.u_min = {-static_cast<float>(b.u_limit)};
  cfg.u_max = {static_cast<float>(b.u_limit)};
  return cfg;
}

cartpole_task::State start_state() {
  return cartpole_task::State{0.0f, 0.0f, 3.14159265f, 0.0f};
}

// The host-timed loop, shared by both CPU backends. Each sample is one
// step() call; the state is held fixed so every iteration does identical
// work and the distribution reflects the controller, not the trajectory.
template <typename Controller>
bench::Stats time_host_controller(Controller& ctrl, const bench::BenchConfig& b) {
  auto x = start_state();
  std::array<float, 1> u{};
  for (int i = 0; i < b.warmup; ++i) {
    ctrl.step(x, u);
  }
  std::vector<double> ms;
  ms.reserve(b.measure);
  for (int i = 0; i < b.measure; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    ctrl.step(x, u);
    auto t1 = std::chrono::steady_clock::now();
    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  return bench::summarize(std::move(ms));
}

bool run_backend(const std::string& backend, const bench::BenchConfig& b, bench::Stats& out,
                 std::string& note) {
  auto cfg = make_config(b);

  if (backend == "cpu_1t") {
    cartpole_task::Controller ctrl(Plant{}, Cost{}, cfg);
    out = time_host_controller(ctrl, b);
    return true;
  }
  if (backend == "cpu_omp") {
#if defined(_OPENMP)
    mppi::Controller<Plant, Cost, 1, mppi::exec::OpenMp> ctrl(Plant{}, Cost{}, cfg);
    out = time_host_controller(ctrl, b);
    return true;
#else
    note = "built without OpenMP";
    return false;
#endif
  }

#if defined(MPPI_BENCH_WITH_CUDA)
  auto time_gpu = [&](mppi::cuda::ReductionPath path, mppi::cuda::NoiseMode mode) {
    mppi::cuda::GpuController<Plant, Cost, 1> ctrl(Plant{}, Cost{}, cfg);
    ctrl.set_reduction_path(path);
    ctrl.set_noise_mode(mode);
    auto x = start_state();

    for (int i = 0; i < b.warmup; ++i) {
      ctrl.step_device(x);
    }
    cudaDeviceSynchronize();

    // Device-timeline timing: CUDA events rather than the host clock, so the
    // number is the iteration itself and not launch queueing.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    std::vector<double> ms;
    ms.reserve(b.measure);
    for (int i = 0; i < b.measure; ++i) {
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
    return bench::summarize(std::move(ms));
  };

  if (backend == "cuda_v1") {
    out = time_gpu(mppi::cuda::ReductionPath::Library, mppi::cuda::NoiseMode::Stored);
    return true;
  }
  if (backend == "cuda_v2") {
    out = time_gpu(mppi::cuda::ReductionPath::HandRolled, mppi::cuda::NoiseMode::Stored);
    return true;
  }
  if (backend == "cuda_regen") {
    out = time_gpu(mppi::cuda::ReductionPath::HandRolled, mppi::cuda::NoiseMode::Regenerated);
    return true;
  }
#else
  if (backend.rfind("cuda", 0) == 0) {
    note = "built without CUDA";
    return false;
  }
#endif

  note = "unknown backend";
  return false;
}

void print_header() {
  std::cout << "config,K,T,backend,mean_ms,p50_ms,p99_ms,min_ms,max_ms,samples,"
               "power_w,power_available,note\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string config_path;
  bool header = true;
  int measure_override = 0;
  int warmup_override = -1;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
      config_path = argv[++i];
    } else if (std::strcmp(argv[i], "--no-header") == 0) {
      header = false;
    } else if (std::strcmp(argv[i], "--measure") == 0 && i + 1 < argc) {
      measure_override = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      warmup_override = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--help") == 0) {
      std::cerr << "usage: mppi_bench --config <file.json> [--no-header]\n"
                   "                  [--measure N] [--warmup N]\n"
                   "\n"
                   "--measure/--warmup override the config for smoke-testing the\n"
                   "pipeline. Published numbers must use the config's own values\n"
                   "(warmup 50 / measure 500, per PLAN.md §4).\n";
      return 0;
    }
  }
  if (config_path.empty()) {
    std::cerr << "error: --config is required\n";
    return 2;
  }

  bench::BenchConfig cfg;
  try {
    cfg = bench::load_config(config_path);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 2;
  }

  if (measure_override > 0) {
    cfg.measure = measure_override;
  }
  if (warmup_override >= 0) {
    cfg.warmup = warmup_override;
  }
  if (measure_override > 0 || warmup_override >= 0) {
    std::cerr << "warning: warmup/measure overridden (" << cfg.warmup << "/" << cfg.measure
              << ") — smoke-test numbers, not publishable ones\n";
  }

  if (header) {
    print_header();
  }

  for (const auto& backend : cfg.backends) {
    bench::PowerSampler sampler(cfg.power_rail);
    // Sample only across the measure window; warmup power is not
    // representative of steady state.
    bool sampling = sampler.start();

    bench::Stats s;
    std::string note;
    bool ok = run_backend(backend, cfg, s, note);

    bench::PowerSummary power = sampling ? sampler.stop() : bench::PowerSummary{};

    if (!ok) {
      std::cout << cfg.name << ',' << cfg.samples << ',' << cfg.horizon << ',' << backend
                << ",,,,,,0,,0," << note << "\n";
      std::cerr << "skipped " << backend << ": " << note << "\n";
      continue;
    }

    std::cout << cfg.name << ',' << cfg.samples << ',' << cfg.horizon << ',' << backend << ','
              << s.mean_ms << ',' << s.p50_ms << ',' << s.p99_ms << ',' << s.min_ms << ','
              << s.max_ms << ',' << s.count << ',';
    if (power.available) {
      std::cout << power.mean_w();
    }
    std::cout << ',' << (power.available ? 1 : 0) << ',' << note << "\n";
    std::cout.flush();
  }
  return 0;
}
