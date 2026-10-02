#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>
#include <type_traits>
#include <utility>
#include <vector>

#include "mppi/controller.hpp"
#include "mppi/cuda/check.cuh"
#include "mppi/cuda/device_buffers.cuh"
#include "mppi/cuda/reduce.cuh"
#include "mppi/cuda/reduce_v2.cuh"
#include "mppi/cuda/rollout.cuh"
#include "mppi/noise_source.hpp"

namespace mppi::cuda {

// Which reduction implementation an iteration uses. Both ship: PLAN.md S2.4
// makes the faster one the default once measured on the Orin, and keeps the
// loser for the writeup.
enum class ReductionPath {
  Library,     // v1: CUB DeviceReduce + cuBLAS Sgemv
  HandRolled,  // v2: warp-shuffle reductions + custom weighted sum
};

// Whether epsilon is kept in global memory or recomputed from its Philox
// counter during the update (PLAN.md S2.5). Regenerating saves K*T*M floats
// and costs one Philox block per (k, t) — a memory-for-compute trade whose
// sign is a hardware question, especially on a bandwidth-limited Orin.
enum class NoiseMode {
  Stored,
  Regenerated,
};

#define CUBLAS_CHECK(expr)                                                                  \
  do {                                                                                      \
    cublasStatus_t st__ = (expr);                                                           \
    if (st__ != CUBLAS_STATUS_SUCCESS) {                                                    \
      std::fprintf(stderr, "cuBLAS error %d for %s at %s:%d\n", int(st__), #expr, __FILE__, \
                   __LINE__);                                                               \
      std::abort();                                                                         \
    }                                                                                       \
  } while (0)

// Reduction path v1 (PLAN.md §1 decision 8): CUB DeviceReduce for the min and
// the normalizer, a small kernel for each element-wise pass, cuBLAS Sgemv for
// the weighted noise sum. S2.4 will add hand-rolled warp-shuffle reductions
// behind the same interface; whichever measures faster becomes the default,
// and both stay in-repo because the comparison is writeup material.
//
// Everything after the launch stays on device: beta and Z are device scalars
// consumed by the next kernel rather than read back, so a steady-state
// iteration performs no host<->device transfer at all. x0 rides in as a
// kernel argument (no memcpy), and only the emitted control comes back —
// which is exactly the "closed loop copies only x0" shape S2.1 asks for.
template <typename Dynamics, typename Cost, int M = 1>
class GpuController {
  static_assert(M >= 1 && M <= 4, "one Philox block per (k,t) covers at most 4 control dims");
  static_assert(std::is_same_v<typename Dynamics::ScalarType, float>,
                "the reduction path is Sgemv: FP32 only, per PLAN.md decision 3");

 public:
  using Scalar = typename Dynamics::ScalarType;
  using State = typename Dynamics::State;
  using Config = ControllerConfig<Scalar, M>;

  GpuController(Dynamics dynamics, Cost cost, Config config)
      : dyn_(dynamics),
        cost_(cost),
        cfg_(config),
        d_u_(static_cast<std::size_t>(config.horizon) * M),
        d_u_next_(static_cast<std::size_t>(config.horizon) * M),
        d_du_(static_cast<std::size_t>(config.horizon) * M),
        d_noise_(static_cast<std::size_t>(config.samples) * config.horizon * M),
        d_costs_(config.samples),
        d_weights_(config.samples),
        d_beta_(1),
        d_z_(1),
        d_emit_(M),
        d_partials_(kMaxPartials) {
    d_u_.zero();
    d_u_next_.zero();

    CUBLAS_CHECK(cublasCreate(&blas_));

    // CUB wants a scratch buffer whose size it reports; ask for both
    // reductions and keep the larger, so the steady-state loop never
    // allocates.
    std::size_t min_bytes = 0, sum_bytes = 0;
    CUDA_CHECK(::cub::DeviceReduce::Min(nullptr, min_bytes, d_costs_.data(), d_beta_.data(),
                                        cfg_.samples));
    CUDA_CHECK(
        ::cub::DeviceReduce::Sum(nullptr, sum_bytes, d_weights_.data(), d_z_.data(), cfg_.samples));
    temp_bytes_ = std::max(min_bytes, sum_bytes);
    d_temp_ = DeviceBuffer<unsigned char>(temp_bytes_);
  }

  ~GpuController() {
    if (blas_ != nullptr) {
      cublasDestroy(blas_);
    }
  }

  GpuController(const GpuController&) = delete;
  GpuController& operator=(const GpuController&) = delete;

  // One full MPPI iteration on device, leaving the emitted control in
  // device memory. Nothing is read back.
  void step_device(const State& x0) {
    const int K = cfg_.samples;
    const int T = cfg_.horizon;
    const auto params = rollout_params();

    if (noise_mode_ == NoiseMode::Stored) {
      launch_rollout_cost<Dynamics, Cost, M, true>(dyn_, cost_, x0, d_u_.data(), params, K,
                                                   d_noise_.data(), d_costs_.data(), block_size_);
    } else {
      // Nothing to write: epsilon gets recomputed during the update.
      launch_rollout_cost<Dynamics, Cost, M, false>(dyn_, cost_, x0, d_u_.data(), params, K,
                                                    nullptr, d_costs_.data(), block_size_);
    }

    reduce_beta(K);
    exp_weights_kernel<Scalar><<<grid_for(K, block_size_), block_size_>>>(
        d_costs_.data(), d_beta_.data(), cfg_.lambda, K, d_weights_.data());
    CUDA_CHECK(cudaGetLastError());
    reduce_normalizer(K);

    normalize_kernel<Scalar>
        <<<grid_for(K, block_size_), block_size_>>>(d_weights_.data(), d_z_.data(), K);
    CUDA_CHECK(cudaGetLastError());

    weighted_sum(params, K, T);

    update_shift_kernel<Scalar, M><<<grid_for(T, block_size_), block_size_>>>(
        d_u_.data(), d_du_.data(), params, d_u_next_.data(), d_emit_.data());
    CUDA_CHECK(cudaGetLastError());

    std::swap(d_u_, d_u_next_);
    ++iteration_;
  }

  void set_reduction_path(ReductionPath p) {
    reduction_path_ = p;
  }
  ReductionPath reduction_path() const {
    return reduction_path_;
  }

  // Regenerating requires the hand-rolled weighted sum: cuBLAS needs a
  // materialized matrix, and the whole point is not to materialize one.
  void set_noise_mode(NoiseMode m) {
    noise_mode_ = m;
    if (m == NoiseMode::Regenerated) {
      reduction_path_ = ReductionPath::HandRolled;
    }
  }
  NoiseMode noise_mode() const {
    return noise_mode_;
  }

  // Device bytes held for noise — zero in regenerated mode. This is the
  // memory saving S2.5 asks to record.
  std::size_t noise_bytes() const {
    return noise_mode_ == NoiseMode::Stored ? d_noise_.size() * sizeof(Scalar) : 0;
  }

  // Closed-loop convenience: one iteration, then bring back the M floats the
  // caller has to apply to the plant.
  void step(const State& x0, std::array<Scalar, M>& u_out) {
    step_device(x0);
    d_emit_.download(u_out.data(), M);
  }

  // Reads the current nominal sequence back to the host. Diagnostics and
  // parity tests only — this synchronizes, so keep it out of timed loops.
  std::vector<Scalar> nominal_sequence() const {
    std::vector<Scalar> u(static_cast<std::size_t>(cfg_.horizon) * M);
    d_u_.download(u.data(), u.size());
    return u;
  }

  std::vector<Scalar> rollout_costs() const {
    std::vector<Scalar> c(cfg_.samples);
    d_costs_.download(c.data(), c.size());
    return c;
  }

  std::vector<Scalar> rollout_weights() const {
    std::vector<Scalar> w(cfg_.samples);
    d_weights_.download(w.data(), w.size());
    return w;
  }

  Scalar beta() const {
    Scalar b{};
    d_beta_.download(&b, 1);
    return b;
  }

  std::uint32_t iteration() const {
    return iteration_;
  }
  void set_block_size(int n) {
    block_size_ = n;
  }

  // Same packing as the CPU controller, except the noise layout: the device
  // path uses the gemv layout so the weighted sum is one Sgemv (see
  // layout_gemv_major's comment for why this is not [T][K][M]).
  RolloutParams<Scalar, M> rollout_params() const {
    RolloutParams<Scalar, M> p{};
    p.horizon = cfg_.horizon;
    p.dt = cfg_.dt;
    p.gamma = costs::coupling_gamma(cfg_.lambda, cfg_.alpha);
    for (int i = 0; i < M; ++i) {
      p.sigma[i] = cfg_.sigma[i];
      p.u_min[i] = cfg_.u_min[i];
      p.u_max[i] = cfg_.u_max[i];
    }
    p.seed = cfg_.seed;
    p.iteration = iteration_;
    p.layout = layout_gemv_major(cfg_.samples, M);
    return p;
  }

 private:
  static constexpr int kReduceBlock = 256;
  static constexpr int kMaxPartials = 1024;

  // beta = min_k S_k.
  void reduce_beta(int K) {
    if (reduction_path_ == ReductionPath::Library) {
      CUDA_CHECK(::cub::DeviceReduce::Min(d_temp_.data(), temp_bytes_, d_costs_.data(),
                                          d_beta_.data(), K));
      return;
    }
    int grid = std::min(kMaxPartials, grid_for(K, kReduceBlock));
    reduce_partials_kernel<Scalar, kReduceBlock, MinOp>
        <<<grid, kReduceBlock>>>(d_costs_.data(), K, d_partials_.data(), MinOp{});
    CUDA_CHECK(cudaGetLastError());
    reduce_final_kernel<Scalar, kReduceBlock, MinOp>
        <<<1, kReduceBlock>>>(d_partials_.data(), grid, d_beta_.data(), MinOp{});
    CUDA_CHECK(cudaGetLastError());
  }

  // Z = sum_k exp(-(S_k - beta)/lambda).
  void reduce_normalizer(int K) {
    if (reduction_path_ == ReductionPath::Library) {
      CUDA_CHECK(
          ::cub::DeviceReduce::Sum(d_temp_.data(), temp_bytes_, d_weights_.data(), d_z_.data(), K));
      return;
    }
    int grid = std::min(kMaxPartials, grid_for(K, kReduceBlock));
    reduce_partials_kernel<Scalar, kReduceBlock, SumOp>
        <<<grid, kReduceBlock>>>(d_weights_.data(), K, d_partials_.data(), SumOp{});
    CUDA_CHECK(cudaGetLastError());
    reduce_final_kernel<Scalar, kReduceBlock, SumOp>
        <<<1, kReduceBlock>>>(d_partials_.data(), grid, d_z_.data(), SumOp{});
    CUDA_CHECK(cudaGetLastError());
  }

  // du[t][i] = sum_k w[k] * eps(k, t, i).
  void weighted_sum(const RolloutParams<Scalar, M>& params, int K, int T) {
    if (reduction_path_ == ReductionPath::Library) {
      // The noise buffer read as a column-major K x (T*M) matrix with
      // leading dimension K — a view that is only valid for the gemv layout,
      // which rollout_params() pins.
      const Scalar one = Scalar(1), zero = Scalar(0);
      CUBLAS_CHECK(cublasSgemv(blas_, CUBLAS_OP_T, K, T * M, &one, d_noise_.data(), K,
                               d_weights_.data(), 1, &zero, d_du_.data(), 1));
      return;
    }
    // d_u_ still holds the nominal sequence the rollouts perturbed —
    // update_shift_kernel writes to the other buffer — so regeneration sees
    // exactly the controls the rollout used.
    if (noise_mode_ == NoiseMode::Stored) {
      StoredNoise<Scalar, M> src{d_noise_.data(), params.layout};
      weighted_sum_kernel<Scalar, M, kReduceBlock, StoredNoise<Scalar, M>>
          <<<T * M, kReduceBlock>>>(src, d_u_.data(), d_weights_.data(), K, T, d_du_.data());
    } else {
      RegeneratedNoise<Scalar, M> src{params};
      weighted_sum_kernel<Scalar, M, kReduceBlock, RegeneratedNoise<Scalar, M>>
          <<<T * M, kReduceBlock>>>(src, d_u_.data(), d_weights_.data(), K, T, d_du_.data());
    }
    CUDA_CHECK(cudaGetLastError());
  }

  Dynamics dyn_;
  Cost cost_;
  Config cfg_;
  std::uint32_t iteration_ = 0;
  int block_size_ = 256;
  ReductionPath reduction_path_ = ReductionPath::Library;
  NoiseMode noise_mode_ = NoiseMode::Stored;

  DeviceBuffer<Scalar> d_u_, d_u_next_, d_du_, d_noise_, d_costs_, d_weights_, d_beta_, d_z_,
      d_emit_, d_partials_;
  DeviceBuffer<unsigned char> d_temp_;
  std::size_t temp_bytes_ = 0;
  cublasHandle_t blas_ = nullptr;
};

}  // namespace mppi::cuda
