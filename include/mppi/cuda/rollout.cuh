#pragma once

#include <cuda_runtime.h>

#include "mppi/cuda/check.cuh"
#include "mppi/rollout.hpp"

namespace mppi::cuda {

inline int grid_for(int n, int block) {
  return (n + block - 1) / block;
}

// Kernel v1 (PLAN.md §1 decision 7): one thread = one rollout, rollout and
// cost accumulation fused, noise in global memory. Simple, correct,
// profilable — cleverness comes later and has to justify itself in Table 3.
//
// The body is deliberately trivial: every line of MPPI semantics lives in
// rollout_and_cost(), which the CPU reference calls too. The kernel
// contributes indexing and nothing else, so CPU<->GPU divergence can only
// come from floating-point evaluation, never from two drifting
// implementations of the algorithm.
template <typename Dynamics, typename Cost, int M, bool StoreNoise = true>
__global__ void rollout_cost_kernel(Dynamics dyn, Cost cost, typename Dynamics::State x0,
                                    const typename Dynamics::ScalarType* __restrict__ u_nominal,
                                    RolloutParams<typename Dynamics::ScalarType, M> params,
                                    int samples, typename Dynamics::ScalarType* __restrict__ noise,
                                    typename Dynamics::ScalarType* __restrict__ costs) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= samples) {
    return;
  }
  costs[k] =
      rollout_and_cost<Dynamics, Cost, M, StoreNoise>(dyn, cost, x0, u_nominal, params, k, noise);
}

// Launches kernel v1. `params.layout` should be the time-major layout for
// device runs so that at fixed t adjacent threads touch adjacent memory.
template <typename Dynamics, typename Cost, int M, bool StoreNoise = true>
void launch_rollout_cost(const Dynamics& dyn, const Cost& cost, const typename Dynamics::State& x0,
                         const typename Dynamics::ScalarType* u_nominal,
                         const RolloutParams<typename Dynamics::ScalarType, M>& params, int samples,
                         typename Dynamics::ScalarType* noise, typename Dynamics::ScalarType* costs,
                         int block_size = 256, cudaStream_t stream = nullptr) {
  rollout_cost_kernel<Dynamics, Cost, M, StoreNoise>
      <<<grid_for(samples, block_size), block_size, 0, stream>>>(dyn, cost, x0, u_nominal, params,
                                                                 samples, noise, costs);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace mppi::cuda
