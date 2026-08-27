#pragma once

#include <cuda_runtime.h>

#include "mppi/noise_source.hpp"
#include "mppi/partition.hpp"
#include "mppi/rollout.hpp"

namespace mppi::cuda {

// Reduction path v2 (PLAN.md S2.4): hand-rolled warp-shuffle reductions and
// a custom weighted-sum kernel, replacing CUB DeviceReduce and cuBLAS Sgemv.
// Both paths stay in the repo — which one is faster is an empirical question
// on the Orin, and the comparison is writeup material either way.
//
// The weighted-sum kernel is templated on the noise source, so the S2.5
// regenerate-instead-of-store variant reuses this reduction unchanged. That
// is what makes "bit-identical to the stored path" hold: same accumulation
// order, same code, only the origin of each epsilon differs.

constexpr int kWarpSize = 32;

struct MinOp {
  template <typename Scalar>
  MPPI_HOST_DEVICE Scalar operator()(Scalar a, Scalar b) const {
    return a < b ? a : b;
  }
  template <typename Scalar>
  MPPI_HOST_DEVICE static Scalar identity() {
    return INFINITY;
  }
};

struct SumOp {
  template <typename Scalar>
  MPPI_HOST_DEVICE Scalar operator()(Scalar a, Scalar b) const {
    return a + b;
  }
  template <typename Scalar>
  MPPI_HOST_DEVICE static Scalar identity() {
    return Scalar(0);
  }
};

template <typename Scalar, typename Op>
__device__ inline Scalar warp_reduce(Scalar v, Op op) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v = op(v, __shfl_down_sync(0xffffffffu, v, offset));
  }
  return v;
}

// Reduces one block's values to a single result in lane 0 of warp 0.
// Two stages: each warp shuffles down to its own leader, the leaders go to
// shared memory, then warp 0 shuffles those.
template <typename Scalar, int BlockSize, typename Op>
__device__ inline Scalar block_reduce(Scalar v, Op op) {
  static_assert(BlockSize % kWarpSize == 0, "block must be whole warps");
  constexpr int kWarps = BlockSize / kWarpSize;

  __shared__ Scalar warp_result[kWarps];
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;

  v = warp_reduce(v, op);
  if (lane == 0) {
    warp_result[warp] = v;
  }
  __syncthreads();

  // Only the first warp participates in the second stage; lanes past the
  // warp count take the identity so the shuffle tree stays well-defined.
  Scalar r = (lane < kWarps) ? warp_result[lane] : Op::template identity<Scalar>();
  if (warp == 0) {
    r = warp_reduce(r, op);
  }
  return r;
}

// Stage 1: each block reduces its slice of `in` into one partial.
template <typename Scalar, int BlockSize, typename Op>
__global__ void reduce_partials_kernel(const Scalar* __restrict__ in, int n,
                                       Scalar* __restrict__ partials, Op op) {
  Scalar acc = Op::template identity<Scalar>();
  for (int i = blockIdx.x * BlockSize + threadIdx.x; i < n; i += BlockSize * gridDim.x) {
    acc = op(acc, in[i]);
  }
  acc = block_reduce<Scalar, BlockSize>(acc, op);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = acc;
  }
}

// Stage 2: one block folds the partials into the final scalar.
template <typename Scalar, int BlockSize, typename Op>
__global__ void reduce_final_kernel(const Scalar* __restrict__ partials, int n,
                                    Scalar* __restrict__ out, Op op) {
  Scalar acc = Op::template identity<Scalar>();
  for (int i = threadIdx.x; i < n; i += BlockSize) {
    acc = op(acc, partials[i]);
  }
  acc = block_reduce<Scalar, BlockSize>(acc, op);
  if (threadIdx.x == 0) {
    *out = acc;
  }
}

// du[t][i] = sum_k w[k] * eps(k, t, i).
//
// One block per output element (there are T*M of them), threads striding
// over k. This is the replacement for the cuBLAS Sgemv, and unlike Sgemv it
// does not care how the noise is laid out — or whether it exists at all,
// which is what lets the regenerating source drop in.
template <typename Scalar, int M, int BlockSize, typename NoiseSrc>
__global__ void weighted_sum_kernel(NoiseSrc src, const Scalar* __restrict__ u_nominal,
                                    const Scalar* __restrict__ weights, int samples, int horizon,
                                    Scalar* __restrict__ du) {
  const int j = blockIdx.x;  // output element index, j = t*M + i
  if (j >= horizon * M) {
    return;
  }
  int t, i;
  split_output_index(j, M, t, i);

  Scalar acc = Scalar(0);
  for (int k = threadIdx.x; k < samples; k += BlockSize) {
    acc += weights[k] * src.eps(k, t, i, u_nominal);
  }
  acc = block_reduce<Scalar, BlockSize>(acc, SumOp{});
  if (threadIdx.x == 0) {
    du[j] = acc;
  }
}

}  // namespace mppi::cuda
