#pragma once

#include <cuda_runtime.h>

#include "mppi/rollout.hpp"

namespace mppi::cuda {

// Softmax and control-update kernels for reduction path v1 (PLAN.md §1
// decision 8). The min and the sum come from CUB; these are the small
// element-wise passes between them.
//
// Note that beta and the normalizer arrive as DEVICE pointers, never as host
// scalars. Reading them back to compute the next launch's arguments would
// put a D2H sync inside the steady-state loop, which is precisely what
// S2.1's "no memcpy in the loop" criterion forbids.

// w[k] = exp(-(S_k - beta) / lambda). Baseline subtraction keeps exp() from
// underflowing when costs are large; it cancels in the normalizer, so the
// weights are mathematically unchanged by it.
template <typename Scalar>
__global__ void exp_weights_kernel(const Scalar* __restrict__ costs,
                                   const Scalar* __restrict__ beta, Scalar lambda, int samples,
                                   Scalar* __restrict__ weights) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= samples) {
    return;
  }
  weights[k] = expf(-(costs[k] - *beta) / lambda);
}

// w[k] /= Z, with Z the sum produced by CUB.
template <typename Scalar>
__global__ void normalize_kernel(Scalar* __restrict__ weights, const Scalar* __restrict__ z,
                                 int samples) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= samples) {
    return;
  }
  weights[k] *= Scalar(1) / *z;
}

// Applies the control update and the receding-horizon shift in one pass.
//
// The CPU reference does these as two steps — update every u[t], emit u[0],
// then shift left — but a shift done in place across blocks would race
// (thread t reading u[t+1] while thread t+1 writes it). Writing the shifted
// result straight into a second buffer is both race-free and one kernel
// instead of two:
//
//   u_out[t]   = clamp(u_in[t+1] + du[t+1])   for t < T-1
//   u_out[T-1] = clamp(u_in[T-1] + du[T-1])   (last entry repeats)
//   u_emit     = clamp(u_in[0] + du[0])       (the control actually applied)
//
// which is exactly the CPU sequence's end state.
template <typename Scalar, int M>
__global__ void update_shift_kernel(const Scalar* __restrict__ u_in,
                                    const Scalar* __restrict__ du,
                                    RolloutParams<Scalar, M> params, Scalar* __restrict__ u_out,
                                    Scalar* __restrict__ u_emit) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int T = params.horizon;
  if (t >= T) {
    return;
  }

  // Source index after the shift: every slot takes the next one's updated
  // value, and the final slot keeps its own.
  int src = (t + 1 < T) ? (t + 1) : t;

  for (int i = 0; i < M; ++i) {
    int si = src * M + i;
    u_out[t * M + i] = clamp_to(u_in[si] + du[si], params.u_min[i], params.u_max[i]);
  }

  if (t == 0) {
    for (int i = 0; i < M; ++i) {
      u_emit[i] = clamp_to(u_in[i] + du[i], params.u_min[i], params.u_max[i]);
    }
  }
}

}  // namespace mppi::cuda
