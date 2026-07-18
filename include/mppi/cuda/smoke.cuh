#pragma once

namespace mppi::cuda_smoke {

// Toolchain-verification kernel for S0.2 — proves the nvcc/CMake/CUDA
// runtime path end to end. Not part of the MPPI algorithm.
void vector_add(const float* a, const float* b, float* c, int n);

}  // namespace mppi::cuda_smoke
