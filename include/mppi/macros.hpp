#pragma once

// Lets dynamics/cost headers be written once and compiled by both a plain
// host compiler (CPU-only builds, ENABLE_CUDA=OFF) and nvcc (device kernels
// from S2.x onward), without an #ifdef at every call site.
#if defined(__CUDACC__)
#define MPPI_HOST_DEVICE __host__ __device__
#else
#define MPPI_HOST_DEVICE
#endif
