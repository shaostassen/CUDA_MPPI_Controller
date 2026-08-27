#pragma once

// Lets dynamics/cost headers be written once and compiled by both a plain
// host compiler (CPU-only builds, ENABLE_CUDA=OFF) and nvcc (device kernels
// from S2.x onward), without an #ifdef at every call site.
// nvcc defines __CUDACC__; clang compiling CUDA defines __CUDA__ instead.
// Checking only the former would make this expand to nothing under clang,
// turning every shared function host-only and failing at the first device
// call site with a confusing diagnostic.
#if defined(__CUDACC__) || defined(__CUDA__)
#define MPPI_HOST_DEVICE __host__ __device__
#else
#define MPPI_HOST_DEVICE
#endif
