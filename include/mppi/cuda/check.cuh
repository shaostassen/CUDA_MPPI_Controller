#pragma once
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(expr)                                                              \
  do {                                                                                \
    cudaError_t err__ = (expr);                                                       \
    if (err__ != cudaSuccess) {                                                       \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, \
                   cudaGetErrorString(err__));                                        \
      std::abort();                                                                   \
    }                                                                                 \
  } while (0)
