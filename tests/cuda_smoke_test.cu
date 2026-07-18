#include <gtest/gtest.h>

#include <vector>

#include "mppi/cuda/smoke.cuh"

// Trivial kernel test required by S0.2's acceptance criteria — exercises
// malloc/memcpy/launch/memcpy-back on real device hardware. Only built when
// ENABLE_CUDA=ON, so it only ever runs on the Orin.
TEST(CudaSmoke, VectorAddMatchesHost) {
  constexpr int n = 1024;
  std::vector<float> a(n), b(n), c(n);
  for (int i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(2 * i);
  }

  mppi::cuda_smoke::vector_add(a.data(), b.data(), c.data(), n);

  for (int i = 0; i < n; ++i) {
    ASSERT_FLOAT_EQ(c[i], a[i] + b[i]);
  }
}
