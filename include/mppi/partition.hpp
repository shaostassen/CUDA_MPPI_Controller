#pragma once

#include "mppi/macros.hpp"

namespace mppi {

// Index arithmetic for the reduction kernels, kept free of any CUDA include
// so the host tests can exercise the same code the kernels run rather than a
// re-typed copy of it. Off-by-ones here would corrupt the control update
// silently, which makes them worth testing directly.

// Flat output index j -> (timestep, control dim) for the T x M sequence.
MPPI_HOST_DEVICE inline void split_output_index(int j, int m, int& t, int& i) {
  t = j / m;
  i = j - t * m;
}

// The half-open range of items a block-strided worker visits:
// for (k = start; k < n; k += stride). Expressed as a helper so a test can
// assert the partition is exactly a cover of [0, n).
MPPI_HOST_DEVICE inline int strided_count(int start, int n, int stride) {
  return start >= n ? 0 : 1 + (n - 1 - start) / stride;
}

}  // namespace mppi
