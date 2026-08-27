#pragma once

namespace mppi::exec {

// Execution policies for the Controller's two hot loops (rollouts over k,
// update over t). Kept as a template parameter so the reference stays a
// plain serial loop and the OpenMP path shares 100% of the algorithm code —
// a second copy of step() would drift.

struct Serial {
  template <typename F>
  static void for_range(int n, F&& f) {
    for (int i = 0; i < n; ++i) {
      f(i);
    }
  }
};

#if defined(_OPENMP)
// schedule(static): every rollout costs the same work (fixed T), and the
// deterministic iteration->thread assignment plus disjoint writes keeps the
// result bit-identical to the serial path — reductions over k happen
// serially inside one lambda call, never across threads.
struct OpenMp {
  template <typename F>
  static void for_range(int n, F&& f) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
      f(i);
    }
  }
};
#endif

}  // namespace mppi::exec
