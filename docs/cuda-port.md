# CUDA port (E2)

## The central design decision: one rollout, two backends

The naive way to port MPPI to CUDA is to write a kernel that reimplements
the rollout, then spend the parity budget hunting down the places where it
quietly disagrees with the CPU reference. This port does not do that.

`include/mppi/rollout.hpp` holds `rollout_and_cost()` — the single definition
of what a rollout *is*: draw the perturbed control from the Philox counter,
clamp it, accumulate the coupling term, integrate, accumulate the running
cost, add the terminal cost. It is marked `MPPI_HOST_DEVICE`, so:

- `Controller::step()` (CPU reference, and the OpenMP path) calls it
- `rollout_cost_kernel` (`include/mppi/cuda/rollout.cuh`) calls it

The kernel body is two lines: compute `k` from the thread index, call the
shared function. It contributes indexing and nothing else. CPU and GPU
therefore cannot disagree about the *algorithm* — only about floating-point
evaluation of the transcendentals inside, which is exactly what S2.2's
`<= 1e-4` relative tolerance is for.

## The noise layout, and why it is testable without a GPU

The one piece of genuinely new logic is the noise buffer layout. The CPU
reference stores noise rollout-major, `[K][T][M]`; the CUDA path stores it
time-major, `[T][K][M]`, so that at fixed `t` adjacent threads (`k`) touch
adjacent memory (PLAN.md §1 decision 9).

Rather than hard-coding two indexing schemes, `NoiseLayout` expresses both as
a pair of strides, and the rollout core is layout-agnostic. That makes the
layout testable on the host: `tests/layout_test.cpp` runs the same rollouts
under both layouts and requires identical costs and identical noise values in
corresponding slots. It passes bit-exactly.

So if GPU/CPU costs ever diverge, the layout is already ruled out.

## Status

| Piece | Verified how |
|---|---|
| Shared rollout core | CPU tests; reproduces controller costs bit-exactly |
| Noise layout equivalence | CPU test, bit-exact both directions |
| Device buffers (S2.1) | syntax-checked only |
| Rollout kernel v1 (S2.2) | syntax-checked only |
| CPU↔GPU cost parity | **pending Orin** (`tests/cuda_rollout_test.cu`) |

Nothing in the CUDA column has ever run. The Orin is the only CUDA hardware
for this project, so `scripts/gpu-test.sh` on that machine is what turns the
bottom three rows into results. Until then, treat the kernel as unproven.

## Iterating on kernels without a GPU

`scripts/cuda-syntax-check.sh` parses every `.cu`/`.cuh` with clang's CUDA
frontend against a minimal API stub (`scripts/cudastub/`), so authoring
mistakes surface locally instead of after an SSH round trip:

```
scripts/cuda-syntax-check.sh
```

It catches template instantiation failures, host/device attribute errors,
bad launch syntax, and typos. It does not prove `nvcc` will accept the code,
and says nothing about numerics or performance. A clean run means "worth
pushing to the Orin", not "verified".

One real portability bug already came out of it: `MPPI_HOST_DEVICE` keyed
only on `__CUDACC__`, which nvcc defines but clang-CUDA does not — under
clang every shared function would silently have become host-only. The guard
now accepts either macro.

## When the Orin is available

```
scripts/gpu-test.sh                     # builds ENABLE_CUDA=ON, runs everything
./build/release/tests/mppi_tests --gtest_filter='Cuda*:Philox*'
```

The parity tests record `max_rel` and `bit_exact` as test properties, so the
XML output says how much of the 1e-4 budget is actually being used rather
than just pass/fail.

Then S2.3 (reductions) can start against a kernel that is known-good.
