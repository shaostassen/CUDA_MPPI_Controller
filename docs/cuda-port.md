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

## A layout tension in the plan, and how it is resolved

PLAN.md §1 decision 8 wants the noise usable as a column-major `K x (T*m)`
matrix so the weighted sum is a single cuBLAS `Sgemv`. Decision 9 pins the
layout as `[T][K][m]`. Those two are only compatible when `m == 1`:

| layout | element (k,t,i) | Sgemv view needs |
|---|---|---|
| `[T][K][m]` | `t*K*m + k*m + i` | `k + (t*m+i)*K` |
| `[T][m][K]` | `t*m*K + i*K + k` | `k + (t*m+i)*K` ✓ |

`[T][m][K]` satisfies the Sgemv view for every `m`, and is *better* coalesced
besides: adjacent `k` are adjacent in memory rather than `m` apart. For
`m == 1` it is the identical mapping, so adopting it changes nothing about
the cartpole work while leaving the quadrotor (`m = 4`) a layout that still
works. The device path uses it (`layout_gemv_major`); `layout_time_major` is
kept because decision 9 as written is still the right thing for a
hand-rolled reduction that does not go through cuBLAS (S2.4).

`tests/layout_test.cpp` emulates the exact `cublasSgemv` call on the host —
same transpose, same `lda`, same indexing — and requires it to reproduce the
reference weighted sum, so the call parameters are checked before cuBLAS is
ever involved.

## Status

| Piece | Verified how |
|---|---|
| Shared rollout core | CPU tests; reproduces controller costs bit-exactly |
| Noise layout equivalence (3 layouts) | CPU test, bit-exact |
| Sgemv call formulation | CPU test emulating the exact cuBLAS call |
| Device buffers (S2.1) | syntax-checked only |
| Rollout kernel v1 (S2.2) | syntax-checked only |
| Reduction path v1 (S2.3) | syntax-checked only |
| CPU↔GPU cost parity | **pending Orin** (`tests/cuda_rollout_test.cu`) |
| Full-iteration parity, GPU swing-up | **pending Orin** (`tests/cuda_iteration_test.cu`) |

Nothing in the CUDA column has ever run. The Orin is the only CUDA hardware
for this project, so `scripts/gpu-test.sh` on that machine is what turns the
bottom rows into results. Until then, treat the kernels as unproven.

## Keeping the iteration on-device

`GpuController::step_device()` runs rollout → CUB min → exp weights → CUB sum
→ normalize → `Sgemv` → fused update+shift without a single host round trip:
beta and the normalizer stay as device scalars consumed by the next kernel
rather than being read back. `x0` rides in as a kernel argument, so even the
closed loop transfers only the emitted control back — the shape S2.1 asks
for.

The update and the receding-horizon shift are fused into one kernel writing
to a second buffer. Shifting in place would race (thread `t` reading `u[t+1]`
while `t+1` writes it); double-buffering makes it race-free and saves a
launch.

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
