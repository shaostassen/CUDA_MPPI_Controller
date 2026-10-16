# OpenMP path (S1.6)

The OpenMP variant is the same `mppi::Controller` with the `mppi::exec::OpenMp`
execution policy — no duplicated algorithm code. Static scheduling, disjoint
per-rollout writes, and per-timestep serial reductions keep its output
bit-identical to the serial path regardless of thread count (the AC's 1e-4
tolerance exists only as headroom for compiler codegen differences).

Bit-exactness is measured, not assumed: `OpenMpPath.ControlUpdateMatches
SerialAcrossIterations` records how many of its 60 iterations matched
exactly. Local run: 60/60 at 8 threads.

## Building

- Linux / Orin: nothing special, `find_package(OpenMP)` picks up libgomp.
- macOS (Apple clang): `brew install libomp`, then configure with
  `cmake --preset cpu -DOpenMP_ROOT=$(brew --prefix libomp)`.

If OpenMP is not found, the OpenMP tests and `omp_bench` are simply skipped;
everything else builds as before.

## Measuring (AC benchmark point: K = 16k, T = 50)

On the Orin (the recorded number — AGX Orin 64 GB, 12 homogeneous Cortex-A78AE cores):

```
OMP_NUM_THREADS=12 OMP_PROC_BIND=close OMP_PLACES=cores ./build/cpu/examples/omp_bench
```

`OMP_PROC_BIND=close OMP_PLACES=cores` pins one thread per physical core and
keeps them from migrating, which matters for run-to-run variance more than
for the mean. AC target: >= 4x vs single-thread.

| Machine | Threads | Serial ms | OpenMP ms | Speedup |
|---|---|---|---|---|
| AGX Orin 64 GB (AC number) | 12 | _pending first hardware run_ | | |
| M-series Mac (dev only) | 8 | 27.36 | 6.56 | 4.16x |

### Scaling measured on the dev Mac (4 performance + 4 efficiency cores)

| Threads | ms | Speedup | Efficiency |
|---|---|---|---|
| 1 | 27.36 | 1.00x | — |
| 2 | 14.42 | 1.89x | 95% |
| 4 | 7.62 | 3.61x | 90% |
| 6 | 7.97 | 3.44x | 57% |
| 8 | 6.56 | 4.16x | 52% |

The 6-thread regression is a heterogeneous-core artifact, not an algorithm
problem: `schedule(static)` hands every thread an equal K/6 chunk, but two
of those threads land on efficiency cores and the whole parallel region
waits on them. At 8 threads each chunk is smaller (K/8), so the straggler
costs less in absolute terms and throughput recovers.

The Orin's twelve A78AE cores are homogeneous, so this straggler effect should
not appear there and `schedule(static)` is the right choice. An earlier
projection of ~5x was for the six-core Orin Nano and is withdrawn: nothing
here predicts twelve cores on different silicon. With twelve cores the AC's 4x
is a low bar, so the number to record is the scaling curve (1, 2, 4, 6, 8, 12
threads) rather than a single point. The recorded numbers come from the Orin.

The workload is compute-bound rather than bandwidth-bound (K*T*M noise
floats = 3.2 MB written and read per iteration, ~240 MB/s at the serial
rate, against the AGX Orin 64 GB's 204.8 GB/s shared budget), and the per-step
cost is dominated by the `sin`/`cos` in the cartpole derivative. That is the
regime where core scaling is expected to hold up, but it is unmeasured on the
Orin.
