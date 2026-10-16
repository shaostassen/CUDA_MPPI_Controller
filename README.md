# cuda_mppi_controller

[![CI](https://github.com/shaostassen/CUDA_MPPI_Controller/actions/workflows/ci.yml/badge.svg)](https://github.com/shaostassen/CUDA_MPPI_Controller/actions/workflows/ci.yml)

GPU-accelerated Model Predictive Path Integral (MPPI) controller written in CUDA, targeting NVIDIA Jetson Orin (sm_87); developed on a Jetson AGX Orin. See [PLAN.md](PLAN.md) for the full backlog.

![cartpole swing-up under MPPI](docs/media/cartpole_swingup.gif)

Cartpole swing-up under the CPU reference controller (K = 512 sampled futures per
control step, T = 50). The faint arcs are a subsample of those futures, replayed
exactly as the controller evaluated them — see [docs/visualization.md](docs/visualization.md).

**Status:**

| Phase | State |
|---|---|
| A — CPU reference | Done: dynamics, counter-based RNG, MPPI core, LQR cross-check, OpenMP path, visualization. 49 CPU tests. |
| B — CUDA port | Written, **not yet run on a GPU**: shared rollout core and kernel, CUB/cuBLAS and hand-rolled warp-shuffle reductions, noise-regeneration variant. Syntax-checked against clang's CUDA frontend only, so there is no GPU parity or timing result yet. |
| B — benchmarks | Harness and power capture done; Table 1 is pending the first GPU run. |
| C — quadrotor | Not started. |

Next: first `scripts/gpu-test.sh` run on the AGX Orin (CPU↔GPU parity,
GPU swing-up), then `scripts/run_table1.sh`. Detail in
[docs/cuda-port.md](docs/cuda-port.md) and [docs/benchmarking.md](docs/benchmarking.md).

## Building

```
cmake --preset cpu      # CPU only, no CUDA toolchain needed
cmake --build --preset cpu
ctest --preset cpu
```

On the Orin, use `--preset release` (adds `ENABLE_CUDA=ON`, `sm_87`) or
`scripts/gpu-test.sh`. macOS needs `brew install libomp` and
`-DOpenMP_ROOT=$(brew --prefix libomp)` for the OpenMP path.

## Formatting

CI runs `clang-format --dry-run --Werror` (version 18, Ubuntu 24.04) over
`include/`, `src/` and `tests/`. To match it locally without installing anything:

```
find include src tests -regex '.*\.\(cpp\|hpp\|cuh\|cu\)' \
  | xargs uvx --from clang-format==18.1.8 clang-format -i
```

Docs: [environment](docs/env.md) · [OpenMP](docs/openmp.md) ·
[visualization](docs/visualization.md) · [CUDA port](docs/cuda-port.md) ·
[benchmarking](docs/benchmarking.md)
