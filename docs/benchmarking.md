# Benchmark harness (S3.1)

One command regenerates Table 1:

```
scripts/run_table1.sh                      # release build, docs/plots/table1.csv
.venv/bin/python scripts/plot_scaling.py   # docs/plots/scaling.png
```

It sweeps every config in `bench/configs/`, timing each backend the config
lists, and writes one CSV row per (config, backend).

## What a row contains

```
config,K,T,backend,mean_ms,p50_ms,p99_ms,min_ms,max_ms,samples,power_w,power_available,note
```

`p99` is reported alongside the mean because the tail is what decides Table
2: a controller averaging 5 ms that spikes to 40 ms misses its deadline.
Percentiles are nearest-rank, so every figure printed is a latency that was
actually observed rather than an interpolation between two that were not.

A backend that cannot run still emits a row, with empty timings and a `note`
saying why (`built without CUDA`, `built without OpenMP`). Dropping the row
instead would make a CPU-only run look like a complete table.

## Backends

| name | what it times |
|---|---|
| `cpu_1t` | single-threaded reference |
| `cpu_omp` | OpenMP execution policy |
| `cuda_v1` | CUB `DeviceReduce` + cuBLAS `Sgemv` |
| `cuda_v2` | warp-shuffle reductions + custom weighted sum |
| `cuda_regen` | v2 with noise regenerated from Philox counters |

CPU backends are timed on the host clock; CUDA backends with CUDA events on
the device timeline, so the figure is the iteration rather than launch
queueing.

## Power

`power_w` comes from `tegrastats`, sampled only across the measure window —
warmup power is not steady state. The rail defaults to `VDD_IN` (whole-board)
and is configurable per config, because rail naming moves between JetPack
releases and carrier boards.

Off a Jetson there is no `tegrastats` and no honest substitute, so the column
is empty and `power_available` is 0. It is never filled with a guess.

This column is what makes the Orin numbers worth publishing: a latency table
alone invites "so what, a desktop GPU is faster", whereas Hz/W is the
comparison an embedded controller is judged on.

## Provenance

Every run writes `<out>_env.txt` next to the CSV: date, host, `nvpmodel`
mode, whether `jetson_clocks` was applied, `nvcc` and L4T versions,
`OMP_NUM_THREADS`, and whether tegrastats was available. A latency table
without that is not reproducible, and the numbers move enough with power mode
alone to make it matter.

## Protocol

Warmup 50 / measure 500 per PLAN.md §4, set in the configs rather than the
code so a config fully describes its own run.

`--measure`/`--warmup` override them for smoke-testing the pipeline, and
`BENCH_ARGS` forwards those through `run_table1.sh`:

```
BENCH_ARGS="--measure 20 --warmup 5" scripts/run_table1.sh build/cpu /tmp/smoke.csv
```

Both paths announce loudly that the result is not publishable — on stderr and
in the env file — because the failure mode here is a smoke run quietly
becoming a committed table.

## Status

The harness runs today and produces real CPU numbers on any machine. What is
pending hardware:

- every `cuda_*` row
- the `power_w` column
- `jetson_clocks` / `nvpmodel` locking, which is what makes run-to-run
  variance small enough for the p99 column to mean anything

So the committed `docs/plots/table1.csv` stays absent until the Orin
produces one. Numbers measured on a dev Mac are useful for checking the
harness works — they are not Table 1.
