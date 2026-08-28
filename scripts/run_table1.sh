#!/usr/bin/env bash
# S3.1 AC: one command regenerates Table 1's CSV from scratch.
#
#   scripts/run_table1.sh [build_dir] [out_csv]
#
# Defaults to the release (CUDA) build on the Orin and
# docs/plots/table1.csv. On a machine without CUDA it still runs and fills
# the CPU rows, marking the GPU ones unavailable rather than omitting them.
#
# Environment captured alongside the numbers, because a latency table with no
# provenance is not reproducible:
#   * nvpmodel power mode and whether jetson_clocks is applied
#   * CUDA / driver / JetPack versions
#   * OMP_NUM_THREADS
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${1:-$ROOT/build/release}"
OUT="${2:-$ROOT/docs/plots/table1.csv}"
META="${OUT%.csv}_env.txt"
BENCH="$BUILD/bench/mppi_bench"

if [ ! -x "$BENCH" ]; then
  echo "error: $BENCH not found — build first, e.g. cmake --build --preset release" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"

{
  echo "date: $(date -Iseconds)"
  echo "host: $(uname -a)"
  echo "bench: $BENCH"
  echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-unset}"
  if command -v nvpmodel > /dev/null 2>&1; then
    echo "nvpmodel: $(sudo nvpmodel -q 2>/dev/null | tr '\n' ' ')"
  else
    echo "nvpmodel: not present (not a Jetson)"
  fi
  if command -v nvcc > /dev/null 2>&1; then
    echo "nvcc: $(nvcc --version | tail -1)"
  else
    echo "nvcc: not present"
  fi
  if [ -f /etc/nv_tegra_release ]; then
    echo "l4t: $(head -1 /etc/nv_tegra_release)"
  fi
  if command -v tegrastats > /dev/null 2>&1; then
    echo "tegrastats: available (power column populated)"
  else
    echo "tegrastats: absent (power column empty)"
  fi
} > "$META"

echo "environment -> $META"

# Lock clocks so run-to-run variance reflects the code, not the governor.
if command -v jetson_clocks > /dev/null 2>&1; then
  echo "locking clocks with jetson_clocks (needs sudo)"
  sudo jetson_clocks || echo "warning: jetson_clocks failed; numbers will be noisier" >&2
  echo "jetson_clocks: applied" >> "$META"
else
  echo "jetson_clocks: not present" >> "$META"
fi

# BENCH_ARGS exists to smoke-test this script quickly, e.g.
#   BENCH_ARGS="--measure 20 --warmup 5" scripts/run_table1.sh
# Published runs leave it unset so the §4 protocol applies.
read -r -a extra_args <<< "${BENCH_ARGS:-}"
if [ -n "${BENCH_ARGS:-}" ]; then
  echo "BENCH_ARGS: ${BENCH_ARGS} (SMOKE TEST — not publishable)" >> "$META"
fi

first=1
for cfg in "$ROOT"/bench/configs/*.json; do
  name="$(basename "$cfg" .json)"
  echo "running $name"
  if [ "$first" = "1" ]; then
    "$BENCH" --config "$cfg" "${extra_args[@]}" > "$OUT" || exit 1
    first=0
  else
    "$BENCH" --config "$cfg" --no-header "${extra_args[@]}" >> "$OUT" || exit 1
  fi
done

echo "wrote $OUT"
echo "plot with: .venv/bin/python scripts/plot_scaling.py $OUT"
