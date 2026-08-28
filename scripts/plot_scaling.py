#!/usr/bin/env python3
"""S3.1: log-log scaling plot from the Table 1 CSV.

Usage:
    .venv/bin/python scripts/plot_scaling.py [csv_in] [png_out]

Defaults: docs/plots/table1.csv -> docs/plots/scaling.png.

Left panel is iteration latency against K (log-log, so a backend that scales
linearly is a straight line of slope 1 and any departure is visible rather
than argued). Right panel is speedup over the single-threaded CPU baseline,
which is the number the writeup actually claims.

Backends with no data — GPU rows from a CPU-only build, say — are skipped
with a note instead of being drawn as zeros.
"""

import csv
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {
    "cpu_1t": ("CPU 1 thread", "tab:gray", "o"),
    "cpu_omp": ("CPU OpenMP", "tab:blue", "s"),
    "cuda_v1": ("CUDA v1 (CUB + cuBLAS)", "tab:red", "^"),
    "cuda_v2": ("CUDA v2 (warp shuffle)", "tab:orange", "v"),
    "cuda_regen": ("CUDA v2 + regenerated noise", "tab:green", "D"),
}
BASELINE = "cpu_1t"
HORIZON = 50  # Table 1's primary sweep holds T fixed


def load(path):
    rows = defaultdict(dict)  # backend -> {K: mean_ms}
    powers = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["T"]) != HORIZON:
                continue
            if not row["mean_ms"]:
                continue
            k = int(row["K"])
            rows[row["backend"]][k] = float(row["mean_ms"])
            if row.get("power_available") == "1" and row.get("power_w"):
                powers[(row["backend"], k)] = float(row["power_w"])
    return rows, powers


def main() -> None:
    csv_in = sys.argv[1] if len(sys.argv) > 1 else "docs/plots/table1.csv"
    png_out = sys.argv[2] if len(sys.argv) > 2 else "docs/plots/scaling.png"

    rows, powers = load(csv_in)
    if not rows:
        print(f"no usable rows in {csv_in} (T={HORIZON})")
        return

    missing = [b for b in STYLE if b not in rows]
    if missing:
        print("no data for: " + ", ".join(missing))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    for backend, series in rows.items():
        label, color, marker = STYLE.get(backend, (backend, None, "o"))
        ks = sorted(series)
        ax.plot(ks, [series[k] for k in ks], marker=marker, color=color, label=label)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("rollouts K")
    ax.set_ylabel("iteration latency [ms]")
    ax.set_title(f"MPPI iteration latency (T = {HORIZON})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    base = rows.get(BASELINE, {})
    if base:
        for backend, series in rows.items():
            if backend == BASELINE:
                continue
            label, color, marker = STYLE.get(backend, (backend, None, "o"))
            ks = sorted(k for k in series if k in base)
            if not ks:
                continue
            ax2.plot(ks, [base[k] / series[k] for k in ks], marker=marker, color=color,
                     label=label)
        ax2.axhline(1.0, color="black", lw=0.8, ls="--")
        ax2.set_xscale("log", base=2)
        ax2.set_xlabel("rollouts K")
        ax2.set_ylabel(f"speedup vs {STYLE[BASELINE][0]}")
        ax2.set_title("Speedup over single-thread CPU")
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend(fontsize=8)
    else:
        ax2.set_visible(False)

    if powers:
        fig.text(0.5, 0.005,
                 "power column populated from tegrastats", ha="center", fontsize=7,
                 color="0.4")

    fig.tight_layout()
    fig.savefig(png_out, dpi=160)
    print(f"wrote {png_out}")


if __name__ == "__main__":
    main()
