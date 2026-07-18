#!/usr/bin/env python3
"""Renders the S1.5 LQR cross-check plot from the committed CSV.

Usage:
    python3 scripts/plot_lqr_check.py [csv_in] [png_out]

Defaults: docs/plots/lqr_cross_check.csv -> docs/plots/lqr_cross_check.png.
Requires matplotlib only.
"""

import csv
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "x": "cart position x [m]",
    "x_dot": "cart velocity [m/s]",
    "theta": "pole angle [rad]",
    "theta_dot": "pole rate [rad/s]",
}


def main() -> None:
    csv_in = sys.argv[1] if len(sys.argv) > 1 else "docs/plots/lqr_cross_check.csv"
    png_out = sys.argv[2] if len(sys.argv) > 2 else "docs/plots/lqr_cross_check.png"

    rows = defaultdict(list)
    with open(csv_in, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["dim"]].append(
                (
                    float(row["value"]),
                    float(row["u_lqr_inf"]),
                    float(row["u_lqr_fin"]),
                    float(row["u_mppi_mean"]),
                    float(row["u_mppi_sem"]),
                )
            )

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharey=False)
    for ax, dim in zip(axes.flat, ["x", "x_dot", "theta", "theta_dot"]):
        data = sorted(rows[dim])
        v = [d[0] for d in data]
        ax.plot(v, [d[1] for d in data], "--", color="gray", label="LQR (infinite horizon)")
        ax.plot(v, [d[2] for d in data], "-", color="tab:blue", label="LQR (T=50, $w_{term}$)")
        ax.errorbar(
            v,
            [d[3] for d in data],
            yerr=[d[4] for d in data],
            fmt="o",
            ms=4,
            color="tab:red",
            label="MPPI mean control",
        )
        ax.axhline(0.0, color="black", lw=0.5)
        ax.axvline(0.0, color="black", lw=0.5)
        ax.set_xlabel(LABELS[dim])
        ax.set_ylabel("first control u [N]")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(
        "MPPI vs LQR near upright (K=512, T=50, $\\lambda$=2, $\\sigma$=3, "
        "$R=\\lambda\\Sigma^{-1}/2$) — single-dim state perturbations"
    )
    fig.tight_layout()
    fig.savefig(png_out, dpi=160)
    print(f"wrote {png_out}")


if __name__ == "__main__":
    main()
