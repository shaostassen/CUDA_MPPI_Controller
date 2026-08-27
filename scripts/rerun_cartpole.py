#!/usr/bin/env python3
"""S1.7: log the cartpole swing-up and its sampled futures to rerun.

Reads the CSVs written by examples/cartpole_replay and produces an .rrd
recording: the executed cart/pole geometry plus the ghost trails MPPI
considered at each control step, alongside control and cost time series.

Usage:
    python3 scripts/rerun_cartpole.py <data_dir> [out.rrd]
    rerun docs/media/cartpole_swingup.rrd     # to view

Requires rerun-sdk (see requirements.txt).
"""

import csv
import sys
from collections import defaultdict

import numpy as np
import rerun as rr

POLE_LENGTH = 0.5  # pivot to point mass, matches dynamics/cartpole.hpp
CART_HALF_W = 0.12
CART_HALF_H = 0.05


def tip(x, theta):
    """Pole tip position; theta = 0 is upright."""
    return x + POLE_LENGTH * np.sin(theta), POLE_LENGTH * np.cos(theta)


def load(data_dir):
    traj = []
    with open(f"{data_dir}/trajectory.csv", newline="") as f:
        for row in csv.DictReader(f):
            traj.append(
                (
                    int(row["frame"]),
                    float(row["t"]),
                    float(row["x"]),
                    float(row["theta"]),
                    float(row["u"]),
                    float(row["beta"]),
                )
            )

    ghosts = defaultdict(lambda: defaultdict(list))
    with open(f"{data_dir}/ghosts.csv", newline="") as f:
        for row in csv.DictReader(f):
            ghosts[int(row["frame"])][int(row["k"])].append(
                (float(row["x"]), float(row["theta"]))
            )
    return traj, ghosts


def main() -> None:
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    out = sys.argv[2] if len(sys.argv) > 2 else "docs/media/cartpole_swingup.rrd"

    traj, ghosts = load(data_dir)

    rr.init("cartpole_swingup")
    # Static: the rail the cart slides along, for visual reference.
    rr.log("world/rail", rr.LineStrips2D([[[-1.5, 0.0], [1.5, 0.0]]], colors=[120, 120, 120]),
           static=True)

    for frame, t, x, theta, u, beta in traj:
        rr.set_time("frame", sequence=frame)
        rr.set_time("sim_time", duration=t)

        tx, ty = tip(x, theta)
        rr.log("world/pole", rr.LineStrips2D([[[x, 0.0], [tx, ty]]], colors=[230, 80, 60],
                                             radii=0.012))
        rr.log("world/tip", rr.Points2D([[tx, ty]], colors=[230, 80, 60], radii=0.03))
        rr.log(
            "world/cart",
            rr.Boxes2D(centers=[[x, 0.0]], half_sizes=[[CART_HALF_W, CART_HALF_H]],
                       colors=[60, 110, 200]),
        )

        # Ghost trails: the futures this control step actually evaluated,
        # drawn as the path each sampled rollout's pole tip would trace.
        strips = []
        for _, pts in sorted(ghosts.get(frame, {}).items()):
            strips.append([list(tip(gx, gth)) for gx, gth in pts])
        if strips:
            rr.log("world/ghosts", rr.LineStrips2D(strips, colors=[90, 160, 230, 60], radii=0.003))

        rr.log("control/u", rr.Scalars(u))
        rr.log("cost/beta", rr.Scalars(beta))

    rr.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
