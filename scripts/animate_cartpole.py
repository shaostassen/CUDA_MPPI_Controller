#!/usr/bin/env python3
"""S1.7: export the swing-up clip as a GIF.

The rerun recording (scripts/rerun_cartpole.py) is the interactive artifact;
this produces the directly-viewable clip for the README, needing no rerun
install to watch.

Usage:
    python3 scripts/animate_cartpole.py <data_dir> [out.gif]
"""

import csv
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

POLE_LENGTH = 0.5
FRAME_STRIDE = 2  # 500 sim steps at 50 Hz -> 250 GIF frames at 25 fps


def tip(x, theta):
    return x + POLE_LENGTH * np.sin(theta), POLE_LENGTH * np.cos(theta)


def main() -> None:
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    out = sys.argv[2] if len(sys.argv) > 2 else "docs/media/cartpole_swingup.gif"

    traj = []
    with open(f"{data_dir}/trajectory.csv", newline="") as f:
        for row in csv.DictReader(f):
            traj.append((int(row["frame"]), float(row["t"]), float(row["x"]),
                         float(row["theta"]), float(row["u"])))

    ghosts = defaultdict(lambda: defaultdict(list))
    with open(f"{data_dir}/ghosts.csv", newline="") as f:
        for row in csv.DictReader(f):
            ghosts[int(row["frame"])][int(row["k"])].append(
                (float(row["x"]), float(row["theta"])))

    frames = traj[::FRAME_STRIDE]

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=100)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.75, 0.85)
    ax.set_aspect("equal")
    ax.axhline(0.0, color="0.75", lw=1.0, zorder=0)
    ax.set_xticks([]), ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    ghost_lines = [ax.plot([], [], color="tab:blue", alpha=0.22, lw=0.9, zorder=1)[0]
                   for _ in range(16)]
    pole_line, = ax.plot([], [], color="tab:red", lw=3.0, solid_capstyle="round", zorder=3)
    tip_dot, = ax.plot([], [], "o", color="tab:red", ms=7, zorder=4)
    cart = Rectangle((0, 0), 0.24, 0.1, facecolor="tab:blue", zorder=2)
    ax.add_patch(cart)
    label = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=9, va="top", family="monospace")

    def update(i):
        frame, t, x, theta, u = frames[i]
        tx, ty = tip(x, theta)
        pole_line.set_data([x, tx], [0.0, ty])
        tip_dot.set_data([tx], [ty])
        cart.set_xy((x - 0.12, -0.05))

        strips = sorted(ghosts.get(frame, {}).items())
        for j, line in enumerate(ghost_lines):
            if j < len(strips):
                pts = strips[j][1]
                gx, gy = zip(*[tip(px, pth) for px, pth in pts])
                line.set_data(gx, gy)
            else:
                line.set_data([], [])

        label.set_text(f"t = {t:5.2f} s\nu = {u:+6.2f} N\nθ = {theta:+6.3f} rad")
        return [*ghost_lines, pole_line, tip_dot, cart, label]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=40, blit=True)
    ani.save(out, writer=animation.PillowWriter(fps=25))
    print(f"wrote {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
