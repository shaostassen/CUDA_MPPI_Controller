# Visualization (S1.7)

Two artifacts come out of the same run: an interactive rerun recording and a
GIF for anyone who does not want to install rerun.

![cartpole swing-up](media/cartpole_swingup.gif)

## What the ghost trails actually are

The faint blue arcs are the futures MPPI evaluated at that control step —
the pole-tip path of a subsample of the K sampled rollouts. They are not a
decorative re-simulation: `Controller::replay_rollout()` regenerates each
rollout from its Philox counters `(seed, iteration, k, t)` and shares
`sample_control()` with the hot loop, so a replayed trajectory is the one
`step()` costed. `ReplayRollout.ReproducesTheTrajectoryStepCosted` pins this
by recomputing replayed rollout costs and comparing against the controller's
own stored costs — 12/12 matched to the last float bit locally.

Because the noise is regenerated rather than stored, visualization costs the
hot loop nothing, which is what lets the same mechanism scale to S4.6's
~200-trail quadrotor render.

## Python environment

The C++ side has no Python dependency; the scripts do. This machine has
several interpreters on `PATH` and their order is not stable, so use a venv
rather than whichever `python3` answers first:

```
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`.venv/` is gitignored. Run every script as `.venv/bin/python scripts/...`.

## Regenerating

```
./build/cpu/examples/cartpole_replay /tmp/s17
.venv/bin/python scripts/rerun_cartpole.py /tmp/s17 docs/media/cartpole_swingup.rrd
.venv/bin/python scripts/animate_cartpole.py /tmp/s17 docs/media/cartpole_swingup.gif
```

`cartpole_replay` writes two intermediates: `trajectory.csv` (the executed
closed loop, one row per control step) and `ghosts.csv` (the sampled
subsample: 10 rollouts per frame, horizon strided by 5). They are build
products, not committed — the `.rrd` and `.gif` are.

View the recording with:

```
.venv/bin/rerun docs/media/cartpole_swingup.rrd
```

It carries two timelines (`frame`, `sim_time`) plus `control/u` and
`cost/beta` time series alongside the 2D scene.
