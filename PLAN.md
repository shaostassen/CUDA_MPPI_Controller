# cuda_mppi_controller — GPU-Accelerated MPPI Controller: Plan & Backlog

**Owner:** Shao · **Window:** ~9 weeks @ 5–8 h/wk (~60–70 h total) · **Hardware:** Jetson Orin Nano (sm_87, JetPack ≥ 6.2) — sole CUDA machine; Mac edits over SSH, no local CUDA · **Stretch target:** desktop-GPU comparison column if lab hardware materializes

**What this project proves:** raw CUDA kernel design + profiling fluency on real embedded hardware, sitting directly on existing control-theory and parallel-computing strengths. Headline artifact: a quadrotor flying a figure-8 under closed-loop MPPI at 100 Hz on a ~15 W board, evaluating 10,000+ sampled futures per control step, backed by an honest speedup table against CPU, OpenMP, and PyTorch (including `torch.compile`) baselines measured on the same SoC — a perf-per-watt story a desktop build can't tell.

---

## 1. Pinned decisions & assumptions

Stated up front so every story below inherits them. Change any of these and the backlog still holds — only parameters shift.

1. **OS / dev flow:** JetPack ≥ 6.2 (L4T, Ubuntu 22.04-based) on the Orin Nano; MAXN/Super power mode + fan profile set once and recorded. The Mac cannot build CUDA — edit locally (VS Code Remote-SSH or rsync-on-save), build/test/profile on the Orin. Phase A's CPU reference also builds natively on the Mac for fast iteration, but every recorded number comes from the Orin.
2. **Toolchain:** JetPack's CUDA (12.6 on JetPack 6.2 — use the newest Orin-supporting JetPack), L4T's GCC, C++20, CMake ≥ 3.28 with presets (pip-install cmake on the Orin if apt's is older). Build targets `sm_87` real + `compute_87` PTX; a desktop arch gets added only if the stretch column happens.
3. **Precision:** FP32 throughout the controller. FP64 appears only in oracle tests for integrator correctness.
4. **RNG:** custom header-only **Philox4x32-10**, marked `__host__ __device__`, counter = (seed, iteration, k, t). No cuRAND dependency. Rationale: exact bit-level CPU↔GPU noise parity (makes parity tests strict instead of statistical) and zero library friction on Jetson.
5. **Integrators:** explicit Euler, dt = 0.02 s inside rollouts (standard MPPI practice, cheap); RK4 ground-truth sim environment. Phase A uses the same model/integrator on both sides for clean parity; deliberate model mismatch is introduced in Phase C as a robustness demonstration.
6. **MPPI flavor:** information-theoretic MPPI (Williams et al.): trajectory cost `S_k`, baseline `β = min_k S_k`, weights `w_k ∝ exp(−(S_k − β)/λ)`, update `u_t += Σ_k w_k ε_{k,t}`, with the control-cost coupling term `γ uᵀ Σ⁻¹ ε`, `γ = λ(1−α)`. Optional Savitzky–Golay smoothing of the updated sequence (ablated in Phase C, off by default).
7. **Kernel v1 design:** 1 thread = 1 rollout, fused rollout + cost accumulation in one kernel, noise read from global memory. Simple, correct, profilable. Cleverness comes later, measured.
8. **Reductions v1:** min via CUB `DeviceReduce`, softmax-normalize in a small kernel, weighted noise sum via cuBLAS `Sgemv` (noise viewed as a K × (T·m) matrix). v2 replaces these with hand-rolled warp-shuffle reductions behind a flag; whichever measures faster becomes default. Both paths stay in-repo — the comparison is writeup content.
9. **Noise layout:** `[T][K][m]` so that at fixed t, adjacent threads (k) read adjacent memory — coalesced for m = 1–4.
10. **Baselines:** CPU single-thread and CPU OpenMP on the Orin's 6× Cortex-A78AE (same-SoC comparison — the honest one), PyTorch eager, `torch.compile(mode="reduce-overhead")` (CUDA graphs — the *fair* GPU baseline), and the UM-ARM-Lab `pytorch_mppi` library at a pinned commit. PyTorch on Jetson means NVIDIA's aarch64 JetPack wheels, not PyPI — pin the wheel in week 1 and smoke-test `torch.compile` immediately (see §8).
11. **Phase C dynamics: quadrotor** (over off-road vehicle). 13-state (pos, vel, quaternion, body rates), 4 rotor-thrust inputs, standard rigid-body + thrust mixing. Reasons: crisper success metrics, underactuated attitude dynamics stress-test the sampler, and it reads better for aerial/legged autonomy audiences.
12. **Visualization:** rerun for 3D scenes and sampled-trajectory "ghost trails" — log `.rrd` on the Orin, view on the Mac; matplotlib for benchmark plots. The ghost-trail figure-8 render is the hero GIF.
13. **Repo hygiene:** MIT license, results committed as CSV + generated plots, every benchmark claim reproducible by one script.

**Algorithm reference (for the repo README):** per control step — sample K noise sequences ε ~ N(0, Σ) over horizon T; roll out `x_{t+1} = f(x_t, u_t + ε_t)`; accumulate running + terminal + control-coupling costs into S_k; softmax-weight with baseline subtraction; update u; execute u_0; shift.

---

## 2. Milestones & schedule

| Milestone | Target week | Exit criteria |
|---|---|---|
| **M1 — CPU cartpole swing-up** | wk 2 | Video of swing-up, all Phase A tests green, LQR cross-check documented |
| **M2 — GPU parity** | wk 4 | GPU iteration matches CPU ≤ 1e-4 rel over 100 seeded iters; side-by-side timing demo |
| **M3 — Benchmarks + profiling log** | wk 6 | Tables 1 & 3 filled, scaling plot, speedup targets evaluated (§4), optimization log written |
| **M4 — Quadrotor closed loop** | wk 9 | Figure-8 @ 100 Hz with K = 10k, ablation table filled, writeup published |
| Stretch (post-window) | — | Desktop-GPU column in Table 1, Nav2 plugin smoke test, persistent-kernel experiment |

Weekly mapping: wk 1 E0+E1 start · wk 2 finish E1 → M1 · wk 3–4 E2 → M2 · wk 5–6 E3 → M3 · wk 7–8 E4 · wk 9 buffer + E6 → M4. Ablations (S4.4) are the designated slip absorber — cut settings, not the milestone.

---

## 3. Epic / story backlog

Stories sized ~2–5 h. AC = acceptance criteria; a story is done only when its AC are demonstrably true (test, committed artifact, or logged measurement).

### E0 — Bootstrap (~5 h)

- **S0.1 Toolchain verification.** AC: JetPack updated (≥ 6.2), power mode + fan set and recorded; deviceQuery reports sm_87; `ncu` and `nsys` successfully profile a sample kernel on the Orin (ncu wants sudo for counters on Tegra); Mac→Orin Remote-SSH flow working; JetPack/CUDA/driver versions logged in `docs/env.md`.
- **S0.2 Repo scaffold.** CMake presets (Debug / Release / Release-Profile with `-lineinfo`), GoogleTest wired, clang-format, `.gitignore`, MIT license, README stub, arch `sm_87` real + `compute_87` PTX. AC: `cmake --preset release && ctest --preset release` green on the Orin on a trivial kernel test; `--ptxas-options=-v` output captured in build log.
- **S0.3 CI.** GitHub Actions: CPU-only build + CPU tests + format check (no free GPU runners). GPU tests gated behind `-DENABLE_CUDA=ON`, run locally via `scripts/gpu-test.sh`. AC: badge green on main.

### E1 — CPU reference, Phase A (~12 h)

- **S1.1 Dynamics library.** Pendulum + cartpole, Euler and RK4 steppers, templated on scalar. AC: unforced pendulum energy drift < 0.1% over 10 s (RK4, dt 0.01, FP64); small-angle period within 1% of analytic.
- **S1.2 Philox RNG header.** Host/device Philox4x32-10, counter-based. AC: known-answer tests against reference vectors; identical sequences host vs device for same counters.
- **S1.3 Cost library + MPPI math.** Quadratic running/terminal costs, control-coupling term, baseline-subtracted softmax weights. AC: unit test reproduces a hand-computed 3-sample weight example exactly.
- **S1.4 MPPI core (CPU, single-thread).** AC: cartpole swing-up from hanging start succeeds in ≥ 95/100 seeded runs (|θ| < 0.1 rad and |x| < 0.5 m within 5 s, held 1 s) at K = 512, T = 50; runs ≥ 100 Hz single-thread.
- **S1.5 LQR cross-check.** Near upright equilibrium with quadratic costs, MPPI mean control ≈ LQR gain response. AC: comparison plot committed with tolerance stated; deviations explained.
- **S1.6 OpenMP path.** Parallel rollouts + reduction. AC: ≥ 4× vs single-thread at K = 16k, T = 50 on the Orin's 6 A78AE cores; control update matches 1T within 1e-4 rel (summation-order tolerance); `OMP_NUM_THREADS` and pinning documented.
- **S1.7 Viz.** rerun logging of cartpole state + sampled-trajectory subsample. AC: 10-second swing-up clip exported — this is the M1 video.

### E2 — CUDA port, Phase B core (~14 h)

- **S2.1 Device data model.** Device-resident control sequence, noise, costs, weights; open-loop benchmark does zero per-iteration H2D/D2H, closed loop copies only x₀ (or keeps the sim env on-device). Orin's DRAM is physically shared CPU/GPU — use pinned zero-copy or managed memory for the x₀/u₀ handoff instead of memcpy. AC: nsys timeline shows no memcpy inside the steady-state loop (open-loop config).
- **S2.2 Fused rollout+cost kernel v1.** 1 thread/rollout, noise from global `[T][K][m]`, costs to global. AC: per-rollout costs match CPU ≤ 1e-4 rel at K ∈ {1k, 16k, 100k}; zero register spills (`ptxas -v`).
- **S2.3 Reduction path v1 (libraries).** CUB min → normalize kernel → cuBLAS Sgemv weighted noise sum → control update kernel. AC: full GPU iteration matches CPU ≤ 1e-4 rel over 100 seeded iterations; cartpole swing-up succeeds when driven by the GPU loop.
- **S2.4 Reduction path v2 (hand-rolled).** Warp-shuffle (`__shfl_down_sync`) block min + grid pass; custom weighted-sum kernel. AC: parity holds; micro-benchmark of v1 vs v2 committed; faster path made default with the number cited.
- **S2.5 Noise regeneration variant.** Regenerate ε in the update kernel from Philox counters instead of storing it. AC: bit-identical results to stored-noise path; memory saving (K·T·m floats) and perf delta recorded in Table 3.

### E3 — Benchmarks & profiling, Phase B perf (~12 h)

- **S3.1 Benchmark harness.** JSON configs, warmup 50 / measure 500 iterations, mean/p50/p99, CSV output, log-log plot script, clock locking (`jetson_clocks`, nvpmodel mode recorded per run) and tegrastats board-power capture over the measure window. AC: one command regenerates Table 1's CSV — including the watts column — from scratch.
- **S3.2 PyTorch baseline (own implementation).** Same algorithm + dynamics in torch; measured eager **and** `torch.compile(mode="reduce-overhead")` if it runs on JetPack (§8 risk). AC: task-success parity on cartpole; eager column mandatory, compile column filled or explicitly N/A-with-reason; JetPack wheel version pinned.
- **S3.3 `pytorch_mppi` library baseline.** Pinned commit, config-matched. AC: numbers recorded; any config deviations documented.
- **S3.4 Nsight pass + top-3 optimizations.** Candidates: coalescing/layout, `__launch_bounds__` / register tuning, occupancy-vs-ILP. AC: each optimization is one commit with before/after ncu metrics appended to Table 3.
- **S3.5 CUDA Graphs.** Capture the full iteration. AC: nsys shows one graph launch per iteration; small-K latency improvement quantified.
- **S3.6 Target evaluation.** AC: written verdict against §4 targets; if any target is missed, a one-paragraph analysis of *why* (this paragraph is writeup gold either way).

### E4 — Quadrotor + closed loop, Phase C (~14 h)

- **S4.1 Quadrotor model.** 13-state quaternion dynamics, rotor mixing, host/device. AC: unit tests for quaternion integration + renorm and mixer; open-loop hover holds ± 5 cm for 10 s with analytically set hover thrust.
- **S4.2 Tasks + eval harness.** (a) hover recovery from 30° tilt, (b) waypoint, (c) figure-8 (lemniscate, A = 1 m, 8 s period). AC: success definitions encoded and seeded; RK4 ground-truth env with parameter perturbation (mass ± 5%) — the deliberate model mismatch.
- **S4.3 Real-time closed-loop runner.** Fixed-rate loop at 50/100 Hz, jitter statistics. AC: 100 Hz sustained for 60 s with K = 10k, T = 50; p99 iteration < 8 ms; log committed.
- **S4.4 Ablations.** λ sweep, noise σ sweep, white vs colored (OU / low-pass) noise, horizon {30, 50, 100}. Metrics: success %, mean cost, control smoothness (mean |Δu|). AC: Table 4 filled, 3 plots, 5-sentence findings paragraph.
- **S4.5 SG-smoothing ablation.** On/off row in Table 4. AC: included with smoothness metric.
- **S4.6 Hero visualization.** Figure-8 with ~200 sampled-trajectory ghost trails rendered in rerun. AC: 15–20 s GIF exported — the M4 headline asset.

### E5 — Stretch (post-window)

- **S5.1 Desktop-GPU column.** Borrowed/lab machine (cluster node or a friend's card): add its arch, run the same harness unchanged. AC: desktop column in Table 1 + a short note on which Orin optimizations transferred.
- **S5.2 Nav2 plugin.** 2D diff-drive MPPI variant wrapped as `nav2_core::Controller`. AC: loads and tracks in a nav2 sim smoke test. (Full AC lives in the Project-3 plan — this is the integration hook.)
- **S5.3 Persistent kernel.** Single-launch iteration via cooperative groups grid sync. AC: perf delta vs CUDA Graphs recorded; writeup section drafted.

### E6 — Writeup & assets (~7 h, threaded through)

- **S6.1 README.** Hero GIF at top, 3-command quickstart, results tables. AC: fresh-clone build verified against the quickstart.
- **S6.2 Long-form writeup** per §7, published on shaostassen.com. AC: draft → published → linked from README.
- **S6.3 Videos.** M1, M2, M4 clips with one-line captions. AC: embedded in writeup.

---

## 4. Benchmark table designs (fill in as you go)

**Protocol (applies to all):** `jetson_clocks` + fixed nvpmodel mode (recorded), fixed seeds, warmup 50 / measure 500, report mean and p99, tegrastats board power sampled over the measure window, JetPack/CUDA/torch versions logged alongside every CSV. All timings are one full MPPI iteration (sample → rollout → reduce → update), state resident where the backend allows. Thermals: fan on, temps logged, repeat-run variance checked before trusting any number.

**Speedup targets (S3.6):** ≥ 50× vs CPU single-thread (one A78AE core) and ≥ 5× vs PyTorch eager at K = 16k, T = 50; vs `torch.compile` the goal is parity-or-better with a written analysis — beating eager PyTorch is table stakes, being honest about compiled-with-graphs is what makes the writeup credible. Also report Hz/W for the closed-loop configs: on a 15–25 W board the efficiency number is the headline, not a footnote.

### Table 1 — Iteration latency (mean ms / p99 ms)

| K | T | CPU 1T | CPU OMP-6 | Torch eager | Torch compile | CUDA v1 | CUDA final | × vs CPU-1T | × vs compile |
|---|---|---|---|---|---|---|---|---|---|
| 1k | 50 | | | | | | | | |
| 4k | 50 | | | | | | | | |
| 16k | 50 | | | | | | | | |
| 64k | 50 | | | | | | | | |
| 100k | 50 | | | | | | | | |
| 16k | 100 | | | | | | | | |
| 64k | 100 | | | | | | | | |

### Table 2 — Max sustainable closed-loop rate (Hz)

| Task (config) | CPU OMP | Torch compile | CUDA final | Board power (W) | Hz/W |
|---|---|---|---|---|---|
| Cartpole (K=4k, T=50) | | | | | |
| Quadrotor (K=10k, T=50) | | | | | |

### Table 3 — Kernel evolution log (K=16k, T=50)

| Ver | Change | Iter μs | Achieved occ % | DRAM % | SM % | Notes |
|---|---|---|---|---|---|---|
| v1 | baseline fused rollout | | | | | |
| | | | | | | |

### Table 4 — Ablations (quadrotor figure-8, K=10k, T=50)

| Setting | Success % | Mean cost | Mean \|Δu\| | Notes |
|---|---|---|---|---|
| λ = … | | | | |
| σ = … | | | | |
| white vs OU noise | | | | |
| T = 30 / 50 / 100 | | | | |
| SG filter on/off | | | | |

---

## 5. Nsight profiling checklist

Run in this order; one optimization per commit; every pass appends a Table 3 row.

1. **Prep:** build the Release-Profile preset (`-lineinfo`), `jetson_clocks`, run headless (no desktop compositor on the GPU), confirm zero spills via `--ptxas-options=-v`; run `ncu` under sudo on Tegra.
2. **System first — nsys:** `nsys profile --stats=true -o iter ./bench --config bench/16k_50.json`. Check: kernels per iteration, gaps between kernels (launch overhead), any memcpy inside the loop, wall time per iteration vs sum of kernel times. Big gap ⇒ CUDA Graphs (S3.5) is your first win.
3. **Kernel — ncu:** `ncu --set full --import-source yes -k regex:rollout -o rollout_v1 ./bench …`. Read in order:
   - *GPU Speed of Light:* SM% vs DRAM% — which resource binds? Rollout with tiny state should be compute/latency-bound; high DRAM% ⇒ noise-read layout problem.
   - *Occupancy:* theoretical vs achieved; if register-limited, try `__launch_bounds__(256)` and compare — but prefer ILP (unroll inner integration steps) over forcing occupancy.
   - *Warp State:* Long Scoreboard ⇒ memory latency (fix coalescing, consider noise regeneration S2.5); Wait ⇒ dependency chains (unroll, restructure).
   - *Memory Workload:* sectors/request near minimal for the ε reads; L2 hit rate.
   - *Source page:* hottest lines; branch divergence at input clamps/constraint checks.
4. **Re-measure** with the harness (never trust profiler-mode timings for the table), append Table 3, commit.
5. **After Graphs:** confirm in nsys that steady state is one graph launch per iteration.

---

## 6. Repo structure

```
cuda_mppi_controller/
├── CMakeLists.txt / CMakePresets.json
├── include/mppi/
│   ├── controller.hpp          # templated MPPI core (host)
│   ├── philox.hpp              # host/device RNG
│   ├── dynamics/               # pendulum, cartpole, quadrotor (host/device)
│   └── costs/
├── src/
│   ├── cpu/                    # reference + OpenMP paths
│   └── cuda/                   # rollout.cu, reduce.cu, update.cu, graphs.cu
├── baselines/
│   ├── torch_mppi.py           # own torch impl (eager + compile)
│   └── pytorch_mppi_bench.py   # library baseline, pinned
├── bench/                      # harness, JSON configs, results/*.csv, plots/
├── tests/                      # unit + CPU↔GPU parity
├── examples/                   # cartpole_demo, quadrotor_fig8
├── scripts/                    # gpu-test.sh, profile.sh, plot.py
└── docs/                       # env.md, optimization-log.md, writeup/
```

---

## 7. Writeup outline — the speedup narrative

Working title: **"10,000 futures, 100 times a second, on 15 watts: MPPI in raw CUDA on Jetson."**

1. **Hook.** MPC by brute-force simulation; why MPPI is the natively GPU-shaped controller, and why I wrote the kernels by hand instead of using PyTorch.
2. **MPPI in five minutes.** The update equation, one diagram, no derivation.
3. **Mapping the algorithm to the GPU.** Thread-per-rollout, memory layout, counter-based RNG (and the parity-testing trick it unlocks).
4. **Making it fast.** The profiling narrative: launch-overhead → Graphs; coalescing; store-vs-regenerate noise; library vs hand-rolled reductions. Each with before/after numbers from Table 3.
5. **Results.** Tables 1–2, log-log scaling plot. Include the honest section: how close `torch.compile` gets, and where raw CUDA still wins (small-K latency, end-to-end loop rate, memory).
6. **Closing the loop.** Quadrotor figure-8 at 100 Hz under model mismatch; ablation findings; the ghost-trail GIF.
7. **Scaling up.** What a desktop GPU would add (if the stretch column lands); which optimizations were Orin-specific vs universal.
8. **Takeaways + repo link.** Two paragraphs max.

---

## 8. Risks & watch-items

- **`torch.compile` on JetPack:** Inductor + CUDA-graphs support in the aarch64 Jetson wheels is the flakiest link — smoke-test in week 1; if broken, the compile column becomes N/A-with-reason and eager stays mandatory.
- **`torch.compile` is a strong baseline:** if the raw-CUDA margin is thin at large K, the narrative pivots to small-K latency and closed-loop Hz (where fusion + graphs dominate). Decided in S3.6, not discovered in the writeup.
- **Quadrotor cost tuning is a time sink:** timebox 4 h; M4 falls back to hover-recovery + waypoint if figure-8 tuning drags, with figure-8 finishing inside the stretch window.
- **Remote-only GPU loop:** every build/test cycle crosses SSH — set up Remote-SSH/rsync properly in S0.1 or the iteration tax eats the schedule. Watch thermals and the 8 GB shared CPU/GPU memory during torch baselines and K = 100k configs.
- **RNG correctness:** Philox known-answer tests in S1.2 before anything depends on it.

**Definition of done:** M1–M4 shipped; README with hero GIF and 3-command quickstart; writeup live on shaostassen.com; every number in Tables 1–4 regenerable by a committed script.

---

## 9. Session 1 — next 2–3 hours

1. Update JetPack (≥ 6.2), set MAXN/Super power mode + fan, run deviceQuery; log sm_87 + JetPack/CUDA/driver versions in `docs/env.md`; confirm `ncu`/`nsys` can profile a toy kernel.
2. Wire up the Mac→Orin dev flow: SSH keys, VS Code Remote-SSH (or rsync-on-save); pip-install cmake ≥ 3.28 on the Orin if apt's is old.
3. Push the §6 scaffold with CMake presets, GTest, and one trivial sm_87 kernel test; `ctest` green on the Orin. (S0.1–S0.2 done.)
4. Start S1.1: pendulum dynamics + the energy-drift test (builds natively on the Mac too — CPU only).
5. Optional: have Claude Code open GitHub issues from the E0–E1 stories.
