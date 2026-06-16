# Residual RL for Robust End-Effector Trajectory Tracking

A 7-DOF Franka Panda (PyBullet) tracks time-varying 3D Cartesian trajectories using
**residual reinforcement learning** layered on top of a **task-priority resolved-rate
inverse-kinematics controller**. The model-based controller does the bulk of the work;
a SAC policy adds a small, bounded correction. The headline result: under realistic
**actuator-gain and control-latency uncertainty**, the residual policy roughly **halves**
the steady-state tracking error of the (already strong) model-based baseline *and* moves
more smoothly — while the controller alone degrades badly.

---

## Headline result

Steady-state tracking error and motion smoothness (EE jerk), averaged over three
trajectory families (circle, figure-eight, moving-target), evaluated on identical
trajectories and seeds:

| Controller | Condition | RMSE (mm) | EE jerk (m/s³) |
|---|---|---|---|
| Task-priority IK (baseline) | clean | **0.21** | **0.4** |
| Residual RL (final) | clean | 4.37 | 2.4 |
| Task-priority IK (baseline) | uncertain | 8.63 | 83.0 |
| **Residual RL (final)** | **uncertain** | **3.34** | **73.8** |

- **Under uncertainty, the residual wins on both axes:** 8.63 → 3.34 mm error (≈61 % lower)
  *and* lower jerk than the baseline.
- **When clean, the baseline is better** (0.21 vs 4.37 mm). This is the honest trade-off —
  see [Limitations](#limitations-and-the-honest-trade-off).

The "uncertain" condition is a fixed actuator gain of 0.70 (30 % velocity undershoot) plus
12 sim-steps of control latency plus sensor/actuation noise — perturbations a reactive
controller cannot fully reject but a learned, anticipatory residual can.

---

## Why residual RL (the core idea)

A pure model-based controller is excellent when its model is right and degrades
systematically when it isn't (an unmodelled actuator gain makes it undershoot; latency
makes it lag). A pure end-to-end RL policy must relearn inverse kinematics from scratch and
has no safety net out of distribution.

**Residual RL keeps the best of both.** The control command is

```
dq = dq_controller(target)  +  residual_scale * policy(observation)
```

- When the model is accurate, the optimal residual is ≈ 0 and behaviour falls back to the
  strong controller.
- When the model is wrong, the policy learns a correction the controller structurally
  cannot produce (e.g. scale commands up to counter gain loss, lead the target to counter
  latency).

Two design choices make this work in practice:

- **Frame-stacked observations (4 frames).** The policy can only compensate for a
  disturbance it can perceive. Stacking recent observations lets it *infer* the condition —
  commanded-vs-achieved velocity reveals the gain, lag patterns reveal the delay — instead
  of applying one blind average correction.
- **Low-pass-filtered residual.** SAC policies tend to output twitchy actions. An
  exponential-moving-average filter on the residual (inside the env, so the policy trains
  with it) removes high-frequency chatter and brings jerk down to near-baseline levels.

---

## Method

**Base controller** (`src/arm_tracking/tasks.py`): recursive task-priority resolved-rate
control with damped-least-squares inversion, hysteresis-gated joint-limit avoidance, and a
null-space posture task. Tracks reachable Cartesian paths to sub-millimetre accuracy.

**RL** (`src/arm_tracking/env.py`, `scripts/train.py`): Soft Actor-Critic (Stable-Baselines3)
in a Gymnasium env.
- *Observation* (46 base dims × 4 stacked): joint angles/velocities, EE-to-target error,
  EE velocity, a short trajectory preview, previous action, and the controller's nominal
  command.
- *Action*: a bounded residual joint-velocity, added to the controller command.
- *Reward*: `w_track·exp(−err/σ) − w_act·‖a‖² − w_smooth·‖a−a_prev‖²`, computed from the
  **true** state (never the noisy observation) to prevent sensor-gaming. A divergence
  penalty equal to the forfeited future reward removes the "suicidal-termination" exploit.

**Uncertainty model** (Phase 4): actuator velocity gain, control-latency buffer, observation
noise, and actuation noise — all applied to the *executed* command so they degrade the base
controller (not just the policy). Domain-randomized per episode during training; fixed for
evaluation.

**Smoothness metrics** (`src/arm_tracking/metrics.py`): RMS EE/joint jerk, command jitter,
and spectral arc length (SPARC), so smoothness is *measured*, not asserted.

See [REPORT.md](REPORT.md) for the full design rationale and the experimental progression.

---

## Repository structure

```
ee-tracking/
├── environment.yml
├── README.md
├── REPORT.md
├── src/arm_tracking/
│   ├── sim.py            # PyBullet Panda wrapper (velocity control, markers, video)
│   ├── trajectories.py   # circle, figure-eight, moving-target generators
│   ├── tasks.py          # task-priority resolved-rate controller + tasks
│   ├── ik_controller.py  # minimal single-objective DLS controller (reference)
│   ├── env.py            # Gymnasium env: residual action, frame stack, uncertainty, reward
│   └── metrics.py        # tracking + smoothness metrics
├── scripts/
│   ├── run_baseline.py   # controller-only tracking (+ GUI/video)
│   ├── train.py          # SAC training (curriculum, uncertainty, frame stack, filtering)
│   ├── watch_policy.py   # roll out a trained policy; save plot/video/CSV
│   ├── compare.py        # pure-IK vs residual vs end-to-end across clean/uncertain
│   ├── test_pose_targets.py
│   └── check_env.py
└── outputs/              # generated plots, videos, tables
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate ee-tracking
# GPU PyTorch matching your CUDA toolkit (example: CUDA 12.x):
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install stable-baselines3 tensorboard gymnasium imageio imageio-ffmpeg
```

Run everything from the repo root as modules (`python -m scripts.<name>`).

---

## Usage

**Baseline (controller only):**
```bash
python -m scripts.run_baseline --traj figure_eight --controller taskpriority --gui --video
```

**Train the final residual policy (curriculum: local → wide → uncertain).** Each stage
continues from the previous; the final stage adds the hardened uncertainty, frame stacking,
residual filtering, and a strong smoothness penalty:
```bash
# Stage 1 — local, clean
python -m scripts.train --mode residual --episode_seconds 16 --n_envs 8 \
    --frame_stack 4 --sigma_pos 0.008 --w_act 0.02 --w_smooth 0.1 \
    --timesteps 200000 --tag res_fs_local

# Stage 2 — wide workspace (some targets intentionally unreachable)
python -m scripts.train --mode residual --episode_seconds 16 --n_envs 8 \
    --frame_stack 4 --sigma_pos 0.008 --w_act 0.02 --w_smooth 0.1 \
    --wide_workspace --unreachable_frac 0.4 \
    --load models/res_fs_local.zip --timesteps 200000 --tag res_fs_wide

# Stage 3 — hardened uncertainty + residual filter + strong smoothness  (final model)
python -m scripts.train --mode residual --episode_seconds 16 --n_envs 8 \
    --frame_stack 4 --action_smoothing 0.1 \
    --sigma_pos 0.008 --w_act 0.02 --w_smooth 1.5 \
    --wide_workspace --unreachable_frac 0.4 --uncertainty \
    --load models/res_fs_wide.zip --timesteps 400000 --tag res_fs_filt6
```

Monitor with `tensorboard --logdir runs` (watch `rollout/mean_pos_error_mm`).

**Watch / record the final policy** (flags must match training: `--frame_stack 4 --action_smoothing 0.1`):
```bash
python -m scripts.watch_policy --model models/res_fs_filt6.zip --mode residual \
    --frame_stack 4 --action_smoothing 0.1 --traj figure_eight --uncertainty --gui --video --csv
```

**Reproduce the comparison table and figures:**
```bash
python -m scripts.compare --residual models/res_fs_filt6.zip \
    --frame_stack 4 --action_smoothing 0.1 \
    --trajs circle figure_eight moving_target --seeds 3 --outdir outputs/filt6_final
```

---

## Results and figures

`compare.py` writes to the output directory:
- `compare_table.csv` — every metric, every controller × condition.
- `compare_rmse.png` — grouped bar chart of steady-state RMSE (clean vs uncertain).
- `compare_error_<traj>.png` — error-vs-time, pure-IK vs residual under uncertainty.

`watch_policy.py` writes a per-policy error-vs-time plot, an optional MP4, and an optional
per-step CSV, named `policy_<model>_<traj>_<clean|unc>.*`.

---

## Limitations and the honest trade-off

This is a deliberately honest result, not a "wins everywhere" claim.

- **The residual costs clean-condition precision.** A single, always-on residual policy
  applies a learned correction even when the model is already accurate, perturbing an
  otherwise near-perfect command (4.4 mm vs the controller's 0.2 mm when clean). Frame
  stacking gives the policy the information to back off, but it does not learn to fully
  zero its output. This is the central tension of an always-on residual.
- **Smoothness vs. responsiveness is a frontier, not a free win.** Filtering and a strong
  smoothness penalty bring jerk to near-baseline, but heavier smoothing eventually trades
  away tracking accuracy. The final model sits at a deliberately chosen operating point.
- **Simulation only.** No sim-to-real transfer is claimed; the residual-over-controller
  design plus domain randomization is what *would* help close that gap.

### Future work
- **Gate the residual** on a learned disturbance estimate, so it is suppressed when the
  model is accurate (recovering clean precision) and active only when needed.
- **Recurrent / longer-history policy** for sharper online system identification.
- **Held-out trajectory shapes** (Lissajous, spiral, raster) to quantify true
  generalization beyond the training families.

---

## License

MIT License — Copyright (c) 2026 Mohammad Haadi Akhter
