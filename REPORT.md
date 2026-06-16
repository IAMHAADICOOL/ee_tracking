# Design Notes & Report

This document explains *why* the system is built the way it is, and walks through the
experimental progression honestly — including what didn't work and what that taught us.
For setup and commands, see [README.md](README.md).

## 1. Problem

Control a simulated 7-DOF arm to track a time-varying 3D Cartesian trajectory: accurately,
smoothly, and robustly to at least one source of uncertainty, with RL as a core component.

## 2. Approach and rationale

### 2.1 Velocity-level control, not a position IK solver
Tracking a *moving* target is a differential problem. We use resolved-rate (velocity-level)
control: at each tick, map the desired EE velocity (feedforward target velocity + a
proportional pull toward the current target point) to joint velocities through the Jacobian.
This is the right tool for continuous tracking, and it is what the RL residual plugs into.

### 2.2 A strong model-based base controller
The base is a recursive **task-priority** controller (damped-least-squares inverse, joint
limits as a high-priority inequality task, a null-space posture task). On reachable paths it
tracks to ~0.2–0.4 mm. This matters: a weak baseline would make RL look good for the wrong
reasons. We deliberately set a high bar.

### 2.3 Why residual RL is the "RL core"
Three paradigms were considered:
- **Pure model-based**: superb when the model is correct, degrades systematically when not.
- **End-to-end RL**: must relearn IK from scratch; rarely reaches sub-mm or smooth on 7-DOF;
  no fallback out of distribution. (Supported in the code via `--mode end2end` for contrast.)
- **Residual RL** (chosen): `dq = controller + residual_scale·policy`. Degrades *gracefully*
  to the controller when the residual is unhelpful, and earns its value exactly where the
  model-based controller fails — under uncertainty. This is a legitimate, well-motivated use
  of RL as the core learning component.

### 2.4 Observation design
The policy sees joint state, EE-to-target error, EE velocity, a **trajectory preview** at
several future times, its previous action, and the **controller's nominal command**. The
preview is what lets a learned policy *anticipate* (e.g. lead a delayed system); the nominal
command lets it reason relative to what the controller is already doing. Reward is computed
from the **true** state, never the (optionally noisy) observation — so the policy cannot
"cheat" by exploiting its own sensor noise.

## 3. The uncertainty model (and a key correction)

A first version perturbed only the policy/observation channel. That was a mistake: it made
the policy's job harder without **degrading the base controller**, so there was nothing for
the residual to *compensate*. The fix was to apply perturbations to the **executed command
and state**, so they hit the controller too:

- **Actuator gain** (`vel_gain`): executed velocity = gain × commanded → systematic
  undershoot. Compensable by scaling commands up.
- **Control latency** (`control_delay_steps`): the command is applied several sim-steps late
  → systematic lag. A reactive controller cannot fix this; an anticipatory policy using the
  preview can.
- **Observation / actuation noise**: robustness stressors.

Levels are domain-randomized per episode during training and fixed for evaluation.
Verified that, with zero residual, these drive the base controller from 0.2 mm to several mm
(and ~10 mm at the hardened levels) — i.e. they create a real gap for RL to close.

## 4. Measuring smoothness
"Smooth" is only meaningful if measured. `metrics.py` computes RMS EE jerk, joint jerk,
command jitter, and SPARC. Jerk is the primary read for continuous tracking (it is highly
sensitive to chatter); SPARC is reported as a secondary, standard reference but is less
discriminating for cyclic motion than for discrete reaches.

## 5. Experimental progression (honest)

The result was *not* achieved in one shot. The path mattered:

1. **Baseline + clean residual.** The pipeline learned (error fell, reward rose), confirming
   the env/reward/training loop was correct. Under clean conditions the residual settled a
   few mm *above* the baseline — the first sign of the always-on-residual tension.
2. **Add uncertainty (first attempt).** With the original (loose) reward and condition-blind
   policy, the residual barely beat the baseline under uncertainty and *hurt* clean tracking.
   Reward tuning (tighter `sigma_pos`, larger action/smoothness weights) moved the problem
   around but did not solve it — a sign the issue was structural, not a weight.
3. **Diagnosis.** Two root causes: (a) the policy could not *perceive* the disturbance, so it
   applied one average correction that was wrong when clean; (b) the strong baseline left
   little headroom at mild uncertainty.
4. **Frame stacking + harder uncertainty.** Stacking 4 observations gave the policy the
   temporal signal to infer the disturbance; hardening the uncertainty (gain 0.70, delay 12)
   degraded the baseline to ~10 mm, creating real headroom. Result: under uncertainty the
   residual roughly **halved** the error (≈4 vs ≈9 mm) — the first clear win. But **jerk
   exploded** (the policy chattered to chase precision).
5. **Residual low-pass filtering.** An EMA filter on the residual (in the env, so the policy
   trains within it) cut jerk ~3× with no accuracy loss. Pushing the smoothness penalty hard
   (`w_smooth` up to 1.5) on top of heavier filtering brought jerk to **near-baseline** while
   keeping — even improving — the uncertainty win.

The final model (`res_fs_filt6`) is the endpoint of this progression.

## 6. Results

| Controller | Condition | RMSE (mm) | EE jerk (m/s³) |
|---|---|---|---|
| Task-priority IK | clean | 0.21 | 0.4 |
| Residual RL (final) | clean | 4.37 | 2.4 |
| Task-priority IK | uncertain | 8.63 | 83.0 |
| Residual RL (final) | uncertain | **3.34** | **73.8** |

**Under uncertainty the residual is better on both accuracy and smoothness.** This is the
core claim, and it is exactly where a model-based controller is weakest.

## 7. The honest trade-off

The residual **costs clean-condition precision** (4.4 mm vs 0.2 mm). The cause is structural:
a single always-on residual applies a correction even when the model is already accurate.
Frame stacking gives the policy the *information* to back off when clean, but with the chosen
reward it does not learn to fully *zero* its output — so it perturbs an already-near-perfect
command. We chose an operating point that prioritizes the (more valuable) robustness and
smoothness over clean precision, and report the trade-off plainly rather than hiding it.

There is also a genuine **smoothness–accuracy frontier**: beyond a point, more filtering
buys lower jerk only by giving up tracking accuracy. We selected a point on that frontier
deliberately; it is not a free optimum.

## 8. What I would do next
- **Gate the residual** on a learned disturbance/condition estimate (or an explicit
  confidence signal), so it is suppressed when the model is accurate — directly targeting the
  clean-precision loss.
- **Recurrent policy** for sharper online identification of gain/latency than a 4-frame stack.
- **Held-out trajectory families** (Lissajous, spiral, raster) to measure generalization
  beyond the circle / figure-eight / moving-target training set.
- **Sim-to-real**: the residual-over-controller structure plus domain randomization is a
  promising basis, but no transfer is claimed here.

## 9. Reproducibility notes
- All evaluation uses fixed trajectories and seeds, identical across controllers; the
  "uncertain" condition uses fixed (non-randomized) levels.
- Eval flags **must** match training (`--frame_stack 4 --action_smoothing 0.1` for the final
  model), because the policy was trained within that stacked, filtered regime.
- `compare.py` treats `pure_ik` as the base controller stepped with a zero action, so the
  comparison is exact: same controller, same env, same disturbance — only the residual differs.
