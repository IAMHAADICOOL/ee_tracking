"""Phase 6 — controller comparison harness.

Runs each controller on the SAME trajectories and seeds, under CLEAN and a
FIXED uncertainty condition, and reports tracking accuracy + smoothness so the
results are apples-to-apples.

Controllers
-----------
pure_ik   : the task-priority resolved-rate controller alone (residual env
            stepped with a zero action) — the model-based baseline.
residual  : residual-RL policy on top of that controller (--residual model).
end2end   : end-to-end RL policy, no controller underneath (--end2end model,
            optional).

Conditions
----------
clean     : no perturbations.
uncertain : fixed (non-randomized) vel_gain + control delay + obs/action noise,
            so every controller faces the identical degradation.

Outputs (in --outdir)
---------------------
compare_table.csv            : every metric, every controller x condition.
compare_rmse.png             : grouped bar chart of steady-state RMSE.
compare_error_<traj>.png     : error-vs-time, pure_ik vs residual, uncertain.

Example
-------
  python -m scripts.compare --residual models/residual_final.zip \\
      --trajs circle figure_eight moving_target --seeds 3
  # include end-to-end and a held-out shape:
  python -m scripts.compare --residual models/residual_final.zip \\
      --end2end models/end2end.zip --trajs figure_eight lissajous --seeds 3
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from src.arm_tracking.env import TrajectoryTrackingEnv
from src.arm_tracking import metrics

# Default fixed uncertainty condition for the comparison. Must match what the
# policy was TRAINED under; overridable from the CLI. NOT randomized so the
# comparison is repeatable and identical for every controller.
UNC_DEFAULTS = dict(obs_noise_std=0.005, action_noise_std=0.02,
                    vel_gain=0.70, control_delay_steps=12)


def _make_env(mode, traj, uncertain, episode_seconds, seed, frame_stack, unc,
              action_smoothing=1.0):
    kw = dict(base_controller=(mode != "end2end"), randomize=False,
              traj_types=(traj,), episode_seconds=episode_seconds,
              frame_stack=frame_stack, action_smoothing=action_smoothing, seed=seed)
    if uncertain:
        kw.update(unc)
    return TrajectoryTrackingEnv(**kw)


def _rollout(env, policy_fn):
    obs, _ = env.reset()
    errs, pos, q, acts = [], [], [], []
    for _ in range(env.max_steps):
        a = policy_fn(obs)
        obs, _, term, trunc, info = env.step(a)
        ee, _ = env.sim.get_ee_pose()
        qq, _ = env.sim.get_joint_state()
        errs.append(info["pos_error"]); pos.append(ee)
        q.append(qq); acts.append(np.asarray(a, dtype=float))
        if term or trunc:
            break
    return np.array(errs), pos, q, acts


def _evaluate(controllers, trajs, conditions, episode_seconds, seeds, frame_stack, unc,
              action_smoothing=1.0):
    """Returns: results[(name, cond, traj)] = list of per-seed metric dicts,
    and err_curves[(name, cond, traj)] = error series from seed 0 (for plots)."""
    results, err_curves = {}, {}
    for name, (mode, policy_fn) in controllers.items():
        for cond in conditions:
            uncertain = (cond == "uncertain")
            for traj in trajs:
                key = (name, cond, traj)
                results[key] = []
                for s in range(seeds):
                    env = _make_env(mode, traj, uncertain, episode_seconds,
                                    seed=100 + s, frame_stack=frame_stack, unc=unc,
                                    action_smoothing=action_smoothing)
                    errs, pos, q, acts = _rollout(env, policy_fn)
                    results[key].append(
                        metrics.summarize(errs, pos, q, env.dt_ctrl, actions=acts))
                    if s == 0:
                        err_curves[key] = (errs, env.dt_ctrl)
                    env.close()
                print(f"  {name:9s} | {cond:9s} | {traj:13s} "
                      f"RMSE(ss)={np.mean([r['rmse_ss_mm'] for r in results[key]]):6.2f} mm  "
                      f"jerk={np.mean([r['ee_jerk_rms'] for r in results[key]]):7.1f}")
    return results, err_curves


def _aggregate(results):
    """Mean each metric over seeds and trajectories -> agg[(name,cond)] = dict."""
    agg = {}
    keys = {(n, c) for (n, c, _t) in results}
    metric_names = next(iter(results.values()))[0].keys()
    for (n, c) in keys:
        rows = [r for (kn, kc, _t), lst in results.items()
                if kn == n and kc == c for r in lst]
        agg[(n, c)] = {m: float(np.nanmean([r[m] for r in rows])) for m in metric_names}
    return agg


def _save_table(agg, outdir):
    path = os.path.join(outdir, "compare_table.csv")
    metric_names = sorted(next(iter(agg.values())).keys())
    with open(path, "w") as f:
        f.write("controller,condition," + ",".join(metric_names) + "\n")
        for (n, c) in sorted(agg):
            f.write(f"{n},{c}," + ",".join(f"{agg[(n,c)][m]:.4f}" for m in metric_names) + "\n")
    print(f"\n  saved table -> {path}")
    # also echo a compact view to the terminal
    print(f"\n  {'controller':10s} {'cond':10s} {'RMSE_ss(mm)':12s} "
          f"{'max(mm)':9s} {'EE_jerk':9s} {'SPARC':8s}")
    for (n, c) in sorted(agg):
        a = agg[(n, c)]
        print(f"  {n:10s} {c:10s} {a['rmse_ss_mm']:12.2f} {a['max_mm']:9.2f} "
              f"{a['ee_jerk_rms']:9.1f} {a['sparc']:8.2f}")


def _plots(agg, err_curves, trajs, controllers, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) grouped bar chart: steady-state RMSE, clean vs uncertain, per controller
    names = list(controllers.keys())
    conds = ["clean", "uncertain"]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, c in enumerate(conds):
        vals = [agg.get((n, c), {}).get("rmse_ss_mm", np.nan) for n in names]
        ax.bar(x + (i - 0.5) * w, vals, w, label=c)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("steady-state RMSE (mm)")
    ax.set_title("Tracking error by controller and condition")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = os.path.join(outdir, "compare_rmse.png"); fig.savefig(p, dpi=120); plt.close(fig)
    print(f"  saved plot  -> {p}")

    # 2) error-vs-time: pure_ik vs residual under uncertainty, per trajectory
    for traj in trajs:
        fig, ax = plt.subplots(figsize=(9, 3.5))
        plotted = False
        for n in names:
            key = (n, "uncertain", traj)
            if key in err_curves:
                errs, dt = err_curves[key]
                ax.plot(np.arange(len(errs)) * dt, errs * 1e3, label=n)
                plotted = True
        if not plotted:
            plt.close(fig); continue
        ax.set_title(f"Error vs time under uncertainty ({traj})")
        ax.set_xlabel("t (s)"); ax.set_ylabel("position error (mm)")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        p = os.path.join(outdir, f"compare_error_{traj}.png")
        fig.savefig(p, dpi=120); plt.close(fig)
        print(f"  saved plot  -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--residual", default=None, help="path to residual-RL .zip")
    ap.add_argument("--end2end", default=None, help="path to end-to-end RL .zip")
    ap.add_argument("--trajs", nargs="+", default=["circle", "figure_eight", "moving_target"])
    ap.add_argument("--seeds", type=int, default=3, help="trajectories averaged per cell")
    ap.add_argument("--episode_seconds", type=float, default=16.0)
    ap.add_argument("--conditions", nargs="+", default=["clean", "uncertain"],
                    choices=["clean", "uncertain"])
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--frame_stack", type=int, default=1,
                    help="MUST match the value used to train the models")
    ap.add_argument("--action_smoothing", type=float, default=1.0,
                    help="MUST match training")
    ap.add_argument("--unc_vel_gain", type=float, default=UNC_DEFAULTS["vel_gain"])
    ap.add_argument("--unc_control_delay", type=int, default=UNC_DEFAULTS["control_delay_steps"])
    ap.add_argument("--unc_obs_noise", type=float, default=UNC_DEFAULTS["obs_noise_std"])
    ap.add_argument("--unc_act_noise", type=float, default=UNC_DEFAULTS["action_noise_std"])
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    unc = dict(vel_gain=args.unc_vel_gain, control_delay_steps=args.unc_control_delay,
               obs_noise_std=args.unc_obs_noise, action_noise_std=args.unc_act_noise,
               randomize_uncertainty=False)

    # Build the controller set. pure_ik is always available (no model needed).
    controllers = {"pure_ik": ("residual", lambda obs: np.zeros(7))}

    if args.residual:
        from stable_baselines3 import SAC
        res_model = SAC.load(args.residual, device="cpu")
        controllers["residual"] = (
            "residual", lambda obs, m=res_model: m.predict(obs, deterministic=True)[0])
        print(f"loaded residual policy: {args.residual}")
    if args.end2end:
        from stable_baselines3 import SAC
        e2e_model = SAC.load(args.end2end, device="cpu")
        controllers["end2end"] = (
            "end2end", lambda obs, m=e2e_model: m.predict(obs, deterministic=True)[0])
        print(f"loaded end2end policy:  {args.end2end}")

    print(f"\nEvaluating {list(controllers)} on {args.trajs} "
          f"({args.seeds} seeds, conditions={args.conditions}, frame_stack={args.frame_stack})\n")
    results, err_curves = _evaluate(
        controllers, args.trajs, args.conditions, args.episode_seconds, args.seeds,
        args.frame_stack, unc, args.action_smoothing)
    agg = _aggregate(results)
    _save_table(agg, args.outdir)
    try:
        _plots(agg, err_curves, args.trajs, controllers, args.outdir)
    except Exception as e:
        print(f"  (plotting skipped: {e})")


if __name__ == "__main__":
    main()
