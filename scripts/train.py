"""Train a tracking policy with SAC (Phase 3). One file, two modes.

  # residual RL (policy corrects the task-priority controller)
  python -m scripts.train --mode residual --timesteps 200000

  # end-to-end RL (policy outputs joint velocities directly, no IK)
  python -m scripts.train --mode end2end  --timesteps 1000000

  # watch it learn live in the PyBullet window (single env, slower)
  python -m scripts.train --mode residual --watch

  # Phase 4: train under uncertainty (obs/action noise + control delay)
  python -m scripts.train --mode residual --uncertainty --timesteps 400000

Outputs:
  models/<mode>.zip            final policy
  models/<mode>_ckpts/         periodic checkpoints (watch their progress)
  models/<mode>_best/          best-by-eval policy
  runs/<mode>/                 TensorBoard logs  ->  tensorboard --logdir runs
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (BaseCallback, CheckpointCallback,
                                                EvalCallback)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from src.arm_tracking.env import TrajectoryTrackingEnv

# Phase-4 uncertainty preset (only used with --uncertainty).
# Default uncertainty levels (HARDER than the first attempt so the strong base
# controller actually degrades, leaving room for the residual to compensate).
# All overridable from the CLI. randomize_uncertainty samples per-episode levels.
UNC_DEFAULTS = dict(obs_noise_std=0.005, action_noise_std=0.02,
                    vel_gain=0.70, control_delay_steps=12)


def make_env(mode, uncertainty, gui=False, randomize=True, seed=0, **env_kw):
    def _f():
        kw = dict(base_controller=(mode == "residual"),
                  randomize=randomize, gui=gui, seed=seed)
        kw.update(env_kw)               # episode_seconds, wide_workspace, reward weights, frame_stack, ...
        if uncertainty:                 # uncertainty is a dict (or None/False)
            kw.update(uncertainty)
        return TrajectoryTrackingEnv(**kw)
    return _f


class ErrorLogCallback(BaseCallback):
    """Log mean Cartesian tracking error (mm) to TensorBoard."""
    def __init__(self):
        super().__init__()
        self._errs = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "pos_error" in info:
                self._errs.append(info["pos_error"])
        if len(self._errs) >= 2000:
            self.logger.record("rollout/mean_pos_error_mm", float(np.mean(self._errs)) * 1000)
            self._errs = []
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["residual", "end2end"], default="residual")
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--episode_seconds", type=float, default=12.0,
                    help="episode length in seconds (>=16 guarantees a full slow loop)")
    ap.add_argument("--wide_workspace", action="store_true",
                    help="spread trajectories across the workspace; some points may be unreachable")
    ap.add_argument("--unreachable_frac", type=float, default=1.0,
                    help="with --wide_workspace, fraction of episodes placed aggressively (try 0.3-0.5)")
    ap.add_argument("--load", default=None,
                    help="continue training from a saved .zip (curriculum / fine-tune)")
    ap.add_argument("--tag", default=None,
                    help="output name for this run; controls models/<tag>* and runs/<tag> "
                         "(default: mode[+_unc])")
    ap.add_argument("--n_envs", type=int, default=4, help="parallel envs (headless only)")
    # --- finalized reward weights (Phase 5); override only to retune ---
    ap.add_argument("--w_track", type=float, default=1.0)
    ap.add_argument("--sigma_pos", type=float, default=0.02)
    ap.add_argument("--w_act", type=float, default=0.01)
    ap.add_argument("--w_smooth", type=float, default=0.05)
    # --- frame stacking: temporal context so the policy can infer the disturbance ---
    ap.add_argument("--frame_stack", type=int, default=1,
                    help="stack the last N observations (use 4 for the uncertainty push)")
    ap.add_argument("--action_smoothing", type=float, default=1.0,
                    help="EMA factor on the residual to cut jerk (1.0=off; try 0.3-0.5)")
    # --- uncertainty levels (only used with --uncertainty); override to harden ---
    ap.add_argument("--unc_vel_gain", type=float, default=UNC_DEFAULTS["vel_gain"],
                    help="actuator gain (executed=gain*commanded); lower = harder")
    ap.add_argument("--unc_control_delay", type=int, default=UNC_DEFAULTS["control_delay_steps"],
                    help="control latency in sim substeps; higher = harder")
    ap.add_argument("--unc_obs_noise", type=float, default=UNC_DEFAULTS["obs_noise_std"])
    ap.add_argument("--unc_act_noise", type=float, default=UNC_DEFAULTS["action_noise_std"])
    ap.add_argument("--watch", action="store_true",
                    help="train in a live GUI window (forces 1 env, slower)")
    ap.add_argument("--watch_all", action="store_true",
                    help="open a GUI window for EVERY parallel env while training "
                         "(needs n_envs>1; resource-heavy/experimental — keep n_envs small)")
    ap.add_argument("--watch_first_only", action="store_true",
                    help="with --watch_all, show only env 0 in a window; the rest "
                         "train headless (robust, keeps full throughput)")
    ap.add_argument("--uncertainty", action="store_true", help="enable Phase-4 toggles")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs("models", exist_ok=True)
    tag = args.tag if args.tag else args.mode + ("_unc" if args.uncertainty else "")

    # ----- uncertainty config (built from CLI flags; None when --uncertainty off) -----
    unc = (dict(vel_gain=args.unc_vel_gain,
                control_delay_steps=args.unc_control_delay,
                obs_noise_std=args.unc_obs_noise,
                action_noise_std=args.unc_act_noise,
                randomize_uncertainty=True)
           if args.uncertainty else None)
    if unc:
        print(f"uncertainty: vel_gain={args.unc_vel_gain} delay={args.unc_control_delay} "
              f"obs_noise={args.unc_obs_noise} act_noise={args.unc_act_noise} (randomized)")

    # ----- training env(s) -----
    common = dict(episode_seconds=args.episode_seconds,
                  wide_workspace=args.wide_workspace,
                  unreachable_frac=args.unreachable_frac,
                  frame_stack=args.frame_stack,
                  action_smoothing=args.action_smoothing,
                  w_track=args.w_track, sigma_pos=args.sigma_pos,
                  w_act=args.w_act, w_smooth=args.w_smooth)
    if args.watch:
        venv = DummyVecEnv([make_env(args.mode, unc, gui=True,
                                     seed=args.seed, **common)])
    elif args.watch_all and args.n_envs > 1:
        # one GUI window PER parallel env (each subprocess opens its own).
        # gui_first_only=True -> only env 0 gets a window, rest headless (robust,
        # keeps throughput). Resource-heavy + OpenGL-across-processes is finicky:
        # use spawn, keep n_envs small, treat as a viz run not a real training run.
        venv = SubprocVecEnv(
            [make_env(args.mode, unc,
                      gui=(i == 0 or not args.watch_first_only),
                      seed=args.seed + i, **common)
             for i in range(args.n_envs)],
            start_method="spawn")
    elif args.n_envs > 1:
        venv = SubprocVecEnv([make_env(args.mode, unc,
                                       seed=args.seed + i, **common)
                              for i in range(args.n_envs)])
    else:
        venv = DummyVecEnv([make_env(args.mode, unc,
                                     seed=args.seed, **common)])
    venv = VecMonitor(venv)

    # ----- eval env (headless, fixed trajectory for a comparable signal) -----
    eval_env = VecMonitor(DummyVecEnv(
        [make_env(args.mode, unc, randomize=False, seed=999, **common)]))

    callbacks = [
        ErrorLogCallback(),
        CheckpointCallback(save_freq=max(1, 20_000 // max(1, args.n_envs)),
                           save_path=f"models/{tag}_ckpts", name_prefix=tag),
        EvalCallback(eval_env, best_model_save_path=f"models/{tag}_best",
                     eval_freq=max(1, 10_000 // max(1, args.n_envs)),
                     n_eval_episodes=3, deterministic=True),
    ]

    if args.load:
        print(f"continuing from {args.load}")
        model = SAC.load(args.load, env=venv, device="auto",
                         tensorboard_log=f"runs/{tag}")
        model.verbose = 1
    else:
        model = SAC("MlpPolicy", venv, verbose=1, seed=args.seed,
                    learning_starts=5_000, batch_size=256,
                    tensorboard_log=f"runs/{tag}")
    n_windows = (1 if args.watch
                 else (1 if (args.watch_all and args.watch_first_only)
                       else args.n_envs) if args.watch_all else 0)
    mode_str = (f"{args.n_envs} env(s), {n_windows} GUI window(s)" if args.watch_all
                else "1 (GUI) env" if args.watch
                else f"{args.n_envs} env(s)")
    print(f"Training {tag}: {args.timesteps} steps, {mode_str}")
    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)

    out = f"models/{tag}.zip"
    model.save(out)
    print(f"saved policy -> {out}")
    venv.close(); eval_env.close()


if __name__ == "__main__":
    main()
