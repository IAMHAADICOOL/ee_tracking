"""Watch (or evaluate) a trained policy tracking a trajectory.

Loads a saved SAC model and runs it in the env, with the same path / target /
EE-tip visualization as the baseline. Point it at the final policy or at any
checkpoint in models/<tag>_ckpts/ to see how tracking improves over training.

  # watch the final residual policy live
  python -m scripts.watch_policy --model models/residual.zip --mode residual --gui

  # see an early checkpoint vs a late one (learning progression)
  python -m scripts.watch_policy --model models/residual_ckpts/residual_20000_steps.zip --mode residual --gui

  # headless: save tracking plot + MP4
  python -m scripts.watch_policy --model models/end2end.zip --mode end2end --traj figure_eight --video

The --mode MUST match how the model was trained (it sets residual vs end-to-end
dynamics). Reports the same steady-state RMSE metric as the baseline so numbers
are directly comparable.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
from stable_baselines3 import SAC

from src.arm_tracking.env import TrajectoryTrackingEnv
from src.arm_tracking import metrics

VIDEO_W, VIDEO_H = 720, 544


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to a saved .zip policy")
    ap.add_argument("--mode", choices=["residual", "end2end"], required=True)
    ap.add_argument("--traj", default="figure_eight",
                    choices=["circle", "figure_eight", "moving_target"])
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--episode_seconds", type=float, default=12.0)
    ap.add_argument("--uncertainty", action="store_true")
    ap.add_argument("--frame_stack", type=int, default=1,
                    help="MUST match the value used to train the model")
    ap.add_argument("--action_smoothing", type=float, default=1.0,
                    help="MUST match training")
    ap.add_argument("--unc_vel_gain", type=float, default=0.70)
    ap.add_argument("--unc_control_delay", type=int, default=12)
    ap.add_argument("--unc_obs_noise", type=float, default=0.005)
    ap.add_argument("--unc_act_noise", type=float, default=0.02)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--csv", action="store_true",
                    help="dump per-step log (t, error, EE xyz, joints) to outputs/")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--label", default=None,
                    help="output name stem (default: model filename); keeps runs from overwriting")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    # Unique stem per (model, trajectory, condition) so videos/plots/csvs from
    # different policies or conditions never overwrite each other.
    cond = "unc" if args.uncertainty else "clean"
    label = args.label or os.path.splitext(os.path.basename(args.model))[0]
    args._stem = f"policy_{label}_{args.traj}_{cond}"
    kw = dict(base_controller=(args.mode == "residual"), randomize=False,
              traj_types=(args.traj,), episode_seconds=args.episode_seconds,
              frame_stack=args.frame_stack, action_smoothing=args.action_smoothing,
              gui=args.gui, seed=0)
    if args.uncertainty:
        # fixed (non-randomized) condition so eval is at a known, repeatable level
        kw.update(obs_noise_std=args.unc_obs_noise, action_noise_std=args.unc_act_noise,
                  vel_gain=args.unc_vel_gain, control_delay_steps=args.unc_control_delay,
                  randomize_uncertainty=False)
    env = TrajectoryTrackingEnv(**kw)
    if args.gui:
        env.sim.setup_gui_camera()

    model = SAC.load(args.model, device="cpu")
    print(f"loaded {args.model}  (mode={args.mode})")

    show = args.gui or args.video
    writer = gui_log = None
    video_path = os.path.join(args.outdir, f"{args._stem}.mp4")
    if args.video and not args.gui:
        import imageio
        writer = imageio.get_writer(video_path, fps=30, macro_block_size=16)
    elif args.video and args.gui:
        gui_log = env.sim.start_video_log(video_path)
    cam = dict(width=VIDEO_W, height=VIDEO_H, cam_target=tuple(env.home_ee),
               distance=0.9, yaw=70, pitch=-20, use_opengl=False)
    frame_every = max(1, int((1.0 / env.dt_ctrl) / 30))

    all_err = []
    try:
        for ep in range(args.episodes):
            if args.gui and not env.sim.is_connected():
                break
            obs, _ = env.reset()
            tgt_m = ee_m = None
            if show:
                pts = [env.traj(p)[0] for p in np.linspace(0, 2 * np.pi, 200)]
                env.sim.draw_path(pts, rgba=(0.15, 0.35, 0.9, 1.0), radius=0.006)
                tgt_m = env.sim.add_marker(env.home_ee, rgba=(0.9, 0.1, 0.1, 1.0), radius=0.017)
                ee_m = env.sim.add_marker(env.home_ee, rgba=(0.1, 0.9, 0.15, 1.0), radius=0.013)
            errs, pos_log, q_log, act_log = [], [], [], []
            for k in range(env.max_steps):
                if args.gui and not env.sim.is_connected():
                    break
                t0 = time.time()
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                ee, _ = env.sim.get_ee_pose()
                q, _ = env.sim.get_joint_state()
                errs.append(info["pos_error"]); pos_log.append(ee)
                q_log.append(q); act_log.append(np.asarray(action, dtype=float))
                if show and k % frame_every == 0:
                    tp = env.traj(env.phase)[0]
                    env.sim.move_marker(tgt_m, tp); env.sim.move_marker(ee_m, ee)
                    if writer is not None:
                        writer.append_data(env.sim.render(**cam))
                if args.gui:
                    time.sleep(max(0.0, env.dt_ctrl - (time.time() - t0)))
                if term or trunc:
                    break
            errs = np.array(errs)
            m = metrics.summarize(errs, pos_log, q_log, env.dt_ctrl, actions=act_log)
            print(f"  episode {ep}: "
                  f"RMSE(steady) {m['rmse_ss_mm']:.2f} mm | max {m['max_mm']:.2f} mm | "
                  f"EE jerk {m['ee_jerk_rms']:.1f} m/s^3 | joint jerk {m['joint_jerk_rms']:.2f} | "
                  f"SPARC {m['sparc']:.2f} | cmd jitter {m['command_jitter']:.3f}")
            all_err.append(errs)
            if args.csv:
                _save_csv(ep, errs, np.array(pos_log), np.array(q_log), env.dt_ctrl, args)
    except KeyboardInterrupt:
        print("\ninterrupted")
    except Exception as e:
        if args.gui:
            print(f"\nwindow closed ({type(e).__name__})")
        else:
            if writer is not None:
                writer.close()
            raise

    if gui_log is not None:
        env.sim.stop_video_log(gui_log)
        print(f"  saved video -> {video_path}  (requires system ffmpeg)")
    if writer is not None:
        writer.close()
        print(f"  saved video -> {video_path}")

    if all_err:
        _save_plot(all_err[0], env.dt_ctrl, args)
    env.close()


def _save_csv(ep, errs, pos, q, dt, args):
    t = (np.arange(len(errs)) * dt).reshape(-1, 1)
    cols = ["t", "error_m", "ee_x", "ee_y", "ee_z"] + [f"q{i}" for i in range(q.shape[1])]
    data = np.hstack([t, errs.reshape(-1, 1), pos, q])
    path = os.path.join(args.outdir, f"{args._stem}_ep{ep}.csv")
    np.savetxt(path, data, delimiter=",", header=",".join(cols), comments="")
    print(f"  saved csv   -> {path}")


def _save_plot(errs, dt, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(len(errs)) * dt
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(t, errs * 1000, "C2")
    ax.set_title(f"{args._stem} tracking error")
    ax.set_xlabel("t (s)"); ax.set_ylabel("position error (mm)"); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(args.outdir, f"{args._stem}.png")
    fig.savefig(path, dpi=120)
    print(f"  saved plot  -> {path}")


if __name__ == "__main__":
    main()
