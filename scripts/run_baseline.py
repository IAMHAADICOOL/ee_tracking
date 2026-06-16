"""Phase 1 baseline: track a Cartesian trajectory with the model-based controller.

No reinforcement learning here. Establishes the baseline later phases must beat
and de-risks the kinematics/plumbing. Logs desired vs. actual EE path, prints
tracking metrics, and saves a plot + (optionally) an MP4.

Headless (fixed duration, for metrics / CI):
    python -m scripts.run_baseline --traj figure_eight --seconds 14 --video

Interactive window (runs until you CLOSE it, then saves graphs + video):
    python -m scripts.run_baseline --traj figure_eight --gui --orientation --video
"""
from __future__ import annotations

import argparse
import os
import time
from collections import deque

import numpy as np

from src.arm_tracking.sim import PandaSim
from src.arm_tracking.trajectories import make_trajectory

MAX_TRAIL = 250          # comet-tail length (bounded so GUI runs forever safely)
VIDEO_W, VIDEO_H = 720, 544


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="circle",
                    choices=["circle", "figure_eight", "moving_target"])
    ap.add_argument("--seconds", type=float, default=12.0,
                    help="run length (ignored with --gui, which runs until you close the window)")
    ap.add_argument("--omega", type=float, default=0.6, help="phase rate rad/s")
    ap.add_argument("--controller", default="taskpriority",
                    choices=["dls", "taskpriority"])
    ap.add_argument("--orientation", action="store_true",
                    help="also hold EE orientation (task-priority controller only)")
    ap.add_argument("--gui", action="store_true",
                    help="open the PyBullet window, run in real time until closed (needs a display)")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sim = PandaSim(gui=args.gui)
    if args.gui:
        sim.setup_gui_camera()

    home_ee, home_quat = sim.get_ee_pose()
    traj = make_trajectory(args.traj, center=home_ee, omega=args.omega)

    if args.controller == "taskpriority":
        from src.arm_tracking.tasks import TaskPriorityController
        ctrl = TaskPriorityController(sim, track_orientation=args.orientation)
    else:
        from src.arm_tracking.ik_controller import DLSController
        ctrl = DLSController(sim)
        if args.orientation:
            print("note: --orientation ignored (use --controller taskpriority)")
            args.orientation = False
    target_quat = home_quat

    dt = sim.timestep
    frame_every = max(1, int((1.0 / dt) / 30))   # ~30 fps capture
    trail_every = frame_every * 3

    # Markers: shown live in --gui and recorded in --video.
    show = args.video or args.gui
    target_marker = ee_marker = None
    trail = deque()
    if show:
        sample_T = max(2 * np.pi, args.omega * (args.seconds if not args.gui else 30.0))
        path_pts = [traj(ph)[0] for ph in np.linspace(0.0, sample_T, 220)]
        sim.draw_path(path_pts, rgba=(0.15, 0.35, 0.9, 1.0), radius=0.006)
        target_marker = sim.add_marker(home_ee, rgba=(0.9, 0.1, 0.1, 1.0), radius=0.017)
        ee_marker = sim.add_marker(home_ee, rgba=(0.1, 0.9, 0.15, 1.0), radius=0.013)

    # Video: GUI records the live window via PyBullet's logger (stable on all GL
    # drivers); headless streams offscreen frames through imageio.
    writer = None
    gui_log = None
    video_path = os.path.join(args.outdir, f"baseline_{args.traj}.mp4")
    if args.video:
        if args.gui:
            gui_log = sim.start_video_log(video_path)
        else:
            import imageio
            writer = imageio.get_writer(video_path, fps=30, macro_block_size=16)

    cam = dict(width=VIDEO_W, height=VIDEO_H, cam_target=tuple(home_ee),
               distance=0.8, yaw=88, pitch=-14, use_opengl=False)

    desired, actual, errors, orn_errors = [], [], [], []
    n_steps = None if args.gui else int(args.seconds / dt)

    if args.gui:
        print("GUI open — orbit with the mouse. Close the window (or Ctrl+C) to "
              "stop; graphs and video are saved on exit.")

    def add_trail(pos):
        if len(trail) < MAX_TRAIL:
            trail.append(sim.add_marker(pos, rgba=(1.0, 0.55, 0.0, 1.0), radius=0.004))
        else:                                   # recycle oldest -> fixed-length tail
            bid = trail.popleft(); sim.move_marker(bid, pos); trail.append(bid)

    phase = 0.0
    k = 0
    try:
        while True:
            if args.gui and not sim.is_connected():
                break
            if n_steps is not None and k >= n_steps:
                break
            t0 = time.time()

            tgt_pos, tgt_vel = traj(phase)
            if args.controller == "taskpriority":
                _, err, oerr, _ = ctrl.step(tgt_pos, tgt_vel,
                                            target_quat if args.orientation else None)
                if oerr is not None:
                    orn_errors.append(float(np.linalg.norm(oerr)))
            else:
                _, err = ctrl.step(tgt_pos, tgt_vel)
            sim.step()
            phase += args.omega * dt

            ee_pos, _ = sim.get_ee_pose()
            desired.append(tgt_pos.copy())
            actual.append(ee_pos.copy())
            errors.append(float(np.linalg.norm(err)))

            if show and k % frame_every == 0:
                sim.move_marker(target_marker, tgt_pos)
                sim.move_marker(ee_marker, ee_pos)
                if k % trail_every == 0:
                    add_trail(ee_pos)
                if writer is not None:
                    writer.append_data(sim.render(**cam))

            if args.gui:
                time.sleep(max(0.0, dt - (time.time() - t0)))
            k += 1
    except KeyboardInterrupt:
        print("\ninterrupted — saving outputs")
    except Exception as e:                      # window closed mid-step in GUI
        if args.gui:
            print(f"\nwindow closed — saving outputs ({type(e).__name__})")
        else:
            if writer is not None:
                writer.close()
            raise

    if gui_log is not None:
        sim.stop_video_log(gui_log)
        print(f"  saved video -> {video_path}  (requires system ffmpeg)")

    _report_and_save(np.array(desired), np.array(actual), np.array(errors),
                     orn_errors, dt, args, writer, video_path)
    sim.close()


def _report_and_save(desired, actual, errors, orn_errors, dt, args, writer, video_path):
    if len(errors) == 0:
        print("no data collected")
        if writer is not None:
            writer.close()
        return

    settle = min(int(1.0 / dt), len(errors) - 1)
    ss = errors[settle:] if len(errors) > settle else errors
    dur = len(errors) * dt
    print(f"\nTrajectory: {args.traj}  |  controller={args.controller}"
          f"{' +orientation' if args.orientation else ''}  |  "
          f"{dur:.1f}s @ {1/dt:.0f} Hz")
    print(f"  steady-state RMSE : {np.sqrt(np.mean(ss**2))*1000:.2f} mm")
    print(f"  steady-state max  : {ss.max()*1000:.2f} mm")
    print(f"  mean error        : {ss.mean()*1000:.2f} mm")
    if orn_errors:
        oss = np.array(orn_errors)[settle:] if len(orn_errors) > settle else np.array(orn_errors)
        print(f"  orient RMSE       : {np.degrees(np.sqrt(np.mean(oss**2))):.3f} deg")
        print(f"  orient max        : {np.degrees(oss.max()):.3f} deg")

    _save_plot(desired, actual, errors, dt, args)
    if writer is not None:
        writer.close()
        print(f"  saved video -> {video_path}")


def _save_plot(desired, actual, errors, dt, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(errors)) * dt
    settle = min(int(1.0 / dt), max(len(errors) - 1, 0))
    fig = plt.figure(figsize=(12, 4.5))

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    d_ss, a_ss = desired[settle:], actual[settle:]
    ax.plot(*d_ss.T, "k--", lw=1.4, label="desired")
    ax.plot(*a_ss.T, "C0", lw=1.0, label="actual")
    ax.set_title(f"{args.traj}: 3D path (steady state)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend(loc="upper left", fontsize=8)
    pts = np.vstack([d_ss, a_ss])
    center = pts.mean(axis=0)
    span = max(pts.max(axis=0) - pts.min(axis=0)) / 2 + 1e-3
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)

    ax2 = fig.add_subplot(1, 3, 2)
    for i, lab in enumerate("xyz"):
        ax2.plot(t, (actual[:, i] - desired[:, i]) * 1000, label=f"{lab}-err")
    ax2.set_title("per-axis error"); ax2.set_xlabel("t (s)")
    ax2.set_ylabel("error (mm)"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(t, errors * 1000, "C3")
    ax3.set_title("position error norm"); ax3.set_xlabel("t (s)")
    ax3.set_ylabel("error (mm)"); ax3.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(args.outdir, f"baseline_{args.traj}.png")
    fig.savefig(path, dpi=120)
    print(f"  saved plot  -> {path}")


if __name__ == "__main__":
    main()
