"""Gymnasium environment for residual-RL trajectory tracking.

The agent does NOT control the arm from scratch. A task-priority resolved-rate
controller produces a nominal joint-velocity command each tick; the policy adds
a bounded *residual* on top:

    dq = dq_nominal(task-priority IK)  +  residual_scale * action

Under clean conditions the best residual is ~0 (so RL can't hurt the strong
baseline); the residual earns its keep when uncertainty (Phase 4: obs/action
noise, control delay) degrades the nominal controller and the policy learns to
anticipate/compensate using the trajectory preview in its observation.

Observation (all scaled to ~unit range):
    q (7), dq (7), tracking error (3), EE velocity (3),
    trajectory preview at several future times (3*K), previous action (7),
    nominal controller command (7).

Action: residual joint velocity, Box(-1, 1, (7,)).

Reward: exp(-||e||/sigma)  -  w_act*||a||^2  -  w_smooth*||a - a_prev||^2
        (tracking; small-residual; smooth-residual). Early-terminate if the EE
        diverges badly.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .sim import PandaSim
from .tasks import TaskPriorityController
from .trajectories import make_trajectory

ARM_VEL_MAX = 2.0   # rad/s per-joint command clamp (conservative vs URDF limits)


class TrajectoryTrackingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, traj_types=("circle", "figure_eight", "moving_target"),
                 episode_seconds: float = 8.0, control_hz: float = 60.0,
                 sim_hz: float = 240.0, residual_scale: float = 0.5,
                 base_controller: bool = True,
                 randomize: bool = True, wide_workspace: bool = False,
                 unreachable_frac: float = 1.0,
                 preview_times=(0.05, 0.1, 0.2, 0.4),
                 w_track: float = 1.0, sigma_pos: float = 0.02,
                 w_act: float = 0.01, w_smooth: float = 0.05,
                 fail_dist: float = 0.30,
                 # --- uncertainty model (Phase 4); defaults OFF (clean) ---
                 # vel_gain & control_delay_steps DEGRADE the base controller
                 # (compensable by an anticipatory residual); obs/action noise
                 # test policy robustness. randomize_uncertainty samples per-episode
                 # levels in [0, level] (gain in [vel_gain, 1]) for domain randomization.
                 obs_noise_std: float = 0.0, action_noise_std: float = 0.0,
                 vel_gain: float = 1.0, control_delay_steps: int = 0,
                 randomize_uncertainty: bool = False,
                 frame_stack: int = 1, action_smoothing: float = 1.0,
                 gui: bool = False, seed: int | None = None):
        super().__init__()
        self.sim = PandaSim(gui=gui, timestep=1.0 / sim_hz)
        self.ctrl = TaskPriorityController(self.sim)   # position-only base controller

        self.traj_types = tuple(traj_types)
        self.randomize = randomize
        self.wide_workspace = wide_workspace
        self.unreachable_frac = float(unreachable_frac)
        self.dt_sim = 1.0 / sim_hz
        self.n_substeps = max(1, int(round(sim_hz / control_hz)))
        self.dt_ctrl = self.n_substeps * self.dt_sim
        self.max_steps = int(episode_seconds / self.dt_ctrl)
        self.residual_scale = residual_scale
        self.base_controller = base_controller
        self.preview_times = np.asarray(preview_times, dtype=float)

        self.w_track, self.sigma_pos = w_track, sigma_pos
        self.w_act, self.w_smooth = w_act, w_smooth
        # In wide-workspace mode the target is often legitimately out of reach,
        # so a large tracking error is expected, not a failure -> don't terminate
        # on it (episodes run full length; unreachable segments just score low).
        self.fail_dist = fail_dist if not wide_workspace else max(fail_dist, 1.5)

        # uncertainty configuration (nominal/max levels)
        self.obs_noise_std = obs_noise_std
        self.action_noise_std = action_noise_std
        self.vel_gain_cfg = float(vel_gain)
        self.control_delay_cfg = int(control_delay_steps)
        self.randomize_uncertainty = bool(randomize_uncertainty)
        # per-episode ACTIVE levels (set in reset)
        self._obs_noise = obs_noise_std
        self._action_noise = action_noise_std
        self._vel_gain = float(vel_gain)
        self._control_delay = int(control_delay_steps)

        self._rng = np.random.default_rng(seed)
        self.home_ee, _ = self.sim.get_ee_pose()

        self._jmid = 0.5 * (self.sim.joint_lower + self.sim.joint_upper)
        self._jhalf = 0.5 * (self.sim.joint_upper - self.sim.joint_lower)

        n_dof = self.sim.n_arm
        self.action_space = spaces.Box(-1.0, 1.0, (n_dof,), np.float32)
        base_dim = n_dof + n_dof + 3 + 3 + 3 * len(self.preview_times) + n_dof + n_dof
        self.frame_stack = max(1, int(frame_stack))
        self.action_smoothing = float(np.clip(action_smoothing, 1e-3, 1.0))
        self._act_filt = np.zeros(self.sim.n_arm)
        self._base_dim = base_dim
        # Stacking the last `frame_stack` observations gives the policy temporal
        # context, so it can infer the disturbance (e.g. commanded-vs-achieved
        # velocity reveals vel_gain; lag patterns reveal control delay) and adapt
        # — backing off when conditions are clean, compensating when they're not.
        obs_dim = base_dim * self.frame_stack
        self.observation_space = spaces.Box(-10.0, 10.0, (obs_dim,), np.float32)
        self._frame_buf = deque(maxlen=self.frame_stack)

        self.traj = None
        self.omega = 0.6
        self.phase = 0.0
        self.prev_action = np.zeros(n_dof)
        self.step_count = 0
        self._cmd_buf = deque()

        # GUI visualisation state (only used when gui=True)
        self._gui = gui
        self.episode_seconds = episode_seconds
        self._path_ids: list[int] = []
        self._tgt_m = None
        self._ee_m = None

    # ------------------------------------------------------------------ helpers
    def _sample_trajectory(self):
        if self.randomize:
            name = self._rng.choice(self.traj_types)
            self.omega = float(self._rng.uniform(0.4, 0.9))
            go_wide = self.wide_workspace and (self._rng.random() < self.unreachable_frac)
            if go_wide:
                # broad placement + larger size: some start/intermediate/end
                # points fall outside the arm's reach (intentional).
                center = self._rng.uniform([0.20, -0.45, 0.15], [0.60, 0.45, 0.75])
                size = float(self._rng.uniform(1.0, 1.8))
            else:
                # reachable, centred near home
                center = self.home_ee + self._rng.uniform(-0.05, 0.05, 3) * np.array([0.5, 1, 1])
                size = 1.0
            kw = {}
            if name == "circle":
                kw["radius"] = float(self._rng.uniform(0.10, 0.18) * size)
            elif name == "figure_eight":
                kw["width"] = float(self._rng.uniform(0.12, 0.20) * size)
                kw["height"] = float(self._rng.uniform(0.08, 0.14) * size)
            else:  # moving_target
                kw["extent"] = float(self._rng.uniform(0.10, 0.18) * size)
                kw["seed"] = int(self._rng.integers(0, 10_000))
            self.traj = make_trajectory(name, np.asarray(center, dtype=float),
                                        omega=self.omega, **kw)
            self.phase = float(self._rng.uniform(0, 2 * np.pi))
        else:
            self.omega = 0.6
            self.traj = make_trajectory(self.traj_types[0], self.home_ee, omega=self.omega)
            self.phase = 0.0

    def _nominal(self):
        tpos, tvel = self.traj(self.phase)
        dq_nom, perr, _, _ = self.ctrl.compute(tpos, tvel)
        return dq_nom, tpos, tvel, perr

    def _get_obs(self, dq_nom):
        q, dq = self.sim.get_joint_state()
        ee_pos, _ = self.sim.get_ee_pose()
        Jp, _ = self.sim.get_jacobian()
        ee_vel = Jp @ dq

        # sensor noise on the policy's view (true sim state is untouched, so the
        # base controller is unaffected — this purely tests policy robustness)
        if self._obs_noise > 0:
            q = q + self._rng.normal(0, self._obs_noise, q.shape)
            dq = dq + self._rng.normal(0, self._obs_noise, dq.shape)
            ee_pos = ee_pos + self._rng.normal(0, self._obs_noise, ee_pos.shape)

        tpos, _ = self.traj(self.phase)
        err = tpos - ee_pos
        preview = np.concatenate([
            self.traj(self.phase + self.omega * t)[0] - ee_pos
            for t in self.preview_times
        ])

        obs = np.concatenate([
            (q - self._jmid) / self._jhalf,    # joint angles -> [-1,1]
            dq / 2.0,                          # joint vels
            err / 0.2,                         # tracking error
            ee_vel / 0.5,                      # EE velocity
            preview / 0.2,                     # trajectory preview
            self.prev_action,                  # previous action
            dq_nom / 2.0,                      # nominal command being corrected
        ]).astype(np.float32)
        return np.clip(obs, -10.0, 10.0)

    def _draw_episode_viz(self):
        """Draw the current episode's path + target/EE markers (GUI only)."""
        for bid in self._path_ids:        # clear last episode's path
            self.sim.remove_body(bid)
        self._path_ids = []
        sample_T = max(2 * np.pi, self.omega * self.episode_seconds)
        vs = self.sim._sphere_shape(0.006, (0.15, 0.35, 0.9, 1.0))
        import pybullet as _p
        for ph in np.linspace(0.0, sample_T, 200):
            pt = self.traj(ph)[0]
            self._path_ids.append(_p.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs,
                basePosition=list(pt), physicsClientId=self.sim.client))
        ee, _ = self.sim.get_ee_pose()
        if self._tgt_m is None:           # create movable markers once
            self._tgt_m = self.sim.add_marker(ee, rgba=(0.9, 0.1, 0.1, 1.0), radius=0.017)
            self._ee_m = self.sim.add_marker(ee, rgba=(0.1, 0.9, 0.15, 1.0), radius=0.013)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.sim.reset()
        self._sample_trajectory()
        self.prev_action = np.zeros(self.sim.n_arm)
        self.step_count = 0
        self._act_filt = np.zeros(self.sim.n_arm)   # reset residual low-pass filter

        # --- resolve this episode's uncertainty levels ---
        if self.randomize_uncertainty:
            # domain randomization: sample levels so the policy sees a RANGE of
            # conditions (gain in [cfg, 1], delay in [0, cfg], noise in [0, cfg])
            self._obs_noise   = float(self._rng.uniform(0.0, self.obs_noise_std))
            self._action_noise = float(self._rng.uniform(0.0, self.action_noise_std))
            self._vel_gain    = float(self._rng.uniform(self.vel_gain_cfg, 1.0))
            self._control_delay = int(self._rng.integers(0, self.control_delay_cfg + 1))
        else:
            self._obs_noise, self._action_noise = self.obs_noise_std, self.action_noise_std
            self._vel_gain, self._control_delay = self.vel_gain_cfg, self.control_delay_cfg

        # command-delay buffer holds the last `control_delay` executed commands
        self._cmd_buf = deque([np.zeros(self.sim.n_arm)] * self._control_delay,
                              maxlen=max(1, self._control_delay + 1))
        if self._gui:
            self._draw_episode_viz()
        dq_nom = self._nominal()[0] if self.base_controller else np.zeros(self.sim.n_arm)
        raw = self._get_obs(dq_nom)
        # prime the stack with the first frame repeated so the initial stacked
        # observation is well-defined
        self._frame_buf.clear()
        for _ in range(self.frame_stack):
            self._frame_buf.append(raw)
        return np.concatenate(list(self._frame_buf)).astype(np.float32), {}

    def _execute(self, dq_total):
        """Apply the executed-command uncertainties to the FULL joint-velocity
        command, then step the sim once. Because these perturb what actually
        runs (not just the residual), they degrade the base controller too:
          - control delay : command is buffered and applied `delay` substeps late
                             -> systematic lag a reactive controller can't fix,
                                but an anticipatory residual (preview) can.
          - vel_gain      : executed = gain * commanded -> systematic undershoot.
          - action noise  : zero-mean perturbation on the executed command.
        """
        if self._control_delay > 0:
            self._cmd_buf.append(np.asarray(dq_total, dtype=float))
            dq_exec = self._cmd_buf.popleft()
        else:
            dq_exec = np.asarray(dq_total, dtype=float)
        dq_exec = self._vel_gain * dq_exec
        if self._action_noise > 0:
            dq_exec = dq_exec + self._rng.normal(0, self._action_noise, dq_exec.shape)
        dq_exec = np.clip(dq_exec, -ARM_VEL_MAX, ARM_VEL_MAX)
        self.sim.set_arm_velocities(dq_exec)
        self.sim.step()
        self.phase += self.omega * self.dt_sim

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)

        if self.base_controller:
            # residual RL: nominal IK command + learned correction (held over substeps)
            residual = self.residual_scale * action
            # low-pass filter (EMA) the residual to suppress high-frequency chatter
            # -> lower jerk. alpha=1.0 disables it; smaller alpha = smoother/slower.
            self._act_filt = (self.action_smoothing * residual
                              + (1.0 - self.action_smoothing) * self._act_filt)
            residual = self._act_filt
            for _ in range(self.n_substeps):
                tpos, tvel = self.traj(self.phase)
                dq_nom, *_ = self.ctrl.compute(tpos, tvel)
                self._execute(dq_nom + residual)
        else:
            # end-to-end RL: the action IS the full joint-velocity command
            cmd = ARM_VEL_MAX * action
            self._act_filt = (self.action_smoothing * cmd
                              + (1.0 - self.action_smoothing) * self._act_filt)
            cmd = self._act_filt
            for _ in range(self.n_substeps):
                self._execute(cmd)

        ee_pos, _ = self.sim.get_ee_pose()
        tpos, _ = self.traj(self.phase)
        err = float(np.linalg.norm(tpos - ee_pos))

        if self._gui and self._tgt_m is not None:
            self.sim.move_marker(self._tgt_m, tpos)
            self.sim.move_marker(self._ee_m, ee_pos)

        reward = (self.w_track * np.exp(-err / self.sigma_pos)
                  - self.w_act * float(np.sum(action ** 2))
                  - self.w_smooth * float(np.sum((action - self.prev_action) ** 2)))

        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        terminated = err > self.fail_dist
        if terminated:
            # Forfeit the best-case remaining return. A fixed penalty (e.g. -10)
            # is exploitable: in a hopeless state the agent could deliberately
            # diverge to END the episode and stop accruing low per-step reward.
            # Tying the penalty to the reward it gives up makes diverging never
            # better than continuing, so there is no "suicide" shortcut.
            reward -= self.w_track * (self.max_steps - self.step_count)

        self.prev_action = action
        dq_nom_next = self._nominal()[0] if self.base_controller else np.zeros(self.sim.n_arm)
        raw = self._get_obs(dq_nom_next)
        self._frame_buf.append(raw)        # drops oldest, appends newest
        obs = np.concatenate(list(self._frame_buf)).astype(np.float32)
        info = {"pos_error": err, "is_success": err < 0.01}
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self.sim.close()
