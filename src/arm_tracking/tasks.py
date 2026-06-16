"""Task-priority control for the Panda arm.

Adapted from the recursive task-priority / null-space projection framework in
`swiftpro_robotics_rrc_2.py` (the SwiftPro VMS code), specialised to a fixed-base
7-DOF arm tracking a Cartesian trajectory.

Idea: instead of one monolithic controller, express each objective as a `Task`
with its own Jacobian + error, order them by priority, and solve recursively so
each task acts only in the *null-space* of all higher-priority tasks. Higher
priority is never disturbed by lower priority. For us:

    [joint-limit tasks]  (highest — safety, one per joint, hysteresis-gated)
        -> position task (primary — track the moving Cartesian target)
        -> orientation task (optional secondary — hold/track EE orientation)
        -> centering task   (lowest — pull toward home posture in the leftover DOF)

This directly gives us joint-limit avoidance (key for the "unreachable target"
uncertainty in Phase 4) and a drop-in EE-orientation objective.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sim import PandaSim, PANDA_HOME


# --------------------------------------------------------------------- helpers
def DLS(A: np.ndarray, damping: float) -> np.ndarray:
    """Damped least-squares pseudo-inverse  A^T (A A^T + lambda^2 I)^-1."""
    m = A.shape[0]
    return A.T @ np.linalg.inv(A @ A.T + (damping ** 2) * np.eye(m))


def quat_conj(q):  # xyzw
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_mul(a, b):  # xyzw, returns a (x) b
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def orientation_error(q_cur, q_des):
    """World-frame orientation error (rotation vector) taking current -> desired.

    Returns a 3-vector e such that a world angular velocity omega = K*e reduces
    the orientation error. Matches PyBullet's world-frame angular Jacobian.
    """
    q_rel = quat_mul(q_des, quat_conj(q_cur))   # world-frame relative rotation
    v, w = q_rel[:3], q_rel[3]
    if w < 0.0:                                  # take the short way round
        v, w = -v, -w
    nv = np.linalg.norm(v)
    if nv < 1e-9:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(nv, w)
    return (v / nv) * angle


@dataclass
class ArmState:
    """Snapshot of everything the tasks need, computed once per control tick."""
    q: np.ndarray        # (7,) joint angles
    dq: np.ndarray       # (7,) joint velocities
    ee_pos: np.ndarray   # (3,) EE position, world
    ee_quat: np.ndarray  # (4,) EE orientation xyzw, world
    J_pos: np.ndarray    # (3,7) linear Jacobian, world
    J_orn: np.ndarray    # (3,7) angular Jacobian, world


# ------------------------------------------------------------------ task types
class Task:
    """Abstract task: holds a Jacobian, an error, a gain and a feed-forward."""

    def __init__(self, name: str, dim: int, n_dof: int = 7):
        self.name = name
        self.dim = dim
        self.n_dof = n_dof
        self.J = np.zeros((dim, n_dof))
        self.err = np.zeros((dim, 1))
        self.K = np.eye(dim)
        self.ff = np.zeros((dim, 1))

    def update(self, state: ArmState):
        raise NotImplementedError

    def is_active(self) -> bool:
        return True

    def set_gain(self, k):
        self.K = k if np.ndim(k) == 2 else np.eye(self.dim) * float(k)

    def set_ff(self, ff):
        self.ff = np.asarray(ff, dtype=float).reshape(self.dim, 1)


class PositionTask(Task):
    """Track a 3-D Cartesian target (with optional feed-forward velocity)."""

    def __init__(self, kp: float = 4.0, n_dof: int = 7):
        super().__init__("position", 3, n_dof)
        self.set_gain(kp)
        self.target = np.zeros(3)

    def set_target(self, pos, vel=None):
        self.target = np.asarray(pos, dtype=float)
        self.set_ff(np.zeros(3) if vel is None else vel)

    def update(self, state):
        self.err = (self.target - state.ee_pos).reshape(3, 1)
        self.J = state.J_pos


class OrientationTask(Task):
    """Track an EE orientation (quaternion). Secondary to position by default."""

    def __init__(self, kp: float = 2.0, n_dof: int = 7):
        super().__init__("orientation", 3, n_dof)
        self.set_gain(kp)
        self.target = np.array([0.0, 0.0, 0.0, 1.0])

    def set_target(self, quat, ang_vel=None):
        self.target = np.asarray(quat, dtype=float)
        self.set_ff(np.zeros(3) if ang_vel is None else ang_vel)

    def update(self, state):
        self.err = orientation_error(state.ee_quat, self.target).reshape(3, 1)
        self.J = state.J_orn


class JointLimitTask(Task):
    """Hysteresis-gated joint-limit avoidance for ONE joint (highest priority).

    Activates when the joint enters `margin` of a hard limit and commands a
    velocity pushing it back; deactivates only once it retreats past
    `margin * hysteresis` (wider) so it doesn't chatter on the boundary.
    Directly adapted from VMSJointLimitsTask.
    """

    def __init__(self, joint_idx: int, q_min: float, q_max: float,
                 margin: float = 0.10, hysteresis: float = 1.5,
                 gain: float = 1.0, n_dof: int = 7):
        super().__init__(f"jlim_{joint_idx}", 1, n_dof)
        self.joint_idx = joint_idx
        self.q_min, self.q_max = q_min, q_max
        self.margin = margin
        self.delta = margin * hysteresis
        self.set_gain(gain)
        self._active = False
        self._dir = 0
        self.J = np.zeros((1, n_dof))
        self.J[0, joint_idx] = 1.0

    def update(self, state):
        q = state.q[self.joint_idx]
        if not self._active:
            if q >= self.q_max - self.margin:
                self._dir, self._active = -1, True
            elif q <= self.q_min + self.margin:
                self._dir, self._active = 1, True
        else:
            if self._dir == -1 and q <= self.q_max - self.delta:
                self._dir, self._active = 0, False
            elif self._dir == 1 and q >= self.q_min + self.delta:
                self._dir, self._active = 0, False
        self.err = np.array([[float(self._dir)]])  # +/-1 push direction

    def is_active(self):
        return self._active


class CenteringTask(Task):
    """Lowest-priority null-space task: gently pull all joints toward a posture."""

    def __init__(self, q_center, gain: float = 0.3, n_dof: int = 7):
        super().__init__("centering", n_dof, n_dof)
        self.q_center = np.asarray(q_center, dtype=float)
        self.set_gain(gain)
        self.J = np.eye(n_dof)

    def update(self, state):
        self.err = (self.q_center - state.q).reshape(self.n_dof, 1)
        self.J = np.eye(self.n_dof)


class ConfigurationTask(Task):
    """Combined 6-DOF pose task: position + orientation solved *simultaneously*.

    Stacks the linear and angular Jacobians into a 6x7 task and the position +
    orientation errors into a 6-vector, so one least-squares solve reduces both
    at once (no artificial priority between them). A block-diagonal gain lets you
    weight translation vs. rotation independently. Preferred over separate
    Position+Orientation tasks whenever a full target pose is specified.

        J   = [J_pos; J_orn]               (6x7)
        err = [p_des - p; rotvec(q_des, q)] (6,1)
        K   = diag(kp_pos*I3, kp_orn*I3)    (6x6)
    """

    def __init__(self, kp_pos: float = 4.0, kp_orn: float = 2.0, n_dof: int = 7):
        super().__init__("configuration", 6, n_dof)
        K = np.zeros((6, 6))
        K[:3, :3] = np.eye(3) * kp_pos
        K[3:, 3:] = np.eye(3) * kp_orn
        self.K = K
        self.target_pos = np.zeros(3)
        self.target_quat = np.array([0.0, 0.0, 0.0, 1.0])

    def set_target(self, pos, quat, vel=None, ang_vel=None):
        self.target_pos = np.asarray(pos, dtype=float)
        self.target_quat = np.asarray(quat, dtype=float)
        ff = np.zeros(6)
        if vel is not None:
            ff[:3] = vel
        if ang_vel is not None:
            ff[3:] = ang_vel
        self.ff = ff.reshape(6, 1)

    def update(self, state):
        pe = self.target_pos - state.ee_pos
        oe = orientation_error(state.ee_quat, self.target_quat)
        self.err = np.concatenate([pe, oe]).reshape(6, 1)
        self.J = np.vstack([state.J_pos, state.J_orn])


# ----------------------------------------------------------------- the solver
def task_priority_step(tasks, state: ArmState, n_dof: int = 7,
                       damping: float = 0.05) -> np.ndarray:
    """Recursive task-priority resolution. Returns joint velocity dq (n_dof,)."""
    P = np.eye(n_dof)
    dq = np.zeros((n_dof, 1))
    for task in tasks:
        task.update(state)
        if not task.is_active():
            continue
        J, err, K, ff = task.J, task.err, task.K, task.ff
        xdot = K @ err + ff
        Jbar = J @ P
        dq = dq + DLS(Jbar, damping) @ (xdot - J @ dq)
        P = P - np.linalg.pinv(Jbar) @ Jbar     # project into remaining null-space
    return dq.flatten()


# ------------------------------------------------------------- the controller
class TaskPriorityController:
    """Stacks the tasks above and drives the arm via resolved-rate velocity."""

    def __init__(self, sim: PandaSim, track_orientation: bool = False,
                 combined: bool = False, q_home=None, damping: float = 0.05,
                 kp_pos: float = 4.0, kp_orn: float = 2.0, limit_margin: float = 0.10,
                 limit_gain: float = 1.0, center_gain: float = 0.3):
        """
        track_orientation : add a *secondary* orientation task below position.
        combined          : use one 6-DOF ConfigurationTask (position+orientation
                            solved together). Implies orientation tracking and
                            overrides track_orientation. Preferred for full-pose
                            targets.
        """
        self.sim = sim
        n = sim.n_arm
        q_home = PANDA_HOME if q_home is None else q_home

        self.limit_tasks = [
            JointLimitTask(i, sim.joint_lower[i], sim.joint_upper[i],
                           margin=limit_margin, gain=limit_gain, n_dof=n)
            for i in range(n)
        ]
        self.combined = combined
        if combined:
            self.config_task = ConfigurationTask(kp_pos=kp_pos, kp_orn=kp_orn, n_dof=n)
            self.pos_task = self.orn_task = None
        else:
            self.config_task = None
            self.pos_task = PositionTask(kp=kp_pos, n_dof=n)
            self.orn_task = OrientationTask(kp=kp_orn, n_dof=n) if track_orientation else None
        self.center_task = CenteringTask(q_home, gain=center_gain, n_dof=n)
        self.damping = damping

    def _snapshot(self) -> ArmState:
        q, dq = self.sim.get_joint_state()
        ee_pos, ee_quat = self.sim.get_ee_pose()
        Jp, Jo = self.sim.get_jacobian()
        return ArmState(q, dq, ee_pos, ee_quat, Jp, Jo)

    @property
    def task_stack(self):
        stack = list(self.limit_tasks)
        if self.combined:
            stack.append(self.config_task)
        else:
            stack.append(self.pos_task)
            if self.orn_task is not None:
                stack.append(self.orn_task)
        stack.append(self.center_task)
        return stack

    def compute(self, target_pos, target_vel=None, target_quat=None):
        """Solve for the joint-velocity command WITHOUT commanding the arm.

        Returns (dq, pos_error_vec, orn_error_vec|None, active_limits). Used by
        the RL env, which adds a residual before commanding.
        """
        if self.combined:
            self.config_task.set_target(target_pos, target_quat, vel=target_vel)
        else:
            self.pos_task.set_target(target_pos, target_vel)
            if self.orn_task is not None and target_quat is not None:
                self.orn_task.set_target(target_quat)

        state = self._snapshot()
        dq = task_priority_step(self.task_stack, state,
                                n_dof=self.sim.n_arm, damping=self.damping)
        pos_err = target_pos - state.ee_pos
        orn_err = (orientation_error(state.ee_quat, target_quat)
                   if target_quat is not None else None)
        active_limits = [t.name for t in self.limit_tasks if t.is_active()]
        return dq, pos_err, orn_err, active_limits

    def step(self, target_pos, target_vel=None, target_quat=None):
        """One control tick: compute and command the arm."""
        dq, pos_err, orn_err, active_limits = self.compute(
            target_pos, target_vel, target_quat)
        self.sim.set_arm_velocities(dq)
        return dq, pos_err, orn_err, active_limits
