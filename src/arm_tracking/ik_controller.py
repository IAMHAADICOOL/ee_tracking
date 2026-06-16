"""Model-based Cartesian tracker: damped least-squares resolved-rate control.

This is the *nominal* controller. It uses the arm's Jacobian to convert a
desired EE velocity (feed-forward path velocity + proportional position error)
into a joint-velocity command, integrates it to a joint-position target, and
regularises the 7-DOF redundancy with a null-space pull toward a home posture
(keeps the elbow from drifting and motion smooth).

In Phase 1 it is the whole system. From Phase 3 on it becomes the baseline the
residual RL policy corrects: q_des = ik_step(...) + policy_residual.
"""
from __future__ import annotations

import numpy as np

from .sim import PandaSim, PANDA_HOME


class DLSController:
    def __init__(self, sim: PandaSim, kp: float = 4.0, damping: float = 0.05,
                 null_gain: float = 0.5, q_home: np.ndarray | None = None):
        self.sim = sim
        self.kp = kp                # proportional gain on Cartesian error
        self.damping = damping      # DLS damping lambda
        self.null_gain = null_gain  # null-space posture gain
        self.q_home = PANDA_HOME if q_home is None else np.asarray(q_home)
        self.dt = sim.timestep

    def joint_velocity(self, target_pos, target_vel=None):
        """Compute the DLS joint-velocity command for a Cartesian target."""
        q, _ = self.sim.get_joint_state()
        ee_pos, _ = self.sim.get_ee_pose()
        J, _ = self.sim.get_jacobian()  # 3 x n_arm position Jacobian

        err = np.asarray(target_pos) - ee_pos
        xdot = self.kp * err
        if target_vel is not None:
            xdot = xdot + np.asarray(target_vel)  # feed-forward path velocity

        # Damped least-squares pseudo-inverse: J^T (J J^T + lambda^2 I)^-1
        n = J.shape[0]
        JJt = J @ J.T + (self.damping ** 2) * np.eye(n)
        J_dls = J.T @ np.linalg.inv(JJt)

        dq = J_dls @ xdot
        # Null-space posture regularisation (redundancy resolution).
        null_proj = np.eye(self.sim.n_arm) - J_dls @ J
        dq = dq + null_proj @ (self.null_gain * (self.q_home - q))
        return dq, err

    def step(self, target_pos, target_vel=None):
        """Advance one control tick; returns (dq_command, position_error_vector)."""
        dq, err = self.joint_velocity(target_pos, target_vel)
        self.sim.set_arm_velocities(dq)
        return dq, err
