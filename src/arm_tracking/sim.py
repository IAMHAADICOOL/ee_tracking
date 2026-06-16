"""PyBullet Franka Panda wrapper.

Thin, dependency-light interface around a position-controlled Panda arm.
Everything the controller / RL env needs (joint states, EE pose, Jacobian,
camera frames) lives here so the rest of the codebase never touches the raw
PyBullet API directly.
"""
from __future__ import annotations

import numpy as np
import pybullet as p
import pybullet_data


# Classic Panda "ready" posture (7 arm joints), radians.
PANDA_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# Per-joint torque limits used as position-control max forces.
PANDA_MAX_FORCE = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])


class PandaSim:
    """A fixed-base Franka Panda controlled in joint-position mode."""

    def __init__(self, gui: bool = False, timestep: float = 1.0 / 240.0,
                 gravity: float = -9.81):
        self.timestep = timestep
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, gravity, physicsClientId=self.client)
        p.setTimeStep(timestep, physicsClientId=self.client)

        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.robot = p.loadURDF(
            "franka_panda/panda.urdf", basePosition=[0, 0, 0],
            useFixedBase=True, physicsClientId=self.client,
        )

        # Discover joints/links rather than hard-coding indices.
        self.arm_joints: list[int] = []      # 7 revolute arm joints
        self.movable_joints: list[int] = []  # revolute + prismatic (Jacobian DOFs)
        self.ee_link = None
        for j in range(p.getNumJoints(self.robot, physicsClientId=self.client)):
            info = p.getJointInfo(self.robot, j, physicsClientId=self.client)
            jtype = info[2]
            link_name = info[12].decode("utf-8")
            if jtype == p.JOINT_REVOLUTE:
                self.arm_joints.append(j)
                self.movable_joints.append(j)
            elif jtype == p.JOINT_PRISMATIC:
                self.movable_joints.append(j)
            if link_name == "panda_grasptarget":
                self.ee_link = j
        self.arm_joints = self.arm_joints[:7]
        if self.ee_link is None:  # fallback if URDF naming differs
            self.ee_link = self.arm_joints[-1]

        self.n_arm = len(self.arm_joints)
        # Index of each arm joint within the movable-joint list (for Jacobian slicing).
        self._arm_in_movable = [self.movable_joints.index(j) for j in self.arm_joints]

        # Joint limits for the arm.
        lows, highs = [], []
        for j in self.arm_joints:
            info = p.getJointInfo(self.robot, j, physicsClientId=self.client)
            lows.append(info[8])
            highs.append(info[9])
        self.joint_lower = np.array(lows)
        self.joint_upper = np.array(highs)

        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self, q: np.ndarray | None = None):
        q = PANDA_HOME if q is None else np.asarray(q)
        for idx, j in enumerate(self.arm_joints):
            p.resetJointState(self.robot, j, float(q[idx]), 0.0,
                              physicsClientId=self.client)
        # Keep fingers closed and still.
        for j in self.movable_joints:
            if j not in self.arm_joints:
                p.resetJointState(self.robot, j, 0.02, 0.0, physicsClientId=self.client)
        return self.get_joint_state()

    def get_joint_state(self):
        states = p.getJointStates(self.robot, self.arm_joints,
                                  physicsClientId=self.client)
        q = np.array([s[0] for s in states])
        dq = np.array([s[1] for s in states])
        return q, dq

    def get_ee_pose(self):
        ls = p.getLinkState(self.robot, self.ee_link, computeForwardKinematics=True,
                            physicsClientId=self.client)
        pos = np.array(ls[4])   # worldLinkFramePosition
        orn = np.array(ls[5])   # worldLinkFrameOrientation (xyzw quaternion)
        return pos, orn

    def get_jacobian(self):
        """Return (J_pos, J_orn), each 3 x n_arm, at the current configuration."""
        q_all, dq_all = [], []
        states = p.getJointStates(self.robot, self.movable_joints,
                                  physicsClientId=self.client)
        q_all = [s[0] for s in states]
        zeros = [0.0] * len(self.movable_joints)
        lin, ang = p.calculateJacobian(
            self.robot, self.ee_link, [0.0, 0.0, 0.0],
            q_all, zeros, zeros, physicsClientId=self.client,
        )
        lin = np.array(lin)[:, self._arm_in_movable]
        ang = np.array(ang)[:, self._arm_in_movable]
        return lin, ang

    # ---------------------------------------------------------------- control
    def set_arm_targets(self, q_des: np.ndarray):
        q_des = np.clip(q_des, self.joint_lower, self.joint_upper)
        p.setJointMotorControlArray(
            self.robot, self.arm_joints, p.POSITION_CONTROL,
            targetPositions=q_des.tolist(),
            forces=PANDA_MAX_FORCE.tolist(),
            physicsClientId=self.client,
        )

    def set_arm_velocities(self, dq: np.ndarray):
        """Resolved-rate velocity control: directly track a joint-velocity command."""
        p.setJointMotorControlArray(
            self.robot, self.arm_joints, p.VELOCITY_CONTROL,
            targetVelocities=np.asarray(dq, dtype=float).tolist(),
            forces=PANDA_MAX_FORCE.tolist(),
            physicsClientId=self.client,
        )

    def step(self):
        p.stepSimulation(physicsClientId=self.client)

    # ----------------------------------------------------------------- render
    def render(self, width: int = 640, height: int = 480,
               cam_target=(0.4, 0.0, 0.5), distance: float = 1.4,
               yaw: float = 50, pitch: float = -30, use_opengl: bool = False):
        view = p.computeViewMatrixFromYawPitchRoll(
            cam_target, distance, yaw, pitch, 0, 2, physicsClientId=self.client)
        proj = p.computeProjectionMatrixFOV(
            60, width / height, 0.1, 3.1, physicsClientId=self.client)
        # TINY is software (works headless); OpenGL is GPU-fast (needs a GUI ctx).
        renderer = p.ER_BULLET_HARDWARE_OPENGL if use_opengl else p.ER_TINY_RENDERER
        _, _, rgb, _, _ = p.getCameraImage(
            width, height, view, proj,
            renderer=renderer, physicsClientId=self.client)
        rgb = np.reshape(np.array(rgb, dtype=np.uint8), (height, width, 4))
        return rgb[:, :, :3]

    def is_connected(self) -> bool:
        """True while the physics client (and, in GUI mode, the window) is open."""
        return bool(p.isConnected(self.client))

    def setup_gui_camera(self, distance: float = 1.0, yaw: float = 90,
                         pitch: float = -25, target=(0.3, 0.0, 0.48)):
        """Frame the on-screen GUI camera on the arm workspace.

        Only meaningful when connected with gui=True. The user can still orbit
        with the mouse; this just sets a sensible initial view. The standard
        PyBullet side panels are left visible.
        """
        p.resetDebugVisualizerCamera(distance, yaw, pitch, list(target),
                                     physicsClientId=self.client)

    def start_video_log(self, path: str):
        """Record the live GUI window to an MP4 via PyBullet's own logger.

        Stable in GUI mode (unlike per-frame getCameraImage on some GL drivers)
        and captures exactly what is shown in the window. Requires system ffmpeg
        on PATH; if absent, PyBullet warns and no file is produced.
        Returns a log id to pass to stop_video_log().
        """
        return p.startStateLogging(
            p.STATE_LOGGING_VIDEO_MP4, path, physicsClientId=self.client)

    def stop_video_log(self, log_id):
        try:
            p.stopStateLogging(log_id, physicsClientId=self.client)
        except Exception:
            pass   # window may already be gone; PyBullet finalises on disconnect

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(physicsClientId=self.client)

    # ------------------------------------------------- visual debug markers
    # Massless, collision-free spheres that render into getCameraImage frames
    # (unlike addUserDebugLine, which only shows in the GUI overlay).
    def _sphere_shape(self, radius, rgba):
        if not hasattr(self, "_vshape_cache"):
            self._vshape_cache = {}
        key = (round(radius, 4), tuple(rgba))
        if key not in self._vshape_cache:
            self._vshape_cache[key] = p.createVisualShape(
                p.GEOM_SPHERE, radius=radius, rgbaColor=list(rgba),
                physicsClientId=self.client)
        return self._vshape_cache[key]

    def add_marker(self, pos, rgba=(1, 0, 0, 1), radius=0.012):
        """Create a single visual-only sphere; returns its body id (movable)."""
        vs = self._sphere_shape(radius, rgba)
        return p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs,
            basePosition=list(pos), physicsClientId=self.client)

    def move_marker(self, body_id, pos):
        p.resetBasePositionAndOrientation(
            body_id, list(pos), [0, 0, 0, 1], physicsClientId=self.client)

    def remove_body(self, body_id):
        try:
            p.removeBody(body_id, physicsClientId=self.client)
        except Exception:
            pass

    def draw_path(self, points, rgba=(0.7, 0.7, 0.7, 1.0), radius=0.005):
        """Render a static polyline of small spheres tracing a path."""
        vs = self._sphere_shape(radius, rgba)
        for pt in points:
            p.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs,
                basePosition=list(pt), physicsClientId=self.client)

    def _rot(self, quat):
        return np.array(p.getMatrixFromQuaternion(list(quat))).reshape(3, 3)

    def add_triad(self, pos, quat, length: float = 0.07, radius: float = 0.007):
        """Three RGB axis-tip spheres showing an orientation; returns 3 body ids.

        Unlike addUserDebugLine, these are real bodies so they show in both the
        GUI window and captured camera frames. Tip i sits at pos + R[:,i]*length.
        """
        R = self._rot(quat)
        cols = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0.4, 1, 1)]   # x=red y=green z=blue
        return [self.add_marker(np.asarray(pos) + R[:, a] * length,
                                rgba=cols[a], radius=radius) for a in range(3)]

    def move_triad(self, ids, pos, quat, length: float = 0.07):
        R = self._rot(quat)
        for a in range(3):
            self.move_marker(ids[a], np.asarray(pos) + R[:, a] * length)
