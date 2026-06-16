"""Time-varying Cartesian trajectories for the end-effector to track.

Each trajectory is a callable phase -> (position, velocity). We parametrise by
*phase* (radians) rather than wall-clock time so the same object works at any
control rate and any traversal speed. The env advances phase by omega * dt.

All trajectories are defined relative to a `center` so we can anchor them on a
configuration we know is reachable (e.g. the home-pose EE position).
"""
from __future__ import annotations

import numpy as np


class Trajectory:
    name = "base"

    def __init__(self, center, omega: float = 0.6):
        self.center = np.asarray(center, dtype=float)
        self.omega = float(omega)  # phase rate (rad/s)

    def __call__(self, phase: float):
        raise NotImplementedError

    def position(self, phase: float):
        return self(phase)[0]


class Circle(Trajectory):
    """Vertical circle in the y-z plane (faces the default camera)."""
    name = "circle"

    def __init__(self, center, radius: float = 0.15, omega: float = 0.6):
        super().__init__(center, omega)
        self.r = radius

    def __call__(self, phase):
        r, c = self.r, self.center
        pos = c + np.array([0.0, r * np.cos(phase), r * np.sin(phase)])
        vel = self.omega * np.array([0.0, -r * np.sin(phase), r * np.cos(phase)])
        return pos, vel


class FigureEight(Trajectory):
    """Gerono lemniscate in the y-z plane: a smooth figure-eight."""
    name = "figure_eight"

    def __init__(self, center, width: float = 0.18, height: float = 0.12,
                 omega: float = 0.6):
        super().__init__(center, omega)
        self.a = width
        self.b = height

    def __call__(self, phase):
        a, b, c = self.a, self.b, self.center
        pos = c + np.array([0.0, a * np.sin(phase), b * np.sin(phase) * np.cos(phase)])
        vel = self.omega * np.array([
            0.0,
            a * np.cos(phase),
            b * (np.cos(phase) ** 2 - np.sin(phase) ** 2),
        ])
        return pos, vel


class MovingTarget(Trajectory):
    """Smooth pseudo-random target: sum of a few incommensurate sinusoids.

    Produces a non-repeating but C-infinity path that stays inside a box, good
    for stress-testing tracking on something other than a closed curve.
    """
    name = "moving_target"

    def __init__(self, center, extent: float = 0.16, omega: float = 0.6,
                 seed: int = 0):
        super().__init__(center, omega)
        rng = np.random.default_rng(seed)
        # 3 axes, each a sum of 3 sinusoids with random freq/phase.
        self.freqs = rng.uniform(0.5, 1.5, size=(3, 3))
        self.phaseoff = rng.uniform(0, 2 * np.pi, size=(3, 3))
        self.amps = rng.uniform(0.4, 1.0, size=(3, 3))
        self.amps /= self.amps.sum(axis=1, keepdims=True)  # normalise per-axis
        self.extent = extent

    def __call__(self, phase):
        f, ph, A = self.freqs, self.phaseoff, self.amps
        offsets, vels = [], []
        for ax in range(3):
            s = np.sum(A[ax] * np.sin(f[ax] * phase + ph[ax]))
            d = np.sum(A[ax] * f[ax] * np.cos(f[ax] * phase + ph[ax]))
            offsets.append(s)
            vels.append(d)
        # x kept shallow so the path stays near the reachable shell.
        scale = np.array([0.4, 1.0, 1.0]) * self.extent
        pos = self.center + scale * np.array(offsets)
        vel = self.omega * scale * np.array(vels)
        return pos, vel


def make_trajectory(name: str, center, **kwargs) -> Trajectory:
    table = {"circle": Circle, "figure_eight": FigureEight,
             "moving_target": MovingTarget}
    if name not in table:
        raise ValueError(f"unknown trajectory '{name}', choose from {list(table)}")
    return table[name](center, **kwargs)
