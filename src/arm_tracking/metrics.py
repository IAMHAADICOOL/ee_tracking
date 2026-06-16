"""Tracking-accuracy and motion-smoothness metrics.

These quantify the two things the project is judged on — *how closely* the EE
follows the path, and *how smoothly* it moves — so claims about smoothness are
measured, not asserted.

Accuracy
--------
rms_error / max_error : Euclidean EE-to-target distance (metres).

Smoothness
----------
ee_jerk_rms   : RMS magnitude of the EE's 3rd position derivative (m/s^3).
                Lower = smoother Cartesian motion.
joint_jerk_rms: RMS of the joints' 3rd derivative (rad/s^3). Catches arm
                jitter that may not show up at the EE.
command_jitter: RMS of consecutive policy-action differences. A direct read on
                how "twitchy" the controller's output is.
sparc         : Spectral Arc Length of the EE speed profile (dimensionless,
                negative). A standard, amplitude/duration-robust smoothness
                measure (Balasubramanian et al., 2015). LESS negative = smoother
                (a single clean speed bump ~ -1.5; jittery motion is more
                negative, e.g. -4 or lower).

All functions take plain numpy arrays so they can run on logged rollouts from
watch_policy or the Phase-6 eval harness without any sim/torch dependency.
"""
from __future__ import annotations

import numpy as np


def rms_error(errors) -> float:
    e = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")


def max_error(errors) -> float:
    e = np.asarray(errors, dtype=float)
    return float(np.max(e)) if e.size else float("nan")


def _nth_derivative(x: np.ndarray, dt: float, n: int) -> np.ndarray:
    """n-th time derivative along axis 0 by repeated finite differencing."""
    d = np.asarray(x, dtype=float)
    for _ in range(n):
        d = np.diff(d, axis=0) / dt
    return d


def ee_jerk_rms(positions, dt: float) -> float:
    """RMS magnitude of EE jerk (m/s^3) from a (N,3) position log."""
    p = np.asarray(positions, dtype=float)
    if p.ndim != 2 or p.shape[0] < 4:
        return float("nan")
    jerk = _nth_derivative(p, dt, 3)           # (N-3, 3)
    return float(np.sqrt(np.mean(np.sum(jerk ** 2, axis=1))))


def joint_jerk_rms(q_log, dt: float) -> float:
    """RMS joint jerk (rad/s^3) from a (N, n_joints) angle log."""
    q = np.asarray(q_log, dtype=float)
    if q.ndim != 2 or q.shape[0] < 4:
        return float("nan")
    jerk = _nth_derivative(q, dt, 3)           # (N-3, n_joints)
    return float(np.sqrt(np.mean(jerk ** 2)))


def command_jitter(actions) -> float:
    """RMS of consecutive action differences (twitchiness of the policy output)."""
    a = np.asarray(actions, dtype=float)
    if a.ndim != 2 or a.shape[0] < 2:
        return float("nan")
    da = np.diff(a, axis=0)
    return float(np.sqrt(np.mean(np.sum(da ** 2, axis=1))))


def sparc(speed, fs: float, padlevel: int = 4,
          fc: float = 10.0, amp_th: float = 0.05) -> float:
    """Spectral Arc Length of a 1-D speed profile.

    Parameters
    ----------
    speed : 1-D array of speed magnitudes (e.g. ||EE velocity||).
    fs    : sampling frequency (Hz) = 1 / control_dt.
    fc    : max frequency considered (Hz).
    amp_th: normalized-spectrum amplitude threshold.

    Returns a negative number; LESS negative = smoother. NaN if the profile is
    too short or essentially still.
    """
    v = np.asarray(speed, dtype=float)
    if v.size < 8 or np.allclose(v, 0.0):
        return float("nan")

    nfft = int(2 ** (np.ceil(np.log2(len(v))) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    Mf = np.abs(np.fft.fft(v, nfft))
    peak = np.max(Mf)
    if peak <= 0:
        return float("nan")
    Mf = Mf / peak

    sel = f <= fc
    f_sel, Mf_sel = f[sel], Mf[sel]

    above = np.where(Mf_sel >= amp_th)[0]
    if above.size < 2:
        return float("nan")
    rng = slice(above[0], above[-1] + 1)
    f_sel, Mf_sel = f_sel[rng], Mf_sel[rng]

    df = np.diff(f_sel) / (f_sel[-1] - f_sel[0])
    dM = np.diff(Mf_sel)
    return float(-np.sum(np.sqrt(df ** 2 + dM ** 2)))


def summarize(errors, positions, q_log, dt, actions=None,
              settle_steps: int = 90) -> dict:
    """Compute all metrics for one rollout. `settle_steps` excludes the initial
    transient (arm moving onto the path) from the steady-state accuracy figure."""
    errors = np.asarray(errors, dtype=float)
    positions = np.asarray(positions, dtype=float)
    s = min(settle_steps, max(0, len(errors) - 1))
    speed = (np.linalg.norm(_nth_derivative(positions, dt, 1), axis=1)
             if len(positions) >= 2 else np.array([0.0]))
    out = {
        "rmse_mm": rms_error(errors) * 1e3,
        "rmse_ss_mm": rms_error(errors[s:]) * 1e3,
        "max_mm": max_error(errors) * 1e3,
        "ee_jerk_rms": ee_jerk_rms(positions, dt),
        "joint_jerk_rms": joint_jerk_rms(q_log, dt),
        "sparc": sparc(speed, fs=1.0 / dt),
    }
    if actions is not None:
        out["command_jitter"] = command_jitter(actions)
    return out
