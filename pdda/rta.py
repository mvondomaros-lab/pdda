"""
Residence-time approach (RTA) for 1D diffusion in an interval.

This module estimates a diffusion constant from a discrete-time trajectory x[k]
sampled at timestep dt, using first-exit (absorbing-boundary) statistics in the
interval Ω = [xmin, xmax).

Key quantities (continuous time):
- Exit time T: time until the trajectory first leaves Ω.
- Mean first-exit time (MFET) τ_x: expected exit time when starting at position x.
- Residence time τ_Ω: MFET averaged over valid frames in Ω (for constant PMF,
  valid frames sample x uniformly in Ω).

For constant diffusivity D and constant PMF in Ω, the bulk RTA identity is:
    D = (xmax - xmin)^2 / (12 τ_Ω)

Terminology (discrete trajectory):
- Frame: one sample x[k] of the trajectory.
- Valid frame: a frame with x[k] ∈ Ω. Exit times are computed from all valid frames.
"""

from typing import Tuple

import numba
import numpy as np

from . import helpers


@numba.njit(cache=True, fastmath=True)
def _count_inside(x: np.ndarray, xmin: float, xmax: float) -> int:
    """
    Count valid frames, i.e., indices k with x[k] ∈ Ω.

    :param x: Trajectory samples x[k].
    :param xmin: Lower bound of Ω (inclusive).
    :param xmax: Upper bound of Ω (exclusive).
    :returns: Number of valid frames.
    """
    n_inside = 0
    for i in range(x.shape[0]):
        xi = x[i]
        if xi >= xmin and xi < xmax:
            n_inside += 1
    return n_inside


@numba.njit(cache=True, fastmath=True)
def exit_times(x: np.ndarray, xmin: float, xmax: float, dt: float) -> np.ndarray:
    """
    Compute first-exit times from Ω = [xmin, xmax) for all valid frames.

    Each exit time is measured from a valid frame k with x[k] ∈ Ω and represents
    the remaining time until the *first* exit from Ω (absorbing boundaries). Exit
    times are therefore conditional on starting inside Ω.

    Implementation:
      The trajectory is decomposed into contiguous segments of frames that lie in Ω.
      For a segment of length m (in frames), the emitted exit times are:
          m·dt, (m−1)·dt, …, 1·dt
      corresponding to starting from each frame in the segment.

    Performance:
      Two passes are used: (1) count valid frames to allocate the output array,
      (2) scan the trajectory to fill the exit-time samples.

    :param x: Trajectory samples x[k].
    :param xmin: Lower bound of Ω (inclusive).
    :param xmax: Upper bound of Ω (exclusive).
    :param dt: Time step between frames.
    :returns: Exit times (same units as dt), one per valid frame.
              Returns an empty array if the trajectory never visits Ω.
    """
    n_inside = _count_inside(x, xmin, xmax)
    if n_inside == 0:
        return np.empty(0, dtype=np.float64)

    out = np.empty(n_inside, dtype=np.float64)
    idx = 0
    run_len = 0

    for i in range(x.shape[0]):
        xi = x[i]
        inside = xi >= xmin and xi < xmax

        if inside:
            run_len += 1
        else:
            if run_len > 0:
                for k in range(run_len):
                    out[idx + k] = (run_len - k) * dt
                idx += run_len
                run_len = 0

    # Flush the final segment of valid frames (if the last samples lie in Ω).
    if run_len > 0:
        for k in range(run_len):
            out[idx + k] = (run_len - k) * dt
        idx += run_len

    return out


def residence_time(
    x: np.ndarray, xmin: float, xmax: float, dt: float
) -> Tuple[float, float]:
    """
    Compute the mean residence time τ_Ω in Ω = [xmin, xmax) and its SEM.

    τ_Ω is defined as the mean of the exit-time samples, averaged over all valid frames
    (all indices k with x[k] ∈ Ω).

    The SEM accounts for temporal correlation in the exit-time series using Jonsson’s
    automated blocking method.

    :param x: Trajectory samples x[k].
    :param xmin: Lower bound of Ω (inclusive).
    :param xmax: Upper bound of Ω (exclusive).
    :param dt: Time step between frames.
    :returns: (tau_omega, tau_omega_sem).
              Returns (NaN, NaN) if the trajectory never visits Ω.
              If the SEM cannot be estimated (e.g., insufficient data), returns SEM = NaN.
    """
    times = exit_times(x, xmin, xmax, dt)
    if times.size == 0:
        return float("nan"), float("nan")

    tau_omega = float(times.mean())
    try:
        tau_omega_sem = helpers.sem(times, corr=True)
    except ValueError:
        tau_omega_sem = float("nan")

    return tau_omega, tau_omega_sem


def diffusivity(
    x: np.ndarray,
    xmin: float,
    xmax: float,
    dt: float,
) -> Tuple[float, float]:
    """
    Estimate the bulk diffusivity in Ω = [xmin, xmax) and its SEM.

    Baseline estimator (constant PMF, constant D):
        D_est = L^2 / (12 τ_Ω)

    SEM is propagated linearly from τ_Ω:
        sem(D_est) ≈ (L^2/12) * sem(τ_Ω) / τ_Ω^2

    :param x: Trajectory samples x[k].
    :param xmin: Lower bound of Ω (inclusive).
    :param xmax: Upper bound of Ω (exclusive).
    :param dt: Time step between frames.
    :returns: (D_est, D_est_sem).
              Returns (NaN, NaN) if τ_Ω is undefined.
              If sem(τ_Ω) is undefined, returns SEM = NaN.
    """
    tau_omega, tau_omega_sem = residence_time(x, xmin, xmax, dt)
    if not np.isfinite(tau_omega) or tau_omega <= 0.0:
        return float("nan"), float("nan")

    L = float(xmax - xmin)
    pref = (L * L) / 12.0

    D_est = pref / tau_omega
    if not np.isfinite(tau_omega_sem):
        return float(D_est), float("nan")

    D_est_sem = pref * tau_omega_sem / (tau_omega * tau_omega)
    return float(D_est), float(D_est_sem)
