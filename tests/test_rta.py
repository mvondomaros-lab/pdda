import numpy as np

from pdda.rta import diffusivity, exit_times, residence_time


def _trajectory_single_inside_run(run_len: int, xmin: float, xmax: float) -> np.ndarray:
    """
    Construct a trajectory that is outside Ω, then inside Ω for run_len frames, then outside again.

    This yields exactly one inside segment of length run_len.

    :param run_len: Length of the inside run in frames (>= 1).
    :param xmin: Lower bound of Ω (inclusive).
    :param xmax: Upper bound of Ω (exclusive).
    :returns: Trajectory x[k].
    """
    if run_len < 1:
        raise ValueError("run_len must be >= 1")

    x_inside = 0.5 * (xmin + xmax)
    x_out = xmin - 1.0

    x = np.empty(run_len + 2, dtype=np.float64)
    x[0] = x_out
    x[1 : 1 + run_len] = x_inside
    x[-1] = x_out
    return x


def test_exit_times_simple_segments() -> None:
    # Two inside segments in Ω = [0, 1): segment lengths 3 and 2.
    x = np.array([-0.1, 0.2, 0.8, 0.9, 1.2, 0.5, 0.6, -0.2], dtype=np.float64)
    xmin, xmax = 0.0, 1.0
    dt = 0.1

    times = exit_times(x, xmin, xmax, dt)
    expected = np.array([0.3, 0.2, 0.1, 0.2, 0.1], dtype=np.float64)

    assert times.dtype == np.float64
    assert times.shape == expected.shape
    assert np.allclose(times, expected)


def test_exit_times_never_inside() -> None:
    # All points are outside Ω = [0, 1).
    x = np.array([-2.0, -1.0, 1.5, 2.0], dtype=np.float64)
    xmin, xmax = 0.0, 1.0
    dt = 0.5

    times = exit_times(x, xmin, xmax, dt)
    assert times.size == 0


def test_exit_times_all_inside() -> None:
    x = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    xmin, xmax = 0.0, 1.0
    dt = 2.0

    times = exit_times(x, xmin, xmax, dt)

    # With N=5 frames all inside, final segment length is 5 and exit times are 5..1 times dt.
    expected = np.array([5, 4, 3, 2, 1], dtype=np.float64) * dt
    assert np.allclose(times, expected)


def test_residence_time_mean_matches_exit_times_mean() -> None:
    xmin, xmax = 0.0, 1.0
    dt = 0.1
    run_len = 64

    x = _trajectory_single_inside_run(run_len=run_len, xmin=xmin, xmax=xmax)
    times = exit_times(x, xmin, xmax, dt)

    tau, tau_sem = residence_time(x, xmin, xmax, dt)

    assert np.isfinite(tau)
    assert np.isclose(tau, float(times.mean()))
    assert (np.isfinite(tau_sem) and tau_sem >= 0.0) or np.isnan(tau_sem)


def test_residence_time_long_single_segment_has_finite_sem() -> None:
    xmin, xmax = 0.0, 1.0
    dt = 0.1
    run_len = 256  # convenient for blocking

    x = _trajectory_single_inside_run(run_len=run_len, xmin=xmin, xmax=xmax)
    times = exit_times(x, xmin, xmax, dt)

    expected = np.arange(run_len, 0, -1, dtype=np.float64) * dt
    assert np.allclose(times, expected)

    tau, tau_sem = residence_time(x, xmin, xmax, dt)

    assert np.isfinite(tau)
    assert np.isclose(tau, float(times.mean()))
    assert np.isfinite(tau_sem)
    assert tau_sem >= 0.0


def test_diffusivity_matches_formula_and_sem_propagation() -> None:
    xmin, xmax = 0.0, 1.0
    dt = 0.1
    run_len = 256

    x = _trajectory_single_inside_run(run_len=run_len, xmin=xmin, xmax=xmax)
    tau, tau_sem = residence_time(x, xmin, xmax, dt)

    D_est, D_sem = diffusivity(x, xmin, xmax, dt)

    L = xmax - xmin
    expected = (L * L) / (12.0 * tau)

    assert np.isfinite(D_est)
    assert np.isclose(D_est, expected)

    if np.isfinite(tau_sem):
        expected_sem = (L * L) / 12.0 * tau_sem / (tau * tau)
        assert np.isfinite(D_sem)
        assert np.isclose(D_sem, expected_sem)
    else:
        assert np.isnan(D_sem)


def test_diffusivity_returns_nan_when_never_visits_interval() -> None:
    x = np.array([-2.0, -1.0, -0.5, -0.1], dtype=np.float64)
    xmin, xmax = 0.0, 1.0
    dt = 0.1

    D_est, D_sem = diffusivity(x, xmin, xmax, dt)
    assert np.isnan(D_est)
    assert np.isnan(D_sem)
