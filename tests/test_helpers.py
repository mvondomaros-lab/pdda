import numpy as np
import pytest

from pdda.helpers import prevpow2, sem


def _ar1(rng: np.random.Generator, n: int, rho: float, sigma: float) -> np.ndarray:
    """
    Generate a stationary AR(1) time series:
        x_t = rho x_{t-1} + eps_t,   eps_t ~ N(0, sigma^2)
    """
    x = np.empty(n, dtype=np.float64)
    # Stationary variance = sigma^2 / (1 - rho^2)
    x[0] = rng.normal(scale=sigma / np.sqrt(1.0 - rho * rho))
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.normal(scale=sigma)
    return x


def test_prevpow2_basic_cases() -> None:
    assert prevpow2(0) == 1
    assert prevpow2(1) == 1
    assert prevpow2(2) == 2
    assert prevpow2(3) == 2
    assert prevpow2(4) == 4
    assert prevpow2(5) == 4
    assert prevpow2(15) == 8
    assert prevpow2(16) == 16
    assert prevpow2(17) == 16


def test_prevpow2_negative_and_float_cases() -> None:
    assert prevpow2(-17) == 16
    assert prevpow2(-1) == 1
    assert prevpow2(0.1) == 1
    assert prevpow2(-0.9) == 1
    assert prevpow2(31.9) == 16  # int(abs(x)) -> 31


@pytest.mark.parametrize("corr", [False, True])
def test_sem_rejects_non_1d(corr: bool) -> None:
    x = np.zeros((8, 2), dtype=np.float64)
    with pytest.raises(ValueError):
        sem(x, corr=corr)


def test_sem_iid_rejects_too_short() -> None:
    x = np.ones(1, dtype=np.float64)
    with pytest.raises(ValueError):
        sem(x, corr=False)


def test_sem_corr_rejects_too_short() -> None:
    x = np.ones(3, dtype=np.float64)
    with pytest.raises(ValueError):
        sem(x, corr=True)


@pytest.mark.parametrize("corr", [False, True])
def test_sem_rejects_non_finite(corr: bool) -> None:
    x = np.array([0.0, 1.0, np.nan, 2.0], dtype=np.float64)
    with pytest.raises(ValueError):
        sem(x, corr=corr)


@pytest.mark.parametrize("corr", [False, True])
def test_sem_constant_series_is_zero(corr: bool) -> None:
    x = np.ones(256, dtype=np.float64)
    sem_val = sem(x, corr=corr)
    assert sem_val == 0.0


def test_sem_corr_truncation_equivalence() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=5000).astype(np.float64)

    n = prevpow2(x.size)
    assert n == 4096  # sanity check

    sem_full = sem(x, corr=True)
    sem_trunc = sem(x[:n], corr=True)

    # corr=True truncates internally; these should match closely.
    assert np.isclose(sem_full, sem_trunc, rtol=0.0, atol=1e-12)


def test_sem_iid_matches_naive_ddof1() -> None:
    """
    For IID data, sem(..., corr=False) should match std(ddof=1)/sqrt(n).
    """
    rng = np.random.default_rng(7)
    n = 4096
    x = rng.normal(loc=0.0, scale=2.0, size=n).astype(np.float64)

    sem_val = float(sem(x, corr=False))
    naive = float(np.std(x, ddof=1) / np.sqrt(n))

    assert np.isfinite(sem_val)
    assert np.isfinite(naive)
    assert np.isclose(sem_val, naive, rtol=0.0, atol=1e-12)


def test_sem_corr_iid_reasonable_vs_naive() -> None:
    """
    For IID data, sem(..., corr=True) should be close to the naive SEM.
    We allow modest tolerance because the blocking estimator has finite-sample variability.
    """
    rng = np.random.default_rng(7)
    n = 4096  # power of two avoids truncation effects
    x = rng.normal(loc=0.0, scale=2.0, size=n).astype(np.float64)

    sem_block = float(sem(x, corr=True))
    naive = float(np.std(x, ddof=1) / np.sqrt(n))

    assert np.isfinite(sem_block)
    assert np.isfinite(naive)

    # Blocking SEM can be slightly larger; keep tolerance modest.
    assert np.isclose(sem_block, naive, rtol=0.15, atol=0.0)


def test_sem_corr_detects_correlation_ar1() -> None:
    """
    For strongly correlated AR(1), sem(..., corr=True) should exceed the naive IID SEM.
    """
    rng = np.random.default_rng(11)
    n = 8192

    x_iid = rng.normal(size=n).astype(np.float64)
    sem_iid = float(sem(x_iid, corr=True))
    naive_iid = float(np.std(x_iid, ddof=1) / np.sqrt(n))

    x_corr = _ar1(rng, n=n, rho=0.95, sigma=1.0)
    sem_corr_est = float(sem(x_corr, corr=True))
    naive_corr = float(np.std(x_corr, ddof=1) / np.sqrt(n))

    assert np.isfinite(sem_iid)
    assert np.isfinite(sem_corr_est)

    # On IID data, blocking SEM should be in the same ballpark as naive.
    assert np.isclose(sem_iid, naive_iid, rtol=0.15, atol=0.0)

    # On correlated data, blocking SEM should increase relative to naive.
    assert sem_corr_est > naive_corr
    assert sem_corr_est > sem_iid

