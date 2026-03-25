import numpy as np


def prevpow2(x: int | float) -> int:
    """
    Largest power of two less than or equal to abs(x).

    :param x: Input value.
    :return: Largest power of 2 <= abs(x). Returns 1 if abs(x) < 1.
    """
    n = int(abs(x))
    return 1 if n == 0 else 1 << (n.bit_length() - 1)


def _sem_iid(x: np.ndarray) -> float:
    """
    SEM of the mean assuming independent samples.

    Computes the standard error of the sample mean under the i.i.d. assumption:
        SEM = s / sqrt(n)
    where s is the sample standard deviation (ddof=1).

    :param x: 1D array of samples.
    :return: SEM of the sample mean (Python float).
    :raises ValueError: If x is not 1D, has <2 samples, or contains non-finite values.
    """
    if x.ndim != 1:
        raise ValueError("sem expects a 1D array")
    if x.size < 2:
        raise ValueError("need at least 2 samples for SEM estimation")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values")

    s = float(np.std(x, ddof=1))
    return float(s / np.sqrt(x.size))


def _sem_corr_blocking(x: np.ndarray) -> float:
    """
    SEM of the mean for correlated time series (automated blocking).

    Implements Jonsson's automated blocking method:
        https://doi.org/10.1103/PhysRevE.98.043304

    The input is truncated to the largest power-of-two length. If the time series
    is too short to determine a stable blocking level, a ValueError is raised.

    :param x: 1D array of samples (correlated time series).
    :return: SEM of the sample mean (Python float).
    :raises ValueError: If x is not 1D, contains non-finite values, or is too short.
    """
    if x.ndim != 1:
        raise ValueError("sem expects a 1D array")
    if x.size < 4:
        raise ValueError("need at least 4 samples for correlated SEM estimation")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values")

    n = prevpow2(x.size)
    if n < 4:
        raise ValueError("need more data for correlated SEM estimation")

    d = int(np.log2(n))
    x = np.asarray(x[:n], dtype=np.float64)

    q = np.array(
        [
            6.634897,
            9.210340,
            11.344867,
            13.276704,
            15.086272,
            16.811894,
            18.475307,
            20.090235,
            21.665994,
            23.209251,
            24.724970,
            26.216967,
            27.688250,
            29.141238,
            30.577914,
            31.999927,
            33.408664,
            34.805306,
            36.190869,
            37.566235,
            38.932173,
            40.289360,
            41.638398,
            42.979820,
            44.314105,
            45.641683,
            46.962942,
            48.278236,
            49.587884,
            50.892181,
        ],
        dtype=np.float64,
    )
    if d >= q.size:
        raise ValueError("need more data for correlated SEM estimation")

    s = np.zeros(d, dtype=np.float64)
    gamma = np.zeros(d, dtype=np.float64)

    # Pairwise-averaging preserves the mean for power-of-two lengths.
    mu = float(np.mean(x))

    for i in range(d):
        s[i] = np.var(x)  # ddof=0
        if s[i] == 0.0:
            return 0.0

        n_i = x.size
        gamma[i] = np.sum((x[:-1] - mu) * (x[1:] - mu)) / n_i
        x = 0.5 * (x[0::2] + x[1::2])

    weights = ((gamma / s) ** 2) * (2.0 ** np.arange(1, d + 1, dtype=np.float64)[::-1])
    m = np.cumsum(weights[::-1])[::-1]

    k = 0
    while k < d and m[k] >= q[k]:
        k += 1
    if k >= d - 1:
        raise ValueError("need more data for correlated SEM estimation")

    return float(np.sqrt(s[k] / (2.0 ** (d - k))))


def sem(x: np.ndarray, *, corr: bool = False) -> float:
    """
    Standard error of the mean (SEM).

    If corr=False (default), assumes independent samples:
        SEM = s / sqrt(n)

    If corr=True, accounts for temporal correlation using Jonsson's automated
    blocking method:
        https://doi.org/10.1103/PhysRevE.98.043304

    :param x: 1D array of samples.
    :param corr: If True, estimate SEM for correlated data (blocking). If False, i.i.d. SEM.
    :return: SEM of the sample mean (Python float).
    :raises ValueError: If x is invalid or (corr=True) there is insufficient data.
    """
    x = np.asarray(x, dtype=np.float64)
    return _sem_corr_blocking(x) if corr else _sem_iid(x)

