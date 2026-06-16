import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["prevpow2", "sem", ]


def prevpow2(x: int | float) -> int:
    """
    Return the largest power of two less than or equal to ``abs(x)``.

    Parameters
    ----------
    x
        Input value.

    Returns
    -------
    int
        Largest power of two less than or equal to ``abs(x)``. Returns ``1`` if
        ``abs(x) < 1``.
    """
    n = int(abs(x))
    return 1 if n == 0 else 1 << (n.bit_length() - 1)


def sem(x: ArrayLike, *, corr: bool = False) -> float:
    """
    Compute the standard error of the mean (SEM).

    If ``corr=False``, assumes independent samples and estimates ``SEM = s / sqrt(n)``.
    If ``corr=True``, accounts for temporal correlation using Jonsson's automated
    blocking method.

    Parameters
    ----------
    x
        One-dimensional array of samples.
    corr
        If ``True``, estimate SEM for correlated data using blocking. If ``False``,
        estimate the i.i.d. SEM.

    Returns
    -------
    float
        Standard error of the sample mean.

    Raises
    ------
    ValueError
        If ``x`` is invalid, contains non-finite values, or, when ``corr=True``,
        contains insufficient data for blocking.
    """
    x = np.asarray(x, dtype=np.float64)
    return _sem_corr_blocking(x) if corr else _sem_iid(x)


def _sem_iid(x: NDArray[np.float64]) -> float:
    """SEM for i.i.d. samples."""
    if x.ndim != 1:
        raise ValueError("sem expects a 1D array")
    if x.size < 2:
        raise ValueError("need at least 2 samples for SEM estimation")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values")

    s = float(np.std(x, ddof=1))
    return float(s / np.sqrt(x.size))


def _sem_corr_blocking(x: NDArray[np.float64]) -> float:
    """SEM via blocking."""
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
        [6.634897, 9.210340, 11.344867, 13.276704, 15.086272, 16.811894, 18.475307, 20.090235, 21.665994, 23.209251,
            24.724970, 26.216967, 27.688250, 29.141238, 30.577914, 31.999927, 33.408664, 34.805306, 36.190869,
            37.566235, 38.932173, 40.289360, 41.638398, 42.979820, 44.314105, 45.641683, 46.962942, 48.278236,
            49.587884, 50.892181, ], dtype=np.float64, )
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
