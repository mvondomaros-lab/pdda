"""Statistical uncertainty estimators used throughout PDDA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["StandardErrorEstimator", "StandardErrorResult"]

StandardErrorMethod = Literal["iid", "blocking"]


@dataclass(frozen=True)
class StandardErrorResult:
    """A standard error and the number of samples used to calculate it."""

    value: float
    method: StandardErrorMethod
    n_input: int
    n_used: int
    blocking_level: int | None


class StandardErrorEstimator:
    """Estimate uncertainty in a one-dimensional series of values.

    Use ``method="iid"`` for independent samples. Use ``method="blocking"``
    for a time series with correlation between adjacent samples.
    """

    def __init__(self, *, method: StandardErrorMethod = "iid") -> None:
        if method not in ("iid", "blocking"):
            raise ValueError("method must be 'iid' or 'blocking'")
        self.method = method

    def estimate(self, samples: ArrayLike) -> StandardErrorResult:
        """Estimate the SEM for one finite one-dimensional sample series."""
        values = _validate_samples(samples)
        if self.method == "iid":
            if values.size < 2:
                raise ValueError("need at least 2 samples for SEM estimation")
            return StandardErrorResult(
                value=float(np.std(values, ddof=1) / np.sqrt(values.size)),
                method=self.method,
                n_input=values.size,
                n_used=values.size,
                blocking_level=None,
            )
        return _blocking_standard_error(values)


def _validate_samples(samples: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("samples must be a one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must contain only finite values")
    return values


_BLOCKING_CRITICAL_VALUES = np.array(
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
    ]
)


def _blocking_standard_error(values: NDArray[np.float64]) -> StandardErrorResult:
    if values.size < 4:
        raise ValueError("need at least 4 samples for blocking SEM estimation")
    n_input = values.size
    n_used = 1 << (n_input.bit_length() - 1)
    values = values[:n_used].copy()
    depth = int(np.log2(n_used))
    if depth >= _BLOCKING_CRITICAL_VALUES.size:
        raise ValueError("too many samples for the supported blocking table")
    variances = np.empty(depth)
    autocovariances = np.empty(depth)
    mean = float(values.mean())
    for level in range(depth):
        variances[level] = np.var(values)
        if variances[level] == 0.0:
            return StandardErrorResult(0.0, "blocking", n_input, n_used, level)
        autocovariances[level] = (
            np.sum((values[:-1] - mean) * (values[1:] - mean)) / values.size
        )
        values = 0.5 * (values[0::2] + values[1::2])
    return _blocking_result(variances, autocovariances, n_input, n_used)


def _blocking_result(
    variances: NDArray[np.float64],
    autocovariances: NDArray[np.float64],
    n_input: int,
    n_used: int,
) -> StandardErrorResult:
    depth = variances.size
    weights = (autocovariances / variances) ** 2 * 2.0 ** np.arange(depth, 0, -1)
    statistic = np.cumsum(weights[::-1])[::-1]
    level = 0
    while level < depth and statistic[level] >= _BLOCKING_CRITICAL_VALUES[level]:
        level += 1
    if level >= depth - 1:
        raise ValueError("insufficient data for a stable blocking SEM estimate")
    return StandardErrorResult(
        float(np.sqrt(variances[level] / 2.0 ** (depth - level))),
        "blocking",
        n_input,
        n_used,
        level,
    )


def _blocking_exit_time_standard_error(
    lengths: NDArray[np.int64],
) -> StandardErrorResult:
    """Block ordered exit-time ramps without expanding every time origin.

    A segment of length L represents L, L-1, ..., 1. Arithmetic runs remain
    arithmetic after pair averaging; pairs crossing run boundaries become
    singleton runs. Centered closed-form moments preserve the same truncation,
    adjacent covariance, and blocking-level selection as the dense estimator.
    Values and the returned SEM are in frames.
    """
    n_input = int(lengths.sum())
    if n_input < 4:
        raise ValueError("need at least 4 samples for blocking SEM estimation")
    n_used = 1 << (n_input.bit_length() - 1)
    depth = n_used.bit_length() - 1
    if depth >= _BLOCKING_CRITICAL_VALUES.size:
        raise ValueError("too many samples for the supported blocking table")
    ends = np.cumsum(lengths)
    last = int(np.searchsorted(ends, n_used))
    counts = lengths[: last + 1].copy()
    first = counts.astype(float)
    counts[-1] -= int(ends[last]) - n_used
    steps = np.full(counts.size, -1.0)
    mean = float(np.sum(counts * (first - (counts - 1) / 2)) / n_used)
    variances = np.empty(depth)
    covariances = np.empty(depth)
    dense = None
    for level in range(depth):
        size = n_used >> level
        # When runs become short, ordinary vectorized blocking is cheaper.
        if dense is None and size <= 2 * counts.size:
            offsets = np.repeat(np.cumsum(counts) - counts, counts)
            dense = np.repeat(first, counts) + np.repeat(steps, counts) * (
                np.arange(size) - offsets
            )
        if dense is not None:
            variances[level] = np.var(dense)
            covariances[level] = np.sum((dense[:-1] - mean) * (dense[1:] - mean)) / size
            dense = 0.5 * (dense[::2] + dense[1::2])
        else:
            count = counts.astype(float)
            centered = first - mean
            midpoint = centered + steps * (count - 1) / 2
            variances[level] = (
                np.sum(count * (midpoint**2 + steps**2 * (count**2 - 1) / 12)) / size
            )
            pairs = count - 1
            internal = np.sum(
                pairs * (midpoint**2 + steps**2 * (pairs**2 - 1) / 12 - steps**2 / 4)
            )
            final = centered + steps * (count - 1)
            covariances[level] = (internal + np.sum(final[:-1] * centered[1:])) / size

            ends = np.cumsum(counts)
            starts = ends - counts
            skip = starts % 2
            paired = (counts - skip) // 2
            full = paired > 0
            crossing = np.flatnonzero(ends[:-1] % 2)
            indices = np.r_[(starts[full] + skip[full]) // 2, ends[crossing] // 2]
            order = np.argsort(indices)
            boundary = 0.5 * (
                first[crossing]
                + steps[crossing] * (counts[crossing] - 1)
                + first[crossing + 1]
            )
            first = np.r_[first[full] + steps[full] * (skip[full] + 0.5), boundary][
                order
            ]
            steps = np.r_[2 * steps[full], np.zeros(crossing.size)][order]
            counts = np.r_[paired[full], np.ones(crossing.size, dtype=np.int64)][order]
        if variances[level] == 0.0:
            return StandardErrorResult(0.0, "blocking", n_input, n_used, level)
    return _blocking_result(variances, covariances, n_input, n_used)
