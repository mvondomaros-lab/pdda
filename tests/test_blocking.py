"""Compressed exit-time blocking versus independent Jonsson calculations.

Reference: M. Jonsson, Phys. Rev. E 98, 043304 (2018),
https://doi.org/10.1103/PhysRevE.98.043304 and
https://github.com/computative/block . The reference below uses explicit
samples and independently computes chi-square quantiles and reverse sums.
PDDA additionally truncates to a power of two, handles zero variance, and
rejects a result when no supported blocking level remains.
"""

from itertools import product

import numpy as np
import pytest
from scipy.stats import chi2

import pdda.rta as rta
import pdda.statistics as statistics
from pdda import SimulationWindow, analyze_diffusivity


def expand(lengths):
    return np.concatenate([np.arange(n, 0, -1.0) for n in lengths])


def jonsson_reference(samples):
    """Explicit blocking transformations and the paper's cumulative test."""
    if len(samples) < 4:
        raise ValueError("need at least 4 samples")
    depth = len(samples).bit_length() - 1
    n_used = 2**depth
    values = samples[:n_used].copy()
    mean = sum(values) / n_used
    variances, covariances = [], []
    for level in range(depth):
        centered = values - mean
        deviations = values - values.mean()
        variance = np.dot(deviations, deviations) / len(values)
        variances.append(variance)
        covariances.append(np.dot(centered[:-1], centered[1:]) / len(values))
        if variance == 0:
            return 0.0, level, n_used, variances, covariances
        values = values.reshape(-1, 2).mean(axis=1)
    for level in range(depth):
        statistic = sum(
            (covariances[j] / variances[j]) ** 2 * 2 ** (depth - j)
            for j in range(level, depth)
        )
        if statistic < chi2.ppf(0.99, level + 1):
            break
    if level >= depth - 1:
        raise ValueError("insufficient data")
    return (
        np.sqrt(variances[level] / (2 ** (depth - level))),
        level,
        n_used,
        variances,
        covariances,
    )


def assert_equivalent(lengths):
    lengths = np.asarray(lengths, dtype=np.int64)
    values = expand(lengths) if lengths.size else np.empty(0)
    try:
        expected = jonsson_reference(values)
    except ValueError:
        with pytest.raises(ValueError):
            statistics._blocking_exit_time_standard_error(lengths)
        with pytest.raises(ValueError):
            statistics.StandardErrorEstimator(method="blocking").estimate(values)
        return
    actual = statistics._blocking_exit_time_standard_error(lengths)
    dense = statistics.StandardErrorEstimator(method="blocking").estimate(values)
    for result in (actual, dense):
        assert result.n_input == len(values)
        assert result.n_used == expected[2]
        assert result.blocking_level == expected[1]
        assert result.method == "blocking"
        assert result.value == pytest.approx(expected[0], rel=2e-12, abs=1e-13)


def test_critical_values_match_jonsson_chi_square_test():
    expected = chi2.ppf(0.99, np.arange(1, 31))
    np.testing.assert_allclose(
        statistics._BLOCKING_CRITICAL_VALUES, expected, rtol=0, atol=5e-7
    )


@pytest.mark.parametrize(
    "lengths, sem, level",
    [
        ([(i * 37) % 100 + 1 for i in range(257)], 0.6570133156247994, 7),
        ([1, 10, 100, 1000] * 64, 3.951844110932261, 11),
        ([31, 1, 32, 1, 63, 1, 64, 1, 127, 1] * 64, 0.7030750335963246, 8),
        ([65537], 4720.406762133958, 12),
    ],
)
def test_golden_sems_from_published_jonsson_code(lengths, sem, level):
    # SEMs obtained by executing sqrt(block(expand(lengths))) from
    # https://github.com/computative/block/blob/master/python/tictoc.py
    # Downloaded 2026-09-09; reference file SHA256:
    # c2aa49ac3a14215060cb078f19e540f911d1f38d666b6f96ab7aa236afa2153c
    result = statistics._blocking_exit_time_standard_error(
        np.array(lengths, dtype=np.int64)
    )
    assert result.value == pytest.approx(sem, rel=2e-12)
    assert result.blocking_level == level


@pytest.mark.parametrize("number_of_runs", range(1, 6))
def test_exhaustive_short_run_boundaries(number_of_runs):
    # Every parity and many singleton-crossing / constant-after-blocking cases.
    for lengths in product(range(1, 5), repeat=number_of_runs):
        assert_equivalent(lengths)


@pytest.mark.parametrize("seed", range(80))
def test_randomized_distributions_and_non_power_of_two_lengths(seed):
    rng = np.random.default_rng(seed)
    size = int(rng.integers(1, 150))
    for lengths in (
        rng.integers(1, 1000, size=size),
        rng.geometric(0.15, size=size),
        rng.geometric(0.002, size=size),
        np.minimum(rng.zipf(1.6, size=size), 10000),
    ):
        assert_equivalent(lengths)


@pytest.mark.parametrize(
    "lengths",
    [
        [],
        [1],
        [2],
        [3],
        [4],
        [5],
        [7],
        [8],
        [9],
        [1023],
        [1024],
        [1025],
        [1] * 4097,
        [2] * 2048,
        [1, 3] * 1024,
        [1, 10001, 1, 3, 2],
        [10001, 1, 2, 3, 1],
        [31, 1, 32, 1, 63, 1, 64, 1, 127, 1],
    ],
)
def test_degenerate_series_truncation_and_long_runs(lengths):
    assert_equivalent(lengths)


@pytest.mark.parametrize("seed", range(20))
def test_every_level_variance_covariance_and_test_statistic(seed, monkeypatch):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, 5000, size=73)
    expected = jonsson_reference(expand(lengths))
    original = statistics._blocking_result
    calls = []

    def inspect(variances, covariances, n_input, n_used):
        np.testing.assert_allclose(variances, expected[3], rtol=2e-12, atol=1e-10)
        np.testing.assert_allclose(covariances, expected[4], rtol=2e-12, atol=1e-10)
        depth = len(variances)
        observed_tests = [
            sum(
                (covariances[j] / variances[j]) ** 2 * 2 ** (depth - j)
                for j in range(k, depth)
            )
            for k in range(depth)
        ]
        reference_tests = [
            sum(
                (expected[4][j] / expected[3][j]) ** 2 * 2 ** (depth - j)
                for j in range(k, depth)
            )
            for k in range(depth)
        ]
        np.testing.assert_allclose(observed_tests, reference_tests, rtol=5e-12)
        calls.append(True)
        return original(variances, covariances, n_input, n_used)

    monkeypatch.setattr(statistics, "_blocking_result", inspect)
    statistics._blocking_exit_time_standard_error(lengths)
    assert calls == [True]


def test_long_ramp_never_allocates_per_time_origin(monkeypatch):
    # Over half a billion samples would require >4 GiB for one dense array.
    # The compressed estimator must reach a small dense tail first.
    original = np.arange

    def bounded_arange(*args, **kwargs):
        if len(args) == 1:
            assert args[0] <= 64
        return original(*args, **kwargs)

    monkeypatch.setattr(statistics.np, "arange", bounded_arange)
    result = statistics._blocking_exit_time_standard_error(np.array([2**29 + 7]))
    assert result.n_input == 2**29 + 7
    assert result.n_used == 2**29
    assert np.isfinite(result.value)


@pytest.mark.parametrize("symmetry_center", [None, 0.0])
@pytest.mark.parametrize("sampling_period", [0.002, 1.0, 1000.0])
@pytest.mark.parametrize("windowed", [False, True])
def test_diffusivity_grid_matches_explicit_blocking(
    symmetry_center,
    sampling_period,
    windowed,
    monkeypatch,
):
    rng = np.random.default_rng(91)
    first = rng.normal(size=8192)
    second = rng.normal(loc=0.25, size=4097)
    original_first, original_second = first.copy(), second.copy()
    source = (
        [SimulationWindow(first, (-3.0, 3.0)), SimulationWindow(second, (-2.0, 2.0))]
        if windowed
        else first
    )
    options = dict(
        positions=[-0.5, 0.0, 0.5],
        interval_widths=[0.5, 1.0, 1.5],
        sampling_period=sampling_period,
        symmetry_center=symmetry_center,
        uncertainty="blocking",
    )
    if not windowed:
        options["bounds"] = (-3.0, 3.0)
    fast = analyze_diffusivity(source, **options)
    np.testing.assert_array_equal(first, original_first)
    np.testing.assert_array_equal(second, original_second)

    def dense_reference(lengths):
        return statistics.StandardErrorEstimator(method="blocking").estimate(
            expand(lengths)
        )

    monkeypatch.setattr(rta, "_blocking_exit_time_standard_error", dense_reference)
    dense = analyze_diffusivity(source, **options)
    np.testing.assert_allclose(
        fast.diffusivity_sems, dense.diffusivity_sems, rtol=2e-12, atol=1e-13
    )
    np.testing.assert_array_equal(fast.diffusivities, dense.diffusivities)
    np.testing.assert_array_equal(fast.n_time_origins, dense.n_time_origins)


def explicit_completed_exits(trajectory, center, width):
    """Enumerate completed observations directly, without PDDA's run helpers."""
    values = []
    run = 0
    for position in trajectory:
        if center - width / 2 <= position < center + width / 2:
            run += 1
        else:
            values.extend(range(run, 0, -1))
            run = 0
    return np.asarray(values, dtype=float)


@pytest.mark.parametrize("reflection", [None, "same", "partner"])
def test_selected_width_sems_match_explicit_exits_and_preserve_sources(reflection):
    rng = np.random.default_rng(148)
    trajectory = rng.normal(size=8192)
    partner = rng.normal(loc=0.2, size=4096)
    originals = trajectory.copy(), partner.copy()
    positions, widths = [-0.5, 0.0, 0.5], [0.5, np.nan, 1.5]
    dt = 0.1
    analyzer = rta.RTAAnalyzer(
        positions,
        [0.5, 1.0, 1.5],
        dt,
        symmetry=None if reflection is None else rta.ReflectionSymmetry(),
    )
    window = rta.RTAWindow((trajectory,), rta.WallDomain(-3.0, 3.0))
    reflected_window = (
        rta.RTAWindow((partner,), window.domain) if reflection == "partner" else None
    )
    sems = analyzer.blocking_diffusivity_sems(
        window, widths, reflected_window=reflected_window
    )
    assert np.isnan(sems[1])
    for index in (0, 2):
        center, width = positions[index], widths[index]
        exits = explicit_completed_exits(trajectory, center, width)
        if reflection is not None:
            source = partner if reflection == "partner" else trajectory
            exits = np.r_[exits, explicit_completed_exits(-source, center, width)]
        residence = dt * exits.mean()
        residence_sem = dt * jonsson_reference(exits)[0]
        expected = width**2 * residence_sem / (12 * residence**2)
        assert sems[index] == pytest.approx(expected, rel=2e-12)
    np.testing.assert_array_equal(trajectory, originals[0])
    np.testing.assert_array_equal(partner, originals[1])


@pytest.mark.parametrize("symmetric", [False, True])
def test_separate_windows_and_reflected_coverage_match_manual_pooling(symmetric):
    rng = np.random.default_rng(332)
    windows = [
        SimulationWindow(rng.uniform(-3, -0.25, 8192), (-3.0, -0.25)),
        SimulationWindow(rng.uniform(-2, 1, 4097), (-2.0, 1.0)),
    ]
    positions = [-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5]
    widths = [0.25, 0.5, 1.0]
    dt = 0.002
    result = analyze_diffusivity(
        windows,
        positions=positions,
        interval_widths=widths,
        sampling_period=dt,
        symmetry_center=0.0 if symmetric else None,
        uncertainty="blocking",
    )
    expected = np.full((len(widths), len(positions)), np.nan)
    for i, width in enumerate(widths):
        for j, center in enumerate(positions):
            counts, means, sems = [], [], []
            for window in windows:
                lower, upper = window.bounds
                trajectory = window.trajectories[0]
                sources = [(trajectory, lower, upper)]
                if symmetric and center != 0:
                    sources.append((-trajectory, -upper, -lower))
                observations = [
                    explicit_completed_exits(path, center, width)
                    for path, lo, hi in sources
                    if lo <= center - width / 2 and center + width / 2 <= hi
                ]
                if not observations:
                    continue
                samples = np.concatenate(observations) * dt
                if not len(samples):
                    continue
                counts.append(len(samples))
                means.append(samples.mean())
                sems.append(jonsson_reference(samples)[0])
            if counts:
                weights = np.asarray(counts) / sum(counts)
                mean = np.dot(weights, means)
                sem = np.sqrt(np.sum((weights * sems) ** 2))
                expected[i, j] = width**2 * sem / (12 * mean**2)
    np.testing.assert_allclose(result.diffusivity_sems, expected, rtol=2e-12)
    assert np.all(np.isfinite(result.diffusivity_sems[np.isfinite(expected)]))


@pytest.mark.parametrize("contributes", [False, True])
def test_pooling_ignores_empty_sources_but_preserves_unavailable_uncertainty(
    contributes,
):
    good = np.random.default_rng(29).normal(size=4096)
    extra = [2.0, 0.0, 2.0] if contributes else [2.0] * 8
    options = dict(
        positions=[0.0],
        interval_widths=[1.0],
        sampling_period=0.1,
        uncertainty="blocking",
    )
    single = analyze_diffusivity(good, bounds=(-3.0, 3.0), **options)
    pooled = analyze_diffusivity(
        [SimulationWindow(good, (-3.0, 3.0)), SimulationWindow(extra, (-3.0, 3.0))],
        **options,
    )
    assert np.isfinite(single.diffusivity_sems[0, 0])
    if contributes:
        assert np.isnan(pooled.diffusivity_sems[0, 0])
    else:
        np.testing.assert_array_equal(pooled.diffusivity_sems, single.diffusivity_sems)
