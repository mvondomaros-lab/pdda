from pathlib import Path

import numpy as np
import pytest

from pdda.rta import (
    DiffusiveSurvivalModel,
    DiffusivityAnalysis,
    ReflectionSymmetry,
    RTAAnalyzer,
    RTAEstimator,
    RTATerminalCensoringWarning,
    RTAWindow,
    WallDomain,
)


def test_estimator_returns_observed_exit_statistics_and_metadata() -> None:
    trajectory = np.array([-0.1, 0.2, 0.8, 0.9, 1.2, 0.5, 0.6, -0.2])
    result = RTAEstimator(0.5, 1.0, 0.1).analyze(trajectory)
    assert np.allclose(result.exit_times, [0.3, 0.2, 0.1, 0.2, 0.1])
    assert result.terminal_censor_times.size == 0
    assert result.n_valid_frames == 5
    assert result.n_observed_segments == 2
    assert result.n_terminal_segments == 0
    assert result.terminal_censored_fraction == 0.0
    assert np.isclose(result.residence_time, result.exit_times.mean())
    assert np.isclose(result.diffusivity, 1.0 / (12.0 * result.residence_time))


def test_segments_expose_indices_and_terminal_censoring_status() -> None:
    segments = RTAEstimator(0.5, 1.0, 0.2).segments([-1.0, 0.1, 0.2, 2.0, 0.4, 0.5])
    assert [(item.start, item.stop, item.exit_observed) for item in segments] == [
        (1, 3, True),
        (4, 6, False),
    ]
    assert np.allclose(segments[0].values, [0.1, 0.2])


def test_terminal_segment_is_discarded_from_rta_estimate() -> None:
    with pytest.warns(RTATerminalCensoringWarning, match="terminal censoring is 66.7%"):
        result = RTAEstimator(0.5, 1.0, 0.1).analyze([-1.0, 0.2, 2.0, 0.3, 0.4])
    assert result.residence_time == pytest.approx(0.1)
    assert result.diffusivity == pytest.approx(1.0 / 1.2)
    assert result.terminal_censored_fraction == pytest.approx(2.0 / 3.0)


def test_terminal_censoring_diagnostics_are_retained() -> None:
    result = RTAEstimator(0.5, 1.0, 0.1).analyze(
        np.r_[-1.0, np.full(100, 0.5), 2.0, 0.3]
    )
    assert result.terminal_censor_times.size == 1
    assert result.n_observed_segments == 1
    assert result.n_terminal_segments == 1
    assert np.isclose(result.terminal_censored_fraction, 1.0 / 101.0)
    assert result.exit_times.size == 100


def test_terminal_censoring_is_excluded_from_empirical_survival() -> None:
    with pytest.warns(RTATerminalCensoringWarning):
        result = RTAEstimator(0.5, 1.0, 0.1).analyze(
            [-1.0, 0.2, 2.0, 0.3, 0.4], survival_times=0.1
        )
    assert np.allclose(result.exit_times, [0.1])
    assert np.allclose(result.terminal_censor_times, [0.1, 0.0])
    assert result.n_observed_segments == 1
    assert result.n_terminal_segments == 1
    assert result.survival_times.shape == (1,)
    assert result.survival[0] == 0.0
    assert result.residence_time == pytest.approx(0.1)


def test_empty_interval_visits_have_explicit_empty_result() -> None:
    result = RTAEstimator(0.5, 1.0, 0.1).analyze([-2.0, 2.0])
    assert result.n_valid_frames == 0
    assert result.n_observed_segments == 0
    assert result.n_terminal_segments == 0
    assert result.exit_times.size == 0
    assert np.isnan(result.residence_time)


@pytest.mark.parametrize("trajectory", [[], [[0.1], [0.2]], [0.1, np.nan]])
def test_rejects_invalid_trajectories(trajectory: object) -> None:
    with pytest.raises(ValueError):
        RTAEstimator(0.5, 1.0, 0.1).analyze(trajectory)


def test_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="sampling_period"):
        RTAEstimator(0.5, 1.0, 0.0)


def test_analyze_many_preserves_independent_terminal_segments() -> None:
    estimator = RTAEstimator(0.5, 1.0, 0.1)
    with pytest.warns(RTATerminalCensoringWarning):
        result = estimator.analyze_many(([-1.0, 0.2, 2.0], [-1.0, 0.3, 0.4]))
    assert np.allclose(result.exit_times, [0.1])
    assert np.allclose(result.terminal_censor_times, [0.1, 0.0])
    assert result.n_observed_segments == 1
    assert result.n_terminal_segments == 1


def test_rta_mean_uses_completed_exit_observations_only() -> None:
    with pytest.warns(RTATerminalCensoringWarning):
        result = RTAEstimator(0.5, 1.0, 0.1).analyze_many(
            ([-1.0, 0.2, 0.3, 0.4, 2.0], [-1.0, 0.5, 0.6])
        )
    assert np.isclose(result.residence_time, 0.2)
    assert np.isclose(result.diffusivity, 1.0 / (12.0 * 0.2))


def test_grid_analyzer_streams_files_and_scores_rta_survival(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.z"
    trajectory = np.array([-1.0, 0.1, 0.2, 1.1, 0.1, 0.2, 1.1])
    np.savetxt(path, trajectory)
    result = RTAAnalyzer([0.0], [1.0, 2.0], 0.1).analyze(
        RTAWindow((path,), WallDomain(-2.0, 2.0))
    )
    assert result.diffusivities.shape == (2, 1)
    assert result.excess_nll.shape == (2, 1)
    assert result.first_step_exit_fractions.shape == (2, 1)
    assert np.all(np.isfinite(result.diffusivities))
    assert np.all(np.isfinite(result.excess_nll))
    assert np.all(result.n_time_origins > 0)
    direct = RTAEstimator(0.0, 1.0, 0.1).analyze(trajectory)
    model = DiffusiveSurvivalModel(-0.5, 0.5, direct.diffusivity)
    score = model.score_discrete_exit_times(direct.exit_times, [], 0.1)
    assert result.diffusivities[0, 0] == pytest.approx(direct.diffusivity)
    assert result.excess_nll[0, 0] == pytest.approx(
        score.mean_excess_negative_log_pseudolikelihood
    )


def test_constant_nll_scan_reports_ordered_discrete_minima() -> None:
    centers = np.arange(3.0)
    widths = np.arange(1.0, 6.0)
    losses = np.array(
        [
            [0.3, 0.3, 0.3],
            [0.1, np.nan, 0.1],
            [0.2, 0.2, 0.2],
            [0.15, 0.15, 0.15],
            [0.25, 0.25, 0.25],
        ]
    )
    result = DiffusivityAnalysis(
        positions=centers,
        interval_widths=widths,
        diffusivities=np.where(np.isfinite(losses), widths[:, None], np.nan),
        excess_nll=losses,
        n_time_origins=np.ones(losses.shape, dtype=np.int64),
        first_step_exit_fractions=np.zeros(losses.shape),
        terminal_censored_fractions=np.zeros(losses.shape),
        sampling_period=1.0,
    )

    minima = result.constant_width_minima()

    assert [selection.method for selection in minima] == ["NLL_0", "NLL_1"]
    assert [selection.interval_width for selection in minima] == [2.0, 4.0]
    assert minima[0].mean_excess_nll == pytest.approx(0.1)
    assert minima[0].interpolated_nll_fraction == pytest.approx(1 / 3)
    assert minima[0].coverage == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "scores, expected",
    [
        ([0.01, 0.02, 0.3, 0.1, 0.2, 0.05, 0.2], 6.0),
        ([0.01, 0.02, 0.3, 0.05, 0.2, 0.1, 0.2], 4.0),
        ([0.01, 0.02, 0.3, 0.2, 0.1, 0.2, 0.3], 5.0),
        ([0.01, 0.02, 0.1, 0.2, 0.3, 0.4, 0.5], np.nan),
    ],
)
def test_select_profile_chooses_lowest_eligible_nll_minimum(scores, expected) -> None:
    centers = np.arange(-5.0, 6.0)
    widths = np.arange(1.0, 8.0)
    bulk_values = np.array([1.3, 1.2, 1.1, 1.0, 1.01, 0.99, 1.0])
    diffusivities = np.array(
        [
            np.where(np.abs(centers) >= 3.0, bulk, 0.3 + 0.1 * np.abs(centers))
            for bulk in bulk_values
        ]
    )
    excess_nll = np.full((widths.size, centers.size), 1.0)
    bulk_scores = np.array(scores)
    excess_nll[:, np.abs(centers) >= 3.0] = bulk_scores[:, None]
    shape = diffusivities.shape
    result = DiffusivityAnalysis(
        positions=centers,
        interval_widths=widths,
        diffusivities=diffusivities,
        excess_nll=excess_nll,
        n_time_origins=np.ones(shape, dtype=np.int64),
        first_step_exit_fractions=np.zeros(shape),
        terminal_censored_fractions=np.zeros(shape),
        sampling_period=1.0,
    )

    selected = result.select_profile()

    assert selected.plateau_tolerance == 0.05
    assert selected.minimum_plateau_points == 2
    assert selected.limiting_bulk_diffusivity == pytest.approx(1.0, rel=0.05)
    assert selected.convergence_width == pytest.approx(4.0)
    assert np.array_equal(
        selected.plateau_compatible, [False, False, False] + [True] * 4
    )
    assert selected.minimum_bulk_width == pytest.approx(4.0)
    assert selected.maximum_bulk_width == pytest.approx(7.0)
    if np.isnan(expected):
        assert np.isnan(selected.interval_width)
        assert np.all(np.isnan(selected.diffusivity))
        assert "No NLL minimum" in selected.summary()
    else:
        assert selected.interval_width == expected
        minimum = next(
            item
            for item in result.constant_width_minima()
            if item.interval_width == expected
        )
        assert selected.profile.nll_rank == minimum.nll_rank
        assert selected.profile.mean_excess_nll == minimum.mean_excess_nll
        np.testing.assert_array_equal(selected.diffusivity, minimum.diffusivity)


def test_select_profile_can_report_no_detected_limit() -> None:
    centers = np.arange(-5.0, 6.0)
    widths = np.arange(1.0, 7.0)
    diffusivities = np.array(
        [np.exp(0.2 * np.abs(centers)) * width for width in widths]
    )
    shape = diffusivities.shape
    result = DiffusivityAnalysis(
        positions=centers,
        interval_widths=widths,
        diffusivities=diffusivities,
        excess_nll=np.ones(shape),
        n_time_origins=np.ones(shape, dtype=np.int64),
        first_step_exit_fractions=np.zeros(shape),
        terminal_censored_fractions=np.zeros(shape),
        sampling_period=1.0,
    )

    selected = result.select_profile()

    assert np.isnan(selected.limiting_bulk_diffusivity)
    assert np.isnan(selected.interval_width)
    assert np.isnan(selected.profile.interval_width)


def test_select_profile_prefers_supported_internal_peak_over_sparse_tail() -> None:
    centers = np.arange(-10.0, 11.0)
    widths = np.arange(1.0, 15.0)
    bulk_values = np.array(
        [0.6, 0.8, 0.9, 1.0, 1.01, 1.0, 1.0, 0.9, 0.8, 0.7, 1.2, 1.21, 1.19, 1.2]
    )
    diffusivities = np.array(
        [np.where(np.abs(centers) >= 7.0, value, 0.3) for value in bulk_values]
    )
    diffusivities[10:, (np.abs(centers) >= 7.0) & (np.abs(centers) < 9.0)] = np.nan
    scores = np.array(
        [0.4, 0.3, 0.2, 0.1, 0.05, 0.1, 0.2, 0.2, 0.2, 0.2, 0.01, 0.2, 0.3, 0.4]
    )
    excess_nll = np.broadcast_to(scores[:, None], diffusivities.shape).copy()
    shape = diffusivities.shape
    analysis = DiffusivityAnalysis(
        positions=centers,
        interval_widths=widths,
        diffusivities=diffusivities,
        excess_nll=excess_nll,
        n_time_origins=np.ones(shape, dtype=np.int64),
        first_step_exit_fractions=np.zeros(shape),
        terminal_censored_fractions=np.zeros(shape),
        sampling_period=1.0,
    )

    selected = analysis.select_profile()

    assert selected.limiting_bulk_diffusivity == pytest.approx(1.0, rel=0.02)
    assert selected.convergence_width == pytest.approx(4.0)
    assert selected.minimum_bulk_width == pytest.approx(4.0)
    assert selected.maximum_bulk_width == pytest.approx(7.0)
    assert selected.interval_width == pytest.approx(5.0)
    assert not np.any(selected.plateau_compatible[7:])
    np.testing.assert_array_equal(selected.spatial_plateau_point_counts[-4:], 2)


def test_select_profile_uses_tail_trend_without_a_separate_endpoint_rule() -> None:
    centers = np.arange(-10.0, 11.0)
    widths = np.arange(1.0, 7.0)
    outer_outlier = np.where(np.abs(centers) == 10.0, 1.03, 1.0)
    diffusivities = np.repeat(outer_outlier[None, :], widths.size, axis=0)
    shape = diffusivities.shape
    result = DiffusivityAnalysis(
        positions=centers,
        interval_widths=widths,
        diffusivities=diffusivities,
        excess_nll=np.ones(shape),
        n_time_origins=np.ones(shape, dtype=np.int64),
        first_step_exit_fractions=np.zeros(shape),
        terminal_censored_fractions=np.zeros(shape),
        sampling_period=1.0,
    )

    selected = result.select_profile(plateau_tolerance=0.025)

    assert np.all(np.isfinite(selected.bulk_diffusivities))
    assert np.all(selected.spatial_plateau_point_counts >= 2)


def test_select_profile_uses_only_the_outermost_contiguous_support() -> None:
    centers = np.arange(-10.0, 11.0)
    widths = np.arange(1.0, 7.0)
    values = np.where(np.abs(centers) <= 5.0, 0.1, np.nan)
    values[np.abs(centers) == 9.0] = 1.0
    values[np.abs(centers) == 10.0] = 1.03
    diffusivities = np.repeat(values[None, :], widths.size, axis=0)
    shape = diffusivities.shape
    result = DiffusivityAnalysis(
        positions=centers,
        interval_widths=widths,
        diffusivities=diffusivities,
        excess_nll=np.ones(shape),
        n_time_origins=np.ones(shape, dtype=np.int64),
        first_step_exit_fractions=np.zeros(shape),
        terminal_censored_fractions=np.zeros(shape),
        sampling_period=1.0,
    )

    selected = result.select_profile()

    np.testing.assert_array_equal(selected.spatial_plateau_starts, 9.0)
    np.testing.assert_array_equal(selected.spatial_plateau_point_counts, 2)
    np.testing.assert_allclose(selected.bulk_diffusivities, np.sqrt(1.03))

    result.diffusivities[:, np.abs(centers) <= 5.0] = 1.0
    result.diffusivities[:, np.abs(centers) == 10.0] = 1.07
    rejected = result.select_profile()
    assert not np.any(np.isfinite(rejected.bulk_diffusivities))


def test_profile_analysis_warns_about_high_terminal_censoring(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.z"
    np.savetxt(path, [0.0, 0.0, 1.0, 0.0, 0.0])

    analyzer = RTAAnalyzer([0.0], [1.0], 0.1)
    with pytest.warns(
        RTATerminalCensoringWarning, match="terminal censoring exceeds 5.0%"
    ):
        result = analyzer.analyze(RTAWindow((path,), WallDomain(-2.0, 2.0)))

    assert result.terminal_censored_fractions[0, 0] == pytest.approx(0.5)


def test_overlapping_windows_pool_only_the_trajectories_that_support_interval(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "first.z", tmp_path / "second.z"
    np.savetxt(first, [-2.0, -1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2])
    np.savetxt(second, [-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6, 2.0])
    analyzer = RTAAnalyzer([-1.5, 0.0, 1.5], [1.0], 0.1)

    pooled = analyzer.analyze_overlapping_windows(
        (
            RTAWindow((first,), WallDomain(-2.0, 1.0)),
            RTAWindow((second,), WallDomain(-1.0, 2.0)),
        )
    )
    first_only = analyzer.analyze(RTAWindow((first,), WallDomain(-2.0, 1.0)))
    second_only = analyzer.analyze(RTAWindow((second,), WallDomain(-1.0, 2.0)))

    assert pooled.n_time_origins[0, 0] == first_only.n_time_origins[0, 0]
    assert pooled.n_time_origins[0, 2] == second_only.n_time_origins[0, 2]
    assert pooled.n_time_origins[0, 1] == (
        first_only.n_time_origins[0, 1] + second_only.n_time_origins[0, 1]
    )


def test_profile_analyzer_accepts_non_integer_intervals(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.z"
    trajectory = np.array([-1.0, -0.4, 0.1, 0.8, 1.2, 0.1, 0.8, 1.2])
    np.savetxt(path, trajectory)
    center, width = 0.25, 1.5
    result = RTAAnalyzer([center], [width], 0.1).analyze(
        RTAWindow((path,), WallDomain(-2.0, 2.0))
    )
    direct = RTAEstimator(center, width, 0.1).analyze(trajectory)
    assert result.diffusivities[0, 0] == pytest.approx(direct.diffusivity)


def test_profile_analyzer_exposes_blocking_sems_without_private_internals(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trajectory.z"
    np.savetxt(path, np.tile([-1.0, 0.0, 1.0], 16))
    analyzer = RTAAnalyzer([0.0], [1.0], 0.1)
    window = RTAWindow((path,), WallDomain(-2.0, 2.0))
    sems = analyzer.blocking_diffusivity_sems(window, [1.0])
    assert sems.shape == (1,)
    assert sems[0] == pytest.approx(0.0)


def test_reflection_symmetry_pools_mirrored_first_exit_data(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.z"
    trajectory = np.array([-3.0, -1.5, -1.4, -0.5, 1.4, 1.5, 2.5, 1.5, 2.5])
    np.savetxt(path, trajectory)
    analyzer = RTAAnalyzer([1.5], [1.0], 0.1, symmetry=ReflectionSymmetry())
    result = analyzer.analyze(RTAWindow((path,), WallDomain(-3.0, 3.0)))
    manual = RTAEstimator(1.5, 1.0, 0.1).analyze_many((trajectory, -trajectory))
    assert result.diffusivities[0, 0] == pytest.approx(manual.diffusivity)


def test_reflection_symmetry_pools_a_partner_window(tmp_path: Path) -> None:
    negative_path = tmp_path / "negative.z"
    positive_path = tmp_path / "positive.z"
    negative = np.array([-3.5, -2.5, -2.4, -1.5, -2.4, -1.5])
    positive = np.array([3.5, 2.5, 2.4, 1.5, 2.4, 1.5])
    np.savetxt(negative_path, negative)
    np.savetxt(positive_path, positive)
    analyzer = RTAAnalyzer([-2.0], [1.0], 0.1, symmetry=ReflectionSymmetry())
    negative_window = RTAWindow((negative_path,), WallDomain(-4.0, -1.0))
    positive_window = RTAWindow((positive_path,), WallDomain(1.0, 4.0))
    result = analyzer.analyze(negative_window, reflected_window=positive_window)
    manual = RTAEstimator(-2.0, 1.0, 0.1).analyze_many((negative, -positive))
    assert result.diffusivities[0, 0] == pytest.approx(manual.diffusivity)


def test_reflection_symmetry_pools_intervals_that_cross_its_plane(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trajectory.z"
    np.savetxt(path, [-2.0, 0.0, 2.0])
    result = RTAAnalyzer([0.0, 0.5], [2.0], 0.1, symmetry=ReflectionSymmetry()).analyze(
        RTAWindow((path,), WallDomain(-3.0, 3.0))
    )
    assert np.isfinite(result.diffusivities[0, 0])
    assert np.isfinite(result.diffusivities[0, 1])


def test_diffusive_survival_model_has_correct_initial_value_and_mean() -> None:
    model = DiffusiveSurvivalModel(-2.0, 2.0, diffusivity=0.5)
    assert np.allclose(model.survival([0.0, 1.0]), [1.0, model.survival(1.0)[0]])
    assert np.isclose(model.residence_time, 4.0**2 / (12.0 * 0.5))


def test_short_time_survival_matches_a_long_eigenmode_sum() -> None:
    model = DiffusiveSurvivalModel(-10.0, 10.0, diffusivity=0.1)
    times = np.array([0.002, 0.02, 0.2, 2.0])
    odd_modes = 2.0 * np.arange(4096) + 1.0
    scaled = model.diffusivity * times / 20.0**2
    expected = (8.0 / np.pi**2) * np.sum(
        np.exp(-(np.pi**2) * np.multiply.outer(scaled, odd_modes**2)) / odd_modes**2,
        axis=1,
    )
    assert np.allclose(model.survival(times), expected, atol=2e-13)


def test_diffusive_survival_comparison_is_zero_for_its_own_curve() -> None:
    model = DiffusiveSurvivalModel(-1.0, 1.0, diffusivity=0.25)
    times = np.linspace(0.0, 4.0, 101)
    comparison = model.compare(times, model.survival(times))
    assert comparison.integrated_squared_error == pytest.approx(0.0)
    assert comparison.normalized_root_mean_squared_error == pytest.approx(0.0)


def test_short_exits_favor_faster_diffusion() -> None:
    times = np.array([0.1, 0.2, 0.3, 0.4])
    faster_model = DiffusiveSurvivalModel(-1.0, 1.0, diffusivity=1.0)
    mismatched_model = DiffusiveSurvivalModel(-1.0, 1.0, diffusivity=0.1)
    assert (
        faster_model.score_discrete_exit_times(
            times, [], 0.1
        ).mean_negative_log_pseudolikelihood
        < mismatched_model.score_discrete_exit_times(
            times, [], 0.1
        ).mean_negative_log_pseudolikelihood
    )


def test_diffusive_survival_fit_is_deterministic_and_positive() -> None:
    fit = DiffusiveSurvivalModel.fit_discrete_exit_times(
        -1.0, 1.0, [0.1, 0.2, 0.3, 0.4], [0.2], 0.1
    )
    assert fit.model.diffusivity > 0.0
    assert np.isfinite(fit.score.mean_negative_log_pseudolikelihood)
    assert fit.n_function_evaluations > 0
    repeated = DiffusiveSurvivalModel.fit_discrete_exit_times(
        -1.0, 1.0, [0.1, 0.2, 0.3, 0.4], [0.2], 0.1
    )
    assert repeated.model.diffusivity == fit.model.diffusivity
    assert repeated.score == fit.score


def test_wall_domain_excludes_intervals_that_cross_a_wall(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.z"
    np.savetxt(path, [-1.0, 0.0, 1.0])
    result = RTAAnalyzer([0.0, 1.0], [1.0, 2.0], 0.1).analyze(
        RTAWindow((path,), WallDomain(-1.0, 1.5))
    )
    assert np.isfinite(result.diffusivities[0, 0])
    assert np.isfinite(result.diffusivities[1, 0])
    assert np.isnan(result.diffusivities[1, 1])
