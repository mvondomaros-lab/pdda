"""Tests for profile and statistics APIs."""

import numpy as np
import pytest

from pdda.profiles import (
    DensityComparator,
    DensityMetrics,
    PlateauNotFoundWarning,
    SmoothingSplineFitter,
    smooth_profile,
    symmetrize_even_profile,
)
from pdda.statistics import (
    StandardErrorEstimator,
    StandardErrorResult,
)


def test_standard_error_reports_its_method_and_sample_count() -> None:
    values = np.arange(16, dtype=float)
    result = StandardErrorEstimator().estimate(values)
    assert isinstance(result, StandardErrorResult)
    assert result.method == "iid"
    assert result.n_input == result.n_used == 16
    assert result.blocking_level is None
    assert np.isclose(result.value, np.std(values, ddof=1) / np.sqrt(16))


def test_blocking_sem_exposes_truncation() -> None:
    result = StandardErrorEstimator(method="blocking").estimate(
        np.arange(5000, dtype=float)
    )
    assert result.method == "blocking"
    assert result.n_input == 5000
    assert result.n_used == 4096
    assert result.blocking_level is not None


def test_standard_error_validates_method_and_data() -> None:
    with pytest.raises(ValueError, match="method"):
        StandardErrorEstimator(method="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="one-dimensional"):
        StandardErrorEstimator().estimate([[1.0], [2.0]])


@pytest.mark.parametrize("lam", [-1.0, np.inf])
def test_spline_fitter_validates_options_at_construction(lam: float) -> None:
    with pytest.raises(ValueError, match="lam"):
        SmoothingSplineFitter(lam=lam)
    with pytest.raises(TypeError, match="log_space"):
        SmoothingSplineFitter(log_space=1)  # type: ignore[arg-type]


def test_smoothing_spline_is_a_reusable_documented_object() -> None:
    x = np.linspace(0.0, 2.0, 21)
    y = np.exp(0.5 * x)
    fitted = SmoothingSplineFitter(lam=1e-8, log_space=True).fit(x, y)
    assert fitted.domain == (0.0, 2.0)
    assert fitted.log_space
    assert np.allclose(fitted(x), y, rtol=1e-3)
    with pytest.raises(ValueError, match="outside"):
        fitted(3.0)


def test_smoothing_spline_supports_even_and_plateau_constraints() -> None:
    x = np.linspace(-2.0, 2.0, 21)
    fitted = SmoothingSplineFitter(even=True, lam=1e-4).fit(x, 1.0 + x**2 + 0.1 * x)
    assert np.isclose(fitted(-0.7), fitted(0.7))
    step = 1e-6
    left_slope = (fitted(0.0) - fitted(-step)) / step
    right_slope = (fitted(step) - fitted(0.0)) / step
    assert left_slope == pytest.approx(0.0, abs=1e-4)
    assert right_slope == pytest.approx(0.0, abs=1e-4)
    plateau_x = np.arange(11, dtype=float)
    plateau_y = np.r_[np.exp(np.linspace(0.0, 3.0, 6)), np.full(5, 20.0)]
    plateau = SmoothingSplineFitter(
        log_space=True,
        lam=1e-8,
        plateau=True,
        plateau_tolerance=0.02,
        extrapolate=True,
    ).fit(plateau_x, plateau_y)
    assert np.isclose(plateau(20.0), 20.0)
    assert plateau.plateaus[1] is not None
    assert plateau.plateaus[1].n_points >= 3


def test_even_spline_has_no_cusp_at_symmetry_center() -> None:
    x = np.array([-15.0, -11.25, -7.5, -3.75, 0.0, 3.75, 7.5, 11.25, 15.0])
    y = np.array([0.1, 0.1, 0.1, 0.354, 0.521, 0.317, 0.109, 0.107, 0.1])
    fitted = SmoothingSplineFitter(log_space=True, even=True).fit(x, y)

    step = 1e-5
    left_slope = (fitted(0.0) - fitted(-step)) / step
    right_slope = (fitted(step) - fitted(0.0)) / step

    assert left_slope == pytest.approx(0.0, abs=1e-4)
    assert right_slope == pytest.approx(0.0, abs=1e-4)


def test_log_space_plateau_detection_is_unit_invariant() -> None:
    x = np.arange(9, dtype=float)
    y = np.r_[np.exp(np.linspace(0.0, 2.0, 5)), np.full(4, 8.0)]
    fitter = SmoothingSplineFitter(
        log_space=True,
        plateau=True,
        plateau_tolerance=0.03,
        minimum_plateau_points=3,
    )

    original = fitter.fit(x, y).plateaus[1]
    rescaled = fitter.fit(x, 1000.0 * y).plateaus[1]

    assert original is not None
    assert rescaled is not None
    assert original.start == rescaled.start
    assert rescaled.value == pytest.approx(1000.0 * original.value)
    assert rescaled.relative_trend == pytest.approx(original.relative_trend)
    assert rescaled.relative_scatter == pytest.approx(original.relative_scatter)


def test_plateau_detection_reports_failure_instead_of_last_point() -> None:
    x = np.arange(8, dtype=float)
    y = np.exp(0.5 * x)
    fitted = SmoothingSplineFitter(
        log_space=True,
        plateau=True,
        plateau_tolerance=0.02,
        minimum_plateau_points=3,
    ).fit(x, y)

    assert fitted.plateaus[1] is None


def test_density_comparison_requires_or_explicitly_requests_normalization() -> None:
    grid = np.linspace(-1.0, 1.0, 21)
    density = np.exp(-(grid**2))
    with pytest.raises(ValueError, match="normalized"):
        DensityComparator(grid).compare(density, density)
    metrics = DensityComparator(grid, normalize=True).compare(density, density)
    assert isinstance(metrics, DensityMetrics)
    assert metrics.total_variation == pytest.approx(0.0)
    assert metrics.reference_mass != 1.0


def test_density_comparison_handles_periodic_distance() -> None:
    grid = np.arange(4, dtype=float)
    reference = np.array([1.0, 0.0, 0.0, 0.0])
    model = np.array([0.0, 0.0, 0.0, 1.0])
    linear = DensityComparator(grid, normalize=True).compare(reference, model)
    periodic = DensityComparator(grid, periodic=True, normalize=True).compare(
        reference, model
    )
    assert linear.wasserstein_1 == pytest.approx(3.0)
    assert periodic.wasserstein_1 == pytest.approx(1.0)
    assert periodic.regularized_model


def test_jensen_shannon_handles_subnormal_model_density() -> None:
    tiny = np.nextafter(0.0, 1.0)
    metrics = DensityComparator([0.0, 1.0], normalize=True).compare(
        [1.0, 0.0], [1.0, tiny]
    )
    assert np.isfinite(metrics.jensen_shannon)
    assert metrics.jensen_shannon == pytest.approx(0.0)


@pytest.mark.parametrize("density_floor", [0.0, -1.0, np.inf])
def test_density_comparator_validates_options_at_construction(
    density_floor: float,
) -> None:
    with pytest.raises(ValueError, match="density_floor"):
        DensityComparator([0.0, 1.0], density_floor=density_floor)
    with pytest.raises(TypeError, match="normalize"):
        DensityComparator([0.0, 1.0], normalize=1)  # type: ignore[arg-type]


def test_symmetrize_even_profile_and_smooth_positive_profile() -> None:
    x = np.linspace(-2.0, 2.0, 21)
    y = 2.0 + x**2 + 0.2 * x
    symmetric_x, symmetric_y = symmetrize_even_profile(x, y)
    assert np.array_equal(symmetric_x, -symmetric_x[::-1])
    assert np.allclose(symmetric_y, symmetric_y[::-1])
    fitted = smooth_profile(
        x,
        np.exp(x**2 + 0.1 * x),
        positive=True,
        symmetry_center=0.0,
    )(x)
    assert np.all(fitted > 0.0)
    assert np.allclose(fitted, fitted[::-1])


def test_smooth_profile_can_continue_outer_plateau() -> None:
    x = np.arange(-10.0, 11.0)
    y = np.where(np.abs(x) >= 6.0, 4.0, np.exp(0.04 * x**2))
    target = np.linspace(-14.0, 14.0, 57)

    fitted = smooth_profile(
        x,
        y,
        positive=True,
        symmetry_center=0.0,
        plateau=True,
        plateau_tolerance=0.02,
        extrapolate=True,
    )(target)

    outer = np.abs(target) >= 10.0
    assert fitted[outer] == pytest.approx(4.0)
    assert np.allclose(fitted, fitted[::-1])


def test_smooth_profile_detects_two_point_outer_blocks_across_missing_data() -> None:
    x = np.arange(-10.0, 11.0)
    y = np.where(np.abs(x) <= 5.0, 0.1, np.nan)
    y[np.abs(x) == 9.0] = 1.0
    y[np.abs(x) == 10.0] = 1.03

    fitted = smooth_profile(x, y, positive=True, symmetry_center=0.0, plateau=True)

    assert fitted.plateau_starts == (-9.0, 9.0)
    assert all(item.n_points == 2 for item in fitted.plateaus)

    y[np.abs(x) <= 5.0] = 1.0
    y[np.abs(x) == 10.0] = 1.07
    with pytest.warns(PlateauNotFoundWarning):
        rejected = smooth_profile(
            x, y, positive=True, symmetry_center=0.0, plateau=True
        )
    assert rejected.plateaus == (None, None)


def test_plateau_search_uses_outer_half_and_produces_exact_plateau() -> None:
    x = np.arange(11, dtype=float)
    y = np.r_[np.exp(np.linspace(0.0, 3.0, 6)), np.full(5, 20.0)]
    fitted = SmoothingSplineFitter(
        log_space=True,
        lam=1e-8,
        plateau=True,
        plateau_tolerance=0.02,
        extrapolate=True,
    ).fit(x, y)

    assert fitted.plateaus[1] is not None
    assert fitted.plateaus[1].start >= 0.5 * (x[0] + x[-1])
    start = fitted.plateaus[1].start
    assert fitted(np.array([start, 9.0, 10.0, 12.0])) == pytest.approx(20.0)
    step = 1e-5
    inner_slope = (fitted(start) - fitted(start - step)) / step
    assert inner_slope == pytest.approx(0.0, abs=1e-6)


def test_smooth_profile_only_fits_a_plateau_when_requested() -> None:
    x = np.arange(-10.0, 11.0)
    y = np.where(np.abs(x) >= 6.0, 20.0, np.exp(0.08 * x**2))

    ordinary = smooth_profile(x, y, positive=True, symmetry_center=0.0)
    plateau = smooth_profile(x, y, positive=True, symmetry_center=0.0, plateau=True)

    assert ordinary.plateaus == (None, None)
    assert plateau.plateaus[1] is not None


def test_requested_unsupported_plateau_warns() -> None:
    x = np.arange(-5.0, 6.0)
    y = np.exp(0.2 * x)

    with pytest.warns(PlateauNotFoundWarning, match="no supported outer plateau"):
        fitted = smooth_profile(x, y, positive=True, plateau=True)

    assert fitted.plateaus == (None, None)


def test_boolean_extrapolation_is_constant() -> None:
    x = np.arange(5.0)
    y = 1.0 + x**2
    no_extrapolation = smooth_profile(x, y)
    extrapolation = smooth_profile(x, y, extrapolate=True)

    with pytest.raises(ValueError, match="outside"):
        no_extrapolation([-1.0, 5.0])
    assert extrapolation([-1.0, 5.0]) == pytest.approx(
        [extrapolation(0.0), extrapolation(4.0)]
    )


@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("smoothing_length", [None, 2.0])
@pytest.mark.parametrize("center", [0.0, 3.0])
def test_plateau_conditions_are_part_of_the_spline(
    positive,
    smoothing_length,
    center,
) -> None:
    x = np.linspace(-10.0, 10.0, 41)
    y = np.where(np.abs(x) >= 6.0, 4.0, 1.0 + 0.08 * x**2)
    fitted = smooth_profile(
        x + center,
        y,
        positive=positive,
        symmetry_center=center,
        plateau=True,
        smoothing_length=smoothing_length,
        extrapolate=True,
    )
    for start, value in zip(fitted.plateau_starts, fitted.plateau_values):
        assert start is not None
        start -= center
        expected = np.log(value) if positive else value
        assert fitted._spline(start) == pytest.approx(expected, abs=1e-10)
        assert fitted._spline(start, nu=1) == pytest.approx(0.0, abs=1e-10)
        assert fitted._spline(start, nu=2) == pytest.approx(0.0, abs=1e-10)
    target = np.linspace(-12, 12, 301)
    np.testing.assert_allclose(
        fitted(center + target), fitted(center - target), atol=1e-9
    )
    assert fitted._spline(0.0, nu=1) == pytest.approx(0.0, abs=1e-9)


def test_missing_plateau_only_constrains_end_slopes_when_extrapolating() -> None:
    x = np.linspace(-5.0, 5.0, 21)
    y = np.exp(0.2 * x)
    with pytest.warns(PlateauNotFoundWarning, match="only zero endpoint slope"):
        fitted = smooth_profile(
            x,
            y,
            positive=True,
            plateau=True,
            extrapolate=True,
            smoothing_length=1.0,
        )
    assert fitted.plateaus == (None, None)
    assert fitted._spline(x[[0, -1]], nu=1) == pytest.approx([0.0, 0.0], abs=1e-10)
    assert fitted(-4.0) != pytest.approx(fitted(-5.0))
    assert fitted(4.0) != pytest.approx(fitted(5.0))
    np.testing.assert_allclose(fitted([-6, 6]), fitted([-5, 5]))


@pytest.mark.parametrize("n_points", [5, 11])
def test_constant_profile_with_plateaus_remains_constant(n_points: int) -> None:
    x = np.arange(float(n_points))
    fitted = smooth_profile(x, np.full(n_points, 3.0), plateau=True, extrapolate=True)
    assert all(plateau is not None for plateau in fitted.plateaus)
    np.testing.assert_allclose(fitted(np.linspace(-1, 12, 100)), 3.0, atol=1e-9)


@pytest.mark.parametrize("lam", [0.0, 0.1, None])
def test_short_interior_can_join_two_different_plateaus(lam: float | None) -> None:
    x = np.arange(11.0)
    y = np.r_[np.ones(5), 1.01, np.full(5, 1.02)]
    fitted = SmoothingSplineFitter(plateau=True, lam=lam).fit(x, y)
    for plateau in fitted.plateaus:
        assert plateau is not None
        assert fitted._spline(plateau.start) == pytest.approx(plateau.value)
        assert fitted._spline(plateau.start, nu=1) == pytest.approx(0, abs=1e-10)
        assert fitted._spline(plateau.start, nu=2) == pytest.approx(0, abs=1e-10)


def test_smooth_profile_accepts_spatial_smoothing_length() -> None:
    x = np.arange(-10.0, 11.0)
    y = np.exp(0.2 * np.cos(2.0 * x) + 0.01 * x**2)
    target = np.linspace(-10.0, 10.0, 101)

    fitted = smooth_profile(
        x, y, positive=True, symmetry_center=0.0, smoothing_length=2.0
    )(target)

    assert np.all(fitted > 0.0)
    assert np.allclose(fitted, fitted[::-1])
    with pytest.raises(ValueError, match="smoothing_length"):
        smooth_profile(x, y, smoothing_length=0.0)
