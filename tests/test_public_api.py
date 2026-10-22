import inspect

import numpy as np
import pytest

import pdda
from pdda import (
    DiffusivityAnalysis,
    DiffusivityEstimate,
    DiffusivityProfile,
    PropagationResult,
    SimulationWindow,
    analyze_diffusivity,
    compare_densities,
    estimate_diffusivity,
    propagate_density,
    smooth_profile,
)


def _trajectory() -> np.ndarray:
    return np.array([-2.0, -0.2, 0.1, 0.4, 2.0] * 8)


def test_top_level_namespace_contains_only_workflow_api() -> None:
    expected = {
        "DensityMetrics",
        "DiffusivityAnalysis",
        "DiffusivityEstimate",
        "DiffusivityProfile",
        "PlateauNotFoundWarning",
        "ProfileSelection",
        "PropagationResult",
        "RTATerminalCensoringWarning",
        "SimulationWindow",
        "SmoothingSpline",
        "analyze_diffusivity",
        "compare_densities",
        "estimate_diffusivity",
        "periodic_grid",
        "propagate_density",
        "smooth_profile",
    }
    assert set(pdda.__all__) == expected
    assert all(getattr(pdda, name) is not None for name in expected)


def test_estimation_api_is_unit_agnostic() -> None:
    original = estimate_diffusivity(
        _trajectory(),
        center=0.0,
        interval_width=1.0,
        sampling_period=0.1,
    )
    length_scale, time_scale = 0.1, 1000.0
    converted = estimate_diffusivity(
        _trajectory() * length_scale,
        center=0.0,
        interval_width=length_scale,
        sampling_period=0.1 * time_scale,
    )
    assert converted.residence_time == pytest.approx(
        original.residence_time * time_scale
    )
    assert converted.diffusivity == pytest.approx(
        original.diffusivity * length_scale**2 / time_scale
    )


def test_smoothing_api_uses_explicit_plateau_and_extrapolation_switches() -> None:
    parameters = inspect.signature(smooth_profile).parameters

    assert parameters["plateau"].default is False
    assert parameters["plateau_tolerance"].default == 0.05
    assert parameters["extrapolate"].default is False
    assert parameters["minimum_plateau_points"].default == 2

    bulk_parameters = inspect.signature(DiffusivityAnalysis.select_profile).parameters
    assert bulk_parameters["plateau_tolerance"].default == 0.05
    assert bulk_parameters["minimum_plateau_points"].default == 2


def test_high_level_estimators_use_sampling_period() -> None:
    single = estimate_diffusivity(
        _trajectory(),
        center=0.0,
        interval_width=1.0,
        sampling_period=0.1,
    )
    assert single.sampling_period == 0.1

    analysis = analyze_diffusivity(
        _trajectory(),
        positions=[0.0],
        interval_widths=[1.0],
        sampling_period=0.1,
        bounds=(-2.0, 2.0),
        uncertainty="blocking",
    )
    assert isinstance(single, DiffusivityEstimate)
    assert isinstance(analysis, DiffusivityAnalysis)
    assert analysis.excess_nll.shape == analysis.diffusivities.shape
    assert analysis.sampling_period == 0.1
    assert np.isfinite(analysis.diffusivity_sems[0, 0])
    assert "1 positions" in analysis.summary()


def test_simulation_windows_accept_arrays_and_reject_files(tmp_path) -> None:
    path = tmp_path / "trajectory.z"
    np.savetxt(path, _trajectory())
    with pytest.raises(TypeError, match="load trajectory files first"):
        SimulationWindow(path, bounds=(-2.0, 2.0))

    analysis = analyze_diffusivity(
        [
            SimulationWindow(np.loadtxt(path), bounds=(-2.0, 2.0)),
            SimulationWindow(_trajectory(), bounds=(-2.0, 2.0)),
        ],
        positions=[0.0],
        interval_widths=[1.0],
        sampling_period=0.1,
    )
    assert np.isfinite(analysis.diffusivities[0, 0])

    replicas = analyze_diffusivity(
        np.vstack((_trajectory(), _trajectory())),
        positions=[0.0],
        interval_widths=[1.0],
        sampling_period=0.1,
        bounds=(-2.0, 2.0),
    )
    assert np.isfinite(replicas.diffusivities[0, 0])


def test_results_offer_selection_navigation() -> None:
    analysis = analyze_diffusivity(
        _trajectory(),
        positions=[0.0],
        interval_widths=[1.0],
        sampling_period=0.1,
        bounds=(-2.0, 2.0),
    )
    selected = analysis.constant_width_minima()[0]
    assert isinstance(selected, DiffusivityProfile)
    assert selected.interval_width == 1.0
    assert np.array_equal(selected.positions, analysis.positions)
    assert selected.nll_rank == 0
    assert analysis.profile_at_width(1.0).diffusivity.shape == (1,)
    assert analysis.at_position(0.0)["diffusivity"].shape == (1,)


@pytest.mark.parametrize("windowed", [False, True])
def test_analysis_accepts_generators_of_trajectories_and_windows(windowed):
    sources = [_trajectory(), _trajectory()]
    options = dict(positions=[0.0], interval_widths=[1.0], sampling_period=0.1)
    if windowed:
        sources = [SimulationWindow(values, (-2.0, 2.0)) for values in sources]
    else:
        options["bounds"] = (-2.0, 2.0)
    expected = analyze_diffusivity(sources, **options)
    actual = analyze_diffusivity((source for source in sources), **options)
    np.testing.assert_array_equal(actual.diffusivities, expected.diffusivities)
    np.testing.assert_array_equal(actual.n_time_origins, expected.n_time_origins)


def test_public_profile_and_propagation_workflow() -> None:
    positions = np.linspace(3.0, 7.0, 9)
    values = 1.0 + (positions - 5.0) ** 2
    profile = smooth_profile(
        positions,
        values,
        positive=True,
        symmetry_center=5.0,
        smoothing_length=0.5,
    )
    assert profile.domain == (3.0, 7.0)
    assert np.allclose(profile([4.0, 6.0]), profile([6.0, 4.0]))

    positions = np.linspace(-1.0, 1.0, 5)
    propagated = propagate_density(
        positions,
        np.ones_like(positions),
        reduced_potential=np.zeros_like(positions),
        times=0.0,
        initial_position=0.1,
    )
    assert isinstance(propagated, PropagationResult)
    assert np.argmax(propagated.densities[0]) == 2

    metric = compare_densities(
        positions,
        np.ones_like(positions),
        np.ones_like(positions),
        normalization="normalize",
    )
    assert metric.total_variation == 0.0
