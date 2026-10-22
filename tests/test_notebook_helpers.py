"""Notebook orchestration uses the same scientific inputs with or without caches."""

import json
import shutil
from dataclasses import replace
from types import SimpleNamespace

import analysis_helpers as helpers
import numpy as np
import pytest


@pytest.fixture(params=["path", "windows"])
def offline_analysis(tmp_path, request):
    raw = tmp_path / "raw"
    raw.mkdir()
    paths = [raw / "first.z", raw / "second.z"]
    for path in paths:
        np.savetxt(path, [-2.0, 0.0, 0.0, 2.0] * 4)
    options = dict(positions=[0.0], interval_widths=[1.0], sampling_period=0.1)
    if request.param == "path":
        source = paths[0]
        options["bounds"] = (-2.0, 2.0)
    else:
        source = [helpers.TrajectoryWindow(p, (-2.0, 2.0)) for p in paths]
    cache = tmp_path / "cache"
    first = helpers.load_or_analyze_diffusivity(cache, source, **options)
    return cache, source, options, first, paths


def test_missing_raw_directory_reuses_complete_cache(offline_analysis, monkeypatch):
    cache, source, options, first, paths = offline_analysis
    for path in paths:
        path.unlink()
    paths[0].parent.rmdir()

    def unexpected_compute(*args, **kwargs):
        pytest.fail("offline reuse must not load or analyze raw data")

    monkeypatch.setattr(helpers, "_load_analysis_trajectories", unexpected_compute)
    monkeypatch.setattr(helpers, "analyze_diffusivity", unexpected_compute)
    before = (cache / "metadata.json").read_bytes()
    with pytest.warns(UserWarning, match="source freshness cannot be verified"):
        cached = helpers.load_or_analyze_diffusivity(cache, source, **options)
    np.testing.assert_array_equal(cached.diffusivities, first.diffusivities)
    np.testing.assert_array_equal(cached.excess_nll, first.excess_nll)
    assert (cache / "metadata.json").read_bytes() == before


def test_committed_cache_works_from_another_checkout_without_raw(tmp_path, monkeypatch):
    original = tmp_path / "original"
    relocated = tmp_path / "relocated"
    raw = original / "data/raw/example/trajectory.z"
    raw.parent.mkdir(parents=True)
    np.savetxt(raw, [-2.0, 0.0, 0.0, 2.0] * 4)
    cache = original / "data/cache/example"
    options = dict(
        positions=[0.0],
        interval_widths=[1.0],
        sampling_period=0.1,
        bounds=(-2.0, 2.0),
    )
    monkeypatch.setattr(helpers, "_PROJECT_ROOT", original)
    first = helpers.load_or_analyze_diffusivity(cache, raw, **options)
    metadata = json.loads((cache / "metadata.json").read_text())
    assert metadata["analysis_inputs"]["trajectories"]["path"] == (
        "data/raw/example/trajectory.z"
    )

    relocated_cache = relocated / "data/cache/example"
    shutil.copytree(cache, relocated_cache)
    monkeypatch.setattr(helpers, "_PROJECT_ROOT", relocated)
    missing_raw = relocated / "data/raw/example/trajectory.z"
    with pytest.warns(UserWarning, match="source freshness cannot be verified"):
        restored = helpers.load_or_analyze_diffusivity(
            relocated_cache, missing_raw, **options
        )
    np.testing.assert_array_equal(restored.diffusivities, first.diffusivities)


@pytest.mark.parametrize(
    "change",
    [
        "sampling_period",
        "positions",
        "interval_widths",
        "symmetry_center",
        "uncertainty",
        "source_path",
        "bounds",
        "missing_cache",
        "incomplete_cache",
        "invalid_version",
    ],
)
def test_offline_cache_rejects_mismatches_and_missing_results(offline_analysis, change):
    cache, source, options, _, paths = offline_analysis
    for path in paths:
        path.unlink()
    if change in {
        "sampling_period",
        "positions",
        "interval_widths",
        "symmetry_center",
        "uncertainty",
    }:
        options[change] = {
            "sampling_period": 0.2,
            "positions": [0.1],
            "interval_widths": [2.0],
            "symmetry_center": 0.0,
            "uncertainty": None,
        }[change]
    elif change == "source_path":
        source = (
            paths[0].with_name("other.z")
            if not isinstance(source, list)
            else [replace(source[0], path=paths[0].with_name("other.z")), source[1]]
        )
    elif change == "bounds":
        if isinstance(source, list):
            source = [replace(source[0], bounds=(-3.0, 3.0)), source[1]]
        else:
            options["bounds"] = (-3.0, 3.0)
    elif change == "missing_cache":
        cache = cache / "absent"
    elif change == "incomplete_cache":
        (cache / "diffusivity_estimates.parquet").unlink()
    else:
        path = cache / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["format_version"] = -1
        path.write_text(json.dumps(metadata))
    with pytest.raises(FileNotFoundError, match="no usable matching analysis cache"):
        helpers.load_or_analyze_diffusivity(cache, source, **options)


def test_offline_reuse_still_checks_available_files(tmp_path):
    paths = [tmp_path / "first.z", tmp_path / "second.z"]
    for path in paths:
        np.savetxt(path, [-2.0, 0.0, 2.0] * 4)
    sources = [helpers.TrajectoryWindow(p, (-2.0, 2.0)) for p in paths]
    options = dict(positions=[0.0], interval_widths=[1.0], sampling_period=0.1)
    helpers.load_or_analyze_diffusivity(tmp_path / "cache", sources, **options)
    paths[0].unlink()
    with pytest.warns(UserWarning, match="1 trajectory file"):
        helpers.load_or_analyze_diffusivity(tmp_path / "cache", sources, **options)
    np.savetxt(paths[1], [-2.0, 0.0, 0.0, 2.0] * 8)
    with pytest.raises(FileNotFoundError, match="no usable matching analysis cache"):
        helpers.load_or_analyze_diffusivity(tmp_path / "cache", sources, **options)


@pytest.fixture(params=[(), ("vacf",), ("pacf",), ("vacf", "pacf")])
def prepared_inputs(tmp_path, request):
    x = np.linspace(-4.0, 4.0, 17)
    for name in request.param:
        np.savetxt(tmp_path / f"diffusivity_{name}.txt", np.c_[x, np.ones_like(x)])
    np.savetxt(tmp_path / "pmf_abf.txt", np.c_[x, np.zeros_like(x)])
    (tmp_path / "md").mkdir()
    np.savetxt(tmp_path / "md/propagator_0001ps.txt", np.c_[x, np.exp(-(x**2))])
    minima = [
        SimpleNamespace(
            positions=x,
            diffusivity=np.full_like(x, 0.5 + rank),
            interval_width=float(rank + 1),
            nll_rank=rank,
        )
        for rank in range(5)
    ]
    inputs = helpers.prepare_propagator_inputs(
        tmp_path,
        minima,
        minima[4],
        box_length=8.0,
        temperature=300.0,
        target_spacing=0.5,
        smoothing_length=1.0,
        symmetric=True,
        plateau=True,
        extrapolate=True,
    )
    assert len(inputs.profiles) == 4 + len(request.param)
    assert {name for name in inputs.profiles if not name.startswith("rta_")} == {
        name.upper() for name in request.param
    }
    assert sum(p.selected for p in inputs.profiles.values()) == 1
    return inputs


def test_comparison_cache_preserves_inputs_and_ignores_presentation(
    tmp_path,
    prepared_inputs,
    monkeypatch,
):
    first = helpers.load_or_compute_propagator_comparison(prepared_inputs, tmp_path)
    restyled = replace(
        prepared_inputs,
        profiles={
            name: replace(profile, selected=not profile.selected)
            for name, profile in prepared_inputs.profiles.items()
        },
    )

    def unexpected_compute(*args):
        pytest.fail("presentation-only changes must not recompute propagation")

    monkeypatch.setattr(helpers, "_compute_propagator_comparison", unexpected_compute)
    cached = helpers.load_or_compute_propagator_comparison(restyled, tmp_path)
    assert cached.inputs is restyled
    assert cached.metric_rows == first.metric_rows
    for name in first.model_propagators:
        for lag in prepared_inputs.lag_times:
            np.testing.assert_array_equal(
                cached.model_propagators[name][lag], first.model_propagators[name][lag]
            )


def test_reference_profiles_can_use_a_distinct_smoothing_length(tmp_path, monkeypatch):
    x = np.linspace(-4.0, 4.0, 17)
    for name in ("vacf", "pacf"):
        np.savetxt(tmp_path / f"diffusivity_{name}.txt", np.c_[x, np.ones_like(x)])
    np.savetxt(tmp_path / "pmf_abf.txt", np.c_[x, np.zeros_like(x)])
    (tmp_path / "md").mkdir()
    np.savetxt(tmp_path / "md/propagator_0001ps.txt", np.c_[x, np.ones_like(x)])
    candidate = SimpleNamespace(
        positions=x,
        diffusivity=np.ones_like(x),
        interval_width=1.0,
        nll_rank=0,
    )
    smoothing_lengths = []

    def record_smoothing_length(x, y, **options):
        smoothing_lengths.append(options["smoothing_length"])
        return lambda positions: np.ones_like(positions)

    monkeypatch.setattr(helpers, "smooth_profile", record_smoothing_length)
    helpers.prepare_propagator_inputs(
        tmp_path,
        [candidate],
        candidate,
        box_length=8.0,
        temperature=300.0,
        smoothing_length=1.0,
        reference_smoothing_length=2.0,
    )

    assert smoothing_lengths == [2.0, 2.0, 1.0]


def test_comparison_refreshes_changed_inputs_and_incomplete_cache(
    tmp_path,
    prepared_inputs,
    monkeypatch,
):
    helpers.load_or_compute_propagator_comparison(prepared_inputs, tmp_path)
    compute = helpers._compute_propagator_comparison
    calls = []

    def counted(inputs):
        calls.append(inputs)
        return compute(inputs)

    monkeypatch.setattr(helpers, "_compute_propagator_comparison", counted)
    changed = replace(prepared_inputs, reduced_potential=prepared_inputs.positions**2)
    helpers.load_or_compute_propagator_comparison(changed, tmp_path)
    (tmp_path / "propagators.parquet").unlink()
    helpers.load_or_compute_propagator_comparison(changed, tmp_path)
    assert len(calls) == 2


def test_window_cache_hit_does_not_read_trajectory(tmp_path, monkeypatch):
    path = tmp_path / "trajectory.z"
    np.savetxt(path, [-2.0, 0.0, 0.0, 2.0] * 4)
    window = helpers.TrajectoryWindow(path, (-2.0, 2.0))
    options = dict(positions=[0.0], interval_widths=[1.0], sampling_period=0.1)
    first = helpers.load_or_analyze_diffusivity(tmp_path / "cache", [window], **options)

    def unexpected_read(*args, **kwargs):
        pytest.fail("a cache hit must not load trajectory files")

    monkeypatch.setattr(np, "loadtxt", unexpected_read)
    cached = helpers.load_or_analyze_diffusivity(
        tmp_path / "cache", [window], **options
    )
    np.testing.assert_array_equal(cached.diffusivities, first.diffusivities)


def test_analysis_checks_metadata_before_loading_cached_arrays(tmp_path, monkeypatch):
    trajectory = np.array([-2.0, 0.0, 0.0, 2.0] * 4)
    options = dict(positions=[0.0], interval_widths=[1.0], bounds=(-2.0, 2.0))
    first = helpers.load_or_analyze_diffusivity(
        tmp_path, trajectory, sampling_period=0.1, **options
    )

    def unexpected_read(*args):
        pytest.fail("a mismatching cache must not load cached arrays")

    monkeypatch.setattr(helpers, "read_diffusivity_analysis_cache", unexpected_read)
    changed = helpers.load_or_analyze_diffusivity(
        tmp_path, trajectory, sampling_period=0.2, **options
    )
    np.testing.assert_allclose(changed.diffusivities, first.diffusivities / 2)


def test_plots_use_prepared_arrays_and_interpolate_missing_estimates(
    tmp_path,
    prepared_inputs,
    monkeypatch,
):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotting_helpers as plots

    monkeypatch.setattr(plt, "show", lambda: None)
    try:
        plots.plot_propagator_inputs(prepared_inputs)
        axes = plt.gcf().axes
        np.testing.assert_array_equal(
            axes[0].lines[0].get_ydata(), prepared_inputs.reduced_potential
        )
        for line, profile in zip(axes[1].lines, prepared_inputs.profiles.values()):
            np.testing.assert_array_equal(line.get_ydata(), profile.diffusivity)
            if profile.selected:
                assert line.get_color() == plots.H2
                assert "$_{4}$" in line.get_label()
        comparison = helpers.load_or_compute_propagator_comparison(
            prepared_inputs, tmp_path
        )
        plots.plot_propagators(comparison)
        assert len(plt.gcf().axes[0].lines) == 1 + len(prepared_inputs.profiles)
        plots.plot_density_metrics(comparison)
        assert all(
            len(axis.lines) == len(prepared_inputs.profiles) for axis in plt.gcf().axes
        )
        candidate = SimpleNamespace(
            positions=np.arange(-40.0, 50.0, 10.0),
            diffusivity=np.array([5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            interval_width=1.0,
        )
        analysis = SimpleNamespace(
            positions=candidate.positions,
            interval_widths=np.array([1.0, 2.0]),
            diffusivities=np.array([candidate.diffusivity, candidate.diffusivity]),
            diffusivity_sems=np.full((2, 9), 0.1),
            symmetry_center=0.0,
        )
        profile_selection = SimpleNamespace(
            origin=0.0,
            spatial_plateau_starts=np.array([20.0, np.nan]),
            spatial_plateau_point_counts=np.array([3, 0]),
            spatial_plateau_relative_trends=np.array([0.0, np.nan]),
            spatial_plateau_relative_scatters=np.array([0.0, np.nan]),
            bulk_diffusivities=np.array([4.0, np.nan]),
            limiting_bulk_diffusivity=4.0,
            minimum_bulk_width=1.0,
            maximum_bulk_width=1.0,
            interval_width=2.0,
            plateau_tolerance=0.05,
        )
        with pytest.raises(
            ValueError, match="selected width 15 is not in the analysis"
        ):
            plots.plot_diffusivity_profiles(
                analysis, profile_selection, selected_widths=(1.0, 15.0)
            )
        figure, axis = plots.plot_diffusivity_profiles(
            analysis, profile_selection, selected_widths=(1.0, 2.0)
        )
        curves = [line for line in axis.lines if line.get_linestyle() == "-"]
        assert len(curves) == 2
        assert all(len(line.get_xdata()) == 400 for line in curves)
        assert all(np.all(np.isfinite(line.get_ydata())) for line in curves)
        assert all(np.all(line.get_xdata() >= 0.0) for line in curves)
        assert all(line.get_xdata()[0] == 0.0 for line in curves)
        for line in curves:
            x = line.get_xdata()
            y = line.get_ydata()
            endpoint_slope = (-3 * y[0] + 4 * y[1] - y[2]) / (2 * (x[1] - x[0]))
            assert abs(endpoint_slope) < 1e-3
        assert len(figure.axes) == 1
        for container in axis.containers:
            markers = container.lines[0]
            assert np.all(markers.get_xdata() >= 0)
            expected = candidate.diffusivity[
                np.searchsorted(candidate.positions, markers.get_xdata())
            ]
            np.testing.assert_array_equal(markers.get_ydata(), expected)
            half_widths = [
                (segment[1, 1] - segment[0, 1]) / 2
                for segment in container.lines[2][0].get_segments()
            ]
            np.testing.assert_allclose(half_widths, 1.959963984540054 * 0.1)
        filled = []
        for container in axis.containers:
            marker = container.lines[0]
            facecolor = marker.get_markerfacecolor()
            if not isinstance(facecolor, str) or facecolor != "none":
                filled.append(marker)
        assert len(filled) == 1
        np.testing.assert_array_equal(filled[0].get_xdata(), [20.0, 30.0, 40.0])
        np.testing.assert_allclose(curves[0].get_ydata()[-10:], 4.0)
        plots.plot_bulk_convergence(analysis, profile_selection)
        assert len(plt.gcf().axes) == 1
        convergence_axis = plt.gcf().axes[0]
        assert convergence_axis.get_title(loc="left") == "Bulk-convergence detection"
        np.testing.assert_array_equal(convergence_axis.lines[0].get_xdata(), [1.0])
        np.testing.assert_array_equal(convergence_axis.lines[0].get_ydata(), [4.0])
        assert convergence_axis.get_ylabel()
        assert not any(
            line.get_label().startswith("Selected") for line in convergence_axis.lines
        )
    finally:
        plt.close("all")


def test_default_cache_computes_sems_for_every_width(tmp_path):
    from cache_helpers import read_diffusivity_analysis_cache

    trajectory = np.random.default_rng(17).normal(size=4096)
    options = dict(
        positions=[-0.25, 0.0, 0.25],
        interval_widths=[0.5, 1.0, 1.5],
        sampling_period=0.1,
        bounds=(-3.0, 3.0),
    )
    # A cache built without uncertainties must be refreshed by the new default.
    helpers.load_or_analyze_diffusivity(
        tmp_path, trajectory, uncertainty=None, **options
    )
    analysis = helpers.load_or_analyze_diffusivity(tmp_path, trajectory, **options)
    assert analysis.diffusivity_sems.shape == analysis.diffusivities.shape
    assert np.all(np.isfinite(analysis.diffusivity_sems))
    assert np.all(analysis.diffusivity_sems >= 0)
    assert np.all(np.any(analysis.diffusivity_sems > 0, axis=1))
    cached = read_diffusivity_analysis_cache(tmp_path)
    np.testing.assert_array_equal(
        cached.analysis.diffusivity_sems, analysis.diffusivity_sems
    )
    for profile in cached.constant_width_minima:
        row = np.flatnonzero(analysis.interval_widths == profile.interval_width)[0]
        np.testing.assert_array_equal(
            profile.diffusivity_sem, analysis.diffusivity_sems[row]
        )


def test_window_trajectory_paths_remain_file_backed(tmp_path):
    path = tmp_path / "trajectory.z"
    np.savetxt(path, [-2.0, 0.0, 2.0])

    window = helpers._load_analysis_trajectories(
        helpers.TrajectoryWindow(path, (-2.0, 2.0))
    )

    assert window.paths == (path,)
    assert (window.domain.lower, window.domain.upper) == (-2.0, 2.0)
