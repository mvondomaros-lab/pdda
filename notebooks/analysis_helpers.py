"""File-backed analysis and prepared propagation comparisons for the notebooks."""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from cache_helpers import (
    DIFFUSIVITY_ANALYSIS_CACHE_FORMAT_VERSION,
    PROPAGATOR_CACHE_FORMAT_VERSION,
    read_diffusivity_analysis_cache,
    read_propagator_cache,
    write_diffusivity_analysis_cache,
    write_propagator_cache,
)
from comparison_types import ComparisonProfile, PropagatorComparison, PropagatorInputs
from numpy.typing import ArrayLike, NDArray

from pdda import (
    DiffusivityAnalysis,
    DiffusivityProfile,
    PlateauNotFoundWarning,
    SimulationWindow,
    analyze_diffusivity,
    periodic_grid,
    smooth_profile,
)
from pdda.profiles import (
    DensityComparator,
    symmetrize_even_profile,
)
from pdda.propagators import propagate_diffusivity_profiles
from pdda.rta import RTAWindow, WallDomain

_BOLTZMANN_KCAL_PER_MOL_K = 0.0019872041
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TrajectoryWindow:
    """A trajectory file and its physical bounds, loaded only on a cache miss."""

    path: str | Path
    bounds: tuple[float, float]


def regular_positions(
    bounds: tuple[float, float], spacing: float
) -> NDArray[np.float64]:
    """Return regularly spaced interval centers that stay inside ``bounds``."""
    lower, upper = bounds
    return np.arange(
        np.ceil((lower + spacing / 2.0) / spacing) * spacing,
        np.floor((upper - spacing / 2.0) / spacing) * spacing + spacing / 2.0,
        spacing,
    )


def load_or_analyze_diffusivity(
    cache_directory: str | Path,
    trajectories: object,
    *,
    positions: ArrayLike,
    interval_widths: ArrayLike,
    sampling_period: float,
    bounds: tuple[float, float] | None = None,
    symmetry_center: float | None = None,
    uncertainty: str | None = "blocking",
) -> DiffusivityAnalysis:
    """Load a matching analysis or calculate and cache it.

    Missing trajectory files permit cache-only use if settings, file paths,
    and all available source fingerprints match. A warning reports that the
    missing sources' freshness cannot be verified. Without a usable matching
    cache, restore the trajectory files before analyzing.

    Blocking SEMs are computed and cached for every interval width by default.
    Pass ``uncertainty=None`` to explicitly omit uncertainty estimation.
    """
    positions = np.asarray(positions, dtype=float)
    interval_widths = np.asarray(interval_widths, dtype=float)
    missing: list[str] = []
    configuration = {
        "positions": positions.tolist(),
        "interval_widths": interval_widths.tolist(),
        "sampling_period": float(sampling_period),
        "bounds": None if bounds is None else list(bounds),
        "window_bounds": _window_bounds(trajectories),
        "symmetry_center": symmetry_center,
        "uncertainty": uncertainty,
        "trajectories": _source_signature(trajectories, missing=missing),
    }
    cache_directory = Path(cache_directory)
    if (cache_directory / "metadata.json").exists():
        try:
            metadata = json.loads((cache_directory / "metadata.json").read_text())
            if (
                metadata.get("format_version")
                == DIFFUSIVITY_ANALYSIS_CACHE_FORMAT_VERSION
            ):
                stored_inputs = metadata.get("analysis_inputs")
                if _cache_inputs_match(configuration, stored_inputs):
                    analysis = read_diffusivity_analysis_cache(cache_directory).analysis
                else:
                    analysis = None
                if analysis is not None:
                    if missing:
                        warnings.warn(
                            f"Using cached analysis: {len(missing)} trajectory file(s) "
                            "are unavailable; source freshness cannot be verified.",
                            UserWarning,
                            stacklevel=2,
                        )
                    return analysis
        except (OSError, ValueError, KeyError):
            pass

    if missing:
        raise FileNotFoundError(
            "Raw trajectories are unavailable and no usable matching analysis "
            "cache was found. Restore the raw files to compute these settings: "
            + ", ".join(missing)
        )
    analysis = analyze_diffusivity(
        _load_analysis_trajectories(trajectories),
        positions=positions,
        interval_widths=interval_widths,
        sampling_period=sampling_period,
        bounds=bounds,
        symmetry_center=symmetry_center,
        uncertainty=uncertainty,
    )
    write_diffusivity_analysis_cache(
        cache_directory,
        analysis,
        metadata={"analysis_inputs": configuration},
    )
    return analysis


def prepare_propagator_inputs(
    data_directory: str | Path,
    nll_minima: Iterable[DiffusivityProfile],
    selected_profile: DiffusivityProfile,
    *,
    box_length: float,
    temperature: float,
    target_spacing: float = 0.1,
    smoothing_length: float | None = None,
    reference_smoothing_length: float | None = None,
    symmetric: bool = False,
    plateau: bool = False,
    plateau_tolerance: float = 0.05,
    extrapolate: bool = False,
) -> PropagatorInputs:
    """Prepare the same PMF, diffusivities, and MD densities for all comparisons.

    Coordinates are in Å, times in ps, and the source PMF in kcal/mol.
    Symmetry, when requested, is reflection about the cell center at zero.
    VACF and PACF reference profiles are included independently when their
    ``diffusivity_vacf.txt`` / ``diffusivity_pacf.txt`` files exist. Their
    smoothing length can be overridden with ``reference_smoothing_length``;
    when omitted, it falls back to ``smoothing_length``. Missing references
    are omitted from propagation, metrics, plots, and caches.
    """
    directory = Path(data_directory)
    grid, spacing = periodic_grid(box_length, target_spacing)
    pmf = np.loadtxt(directory / "pmf_abf.txt")
    pmf_x, pmf_y = pmf[:, 0], pmf[:, 1] / (_BOLTZMANN_KCAL_PER_MOL_K * temperature)
    if symmetric:
        pmf_x, pmf_y = symmetrize_even_profile(pmf_x, pmf_y)
    potential = _periodic_interpolate(pmf_x, pmf_y, grid, box_length)
    profiles = {}

    def fit(
        source,
        name,
        rank=None,
        width=None,
        selected=False,
        profile_smoothing_length=None,
    ):
        values = _profile_within_periodic_cell(_load_profile(source), box_length)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", PlateauNotFoundWarning)
            fitted = smooth_profile(
                values[:, 0],
                values[:, 1],
                positive=True,
                symmetry_center=0.0 if symmetric else None,
                smoothing_length=profile_smoothing_length,
                plateau=plateau,
                plateau_tolerance=plateau_tolerance,
                extrapolate=extrapolate,
            )
        for warning in caught:
            warnings.warn(f"{name}: {warning.message}", warning.category, stacklevel=2)
        profiles[name] = ComparisonProfile(fitted(grid), rank, width, selected)

    for name in ("VACF", "PACF"):
        path = directory / f"diffusivity_{name.lower()}.txt"
        if path.exists():
            fit(
                path,
                name,
                profile_smoothing_length=(
                    smoothing_length
                    if reference_smoothing_length is None
                    else reference_smoothing_length
                ),
            )
    for candidate in nll_minima:
        selected = candidate.interval_width == selected_profile.interval_width
        if candidate.nll_rank >= 3 and not selected:
            continue
        fit(
            np.column_stack((candidate.positions, candidate.diffusivity)),
            f"rta_L{candidate.interval_width:g}",
            candidate.nll_rank,
            candidate.interval_width,
            selected,
            profile_smoothing_length=smoothing_length,
        )
    md = {}
    for lag, (positions, density) in _load_md_propagators(directory / "md").items():
        values = _resample_density(positions, density, grid, spacing)
        md[lag] = _symmetrize_density(values, spacing) if symmetric else values
    return PropagatorInputs(grid, spacing, potential, profiles, md)


def load_or_compute_propagator_comparison(
    inputs: PropagatorInputs,
    cache_directory: str | Path,
) -> PropagatorComparison:
    """Reuse predictions for identical numerical inputs, or compute and save them."""
    directory = Path(cache_directory)
    expected = {
        "positions": _source_signature(inputs.positions),
        "reduced_potential": _source_signature(inputs.reduced_potential),
        "diffusivities": {
            name: _source_signature(profile.diffusivity)
            for name, profile in inputs.profiles.items()
        },
        "md_propagators": {
            str(lag): _source_signature(density)
            for lag, density in inputs.md_propagators.items()
        },
    }
    try:
        metadata = json.loads((directory / "propagator_metadata.json").read_text())
        if metadata.get("format_version") == PROPAGATOR_CACHE_FORMAT_VERSION and all(
            metadata.get(key) == value for key, value in expected.items()
        ):
            return read_propagator_cache(directory, inputs)
    except (OSError, ValueError, KeyError):
        pass
    comparison = _compute_propagator_comparison(inputs)
    write_propagator_cache(directory, comparison, metadata=expected)
    return comparison


def _compute_propagator_comparison(inputs: PropagatorInputs) -> PropagatorComparison:
    propagated = propagate_diffusivity_profiles(
        inputs.positions,
        inputs.reduced_potential,
        {name: profile.diffusivity for name, profile in inputs.profiles.items()},
        inputs.lag_times,
        periodic=True,
    )
    model_propagators = {
        name: dict(zip(inputs.lag_times, result.densities, strict=True))
        for name, result in propagated.items()
    }
    comparator = DensityComparator(inputs.positions, periodic=True, normalize=True)
    rows = []
    for lag in inputs.lag_times:
        for name, densities in model_propagators.items():
            metric = comparator.compare(inputs.md_propagators[lag], densities[lag])
            rows.append(
                {
                    "lag_ps": lag,
                    "model": name,
                    "total_variation": metric.total_variation,
                    "wasserstein_1_angstrom": metric.wasserstein_1,
                    "hellinger": metric.hellinger,
                    "jensen_shannon": metric.jensen_shannon,
                    "kl_reference_model": metric.kl_reference_model,
                    "renyi_2": metric.renyi_2,
                }
            )
    return PropagatorComparison(inputs, model_propagators, tuple(rows))


def _load_profile(source: ArrayLike | str | Path) -> NDArray[np.float64]:
    return (
        np.loadtxt(source)
        if isinstance(source, (str, Path))
        else np.asarray(source, dtype=float)
    )


def _window_bounds(trajectories: object) -> list[list[float]] | None:
    if isinstance(trajectories, (SimulationWindow, TrajectoryWindow)):
        return [list(trajectories.bounds)]
    if (
        isinstance(trajectories, (list, tuple))
        and trajectories
        and all(
            isinstance(item, (SimulationWindow, TrajectoryWindow))
            for item in trajectories
        )
    ):
        return [list(item.bounds) for item in trajectories]
    return None


def _load_analysis_trajectories(trajectories: object) -> object:
    """Keep windowed trajectory paths file-backed for streaming analysis."""
    if isinstance(trajectories, TrajectoryWindow):
        return RTAWindow(trajectories.path, WallDomain(*trajectories.bounds))
    if isinstance(trajectories, (list, tuple)):
        return [_load_analysis_trajectories(item) for item in trajectories]
    if isinstance(trajectories, (str, Path)):
        return np.loadtxt(trajectories)
    return trajectories


def _cache_inputs_match(current: object, stored: object) -> bool:
    """Match complete settings, allowing only missing files' stat fields to differ."""
    if isinstance(current, dict):
        if not isinstance(stored, dict):
            return False
        if set(current) == {"path"}:
            return (
                set(stored) == {"path", "size", "modified_ns"}
                and current["path"] == stored["path"]
            )
        return current.keys() == stored.keys() and all(
            _cache_inputs_match(value, stored[key]) for key, value in current.items()
        )
    if isinstance(current, list):
        return (
            isinstance(stored, list)
            and len(current) == len(stored)
            and all(
                _cache_inputs_match(a, b) for a, b in zip(current, stored, strict=True)
            )
        )
    return current == stored


def _source_signature(source: object, *, missing: list[str] | None = None) -> Any:
    if isinstance(source, TrajectoryWindow):
        return {
            "bounds": list(source.bounds),
            "trajectories": _source_signature(source.path, missing=missing),
        }
    if isinstance(source, SimulationWindow):
        return {
            "bounds": list(source.bounds),
            "trajectories": _source_signature(source.trajectories, missing=missing),
        }
    if isinstance(source, (list, tuple)):
        return [_source_signature(item, missing=missing) for item in source]
    if isinstance(source, (str, Path)):
        path = Path(source)
        resolved = path.resolve()
        try:
            source_path = resolved.relative_to(_PROJECT_ROOT).as_posix()
        except ValueError:
            source_path = str(resolved)
        try:
            stat = path.stat()
        except FileNotFoundError:
            if missing is None:
                raise
            missing.append(str(resolved))
            return {"path": source_path}
        return {
            "path": source_path,
            "size": stat.st_size,
            "modified_ns": stat.st_mtime_ns,
        }
    values = np.asarray(source)
    digest = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
    return {"shape": list(values.shape), "dtype": str(values.dtype), "sha256": digest}


def _profile_within_periodic_cell(
    profile: NDArray[np.float64], box_length: float
) -> NDArray[np.float64]:
    half_box = 0.5 * box_length
    return profile[np.abs(profile[:, 0]) <= half_box]


def _periodic_interpolate(
    x: ArrayLike, y: ArrayLike, target: NDArray[np.float64], period: float
) -> NDArray[np.float64]:
    x_values, y_values = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return np.interp(
        target,
        np.r_[x_values[-1] - period, x_values, x_values[0] + period],
        np.r_[y_values[-1], y_values, y_values[0]],
    )


def _resample_density(
    x: ArrayLike, density: ArrayLike, target: NDArray[np.float64], spacing: float
) -> NDArray[np.float64]:
    values = np.maximum(np.interp(target, x, density), 0.0)
    return values / (values.sum() * spacing)


def _symmetrize_density(
    density: NDArray[np.float64], spacing: float
) -> NDArray[np.float64]:
    values = 0.5 * (density + density[::-1])
    return values / (values.sum() * spacing)


def _load_md_propagators(
    directory: Path,
) -> dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    values: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    for path in sorted(directory.glob("propagator_*ps.txt")):
        lag = int(path.stem.removeprefix("propagator_").removesuffix("ps"))
        table = np.loadtxt(path)
        values[lag] = table[:, 0], table[:, 1]
    if not values:
        raise ValueError(f"no MD propagators found in {directory}")
    return values
