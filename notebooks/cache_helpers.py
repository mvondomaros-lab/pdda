"""Notebook cache storage for diffusivity analyses and propagator comparisons."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from comparison_types import PropagatorComparison, PropagatorInputs
from numpy.typing import NDArray

from pdda.rta import (
    DiffusivityAnalysis,
    DiffusivityProfile,
    FirstExitHistogramGrid,
)

__all__ = [
    "DIFFUSIVITY_ANALYSIS_CACHE_FORMAT_VERSION",
    "PROPAGATOR_CACHE_FORMAT_VERSION",
    "DiffusivityAnalysisCache",
    "read_diffusivity_analysis_cache",
    "read_propagator_cache",
    "write_diffusivity_analysis_cache",
    "write_propagator_cache",
]

DIFFUSIVITY_ANALYSIS_CACHE_FORMAT_VERSION = 10
PROPAGATOR_CACHE_FORMAT_VERSION = 5
_ESTIMATES_FILENAME = "diffusivity_estimates.parquet"
_MINIMA_FILENAME = "constant_nll_minima.parquet"
_OBSERVED_FILENAME = "observed_exit_histograms.parquet"
_CENSORED_FILENAME = "terminal_censor_histograms.parquet"
_METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True)
class DiffusivityAnalysisCache:
    """A cached analysis, all ranked constant-width minima, and provenance."""

    analysis: DiffusivityAnalysis
    constant_width_minima: tuple[DiffusivityProfile, ...]
    metadata: dict[str, Any]


def write_diffusivity_analysis_cache(
    directory: str | Path,
    analysis: DiffusivityAnalysis,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write a complete position-dependent diffusivity analysis.

    Every local minimum of the mean excess-NLL curve is ranked and stored
    alongside all constant-width estimates. Profiles are not duplicated: each
    minimum refers to its interval-width row in the analysis.
    The sparse segment-length histograms preserve the complete discrete
    first-exit data, not merely a display-resolution survival curve.
    """
    pa, pq = _pyarrow()
    if analysis.histograms is None:
        raise ValueError("the analysis does not contain exit histograms")
    minima = analysis.constant_width_minima()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    shape = (analysis.interval_widths.size, analysis.positions.size)
    if analysis.diffusivities.shape != shape:
        raise ValueError("analysis arrays must match interval widths by positions")
    width_index, center_index = np.indices(shape)
    sems = (
        np.full(shape, np.nan)
        if analysis.diffusivity_sems is None
        else analysis.diffusivity_sems
    )
    pq.write_table(
        pa.table(
            {
                "width_index": width_index.ravel().astype(np.int32),
                "center_index": center_index.ravel().astype(np.int32),
                "interval_width": analysis.interval_widths[width_index].ravel(),
                "position": analysis.positions[center_index].ravel(),
                "diffusivity": analysis.diffusivities.ravel(),
                "diffusivity_sem": sems.ravel(),
                "excess_nll": analysis.excess_nll.ravel(),
                "n_time_origins": analysis.n_time_origins.ravel(),
                "first_step_exit_fraction": analysis.first_step_exit_fractions.ravel(),
                "terminal_censored_fraction": (
                    analysis.terminal_censored_fractions.ravel()
                ),
            }
        ),
        directory / _ESTIMATES_FILENAME,
        compression="zstd",
    )
    minimum_rows = []
    for selection in minima:
        width_index = int(
            np.flatnonzero(analysis.interval_widths == selection.interval_width)[0]
        )
        minimum_rows.append(
            {
                "rank": selection.nll_rank,
                "width_index": width_index,
                "interval_width": selection.interval_width,
                "mean_excess_nll": selection.mean_excess_nll,
                "direct_nll_fraction": selection.direct_nll_fraction,
                "interpolated_nll_fraction": selection.interpolated_nll_fraction,
                "extrapolated_nll_fraction": selection.extrapolated_nll_fraction,
                "profile_coverage": selection.coverage,
            }
        )
    pq.write_table(
        pa.Table.from_pylist(
            minimum_rows,
            schema=pa.schema(
                [
                    ("rank", pa.int32()),
                    ("width_index", pa.int32()),
                    ("interval_width", pa.float64()),
                    ("mean_excess_nll", pa.float64()),
                    ("direct_nll_fraction", pa.float64()),
                    ("interpolated_nll_fraction", pa.float64()),
                    ("extrapolated_nll_fraction", pa.float64()),
                    ("profile_coverage", pa.float64()),
                ]
            ),
        ),
        directory / _MINIMA_FILENAME,
        compression="zstd",
    )
    _write_histogram_table(
        pa, pq, directory / _OBSERVED_FILENAME, analysis.histograms, False
    )
    _write_histogram_table(
        pa, pq, directory / _CENSORED_FILENAME, analysis.histograms, True
    )
    payload = {
        **({} if metadata is None else metadata),
        "format_version": DIFFUSIVITY_ANALYSIS_CACHE_FORMAT_VERSION,
        "sampling_period": analysis.sampling_period,
        "symmetry_center": analysis.symmetry_center,
        "constant_nll_minima_count": len(minima),
    }
    (directory / _METADATA_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_diffusivity_analysis_cache(
    directory: str | Path,
) -> DiffusivityAnalysisCache:
    """Load a cache written in the current analysis format."""
    _, pq = _pyarrow()
    directory = Path(directory)
    metadata = json.loads((directory / _METADATA_FILENAME).read_text(encoding="utf-8"))
    version = metadata.get("format_version")
    if version != DIFFUSIVITY_ANALYSIS_CACHE_FORMAT_VERSION:
        raise ValueError("unsupported diffusivity-analysis cache format version")
    estimates = pq.read_table(directory / _ESTIMATES_FILENAME).to_pydict()
    width_indices = np.asarray(estimates["width_index"], dtype=np.int64)
    center_indices = np.asarray(estimates["center_index"], dtype=np.int64)
    n_widths = int(width_indices.max()) + 1
    n_centers = int(center_indices.max()) + 1
    shape = (n_widths, n_centers)

    def reshape(name: str, dtype: type[np.generic]) -> NDArray[Any]:
        values = np.asarray(estimates[name], dtype=dtype)
        output = np.empty(shape, dtype=dtype)
        output[width_indices, center_indices] = values
        return output

    widths = reshape("interval_width", np.float64)[:, 0]
    centers = reshape("position", np.float64)[0, :]
    histograms = FirstExitHistogramGrid(
        *_read_histogram_table(pq, directory / _OBSERVED_FILENAME),
        *_read_histogram_table(pq, directory / _CENSORED_FILENAME),
    )
    analysis = DiffusivityAnalysis(
        positions=centers,
        interval_widths=widths,
        diffusivities=reshape("diffusivity", np.float64),
        diffusivity_sems=reshape("diffusivity_sem", np.float64),
        excess_nll=reshape("excess_nll", np.float64),
        n_time_origins=reshape("n_time_origins", np.int64),
        first_step_exit_fractions=reshape("first_step_exit_fraction", np.float64),
        terminal_censored_fractions=reshape("terminal_censored_fraction", np.float64),
        histograms=histograms,
        sampling_period=metadata["sampling_period"],
        symmetry_center=metadata["symmetry_center"],
    )
    stored = pq.read_table(directory / _MINIMA_FILENAME).to_pylist()
    minima = tuple(_minimum_from_cache_row(analysis, row) for row in stored)
    return DiffusivityAnalysisCache(
        analysis=analysis, constant_width_minima=minima, metadata=metadata
    )


def _minimum_from_cache_row(
    analysis: DiffusivityAnalysis, row: dict[str, Any]
) -> DiffusivityProfile:
    """Reconstruct one ranked minimum from its cached analysis row."""
    rank = int(row["rank"])
    width_index = int(row["width_index"])
    if width_index < 0 or width_index >= analysis.interval_widths.size:
        raise ValueError("cached constant-width minimum has an invalid width index")
    if not np.isclose(
        analysis.interval_widths[width_index], float(row["interval_width"])
    ):
        raise ValueError("cached constant-width minimum does not match the analysis")
    selected = analysis.profile_at_width(float(row["interval_width"]))
    return DiffusivityProfile(
        positions=selected.positions,
        diffusivity=selected.diffusivity,
        interval_width=selected.interval_width,
        diffusivity_sem=selected.diffusivity_sem,
        method=f"NLL_{rank}",
        nll_rank=rank,
        mean_excess_nll=float(row["mean_excess_nll"]),
        direct_nll_fraction=float(row["direct_nll_fraction"]),
        interpolated_nll_fraction=float(row["interpolated_nll_fraction"]),
        extrapolated_nll_fraction=float(row["extrapolated_nll_fraction"]),
        coverage=float(row["profile_coverage"]),
    )


def _write_histogram_table(
    pa: Any, pq: Any, path: Path, histograms: FirstExitHistogramGrid, censored: bool
) -> None:
    prefix = "censored" if censored else "observed"
    pq.write_table(
        pa.table(
            {
                "width_index": getattr(histograms, f"{prefix}_width_indices"),
                "center_index": getattr(histograms, f"{prefix}_center_indices"),
                "duration_steps": getattr(histograms, f"{prefix}_duration_steps"),
                "count": getattr(histograms, f"{prefix}_counts"),
            }
        ),
        path,
        compression="zstd",
    )


def _read_histogram_table(pq: Any, path: Path) -> tuple[NDArray[np.int64], ...]:
    values = pq.read_table(path).to_pydict()
    return tuple(
        np.asarray(values[column], dtype=np.int64)
        for column in ("width_index", "center_index", "duration_steps", "count")
    )


def write_propagator_cache(
    directory: Path, comparison: PropagatorComparison, *, metadata: Mapping[str, Any]
) -> None:
    """Store predicted and reference densities with numerical-input fingerprints."""
    pa, pq = _pyarrow()
    directory.mkdir(parents=True, exist_ok=True)
    rows = {
        "lag_ps": [],
        "model": [],
        "grid_index": [],
        "position_angstrom": [],
        "density_per_angstrom": [],
    }
    grid_index = np.arange(comparison.inputs.positions.size, dtype=np.int32)
    for lag in comparison.inputs.lag_times:
        densities = {"MD": comparison.inputs.md_propagators[lag]} | {
            name: values[lag] for name, values in comparison.model_propagators.items()
        }
        for name, density in densities.items():
            rows["lag_ps"].append(
                np.full(comparison.inputs.positions.size, lag, dtype=np.int32)
            )
            rows["model"].append(
                np.full(comparison.inputs.positions.size, name, dtype=object)
            )
            rows["grid_index"].append(grid_index)
            rows["position_angstrom"].append(comparison.inputs.positions)
            rows["density_per_angstrom"].append(density)
    pq.write_table(
        pa.table({key: np.concatenate(value) for key, value in rows.items()}),
        directory / "propagators.parquet",
        compression="zstd",
    )
    pq.write_table(
        pa.Table.from_pylist(list(comparison.metric_rows)),
        directory / "propagator_metrics.parquet",
        compression="zstd",
    )
    (directory / "propagator_metadata.json").write_text(
        json.dumps(
            {"format_version": PROPAGATOR_CACHE_FORMAT_VERSION, **metadata},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def read_propagator_cache(
    directory: Path,
    inputs: PropagatorInputs,
) -> PropagatorComparison:
    """Restore predictions alongside the notebook's current prepared inputs."""
    _, pq = _pyarrow()
    metadata = json.loads(
        (directory / "propagator_metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("format_version") != PROPAGATOR_CACHE_FORMAT_VERSION:
        raise ValueError("unsupported propagator cache format version")
    table = pq.read_table(directory / "propagators.parquet").to_pydict()
    lags = tuple(sorted(set(table["lag_ps"])))
    models = tuple(dict.fromkeys(table["model"]))
    if lags != inputs.lag_times or set(models) != {"MD", *inputs.profiles}:
        raise ValueError("cached models or lag times do not match the inputs")
    indices = np.asarray(table["grid_index"], dtype=np.int64)
    positions = np.asarray(table["position_angstrom"], dtype=float)
    grid = np.empty(int(indices.max()) + 1, dtype=float)
    grid[indices] = positions
    values, lag_values, model_values = (
        np.asarray(table["density_per_angstrom"], dtype=float),
        np.asarray(table["lag_ps"], dtype=int),
        np.asarray(table["model"], dtype=str),
    )
    densities: dict[str, dict[int, NDArray[np.float64]]] = {name: {} for name in models}
    for lag in lags:
        for model in models:
            mask = (lag_values == lag) & (model_values == model)
            if not np.array_equal(np.sort(indices[mask]), np.arange(grid.size)):
                raise ValueError("cached density has missing or duplicate positions")
            densities[model][lag] = values[mask][np.argsort(indices[mask])]
    if not np.array_equal(grid, inputs.positions):
        raise ValueError("cached propagation positions do not match the inputs")
    metrics = tuple(pq.read_table(directory / "propagator_metrics.parquet").to_pylist())
    return PropagatorComparison(
        inputs, {name: densities[name] for name in inputs.profiles}, metrics
    )


def _pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError(
            "Parquet cache support requires pyarrow; install the development "
            "environment "
            "with `pixi install`."
        ) from error
    return pa, pq
