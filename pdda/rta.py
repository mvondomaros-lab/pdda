"""Residence-time estimation for one-dimensional trajectories."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize_scalar

from .profiles import Plateau, SmoothingSplineFitter, _find_right_plateau
from .statistics import StandardErrorEstimator, _blocking_exit_time_standard_error

_TERMINAL_CENSORING_WARNING_FRACTION: Final = 0.05

__all__ = [
    "DiffusivityAnalysis",
    "DiffusivityEstimate",
    "DiffusivityProfile",
    "ProfileSelection",
    "RTATerminalCensoringWarning",
    "SimulationWindow",
    "analyze_diffusivity",
    "estimate_diffusivity",
]


class RTATerminalCensoringWarning(RuntimeWarning):
    """A trajectory ends inside an interval often enough to affect an RTA estimate."""


@dataclass(frozen=True)
class WallDomain:
    """Coordinate range sampled safely between two simulation walls.

    RTA intervals must lie completely inside ``[lower, upper]``. This avoids
    treating wall encounters as physical first exits.
    """

    lower: float
    upper: float

    def __post_init__(self) -> None:
        lower, upper = float(self.lower), float(self.upper)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError("wall bounds must be finite and satisfy lower < upper")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def contains_interval(self, center: float, width: float) -> bool:
        """Whether a centered interval lies wholly between the walls."""
        return bool(
            center - width / 2.0 >= self.lower and center + width / 2.0 <= self.upper
        )


@dataclass(frozen=True)
class ReflectionSymmetry:
    """Pool equivalent observations on the two sides of a symmetry plane.

    With ``origin=0.0``, observations at ``z`` and ``-z`` contribute to the
    same diffusivity estimate.
    """

    origin: float = 0.0

    def __post_init__(self) -> None:
        origin = float(self.origin)
        if not np.isfinite(origin):
            raise ValueError("symmetry origin must be finite")
        object.__setattr__(self, "origin", origin)


@dataclass(frozen=True)
class RTAWindow:
    """One or more continuous trajectories obtained within the same walls.

    Separate windows are never joined end to end. Use
    :meth:`RTAAnalyzer.analyze_overlapping_windows` to pool the
    observations from windows that each support a requested interval.
    """

    paths: object
    domain: WallDomain
    label: str | None = None

    def __post_init__(self) -> None:
        sources = _normalize_trajectory_sources(self.paths)
        if not sources:
            raise ValueError("an RTA window needs at least one trajectory")
        if not isinstance(self.domain, WallDomain):
            raise TypeError("domain must be a WallDomain")
        if self.label is not None and not self.label:
            raise ValueError("label must be non-empty when provided")
        object.__setattr__(self, "paths", sources)

    @property
    def trajectories(self) -> tuple[object, ...]:
        """Trajectory sources belonging to this internal simulation window."""
        return self.paths  # type: ignore[return-value]


@dataclass(frozen=True)
class SimulationWindow:
    """Independent trajectories sampled between the same restraining walls.

    ``trajectories`` may be one array or a sequence of independent arrays.
    ``bounds`` are the lower and upper wall positions. An interval is used only
    when both its boundaries lie inside this range.
    """

    trajectories: object
    bounds: tuple[float, float]
    label: str | None = None

    def __post_init__(self) -> None:
        trajectories = _normalize_trajectory_arrays(self.trajectories)
        if not trajectories:
            raise ValueError("a simulation window needs at least one trajectory")
        domain = WallDomain(*self.bounds)
        object.__setattr__(self, "trajectories", trajectories)
        object.__setattr__(self, "bounds", (domain.lower, domain.upper))
        if self.label is not None and not self.label:
            raise ValueError("label must be non-empty when provided")

    @property
    def domain(self) -> WallDomain:
        """Physical region between the lower and upper walls."""
        return WallDomain(*self.bounds)

    def as_rta_window(self) -> RTAWindow:
        """Represent this simulation window for residence-time analysis."""
        return RTAWindow(
            self.trajectories,
            self.domain,
            label=self.label,
        )


@dataclass(frozen=True)
class ResidenceSegment:
    """A maximal contiguous run of trajectory frames inside the interval."""

    start: int
    stop: int
    values: NDArray[np.float64]
    exit_observed: bool


@dataclass(frozen=True)
class DiffusivityEstimate:
    """Diffusivity and first-exit information for one RTA interval.

    ``diffusivity`` is calculated from completed first exits. If a trajectory
    ends while it remains inside the interval, the corresponding time origins
    are reported in ``terminal_censor_times`` and excluded from the estimate.
    Inspect ``terminal_censored_fraction`` when finite trajectory length may
    matter.
    """

    exit_times: NDArray[np.float64]
    terminal_censor_times: NDArray[np.float64]
    survival_times: NDArray[np.float64]
    survival: NDArray[np.float64]
    residence_time: float
    residence_time_sem: float
    diffusivity: float
    diffusivity_sem: float
    n_valid_frames: int
    n_observed_segments: int
    n_terminal_segments: int
    terminal_censored_fraction: float
    sampling_period: float


@dataclass(frozen=True)
class SurvivalComparison:
    """Shape discrepancy between empirical and constant-diffusion survival."""

    times: NDArray[np.float64]
    empirical: NDArray[np.float64]
    predicted: NDArray[np.float64]
    integrated_squared_error: float
    normalized_root_mean_squared_error: float


@dataclass(frozen=True)
class DiscreteExitModelScore:
    """A censoring-aware discrete first-exit pseudo-likelihood score.

    Time origins extracted from a trajectory overlap and are therefore not
    independent. The score is useful for comparing survival-model shape, but
    is not an independent-observation likelihood for uncertainty inference.
    """

    mean_negative_log_pseudolikelihood: float
    saturated_mean_negative_log_pseudolikelihood: float
    mean_excess_negative_log_pseudolikelihood: float
    observed_exit_contribution: float
    censored_contribution: float
    n_observed_exits: int
    n_censored: int


@dataclass(frozen=True)
class FirstExitHistogramGrid:
    """Sparse residence-segment length histograms.

    Every row identifies one ``(width_index, center_index, duration_steps)``
    segment-length bin and records its multiplicity.  Reverse cumulative sums
    convert observed segment lengths into the first-exit time-origin histogram;
    the corresponding terminal lengths give the censoring histogram. This
    avoids materializing every time origin while preserving all data needed to
    recompute survival curves, RTA diffusivities, and discrete NLL scores.
    """

    observed_width_indices: NDArray[np.int64]
    observed_center_indices: NDArray[np.int64]
    observed_duration_steps: NDArray[np.int64]
    observed_counts: NDArray[np.int64]
    censored_width_indices: NDArray[np.int64]
    censored_center_indices: NDArray[np.int64]
    censored_duration_steps: NDArray[np.int64]
    censored_counts: NDArray[np.int64]


@dataclass(frozen=True)
class DiffusiveSurvivalFit:
    """A deterministic maximum-pseudo-likelihood constant-diffusion fit."""

    model: DiffusiveSurvivalModel
    score: DiscreteExitModelScore
    n_function_evaluations: int


@dataclass(frozen=True)
class ProfileSelection:
    """An NLL minimum selected within a supported bulk-width range.

    For each tested width, the outer part of the diffusivity profile is checked
    for a spatial plateau. The accepted bulk diffusivities are then examined
    as a function of width. The highest supported width plateau defines a
    finite range of bulk-compatible widths. ``profile`` is the lowest-NLL
    minimum within that range. Its rank and sampling diagnostics are retained.
    """

    profile: DiffusivityProfile
    minimum_bulk_width: float
    maximum_bulk_width: float
    plateau_tolerance: float
    minimum_plateau_points: int
    origin: float
    limiting_bulk_diffusivity: float
    convergence_width: float
    convergence_relative_trend: float
    convergence_relative_scatter: float
    bulk_diffusivities: NDArray[np.float64]
    spatial_plateau_starts: NDArray[np.float64]
    spatial_plateau_point_counts: NDArray[np.int64]
    spatial_plateau_relative_trends: NDArray[np.float64]
    spatial_plateau_relative_scatters: NDArray[np.float64]
    relative_bulk_deviations: NDArray[np.float64]
    plateau_compatible: NDArray[np.bool_]

    @property
    def interval_width(self) -> float:
        """Selected NLL minimum's interval width, in coordinate units."""
        return self.profile.interval_width

    @property
    def positions(self) -> NDArray[np.float64]:
        return self.profile.positions

    @property
    def diffusivity(self) -> NDArray[np.float64]:
        return self.profile.diffusivity

    def summary(self) -> str:
        """Describe the detected bulk bound and selected NLL minimum."""
        if not np.isfinite(self.minimum_bulk_width):
            return "No bulk-convergence width could be detected."
        if not np.isfinite(self.interval_width):
            return (
                "No NLL minimum lies within the bulk-compatible width range "
                f"{self.minimum_bulk_width:g} to {self.maximum_bulk_width:g}."
            )
        return (
            f"Selected NLL_{self.profile.nll_rank} at interval width "
            f"{self.interval_width:g}; bulk width range "
            f"{self.minimum_bulk_width:g} to {self.maximum_bulk_width:g}; bulk "
            f"diffusivity {self.limiting_bulk_diffusivity:.6g}."
        )


@dataclass(frozen=True)
class DiffusivityAnalysis:
    """RTA diffusivities evaluated at several positions and interval widths.

    Rows correspond to ``interval_widths`` and columns to ``positions``.
    ``diffusivities`` contains the estimates.
    ``excess_nll`` measures disagreement between the constant-diffusion model
    and the observed first-exit-time distribution; lower values indicate
    better agreement.
    ``NaN`` marks an interval that crosses a wall or lacks sufficient exits.
    """

    positions: NDArray[np.float64]
    interval_widths: NDArray[np.float64]
    diffusivities: NDArray[np.float64]
    excess_nll: NDArray[np.float64]
    n_time_origins: NDArray[np.int64]
    first_step_exit_fractions: NDArray[np.float64]
    terminal_censored_fractions: NDArray[np.float64]
    sampling_period: float
    histograms: FirstExitHistogramGrid | None = None
    diffusivity_sems: NDArray[np.float64] | None = None
    symmetry_center: float | None = None

    def profile_at_width(self, width: float) -> DiffusivityProfile:
        """Return the profile for one analyzed interval width."""
        index = self._width_index(width)
        valid = np.isfinite(self.diffusivities[index])
        return _profile_for_width(
            self,
            index,
            np.broadcast_to(valid, self.diffusivities.shape),
            method="specified",
        )

    def at_position(self, position: float) -> dict[str, NDArray[np.float64] | float]:
        """Return all width-dependent quantities at one scanned position."""
        matches = np.flatnonzero(
            np.isclose(self.positions, float(position), rtol=0.0, atol=0.0)
        )
        if matches.size != 1:
            raise ValueError("position must exactly match one analyzed center")
        index = int(matches[0])
        return {
            "position": float(self.positions[index]),
            "interval_width": self.interval_widths.copy(),
            "diffusivity": self.diffusivities[:, index].copy(),
            "diffusivity_sem": (
                np.full(self.interval_widths.size, np.nan)
                if self.diffusivity_sems is None
                else self.diffusivity_sems[:, index].copy()
            ),
            "excess_nll": self.excess_nll[:, index].copy(),
        }

    def summary(self) -> str:
        """Return a compact, notebook-friendly description of the result."""
        finite = int(np.isfinite(self.diffusivities).sum())
        total = int(self.diffusivities.size)
        return (
            f"Diffusivity analysis: {self.positions.size} positions x "
            f"{self.interval_widths.size} interval widths; "
            f"{finite}/{total} estimates available."
        )

    def _width_index(self, width: float) -> int:
        matches = np.flatnonzero(
            np.isclose(self.interval_widths, float(width), rtol=0.0, atol=0.0)
        )
        if matches.size != 1:
            raise ValueError("width must exactly match one analyzed interval width")
        return int(matches[0])

    def constant_width_minima(
        self,
    ) -> tuple[DiffusivityProfile, ...]:
        """Rank constant interval widths by agreement with the exit-time data.

        For each width, excess NLL is averaged equally over position. Internal
        gaps are linearly interpolated and terminal gaps are continued with the
        nearest available value so that walls and window seams do not favor
        narrower intervals merely because they cover more positions. Every
        local minimum among the tested widths is returned. ``NLL_0`` has the
        lowest mean excess NLL; the remaining minima follow in increasing
        order. The diffusivity profile itself is never interpolated.
        """
        directly_supported = np.isfinite(self.excess_nll) & np.isfinite(
            self.diffusivities
        )
        completed, sources = _complete_nll_map(
            self.positions, self.excess_nll, directly_supported
        )
        mean_nll = np.mean(completed, axis=1)
        minimum_indices = _discrete_local_minimum_indices(mean_nll)
        minimum_indices = minimum_indices[
            np.argsort(mean_nll[minimum_indices], kind="stable")
        ]
        direct_fractions = np.mean(sources == 1, axis=1)
        interpolated_fractions = np.mean(sources == 2, axis=1)
        extrapolated_fractions = np.mean(sources == 3, axis=1)
        coverages = np.mean(np.isfinite(self.diffusivities), axis=1)
        return tuple(
            _profile_for_width(
                self,
                int(width_index),
                directly_supported,
                method=f"NLL_{rank}",
                nll_rank=rank,
                mean_excess_nll=float(mean_nll[width_index]),
                direct_nll_fraction=float(direct_fractions[width_index]),
                interpolated_nll_fraction=float(interpolated_fractions[width_index]),
                extrapolated_nll_fraction=float(extrapolated_fractions[width_index]),
                coverage=float(coverages[width_index]),
            )
            for rank, width_index in enumerate(minimum_indices)
        )

    def select_profile(
        self,
        *,
        plateau_tolerance: float = 0.05,
        minimum_plateau_points: int = 2,
        origin: float = 0.0,
    ) -> ProfileSelection:
        """Select the lowest NLL minimum in the bulk-compatible width range.

        Bulk detection has two stages. First, each width is treated separately:
        an outer spatial plateau is identified in its diffusivity profile, and
        its representative value is recorded as that width's bulk diffusivity.
        The plateau is sought only in the outermost contiguous block of
        supported radii, within the outer half of the sampled radial domain.
        It must extend to the outermost finite supported point.
        This produces one bulk-diffusivity estimate for every width with a
        credible spatial plateau.

        Second, find the highest nearly constant run of at least four
        consecutive width estimates with at least three spatial plateau points
        each. This run need not reach the largest tested width: wide intervals
        can mix bulk and membrane behavior. Adjacent widths within tolerance
        of its value extend the bulk-compatible range.

        Finally, select the minimum with the lowest mean excess NLL within
        that range, including its endpoints.
        Use :meth:`constant_width_minima` to inspect or choose any NLL minimum.

        Both stages use the same trend and scatter criterion.
        ``minimum_plateau_points`` applies to spatial plateau detection;
        the width-stage minimum is at least four points.

        The NLL values in the analysis are neither modified nor filtered by this
        procedure. If no credible selection can be made, ``interval_width`` is
        ``NaN`` and the returned profile contains no selected positions.

        Cite von Domaros, Thomas, and Prabhakar (2026),
        https://arxiv.org/abs/2610.29703, when using interval-width selection
        or bulk-convergence diagnostics, in addition to the original RTA paper.
        """
        origin = float(origin)
        if not np.isfinite(origin):
            raise ValueError("origin must be finite")
        plateau_fitter = SmoothingSplineFitter(
            log_space=True,
            plateau=True,
            plateau_tolerance=plateau_tolerance,
            minimum_plateau_points=minimum_plateau_points,
            extrapolate=True,
        )
        plateau_tolerance = plateau_fitter.plateau_tolerance
        minimum_plateau_points = plateau_fitter.minimum_plateau_points

        n_widths = self.interval_widths.size
        sampled_radii = np.unique(np.abs(self.positions - origin))
        bulk_diffusivities = np.full(n_widths, np.nan)
        spatial_starts = np.full(n_widths, np.nan)
        spatial_counts = np.zeros(n_widths, dtype=np.int64)
        spatial_trends = np.full(n_widths, np.nan)
        spatial_scatters = np.full(n_widths, np.nan)
        for index, diffusivity in enumerate(self.diffusivities):
            radii, radial_diffusivity = _fold_positive_profile(
                self.positions, diffusivity, origin
            )
            if radii.size < minimum_plateau_points:
                continue
            plateau = _find_right_plateau(
                radii,
                radial_diffusivity,
                plateau_tolerance,
                log_space=True,
                minimum_points=minimum_plateau_points,
                outer_half=True,
                support_positions=sampled_radii,
            )
            if plateau is None:
                continue
            bulk_diffusivities[index] = plateau.value
            spatial_starts[index] = plateau.start
            spatial_counts[index] = plateau.n_points
            spatial_trends[index] = plateau.relative_trend
            spatial_scatters[index] = plateau.relative_scatter

        limiting_bulk_diffusivity = float("nan")
        convergence_width = float("nan")
        convergence_trend = float("nan")
        convergence_scatter = float("nan")
        relative_deviations = np.full(n_widths, np.nan)
        compatible = np.zeros(n_widths, dtype=bool)
        detected = np.isfinite(bulk_diffusivities)
        order = np.argsort(self.interval_widths)
        sorted_widths = self.interval_widths[order]
        if np.any(np.diff(sorted_widths) <= 0.0):
            raise ValueError(
                "bulk-convergence detection requires unique interval widths"
            )
        supported = detected[order] & (
            spatial_counts[order] >= max(3, minimum_plateau_points)
        )
        width_plateau = _find_maximum_width_plateau(
            sorted_widths,
            bulk_diffusivities[order],
            supported,
            plateau_tolerance,
            max(4, minimum_plateau_points),
        )
        if width_plateau is not None:
            convergence_plateau, first, last = width_plateau
            limiting_bulk_diffusivity = convergence_plateau.value
            convergence_width = convergence_plateau.start
            convergence_trend = convergence_plateau.relative_trend
            convergence_scatter = convergence_plateau.relative_scatter
            relative_deviations[detected] = np.abs(
                bulk_diffusivities[detected] / limiting_bulk_diffusivity - 1.0
            )
            acceptable = supported & (relative_deviations[order] <= plateau_tolerance)
            while first > 0 and acceptable[first - 1]:
                first -= 1
            while last + 1 < n_widths and acceptable[last + 1]:
                last += 1
            compatible[order[first : last + 1]] = acceptable[first : last + 1]

        minimum_bulk_width = float("nan")
        maximum_bulk_width = float("nan")
        profile = _profile_for_width(
            self, None, np.isfinite(self.diffusivities), method="bulk-constrained NLL"
        )
        if np.any(compatible):
            minimum_bulk_width = float(np.min(self.interval_widths[compatible]))
            maximum_bulk_width = float(np.max(self.interval_widths[compatible]))
            candidates = [
                candidate
                for candidate in self.constant_width_minima()
                if candidate.interval_width in self.interval_widths[compatible]
            ]
            if candidates:
                profile = candidates[0]  # Minima are ranked by mean excess NLL.
        result = ProfileSelection(
            profile=profile,
            minimum_bulk_width=minimum_bulk_width,
            maximum_bulk_width=maximum_bulk_width,
            plateau_tolerance=plateau_tolerance,
            minimum_plateau_points=minimum_plateau_points,
            origin=origin,
            limiting_bulk_diffusivity=limiting_bulk_diffusivity,
            convergence_width=convergence_width,
            convergence_relative_trend=convergence_trend,
            convergence_relative_scatter=convergence_scatter,
            bulk_diffusivities=bulk_diffusivities,
            spatial_plateau_starts=spatial_starts,
            spatial_plateau_point_counts=spatial_counts,
            spatial_plateau_relative_trends=spatial_trends,
            spatial_plateau_relative_scatters=spatial_scatters,
            relative_bulk_deviations=relative_deviations,
            plateau_compatible=compatible,
        )
        return result


def _find_maximum_width_plateau(
    widths: NDArray[np.float64],
    values: NDArray[np.float64],
    supported: NDArray[np.bool_],
    tolerance: float,
    minimum_points: int,
) -> tuple[Plateau, int, int] | None:
    """Find the highest flat segment of consecutive, spatially supported widths."""
    indices = np.flatnonzero(supported)
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    best: tuple[Plateau, int, int] | None = None
    for run in runs:
        for first in range(run.size - minimum_points + 1):
            for stop in range(first + minimum_points, run.size + 1):
                segment = run[first:stop]
                plateau = _find_right_plateau(
                    widths[segment],
                    values[segment],
                    tolerance,
                    log_space=True,
                    minimum_points=segment.size,
                    outer_half=False,
                )
                if plateau is None:
                    continue
                if np.any(np.abs(values[segment] / plateau.value - 1.0) > tolerance):
                    continue
                candidate = (plateau, int(segment[0]), int(segment[-1]))
                if best is None or (plateau.value, plateau.n_points) > (
                    best[0].value,
                    best[0].n_points,
                ):
                    best = candidate
    return best


def _fold_positive_profile(
    centers: NDArray[np.float64],
    values: NDArray[np.float64],
    origin: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fold finite positive profile values about an origin using geometric means."""
    available = np.isfinite(values) & (values > 0.0)
    radii, inverse = np.unique(np.abs(centers[available] - origin), return_inverse=True)
    folded = np.empty(radii.size, dtype=np.float64)
    logs = np.log(values[available])
    for index in range(radii.size):
        folded[index] = np.exp(logs[inverse == index].mean())
    return radii, folded


@dataclass(frozen=True)
class DiffusivityProfile:
    """An RTA diffusivity profile obtained with one selected interval width.

    The diffusivity arrays are aligned with ``positions`` and contain ``NaN``
    where the selected interval is not supported. Profiles returned by
    :meth:`DiffusivityAnalysis.constant_width_minima` also report their NLL
    rank, mean excess NLL, sampling-source fractions, and spatial coverage.
    """

    positions: NDArray[np.float64]
    diffusivity: NDArray[np.float64]
    interval_width: float
    diffusivity_sem: NDArray[np.float64] | None = None
    method: str = "unspecified"
    nll_rank: int | None = None
    mean_excess_nll: float | None = None
    direct_nll_fraction: float | None = None
    interpolated_nll_fraction: float | None = None
    extrapolated_nll_fraction: float | None = None
    coverage: float | None = None

    def summary(self) -> str:
        """Return a compact description without printing automatically."""
        if not np.isfinite(self.interval_width):
            return f"No interval width could be selected using {self.method}."
        available = int(np.isfinite(self.diffusivity).sum())
        return (
            f"Selected interval width {self.interval_width:g} using {self.method}; "
            "profile available at "
            f"{available}/{self.positions.size} positions."
        )


class DiffusiveSurvivalModel:
    """Analytical first-exit survival model for uniform diffusion in an interval.

    The initial position is uniformly distributed over ``[xmin, xmax)``. This
    is the discrete RTA model's relevant initial distribution in the flat-PMF
    limit. The survival series contains odd eigenmodes of the absorbing
    diffusion equation.
    """

    def __init__(
        self, xmin: float, xmax: float, diffusivity: float, *, n_terms: int = 256
    ) -> None:
        self.xmin, self.xmax, _ = _validate_configuration(xmin, xmax, 1.0)
        self.diffusivity = float(diffusivity)
        if not np.isfinite(self.diffusivity) or self.diffusivity <= 0.0:
            raise ValueError("diffusivity must be finite and greater than zero")
        if not isinstance(n_terms, int) or n_terms < 1:
            raise ValueError("n_terms must be a positive integer")
        self.n_terms = n_terms

    @property
    def residence_time(self) -> float:
        """Exact interval-averaged mean first-exit time."""
        return (self.xmax - self.xmin) ** 2 / (12.0 * self.diffusivity)

    def survival(self, times: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the interval-averaged survival probability at given times."""
        values = _validate_nonnegative_times(times, "times")
        scaled_times = self.diffusivity * values / (self.xmax - self.xmin) ** 2
        result = np.empty_like(scaled_times)
        result[scaled_times == 0.0] = 1.0

        # The heat-content expansion is exact up to exponentially small image
        # terms in this regime and avoids hundreds of eigenmodes near t = 0.
        short = (scaled_times > 0.0) & (scaled_times <= 0.01)
        result[short] = 1.0 - 4.0 * np.sqrt(scaled_times[short] / np.pi)

        eigen = scaled_times > 0.01
        if np.any(eigen):
            minimum = float(scaled_times[eigen].min())
            maximum_odd_mode = np.sqrt(
                -np.log(np.finfo(np.float64).eps) / (np.pi**2 * minimum)
            )
            required_terms = max(1, int(np.ceil((maximum_odd_mode + 1.0) / 2.0)))
            odd_modes = (
                2.0 * np.arange(min(self.n_terms, required_terms), dtype=np.float64)
                + 1.0
            )
            result[eigen] = (8.0 / np.pi**2) * np.sum(
                np.exp(
                    -(np.pi**2) * np.multiply.outer(scaled_times[eigen], odd_modes**2)
                )
                / odd_modes**2,
                axis=1,
            )
        return result

    def compare(self, times: ArrayLike, survival: ArrayLike) -> SurvivalComparison:
        """Compare empirical survival using an integrated squared-error score.

        KL divergence is not appropriate here: survival ordinates are neither
        a probability mass function nor a density.
        """
        eval_times = _validate_nonnegative_times(times, "times")
        empirical = np.asarray(survival, dtype=np.float64)
        if empirical.ndim != 1 or empirical.shape != eval_times.shape:
            raise ValueError("survival must be a one-dimensional array matching times")
        if not np.all(np.isfinite(empirical)) or np.any(
            (empirical < 0.0) | (empirical > 1.0)
        ):
            raise ValueError("survival must contain finite probabilities in [0, 1]")
        if np.any(np.diff(eval_times) < 0.0):
            raise ValueError("times must be non-decreasing for survival comparison")
        predicted = self.survival(eval_times)
        integrated_squared_error = float(
            np.trapezoid((empirical - predicted) ** 2, eval_times)
        )
        scale = float(np.trapezoid(empirical**2, eval_times))
        normalized_rmse = (
            float(np.sqrt(integrated_squared_error / scale))
            if scale > 0.0
            else float("nan")
        )
        return SurvivalComparison(
            times=eval_times,
            empirical=empirical.copy(),
            predicted=predicted,
            integrated_squared_error=integrated_squared_error,
            normalized_root_mean_squared_error=normalized_rmse,
        )

    def score_discrete_exit_times(
        self,
        exit_times: ArrayLike,
        censor_times: ArrayLike,
        sampling_period: float,
    ) -> DiscreteExitModelScore:
        """Score sampled exit/censor observations without an arbitrary tail cut.

        An exit reported at ``t`` is assigned probability ``S(t - dt) - S(t)``;
        a right-censored observation at ``t`` is assigned probability ``S(t)``.
        The returned mean negative log pseudo-likelihood is lower for a better
        discrete-time constant-diffusion fit. Its excess over a saturated
        discrete survival model is comparable between interval widths.
        """
        observed = _validate_nonnegative_times(exit_times, "exit_times")
        censored = _validate_nonnegative_times(censor_times, "censor_times")
        if np.any(observed <= 0.0):
            raise ValueError("exit_times must be strictly positive")
        sampling_period = _validate_sampling_period(sampling_period)
        if np.any(observed < sampling_period):
            raise ValueError("exit_times must be at least one sampling period")
        # Times live on a discrete grid and repeat heavily for overlapping RTA
        # origins. Histograms keep the calculation practical for long traces.
        unique_exits, exit_counts = np.unique(observed, return_counts=True)
        unique_censors, censor_counts = np.unique(censored, return_counts=True)
        observed_contribution, censored_contribution = (
            _negative_log_likelihood_contributions(
                self,
                unique_exits,
                exit_counts,
                unique_censors,
                censor_counts,
                sampling_period,
            )
        )
        count = observed.size + censored.size
        mean_nll = (
            (observed_contribution + censored_contribution) / count
            if count
            else float("nan")
        )
        saturated_mean_nll = _saturated_discrete_mean_nll(observed, censored)
        return DiscreteExitModelScore(
            mean_negative_log_pseudolikelihood=mean_nll,
            saturated_mean_negative_log_pseudolikelihood=saturated_mean_nll,
            mean_excess_negative_log_pseudolikelihood=mean_nll - saturated_mean_nll,
            observed_exit_contribution=observed_contribution,
            censored_contribution=censored_contribution,
            n_observed_exits=observed.size,
            n_censored=censored.size,
        )

    @classmethod
    def fit_discrete_exit_times(
        cls,
        xmin: float,
        xmax: float,
        exit_times: ArrayLike,
        censor_times: ArrayLike,
        sampling_period: float,
        *,
        n_terms: int = 256,
    ) -> DiffusiveSurvivalFit:
        """Fit diffusivity by bounded scalar maximum pseudo-likelihood.

        The optimization is deterministic in log diffusivity and incorporates
        terminal right-censoring. It does not perform Bayesian sampling.
        """
        observed = _validate_nonnegative_times(exit_times, "exit_times")
        censored = _validate_nonnegative_times(censor_times, "censor_times")
        if observed.size == 0:
            raise ValueError(
                "at least one observed exit is required to fit diffusivity"
            )
        xmin, xmax, _ = _validate_configuration(xmin, xmax, 1.0)
        sampling_period = _validate_sampling_period(sampling_period)
        initial_diffusivity = (xmax - xmin) ** 2 / (12.0 * observed.mean())
        unique_exits, exit_counts = np.unique(observed, return_counts=True)
        unique_censors, censor_counts = np.unique(censored, return_counts=True)

        def objective(log_diffusivity: float) -> float:
            model = cls(xmin, xmax, float(np.exp(log_diffusivity)), n_terms=n_terms)
            contributions = _negative_log_likelihood_contributions(
                model,
                unique_exits,
                exit_counts,
                unique_censors,
                censor_counts,
                sampling_period,
            )
            return sum(contributions) / (observed.size + censored.size)

        optimization = minimize_scalar(
            objective,
            bounds=(
                np.log(initial_diffusivity) - 10.0,
                np.log(initial_diffusivity) + 10.0,
            ),
            method="bounded",
            options={"xatol": 1e-8},
        )
        if not optimization.success:
            raise RuntimeError("diffusive survival fit did not converge")
        model = cls(xmin, xmax, float(np.exp(optimization.x)), n_terms=n_terms)
        return DiffusiveSurvivalFit(
            model=model,
            score=model.score_discrete_exit_times(observed, censored, sampling_period),
            n_function_evaluations=optimization.nfev,
        )


def _negative_log_likelihood_contributions(
    model: DiffusiveSurvivalModel,
    unique_exits: NDArray[np.float64],
    exit_counts: NDArray[np.int64],
    unique_censors: NDArray[np.float64],
    censor_counts: NDArray[np.int64],
    sampling_period: float,
) -> tuple[float, float]:
    exit_probability = model.survival(unique_exits - sampling_period) - model.survival(
        unique_exits
    )
    censor_probability = model.survival(unique_censors)
    tiny = np.finfo(np.float64).tiny
    observed = float(-np.dot(exit_counts, np.log(np.maximum(exit_probability, tiny))))
    censored = float(
        -np.dot(censor_counts, np.log(np.maximum(censor_probability, tiny)))
    )
    return observed, censored


class ResidenceTimeEstimator:
    """Estimate bulk diffusivity from first exits in a fixed interval.

    The interval is ``[xmin, xmax)`` and ``sampling_period`` is the elapsed
    time between saved samples. Incomplete final residence segments are
    discarded from the RTA estimate. Their time-origin fraction is returned as
    a diagnostic.
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        sampling_period: float,
    ) -> None:
        self.xmin, self.xmax, self.sampling_period = _validate_configuration(
            xmin, xmax, sampling_period
        )

    def segments(self, trajectory: ArrayLike) -> tuple[ResidenceSegment, ...]:
        """Return in-interval segments with frame indices and exit status."""
        x = _validate_trajectory(trajectory)
        inside = (self.xmin <= x) & (x < self.xmax)
        starts = np.flatnonzero(inside & np.concatenate(([True], ~inside[:-1])))
        stops = np.flatnonzero(inside & np.concatenate((~inside[1:], [True]))) + 1
        return tuple(
            ResidenceSegment(
                int(start),
                int(stop),
                x[start:stop].copy(),
                bool(stop < x.size),
            )
            for start, stop in zip(starts, stops, strict=True)
        )

    def analyze(
        self, trajectory: ArrayLike, *, survival_times: ArrayLike | None = None
    ) -> DiffusivityEstimate:
        """Analyze one trajectory and return estimates and censoring diagnostics."""
        return self.analyze_many((trajectory,), survival_times=survival_times)

    def analyze_many(
        self,
        trajectories: Iterable[ArrayLike],
        *,
        survival_times: ArrayLike | None = None,
    ) -> DiffusivityEstimate:
        """Pool independent trajectories without treating joins as continuous.

        Each input trajectory retains its own terminal-censoring status. This
        is appropriate for independent replicas or separately stored chunks.
        """
        segments = tuple(
            segment
            for trajectory in trajectories
            for segment in self.segments(trajectory)
        )
        observed, censored = _observations_from_segments(segments, self.sampling_period)
        n_observed_segments = sum(segment.exit_observed for segment in segments)
        n_terminal_segments = len(segments) - n_observed_segments
        n_valid = observed.size + censored.size
        fraction = 0.0 if n_valid == 0 else censored.size / n_valid
        eval_times = _validate_survival_times(
            survival_times, observed, np.empty(0), self.sampling_period
        )
        survival = _empirical_survival(observed, eval_times)
        residence, residence_sem, diffusion, diffusion_sem = _estimates(
            observed, self.xmax - self.xmin
        )

        result = DiffusivityEstimate(
            exit_times=observed,
            terminal_censor_times=censored,
            survival_times=eval_times,
            survival=survival,
            residence_time=residence,
            residence_time_sem=residence_sem,
            diffusivity=diffusion,
            diffusivity_sem=diffusion_sem,
            n_valid_frames=n_valid,
            n_observed_segments=n_observed_segments,
            n_terminal_segments=n_terminal_segments,
            terminal_censored_fraction=float(fraction),
            sampling_period=self.sampling_period,
        )
        _warn_high_terminal_censoring_for_result(result, self.xmin, self.xmax)
        return result


class _RTAScanEngine:
    """Analyze many RTA intervals while streaming trajectory files once.

    Centers and widths may be any finite floating-point values. The analyzer
    forms an adaptive set of position states from the requested interval
    boundaries, run-length-encodes those states once per file, and derives all
    interval residence segments from that shared representation.  It avoids
    loading the full trajectory collection or repeatedly parsing each file.
    """

    def __init__(
        self, centers: ArrayLike, widths: ArrayLike, sampling_period: float
    ) -> None:
        centers = np.asarray(centers, dtype=np.float64)
        widths = np.asarray(widths, dtype=np.float64)
        if centers.ndim != 1 or centers.size == 0 or not np.all(np.isfinite(centers)):
            raise ValueError("centers must be a non-empty one-dimensional finite array")
        if (
            widths.ndim != 1
            or widths.size == 0
            or not np.all(np.isfinite(widths))
            or np.any(widths <= 0.0)
        ):
            raise ValueError(
                "widths must be a non-empty one-dimensional array of positive "
                "finite values"
            )
        _, _, self.sampling_period = _validate_configuration(0.0, 1.0, sampling_period)
        self.centers = centers.copy()
        self.widths = widths.copy()
        lower = centers[None, :] - widths[:, None] / 2.0
        upper = centers[None, :] + widths[:, None] / 2.0
        self._boundaries = np.unique(np.concatenate((lower.ravel(), upper.ravel())))
        self._lower_state = np.searchsorted(self._boundaries, lower).astype(np.int64)
        self._upper_state = np.searchsorted(self._boundaries, upper).astype(np.int64)

    def analyze_paths(
        self,
        paths: Iterable[object],
        *,
        reflection_origin: float | None = None,
        reflection_paths: Iterable[object] | None = None,
        candidate_masks: Iterable[ArrayLike] | None = None,
        reflected_candidate_masks: Iterable[ArrayLike] | None = None,
        include_histograms: bool = False,
    ) -> DiffusivityAnalysis:
        """Return full-data grid diagnostics without joining file boundaries."""
        path_values = tuple(paths)
        if not path_values:
            raise ValueError("at least one trajectory path is required")
        shape = (self.widths.size, self.centers.size)
        observed = [[{} for _ in self.centers] for _ in self.widths]
        censored = [[{} for _ in self.centers] for _ in self.widths]
        reflection_candidates = self._reflection_candidates(reflection_origin)
        all_candidates = np.ones(shape, dtype=bool)
        masks = _candidate_masks(path_values, candidate_masks, all_candidates)
        reflected_masks = _candidate_masks(
            path_values,
            reflected_candidate_masks,
            reflection_candidates,
        )

        def accumulate(
            trajectory: NDArray[np.float64], candidates: NDArray[np.bool_]
        ) -> None:
            states, durations = _boundary_run_lengths(trajectory, self._boundaries)
            state_min, state_max = int(states.min()), int(states.max())
            for width_index, center_index in zip(
                *np.nonzero(
                    candidates
                    & (self._lower_state <= state_max)
                    & (self._upper_state > state_min)
                ),
                strict=True,
            ):
                lengths, terminal = _interval_run_lengths(
                    states,
                    durations,
                    self._lower_state[width_index, center_index],
                    self._upper_state[width_index, center_index],
                )
                if lengths.size == 0:
                    continue
                if terminal and lengths.size > 1:
                    _accumulate_length_counts(
                        observed[width_index][center_index], lengths[:-1]
                    )
                elif not terminal:
                    _accumulate_length_counts(
                        observed[width_index][center_index], lengths
                    )
                if terminal:
                    _accumulate_length_counts(
                        censored[width_index][center_index], lengths[-1:]
                    )

        for path, candidates, reflected_candidates in zip(
            path_values, masks, reflected_masks, strict=True
        ):
            trajectory = _load_trajectory_source(path)
            accumulate(trajectory, candidates)
            if reflection_origin is not None and reflection_paths is None:
                reflected = 2.0 * reflection_origin - trajectory
                accumulate(reflected, reflected_candidates)
        if reflection_paths is not None:
            if reflection_origin is None:
                raise ValueError("reflection_paths requires reflection symmetry")
            reflected_path_values = tuple(reflection_paths)
            if not reflected_path_values:
                raise ValueError("at least one reflected trajectory path is required")
            reflected_masks = _candidate_masks(
                reflected_path_values,
                reflected_candidate_masks,
                reflection_candidates,
            )
            for path, candidates in zip(
                reflected_path_values, reflected_masks, strict=True
            ):
                trajectory = _load_trajectory_source(path)
                reflected = 2.0 * reflection_origin - trajectory
                accumulate(reflected, candidates)

        diffusivities = np.full(shape, np.nan)
        excess_nll = np.full(shape, np.nan)
        n_origins = np.zeros(shape, dtype=np.int64)
        first_step_fraction = np.full(shape, np.nan)
        censor_fraction = np.full(shape, np.nan)
        for width_index, width in enumerate(self.widths):
            for center_index, center in enumerate(self.centers):
                event_counts, censor_counts = _time_origin_histograms(
                    observed[width_index][center_index],
                    censored[width_index][center_index],
                )
                total = int(event_counts.sum() + censor_counts.sum())
                n_origins[width_index, center_index] = total
                observed_total = int(event_counts.sum())
                if observed_total == 0:
                    continue
                censor_fraction[width_index, center_index] = censor_counts.sum() / total
                first_step_fraction[width_index, center_index] = (
                    event_counts[1] / observed_total
                )
                residence = (
                    self.sampling_period
                    * np.dot(np.arange(event_counts.size), event_counts)
                    / observed_total
                )
                if residence <= 0.0:
                    continue
                diffusivity = width**2 / (12.0 * residence)
                diffusivities[width_index, center_index] = diffusivity
                model = DiffusiveSurvivalModel(
                    center - width / 2.0, center + width / 2.0, diffusivity
                )
                predicted = model.survival(
                    np.arange(event_counts.size, dtype=np.float64)
                    * self.sampling_period
                )
                excess_nll[width_index, center_index] = _histogram_excess_nll(
                    event_counts, np.zeros_like(censor_counts), predicted
                )
        histograms = (
            _sparse_histogram_grid(observed, censored) if include_histograms else None
        )
        return DiffusivityAnalysis(
            positions=self.centers.copy(),
            interval_widths=self.widths.copy(),
            diffusivities=diffusivities,
            excess_nll=excess_nll,
            n_time_origins=n_origins,
            first_step_exit_fractions=first_step_fraction,
            terminal_censored_fractions=censor_fraction,
            sampling_period=self.sampling_period,
            histograms=histograms,
        )

    def _reflection_candidates(
        self, reflection_origin: float | None
    ) -> NDArray[np.bool_]:
        """Return off-center candidates to pool with their reflection."""
        if reflection_origin is None:
            return np.zeros((self.widths.size, self.centers.size), dtype=bool)
        centered = np.isclose(self.centers, reflection_origin)
        return np.broadcast_to(
            ~centered[None, :], (self.widths.size, self.centers.size)
        ).copy()

    def blocking_diffusivity_sems(
        self,
        paths: Iterable[object],
        selected_widths: ArrayLike,
        *,
        reflection_origin: float | None = None,
        reflection_paths: Iterable[object] | None = None,
    ) -> NDArray[np.float64]:
        """Return blocking SEMs for one selected width at each center.

        Blocking accounts for correlation between overlapping time origins and
        therefore requires one continuous trajectory file.  Pass ``NaN`` for
        centers without a selected interval; their returned SEM is ``NaN``.
        """
        path_values = tuple(paths)
        if len(path_values) != 1:
            raise ValueError(
                "blocking SEMs require exactly one continuous trajectory path"
            )
        reflected_path_values = (
            None if reflection_paths is None else tuple(reflection_paths)
        )
        if reflected_path_values is not None and len(reflected_path_values) != 1:
            raise ValueError(
                "blocking SEMs require exactly one reflected trajectory path"
            )
        if reflected_path_values is not None and reflection_origin is None:
            raise ValueError("reflection_paths requires reflection symmetry")
        selected = np.asarray(selected_widths, dtype=np.float64)
        if selected.shape != self.centers.shape:
            raise ValueError("selected_widths must have one value per center")
        if np.any(~np.isfinite(selected) & ~np.isnan(selected)):
            raise ValueError("selected widths must be finite or NaN")
        active = np.isfinite(selected)
        if np.any(selected[active] <= 0.0):
            raise ValueError("selected widths must be positive or NaN")
        if not np.all(np.isin(selected[active], self.widths)):
            raise ValueError("selected widths must be chosen from the analyzed widths")
        candidates = self.widths[:, None] == selected[None, :]
        _, grid_sems, _ = self.blocking_diffusivity_summary_grid(
            path_values[0],
            candidates,
            reflection_origin=reflection_origin,
            reflection_path=(
                None if reflected_path_values is None else reflected_path_values[0]
            ),
        )
        sems = np.full(self.centers.size, np.nan)
        rows, columns = np.nonzero(candidates)
        sems[columns] = grid_sems[rows, columns]
        return sems

    def blocking_diffusivity_summary_grid(
        self,
        path: object,
        candidates: NDArray[np.bool_],
        *,
        reflection_origin: float | None = None,
        reflection_path: object | None = None,
        reflected_candidates: NDArray[np.bool_] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
        """Return per-source diffusivities, blocking SEMs, and exit counts.

        A reflected coordinate, when requested, is pooled with its originating
        trajectory before uncertainty is estimated; it is not treated as an
        independent source.
        """
        if candidates.shape != (self.widths.size, self.centers.size):
            raise ValueError("candidate mask must match the analysis grid")
        trajectory = _load_trajectory_source(path)
        states, durations = _boundary_run_lengths(trajectory, self._boundaries)
        reflected_states: NDArray[np.int64] | None = None
        reflected_durations: NDArray[np.int64] | None = None
        if reflection_origin is not None:
            source = (
                trajectory
                if reflection_path is None
                else _load_trajectory_source(reflection_path)
            )
            reflected = 2.0 * reflection_origin - source
            reflected_states, reflected_durations = _boundary_run_lengths(
                reflected, self._boundaries
            )
        shape = candidates.shape
        diffusivities = np.full(shape, np.nan)
        sems = np.full(shape, np.nan)
        counts = np.zeros(shape, dtype=np.int64)
        reflection_candidates = self._reflection_candidates(reflection_origin)
        reflection_candidates &= (
            candidates if reflected_candidates is None else reflected_candidates
        )
        for width_index, center_index in zip(
            *np.nonzero(candidates | reflection_candidates), strict=True
        ):
            exit_lengths = (
                _observed_exit_lengths(
                    states,
                    durations,
                    self._lower_state[width_index, center_index],
                    self._upper_state[width_index, center_index],
                )
                if candidates[width_index, center_index]
                else np.empty(0, dtype=np.int64)
            )
            if reflection_candidates[width_index, center_index]:
                assert reflected_states is not None and reflected_durations is not None
                reflected_exit_lengths = _observed_exit_lengths(
                    reflected_states,
                    reflected_durations,
                    self._lower_state[width_index, center_index],
                    self._upper_state[width_index, center_index],
                )
                exit_lengths = np.concatenate((exit_lengths, reflected_exit_lengths))
            if exit_lengths.size == 0:
                continue
            count = int(exit_lengths.sum())
            counts[width_index, center_index] = count
            lengths = exit_lengths.astype(float)
            residence = float(
                self.sampling_period * np.sum(lengths * (lengths + 1) / 2) / count
            )
            width = self.widths[width_index]
            diffusivities[width_index, center_index] = width**2 / (12.0 * residence)
            try:
                residence_sem = (
                    _blocking_exit_time_standard_error(exit_lengths).value
                    * self.sampling_period
                )
            except ValueError:
                continue
            sems[width_index, center_index] = (
                width**2 * residence_sem / (12.0 * residence**2)
            )
        return diffusivities, sems, counts


def _warn_high_terminal_censoring(result: DiffusivityAnalysis) -> None:
    """Warn when an analysis contains substantial terminal censoring."""
    fractions = result.terminal_censored_fractions
    affected = np.isfinite(fractions) & (
        fractions > _TERMINAL_CENSORING_WARNING_FRACTION
    )
    if not np.any(affected):
        return
    maximum_index = np.unravel_index(
        np.nanargmax(np.where(affected, fractions, np.nan)), fractions.shape
    )
    maximum = fractions[maximum_index]
    width = result.interval_widths[maximum_index[0]]
    center = result.positions[maximum_index[1]]
    warnings.warn(
        "terminal censoring exceeds "
        f"{_TERMINAL_CENSORING_WARNING_FRACTION:.1%} for {affected.sum()} RTA "
        f"position-width combinations; maximum {maximum:.1%} at width {width:g} "
        f"and center {center:g}. "
        "Diffusivity estimates use completed exits only; "
        "inspect terminal_censored_fractions before interpreting these values.",
        RTATerminalCensoringWarning,
        stacklevel=3,
    )


def _warn_high_terminal_censoring_for_result(
    result: DiffusivityEstimate, xmin: float, xmax: float
) -> None:
    """Warn as soon as a single-interval estimate has high censoring."""
    fraction = result.terminal_censored_fraction
    if not np.isfinite(result.diffusivity) or (
        fraction <= _TERMINAL_CENSORING_WARNING_FRACTION
    ):
        return
    warnings.warn(
        "terminal censoring is "
        f"{fraction:.1%} for the RTA interval [{xmin:g}, {xmax:g}). "
        "This diffusivity estimate uses completed exits only; inspect "
        "terminal_censor_times before interpreting it.",
        RTATerminalCensoringWarning,
        stacklevel=3,
    )


class RTAEstimator:
    """Estimate diffusivity in one specified residence interval.

    ``center`` and ``width`` define the interval. Use this class when the
    interval is already known. Use :class:`RTAAnalyzer` to analyze several
    positions and widths while respecting simulation walls.
    """

    def __init__(
        self,
        center: float,
        width: float,
        sampling_period: float,
    ) -> None:
        center, width = float(center), float(width)
        if not np.isfinite(center):
            raise ValueError("center must be finite")
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError("width must be finite and greater than zero")
        self.center = center
        self.width = width
        self.sampling_period = _validate_sampling_period(sampling_period)
        self._estimator = ResidenceTimeEstimator(
            center - width / 2.0,
            center + width / 2.0,
            self.sampling_period,
        )

    @property
    def interval(self) -> tuple[float, float]:
        """Lower and upper interval boundaries."""
        return self._estimator.xmin, self._estimator.xmax

    def segments(self, trajectory: ArrayLike) -> tuple[ResidenceSegment, ...]:
        """Return contiguous residence segments in the configured interval."""
        return self._estimator.segments(trajectory)

    def analyze(
        self, trajectory: ArrayLike, *, survival_times: ArrayLike | None = None
    ) -> DiffusivityEstimate:
        """Analyze one continuous trajectory."""
        return self._estimator.analyze(trajectory, survival_times=survival_times)

    def analyze_many(
        self,
        trajectories: Iterable[ArrayLike],
        *,
        survival_times: ArrayLike | None = None,
    ) -> DiffusivityEstimate:
        """Analyze independent trajectories without joining their boundaries."""
        return self._estimator.analyze_many(trajectories, survival_times=survival_times)


class RTAAnalyzer:
    """Analyze RTA diffusivity at several positions and interval widths.

    Each :class:`RTAWindow` supplies one or more continuous trajectories and
    the walls that bound them. Intervals crossing a wall are omitted. Use
    :meth:`analyze_overlapping_windows` when several windows cover different
    parts of the coordinate. ``symmetry`` pools observations at mirrored
    positions before fitting.
    """

    def __init__(
        self,
        positions: ArrayLike,
        interval_widths: ArrayLike,
        sampling_period: float,
        *,
        symmetry: ReflectionSymmetry | None = None,
    ) -> None:
        if symmetry is not None and not isinstance(symmetry, ReflectionSymmetry):
            raise TypeError("symmetry must be a ReflectionSymmetry or None")
        period = _validate_sampling_period(sampling_period)
        self._grid_analyzer = _RTAScanEngine(positions, interval_widths, period)
        self.positions = self._grid_analyzer.centers.copy()
        self.interval_widths = self._grid_analyzer.widths.copy()
        self.sampling_period = self._grid_analyzer.sampling_period
        self.symmetry = symmetry

    def analyze(
        self,
        window: RTAWindow | SimulationWindow,
        *,
        reflected_window: RTAWindow | SimulationWindow | None = None,
        include_histograms: bool = False,
        uncertainty: Literal["blocking"] | None = None,
    ) -> DiffusivityAnalysis:
        """Analyze one window and, optionally, its reflected partner."""
        window = _coerce_window(window)
        if uncertainty is not None and reflected_window is not None:
            raise ValueError(
                "blocking uncertainty with a separate reflected window is not "
                "currently supported"
            )
        domain = window.domain
        if reflected_window is None:
            self._validate_reflection_domain(domain)
        reflected_paths = self._reflected_paths(
            domain,
            None if reflected_window is None else _coerce_window(reflected_window),
        )
        result = self._grid_analyzer.analyze_paths(
            window.paths,
            reflection_origin=(None if self.symmetry is None else self.symmetry.origin),
            reflection_paths=reflected_paths,
            include_histograms=include_histograms,
        )
        allowed = self._allowed_candidates(domain)
        masked_result = replace(
            result,
            diffusivities=np.where(allowed, result.diffusivities, np.nan),
            excess_nll=np.where(allowed, result.excess_nll, np.nan),
            n_time_origins=np.where(allowed, result.n_time_origins, 0),
            first_step_exit_fractions=np.where(
                allowed, result.first_step_exit_fractions, np.nan
            ),
            terminal_censored_fractions=np.where(
                allowed, result.terminal_censored_fractions, np.nan
            ),
            sampling_period=self.sampling_period,
            symmetry_center=(None if self.symmetry is None else self.symmetry.origin),
        )
        if uncertainty is not None:
            if uncertainty != "blocking":
                raise ValueError("uncertainty must be 'blocking' or None")
            masked_result = replace(
                masked_result,
                diffusivity_sems=self.pooled_blocking_diffusivity_sems((window,)),
            )
        _warn_high_terminal_censoring(masked_result)
        return masked_result

    def analyze_many(
        self, windows: Iterable[RTAWindow | SimulationWindow]
    ) -> tuple[DiffusivityAnalysis, ...]:
        """Analyze independent windows separately, preserving their identities."""
        return tuple(self.analyze(window) for window in windows)

    def analyze_separately(
        self, windows: Iterable[RTAWindow | SimulationWindow]
    ) -> tuple[DiffusivityAnalysis, ...]:
        """Readable alias for :meth:`analyze_many`."""
        return self.analyze_many(windows)

    def analyze_overlapping_windows(
        self,
        windows: Iterable[RTAWindow | SimulationWindow],
        *,
        include_histograms: bool = False,
        uncertainty: Literal["blocking"] | None = None,
    ) -> DiffusivityAnalysis:
        """Pool every source that physically contains each scanned interval.

        Each trajectory remains an independent source.  For a candidate
        interval, only trajectories whose own wall domain contains that whole
        interval contribute observations.  With reflection symmetry, each
        source also contributes its reflected coordinates where its reflected
        wall domain contains the interval.
        """
        window_values = tuple(_coerce_window(window) for window in windows)
        if not window_values:
            raise ValueError("at least one RTAWindow is required")
        domains = tuple(window.domain for window in window_values)
        paths = tuple(path for window in window_values for path in window.paths)
        source_domains = tuple(
            domain
            for window, domain in zip(window_values, domains, strict=True)
            for _ in window.paths
        )
        candidate_masks = tuple(
            self._allowed_candidates(domain) for domain in source_domains
        )
        if self.symmetry is None:
            result = self._grid_analyzer.analyze_paths(
                paths,
                candidate_masks=candidate_masks,
                include_histograms=include_histograms,
            )
        else:
            origin = self.symmetry.origin
            reflected_masks = tuple(
                self._allowed_candidates(
                    WallDomain(2.0 * origin - domain.upper, 2.0 * origin - domain.lower)
                )
                for domain in source_domains
            )
            result = self._grid_analyzer.analyze_paths(
                paths,
                reflection_origin=origin,
                candidate_masks=candidate_masks,
                reflected_candidate_masks=reflected_masks,
                include_histograms=include_histograms,
            )
        result = replace(
            result,
            sampling_period=self.sampling_period,
            symmetry_center=(None if self.symmetry is None else self.symmetry.origin),
        )
        if uncertainty is not None:
            if uncertainty != "blocking":
                raise ValueError("uncertainty must be 'blocking' or None")
            result = replace(
                result,
                diffusivity_sems=self.pooled_blocking_diffusivity_sems(window_values),
            )
        _warn_high_terminal_censoring(result)
        return result

    def analyze_combined(
        self,
        windows: Iterable[RTAWindow | SimulationWindow],
        **options: Any,
    ) -> DiffusivityAnalysis:
        """Pool compatible observations from multiple simulation windows."""
        return self.analyze_overlapping_windows(windows, **options)

    def blocking_diffusivity_sems(
        self,
        window: RTAWindow,
        interval_widths: ArrayLike,
        *,
        reflected_window: RTAWindow | None = None,
    ) -> NDArray[np.float64]:
        """Return blocking SEMs for selected intervals in one continuous window.

        When a reflected partner is provided, its reflected exit times are
        pooled before estimating the SEM.
        """
        if not isinstance(window, RTAWindow):
            raise TypeError("window must be an RTAWindow")
        domain = window.domain
        if reflected_window is None:
            self._validate_reflection_domain(domain)
        reflected_paths = self._reflected_paths(domain, reflected_window)
        selected = np.asarray(interval_widths, dtype=np.float64)
        if selected.shape != self.positions.shape:
            raise ValueError("interval_widths must have one value per position")
        for center, width in zip(self.positions, selected, strict=True):
            if np.isfinite(width) and not domain.contains_interval(center, width):
                raise ValueError("selected intervals must not cross a wall")
        return self._grid_analyzer.blocking_diffusivity_sems(
            window.paths,
            selected,
            reflection_origin=(None if self.symmetry is None else self.symmetry.origin),
            reflection_paths=reflected_paths,
        )

    def pooled_blocking_diffusivity_sems(
        self, windows: Iterable[RTAWindow | SimulationWindow]
    ) -> NDArray[np.float64]:
        """Estimate SEMs for an analysis of overlapping simulation windows.

        Each trajectory file is one independent blocking source.  Source
        residence-time variances are combined with the same observed-exit
        weights used by the pooled RTA estimator and transformed to
        diffusivity with the delta method.  A candidate is ``NaN`` when any
        contributing source lacks a blocking SEM.
        """
        window_values = tuple(_coerce_window(window) for window in windows)
        if not window_values:
            raise ValueError("at least one RTAWindow is required")
        shape = (self.interval_widths.size, self.positions.size)
        source_diffusivities: list[NDArray[np.float64]] = []
        source_sems: list[NDArray[np.float64]] = []
        source_counts: list[NDArray[np.int64]] = []
        for window in window_values:
            allowed = self._allowed_candidates(window.domain)
            reflected_allowed = None
            if self.symmetry is not None:
                origin = self.symmetry.origin
                reflected_allowed = self._allowed_candidates(
                    WallDomain(
                        2 * origin - window.domain.upper,
                        2 * origin - window.domain.lower,
                    )
                )
            for path in window.paths:
                diffusivity, sem, count = (
                    self._grid_analyzer.blocking_diffusivity_summary_grid(
                        path,
                        allowed,
                        reflection_origin=(
                            None if self.symmetry is None else self.symmetry.origin
                        ),
                        reflected_candidates=reflected_allowed,
                    )
                )
                source_diffusivities.append(diffusivity)
                source_sems.append(sem)
                source_counts.append(count)
        if not source_diffusivities:
            return np.full(shape, np.nan)
        diffusivities = np.stack(source_diffusivities)
        sems = np.stack(source_sems)
        counts = np.stack(source_counts)
        contributing = counts > 0
        complete = np.all(~contributing | np.isfinite(sems), axis=0)
        weights = counts.astype(np.float64)
        widths_squared = self.interval_widths[:, None] ** 2
        usable = contributing & np.isfinite(diffusivities) & (diffusivities > 0.0)
        # Convert each source's diffusivity and SEM back to residence times.
        # Empty sources contribute zero; missing SEMs are rejected by `complete`.
        residence = np.divide(
            widths_squared[None, :, :],
            12.0 * diffusivities,
            out=np.zeros_like(diffusivities),
            where=usable,
        )
        residence_sem = np.divide(
            widths_squared[None, :, :] * sems,
            12.0 * diffusivities**2,
            out=np.zeros_like(sems),
            where=usable & np.isfinite(sems),
        )
        total_weight = weights.sum(axis=0)
        pooled_residence = np.divide(
            (weights * residence).sum(axis=0),
            total_weight,
            out=np.full(shape, np.nan),
            where=total_weight > 0.0,
        )
        pooled_residence_sem = np.divide(
            np.sqrt(((weights * residence_sem) ** 2).sum(axis=0)),
            total_weight,
            out=np.full(shape, np.nan),
            where=total_weight > 0.0,
        )
        pooled_sem = (
            widths_squared * pooled_residence_sem / (12.0 * pooled_residence**2)
        )
        return np.where(complete & np.isfinite(pooled_sem), pooled_sem, np.nan)

    def _validate_reflection_domain(self, domain: WallDomain) -> None:
        if self.symmetry is not None and not np.isclose(
            domain.lower + domain.upper, 2.0 * self.symmetry.origin
        ):
            raise ValueError(
                "reflection symmetry requires walls symmetric about origin"
            )

    def _allowed_candidates(self, domain: WallDomain) -> NDArray[np.bool_]:
        return (
            self.positions[None, :] - self.interval_widths[:, None] / 2.0
            >= domain.lower
        ) & (
            self.positions[None, :] + self.interval_widths[:, None] / 2.0
            <= domain.upper
        )

    def _reflected_paths(
        self, domain: WallDomain, reflected_window: RTAWindow | None
    ) -> tuple[Path, ...] | None:
        if reflected_window is None:
            return None
        if self.symmetry is None:
            raise ValueError("a reflected window requires ReflectionSymmetry")
        if not isinstance(reflected_window, RTAWindow):
            raise TypeError("reflected_window must be an RTAWindow")
        reflected_domain = reflected_window.domain
        origin = self.symmetry.origin
        expected_bounds = (2.0 * origin - domain.upper, 2.0 * origin - domain.lower)
        if not (
            np.isclose(reflected_domain.lower, expected_bounds[0])
            and np.isclose(reflected_domain.upper, expected_bounds[1])
        ):
            raise ValueError(
                "reflected window walls must be the reflection of the source walls"
            )
        return reflected_window.paths


def estimate_diffusivity(
    trajectory: ArrayLike,
    *,
    center: float,
    interval_width: float,
    sampling_period: float,
    survival_times: ArrayLike | None = None,
) -> DiffusivityEstimate:
    """Estimate diffusivity from first exits out of one spatial interval.

    ``sampling_period`` is the time between saved trajectory samples, not the
    molecular-dynamics integration timestep. Coordinate-like inputs must use
    one consistent length unit. The returned diffusivity is expressed in that
    length unit squared per the time unit used for ``sampling_period``.
    """
    return RTAEstimator(
        center,
        interval_width,
        sampling_period,
    ).analyze(_validate_trajectory(trajectory), survival_times=survival_times)


def analyze_diffusivity(
    trajectories: (
        ArrayLike | Iterable[ArrayLike] | SimulationWindow | Iterable[SimulationWindow]
    ),
    *,
    positions: ArrayLike,
    interval_widths: ArrayLike,
    sampling_period: float,
    bounds: tuple[float, float] | None = None,
    symmetry_center: float | None = None,
    uncertainty: Literal["blocking"] | None = None,
) -> DiffusivityAnalysis:
    """Estimate local diffusivity for each position and interval width.

    ``trajectories`` may be one array, a sequence of independent arrays, or a
    sequence of :class:`SimulationWindow` objects. When
    passing raw trajectories, ``bounds`` gives the physical wall positions.
    An interval contributes only when it lies completely inside those bounds.
    ``symmetry_center`` pools physically equivalent observations on opposite
    sides of a symmetric system before estimating diffusivity. All
    coordinate-like inputs must use one consistent length unit. Diffusivity is
    returned in that unit squared per the time unit of ``sampling_period``.
    """
    symmetry = None if symmetry_center is None else ReflectionSymmetry(symmetry_center)
    analyzer = RTAAnalyzer(
        positions,
        interval_widths,
        sampling_period,
        symmetry=symmetry,
    )
    if isinstance(trajectories, SimulationWindow):
        window = trajectories
    else:
        sources = _normalize_trajectory_sources(trajectories)
        if sources and all(
            isinstance(item, (SimulationWindow, RTAWindow)) for item in sources
        ):
            return analyzer.analyze_overlapping_windows(
                sources, include_histograms=True, uncertainty=uncertainty
            )
        if bounds is None:
            raise TypeError("bounds=(lower, upper) is required for raw trajectories")
        window = SimulationWindow(sources, bounds)
    return analyzer.analyze(window, include_histograms=True, uncertainty=uncertainty)


def _candidate_masks(
    paths: tuple[Path, ...],
    masks: Iterable[ArrayLike] | None,
    default: NDArray[np.bool_],
) -> tuple[NDArray[np.bool_], ...]:
    """Validate one candidate mask per independent trajectory path."""
    if masks is None:
        return tuple(default for _ in paths)
    values = tuple(np.asarray(mask, dtype=bool) for mask in masks)
    if len(values) != len(paths) or any(mask.shape != default.shape for mask in values):
        raise ValueError("candidate masks must match the paths and scanned grid")
    return values


def _profile_for_width(
    result: DiffusivityAnalysis,
    selected_index: int | None,
    available: NDArray[np.bool_],
    *,
    method: str = "unspecified",
    nll_rank: int | None = None,
    mean_excess_nll: float | None = None,
    direct_nll_fraction: float | None = None,
    interpolated_nll_fraction: float | None = None,
    extrapolated_nll_fraction: float | None = None,
    coverage: float | None = None,
) -> DiffusivityProfile:
    """Build a selected profile from one row of a grid result."""
    selected_diffusivities = np.full(result.positions.size, np.nan)
    selected_sems = np.full(result.positions.size, np.nan)
    interval_width = float("nan")
    if selected_index is not None:
        positions = np.flatnonzero(available[selected_index])
        interval_width = float(result.interval_widths[selected_index])
        selected_diffusivities[positions] = result.diffusivities[
            selected_index, positions
        ]
        if result.diffusivity_sems is not None:
            selected_sems[positions] = result.diffusivity_sems[
                selected_index, positions
            ]
    return DiffusivityProfile(
        positions=result.positions.copy(),
        diffusivity=selected_diffusivities,
        interval_width=interval_width,
        diffusivity_sem=selected_sems,
        method=method,
        nll_rank=nll_rank,
        mean_excess_nll=mean_excess_nll,
        direct_nll_fraction=direct_nll_fraction,
        interpolated_nll_fraction=interpolated_nll_fraction,
        extrapolated_nll_fraction=extrapolated_nll_fraction,
        coverage=coverage,
    )


def _complete_nll_map(
    centers: NDArray[np.float64],
    losses: NDArray[np.float64],
    directly_supported: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.int8]]:
    """Fill selection-only NLL gaps without inventing diffusivity values.

    Source codes are 0 for unavailable, 1 for evaluated, 2 for interpolated,
    and 3 for constant terminal extrapolation.
    """
    completed = np.full(losses.shape, np.nan)
    sources = np.zeros(losses.shape, dtype=np.int8)
    order = np.argsort(centers, kind="stable")
    sorted_centers = centers[order]
    for width_index in range(losses.shape[0]):
        supported = directly_supported[width_index, order]
        if not np.any(supported):
            continue
        support_positions = sorted_centers[supported]
        support_losses = losses[width_index, order][supported]
        completed[width_index, order] = np.interp(
            sorted_centers, support_positions, support_losses
        )
        row_sources = np.full(centers.size, 2, dtype=np.int8)
        row_sources[sorted_centers < support_positions[0]] = 3
        row_sources[sorted_centers > support_positions[-1]] = 3
        row_sources[supported] = 1
        sources[width_index, order] = row_sources
    return completed, sources


def _discrete_local_minimum_indices(
    values: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Return one representative index for each finite discrete minimum."""
    minima: list[int] = []
    index = 0
    while index < values.size:
        if not np.isfinite(values[index]):
            index += 1
            continue
        stop = index + 1
        while stop < values.size and values[stop] == values[index]:
            stop += 1
        left = values[index - 1] if index > 0 else np.inf
        right = values[stop] if stop < values.size else np.inf
        if values[index] < left and values[index] < right:
            minima.append((index + stop - 1) // 2)
        index = stop
    return np.asarray(minima, dtype=np.int64)


def _boundary_run_lengths(
    trajectory: NDArray[np.float64],
    boundaries: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Run-length encode states induced by ordered interval boundaries.

    A state is the half-open region between two neighboring boundaries.  This
    makes every requested interval an exact union of states, irrespective of
    its center or width.
    """
    states_per_frame = np.searchsorted(boundaries, trajectory, side="right") - 1
    starts = np.flatnonzero(np.r_[True, states_per_frame[1:] != states_per_frame[:-1]])
    return states_per_frame[starts], np.diff(np.r_[starts, trajectory.size]).astype(
        np.int64
    )


def _interval_run_lengths(
    states: NDArray[np.int64],
    durations: NDArray[np.int64],
    lower: np.int64,
    upper: np.int64,
) -> tuple[NDArray[np.int64], bool]:
    """Return consecutive frame counts inside one interval represented by states."""
    inside = (lower <= states) & (states < upper)
    if not np.any(inside):
        return np.empty(0, dtype=np.int64), False
    starts = np.flatnonzero(inside & np.r_[True, ~inside[:-1]])
    stops = np.flatnonzero(inside & np.r_[~inside[1:], True]) + 1
    cumulative = np.r_[0, np.cumsum(durations, dtype=np.int64)]
    lengths = cumulative[stops] - cumulative[starts]
    return lengths, bool(inside[-1])


def _observed_exit_lengths(
    states: NDArray[np.int64],
    durations: NDArray[np.int64],
    lower: int,
    upper: int,
) -> NDArray[np.int64]:
    """Return ordered completed residence lengths, excluding the censored tail."""
    lengths, terminal = _interval_run_lengths(states, durations, lower, upper)
    return lengths[:-1] if terminal else lengths


def _accumulate_length_counts(
    counts: dict[int, int], lengths: NDArray[np.int64]
) -> None:
    """Accumulate segment lengths without storing individual time origins."""
    values, multiplicities = np.unique(lengths, return_counts=True)
    for value, multiplicity in zip(values, multiplicities, strict=True):
        key = int(value)
        counts[key] = counts.get(key, 0) + int(multiplicity)


def _time_origin_histograms(
    observed_lengths: dict[int, int], censored_lengths: dict[int, int]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Expand segment-length counts into compact first-exit/censor histograms."""
    maximum = max((*observed_lengths, *censored_lengths), default=0)
    if maximum == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    observed_by_length = np.zeros(maximum + 1, dtype=np.int64)
    censored_by_length = np.zeros(maximum + 1, dtype=np.int64)
    for length, count in observed_lengths.items():
        observed_by_length[length] = count
    for length, count in censored_lengths.items():
        censored_by_length[length] = count
    # A completed segment of length L contributes exits at 1, ..., L.  A
    # terminal segment contributes right-censoring times 0, ..., L - 1.
    event_counts = np.cumsum(observed_by_length[::-1], dtype=np.int64)[::-1]
    event_counts[0] = 0
    censor_counts = np.empty_like(censored_by_length)
    censor_counts[:-1] = np.cumsum(censored_by_length[:0:-1], dtype=np.int64)[::-1]
    censor_counts[-1] = 0
    return event_counts, censor_counts


def _sparse_histogram_grid(
    observed: list[list[dict[int, int]]], censored: list[list[dict[int, int]]]
) -> FirstExitHistogramGrid:
    """Collect per-interval segment counts as sparse first-exit histograms."""
    observed_rows: list[tuple[int, int, int, int]] = []
    censored_rows: list[tuple[int, int, int, int]] = []
    for width_index, (observed_row, censored_row) in enumerate(
        zip(observed, censored, strict=True)
    ):
        for center_index, (observed_lengths, censored_lengths) in enumerate(
            zip(observed_row, censored_row, strict=True)
        ):
            for duration, count in observed_lengths.items():
                observed_rows.append(
                    (width_index, center_index, int(duration), int(count))
                )
            for duration, count in censored_lengths.items():
                censored_rows.append(
                    (width_index, center_index, int(duration), int(count))
                )

    def columns(
        rows: list[tuple[int, int, int, int]],
    ) -> tuple[
        NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]
    ]:
        if not rows:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty.copy(), empty.copy(), empty.copy()
        values = np.asarray(rows, dtype=np.int64)
        return values[:, 0], values[:, 1], values[:, 2], values[:, 3]

    return FirstExitHistogramGrid(*columns(observed_rows), *columns(censored_rows))


def _kaplan_meier_from_histograms(
    event_counts: NDArray[np.int64], censor_counts: NDArray[np.int64]
) -> tuple[NDArray[np.float64], bool]:
    """Return discrete Kaplan--Meier survival and whether its tail is open."""
    total_counts = event_counts + censor_counts
    if total_counts.size == 0 or total_counts.sum() == 0:
        return np.empty(0, dtype=np.float64), False
    risk = np.cumsum(total_counts[::-1], dtype=np.int64)[::-1]
    hazards = np.divide(
        event_counts,
        risk,
        out=np.zeros_like(event_counts, dtype=np.float64),
        where=risk > 0,
    )
    survival = np.cumprod(1.0 - hazards)
    return survival, bool(survival[-1] > 0.0)


def _histogram_excess_nll(
    event_counts: NDArray[np.int64],
    censor_counts: NDArray[np.int64],
    predicted_survival: NDArray[np.float64],
) -> float:
    """Score a model survival curve against discrete empirical observations."""
    if predicted_survival.shape != event_counts.shape:
        raise ValueError("predicted survival and histogram must have matching shapes")
    total = event_counts.sum() + censor_counts.sum()
    if total == 0:
        return float("nan")
    tiny = np.finfo(np.float64).tiny
    event_probability = np.maximum(
        np.r_[1.0, predicted_survival[:-1]] - predicted_survival,
        tiny,
    )
    model_nll = -np.dot(event_counts, np.log(event_probability))
    model_nll -= np.dot(censor_counts, np.log(np.maximum(predicted_survival, tiny)))
    empirical_survival, _ = _kaplan_meier_from_histograms(event_counts, censor_counts)
    risk = np.cumsum((event_counts + censor_counts)[::-1], dtype=np.int64)[::-1]
    hazard = np.divide(
        event_counts,
        risk,
        out=np.zeros_like(event_counts, dtype=np.float64),
        where=risk > 0,
    )
    empirical_before = np.r_[1.0, empirical_survival[:-1]]
    saturated_nll = -np.dot(
        event_counts, np.log(np.maximum(empirical_before * hazard, tiny))
    )
    saturated_nll -= np.dot(censor_counts, np.log(np.maximum(empirical_survival, tiny)))
    return float((model_nll - saturated_nll) / total)


def _observations_from_segments(
    segments: tuple[ResidenceSegment, ...], sampling_period: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    observed: list[NDArray[np.float64]] = []
    censored: list[NDArray[np.float64]] = []
    for segment in segments:
        length = segment.stop - segment.start
        if segment.exit_observed:
            observed.append(
                np.arange(length, 0, -1, dtype=np.float64) * sampling_period
            )
        else:
            # At the final sampled frame the particle is still inside, so this
            # time origin only proves survival for zero additional time.
            censored.append(
                np.arange(length - 1, -1, -1, dtype=np.float64) * sampling_period
            )
    return (
        np.concatenate(observed) if observed else np.empty(0, dtype=np.float64),
        np.concatenate(censored) if censored else np.empty(0, dtype=np.float64),
    )


def _validate_survival_times(
    times: ArrayLike | None,
    observed: NDArray[np.float64],
    censored: NDArray[np.float64],
    sampling_period: float,
) -> NDArray[np.float64]:
    if times is not None:
        values = np.asarray(times, dtype=np.float64)
        if values.ndim == 0:
            values = values.reshape(1)
        if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(
                "survival_times must be a scalar or one-dimensional finite "
                "non-negative array"
            )
        return values.copy()
    durations = np.concatenate((observed, censored))
    if durations.size == 0:
        return np.empty(0, dtype=np.float64)
    return (
        np.arange(
            int(np.rint(durations.max() / sampling_period)) + 1,
            dtype=np.float64,
        )
        * sampling_period
    )


def _validate_nonnegative_times(times: ArrayLike, name: str) -> NDArray[np.float64]:
    values = np.asarray(times, dtype=np.float64)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(
            f"{name} must be a scalar or one-dimensional finite non-negative array"
        )
    return values.copy()


def _empirical_survival(
    exit_times: NDArray[np.float64], times: NDArray[np.float64]
) -> NDArray[np.float64]:
    if exit_times.size == 0:
        return np.full(times.shape, np.nan)
    return (
        exit_times.size - np.searchsorted(np.sort(exit_times), times, side="right")
    ) / exit_times.size


def _saturated_discrete_mean_nll(
    exit_times: NDArray[np.float64], censor_times: NDArray[np.float64]
) -> float:
    """Return the discrete Kaplan--Meier model's per-observation NLL.

    Subtracting this quantity from a parametric NLL removes the unavoidable
    width-dependent entropy of the sampled first-exit distribution.
    """
    durations = np.concatenate((exit_times, censor_times))
    if durations.size == 0:
        return float("nan")
    event_times, event_counts = np.unique(exit_times, return_counts=True)
    censor_times_unique, censor_counts = np.unique(censor_times, return_counts=True)
    support = np.union1d(event_times, censor_times_unique)
    sorted_durations = np.sort(durations)
    risk = durations.size - np.searchsorted(sorted_durations, support, side="left")
    events = np.zeros(support.size, dtype=np.int64)
    censors = np.zeros(support.size, dtype=np.int64)
    events[np.searchsorted(support, event_times)] = event_counts
    censors[np.searchsorted(support, censor_times_unique)] = censor_counts
    hazard = events / risk
    survival_after = np.cumprod(1.0 - hazard)
    survival_before = np.concatenate(([1.0], survival_after[:-1]))
    tiny = np.finfo(np.float64).tiny
    event_probability = np.maximum(survival_before * hazard, tiny)
    censor_probability = np.maximum(survival_after, tiny)
    nll = -np.dot(events, np.log(event_probability)) - np.dot(
        censors, np.log(censor_probability)
    )
    return float(nll / durations.size)


def _estimates(
    exit_times: NDArray[np.float64], length: float
) -> tuple[float, float, float, float]:
    """Return the ordinary RTA estimate from completed exit observations."""
    if exit_times.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    residence = float(exit_times.mean())
    try:
        residence_sem = (
            StandardErrorEstimator(method="blocking").estimate(exit_times).value
        )
    except ValueError:
        residence_sem = float("nan")
    diffusion = length**2 / (12.0 * residence)
    diffusion_sem = (
        float("nan")
        if not np.isfinite(residence_sem)
        else length**2 * residence_sem / (12.0 * residence**2)
    )
    return residence, residence_sem, float(diffusion), float(diffusion_sem)


def _validate_trajectory(trajectory: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(trajectory, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("trajectory must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("trajectory must contain only finite values")
    return values


def _normalize_trajectory_arrays(sources: object) -> tuple[NDArray[np.float64], ...]:
    """Normalize one trajectory array or independent trajectory arrays."""
    normalized = _normalize_trajectory_sources(sources)
    if any(isinstance(source, (str, Path)) for source in normalized):
        raise TypeError("trajectories must be arrays; load trajectory files first")
    return tuple(_validate_trajectory(source) for source in normalized)  # type: ignore[arg-type]


def _load_trajectory_source(source: object) -> NDArray[np.float64]:
    """Load one trajectory from an array-like object or a text file."""
    if isinstance(source, (str, Path)):
        return _validate_trajectory(np.loadtxt(source))
    return _validate_trajectory(source)  # type: ignore[arg-type]


def _normalize_trajectory_sources(sources: object) -> tuple[object, ...]:
    """Normalize one trajectory or a collection of independent trajectories."""
    if isinstance(sources, np.ndarray):
        if sources.ndim == 2:
            return tuple(row for row in sources)
        return (sources,)
    if isinstance(sources, (str, Path)):
        return (sources,)
    try:
        values = tuple(sources)  # type: ignore[arg-type]
    except TypeError:
        return (sources,)
    if not values:
        return ()
    if all(
        np.isscalar(value) and not isinstance(value, (str, Path)) for value in values
    ):
        return (np.asarray(values, dtype=np.float64),)
    return values


def _coerce_window(window: RTAWindow | SimulationWindow) -> RTAWindow:
    if isinstance(window, SimulationWindow):
        return window.as_rta_window()
    if isinstance(window, RTAWindow):
        return window
    raise TypeError("window must be a SimulationWindow or RTAWindow")


def _validate_configuration(
    xmin: float, xmax: float, sampling_period: float
) -> tuple[float, float, float]:
    xmin, xmax, sampling_period = map(float, (xmin, xmax, sampling_period))
    if not np.all(np.isfinite((xmin, xmax, sampling_period))) or xmin >= xmax:
        raise ValueError("xmin and xmax must be finite and satisfy xmin < xmax")
    if sampling_period <= 0.0:
        raise ValueError("sampling_period must be finite and greater than zero")
    return xmin, xmax, sampling_period


def _validate_sampling_period(value: float) -> float:
    """Return a finite, positive time between saved trajectory frames."""
    try:
        interval = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "sampling_period must be finite and greater than zero"
        ) from error
    if not np.isfinite(interval) or interval <= 0.0:
        raise ValueError("sampling_period must be finite and greater than zero")
    return interval
