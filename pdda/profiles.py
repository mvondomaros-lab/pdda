"""Profile fitting and density-comparison tools for PDDA."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import BSpline, make_smoothing_spline
from scipy.linalg import null_space
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import wasserstein_distance

__all__ = [
    "DensityMetrics",
    "PlateauNotFoundWarning",
    "SmoothingSpline",
    "compare_densities",
    "smooth_profile",
]


class PlateauNotFoundWarning(RuntimeWarning):
    """The requested outer plateau is not supported by the sampled profile."""


@dataclass(frozen=True)
class Plateau:
    """A nearly constant outer part of a one-dimensional profile."""

    start: float
    value: float
    n_points: int
    span: float
    relative_trend: float
    relative_scatter: float


@dataclass(frozen=True)
class SmoothingSpline:
    """A smooth representation of a sampled spatial profile.

    Calling the object evaluates the fitted profile at new coordinates.
    ``domain`` is the range containing input observations. When plateau
    detection is enabled, ``plateaus`` describes any constant outer regions.
    Evaluation outside ``domain`` is allowed only when ``extrapolate`` is true;
    the nearest boundary value is then continued.
    """

    domain: tuple[float, float]
    log_space: bool
    even: bool
    extrapolate: bool
    plateaus: tuple[Plateau | None, Plateau | None]
    _spline: object
    symmetry_center: float = 0.0

    @property
    def plateau_starts(self) -> tuple[float | None, float | None]:
        """Coordinates where the left and right plateaus begin."""
        return tuple(None if item is None else item.start for item in self.plateaus)

    @property
    def plateau_values(self) -> tuple[float | None, float | None]:
        """Detected left and right plateau values in the original value scale."""
        return tuple(None if item is None else item.value for item in self.plateaus)

    def __call__(self, points: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the profile at one or more coordinates."""
        values = np.asarray(points, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("evaluation points must contain only finite values")
        lower, upper = self.domain
        outside = (values < lower) | (values > upper)
        if not self.extrapolate and np.any(outside):
            raise ValueError("evaluation points fall outside the fitted domain")
        physical_points = np.clip(values, lower, upper) if self.extrapolate else values
        spline_points = (
            physical_points - self.symmetry_center if self.even else physical_points
        )
        fitted = np.asarray(self._spline(spline_points), dtype=np.float64)
        for side, plateau in zip(("left", "right"), self.plateaus, strict=True):
            if plateau is not None:
                outside = (
                    physical_points <= plateau.start
                    if side == "left"
                    else physical_points >= plateau.start
                )
                value = np.log(plateau.value) if self.log_space else plateau.value
                fitted = np.where(outside, value, fitted)
        return np.exp(fitted) if self.log_space else fitted


class SmoothingSplineFitter:
    """Configure and fit a smooth one-dimensional profile.

    Use ``log_space=True`` for positive quantities such as diffusivity. Set
    ``even=True`` only for a profile known to be reflection-symmetric. The
    observations are averaged with their reflected partners before fitting a
    smooth full-domain spline, which enforces zero slope at the symmetry center.
    Set ``plateau=True`` to identify approximately constant outer regions.
    """

    def __init__(
        self,
        *,
        lam: float | None = None,
        even: bool = False,
        log_space: bool = False,
        plateau: bool = False,
        plateau_tolerance: float = 0.05,
        minimum_plateau_points: int = 2,
        extrapolate: bool = False,
        symmetry_center: float = 0.0,
    ) -> None:
        if not isinstance(even, (bool, np.bool_)):
            raise TypeError("even must be a boolean")
        if not isinstance(log_space, (bool, np.bool_)):
            raise TypeError("log_space must be a boolean")
        if not isinstance(plateau, (bool, np.bool_)):
            raise TypeError("plateau must be a boolean")
        if not isinstance(extrapolate, (bool, np.bool_)):
            raise TypeError("extrapolate must be a boolean")
        if lam is not None:
            lam = float(lam)
            if not np.isfinite(lam) or lam < 0.0:
                raise ValueError("lam must be a finite non-negative number or None")
        self.lam = lam
        self.even = bool(even)
        self.log_space = bool(log_space)
        self.plateau = bool(plateau)
        self.plateau_tolerance = _validate_plateau_tolerance(plateau_tolerance)
        if (
            not isinstance(minimum_plateau_points, (int, np.integer))
            or isinstance(minimum_plateau_points, (bool, np.bool_))
            or minimum_plateau_points < 2
        ):
            raise ValueError("minimum_plateau_points must be an integer of at least 2")
        self.minimum_plateau_points = int(minimum_plateau_points)
        self.extrapolate = bool(extrapolate)
        self.symmetry_center = float(symmetry_center)
        if not np.isfinite(self.symmetry_center):
            raise ValueError("symmetry_center must be finite")
        if not self.even and self.symmetry_center != 0.0:
            raise ValueError("symmetry_center requires even=True")

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        support_positions: ArrayLike | None = None,
    ) -> SmoothingSpline:
        """Fit a spline; support_positions retains gaps for plateau detection."""
        x_values, y_values = _validate_spline_data(x, y, self.log_space)
        support_grid = (
            x_values
            if support_positions is None
            else np.asarray(support_positions, dtype=np.float64)
        )
        if (
            support_grid.ndim != 1
            or not np.all(np.isfinite(support_grid))
            or np.any(np.diff(support_grid) <= 0.0)
            or not np.all(np.isin(x_values, support_grid))
        ):
            raise ValueError("support_positions must be increasing and contain x")
        if self.even:
            x_values = x_values - self.symmetry_center
            x_values, y_values = _symmetrize_spline_data(x_values, y_values)
            extent = min(-x_values[0], x_values[-1])
            support_grid = np.unique(np.abs(support_grid - self.symmetry_center))
            support_grid = support_grid[support_grid <= extent]
        values = np.log(y_values) if self.log_space else y_values
        plateaus = _find_plateaus(
            x_values,
            y_values,
            support_grid,
            self.plateau_tolerance if self.plateau else None,
            self.even,
            self.log_space,
            self.minimum_plateau_points,
            True,
        )
        domain = (float(x_values[0]), float(x_values[-1]))
        spline = (
            _fit_constrained_spline(
                x_values,
                values,
                plateaus,
                self.lam,
                log_space=self.log_space,
                extrapolate=self.extrapolate,
            )
            if self.extrapolate or any(item is not None for item in plateaus)
            else make_smoothing_spline(x_values, values, lam=self.lam)
        )
        if self.even:
            domain = tuple(value + self.symmetry_center for value in domain)
            plateaus = tuple(
                _shift_plateau(plateau, self.symmetry_center) for plateau in plateaus
            )
        return SmoothingSpline(
            domain=domain,
            log_space=self.log_space,
            even=self.even,
            extrapolate=self.extrapolate,
            plateaus=plateaus,
            _spline=spline,
            symmetry_center=self.symmetry_center,
        )


def smooth_profile(
    positions: ArrayLike,
    values: ArrayLike,
    *,
    positive: bool = False,
    symmetry_center: float | None = None,
    smoothing_length: float | None = None,
    plateau: bool = False,
    plateau_tolerance: float = 0.05,
    minimum_plateau_points: int = 2,
    extrapolate: bool = False,
) -> SmoothingSpline:
    """Fit a smooth one-dimensional profile for interpolation or propagation.

    Non-finite observations are omitted. With ``positive=True``, non-positive
    observations are also omitted and the logarithm of the profile is fitted,
    which guarantees positive interpolated values.
    ``smoothing_length`` controls smoothness in coordinate units; larger
    values suppress variations on longer spatial scales. When omitted, the
    smoothing strength is selected from the data by generalized
    cross-validation.

    ``symmetry_center`` averages equivalent observations on both sides and
    gives the fitted profile zero slope at the symmetry plane. Set
    ``plateau=True`` for a profile expected to reach a bulk value. Plateau
    detection uses the outermost contiguous supported block in the outer half
    of each side. Two supported points suffice by default when relative trend
    and scatter meet the tolerance. A detected bulk region is imposed during
    fitting: the spline reaches its bulk value with zero slope and curvature
    and is exactly constant from its detected start.

    Evaluation outside the fitted domain raises an error unless
    ``extrapolate=True``. Extrapolation continues the nearest boundary value.
    On sides without a detected plateau, this also imposes zero endpoint slope
    without prescribing a bulk value or flattening an interior region.
    """
    x = np.asarray(positions, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError(
            "positions and values must be equal-length one-dimensional arrays"
        )
    support_positions = x[np.isfinite(x)]
    keep = np.isfinite(x) & np.isfinite(y)
    if positive:
        keep &= y > 0.0
    x, y = x[keep], y[keep]
    smoothing = (
        None
        if smoothing_length is None
        else _lam_from_smoothing_length(x, smoothing_length)
    )
    fitted = SmoothingSplineFitter(
        lam=smoothing,
        even=symmetry_center is not None,
        symmetry_center=0.0 if symmetry_center is None else symmetry_center,
        log_space=positive,
        plateau=plateau,
        plateau_tolerance=plateau_tolerance,
        minimum_plateau_points=minimum_plateau_points,
        extrapolate=extrapolate,
    ).fit(x, y, support_positions=support_positions)
    if plateau:
        missing = [
            side
            for side, detected in zip(("left", "right"), fitted.plateaus, strict=True)
            if detected is None
        ]
        if missing:
            warnings.warn(
                "no supported outer plateau was found on the "
                f"{' and '.join(missing)} side"
                + ("s" if len(missing) == 2 else "")
                + (
                    "; only zero endpoint slope is imposed for constant extrapolation"
                    if extrapolate
                    else "; the smooth profile is unconstrained there"
                ),
                PlateauNotFoundWarning,
                stacklevel=2,
            )
    return fitted


def symmetrize_even_profile(
    x: ArrayLike, y: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reflect-average a profile about zero over its shared domain.

    This is useful for membrane profiles whose two leaflets are pooled before
    fitting or propagation. Input coordinates must be strictly increasing and
    span zero.
    """
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if (
        x_values.ndim != 1
        or y_values.ndim != 1
        or x_values.size != y_values.size
        or x_values.size < 2
        or not np.all(np.isfinite(x_values))
        or not np.all(np.isfinite(y_values))
        or np.any(np.diff(x_values) <= 0.0)
        or x_values[0] >= 0.0
        or x_values[-1] <= 0.0
    ):
        raise ValueError("x and y must be finite, increasing profiles spanning zero")
    spacing = float(np.median(np.diff(x_values)))
    extent = min(abs(x_values[0]), abs(x_values[-1]))
    positive = np.arange(0.0, extent + spacing / 2.0, spacing)
    positive = positive[positive <= extent + 1e-10]
    values = 0.5 * (
        np.interp(positive, x_values, y_values)
        + np.interp(-positive, x_values, y_values)
    )
    return np.r_[-positive[:0:-1], positive], np.r_[values[:0:-1], values]


def _lam_from_smoothing_length(x: NDArray[np.float64], length: float) -> float:
    length = float(length)
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("smoothing_length must be a finite positive number or None")
    unique = np.sort(np.unique(x))
    if unique.size < 2:
        raise ValueError("smoothing_length requires at least two distinct positions")
    spacing = float(np.median(np.diff(unique)))
    return length**4 / spacing


def _validate_spline_data(
    x: ArrayLike, y: ArrayLike, log_space: bool
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if (
        x_values.ndim != 1
        or y_values.ndim != 1
        or x_values.size != y_values.size
        or x_values.size < 5
    ):
        raise ValueError(
            "x and y must be equal-length one-dimensional arrays with at least "
            "five points"
        )
    if (
        not np.all(np.isfinite(x_values))
        or not np.all(np.isfinite(y_values))
        or np.any(np.diff(x_values) <= 0.0)
    ):
        raise ValueError("x and y must be finite, with x strictly increasing")
    if log_space and np.any(y_values <= 0.0):
        raise ValueError("log_space requires strictly positive y values")
    return x_values, y_values


def _validate_plateau_tolerance(value: float) -> float:
    tolerance = float(value)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("plateau_tolerance must be a finite positive number")
    return tolerance


def _symmetrize_spline_data(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if x[0] >= 0.0 or x[-1] <= 0.0:
        raise ValueError("even fitting requires x to span zero")
    extent = min(-x[0], x[-1])
    raw_radii = np.sort(np.abs(x[np.abs(x) <= extent]))
    tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, extent)
    starts = np.r_[0, np.flatnonzero(np.diff(raw_radii) > tolerance) + 1]
    counts = np.diff(np.r_[starts, raw_radii.size])
    radii = np.add.reduceat(raw_radii, starts) / counts
    if radii.size < 5:
        raise ValueError("even fitting requires five mirrored data points")
    averaged = 0.5 * (np.interp(radii, x, y) + np.interp(-radii, x, y))
    return (
        np.r_[-radii[:0:-1], radii],
        np.r_[averaged[:0:-1], averaged],
    )


def _find_plateaus(
    x: NDArray[np.float64],
    values: NDArray[np.float64],
    support_positions: NDArray[np.float64],
    tolerance: float | None,
    even: bool,
    log_space: bool,
    minimum_points: int,
    outer_half: bool,
) -> tuple[Plateau | None, Plateau | None]:
    if tolerance is None:
        return None, None
    if even:
        positive = x >= 0.0
        right = _find_right_plateau(
            x[positive],
            values[positive],
            tolerance,
            log_space=log_space,
            minimum_points=minimum_points,
            outer_half=outer_half,
            support_positions=support_positions[support_positions >= 0.0],
        )
        if right is None:
            return None, None
        left = Plateau(
            start=-right.start,
            value=right.value,
            n_points=right.n_points,
            span=right.span,
            relative_trend=right.relative_trend,
            relative_scatter=right.relative_scatter,
        )
        return left, right
    right = _find_right_plateau(
        x,
        values,
        tolerance,
        log_space=log_space,
        minimum_points=minimum_points,
        outer_half=outer_half,
        support_positions=support_positions,
    )
    left = _find_right_plateau(
        -x[::-1],
        values[::-1],
        tolerance,
        log_space=log_space,
        minimum_points=minimum_points,
        outer_half=outer_half,
        support_positions=-support_positions[::-1],
    )
    if left is not None:
        left = Plateau(
            start=-left.start,
            value=left.value,
            n_points=left.n_points,
            span=left.span,
            relative_trend=left.relative_trend,
            relative_scatter=left.relative_scatter,
        )
    if (
        left is not None
        and right is not None
        and left.start >= right.start
        and left.value != right.value
    ):
        # Leave an interior span between independently estimated bulk values.
        # Sharing a join would impose two potentially different values there.
        middle = 0.5 * (x[0] + x[-1])
        outer_left = x < middle
        outer_right = x > middle
        left_tail = _find_right_plateau(
            -x[outer_left][::-1],
            values[outer_left][::-1],
            tolerance,
            log_space=log_space,
            minimum_points=minimum_points,
            outer_half=False,
            support_positions=-support_positions[support_positions < middle][::-1],
        )
        left = (
            None
            if left_tail is None
            else Plateau(
                start=-left_tail.start,
                value=left_tail.value,
                n_points=left_tail.n_points,
                span=left_tail.span,
                relative_trend=left_tail.relative_trend,
                relative_scatter=left_tail.relative_scatter,
            )
        )
        right = _find_right_plateau(
            x[outer_right],
            values[outer_right],
            tolerance,
            log_space=log_space,
            minimum_points=minimum_points,
            outer_half=False,
            support_positions=support_positions[support_positions > middle],
        )
    return left, right


def _shift_plateau(plateau: Plateau | None, offset: float) -> Plateau | None:
    if plateau is None:
        return None
    return Plateau(
        start=plateau.start + offset,
        value=plateau.value,
        n_points=plateau.n_points,
        span=plateau.span,
        relative_trend=plateau.relative_trend,
        relative_scatter=plateau.relative_scatter,
    )


def _find_right_plateau(
    x: NDArray[np.float64],
    values: NDArray[np.float64],
    tolerance: float,
    *,
    log_space: bool,
    minimum_points: int,
    outer_half: bool,
    support_positions: NDArray[np.float64] | None = None,
) -> Plateau | None:
    grid = x if support_positions is None else support_positions
    last_start = x.size - minimum_points
    first_start = (
        int(np.searchsorted(x, 0.5 * (grid[0] + grid[-1]))) if outer_half else 0
    )
    grid_indices = np.searchsorted(grid, x)
    gaps = np.flatnonzero(np.diff(grid_indices) > 1)
    if gaps.size:
        first_start = max(first_start, int(gaps[-1] + 1))
    for start in range(first_start, last_start + 1):
        tail_x, tail_values = x[start:], values[start:]
        if log_space:
            transformed = np.log(tail_values)
            plateau_value = float(np.exp(np.median(transformed)))
            allowed = np.log1p(tolerance)
            trend = abs(np.polyfit(tail_x, transformed, 1)[0]) * (
                tail_x[-1] - tail_x[0]
            )
            scatter = 1.4826 * np.median(np.abs(transformed - np.median(transformed)))
            relative_trend = float(np.expm1(trend))
            relative_scatter = float(np.expm1(scatter))
            accepted = trend <= allowed and scatter <= allowed
        else:
            plateau_value = float(np.median(tail_values))
            scale = max(abs(plateau_value), np.finfo(float).eps)
            trend = abs(np.polyfit(tail_x, tail_values, 1)[0]) * (
                tail_x[-1] - tail_x[0]
            )
            scatter = 1.4826 * np.median(np.abs(tail_values - plateau_value))
            relative_trend = float(trend / scale)
            relative_scatter = float(scatter / scale)
            accepted = relative_trend <= tolerance and relative_scatter <= tolerance
        if accepted:
            return Plateau(
                start=float(tail_x[0]),
                value=plateau_value,
                n_points=int(tail_x.size),
                span=float(tail_x[-1] - tail_x[0]),
                relative_trend=relative_trend,
                relative_scatter=relative_scatter,
            )
    return None


def _fit_constrained_spline(
    x: NDArray[np.float64],
    values: NDArray[np.float64],
    plateaus: tuple[Plateau | None, Plateau | None],
    lam: float | None,
    *,
    log_space: bool,
    extrapolate: bool,
) -> BSpline:
    """Minimize squared residuals plus integrated squared curvature.

    Boundary conditions are exact linear constraints on cubic B-spline
    coefficients. GCV, when requested, uses the constrained fit's effective
    degrees of freedom, conditional on the detected plateau values and starts.
    """
    left, right = plateaus
    lower = x[0] if left is None else left.start
    upper = x[-1] if right is None else right.start
    if lower == upper:
        # Identical touching plateaus describe a constant profile throughout.
        value = np.log(left.value) if log_space else left.value
        return BSpline(
            np.r_[np.repeat(x[0], 4), np.repeat(x[-1], 4)], np.full(4, value), 3
        )
    interior = (x >= lower) & (x <= upper)
    fit_x, fit_y = x[interior], values[interior]
    breaks = np.unique(np.r_[lower, fit_x, upper])
    if breaks.size < 5:
        # A short interior still needs enough freedom to satisfy both ends'
        # value, slope, and curvature conditions independently.
        breaks = np.unique(np.r_[breaks, np.linspace(lower, upper, 5)])
    knots = np.r_[np.repeat(lower, 3), breaks, np.repeat(upper, 3)]
    basis = BSpline(knots, np.eye(len(knots) - 4), 3)
    rows, targets = [], []
    for endpoint, plateau in ((lower, left), (upper, right)):
        if plateau is not None:
            rows.extend([basis(endpoint, nu=order) for order in (0, 1, 2)])
            value = np.log(plateau.value) if log_space else plateau.value
            targets.extend([value, 0.0, 0.0])
        elif extrapolate:
            rows.append(basis(endpoint, nu=1))
            targets.append(0.0)
    constraints = np.asarray(rows)
    scales = np.linalg.norm(constraints, axis=1)
    constraints = constraints / scales[:, None]
    particular = np.linalg.lstsq(constraints, np.asarray(targets) / scales, rcond=None)[
        0
    ]
    free = null_space(constraints)
    design = basis(fit_x)
    # Two-point Gauss quadrature integrates products of cubic-spline second
    # derivatives exactly on each knot span.
    midpoints = (breaks[:-1] + breaks[1:]) / 2
    half_spans = np.diff(breaks) / 2
    quadrature_x = (
        midpoints[:, None] + half_spans[:, None] * np.array([-1, 1]) / np.sqrt(3)
    ).ravel()
    curvature = basis(quadrature_x, nu=2) * np.sqrt(np.repeat(half_spans, 2))[:, None]
    a, b = design @ free, fit_y - design @ particular
    r, s = curvature @ free, curvature @ particular
    gram, penalty = a.T @ a, r.T @ r

    def solve(strength: float) -> tuple[NDArray[np.float64], float]:
        if strength == 0.0:
            coefficients, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
            return coefficients, float(rank)
        matrix = gram + strength * penalty
        coefficients = np.linalg.solve(matrix, a.T @ b - strength * r.T @ s)
        degrees = np.trace(np.linalg.solve(matrix, gram))
        return coefficients, float(degrees)

    if free.shape[1] == 0:
        return BSpline(knots, particular, 3)
    if lam is None:
        scale = np.trace(gram) / max(np.trace(penalty), np.finfo(float).tiny)

        def gcv(log_strength: float) -> float:
            coefficients, degrees = solve(scale * np.exp(log_strength))
            residual = a @ coefficients - b
            return float(np.sum(residual**2) / max(len(b) - degrees, 1e-12) ** 2)

        optimum = minimize_scalar(gcv, bounds=(-20.0, 20.0), method="bounded")
        lam = scale * np.exp(optimum.x)
    coefficients, _ = solve(lam)
    return BSpline(knots, particular + free @ coefficients, 3)


@dataclass(frozen=True)
class DensityMetrics:
    """Differences between a reference and a model probability density.

    Every measure is zero for identical densities. Total variation, Hellinger,
    and Jensen--Shannon distances are dimensionless. ``wasserstein_1`` has the
    coordinate's length unit and measures the average transport distance. The
    KL and Renyi quantities emphasize regions where the model assigns too
    little probability.
    """

    total_variation: float
    wasserstein_1: float
    hellinger: float
    jensen_shannon: float
    kl_reference_model: float
    renyi_2: float
    reference_mass: float
    model_mass: float
    regularized_model: bool


class DensityComparator:
    """Reusable comparison configuration for densities on one fixed grid."""

    def __init__(
        self,
        grid: ArrayLike,
        *,
        periodic: bool = False,
        normalize: bool = False,
        density_floor: float = 1e-15,
        normalization: Literal["check", "normalize"] | None = None,
    ) -> None:
        if normalization is not None:
            if normalization not in ("check", "normalize"):
                raise ValueError("normalization must be 'check' or 'normalize'")
            if normalize is not False:
                raise TypeError(
                    "pass normalization, not both normalization and normalize"
                )
            normalize = normalization == "normalize"
        if not isinstance(periodic, (bool, np.bool_)):
            raise TypeError("periodic must be a boolean")
        if not isinstance(normalize, (bool, np.bool_)):
            raise TypeError("normalize must be a boolean")
        density_floor = float(density_floor)
        if not np.isfinite(density_floor) or density_floor <= 0.0:
            raise ValueError("density_floor must be finite and greater than zero")
        self.grid = np.asarray(grid, dtype=np.float64).copy()
        self.periodic = bool(periodic)
        self.normalize = bool(normalize)
        self.density_floor = density_floor
        self.normalization = "normalize" if self.normalize else "check"
        # Validate grid and configuration once, independent of the profiles.
        _validate_densities(
            np.ones_like(self.grid), np.ones_like(self.grid), self.grid, periodic
        )

    def compare(self, reference: ArrayLike, model: ArrayLike) -> DensityMetrics:
        """Compare reference and model densities on the configured grid."""
        return _compare_densities(
            reference,
            model,
            self.grid,
            periodic=self.periodic,
            normalize=self.normalize,
            density_floor=self.density_floor,
        )


def compare_densities(
    grid: ArrayLike,
    reference: ArrayLike,
    model: ArrayLike,
    *,
    periodic: bool = False,
    normalization: Literal["check", "normalize"] = "check",
) -> DensityMetrics:
    """Compare two one-dimensional probability densities on a shared grid.

    Set ``periodic=True`` when the first and last grid cells are adjacent.
    With ``normalization="check"``, both profiles must already integrate to
    one; ``normalization="normalize"`` rescales them before comparison.
    """
    return DensityComparator(
        grid,
        periodic=periodic,
        normalization=normalization,
    ).compare(reference, model)


def _compare_densities(
    reference: ArrayLike,
    model: ArrayLike,
    grid: ArrayLike,
    *,
    periodic: bool = False,
    normalize: bool = False,
    density_floor: float = 1e-15,
) -> DensityMetrics:
    """Compare normalized densities, optionally normalizing supplied profiles."""
    grid_values, reference_values, model_values, weights = _validate_densities(
        reference, model, grid, periodic
    )
    reference_mass, model_mass = (
        float(np.dot(reference_values, weights)),
        float(np.dot(model_values, weights)),
    )
    if not normalize and (
        not np.isclose(reference_mass, 1.0, rtol=1e-10, atol=1e-12)
        or not np.isclose(model_mass, 1.0, rtol=1e-10, atol=1e-12)
    ):
        raise ValueError(
            "densities must be normalized; pass normalize=True to normalize them"
        )
    reference_values, model_values = (
        reference_values / reference_mass,
        model_values / model_mass,
    )
    positive_reference, positive_model = reference_values > 0.0, model_values > 0.0

    def js_term(values: NDArray[np.float64], positive: NDArray[np.bool_]) -> float:
        # Form log(2 * value / (reference + model)) without halving a
        # subnormal mixture density to zero.
        supported = values[positive]
        log_ratio = (
            np.log(2.0)
            + np.log(supported)
            - np.log(reference_values[positive] + model_values[positive])
        )
        return float(np.dot(supported * log_ratio, weights[positive]))

    js = 0.5 * (
        js_term(reference_values, positive_reference)
        + js_term(model_values, positive_model)
    )
    regularized = bool(np.any(model_values == 0.0))
    model_for_divergence = model_values.copy()
    if regularized:
        model_for_divergence = np.maximum(
            model_for_divergence, density_floor / weights.sum()
        )
        model_for_divergence /= np.dot(model_for_divergence, weights)
    kl = np.dot(
        np.where(
            positive_reference,
            reference_values
            * np.log(
                np.divide(
                    reference_values,
                    model_for_divergence,
                    out=np.ones_like(reference_values),
                    where=positive_reference,
                )
            ),
            0.0,
        ),
        weights,
    )
    log_renyi = logsumexp(
        2.0 * np.log(reference_values[positive_reference])
        - np.log(model_for_divergence[positive_reference])
        + np.log(weights[positive_reference])
    )
    if periodic:
        cumulative = np.cumsum(reference_values - model_values) * weights[0]
        wasserstein = np.sum(np.abs(cumulative - np.median(cumulative))) * weights[0]
    else:
        wasserstein = wasserstein_distance(
            grid_values,
            grid_values,
            u_weights=reference_values * weights,
            v_weights=model_values * weights,
        )
    return DensityMetrics(
        float(0.5 * np.dot(np.abs(reference_values - model_values), weights)),
        float(wasserstein),
        float(
            np.sqrt(
                0.5
                * np.dot(
                    (np.sqrt(reference_values) - np.sqrt(model_values)) ** 2, weights
                )
            )
        ),
        float(js),
        float(kl),
        float(max(log_renyi, 0.0)),
        reference_mass,
        model_mass,
        regularized,
    )


def _validate_densities(
    reference: ArrayLike, model: ArrayLike, grid: ArrayLike, periodic: bool
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    if not isinstance(periodic, (bool, np.bool_)):
        raise TypeError("periodic must be a boolean")
    grid_values, reference_values, model_values = (
        np.asarray(grid, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
        np.asarray(model, dtype=np.float64),
    )
    if (
        grid_values.ndim != 1
        or grid_values.size < 2
        or reference_values.shape != grid_values.shape
        or model_values.shape != grid_values.shape
    ):
        raise ValueError(
            "grid, reference, and model must be matching one-dimensional arrays "
            "with at least two points"
        )
    spacing = np.diff(grid_values)
    if not np.all(np.isfinite(grid_values)) or np.any(spacing <= 0.0):
        raise ValueError("grid must be finite and strictly increasing")
    if (
        not np.all(np.isfinite(reference_values))
        or not np.all(np.isfinite(model_values))
        or np.any(reference_values < 0.0)
        or np.any(model_values < 0.0)
    ):
        raise ValueError("densities must contain finite, non-negative values")
    if periodic:
        tolerance = 32 * np.finfo(np.float64).eps * max(1.0, abs(float(spacing[0])))
        if not np.all(np.abs(spacing - spacing[0]) <= tolerance):
            raise ValueError("a periodic grid must be uniformly spaced")
        weights = np.full(grid_values.size, spacing[0])
    else:
        weights = np.empty(grid_values.size)
        weights[0], weights[-1], weights[1:-1] = (
            spacing[0] / 2.0,
            spacing[-1] / 2.0,
            (spacing[:-1] + spacing[1:]) / 2.0,
        )
    if np.dot(reference_values, weights) <= 0.0 or np.dot(model_values, weights) <= 0.0:
        raise ValueError("densities must have positive mass")
    return grid_values, reference_values, model_values, weights
