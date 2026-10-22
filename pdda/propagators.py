"""Propagate densities with the one-dimensional Smoluchowski equation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import expm_multiply

__all__ = [
    "PropagationResult",
    "periodic_grid",
    "propagate_density",
]

_FLOAT: Final = np.float64


@dataclass(frozen=True)
class PropagationResult:
    """Probability densities at the requested propagation times.

    ``densities[i]`` is the density at ``times[i]``. Each row integrates to
    one with respect to the uniform spacing of ``positions``.
    """

    positions: NDArray[np.float64]
    times: NDArray[np.float64]
    densities: NDArray[np.float64]
    equilibrium: NDArray[np.float64]


class SmoluchowskiSolver:
    """Reusable solver for advanced or repeated propagation calculations.

    Most analyses should use :func:`propagate_density`. This class exposes the
    precomputed Smoluchowski generator when several initial conditions or time
    series will be propagated through the same physical profiles.

    Parameters
    ----------
    grid
        Strictly increasing, uniformly spaced one-dimensional cell centers.
        The supplied values are retained exactly.
    diffusivity, reduced_potential
        Finite one-dimensional arrays matching ``grid``. Diffusivity must be
        strictly positive. ``reduced_potential`` is dimensionless, i.e. the
        potential of mean force divided by :math:`k_B T`.
    periodic
        If true, probability can cross between the first and last grid cells.
        If false, both ends are reflecting boundaries with zero flux.
    """

    def __init__(
        self,
        grid: ArrayLike,
        diffusivity: ArrayLike,
        reduced_potential: ArrayLike,
        *,
        periodic: bool = False,
    ) -> None:
        self.grid, self._spacing = _validate_grid(grid)
        self.diffusivity = _validate_profile(diffusivity, self.grid, "diffusivity")
        if np.any(self.diffusivity <= 0.0):
            raise ValueError("diffusivity must be strictly positive")
        self.reduced_potential = _validate_profile(
            reduced_potential, self.grid, "reduced_potential"
        )
        if not isinstance(periodic, (bool, np.bool_)):
            raise TypeError("periodic must be a boolean")
        self.periodic = bool(periodic)

        # This column generator evolves cell densities and is built once for
        # all propagation requests made through this solver.
        self._generator = _generator(
            self.reduced_potential, self.diffusivity, self._spacing, self.periodic
        )
        equilibrium = np.exp(-(self.reduced_potential - np.min(self.reduced_potential)))
        self.equilibrium = equilibrium / (equilibrium.sum() * self._spacing)

    def propagate(
        self,
        times: ArrayLike,
        initial_position: float | None = None,
        *,
        initial_density: ArrayLike | None = None,
    ) -> PropagationResult:
        """Calculate the probability density at one or more non-negative times.

        By default, all probability starts in the central grid cell. Set
        ``initial_position`` to start near another coordinate, or provide a
        non-negative ``initial_density``. Supplied densities are normalized to
        unit probability. A scalar time produces one row in ``densities``.
        """
        requested_times = _validate_times(times)
        if initial_density is not None and initial_position is not None:
            raise TypeError("pass either initial_position or initial_density, not both")
        if initial_density is None:
            initial_index = self._initial_index(initial_position)
            starting_density = np.zeros(self.grid.size, dtype=_FLOAT)
            starting_density[initial_index] = 1.0 / self._spacing
        else:
            starting_density = _validate_profile(
                initial_density, self.grid, "initial_density"
            )
            if np.any(starting_density < 0.0):
                raise ValueError("initial_density must be non-negative")
            mass = float(starting_density.sum() * self._spacing)
            if mass <= 0.0:
                raise ValueError("initial_density must have positive mass")
            starting_density /= mass

        densities = np.empty((requested_times.size, self.grid.size), dtype=_FLOAT)
        for index, time in enumerate(requested_times):
            densities[index] = expm_multiply(self._generator * time, starting_density)
        if np.any(densities < -1e-12):
            raise RuntimeError("propagation produced a negative density")
        densities = np.maximum(densities, 0.0)
        return PropagationResult(
            positions=self.grid.copy(),
            times=requested_times,
            densities=densities,
            equilibrium=self.equilibrium.copy(),
        )

    def _initial_index(self, initial_position: float | None) -> int:
        if initial_position is None:
            return self.grid.size // 2
        try:
            position = float(initial_position)
        except (TypeError, ValueError) as error:
            raise ValueError("initial_position must be an exact grid point") from error
        if not np.isfinite(position):
            raise ValueError("initial_position must be an exact grid point")
        index = int(np.argmin(np.abs(self.grid - position)))
        if abs(self.grid[index] - position) >= self._spacing / 2.0 - 1e-12:
            raise ValueError(
                "initial_position must be an exact grid point or unambiguously "
                "nearest to one"
            )
        return index


def periodic_grid(
    period: float, target_spacing: float
) -> tuple[NDArray[np.float64], float]:
    """Create a zero-centered periodic grid for a cell of length ``period``.

    The number of cells is chosen so that the actual spacing is close to
    ``target_spacing`` and the grid contains a cell centered at zero. At least
    three cells are retained so the grid can be used for propagation. Both
    the grid and its actual spacing are returned.
    """
    period = float(period)
    target_spacing = float(target_spacing)
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("period must be finite and positive")
    if not np.isfinite(target_spacing) or target_spacing <= 0.0:
        raise ValueError("target_spacing must be finite and positive")
    n_cells = max(3, round(period / target_spacing))
    n_cells += (n_cells + 1) % 2
    spacing = period / n_cells
    return (np.arange(n_cells, dtype=_FLOAT) - n_cells // 2) * spacing, spacing


def propagate_density(
    positions: ArrayLike,
    diffusivity: ArrayLike,
    reduced_potential: ArrayLike,
    times: ArrayLike,
    *,
    initial_position: float | None = None,
    initial_density: ArrayLike | None = None,
    periodic: bool = False,
) -> PropagationResult:
    """Predict how a density evolves along a molecular coordinate.

    Parameters
    ----------
    positions
        Strictly increasing, uniformly spaced coordinates. Their unit must be
        consistent with the length unit used for ``diffusivity``.
    diffusivity
        Positive local diffusivity at every position, in length squared per
        time. Its time unit determines the unit of ``times``.
    reduced_potential
        Dimensionless potential of mean force, :math:`F/(k_B T)`, at every
        position. Do not pass an energy-valued PMF.
    times
        Non-negative propagation times. A scalar is accepted and produces one
        row in ``densities``.
    initial_position
        Coordinate at which to localize the initial probability. The nearest
        unambiguous cell is used. By default, the central cell is used.
    initial_density
        Alternative non-negative initial density sampled at ``positions``.
        It is normalized to unit probability. Pass this or
        ``initial_position``, not both.
    periodic
        If true, probability crosses between the two ends of the coordinate.
        Otherwise, both ends are reflecting boundaries with zero flux.

    Returns
    -------
    PropagationResult
        Densities at each requested time and the equilibrium density implied
        by ``reduced_potential``.

    Notes
    -----
    When using propagator calculations, cite von Domaros, Thomas, and
    Prabhakar (2026), https://arxiv.org/abs/2610.29703.
    """
    return SmoluchowskiSolver(
        positions,
        diffusivity,
        reduced_potential=reduced_potential,
        periodic=periodic,
    ).propagate(
        times,
        initial_position=initial_position,
        initial_density=initial_density,
    )


def propagate_diffusivity_profiles(
    positions: ArrayLike,
    reduced_potential: ArrayLike,
    diffusivity_profiles: Mapping[str, ArrayLike],
    times: ArrayLike,
    *,
    initial_position: float | None = None,
    initial_density: ArrayLike | None = None,
    periodic: bool = False,
) -> dict[str, PropagationResult]:
    """Propagate several diffusivity profiles through the same free energy.

    The insertion order of ``diffusivity_profiles`` is retained in the output.
    ``reduced_potential`` is the dimensionless PMF, :math:`F/(k_B T)`, rather
    than an energy-valued PMF. Boundaries are reflecting unless ``periodic`` is
    true.
    """
    return {
        name: propagate_density(
            positions,
            diffusivity,
            reduced_potential,
            times,
            initial_position=initial_position,
            initial_density=initial_density,
            periodic=periodic,
        )
        for name, diffusivity in diffusivity_profiles.items()
    }


def _validate_grid(grid: ArrayLike) -> tuple[NDArray[np.float64], float]:
    values = np.asarray(grid, dtype=_FLOAT)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(
            "grid must be a one-dimensional array with at least two cell centers"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("grid must contain only finite values")
    differences = np.diff(values)
    if np.any(differences <= 0.0):
        raise ValueError("grid must be strictly increasing")
    spacing = float(differences[0])
    tolerance = 32.0 * np.finfo(_FLOAT).eps * max(1.0, abs(spacing))
    if not np.all(np.abs(differences - spacing) <= tolerance):
        raise ValueError("grid must be uniformly spaced")
    return values.copy(), spacing


def _validate_profile(
    values: ArrayLike, grid: NDArray[np.float64], name: str
) -> NDArray[np.float64]:
    profile = np.asarray(values, dtype=_FLOAT)
    if profile.ndim != 1 or profile.shape != grid.shape:
        raise ValueError(f"{name} must be a one-dimensional array matching grid")
    if not np.all(np.isfinite(profile)):
        raise ValueError(f"{name} must contain only finite values")
    return profile.copy()


def _validate_times(times: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(times, dtype=_FLOAT)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1:
        raise ValueError("times must be a scalar or one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("times must contain only finite non-negative values")
    return values.copy()


def _generator(
    potential: NDArray[np.float64],
    diffusivity: NDArray[np.float64],
    spacing: float,
    periodic: bool,
) -> csc_matrix:
    if periodic:
        left = np.arange(potential.size)
        right = (left + 1) % potential.size
    else:
        left = np.arange(potential.size - 1)
        right = left + 1
    delta_potential = potential[right] - potential[left]
    face_diffusivity = (
        2.0
        * diffusivity[left]
        * diffusivity[right]
        / (diffusivity[left] + diffusivity[right])
    )
    forward = face_diffusivity * _bernoulli(delta_potential) / spacing**2
    backward = face_diffusivity * _bernoulli(-delta_potential) / spacing**2
    rows = np.concatenate((right, left, left, right))
    columns = np.concatenate((left, left, right, right))
    values = np.concatenate((forward, -forward, backward, -backward))
    return coo_matrix(
        (values, (rows, columns)), shape=(potential.size, potential.size)
    ).tocsc()


def _bernoulli(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate ``x / (exp(x) - 1)`` accurately and without overflow."""
    result = np.empty_like(values)
    small = np.abs(values) <= 1e-5
    x = values[small]
    result[small] = 1.0 - x / 2.0 + x**2 / 12.0 - x**4 / 720.0
    positive = values > 1e-5
    # The limiting value is zero. Avoid ``large * 0`` for finite but very
    # large potential differences.
    result[positive] = 0.0
    moderate_positive = positive & (values <= 700.0)
    x = values[moderate_positive]
    result[moderate_positive] = x * np.exp(-x) / -np.expm1(-x)
    negative = values < -1e-5
    x = values[negative]
    result[negative] = x / np.expm1(x)
    return result
