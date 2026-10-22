"""Prepared scientific inputs and results shared by notebook helpers."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ComparisonProfile:
    """One fitted diffusivity and its optional NLL-selection information."""

    diffusivity: NDArray[np.float64]
    nll_rank: int | None = None
    interval_width: float | None = None
    selected: bool = False


@dataclass(frozen=True)
class PropagatorInputs:
    """Actual propagation inputs: Å, ps, Å²/ps, and dimensionless PMF."""

    positions: NDArray[np.float64]
    spacing: float
    reduced_potential: NDArray[np.float64]
    profiles: dict[str, ComparisonProfile]
    md_propagators: dict[int, NDArray[np.float64]]

    @property
    def lag_times(self) -> tuple[int, ...]:
        return tuple(sorted(self.md_propagators))


@dataclass(frozen=True)
class PropagatorComparison:
    """Prepared inputs, predicted densities, and discrepancies from MD."""

    inputs: PropagatorInputs
    model_propagators: dict[str, dict[int, NDArray[np.float64]]]
    metric_rows: tuple[dict[str, float | int | str], ...]
