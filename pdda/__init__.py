"""Analyze position-dependent diffusion along a molecular coordinate.

PDDA estimates local diffusivities from first-exit times, compares estimates
obtained with different interval widths, fits smooth physical profiles, and
propagates densities with the one-dimensional Smoluchowski equation.

For methods beyond core residence-time diffusivity estimation, including
interval selection and propagator calculation, also cite von Domaros, Thomas,
and Prabhakar (2026), https://arxiv.org/abs/2610.29703. See README.md and
CITATION.cff for the original RTA paper and full citation guidance.
"""

from .profiles import (
    DensityMetrics,
    PlateauNotFoundWarning,
    SmoothingSpline,
    compare_densities,
    smooth_profile,
)
from .propagators import (
    PropagationResult,
    periodic_grid,
    propagate_density,
)
from .rta import (
    DiffusivityAnalysis,
    DiffusivityEstimate,
    DiffusivityProfile,
    ProfileSelection,
    RTATerminalCensoringWarning,
    SimulationWindow,
    analyze_diffusivity,
    estimate_diffusivity,
)

__all__ = [
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
]
