# PDDA: Position-Dependent Diffusivity Analysis

PDDA estimates how molecular diffusivity varies along a coordinate from
molecular-dynamics trajectories—for example, how water mobility changes across
a membrane. It helps you choose a spatial resolution supported by the data,
obtain a smooth diffusivity profile, and predict how a molecular distribution
evolves in a potential of mean force.

## Citation and attribution

**If you use PDDA in scientific work, please cite the primary RTA paper:**

Rinto Thomas, Praveen Ranganath Prabhakar, and Michael von Domaros.
*A Residence-Time Approach for Determining Position-Dependent Diffusivities
from Biased Molecular Simulations*. Journal of Chemical Theory and Computation
**22** (18), 9235–9246 (2026).
[DOI: 10.1021/acs.jctc.6c00645](https://doi.org/10.1021/acs.jctc.6c00645).

Additional citations depend on the methods and reference data used:

- **Methods beyond core residence-time diffusivity estimation:** when using
  interval-width selection, bulk-convergence diagnostics, exit-time model
  comparisons, profile smoothing, or propagator calculation and validation,
  also cite Michael von Domaros, Rinto Thomas, and Praveen Ranganath Prabhakar.
  *Choosing Spatial Resolution in the Residence-Time Approach to Local
  Diffusivity*. arXiv (2026).
  [arXiv:2610.29703](https://arxiv.org/abs/2610.29703).
  This paper covers the methodological extensions to the original RTA
  implementation. Using only the core residence-time-based diffusivity
  estimator requires the primary RTA citation above.
- **Stratum corneum reference comparisons:** when using the SC PMF, PACF/VACF
  diffusivity profiles, or MD propagators, also cite Rinto Thomas, Praveen
  Ranganath Prabhakar, Douglas J. Tobias, and Michael von Domaros.
  *Modeling Diffusion and Permeation Across the Stratum Corneum Lipid Barrier*.
  ACS ES&T Air **3** (2), 580–589 (2026).
  [DOI: 10.1021/acsestair.5c00412](https://doi.org/10.1021/acsestair.5c00412).
- **Blocking-based uncertainty estimates:** when reporting SEMs calculated with
  `uncertainty="blocking"`, also cite Marius Jonsson.
  *Standard error estimation by an automated blocking method*.
  Physical Review E **98**, 043304 (2018).
  [DOI: 10.1103/PhysRevE.98.043304](https://doi.org/10.1103/PhysRevE.98.043304).
  The RTA paper describes PDDA's application to correlated residence-time data.

For other externally supplied data or methods, cite their original sources as
appropriate. A repository citation may identify the software version used, but
please cite the scientific papers as well. [CITATION.cff](CITATION.cff) sets the
RTA paper as `preferred-citation` and includes the additional papers as references.

Michael von Domaros is the sole author of the PDDA library. Rinto Thomas and
Praveen Ranganath Prabhakar contribute to the underlying science and are
coauthors of the relevant research papers, as credited above.

## Scientific approach

The residence-time approach (RTA) estimates local diffusivity from the mean
first-exit time $\hat\tau$ from a spatial interval of width $L$:

$$
\hat D(z,L) = \frac{L^2}{12\hat\tau(z,L)}.
$$

The interval width sets the spatial scale of the estimate. PDDA compares
several widths using the observed exit-time distributions. Where a bulk region
is present, it also checks whether the estimated bulk diffusivity has converged
with interval width.

The approach assumes approximately diffusive motion within each interval, with
slowly varying diffusivity and weak effective drift on that scale. It is
intended for a single coordinate sampled at regular time intervals.

## Getting started

For the scientific library alone, install from the repository directory:

```bash
python -m pip install .
```

For the examples and development tools, install the Pixi environment:

```bash
pixi install
```

The release builds use Pixi 0.82.0, matching CI.

The library requires Python 3.10 or later, NumPy, and SciPy. Run Python in the
Pixi environment with `pixi run python`; this environment uses Python 3.12 or
later to support the notebook and plotting tools.

For complete worked examples, start with the [POPC/water notebook](notebooks/popc_water.ipynb)
or the [SC/water notebook](notebooks/sc_water.ipynb). They show bulk detection,
NLL ranking, diffusivity estimates, and comparison of predicted propagators
with molecular dynamics. See the [notebook guide](notebooks/README.md) for the
required data and helper functions.

The notebooks can run from the committed caches without downloading the raw
trajectories. Set their working directory to `notebooks/`; the raw trajectories
are needed only when recomputing analyses with different settings.

## Analyze a trajectory

Supply the sampled coordinate as an array, the positions at which to estimate
diffusivity, candidate interval widths, and the physical bounds of the sampled
region. `sampling_period` is the time between **saved samples**, not necessarily
the MD integration timestep.

```python
import numpy as np
from pdda import analyze_diffusivity

z = np.loadtxt("trajectory_z.dat")  # Å; one coordinate value per saved sample

analysis = analyze_diffusivity(
    z,
    positions=np.arange(-50.0, 51.0, 1.0),  # Å; interval centers
    interval_widths=np.arange(1.0, 20.5, 0.5),  # Å
    sampling_period=0.002,  # ps
    bounds=(-54.1, 54.1),  # Å; replace with your system's bounds
    uncertainty="blocking",  # SEMs for each position and interval width
)
print(analysis.summary())
```

Use one consistent unit system throughout. With coordinates in Å and times in
ps, diffusivities are in Å²/ps. PDDA does not convert units.

`analysis.diffusivities` and `analysis.diffusivity_sems` have one row per
interval width and one column per position. Blocking accounts for correlation
between overlapping exit-time observations. SEMs are `NaN` where the data do
not support an uncertainty estimate. Omit `uncertainty` to compute diffusivities
without SEMs. The example notebooks compute and cache SEMs by default.

For independent trajectories, pass a list of arrays rather than concatenating
them. For separately confined simulation windows, use `SimulationWindow` with
the bounds of each window. Only windows containing the complete residence
interval contribute. If the system is physically reflection-symmetric, add
`symmetry_center=0.0` (in coordinate units) to pool equivalent observations at
$z$ and $-z$.

## Choose an interval width

For systems with a bulk region:

```python
selection = analysis.select_profile()
print(selection.summary())
profile = selection.profile
```

Selection combines two scientific checks:

1. **Bulk detection:** for each interval width, check the outermost contiguous
   block of supported radial positions for a plateau. Two positions suffice
   when their trend and scatter are within the 5% tolerance. Across widths,
   find the highest nearly constant run of at least four consecutive widths,
   each supported by at least three spatial plateau points. Adjacent values
   within tolerance extend the compatible range; wider intervals may leave it.
2. **NLL ranking:** among local minima in that width range, choose the one with
   the lowest mean excess negative log likelihood (NLL).

The excess NLL measures disagreement between the RTA model and the observed
exit-time distribution; lower values indicate closer agreement. It is averaged
across position to compare interval widths. Internal gaps in the NLL map are
interpolated and terminal gaps use the nearest supported value for this ranking
only; diffusivity estimates are never filled. The chosen profile provides
`positions`, `diffusivity`, and `interval_width`.

If bulk convergence is not supported, or no NLL minimum lies in its width range,
selection returns no diffusivity estimates. Inspect
`selection.summary()` before proceeding. To inspect or choose other candidates,
use `analysis.constant_width_minima()`: it returns all local minima ranked by
increasing mean excess NLL, with the global minimum first.

## Smooth a profile and predict transport

Once a suitable profile has been selected:

```python
from pdda import smooth_profile

D = smooth_profile(
    profile.positions,  # Å
    profile.diffusivity,  # Å²/ps
    positive=True,
    smoothing_length=2.0,  # Å; larger values suppress broader variations
    plateau=True,
    extrapolate=True,
)
```

`D(z)` evaluates the smooth diffusivity. Positive fitting preserves positive
values. With `plateau=True`, a supported outer bulk region is joined with zero
slope and curvature and continued exactly constant. If no plateau is detected,
a warning is issued and no bulk value is imposed. `extrapolate=True` permits
constant continuation beyond the data, with zero endpoint slope; this boundary
condition is not evidence of a bulk plateau.

Use `propagate_density()` to predict time-dependent probability densities from
this diffusivity and a potential of mean force. Supply the **dimensionless PMF**,
$F/(k_B T)$, as `reduced_potential`, and evaluate both inputs on the same uniformly
spaced coordinate array. Boundaries are reflecting by default; use
`periodic=True` for a periodic coordinate. The notebooks demonstrate the full
calculation and comparison with MD.

## Interpreting the results

Check the interval-width dependence and sampling diagnostics before treating a
profile as a physical diffusivity. `NaN` estimates indicate unsupported
intervals, such as intervals crossing a wall or lacking usable exits. A terminal
censoring warning means that a substantial fraction of residence segments ended
before their exit was observed; those incomplete segments are excluded from
the estimate. Smoothing cannot establish the validity of an unsupported estimate.

Function docstrings describe the available parameters and returned quantities,
for example `help(analyze_diffusivity)` and `help(smooth_profile)`.

## Development and releases

Run `pixi run pytest -q`, `pixi run ruff check .`, and
`pixi run ruff format --check pdda tests notebooks/*.py` before submitting
changes. Tests cover the scientific estimators, numerical propagation, public
API, and notebook caches.

Dependency requirements specify minimum versions without speculative upper
bounds. The library needs NumPy 2.0 or later (`numpy.trapezoid`) and SciPy 1.13
or later (NumPy 2 support). `pixi.lock` records exact development versions for
both supported platforms. CI also tests the scientific library with Python
3.10, NumPy 2.0, and SciPy 1.13 to check the supported minimums.

Version 0.8 is recorded in `pyproject.toml` and `CITATION.cff`.
`pixi run build-wheel` creates the wheel and source distribution in `dist/`.
`pixi run build-conda` builds a local Conda package in `dist/conda/`.
`pixi run build-artifacts` waits for both builds and stages their results in
`out/pypi/` and `out/conda/noarch/`.

Pull requests run tests, style checks, and artifact builds. Pushes to `main`
also open a publication PR in the lab's package-hosting repository; this needs
the repository secret `PAGES_PUSH_TOKEN`. The notebooks and reference data are
available in the Git repository; the wheel contains the scientific library.
Generated `tmp/` and `figures/` directories are optional and ignored by Git.
