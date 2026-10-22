# Example analyses

Start with `popc_water.ipynb` for one continuous trajectory or `sc_water.ipynb`
for separately confined simulation windows. Both notebooks estimate diffusivity,
choose an interval width, and compare predicted propagators with MD.

## Citing example analyses

Please cite the [primary RTA paper](../README.md#citation-and-attribution) when
using these analyses in scientific work. The interval selection, bulk-convergence
diagnostics, profile smoothing, and propagator calculations and validation also
require citing
[Choosing Spatial Resolution in the Residence-Time Approach to Local Diffusivity](https://arxiv.org/abs/2610.29703).
Also cite the stratum corneum paper
when using SC reference profiles or MD propagators, and Jonsson's blocking paper
when reporting the blocking-based SEMs computed by these notebooks. The
[project citation guide](../README.md#citation-and-attribution) gives full author
lists, publication details, and DOIs; [CITATION.cff](../CITATION.cff) provides
machine-readable metadata.

## Running examples

Use the project's Pixi environment with `notebooks/` as the working directory.
Positions are in Å, times in ps, and diffusivities in Å²/ps.
The committed caches let you run both notebooks without the raw trajectories.
To execute them without changing the saved notebooks, run from the repository:

```bash
pixi run jupyter nbconvert --to notebook --execute notebooks/popc_water.ipynb --output-dir /tmp/pdda-notebooks
pixi run jupyter nbconvert --to notebook --execute notebooks/sc_water.ipynb --output-dir /tmp/pdda-notebooks
```

The notebooks warn that source freshness cannot be checked in that case. Restoring
the raw files is necessary to change the analysis grid, interval widths, or
uncertainty settings.

## Adapt a notebook to your system

1. Copy the closest example and set `DATA_PATH`, `CACHE_PATH`, and the raw
   trajectory paths. Give each system its own cache directory.
2. Set the sampling period between saved frames, temperature, box length,
   and physical bounds. For windowed simulations, specify each window's bounds.
3. Choose the position grid and interval widths. Set `symmetry_center=None`
   and `symmetric=False` if reflection symmetry is not justified. Set
   `PLOT_INTERVAL_WIDTHS` to the analyzed widths you want in the local-profile
   figure; every listed width must occur in `INTERVAL_WIDTHS`.
4. Run the cells in order. Inspect the sampling diagnostics, NLL minima, and
   bulk-convergence plot before using the selected profile.

Raw trajectories are one-column coordinate files. Independent trajectories and
windows stay separate; do not concatenate them. See the [data guide](../data/README.md)
for file layouts and the provenance of the bundled examples.

Propagator comparisons need `pmf_abf.txt` and `md/propagator_XXXXps.txt` in
`DATA_PATH`. The PMF uses kcal/mol; the helper converts it to dimensionless
units using the configured temperature. `diffusivity_pacf.txt` and
`diffusivity_vacf.txt` are independently optional. Only available reference
profiles appear in the plots, predictions, and comparison metrics.

## Results and uncertainty

`analysis.diffusivities` and `analysis.diffusivity_sems` have one row per
interval width and one column per position. The notebooks use
`uncertainty="blocking"` to calculate SEMs for every analyzed profile. Selected
profiles retain their corresponding SEMs. Unsupported estimates and SEMs
remain `NaN`.

`analysis.constant_width_minima()` lists candidate profiles in NLL order.
`analysis.select_profile()` chooses a candidate supported by bulk convergence.
If no candidate qualifies, inspect `selection.summary()` and the diagnostic
plots before proceeding. Smoothing cannot make an unsupported estimate valid.

## Caching

The helpers reuse results when the inputs and settings match. Plotting saved
results does not require raw trajectories. Changed settings or an incompatible
cache require raw data to recompute the analysis. Cache metadata records source
paths relative to this repository so a fresh clone can use the committed caches.

Each analysis cache contains diffusivities, SEMs, NLL rankings, and first-exit
histograms. SEMs are stored in `diffusivity_estimates.parquet`. Propagator
predictions and metrics have a separate cache, so changing a comparison does
not repeat the trajectory analysis. Parquet support comes from `pyarrow` in
the Pixi environment.

## Helper files

- `analysis_helpers.py`: load data, prepare comparisons, and reuse caches.
- `cache_helpers.py`: read and write the result tables.
- `comparison_types.py`: the prepared inputs and comparison results.
- `plotting_helpers.py`: build figures from those results.

These helpers belong to the notebooks. The installed scientific API and its
usage examples are described in the [project guide](../README.md).
