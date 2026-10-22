# Analysis data

This directory contains reference profiles and cached results for the example
analyses. Raw molecular-coordinate trajectories are ignored by Git and must be
obtained separately to change the analysis settings. The committed Parquet
caches allow the notebooks to run without those trajectories.

## Citation guidance

When using the bundled data or cached RTA analyses in scientific work, cite the
[primary RTA paper](../README.md#citation-and-attribution). When using interval
selection, exit-time model comparisons, smoothed profiles, or propagator
calculations and validation, also cite
[Choosing Spatial Resolution in the Residence-Time Approach to Local Diffusivity](https://arxiv.org/abs/2610.29703).
For the SC PMF,
PACF/VACF profiles, and MD propagator comparisons, also cite
[Modeling Diffusion and Permeation Across the Stratum Corneum Lipid Barrier](https://doi.org/10.1021/acsestair.5c00412).
When reporting blocking-based SEMs from the caches, also cite
[Jonsson's blocking paper](https://doi.org/10.1103/PhysRevE.98.043304).
The [project citation guide](../README.md#citation-and-attribution) lists every
paper's full authors and publication details.

## Layout

```text
data/
├── cache/
│   ├── popc_water/
│   │   ├── diffusivity_estimates.parquet
│   │   ├── constant_nll_minima.parquet
│   │   ├── observed_exit_histograms.parquet
│   │   ├── terminal_censor_histograms.parquet
│   │   ├── metadata.json
│   │   ├── propagators.parquet
│   │   ├── propagator_metrics.parquet
│   │   └── propagator_metadata.json
│   └── sc_water/
├── popc_water/
│   ├── pmf_abf.txt
│   ├── diffusivity_pacf.txt
│   ├── diffusivity_vacf.txt
│   ├── diffusivity_rta.txt          # historical reference table
│   └── md/                         # unbiased-MD propagators, by lag time
├── sc_water/                       # same reference-data layout as POPC/water
└── raw/                           # ignored by Git; absent from a fresh clone
    ├── popc_water/trajectory.z
    └── sc_water/window_01.z ... window_07.z
```

Every raw `*.z` file is a headerless, one-column text file of finite Ångström
values ordered by simulation step. The step index and all other Colvars columns
are intentionally discarded.

`raw/popc_water` is the one POPC/water ABF window covering approximately -54 to
+54 Å. `trajectory.z` is a single continuous trajectory assembled from the 201
restart runs. At every checkpoint, the later run supplies the retained frame
and all overlapping future frames from the preceding run are discarded.

`raw/sc_water` contains seven ABF windows collectively spanning approximately
-45 to +45 Å. Each `window_*.z` file is one continuous trajectory. Repeated
checkpoint frames at joins between source runs have been removed.

Successive saved coordinate frames are separated by 2 fs (`0.002` ps).
Analyses use Å for position and ps for time, yielding diffusivities in Å²/ps.
Interval-width selection based on exit-time model mismatch is unchanged by a
global time-unit conversion.

Each analysis cache stores diffusivity, its blocking SEM, and model mismatch
at every analyzed position and interval width, the ranked constant-width NLL
minima, and sparse histograms of observed and terminally censored residence
lengths. These histograms are sufficient to recover the first-exit and censoring-time
distributions for the analyzed intervals. They do not retain temporal ordering,
so computing blocking SEMs requires the raw trajectories. Reading the saved
SEMs does not.

## Reference profiles and propagators

For each system, `pmf_abf.txt` contains position (Å) and the ABF potential of
mean force (kcal/mol). `diffusivity_pacf.txt` and `diffusivity_vacf.txt` contain
position (Å) and diffusivity (Å²/ps) determined from PACF and VACF analyses.
PACF and VACF files are optional independently; comparisons include those present.
Files named `md/propagator_XXXXps.txt` contain position (Å) and an unbiased-MD
probability density (Å⁻¹); the filename gives the lag time in ps.
The `diffusivity_rta.txt` files supplied for POPC/water and SC/water are
reference tables; the notebooks calculate their own RTA profiles from the
analysis caches.

The POPC/water reference data originate from the smaller POPC system used in
the RTA paper (DOI: [10.1021/acs.jctc.6c00645](https://doi.org/10.1021/acs.jctc.6c00645)),
not from PDDA's larger POPC ABF trajectory. That POPC system has a 76.34 Å
periodic cell at 310 K. Its PMF spans approximately -38 to +38 Å, while the
PACF and VACF profiles span approximately -32 to +32 Å. Do not conflate these
reference profiles or propagators with the caches generated from `raw/`.
