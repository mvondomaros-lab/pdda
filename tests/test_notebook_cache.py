import numpy as np
import pytest
from cache_helpers import (
    read_diffusivity_analysis_cache,
    write_diffusivity_analysis_cache,
)

from pdda import RTATerminalCensoringWarning, analyze_diffusivity
from pdda.rta import (
    DiffusivityAnalysis,
    FirstExitHistogramGrid,
)


def test_parquet_cache_round_trips_analysis_and_sparse_histograms(tmp_path) -> None:
    with pytest.warns(RTATerminalCensoringWarning, match="terminal censoring"):
        analysis = analyze_diffusivity(
            [-2.0, -0.2, 0.0, 0.2, 2.0, -0.1, 0.1],
            positions=[0.0],
            interval_widths=[1.0],
            sampling_period=0.1,
            bounds=(-2.0, 2.0),
            symmetry_center=0.0,
        )
    cache_path = tmp_path / "cache"
    write_diffusivity_analysis_cache(cache_path, analysis, metadata={"system": "test"})

    cached = read_diffusivity_analysis_cache(cache_path)

    assert cached.metadata["system"] == "test"
    assert cached.metadata["sampling_period"] == 0.1
    assert cached.analysis.sampling_period == 0.1
    assert cached.analysis.symmetry_center == 0.0
    assert len(cached.constant_width_minima) == 1
    assert np.array_equal(cached.constant_width_minima[0].positions, analysis.positions)
    assert cached.constant_width_minima[0].interval_width == 1.0
    assert cached.constant_width_minima[0].nll_rank == 0
    assert cached.constant_width_minima[0].coverage == 1.0
    assert (cache_path / "diffusivity_estimates.parquet").is_file()
    assert (cache_path / "constant_nll_minima.parquet").is_file()
    assert np.allclose(
        cached.analysis.diffusivities, analysis.diffusivities, equal_nan=True
    )
    assert np.allclose(
        cached.analysis.excess_nll,
        analysis.excess_nll,
        equal_nan=True,
    )
    assert cached.analysis.histograms is not None
    assert analysis.histograms is not None
    assert np.array_equal(
        cached.analysis.histograms.observed_counts,
        analysis.histograms.observed_counts,
    )
    assert np.array_equal(
        cached.analysis.histograms.censored_counts,
        analysis.histograms.censored_counts,
    )


def test_cache_stores_every_ranked_constant_width_minimum(tmp_path) -> None:
    empty = np.array([], dtype=np.int64)
    histograms = FirstExitHistogramGrid(*(empty.copy() for _ in range(8)))
    losses = np.array(
        [
            [0.3, 0.3, 0.3],
            [0.1, np.nan, 0.1],
            [0.2, 0.2, 0.2],
            [0.15, 0.15, 0.15],
            [0.25, 0.25, 0.25],
        ]
    )
    analysis = DiffusivityAnalysis(
        positions=np.arange(3.0),
        interval_widths=np.arange(1.0, 6.0),
        diffusivities=np.where(
            np.isfinite(losses), np.arange(1.0, 6.0)[:, None], np.nan
        ),
        excess_nll=losses,
        n_time_origins=np.ones(losses.shape, dtype=np.int64),
        first_step_exit_fractions=np.zeros(losses.shape),
        terminal_censored_fractions=np.zeros(losses.shape),
        sampling_period=1.0,
        histograms=histograms,
    )

    cache_path = tmp_path / "cache"
    write_diffusivity_analysis_cache(cache_path, analysis)
    cached = read_diffusivity_analysis_cache(cache_path)

    assert [item.interval_width for item in cached.constant_width_minima] == [
        2.0,
        4.0,
    ]
    assert [item.method for item in cached.constant_width_minima] == [
        "NLL_0",
        "NLL_1",
    ]
    selected = cached.constant_width_minima[0]
    assert selected.nll_rank == 0
    assert selected.mean_excess_nll == pytest.approx(0.1)
    assert selected.direct_nll_fraction == pytest.approx(2 / 3)
    assert selected.interpolated_nll_fraction == pytest.approx(1 / 3)
    assert selected.extrapolated_nll_fraction == 0.0
    assert selected.coverage == pytest.approx(2 / 3)
