"""Consistent figure builders for the PDDA showcase notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from mvdlib.plots import FigSize
from mvdlib.style import BASE, H1, H2, INK, SEQ

from pdda import smooth_profile

plt.style.use("mvdlib.style.default")
_FIGURE_SIZE = FigSize()
_COLORS = (BASE, H1, H2, SEQ)
_NLL_LINESTYLES = ("-", "--", ":")
_FIGURE_WIDTH = "double"
_TITLE_SIZE = 10
_LABEL_SIZE = 9
_TICK_SIZE = 8
_LEGEND_SIZE = 8


def _style_figure(fig) -> None:
    """Keep type sizes consistent across single and multipanel figures."""
    for axis in fig.axes:
        axis.title.set_fontsize(_TITLE_SIZE)
        axis.xaxis.label.set_fontsize(_LABEL_SIZE)
        axis.yaxis.label.set_fontsize(_LABEL_SIZE)
        axis.tick_params(labelsize=_TICK_SIZE)


def _model_legend(fig, axis) -> None:
    """Place the shared model key above the data panels."""
    fig.legend(
        *axis.get_legend_handles_labels(),
        loc="outside upper center",
        ncols=3,
        frameon=False,
        fontsize=_LEGEND_SIZE,
        handlelength=1.6,
        columnspacing=1.1,
    )


def _model_styles(profiles) -> dict[str, tuple[str, str]]:
    styles = {}
    for name, profile in profiles.items():
        if profile.nll_rank is None:
            styles[name] = (BASE, "--" if name == "VACF" else "-")
        else:
            rank = profile.nll_rank
            styles[name] = (
                H2 if profile.selected else H1,
                _NLL_LINESTYLES[rank] if rank < 3 else "-",
            )
    return styles


def _model_label(name, profile) -> str:
    if profile.nll_rank is None:
        return name
    label = rf"RTA NLL$_{{{profile.nll_rank}}}$ ({profile.interval_width:g} Å)"
    return label + (" (selected)" if profile.selected else "")


def plot_nll_minima(
    analysis, nll_minima, selection, *, colors=None, figure_size=None
) -> None:
    """Show ranked NLL minima and highlight the bulk-compatible selection."""
    nll_minima = tuple(nll_minima)
    if not nll_minima:
        raise ValueError("at least one constant-width NLL minimum is required")
    selected_width = selection.interval_width
    colors = _COLORS if colors is None else colors
    figure_size = _FIGURE_SIZE if figure_size is None else figure_size
    _, nll, plateau, sequential = colors
    height = max(3.6, 0.31 * len(nll_minima) + 0.8)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=figure_size(_FIGURE_WIDTH, height=height),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.25, 1)},
    )
    image = axes[0].pcolormesh(
        analysis.positions,
        analysis.interval_widths,
        analysis.excess_nll,
        shading="nearest",
        cmap=sequential,
    )
    for rank, selection in enumerate(nll_minima):
        is_selected = selection.interval_width == selected_width
        if rank >= 3 and not is_selected:
            continue
        color = plateau if is_selected else nll
        axes[0].axhline(
            selection.interval_width,
            color=color,
            lw=1.8,
            ls=_NLL_LINESTYLES[rank] if rank < 3 else "-",
        )
    axes[0].set(
        title="Normalized excess NLL",
        xlabel=r"Interval center $z / \mathrm{\AA}$",
        ylabel=r"Interval width / $\mathrm{\AA}$",
    )
    fig.colorbar(
        image, ax=axes[0], fraction=0.045, pad=0.025, label=r"$\Delta\mathrm{NLL}$"
    )

    ranks = np.arange(len(nll_minima))
    costs = [selection.mean_excess_nll for selection in nll_minima]
    bar_colors = [
        plateau if item.interval_width == selected_width else nll for item in nll_minima
    ]
    axes[1].barh(ranks, costs, color=bar_colors, height=0.72)
    axes[1].set_yticks(
        ranks,
        [
            rf"NLL$_{{{rank}}}$  {item.interval_width:g} Å"
            for rank, item in enumerate(nll_minima)
        ],
    )
    axes[1].invert_yaxis()
    axes[1].set(
        title="Mean NLL at each minimum",
        xlabel=r"Mean $\Delta\mathrm{NLL}$",
        xlim=(0.0, 1.18 * max(costs)),
    )
    axes[1].grid(axis="x", alpha=0.2)
    axes[1].set_axisbelow(True)
    _style_figure(fig)
    plt.show()


def plot_bulk_convergence(analysis, bulk, *, colors=None, figure_size=None) -> None:
    """Show bulk diffusivity convergence across interval widths."""
    colors = _COLORS if colors is None else colors
    figure_size = _FIGURE_SIZE if figure_size is None else figure_size
    base, _, plateau, _ = colors
    fig, axis = plt.subplots(
        figsize=figure_size(_FIGURE_WIDTH, height=3.1), constrained_layout=True
    )
    widths = analysis.interval_widths
    detected = np.isfinite(bulk.bulk_diffusivities)
    last_bulk_width = (
        float(np.max(widths[detected])) if detected.any() else float("nan")
    )
    axis.plot(
        widths[detected],
        bulk.bulk_diffusivities[detected],
        color=base,
        marker="o",
        ms=3,
        label="Bulk estimates",
    )
    limit = bulk.limiting_bulk_diffusivity
    if np.isfinite(bulk.minimum_bulk_width):
        axis.fill_between(
            (bulk.minimum_bulk_width, bulk.maximum_bulk_width),
            limit * (1 - bulk.plateau_tolerance),
            limit * (1 + bulk.plateau_tolerance),
            color=plateau,
            alpha=0.12,
            zorder=0,
        )
    if np.isfinite(limit):
        axis.axhline(limit, color=plateau, lw=1.5, label="Bulk plateau level")
    if np.isfinite(bulk.minimum_bulk_width):
        axis.axvline(
            bulk.minimum_bulk_width,
            color=base,
            ls=":",
            label=f"Minimum bulk width: {bulk.minimum_bulk_width:g} Å",
        )
    axis.set(
        title="Bulk-convergence detection",
        xlabel=r"Interval width / $\mathrm{\AA}$",
        ylabel=r"$D / \mathrm{\AA}^2\,\mathrm{ps}^{-1}$",
        xlim=(widths.min(), widths.max()),
    )
    if np.isfinite(last_bulk_width) and last_bulk_width < widths.max():
        message = "No spatial bulk\nplateau detected"
        if not np.isfinite(bulk.minimum_bulk_width):
            message = "No converged bulk width;\nno spatial plateau detected"
        axis.annotate(
            message,
            xy=((last_bulk_width + widths.max()) / 2, 0.5),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="center",
            color=INK,
            fontsize=_TICK_SIZE,
        )
    elif not np.isfinite(bulk.minimum_bulk_width):
        axis.annotate(
            "Bulk convergence not detected",
            xy=(0.7, 0.88),
            xycoords="axes fraction",
            ha="center",
            color=INK,
            fontsize=_TICK_SIZE,
        )
    axis.grid(alpha=0.2)
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=_LEGEND_SIZE,
        borderaxespad=0,
    )
    _style_figure(fig)
    plt.show()


def plot_diffusivity_profiles(
    analysis,
    bulk,
    *,
    selected_widths,
    smoothing_length=2.0,
    colors=None,
    figure_size=None,
):
    """Plot explicitly chosen widths using the detected bulk regions."""
    from dataclasses import replace

    from matplotlib.lines import Line2D

    from pdda.profiles import (
        Plateau,
        _fit_constrained_spline,
        _lam_from_smoothing_length,
        _symmetrize_spline_data,
    )

    selected_widths = np.asarray(selected_widths, dtype=float)
    if selected_widths.ndim != 1 or selected_widths.size == 0:
        raise ValueError("selected_widths must be a nonempty one-dimensional sequence")
    shown_indices = []
    for width in selected_widths:
        matches = np.flatnonzero(analysis.interval_widths == width)
        if matches.size != 1:
            raise ValueError(f"selected width {width:g} is not in the analysis")
        shown_indices.append(int(matches[0]))
    if len(set(shown_indices)) != len(shown_indices):
        raise ValueError("selected_widths must be unique")

    figure_size = _FIGURE_SIZE if figure_size is None else figure_size
    sequential = _COLORS[3] if colors is None else colors[3]
    fig, axis = plt.subplots(
        figsize=figure_size(_FIGURE_WIDTH, height=3.5), constrained_layout=True
    )
    origin = bulk.origin
    positions = analysis.positions
    colors = sequential(np.linspace(0.0, 1.0, len(shown_indices)))
    legend_handles = []
    for color, index in zip(colors, shown_indices, strict=True):
        width = analysis.interval_widths[index]
        legend_handles.append(
            Line2D(
                [],
                [],
                color=color,
                marker="o",
                markersize=3,
                linewidth=1.0,
                label=f"{width:g} Å",
            )
        )
        values = analysis.diffusivities[index]
        valid = np.isfinite(positions) & (positions >= origin) & np.isfinite(values)
        errors = None
        if analysis.diffusivity_sems is not None:
            sems = analysis.diffusivity_sems[index]
            errors = (
                1.959963984540054
                * np.where(np.isfinite(sems) & (sems >= 0.0), sems, np.nan)[valid]
            )
        supported = np.isfinite(positions) & np.isfinite(values) & (values > 0.0)
        observed_positions = positions[supported]
        if (
            supported.sum() >= 5
            and observed_positions[0] < origin < observed_positions[-1]
            and np.unique(np.abs(observed_positions - origin)).size >= 5
        ):
            fitted = smooth_profile(
                positions,
                values,
                positive=True,
                symmetry_center=origin,
                smoothing_length=smoothing_length,
                plateau=False,
            )
            if np.isfinite(bulk.spatial_plateau_starts[index]):
                x, y = _symmetrize_spline_data(
                    observed_positions - origin, values[supported]
                )
                plateau_radius = bulk.spatial_plateau_starts[index]
                right_plateau = Plateau(
                    start=plateau_radius,
                    value=bulk.bulk_diffusivities[index],
                    n_points=int(bulk.spatial_plateau_point_counts[index]),
                    span=float(x[-1] - plateau_radius),
                    relative_trend=bulk.spatial_plateau_relative_trends[index],
                    relative_scatter=bulk.spatial_plateau_relative_scatters[index],
                )
                left_plateau = replace(right_plateau, start=-plateau_radius)
                lam = (
                    None
                    if smoothing_length is None
                    else _lam_from_smoothing_length(x, smoothing_length)
                )
                spline = _fit_constrained_spline(
                    x,
                    np.log(y),
                    (left_plateau, right_plateau),
                    lam,
                    log_space=True,
                    extrapolate=False,
                )
                fitted = replace(
                    fitted,
                    plateaus=(
                        replace(left_plateau, start=origin - plateau_radius),
                        replace(right_plateau, start=origin + plateau_radius),
                    ),
                    _spline=spline,
                )
            curve_positions = np.linspace(origin, fitted.domain[1], 400)
            axis.plot(
                curve_positions, fitted(curve_positions), color=color, linewidth=1.0
            )
        plotted_positions = positions[valid]
        in_plateau = np.zeros(plotted_positions.size, dtype=bool)
        if np.isfinite(bulk.spatial_plateau_starts[index]):
            in_plateau = (
                plotted_positions >= origin + bulk.spatial_plateau_starts[index]
            )
        for mask, facecolor in ((~in_plateau, "none"), (in_plateau, color)):
            if not mask.any():
                continue
            axis.errorbar(
                plotted_positions[mask],
                values[valid][mask],
                yerr=None if errors is None else errors[mask],
                fmt="o",
                linestyle="none",
                markersize=3,
                markerfacecolor=facecolor,
                markeredgecolor=color,
                markeredgewidth=0.8,
                elinewidth=0.7,
                capsize=1.5,
                color=color,
                zorder=3,
            )
    axis.set_title("RTA diffusivity profiles and bulk detection", y=1.25, pad=3)
    axis.set(
        xlabel=r"$z\ /\ \mathrm{\AA}$",
        ylabel=r"$\hat D\ /\ \mathrm{\AA}^2\,\mathrm{ps}^{-1}$",
    )
    width_legend = axis.legend(
        handles=legend_handles,
        ncols=min(8, len(legend_handles)),
        alignment="left",
        borderaxespad=0,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.18),
        frameon=False,
        fontsize=_LEGEND_SIZE,
        handlelength=1.5,
        columnspacing=1.2,
    )
    axis.add_artist(width_legend)
    region_handles = [
        Line2D(
            [],
            [],
            color="black",
            marker="o",
            markerfacecolor="none",
            linestyle="none",
            markersize=4,
            label="membrane",
        ),
        Line2D(
            [],
            [],
            color="black",
            marker="o",
            markerfacecolor="black",
            linestyle="none",
            markersize=4,
            label="bulk",
        ),
    ]
    axis.legend(
        handles=region_handles,
        alignment="left",
        borderaxespad=0,
        ncols=2,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.08),
        frameon=False,
        fontsize=_LEGEND_SIZE,
        handlelength=1.0,
        columnspacing=1.2,
    )
    axis.set_xlim(left=origin)
    axis.grid(alpha=0.2)
    _style_figure(fig)
    plt.show()
    return fig, axis


def plot_propagator_inputs(inputs, *, figure_size=None) -> None:
    """Display the exact PMF and diffusivity arrays used for propagation."""
    styles = _model_styles(inputs.profiles)
    figure_size = _FIGURE_SIZE if figure_size is None else figure_size
    fig, axes = plt.subplots(
        2,
        1,
        figsize=figure_size(_FIGURE_WIDTH, height=4.6),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].plot(inputs.positions, inputs.reduced_potential, color=INK)
    axes[0].set(ylabel=r"$\beta F(z)$", title="PMF")
    for name, profile in inputs.profiles.items():
        color, linestyle = styles[name]
        axes[1].plot(
            inputs.positions,
            profile.diffusivity,
            color=color,
            ls=linestyle,
            label=_model_label(name, profile),
        )
    axes[1].set(
        xlabel=r"Position, $z$ / $\mathrm{\AA}$",
        ylabel=r"$D(z)$ / $\mathrm{\AA}^2\,\mathrm{ps}^{-1}$",
        title="Diffusivity profiles",
    )
    _model_legend(fig, axes[1])
    for axis in axes:
        axis.grid(alpha=0.2)
    _style_figure(fig)
    plt.show()


def plot_propagators(comparison, *, figure_size=None) -> None:
    """Plot MD and model propagators for every cached lag time."""
    styles = _model_styles(comparison.inputs.profiles)
    figure_size = _FIGURE_SIZE if figure_size is None else figure_size
    n_columns = 3
    n_rows = int(np.ceil(len(comparison.inputs.lag_times) / n_columns))
    maximum_density = max(
        [np.max(values) for values in comparison.inputs.md_propagators.values()]
        + [
            np.max(values)
            for densities in comparison.model_propagators.values()
            for values in densities.values()
        ]
    )
    upper_density = 10.0 ** np.ceil(np.log10(1.1 * maximum_density))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=figure_size(_FIGURE_WIDTH, height=2.2 * n_rows + 0.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for axis, lag in zip(axes.flat, comparison.inputs.lag_times):
        axis.plot(
            comparison.inputs.positions,
            comparison.inputs.md_propagators[lag],
            color=INK,
            ls="-",
            label="MD",
        )
        for label, densities in comparison.model_propagators.items():
            color, linestyle = styles[label]
            axis.plot(
                comparison.inputs.positions,
                densities[lag],
                color=color,
                ls=linestyle,
                label=_model_label(label, comparison.inputs.profiles[label]),
            )
        axis.set(title=f"{lag} ps", yscale="log", ylim=(1e-7, upper_density))
    for axis in axes.flat[len(comparison.inputs.lag_times) :]:
        axis.set_visible(False)
    _model_legend(fig, axes.flat[0])
    fig.supxlabel(r"Position, $z$ / $\mathrm{\AA}$", fontsize=_LABEL_SIZE)
    fig.supylabel(r"Probability density / $\mathrm{\AA}^{-1}$", fontsize=_LABEL_SIZE)
    _style_figure(fig)
    plt.show()


def plot_density_metrics(comparison, *, figure_size=None) -> None:
    """Plot every density-comparison metric against lag time."""
    styles = _model_styles(comparison.inputs.profiles)
    figure_size = _FIGURE_SIZE if figure_size is None else figure_size
    labels = {
        "total_variation": "Total variation",
        "wasserstein_1_angstrom": r"Wasserstein-1 / $\mathrm{\AA}$",
        "hellinger": "Hellinger",
        "jensen_shannon": "Jensen-Shannon",
        "kl_reference_model": r"KL (MD $\Vert$ model)",
        "renyi_2": "Rényi-2",
    }
    fig, axes = plt.subplots(
        2,
        3,
        figsize=figure_size(_FIGURE_WIDTH, height=5.6),
        sharex=True,
        layout="constrained",
    )
    for axis, (metric, label) in zip(axes.flat, labels.items(), strict=True):
        for model in comparison.model_propagators:
            rows = [row for row in comparison.metric_rows if row["model"] == model]
            color, linestyle = styles[model]
            axis.plot(
                [row["lag_ps"] for row in rows],
                [max(row[metric], np.finfo(float).tiny) for row in rows],
                color=color,
                ls=linestyle,
                marker="o",
                ms=3,
                label=_model_label(model, comparison.inputs.profiles[model]),
            )
        axis.set(title=label, xscale="log", yscale="log")
        axis.grid(alpha=0.2)
    _model_legend(fig, axes.flat[0])
    fig.supxlabel(r"Lag time / $\mathrm{ps}$", fontsize=_LABEL_SIZE)
    fig.supylabel("Discrepancy from MD", fontsize=_LABEL_SIZE)
    _style_figure(fig)
    plt.show()
