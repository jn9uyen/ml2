from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.container import BarContainer, ErrorbarContainer
from matplotlib.figure import Figure

from .base import save_figure, setup_multiplot


def plot_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    min_count: int = 1,
    figsize: tuple[int, int] = (12, 8),
    bar_spacing: float = 0.15,
    order: list | None = None,
    hue_order: list | None = None,
    dodge_direction: str = "x",
    title: str = "",
    x_scale: Literal["linear", "log"] | None = None,
    y_scale: Literal["linear", "log"] | None = None,
    saveas_filename: str | None = None,
    **kwargs,
) -> tuple[Figure | None, Axes | None]:
    """
    Generate a bar plot with hue with an option to dodge bars/error bars along the
    x or y axis.

    Parameters
    ----------
    df : pd.DataFrame
        The data to plot.
    x : str
        The column name for the x-axis.
    y : str
        The column name for the y-axis.
    hue : str
        The column name for the hue (color) grouping.
    min_count : int
        The minimum count of observations required to plot a bar.
    figsize : tuple[int, int]
        The size of the figure.
    bar_spacing : float
        The spacing between bars.
    hue_order : list | None
        The order of hue categories.
    dodge_direction : str
        The axis ('x' or 'y') along which to dodge the bars.
        'y' for horizontal plots, 'x' for vertical plots.
    title : str
        The title of the plot.
    x_scale : str | None
        The scale for the x-axis (e.g., 'log', 'linear').
    y_scale : str | None
        The scale for the y-axis (e.g., 'log', 'linear').
    saveas_filename : str | None
        The filename to save the figure. If None, the figure is not saved.
    **kwargs
        Additional keyword arguments passed to `save_figure`.
    """
    # --- Data Preparation ---
    plot_df = df[df["count"] >= min_count].copy()
    if plot_df.empty:
        print("DataFrame is empty after filtering. Cannot generate plot.")
        return None, None

    if hue_order is None:
        # Create hue_order from the filtered plot_df.
        hue_order = sorted(plot_df[hue].dropna().unique().tolist())
    else:
        # Filter the provided hue_order against the values in the filtered plot_df.
        hue_values = set(plot_df[hue].dropna().unique())
        hue_order = [val for val in hue_order if val in hue_values]

    if not hue_order:
        print(
            "No data remains for the specified hue categories after filtering. "
            "Cannot generate plot."
        )
        return None, None

    print(f"Data shape: {df.shape}")
    print(f"Filtered data shape: {plot_df.shape}")
    print(f"Filtered data proportion: {plot_df.shape[0] / df.shape[0]:.2%}")

    # --- Create Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=plot_df,
        x=x,
        y=y,
        hue=hue,
        err_kws={"color": "darkgray"},
        order=order,
        hue_order=hue_order,
        palette=kwargs.pop("palette", None),
        ax=ax,
    )

    # --- Manually Shift Bars and Error Bars ---
    error_map = {
        c.get_label(): c for c in ax.containers if isinstance(c, ErrorbarContainer)
    }
    num_hue_levels = len(hue_order)

    for i, hue_name in enumerate(hue_order):
        shift = (i - (num_hue_levels - 1) / 2) * bar_spacing
        bar_container = ax.containers[i]

        if dodge_direction == "y":
            for patch in bar_container:
                patch.set_y(patch.get_y() + shift)
            if hue_name in error_map:
                plotline, caplines, barlinecols = error_map[hue_name].lines
                plotline.set_ydata(np.asarray(plotline.get_ydata()) + shift)
                for cap in caplines:
                    cap.set_ydata(np.asarray(cap.get_ydata()) + shift)
                for barline in barlinecols:
                    segs = barline.get_segments()
                    barline.set_segments([s + [0, shift] for s in segs])

        elif dodge_direction == "x":
            for patch in bar_container:
                patch.set_x(patch.get_x() + shift)
            if hue_name in error_map:
                plotline, caplines, barlinecols = error_map[hue_name].lines
                plotline.set_xdata(np.asarray(plotline.get_xdata()) + shift)
                for cap in caplines:
                    cap.set_xdata(np.asarray(cap.get_xdata()) + shift)
                for barline in barlinecols:
                    segs = barline.get_segments()
                    barline.set_segments([s + [shift, 0] for s in segs])

    # --- Add Text Labels ---
    textbox_props = dict(
        boxstyle="round,pad=0.3", edgecolor="none", facecolor="lightgray", alpha=0.7
    )
    for i in range(num_hue_levels):
        container = ax.containers[i]
        if isinstance(container, BarContainer):
            ax.bar_label(
                container,
                fmt="%.1f",
                padding=8,
                color="black",
                bbox=textbox_props,
                size="small",
            )

    ax.set_title(title)

    if x_scale:
        ax.set_xscale(x_scale)
    if y_scale:
        ax.set_yscale(y_scale)

    if saveas_filename:
        save_figure(fig, saveas_filename, **kwargs)

    return fig, ax


def multi_boxplot(
    df: pd.DataFrame,
    x: str,
    hue: str,
    hue_order: list | None = None,
    group_col: str = "is_modal_purchase_interval",
    min_count: int = 1,
    title: str = "",
    saveas_filename: str | None = None,
    **kwargs,
) -> tuple[Figure | None, list[Axes] | None]:
    """
    Create multiple boxplots (subplots) based on a grouping column.
    """
    plot_df = df[df["count"] >= min_count].copy()
    if plot_df.empty:
        print("DataFrame is empty after filtering. Cannot generate plot.")
        return None, None

    if hue_order is None:
        # Create hue_order from the filtered plot_df.
        hue_order = sorted(plot_df[hue].dropna().unique().tolist())
    else:
        # Filter the provided hue_order against the values in the filtered plot_df.
        hue_values = set(plot_df[hue].dropna().unique())
        hue_order = [val for val in hue_order if val in hue_values]

    if not hue_order:
        print(
            "No data remains for the specified hue categories after filtering. "
            "Cannot generate plot."
        )
        return None, None

    if group_col not in plot_df.columns:
        print(f"Column '{group_col}' not found in DataFrame.")
        return None, None

    group_values = sorted(plot_df[group_col].unique())

    fig, axes = setup_multiplot(
        num_subplots=len(group_values),
        num_cols=len(group_values),
        sharex=True,
        figsize=kwargs.pop("figsize", None),
    )

    for i, group_val in enumerate(group_values):
        cond = plot_df[group_col] == group_val
        sns.boxplot(
            data=plot_df[cond],
            x=x,
            hue=hue,
            hue_order=hue_order,
            showmeans=True,
            meanprops={"markersize": 10, "markeredgecolor": "k"},
            boxprops={"edgecolor": "darkgray"},
            whiskerprops={"color": "darkgray"},
            capprops={"color": "darkgray"},
            medianprops={"color": "darkgray"},
            flierprops={"markeredgecolor": "darkgray"},
            palette=kwargs.pop("palette", None),
            ax=axes[i],
            # legend=False,
        )
        axes[i].set_title(f"{group_col} = {group_val}")

    # Create single shared legend.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",  # Position the legend at the top center.
        bbox_to_anchor=(0.5, 1.22),  # Fine-tune the position (horizontal, vertical).
        ncol=3,  # Arrange in 3 columns
    )

    for i in range(len(group_values)):
        legend = axes[i].get_legend()
        if legend is not None:
            legend.remove()

    # Adjust layout to prevent the legend from overlapping the plot titles.
    plt.tight_layout(rect=(0, 0, 1, 0.95))  # rect=[left, bottom, right, top]
    fig.suptitle(title)

    if saveas_filename:
        save_figure(fig, saveas_filename, **kwargs)

    return fig, axes
