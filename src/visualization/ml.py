from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .base import HEIGHT, WIDTH, save_figure


def _compute_relative_importance(df: pd.DataFrame, importance_col: str):
    """
    Compute relative and cumulative importance given a list of features and their
    importances. Return input df with two additional columns:
    ["relative_importance", "cumulative_importance"].
    """
    df["relative_importance"] = df[importance_col] / df[importance_col].max()
    df["cumulative_importance"] = df[importance_col].cumsum() / df[importance_col].sum()
    return df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    importance_type: Literal[
        "weight", "split", "gain", "cover", "total_gain", "total_cover"
    ],
    top_n: int = 20,
    relative: bool = False,
    title: str | None = None,
    saveas_filename: str | None = None,
    model_architecture: Literal["xgb", "lgbm"] = "lgbm",
    **kwargs,
) -> None:
    """
    Plot the top N most important features from a feature importance DataFrame.

    Parameters
    ----------
    importance_df : pd.DataFrame
        A DataFrame with columns ["feature", "importance"].
    importance_type : Literal["weight", "split", "gain", "cover", "total_gain",
        "total_cover"]
        The type of importance for XGBoost and LightGBM models.
    top_n : int, optional
        The number of top features to display, by default 20.
    relative : bool, optional
        If True, importance values are displayed as relative percentages.
    model_architecture : Literal["xgb", "lgbm"]
        The architecture of the model used to align `importance_type`:
        - "xgb": XGBoost: "gain": Average Gain, "weight": Split Count
        - "lgbm": LightGBM: "gain": Total Gain, "split": Split Count
    """
    IMPORTANCE_TYPE_MAP = {
        "xgb": {
            "weight": "Split Count",
            "split": "Split Count",  # Still a valid mapping; 'weight' is more common.
            "gain": "Average Gain",
            "cover": "Average Coverage",
            "total_gain": "Total Gain",
            "total_cover": "Total Coverage",
        },
        "lgbm": {
            "weight": "Split Count",  # 'weight' is used similarly to 'split' in lgbm.
            "split": "Split Count",
            "gain": "Total Gain",
            "cover": "Total Coverage",  # 'cover' in LGBM means 'total_cover'.
            "total_gain": "Total Gain",
            "total_cover": "Total Coverage",
        },
    }

    if model_architecture not in IMPORTANCE_TYPE_MAP:
        raise ValueError(f"Unsupported model_architecture: {model_architecture}")

    if importance_type not in IMPORTANCE_TYPE_MAP[model_architecture]:
        importance_label = importance_type  # Fallback to the raw importance_type.
    else:
        importance_label = IMPORTANCE_TYPE_MAP[model_architecture][importance_type]

    importance_df = importance_df.sort_values(
        by="importance", ascending=False
    ).reset_index(drop=True)
    importance_df["feature"] = (
        (importance_df.index + 1).astype(str) + ". " + importance_df["feature"]
    )

    if relative:
        importance_df = _compute_relative_importance(importance_df, "importance")

    importance_df = importance_df.head(top_n)

    # --- Plot ---
    dynamic_height = max(HEIGHT, top_n * 0.5)
    fig, ax1 = plt.subplots(figsize=(WIDTH, dynamic_height))
    display_col = "relative_importance" if relative else "importance"
    sns.barplot(
        x=display_col, y="feature", data=importance_df, palette="viridis", ax=ax1
    )

    if relative:
        ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        xlabel = "Relative Importance"
    else:
        xlabel = f"Importance ({importance_label})"

    for index, value in enumerate(importance_df[display_col]):
        value_label = f"{value:.1%}" if relative else f"{value:.1f}"
        ax1.text(
            value + 0.01 * max(importance_df[display_col]),
            index,
            value_label,  # + f" ({index + 1})",
            va="center",
        )

    if relative:
        ax2 = ax1.twiny()
        ax2.set_xticks([])
        ax2.plot(
            importance_df["cumulative_importance"],
            importance_df["feature"],
            marker="o",
            linestyle="-",
        )
        for index, value in enumerate(importance_df["cumulative_importance"]):
            ax2.text(
                value,
                index,
                f"{value:.1%}",
                va="bottom",
                ha="right",
                fontsize=10,
                bbox=dict(
                    facecolor="yellow",
                    edgecolor="none",
                    boxstyle="round,pad=0.2",
                    alpha=0.4,
                ),
            )

    ax1.set_xlabel(xlabel)
    title = title or f"Feature Importance ({importance_label}): Top {top_n}"
    ax1.set_title("Relative " + title if relative else title)

    save_figure(fig, saveas_filename or "feature_importance", **kwargs)


def _prep_multicolor_lineplot(
    x: np.ndarray | list,
    y: np.ndarray | list,
    c: np.ndarray | list,
    ax: Axes,
    linewidth: int = 3,
) -> tuple[LineCollection, Axes]:
    """
    Boilerplate for multicolor lineplot.
    https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
    """
    # Create a set of line segments so that we can color them individually
    # This creates the points as an N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments_ndarray = np.concatenate([points[:-1], points[1:]], axis=1)
    # segments_ndarray = np.stack((points[:-1], points[1:]), axis=1)
    segments = segments_ndarray.tolist()

    # Create a continuous norm to map from data points to colors.
    # norm = Normalize(c.min(), c.max())
    norm = Normalize(0, 1)
    lc = LineCollection(segments, cmap="coolwarm", norm=norm)
    # Set the values used for colormapping.
    lc.set_array(c)
    lc.set_linewidth(linewidth)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    return lc, cax


def plot_auc(
    x_vals: np.ndarray | list,
    y_vals: np.ndarray | list,
    thresholds: np.ndarray | list,
    title: str,
    xlabel: str,
    ylabel: str,
    metrics_dict: dict[str, float] | None,
    figsize: int | float | None = None,
    saveas_filename: str | None = None,
    **kwargs,
) -> Figure:
    """
    Plot the area under a curve (AUC) of x-values and y-values that are generated from
    a set of thresholds.

    For example, the ROC AUC curve has x-values = FPR and y-values = TPR.

    The values in `thresholds` determine the color of the line segments.
    """
    x_vals_arr = np.asarray(x_vals)
    y_vals_arr = np.asarray(y_vals)
    thresholds_arr = np.asarray(thresholds)

    if figsize is None:
        figsize = HEIGHT

    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.set_aspect("equal", adjustable="box")

    lc, cax = _prep_multicolor_lineplot(x_vals_arr, y_vals_arr, thresholds_arr, ax)
    line = ax.add_collection(lc)  # type: ignore

    if metrics_dict is not None:
        # Create a single, multi-line string for the legend.
        combined_label = "\n".join([f"{k}: {v:.3f}" for k, v in metrics_dict.items()])
        ax.plot([], [], " ", label=combined_label)

    ax.legend(frameon=True)
    ax.fill_between(x_vals_arr, y_vals_arr, color="skyblue", alpha=0.4)

    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(line, ax=ax, cax=cax, label="Threshold")

    save_figure(fig, saveas_filename or "auc.png", **kwargs)

    return fig


def plot_lift_chart(
    data: pd.DataFrame,
    bin_col: str,
    actual_rate_col: str,
    prediction_rate_col: str,
    average_rate: float,
    title: str = "Decile Lift Chart",
    figsize: tuple[int | float, int | float] = (WIDTH, HEIGHT),
    xlabel: str | None = None,
    ylabel: str | None = None,
    saveas_filename: str | None = None,
    **kwargs,
) -> Figure:
    """
    Plot a binned lift chart comparing the actual positive rate to the model's
    prediction rate.

    This function generates a line plot for a binned analysis (e.g., deciles), showing
    how the actual positive rate and the model's predicted rate vary across bins. It
    also includes a horizontal dashed line representing the overall average rate.

    Parameters
    ----------
    data : pd.DataFrame
        An aggregated DataFrame containing bins and their corresponding metrics.
        Expected columns: a bin column, an actual rate column, and a prediction rate
        column.
    bin_col : str
        The name of the bin column (e.g., 'decile').
    actual_rate_col : str
        The name of the actual positive rate column (e.g., 'churn_rate').
    prediction_rate_col : str
        The name of the model's prediction rate column (e.g., 'prediction').
    average_rate : float
        The overall average rate for the entire dataset.
    title : str, default 'Decile Lift Chart'
        The title of the plot.
    figsize : tuple[int, int], default (10, 6)
        The size of the figure.
    xlabel : str, optional
        The label for the x-axis. If None, defaults to the capitalized bin column name.
    ylabel : str, optional
        The label for the y-axis. If None, defaults to 'Rate / Prediction'.
    saveas_filename : str | None, optional
        The filename to save the plot. If None, the plot will not be saved.

    Returns
    -------
    Figure
        The matplotlib Figure object of the generated plot.
    """
    # Convert to long format to plot actual and predicted rates in a single plot.
    melted_data = data.melt(
        id_vars=bin_col,
        value_vars=[actual_rate_col, prediction_rate_col],
        var_name="metric",
        value_name="value",
    )

    fig, ax = plt.subplots(figsize=figsize)

    sns.lineplot(
        data=melted_data,
        x=bin_col,
        y="value",
        hue="metric",
        markers=True,
        marker="o",
        ax=ax,
    )

    # Add text labels for each marker.
    y_offset = 0.015
    for _, row in melted_data.iterrows():
        ax.text(
            x=row[bin_col],  # type: ignore reportArgumentType
            y=row["value"] + y_offset,  # type: ignore reportArgumentType
            s=f"{row['value']:.1%}",
            ha="center",
            va="bottom",
        )

    # Add horizontal dashed line for the average rate.
    ax.axhline(
        y=average_rate,
        color="lightgray",
        linestyle="--",
        label=f"Avg {actual_rate_col}: {average_rate:.2%}",
    )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel or bin_col.capitalize(), fontsize=12)
    ax.set_ylabel(ylabel or "Rate / Prediction", fontsize=12)
    ax.legend()
    plt.tight_layout()

    save_figure(fig, saveas_filename or "auc.png", **kwargs)

    return fig
