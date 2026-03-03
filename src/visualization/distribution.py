import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

from .base import save_figure, setup_multiplot


def _calculate_rate_by_quantile(
    df: pd.DataFrame, metric_col: str, rate_col: str, n_quantiles: int = 20
) -> pd.DataFrame:
    """
    Calculate rates by metric quantiles.

    This function bins a continuous metric into quantiles, then calculates the
    mean rate, total volume (count), and metric range for each quantile.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    metric_col : str
        The column name representing the continuous metric (e.g., 'income').
    rate_col : str
        The column name representing the rate status (e.g., 'has_defaulted').
    n_quantiles : int, optional
        Number of quantiles to split the metric column into, by default 20.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns for quantile, rate, volume, and the
        metric's min/max value for that quantile.
    """
    df_copy = df.copy()
    df_copy[rate_col] = df_copy[rate_col].astype(float)

    quantiles = pd.qcut(
        df_copy[metric_col], q=n_quantiles, duplicates="drop", labels=False
    )
    rate_by_quantile = (
        df_copy.groupby(quantiles)
        .agg(
            rate=(rate_col, "mean"),
            volume=(rate_col, "size"),
            metric_min=(metric_col, "min"),
            metric_max=(metric_col, "max"),
        )
        .rename_axis("quantile")
        .reset_index()
    )

    return rate_by_quantile


def plot_target_vs_feature(
    df_base: pd.DataFrame,
    feature_names: list[str],
    target_col: str,
    num_cols: int | None = None,
    categorical_top_n: int | None = None,
    obs_series: pd.Series | None = None,
    format_as_percentage: bool = False,
    saveas_filename: str | None = None,
    **kwargs,
) -> None:
    """Plot target rate and volume against feature values.

    This function generates plots showing the relationship between specified
    features and a target variable's rate (e.g., default rate). It includes
    a secondary axis to display the volume of data for each point.

    It handles both numeric and categorical features:
    - For numeric features, it plots rate by quantile with a LOWESS trend line.
    - For categorical features, it plots rate by category, sorted by rate.

    Parameters
    ----------
    df_base : pd.DataFrame
        The base dataset containing features and the target variable.
    feature_names : list[str]
        A list of feature column names to plot.
    target_col : str
        The name of the target variable column (e.g., 'has_defaulted').
    num_cols : int | None, optional
        The number of columns to use for the subplots. If None, it will be determined
        automatically based on the number of features.
    categorical_top_n : int | None, optional
        Limits the display to the top N categories for categorical features, based on
        volume. By default None, which shows all.
    obs_series : pd.Series | None, optional
        A Series representing a specific observation. If provided, a vertical line
        indicates its value on the plot. By default None.
    """
    n_quantiles = kwargs.get("n_quantiles", 20)

    num_subplots = len(feature_names)
    if num_subplots == 1:
        num_cols = 1
    elif num_cols is None:
        num_cols = 2
    fig, axes = setup_multiplot(num_subplots, num_cols=num_cols, sharex=False)

    for i, (ax, col) in enumerate(zip(axes, feature_names)):
        # --- Setup Axes ---
        ax2 = ax.twinx()  # Create secondary y-axis for volume
        ax2.set_ylabel("Volume")

        # Dynamic label for target_col being rate (classification) or avg (regression).
        y_axis_label = (
            f"{target_col.replace('_', ' ').title()} Rate"
            if format_as_percentage
            else f"Avg {target_col.replace('_', ' ').title()}"
        )
        legend_label = (
            f"{target_col} Rate" if format_as_percentage else f"Avg {target_col}"
        )

        # --- Numeric Feature ---
        if pd.api.types.is_numeric_dtype(df_base[col]):
            rate_data = _calculate_rate_by_quantile(
                df_base, col, rate_col=target_col, n_quantiles=n_quantiles
            )

            # Volume Bars (Right Axis)
            bar_widths = rate_data["metric_max"] - rate_data["metric_min"]
            ax2.bar(
                x=rate_data["metric_min"],
                height=rate_data["volume"],
                width=bar_widths,
                align="edge",  # Align bars starting from the left edge (metric_min).
                color="lightgray",
                alpha=0.2,
                label="Volume",
            )
            # Rate Line (Left Axis)
            sns.lineplot(
                data=rate_data,
                x="metric_min",
                y="rate",
                ax=ax,
                label=legend_label,
                linewidth=1,
                marker="o",
            )
            # Trend Line (Left Axis)
            sns.regplot(
                data=rate_data,
                x="metric_min",
                y="rate",
                scatter=False,
                lowess=True,
                ci=None,
                color="orange",
                label="Trend",
                ax=ax,
            )

        # --- Categorical Feature ---
        else:
            rate_data = (
                df_base.groupby(col)
                .agg(
                    rate=(target_col, "mean"),
                    volume=(target_col, "size"),
                )
                .reset_index()
            )

            if categorical_top_n is not None:
                rate_data = rate_data.nlargest(categorical_top_n, "volume")

            rate_data = rate_data.sort_values("rate", ascending=False)

            # Volume Bars (Right Axis)
            sns.barplot(
                data=rate_data,
                x=col,
                y="volume",
                ax=ax2,
                color="lightgray",
                alpha=0.2,
                label="Volume",
            )
            # Rate Line (Left Axis)
            sns.lineplot(
                data=rate_data,
                x=col,
                y="rate",
                ax=ax,
                label=legend_label,
                marker="o",
            )
            ax.set_xticks(
                ax.get_xticks(), ax.get_xticklabels(), rotation=30, ha="right"
            )

        ax.set_ylabel(y_axis_label)
        if format_as_percentage:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        # --- Add Indicator for the specific data point (if provided) ---
        if obs_series is not None:
            obs_value = obs_series[col]
            label = f"Observation: {obs_value}"
            x_pos = None

            if pd.api.types.is_numeric_dtype(df_base[col]):
                label = f"Observation: {obs_value:,.0f}"
                x_pos = obs_value
            else:  # Categorical
                categories = rate_data[col].tolist()
                if obs_value in categories:
                    x_pos = categories.index(obs_value)
                else:
                    # Add an invisible plot element for the legend only.
                    ax.plot([], [], "x", label=label)

            if x_pos is not None:
                ax.axvline(
                    x=x_pos,
                    color="black",
                    linestyle="-.",
                    linewidth=1.5,
                    alpha=1,
                    label=label,
                )

        ax.set_title(f"{i + 1}. {col.replace('_', ' ').title()}")
        ax2.grid(False)  # Turn off grid for the volume axis.

        # Combine legends from both axes
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2)
        legend_ax2 = ax2.get_legend()
        if legend_ax2:
            legend_ax2.remove()

    # --- High-Level Overview ---
    if obs_series is not None:
        fig.suptitle(f"Observation: {obs_series.name}")

    fig.tight_layout()
    if obs_series is not None:
        # Add extra space for the suptitle if it exists.
        fig.subplots_adjust(top=0.95)

    # Hide unused subplots.
    for j in range(num_subplots, len(axes)):
        axes[j].set_visible(False)

    save_figure(fig, saveas_filename or "target_vs_feature.png", **kwargs)


def plot_trend_and_volume(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    ax: Axes | None = None,
    order: list[str] | None = None,
    hue_order: list[str] | None = None,
    point_palette: str | None = None,  # "viridis",
    point_alpha: float = 1.0,
    count_color: str = "lightgray",
    count_alpha: float = 0.4,
    dodge: float = 0.3,
    saveas_filename: str | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    """
    Generate a dual-axis plot with a point plot and a count plot.

    This function creates a visualization to show a primary metric's trend
    (e.g., mean) on the left y-axis and the corresponding data volume (count)
    on the right y-axis. It is useful for assessing if a trend is supported
    by a sufficient amount of data.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to plot.
    x : str
        The name of the column to use for the x-axis categories.
    y : str
        The name of the column for the primary y-axis (point plot).
    hue : str | None, optional
        The name of the column for hue-based segmentation. Defaults to None.
    ax : plt.Axes | None, optional
        A matplotlib axes object to plot on. If None, a new figure and axes
        are created. Defaults to None.
    order : list[str] | None, optional
        The specific order for the x-axis categories. If None, the order is
        inferred from the data. Defaults to None.
    hue_order : list[str] | None, optional
        The specific order for the hue categories. If None, the order is
        inferred from the data. Defaults to None.
    point_palette : str, default 'viridis'
        The color palette for the point plot when 'hue' is used.
    point_alpha : float, default 1.0
        The alpha transparency for the point plot markers.
    count_color : str, default 'lightgray'
        The color for the count plot bars on the secondary axis.
    count_alpha : float, default 0.6
        The alpha transparency for the count plot bars.
    dodge : float, default 0.3
        The amount to dodge the point plot markers when 'hue' is used.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object.
    ax : matplotlib.axes.Axes
        The primary matplotlib Axes object.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create a sample DataFrame
    >>> np.random.seed(42)
    >>> data = {
    ...     'img_sex': np.random.choice(['Male', 'Female'], size=300),
    ...     'cost_per_booking': np.random.uniform(70, 250, size=300),
    ...     'device_type': np.random.choice(['Mobile', 'Desktop'], size=300)
    ... }
    >>> df_sample = pd.DataFrame(data)
    >>> # Generate the plot
    >>> fig, ax = plot_trend_and_volume(
    ...     df=df_sample,
    ...     x='img_sex',
    ...     y='cost_per_booking',
    ...     hue='device_type'
    ... )
    >>> # Save the figure
    >>> fig.savefig("cost_per_booking_trend_volume.png", dpi=300, bbox_inches='tight')
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
        if fig is None:
            raise ValueError(
                "The provided 'ax' object is not associated with a Figure. "
                "Please ensure the Axes has been added to a figure before passing."
            )

    ax2 = ax.twinx()

    if order is None:
        order = df[x].value_counts().index.tolist()  # frequency descending order.

    # Plot 1: Count plot on the secondary axis (ax2)
    sns.countplot(
        data=df,
        x=x,
        ax=ax2,
        color=count_color,
        alpha=count_alpha,
        order=order,
    )
    ax2.set_ylabel("Volume (Count)")
    ax2.grid(False)

    # Plot 2: Point plot on the primary axis (ax)
    if hue:
        # If hue is used, pass palette and dodge.
        if hue_order is None:
            hue_order = sorted(df[hue].dropna().unique())

        df[hue] = pd.Categorical(df[hue], categories=hue_order, ordered=True)

        sns.pointplot(
            data=df,
            x=x,
            y=y,
            ax=ax,
            order=order,
            hue=hue,
            palette=point_palette,
            alpha=point_alpha,
            dodge=dodge,  # type: ignore
            markers="o",
            linestyles="-",
        )
    else:
        # If no hue, pass a single color instead of a palette.
        sns.pointplot(
            data=df,
            x=x,
            y=y,
            ax=ax,
            order=order,
            # color=color,
            markers="o",
            linestyles="-",
            alpha=point_alpha,
        )
    ax.set_ylabel(f"Average {y.replace('_', ' ').title()}")
    ax.set_xlabel(x.replace("_", " ").title())

    # --- Formatting ---
    title = f"Average {y.replace('_', ' ').title()} and Volume by {x.replace('_', ' ').title()}"
    if hue:
        title += f" and {hue.replace('_', ' ').title()}"
    ax.set_title(title, pad=20)

    if hue:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            title=hue.replace("_", " ").title(),
            bbox_to_anchor=(1.18, 1),
            loc="upper left",
        )
    # Ensure the right-side y-axis labels don't overlap with the legend
    # fig.tight_layout()

    if saveas_filename:
        save_figure(fig, saveas_filename)

    return fig, ax
