from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, ListedColormap
from matplotlib.figure import Figure

from utils import get_project_root

DPI = 300
WIDTH, HEIGHT = 20 / 2.54, 12 / 2.54
FIGURE_FACE_COLOR = "#212946"
AXES_FACE_COLOR = "#354066"
FIGURES_FOLDER = get_project_root() / "figures"


def configure_plotting():
    plt.rcParams.update(
        {
            "figure.autolayout": True,
            "figure.facecolor": FIGURE_FACE_COLOR,
            "figure.figsize": (WIDTH, HEIGHT),
            "figure.dpi": DPI,
            "axes.facecolor": AXES_FACE_COLOR,
            "axes.spines.left": False,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.spines.bottom": False,
            "axes.grid": True,
            "axes.grid.which": "both",
            "grid.linewidth": "0.3",
            "grid.alpha": 0.5,
            "text.color": "0.7",
            "axes.labelcolor": "0.7",
            "xtick.color": "0.7",
            "ytick.color": "0.7",
            "font.size": 10,
            "axes.titlesize": 14,
            "figure.titlesize": 14,
            "legend.fontsize": 9,
            # --- Settings for boxplot ---
            "boxplot.boxprops.color": "lightgray",
            "boxplot.whiskerprops.color": "lightgray",
            "boxplot.capprops.color": "lightgray",
            "boxplot.medianprops.color": "lightgray",
        }
    )


# Configure plotting when the module is imported
_has_configured_plotting = False


def _auto_configure_plotting_once():
    global _has_configured_plotting
    if not _has_configured_plotting:
        configure_plotting()
        _has_configured_plotting = True


_auto_configure_plotting_once()


def save_figure(fig, filename: str, extension: str = ".png", **kwargs) -> None:
    """Save a Matplotlib or Plotly figure to a file."""
    folder: Path = kwargs.pop("folder", FIGURES_FOLDER)
    dpi: int = kwargs.pop("dpi", DPI)
    folder.mkdir(parents=True, exist_ok=True)

    if not any(
        filename.lower().endswith(ext)
        for ext in [".png", ".jpg", ".jpeg", ".pdf", ".svg"]
    ):
        filename_with_ext = f"{filename}{extension}"
    else:
        filename_with_ext = filename

    if isinstance(fig, matplotlib.figure.Figure):
        fig.savefig(folder / filename_with_ext, dpi=dpi, **kwargs)
        plt.close(fig)
    elif isinstance(fig, go.Figure):
        scale_factor = dpi / 96  # Common baseline screen resolution is 96 DPI.
        fig.write_image(folder / filename_with_ext, scale=scale_factor, **kwargs)
    else:
        raise TypeError(f"Unsupported figure type: {type(fig)}")


def setup_multiplot(
    num_subplots: int,
    num_cols: int = 2,
    sharex: bool = True,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, list[Axes]]:
    num_rows = int(np.ceil(num_subplots / num_cols))

    if figsize is None:
        figsize = (
            (1.5 * WIDTH, num_rows * HEIGHT)
            if num_subplots > 1
            else (WIDTH, num_rows * HEIGHT)
        )

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=sharex)

    if num_subplots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    return fig, axes


def truncate_colormap(
    cmap: Colormap, minval: float = 0.25, maxval: float = 0.85, n: int = 256
) -> ListedColormap:
    """Truncate a colormap to only use values between minval and maxval."""
    new_cmap = ListedColormap(cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def autoformat_number(x: float) -> str:
    """
    Format large numbers without decimals, small numbers with 2 decimals.

    Example
    -------
    ```
    formatted_annot = np.vectorize(autoformat_number)(self.asset_cov_)
    sns.heatmap(df, annot=formatted_annot, fmt="",)
    ```
    instead of:
    ```
    sns.heatmap(df, annot=True, fmt=".2f",)
    ```
    """
    if np.abs(x) >= 100:
        return f"{x:.0f}"
    elif np.abs(x) >= 10:
        return f"{x:.1f}"
    else:
        return f"{x:.2f}"


def flatten_multiindex(multiindex: pd.MultiIndex | pd.Index) -> list:
    """
    Flatten a pandas MultiIndex into a list of concatenated strings for each row.
    For a MultiIndex with one level, the values are returned as they are.
    For a MultiIndex with multiple levels, the levels are concatenated with ", ".

    E.g. for a 2-level MultiIndex [('A', 'X'), ('B', 'Y')], return ['A, X', 'B, Y'].
    """
    if multiindex.nlevels == 1:
        return [str(idx[0]) for idx in multiindex]
    return [", ".join(map(str, idx)) for idx in multiindex]


def plot_heatmap(
    data: np.ndarray,
    title: str,
    cbar_label: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xticklabels: list[str] | None = None,
    yticklabels: list[str] | None = None,
    saveas_filename: str | None = None,
    **kwargs,
) -> None:
    fig, ax = plt.subplots(figsize=(WIDTH * 1.2, HEIGHT))
    sns.heatmap(
        data,
        annot=np.vectorize(autoformat_number)(data),
        fmt="",
        cmap=truncate_colormap(colormaps["mako"], minval=0.2, maxval=0.6),
        vmin=float(np.quantile(data, 0.1)),
        vmax=float(np.quantile(data, 0.9)),
        xticklabels=xticklabels or [str(i) for i in range(data.shape[1])],
        yticklabels=yticklabels or [str(i) for i in range(data.shape[0])],
        annot_kws={"size": 6, "color": "lightgray"},
        cbar_kws={"label": cbar_label},
        ax=ax,
    )
    ax.grid(False)
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    save_figure(fig, saveas_filename or "heatmap", **kwargs)
