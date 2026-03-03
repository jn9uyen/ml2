from typing import Union
import os
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.calibration import CalibrationDisplay
import shap
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.axes import Axes
import seaborn as sns

from helper import logging_config

logger = logging_config.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))
folder = f"{current_dir}/figures/"

if not os.path.exists(folder):
    os.makedirs(folder)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context(
    "notebook",
    font_scale=1.0,
    rc={
        "axes.titlesize": 18,
        "axes.labelsize": 13,
        "axes.ticksize": 13,
        "axes.textsize": 13,
        "font.size": 11,
        "legend.fontsize": 13,
    },
)
# sns.set_style('darkgrid', rc={'legend.frameon': True})
plt.rcParams["figure.figsize"] = (9, 5)


def plot_default_by_time(
    pdf: pd.DataFrame,
    date_col: str,
    default_col: str = "has_defaulted",
    time_period: str = "year-quarter",
    figsize: tuple = None,
) -> None:
    """
    Calculate default rate by time period and plot it.
    """
    match time_period:
        case "year-quarter":
            pdf["time_period"] = pdf[date_col].dt.strftime("%Y-Q") + (
                pdf[date_col].dt.quarter
            ).astype(str)
        case "year-month":
            pdf["time_period"] = pdf[date_col].dt.strftime("%Y-%m")
        case _:
            raise ValueError(f"Invalid time_period: {time_period}")

    group = pdf.groupby("time_period")[default_col]

    default_df = pd.merge(
        group.sum().rename("defaulted_sum").reset_index(),
        group.size().rename("vol").reset_index(),
        on="time_period",
    )
    default_df["default_rate"] = default_df["defaulted_sum"] / default_df["vol"]

    f, ax = plt.subplots(figsize=figsize)
    sns.lineplot(data=default_df, x="time_period", y="default_rate", ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(f"Default Rate by {time_period}")


def plot_feature_importances(
    feat_imp: pd.DataFrame,
    importance_type: str,
    mdl_name: str,
    max_feat_display: int = 15,
    is_relative_feat_imp: bool = False,
    saveas_filename: str = None,
):

    title = f"Feature Importance ({importance_type}) [{mdl_name}]"
    if is_relative_feat_imp:
        title = "Relative " + title

    df = (
        feat_imp.iloc[:max_feat_display].round(2)
        if max_feat_display is not None
        else feat_imp
    )

    fig, ax = plt.subplots()
    sns.barplot(df, y="feature", x="importance", color="steelblue", ax=ax)
    ax.bar_label(ax.containers[0], fontsize=10)
    ax.set_title(title)

    saveas_filename = saveas_filename or f"feat_imp_{mdl_name}"
    fig.savefig(f"{folder}/{saveas_filename}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_shap_beeswarm(
    shap_values, mdl_name: str, max_display: int = 15, saveas_filename: str = None
):
    saveas_filename = saveas_filename or f"shap_beeswarm_{mdl_name}"
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.gcf().savefig(f"{folder}/{saveas_filename}.png", bbox_inches="tight", dpi=300)
    plt.close()


def _prep_multicolor_lineplot(
    x: list, y: list, c: list, ax: plt.Axes, linewidth: int = 3
):
    """
    Boilerplate for multicolor lineplot
    """
    # Create a set of line segments so that we can color them individually
    # This creates the points as an N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    # norm = plt.Normalize(c.min(), c.max())
    norm = plt.Normalize(0, 1)
    lc = LineCollection(segments, cmap="coolwarm", norm=norm)
    # Set the values used for colormapping
    lc.set_array(c)
    lc.set_linewidth(linewidth)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    return lc, cax


def plot_roc_curve(
    fpr: list,
    tpr: list,
    thresholds: list,
    roc_auc: float,
    gini: float,
    figsize: int = 7,
    title: str = "ROC curve",
    saveas_filename: str = "roc_curve",
) -> None:
    """
    ROC curve

    Multicoloured line: based on `threshold`
    https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
    """
    # random.seed(123)
    # i = sorted(np.random.randint(len(fpr), size=5000))

    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.set_aspect("equal", adjustable="box")

    lc, cax = _prep_multicolor_lineplot(fpr, tpr, thresholds, ax)
    line = ax.add_collection(lc)
    line.set_label(f"ROC curve\nAUC: {roc_auc:.2f}\nGini: {gini:.2f}")
    ax.legend(loc="lower right", frameon=True)
    ax.fill_between(fpr, tpr, color="skyblue", alpha=0.4)

    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    fig.colorbar(line, ax=ax, cax=cax, label="Threshold")
    fig.savefig(f"{folder}/{saveas_filename}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_precision_recall_curve(
    precision: list,
    recall: list,
    thresholds: list,
    ap: float,
    ap0: float,
    map: float,
    figsize: int = 7,
    title: str = "Precision-Recall",
    saveas_filename: str = "pr_curve",
):
    """
    Precision-recall curve
    """
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.set_aspect("equal", adjustable="box")

    lc, cax = _prep_multicolor_lineplot(recall, precision, thresholds, ax)
    line = ax.add_collection(lc)
    line.set_label(f"AP: {ap:.2f}\nAP (class 0): {ap0:.2f}\nMAP: {map:.2f}")
    ax.legend(loc="upper right", frameon=True)
    ax.fill_between(recall, precision, color="skyblue", alpha=0.4)

    # ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("Recall (True Positive Rate)")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    fig.colorbar(line, ax=ax, cax=cax, label="Threshold")
    fig.savefig(f"{folder}/{saveas_filename}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_binary_confusion_matrix(
    y: list,
    y_prob: list,
    thres: float = 0.5,
    class_names: list = ["Non-defaulted", "Defaulted"],
    saveas_filename: str = None,
    figsize: tuple = (5, 10),
) -> None:
    """
    Plot confusion maxtrix for binary classification
    """
    y_pred = np.where(y_prob >= thres, 1, 0)

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    titles_options = [
        (f"Confusion matrix\nthreshold = {thres}", None),
        ("Normalised confusion matrix (by row)", "true"),
    ]
    for i, (title, normalize) in enumerate(titles_options):
        disp = metrics.ConfusionMatrixDisplay.from_predictions(
            y,
            y_pred,
            display_labels=class_names,
            cmap=plt.cm.Blues,
            normalize=normalize,
            ax=ax[i],
        )
        disp.ax_.set_title(title)
        disp.ax_.grid(None)

    saveas_filename = saveas_filename or f"confusion_matrix_thres_{thres}"
    fig.savefig(f"{folder}/{saveas_filename}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_calibration_curve(
    y: Union[list, np.array, pd.Series],
    y_prob: Union[list, np.array, pd.Series],
    n_bins: int = 10,
    strategy: str = "uniform",
    model_name: str = "classifier",
    title: str = "Calibration Curve",
    saveas_filename: str = "calibration",
):
    disp = CalibrationDisplay.from_predictions(
        y, y_prob, n_bins=n_bins, strategy=strategy, name=model_name
    )
    disp.ax_.set_title(title)
    disp.figure_.savefig(
        f"{folder}/{saveas_filename}.png", bbox_inches="tight", dpi=300
    )
    plt.close(disp.figure_)


def plot_prediction_distribution(
    df: pd.DataFrame,
    class_mapping: dict = None,
    figsize: tuple = (9, 5),
    title: str = "Prediction Distribution",
    saveas_filename: str = "pred_distribution",
) -> None:
    """
    Plot prediction distribution

    Parameters
    ----------
    df : pd.DataFrame
        Predictions with columns [y_pred, y]; y optional
    class_mapping: dict
        Mapping of class labels; e.g. {0: "Non-default", 1: "Default"}
    """
    hue_order = [0, 1]
    if class_mapping is not None:
        df["y"] = df["y"].map(class_mapping)
        hue_order = [class_mapping.get(key, None) for key in hue_order]

    fig, ax = plt.subplots(figsize=figsize)
    hue = "y" if "y" in df.columns else None
    sns.histplot(
        data=df, x="y_pred", hue=hue, hue_order=hue_order, kde=True, fill=True, ax=ax
    )
    ax.set_title(f"{title}")
    ax.set_xlabel("Prediction")
    fig.savefig(f"{folder}/{saveas_filename}.png", bbox_inches="tight", dpi=300)
    plt.close()


def plot_ranked_probabilities(
    df: pd.DataFrame,
    figsize: tuple = (9, 5),
    sample_size: int = 5000,
    title: str = "Ranked probabilities",
    saveas_filename: str = "ranked_probabilities",
) -> None:
    """
    Plot ranked probabilities

    Parameters
    ----------
    df : pd.DataFrame
        ranked probabilities with columns [y, y_prob, rank_pct, percentile, decile]
    """
    df = df.sample(sample_size) if len(df) > sample_size else df
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=df,
        x="rank_pct",
        y="y_prob",
        hue="y",
        hue_order=[0, 1],
        style="y",
        size="y",
        sizes=(50, 150),
        size_order=[1, 0],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Ranked Percentage of Dataset")
    ax.set_ylabel("Predicted Probability")

    fig.savefig(f"{folder}/{saveas_filename}.png", bbox_inches="tight", dpi=300)
    plt.close()


def _add_text_label(
    ax: Axes,
    x: Union[int, float],
    y: Union[int, float],
    label: str,
    offset: float = 0.02,
    bbox: dict = dict(
        facecolor="yellow",
        alpha=0.4,
        edgecolor="none",
        boxstyle="round,pad=0.3",
    ),
    **kwargs,
) -> None:
    ax.text(
        x,
        y + np.sign(y) * offset,
        f"{label}",
        bbox=bbox,
        **kwargs,
    )


def plot_lift_curve(
    df: pd.DataFrame,
    baseline_val: float = 1,
    show_rate: bool = False,
    figsize: tuple = (9, 5),
    title: str = "Lift Chart",
    saveas_filename: str = "lift_chart",
) -> None:
    """
    Plot lift chart. If show_rate = True, plot the rate and cumulative rate; otherwise
    plot lift and gain.

    Parameters
    ----------
    df : pd.DataFrame
        lift by percentile or decile including columns:
        [decile/percentile, rate, lift, cumulative_gain]
    show_rate : bool
        show rate if True, otherwise show lift
    baseline_val : float
        baseline rate / lift horizontal line
    """
    y = "rate" if show_rate else "lift"

    fig, ax = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    if "decile" in df.columns:
        x = "decile"
        sns.barplot(data=df, x=x, y=y, ax=ax[0])
    else:
        x = "percentile"
        sns.lineplot(data=df, x=x, y=y, ax=ax[0])
        ax[0].fill_between(df[x], df[y], color="skyblue", alpha=0.4)

    # Cumulative chart
    if show_rate:
        y = "cumulative_rate"
        sns.barplot(data=df, x=x, y=y, ax=ax[1])

        # Use index because barplot treats x as categorical, so locations are 0,1,etc...
        df["index"] = df.index
        x = "index"

        for i in range(df.shape[0]):
            _add_text_label(ax[1], df[x][i], df[y][i], f"{df[y][i]:.2f}", ha="center")

        ax2 = ax[1].twinx()
        y2 = "cumulative_rate_pct_diff"
        df[y2] = df[y2] * 100
        sns.lineplot(data=df, x=x, y=y2, ax=ax2, color="orange", marker="o")
        for i in range(df.shape[0]):
            _add_text_label(
                ax2,
                df[x][i],
                df[y2][i],
                f"{df[y2][i]:.0f}",
                offset=0.1,
                ha="center",
                bbox=dict(
                    facecolor="orange",
                    alpha=0.5,
                    edgecolor="black",
                    boxstyle="round,pad=0.3",
                ),
            )
        ax2.set_ylabel("percentage difference from base rate (%)")
        ax[1].set_title("Cumulative Rates")
    else:
        y = "cumulative_gain"
        sns.lineplot(data=df, x=x, y=y, ax=ax[1])
        ax[1].fill_between(df[x], df[y], color="skyblue", alpha=0.4)
        ax[1].set_title("Cumulative Gain")

    baseline_val = baseline_val if show_rate else 1
    ax[0].axhline(y=baseline_val, color="orange", linestyle="--")
    _add_text_label(
        ax[0],
        int(df[x].mean()),
        baseline_val,
        f"{baseline_val:.4f}",
        ha="left",
        va="bottom",
    )
    ax[0].set_title(title)

    fig.savefig(f"{folder}/{saveas_filename}.png", bbox_inches="tight", dpi=300)
    plt.close()


def plot_feature_distribution(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    title: str = None,
    saveas_filename: str = None,
    unique_values_max: int = 5,
    figsize: tuple = (9, 5),
    show_proportion: bool = True,
    ax=None,
) -> None:
    """
    Plot feature distribution using sns.countplot if the feature has fewer than 5 unique
    values, otherwise use sns.kdeplot.
    """
    unique_values = df[feature_col].nunique()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if unique_values <= unique_values_max:
        if show_proportion:
            # Calculate proportions within each group
            count_df = (
                df.groupby([feature_col, target_col]).size().reset_index(name="count")
            )
            total_counts = count_df.groupby(feature_col)["count"].transform("sum")
            count_df["proportion"] = count_df["count"] / total_counts
            sns.barplot(
                data=count_df,
                x=feature_col,
                y="proportion",
                hue=target_col,
                hue_order=[0, 1],
                alpha=0.7,
                ax=ax,
            )
        else:
            sns.countplot(
                data=df,
                x=feature_col,
                hue=target_col,
                hue_order=[0, 1],
                alpha=0.7,
                ax=ax,
            )
    else:
        sns.kdeplot(
            data=df,
            x=feature_col,
            hue=target_col,
            hue_order=[0, 1],
            fill=True,
            common_norm=False,
            alpha=0.5,
            linewidth=0.1,
            ax=ax,
        )

    title = title or f"{feature_col}"
    ax.set_title(title)

    if saveas_filename:
        fig.savefig(f"{folder}/{saveas_filename}.png", bbox_inches="tight", dpi=300)

    plt.close()


def plot_multifaceted_features(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    titles: list,
    saveas_filename: str,
    max_cols: int = 3,
    figsize: tuple = (15, 5),
) -> None:
    """
    Set up multifaceted plots for multiple features with max multifaceted columns = 3.
    """
    num_features = len(feature_cols)
    num_rows = (num_features // max_cols) + int(num_features % max_cols > 0)
    fig, axes = plt.subplots(
        num_rows, max_cols, figsize=(figsize[0], figsize[1] * num_rows)
    )

    axes = axes.flatten()

    for i, feature_col in enumerate(feature_cols):
        plot_feature_distribution(
            df,
            feature_col=feature_col,
            target_col=target_col,
            title=titles[i] if titles else f"{feature_col}",
            ax=axes[i],
        )

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    fig.savefig(f"{folder}/{saveas_filename}.png", bbox_inches="tight", dpi=300)
    plt.close()
