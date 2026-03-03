from typing import Literal

import numpy as np
import pandas as pd

from .classification import compute_classification_metrics
from .regression import compute_regression_metrics


def weighted_average(
    df: pd.DataFrame,
    value_col: str,
    weight_col: str,
    group_cols: list[str],
    new_col_name: str = "weighted_average",
) -> pd.DataFrame:
    """
    Calculate the weighted average of a column for specified groups.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    value_col : str
        Column name for the values to average.
    weight_col : str
        Column name for the weights.
    group_cols : list of str
        List of column names to group by.
    new_col_name : str, optional
        Name for the new column containing the weighted averages (default is
        "weighted_average").

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the grouping columns and the calculated weighted average
        for each group.
    """
    df_copy = df.copy()
    product_col = f"_{value_col}_x_{weight_col}"
    df_copy[product_col] = df_copy[value_col] * df_copy[weight_col]
    grouped = df_copy.groupby(group_cols).agg(
        sum_product=(product_col, "sum"), sum_weight=(weight_col, "sum")
    )
    grouped[new_col_name] = grouped["sum_product"] / grouped["sum_weight"]

    return grouped[[new_col_name]].reset_index()


def compute_model_metrics(
    model_type: Literal["classification", "regression"],
    dataset_list: list[tuple[Literal["train", "val", "test"], np.ndarray, np.ndarray]],
    y_train_mean: float,
) -> pd.DataFrame:
    """
    Compute model evaluation metrics for a list of datasets, e.g. train, test.

    Parameters
    ----------
    model_type : Literal["classification", "regression"]
        The type of model being evaluated.
    dataset_list : list[tuple[Literal["train", "val", "test"], ArrayLike, ArrayLike]]
        A list of tuples containing dataset information
        (name, true values, predicted values), e.g. [("test", y_test, y_pred)].
    y_train_mean : float
        The mean of the training target values; used to compute the skill score.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the evaluation metrics for each dataset.
    """
    dataset_order = {"train": 1, "val": 2, "test": 3}
    dataset_list_ordered = sorted(
        dataset_list, key=lambda item: dataset_order.get(item[0], 99)
    )

    metrics_list = []
    for dataset_name, y_true, y_pred in dataset_list_ordered:
        match model_type:
            case "regression":
                metrics = compute_regression_metrics(
                    y_true, y_pred, dataset_name, y_train_mean
                )
            case "classification":
                metrics = compute_classification_metrics(
                    y_true, y_pred, dataset_name, y_train_mean
                )
        metrics_list.append(metrics)

    dataset_names = [name for name, _, _ in dataset_list_ordered]
    metrics_df = pd.DataFrame(metrics_list, index=dataset_names)

    if "size" in metrics_df.columns:
        metrics_df.insert(1, "pct", metrics_df["size"] / metrics_df["size"].sum())

    return metrics_df
