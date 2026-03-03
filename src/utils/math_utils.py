import pandas as pd


def compute_relative_importance(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Compute relative and cumulative importance of a series,
    e.g. df["feature_importance"].
    """
    sorted_series = series.sort_values(ascending=False)

    relative_importance = sorted_series / sorted_series.iloc[0]
    cumulative_importance = sorted_series.cumsum() / sorted_series.sum()

    return relative_importance, cumulative_importance
