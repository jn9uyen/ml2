import math

import numpy as np
import pandas as pd
import pandas_gbq
from google.cloud import bigquery
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle


def load_data_bq(
    table_id: str, project_id: str, dataset_id: str, client: bigquery.Client
) -> pd.DataFrame | None:
    table_path = f"{project_id}.{dataset_id}.{table_id}"
    query = f"""select * from `{table_path}`"""

    try:
        # The use_bqstorage_client=True argument is important for performance
        query_job = client.query(query)
        return query_job.to_dataframe(create_bqstorage_client=True)
    except Exception as e:
        print(f"Error loading data from BigQuery table {table_path}: {e}")


def saved_data_bq(
    data: pd.DataFrame, table_id: str, project_id: str, dataset_id: str
) -> None:
    table_path = f"{project_id}.{dataset_id}.{table_id}"
    try:
        pandas_gbq.to_gbq(
            data,
            destination_table=table_path,
            project_id=project_id,
            if_exists="replace",
        )
        print(f"Saved data to BigQuery table: {table_path}")
    except Exception as e:
        print(f"Error saving data to BigQuery table {table_path}: {e}")


def custom_train_test_split(
    df: pd.DataFrame,
    date_col: str | None = None,
    split_date: str | None = None,
    target_col: str = "target",
    feature_cols: list[str] | None = None,
    id_col: str | None = None,
    by_time: bool = True,
    balance: bool = False,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split a dataset by time or randomly, with optional class balancing and feature
    selection.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to split.
    date_col : str, optional
        Column with datetime values (needed if `by_time=True`).
    split_date : str, optional
        Date to split on (needed if `by_time=True`).
    target_col : str, default='target'
        Name of target variable.
    feature_cols : list of str, optional
        List of feature columns to include in X_train and X_test.
    id_col : str, optional
        Column to use as index for X_train, X_test, y_train, and y_test.
    by_time : bool, default=True
        If True, split by time; otherwise, use random split.
    balance : bool, default=False
        If True and `by_time=True`, undersample test set for class balance.
    **kwargs : dict
        Passed to `train_test_split` (e.g., `test_size`, `stratify`).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        (X_train, X_test, y_train, y_test) with optional indexing.
    """

    df = df.copy()

    if id_col:
        if id_col not in df.columns:
            raise ValueError(f"`id_col` '{id_col}' is not found in the dataset.")
        df.set_index(id_col, inplace=True)

    if by_time:
        if date_col is None or split_date is None:
            raise ValueError(
                "`date_col` and `split_date` are required for `by_time=True`."
            )

        df[date_col] = pd.to_datetime(df[date_col])
        train = df[df[date_col] < split_date]
        test = df[df[date_col] >= split_date]

        if balance:
            rus = RandomUnderSampler(random_state=42)
            # Ensure we don't try to resample an empty test set
            if not test.empty:
                X_test_resampled, y_test_resampled = rus.fit_resample(
                    test.drop(columns=[target_col]), test[target_col]
                )
                test = pd.concat([X_test_resampled, y_test_resampled], axis=1)

    else:
        train, test = train_test_split(df, **kwargs)

    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != target_col]

    X_train = pd.DataFrame(train[feature_cols])
    X_test = pd.DataFrame(test[feature_cols])

    y_train = train[target_col]
    y_test = test[target_col]

    return X_train, X_test, y_train, y_test


def calc_ntile(series: pd.Series, n: int = 100) -> tuple[pd.Series, np.ndarray]:
    """
    Calculate the n-tile ranking.

    Parameters
    ----------
    series : pd.Series
        The input Series for which n-tiles are calculated.
    n : int, optional (default=100)
        The number of quantiles to divide the data into.

    Returns
    -------
    tuple[pd.Series, np.ndarray]
        - A Series of integer n-tile ranks ranging from 1 to n.
        - A NumPy array of bin edges used for quantile calculation.
    """
    series = series.copy()

    # Use qcut to assign percentile bins; drop duplicates to get unique bin edges.
    try:
        ntile, bins = pd.qcut(
            series, q=n, labels=False, retbins=True, duplicates="drop"
        )
    except ValueError as e:
        raise ValueError(
            f"Could not create n-tiles with `n={n}`. The data may have too many "
            f"duplicate values. Original error: {e}"
        )

    # Convert to 1-based index (1 to n)
    ntile = ntile + 1

    return ntile, bins


def group_by_volume(
    df: pd.DataFrame,
    value_col: str,
    group_keys: list[str],
    min_size: int | None = None,
    n_groups: int | None = None,
) -> pd.Series:
    """
    Group a DataFrame by discrete values based on volume constraints.

    This function bins data by operating in one of two modes:
    1. `min_size`: Bins data ensuring each bin has at least `min_size` rows.
    2. `n_groups`: Bins data into exactly `n_groups` with roughly equal volume.

    This function uses NumPy for efficient processing within each group.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    value_col : str
        The name of the column containing discrete values to be binned.
    group_keys : list[str]
        A list of column names to group by before applying the binning logic.
    min_size : int, optional
        The minimum number of rows for each bin. Mutually exclusive with `n_groups`.
    n_groups : int, optional
        The desired number of bins. Mutually exclusive with `min_size`.

    Returns
    -------
    pd.Series
        A Series with the same index as the input df, containing the new group labels.

    Raises
    ------
    ValueError
        If neither or both `min_size` and `n_groups` are provided.

    Examples
    --------
    >>> data = {
    ...     'product': ['A'] * 150 + ['B'] * 80,
    ...     'value':   ([10]*20 + [20]*30 + [30]*40 + [40]*50 + [50]*10) +
    ...                ([100]*50 + [200]*30)
    ... }
    >>> sample_df = pd.DataFrame(data)
    >>> sample_df["value_group"] = group_by_volume(
    ...     sample_df, value_col='value', group_keys=['product'], n_groups=2
    ... )
    >>> print(sample_df.groupby(['product', 'value_group']).size())
    product  value_group
    A        01. [10.00, 20.00]     50
             02. [30.00, 50.00]    100
    B        01. [100.00, 100.00]    50
             02. [200.00, 200.00]    30
    dtype: int64
    """
    if not (min_size is None) ^ (n_groups is None):
        raise ValueError("Please provide exactly one of 'min_size' or 'n_groups'.")

    new_col_name = f"{value_col}_group"

    if new_col_name in df.columns:
        df = df.drop(columns=[new_col_name])

    def _process_group(group: pd.DataFrame) -> pd.DataFrame:
        """Process a single group for binning."""
        if group.empty:
            return pd.DataFrame({new_col_name: pd.Series(dtype="str")})

        # Get sorted unique values and their counts.
        sorted_group = group.sort_values(value_col)
        values = sorted_group[value_col].to_numpy()
        unique_vals, counts = np.unique(values, return_counts=True)

        bin_ids = np.zeros(len(group), dtype=np.int16)
        bin_edges = []
        cursor = 0

        # Determine bin assignments.
        if min_size:
            current_bin_size = 0
            bin_start_idx = 0
            for i in range(len(unique_vals)):
                current_bin_size += counts[i]
                if current_bin_size >= min_size:
                    bin_end_idx = bin_start_idx + current_bin_size
                    bin_ids[bin_start_idx:bin_end_idx] = len(bin_edges)
                    bin_edges.append((unique_vals[cursor], unique_vals[i]))
                    cursor = i + 1
                    bin_start_idx = bin_end_idx
                    current_bin_size = 0

            if current_bin_size > 0:
                if bin_edges:
                    last_bin_idx = len(bin_edges) - 1
                    bin_ids[bin_start_idx:] = last_bin_idx
                    min_val, _ = bin_edges[last_bin_idx]
                    bin_edges[last_bin_idx] = (min_val, unique_vals[-1])
                else:
                    bin_ids.fill(0)
                    bin_edges.append((unique_vals[0], unique_vals[-1]))
        else:
            assert n_groups is not None

            rows_processed = 0
            for i in range(n_groups):
                remaining_bins = n_groups - i
                if remaining_bins == 0:
                    continue
                remaining_rows = len(group) - rows_processed
                target_size = math.ceil(remaining_rows / remaining_bins)
                current_bin_size = 0
                start_cursor = cursor
                while cursor < len(unique_vals):
                    current_bin_size += counts[cursor]
                    cursor += 1
                    if current_bin_size >= target_size and i < n_groups - 1:
                        break
                if current_bin_size > 0:
                    bin_start_idx = rows_processed
                    bin_end_idx = rows_processed + current_bin_size
                    bin_ids[bin_start_idx:bin_end_idx] = i
                    bin_edges.append(
                        (unique_vals[start_cursor], unique_vals[cursor - 1])
                    )
                    rows_processed = bin_end_idx

        label_map = {
            i: f"{i + 1:02d}. [{min_val:.2f}, {max_val:.2f}]"
            for i, (min_val, max_val) in enumerate(bin_edges)
        }

        sorted_group["bin_id"] = bin_ids
        sorted_group[new_col_name] = sorted_group["bin_id"].map(label_map)
        return sorted_group[[new_col_name]]

    # --- Apply _process_group() to each group ---
    grouped_results = df.groupby(group_keys, group_keys=False).apply(_process_group)
    df_with_groups = df.join(grouped_results)

    return df_with_groups[new_col_name]
