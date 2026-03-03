from collections.abc import Callable, Sequence

import pandas as pd


def generate_grouped_value_counts_over_time(
    df: pd.DataFrame,
    entity_cols: list[str],
    value_col: str,
    date_col: str,
    time_windows_days: list[int] = [180],
    break_ties: bool = True,
) -> pd.DataFrame:
    """
    Generate value counts for entity groups over tumbling time windows.

    Over non-overlapping time windows, this function groups by a set of entity
    columns, counts the occurrences of values in a specified value column,
    and flags the modal (most frequent) value for each window.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    entity_cols : list of str
        List of column names that define a unique entity (e.g.,
        ["pms_patient_code", "product_category"]).
    value_col : str
        The name of the column whose values will be counted (e.g.,
        "product_category_purchase_interval_days").
    date_col : str
        The name of the date column used for creating time windows.
    time_windows_days : list of int, default=[180]
        List of time window sizes in days.
    break_ties : bool, default=True
        Whether to break ties when flagging the modal value.

    Returns
    -------
    pd.DataFrame
        A DataFrame with counts of each value per entity and time window.
    """
    required_cols = entity_cols + [value_col, date_col]
    if not set(required_cols).issubset(df.columns):
        missing = set(required_cols) - set(df.columns)
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    window_groupby_cols = entity_cols + ["window_start_date"]

    all_results = []

    for window_days in time_windows_days:
        # Group by entities, the value itself, and the time window.
        initial_groupby_cols = entity_cols + [value_col]
        stepped_groups = df.groupby(
            initial_groupby_cols + [pd.Grouper(key=date_col, freq=f"{window_days}D")]
        )

        # Aggregate to get counts and date ranges for each value.
        agg_dict = {
            "count": (value_col, "count"),
            f"first_{date_col}": (date_col, "min"),
            f"last_{date_col}": (date_col, "max"),
        }
        result_df = stepped_groups.agg(**agg_dict).reset_index()  # type: ignore
        result_df["window_days"] = window_days
        result_df = result_df.rename(columns={date_col: "window_start_date"})

        # Create a sub-index for each value's appearance within a window.
        result_df["value_index"] = result_df.groupby(window_groupby_cols).cumcount() + 1
        all_results.append(result_df)

    if not all_results:
        return pd.DataFrame()

    combined_df = pd.concat(all_results, ignore_index=True)

    # Create a global index for each unique time window start date.
    unique_window_starts = combined_df["window_start_date"].sort_values().unique()
    window_map = {date: i + 1 for i, date in enumerate(unique_window_starts)}
    combined_df["window_index"] = combined_df["window_start_date"].map(window_map)

    # Flag the modal value within each patient/category/window group
    combined_df["is_modal"] = flag_group_max_value(
        combined_df,
        groupby_cols=window_groupby_cols,
        value_col="count",
        break_ties=break_ties,
    )

    final_cols = entity_cols + [
        "window_days",
        "window_start_date",
        "window_index",
        value_col,
        "value_index",
        "count",
        "is_modal",
        f"first_{date_col}",
        f"last_{date_col}",
    ]
    return combined_df[final_cols]


def flag_group_max_value(
    df: pd.DataFrame,
    groupby_cols: list[str],
    value_col: str,
    break_ties: bool = True,
    as_type_int: bool = True,
) -> pd.Series:
    """
    Identify the row(s) with the maximum value within each group of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    groupby_cols : list of str
        The columns to group by.
    value_col : str
        The column in which to find the maximum value.
    break_ties : bool, optional
        If True (default), flags only the first occurrence of the maximum value
        in case of a tie. If False, flags all rows that share the maximum value.

    Returns
    -------
    pd.Series
        A boolean Series with the same index as the input DataFrame,
        marking the row(s) with the maximum value as True.
    """
    assert set(groupby_cols + [value_col]).issubset(
        df.columns
    ), "DataFrame must contain the specified grouping and value columns."

    if break_ties:
        max_indices = df.groupby(groupby_cols)[value_col].idxmax()
        output_series = pd.Series(False, index=df.index)
        output_series.loc[max_indices] = True
    else:
        # Find the maximum values within each group and broadcast it back.
        max_values_in_group = df.groupby(groupby_cols)[value_col].transform("max")
        output_series = df[value_col] == max_values_in_group

    if as_type_int:
        return output_series.astype("Int8")
    return output_series


def partition_numbers_by_proximity(
    numbers: Sequence[int | float], threshold: int | float
) -> list[Sequence[int | float]]:
    """
    Partition numbers to maximize the largest group's size.

    This function takes a list of numbers and partitions them into non-overlapping
    groups. A group is considered valid if the difference between its maximum and
    minimum value is ≤ the specified threshold. The function uses dynamic programming
    to find an optimal partition that maximizes the size of the largest group.

    Parameters
    ----------
    numbers : Sequence[int | float]
        A list of numbers to be partitioned. The list can contain duplicates
        and will be sorted internally.
    threshold : int | float
        The maximum allowed difference (`max - min`) within a valid group.
        Must be a non-negative number.

    Returns
    -------
    list[Sequence[int | float]]
        The optimal partitioning of the numbers, returned as a list of lists.

    Examples
    --------
    >>> numbers = [15, 17, 18, 20]
    >>> threshold = 3
    >>> group_numbers_by_proximity(numbers, threshold)
    [[15, 17, 18], [20]]

    >>> numbers2 = [1, 2, 3, 8, 9, 12]
    >>> threshold2 = 2
    >>> group_numbers_by_proximity(numbers2, threshold2)
    [[1, 2, 3], [8, 9], [12]]
    """
    if not numbers:
        return []

    assert threshold > 0, "Threshold must be > 0."

    # Sort unique numbers to enable efficient group validation.
    nums: Sequence[int | float] = sorted(list(set(numbers)))
    n: int = len(nums)

    # Memoization cache for the dynamic programming solution.
    memo: dict[int, tuple[int, list[Sequence[int | float]]]] = {}

    def solve(i: int) -> tuple[int, list[Sequence[int | float]]]:
        """
        Find the optimal partition for the prefix nums[:i].
        Returns a tuple: (max_group_size, partition).
        """
        if i == 0:
            return (0, [])
        if i in memo:
            return memo[i]

        best_option: tuple[int, list[Sequence[int | float]]] = (-1, [])

        # For the i-th number, try all valid "last groups".
        for j in range(i):
            new_group: Sequence[int | float] = nums[j:i]

            # Check if the new group is valid.
            if new_group[-1] - new_group[0] <= threshold:
                prev_max_size, prev_partition = solve(j)
                current_max_size: int = max(prev_max_size, len(new_group))

                # If this option gives a larger max group, it's the new best.
                if current_max_size > best_option[0]:
                    current_partition: list[Sequence[int | float]] = prev_partition + [
                        new_group
                    ]
                    best_option = (current_max_size, current_partition)

        memo[i] = best_option
        return best_option

    # Solve for the entire list.
    _max_size, final_partition = solve(n)
    return final_partition


def apply_partitioning_to_groups(
    df: pd.DataFrame,
    groupby_cols: list[str],
    value_col: str,
    partition_func: Callable[
        [Sequence[int | float], int | float], list[Sequence[int | float]]
    ],
    proximity_threshold: int | float,
    output_value_col: str | None = None,
    output_group_col: str | None = None,
) -> pd.DataFrame:
    """
    Apply a partitioning function to a DataFrame's groups.

    This function groups a DataFrame, applies a partitioning function to a
    specified column in each group, and returns a tidy DataFrame of the results.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    groupby_cols : list of str
        A list of column names to group the DataFrame by.
    value_col : str
        The name of the column containing the numbers to be partitioned.
    partition_func : callable
        A function that takes a sequence of numbers and a threshold, and returns
        a list of partitioned groups (list of sequences).
    proximity_threshold : int or float
        The threshold value passed to the partitioning function.
    output_group_name : str, optional
        The name for the output column identifying the partition group index.
    output_interval_name : str, optional
        The name for the output column containing the partitioned values.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the grouping columns and the partitioned results.
    """
    output_value_col = output_value_col or value_col
    output_group_col = output_group_col or "group_index"

    required_cols = groupby_cols + [value_col]
    if not set(required_cols).issubset(df.columns):
        missing = set(required_cols) - set(df.columns)
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    results_list = []
    grouped = df.groupby(groupby_cols)

    for group_keys, group_df in grouped:
        values = group_df[value_col].tolist()
        partitions = partition_func(values, proximity_threshold)

        # Dynamically create the base dictionary for each result row.
        base_result_row = dict(zip(groupby_cols, group_keys))

        for partition_idx, partition_group in enumerate(partitions):
            for value in partition_group:
                result_row = base_result_row.copy()
                result_row[output_value_col] = value
                result_row[output_group_col] = partition_idx + 1
                results_list.append(result_row)

    return pd.DataFrame(results_list)
