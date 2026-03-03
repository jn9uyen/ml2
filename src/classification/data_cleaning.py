"""
Data Cleaning
-------------
Functions used for cleaning data in a pandas DataFrame.
"""

import numpy as np
import pandas as pd
from helper import logging_config

logger = logging_config.getLogger(__name__)

def clean_missing_values(
    pdf: pd.DataFrame,
    missing_values: list[str] = ["", "none", "unknown"],
    excl_cols: list[str] = ["id"],
) -> pd.DataFrame:
    """
    Replace missing values in string columns with 'NA' and numeric columns with np.nan.
    """
    cols_to_clean = [col for col in pdf.columns if col not in excl_cols]
    str_cols = pdf[cols_to_clean].select_dtypes(include=["object"]).applymap(
        lambda x: x.lower() if isinstance(x, str) else x
    )
    pdf[str_cols.columns] = str_cols.replace(missing_values, None)
    num_cols = pdf.select_dtypes(include=[np.number]).columns
    pdf[num_cols] = pdf[num_cols].replace({None: np.nan})

    return pdf

def clean_special_chars(
    pdf: pd.DataFrame,
    special_chars: list = ["[", ",", "]", "<", ">"],
    replace_char: str = "?",
) -> pd.DataFrame:
    """
    Replace all special characters in the DataFrame with an sklearn-compatible character
    """

    def replace_special_chars(val, chars):
        if isinstance(val, str):
            for char in chars:
                val = val.replace(char, replace_char)
        return val

    return pdf.applymap(lambda x: replace_special_chars(x, special_chars))

def clean_dollar_cols(
    pdf: pd.DataFrame, chars: list[str] = ["$", ",", "."]
) -> pd.DataFrame:
    """
    Convert str columns containing values like $2000.00 to floats: 2000.00
    """
    pdf_str_cols = pdf.select_dtypes(include="object").applymap(str)

    # Pattern to match columns with dollar amounts
    pattern = "".join([f"(?=.*{char})" for char in chars])
    dollar_cols = pdf_str_cols.columns[
        pdf_str_cols.apply(lambda col: col.str.contains(pattern).any())
    ]

    for col in dollar_cols:
        replaced = pdf[col]
        for char in chars:
            if char in ["$", ","]:
                replaced = replaced.str.replace(char, "")
        try:
            pdf[col] = replaced.astype(float)
        except ValueError:
            logger.info("string contains non-numeric characters")
            pass

    return pdf

def clean_to_binary_cols(
    pdf: pd.DataFrame,
    binary_val_sets: list = [{"yes", "no"}, {"true", "false"}, {True, False}],
) -> pd.DataFrame:
    """
    Convert column with values like ["yes", "no"] to binary categorical values
    """

    binary_mapping = {}
    for val_set in binary_val_sets:
        val_list = list(val_set)
        binary_mapping[val_list[0]] = 1
        binary_mapping[val_list[1]] = 0

    str_bool_cols = pdf.select_dtypes(include=["object", "bool"])

    def _is_subset_of_allowed_values(column, allowed_values_sets):
        col_vals = set(column.dropna())
        return any(
            col_vals.issubset(allowed_set) for allowed_set in allowed_values_sets
        )

    binary_cols = str_bool_cols.columns[
        str_bool_cols.apply(
            lambda col: _is_subset_of_allowed_values(col, binary_val_sets)
        )
    ]
    pdf[binary_cols] = pdf[binary_cols].applymap(lambda x: binary_mapping.get(x, x))

    return pdf

def convert_date_cols(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Identify and convert columns with date values in yyyy-mm-dd format to datetime,
    excluding columns with only missing values or those not in the correct date format.
    """

    def _is_date_column(series):
        non_na_values = series.dropna().astype(str)
        return (
            not non_na_values.empty and
            non_na_values.str.match(r"^\d{4}-\d{2}-\d{2}$").all()
        )

    date_cols = [col for col in pdf.columns if _is_date_column(pdf[col])]

    for col in date_cols:
        print(f"Converting column '{col}' from {pdf[col].dtype} to datetime")
        pdf[col] = pd.to_datetime(pdf[col])

    return pdf