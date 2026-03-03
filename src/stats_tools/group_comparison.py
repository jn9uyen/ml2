from pprint import pprint
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kruskal, levene
from statsmodels.formula.api import ols
from statsmodels.stats import multicomp


def classify_effect_size(
    value: float, test_type: Literal["eta_sq", "omega_sq", "eta_sq_H", "cohen_d"]
) -> str:
    """Classify an effect size value as trivial, small, medium, or large."""
    if pd.isna(value):
        return ""

    if test_type == "cohen_d":
        abs_value = abs(value)
        if abs_value >= 0.8:
            return "large"
        elif abs_value >= 0.5:
            return "medium"
        elif abs_value >= 0.2:
            return "small"
        else:
            return "trivial"
    elif test_type in ["eta_sq", "omega_sq", "eta_sq_H"]:
        if value >= 0.14:
            return "large"
        elif value >= 0.06:
            return "medium"
        elif value >= 0.01:
            return "small"
        else:
            return "trivial"
    return "unknown"


def analyze_differences_by_factors(
    df: pd.DataFrame,
    value_col: str,
    factor_cols: list[str],
    min_stratum_size: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """
    Perform Levene's test, ANOVA (1-way or 2-way), Tukey HSD, and Kruskal-Wallis test,
    all including effect size measures and classifications.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    value_col : str
        The name of the numeric dependent variable.
    factor_cols : list[str]
        A list with the names of one or two columns to use as factors.
    min_stratum_size : int | None
        If provided, only includes strata with at least this many samples.
    random_state : int
        Random seed for reproducibility when sampling.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None
        A tuple containing the ANOVA table, Tukey HSD results,
        and Kruskal-Wallis grouped data, or None if the analysis cannot be run.
    """
    if not 1 <= len(factor_cols) <= 2:
        raise ValueError("This function supports one or two factors only.")

    print(f"--- Running Analysis for Factor(s): {', '.join(factor_cols)} ---")

    analysis_df = df[factor_cols + [value_col]].copy()
    analysis_df[value_col] = pd.to_numeric(analysis_df[value_col], errors="coerce")
    analysis_df.dropna(inplace=True)

    if min_stratum_size is not None:
        analysis_df = analysis_df.groupby(
            factor_cols, group_keys=False
        ).apply(  # type: ignore
            lambda x: x.sample(
                n=min(len(x), min_stratum_size), random_state=random_state
            )
        )
        print(f"Limited to strata with min {min_stratum_size} samples.")

    if analysis_df.empty:
        print("DataFrame is empty after filtering/sampling. Cannot run analysis.")
        return None

    analysis_df[factor_cols] = analysis_df[factor_cols].astype("category")

    # --- 1. Assumption Check: Levene's Test ---
    print("\n--- 1. Assumption Check: Levene's Test for Equal Variances ⚖️ ---")

    grouped_data = [
        group[value_col].values for _, group in analysis_df.groupby(factor_cols)
    ]
    if len(grouped_data) > 1:
        stat, p_value = levene(*grouped_data)
        print(f"  - Statistic: {stat:.4f}, p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("⚠️ Result: Variances are UNEQUAL. ANOVA results may be unreliable.")
        else:
            print("✅ Result: Variances are EQUAL. ANOVA assumption is met.")
    else:
        print("Not enough groups for Levene's test.")

    # --- 2. ANOVA (1-Way or 2-Way) ---
    print(f"\n--- 2. {len(factor_cols)}-Way ANOVA Results ---")

    terms = [f'C(Q("{col}"))' for col in factor_cols]
    formula = f'Q("{value_col}") ~ {" + ".join(terms)}'
    if len(factor_cols) == 2:
        interaction_term = ":".join(terms)
        formula += f" + {interaction_term}"

    model = ols(formula, data=analysis_df).fit()
    anova_df = sm.stats.anova_lm(model, typ=2)

    # Effect size measures.
    anova_df["eta_sq"] = anova_df["sum_sq"] / (anova_df["sum_sq"].sum())
    anova_df["omega_sq"] = (anova_df["sum_sq"] - (anova_df["df"] * model.mse_resid)) / (
        anova_df["sum_sq"].sum() + model.mse_resid
    )
    anova_df["effect_size"] = anova_df["omega_sq"].apply(
        lambda x: classify_effect_size(x, "omega_sq")
    )

    pprint(anova_df)
    print(
        "\nInterpretation Guide for Eta/Omega Squared:"
        "0.01 (small), 0.06 (medium), 0.14 (large)"
        "\nANOVA Interpretation:"
    )
    for effect, row in anova_df.iterrows():
        if str(effect).lower() == "residual":
            continue
        p_value = row["PR(>F)"]
        significance_str = "Significant" if p_value < 0.05 else "Not significant"
        effect_size = row["effect_size"]
        omega_sq_value = row["omega_sq"]
        print(
            f"Effect of {effect}: p-value={p_value:.4f}, Result: {significance_str}, "
            f"Effect Size: {effect_size} (ω² = {omega_sq_value:.4f})"
        )

    # --- 3. Tukey's HSD & 4. Kruskal-Wallis Test ---
    print("\n--- 3. Tukey's HSD Results (compares all combined groups) ---")

    # These tests require a single grouping column.
    analysis_df["combined_group"] = (
        analysis_df[factor_cols].astype(str).agg(" | ".join, axis=1)
    )
    multi_comp = multicomp.MultiComparison(
        analysis_df[value_col], analysis_df["combined_group"]
    )
    tukey_result = multi_comp.tukeyhsd()
    tukey_df = pd.DataFrame(
        data=tukey_result._results_table.data[1:],
        columns=tukey_result._results_table.data[0],
    )
    tukey_df["meandiff"] = pd.to_numeric(tukey_df["meandiff"])
    tukey_df["p-adj"] = pd.to_numeric(tukey_df["p-adj"])

    # Calculate Cohen's d and classify it.
    pooled_sd = np.sqrt(model.mse_resid)
    tukey_df["cohen_d"] = tukey_df["meandiff"].astype(float) / pooled_sd
    tukey_df["effect_size"] = tukey_df["cohen_d"].apply(
        lambda x: classify_effect_size(x, "cohen_d")
    )

    significant_results = tukey_df[tukey_df["reject"]]
    if significant_results.empty:
        print("No significant differences found between any groups.")
    else:
        cols_to_show = [
            "group1",
            "group2",
            "meandiff",
            "p-adj",
            "cohen_d",
            "effect_size",
        ]
        print("Significant differences were found between the following groups:")
        pprint(significant_results[cols_to_show].round(4))

    # --- 4. Kruskal-Wallis Test with Eta Squared ---
    print("\n--- 4. Kruskal-Wallis Test (with Eta Squared Effect Size) ---")
    kruskal_grouped_data = [
        g[value_col].values for _, g in analysis_df.groupby("combined_group")
    ]
    kw_df = pd.DataFrame()

    if len(kruskal_grouped_data) > 1:
        H_stat, p_value = kruskal(*kruskal_grouped_data)
        N = len(analysis_df)
        k = len(kruskal_grouped_data)
        eta_squared_H = (H_stat - k + 1) / (N - k) if (N - k) != 0 else 0

        results_dict = {
            "H_statistic": H_stat,
            "p_value": p_value,
            "eta_squared_H": eta_squared_H,
        }
        kw_df = pd.DataFrame(results_dict, index=[0])
        kw_df["effect_size"] = kw_df["eta_squared_H"].apply(
            lambda x: classify_effect_size(x, "eta_sq_H")
        )

        pprint(kw_df)
        print(f"\nResult: {'Significant' if p_value < 0.05 else 'Not significant'}")
        print(
            "Interpretation Guide for Eta Squared (η²H): "
            "0.01 (small), 0.06 (medium), 0.14 (large)"
        )
    else:
        print("Not enough groups for Kruskal-Wallis test.")

    return anova_df, tukey_df, kw_df
