from typing import cast

import networkx as nx
import numpy as np
import pandas as pd


def feature_bundling(
    df: pd.DataFrame, correlation_threshold: float = 0.8
) -> tuple[dict, dict, pd.DataFrame, nx.Graph, list[str]]:
    """
    Bundle features by grouping highly correlated features using connected components,
    and select one representative feature from each bundle using a two-step approach.

    This function automatically selects and operates only on the numeric columns
    of the input DataFrame.

    1. The primary selection metric is the sum of a feature's absolute correlations
       with other features in the bundle.
    2. Ties are broken by selecting the feature with the highest Coefficient of
       Variation (CV).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing numerical features.
    correlation_threshold : float, optional
        Minimum absolute correlation required to consider two features are connected.
        Default is 0.8.

    Returns
    -------
    selected_features : dict
        A dictionary mapping each bundle (component id) to the selected representative
        feature.
    bundles : dict
        A dictionary mapping each bundle (component id) to the list of features in
        that bundle.
    bundle_scores_df : pd.DataFrame
        A DataFrame containing detailed scores and selection status for each
        feature across all bundles.
    G : networkx.Graph
        The feature correlation graph used for bundling.
    non_numeric_features : list[str]
        A list of column names that were identified as non-numeric and
        excluded from the bundling process.
    """
    numeric_df = df.select_dtypes(include=np.number)
    non_numeric_features = df.select_dtypes(exclude=np.number).columns.tolist()

    corr = numeric_df.corr()
    mask = abs(corr) >= correlation_threshold
    # Only consider the upper triangle (excluding the diagonal).
    mask &= np.triu(np.ones(corr.shape, dtype=bool), k=1)
    rows, cols = np.where(mask)
    edges = [(corr.index[i], corr.columns[j]) for i, j in zip(rows, cols)]

    G = nx.Graph()
    G.add_nodes_from(corr.index)
    G.add_edges_from(edges)

    # Use connected components to group features into bundles.
    connected_components = list(nx.connected_components(G))
    bundles = {i: sorted(list(comp)) for i, comp in enumerate(connected_components)}

    # Compute combined scores and select representative features.
    selected_features = {}
    scores_list = []
    epsilon = 1e-9  # To prevent division by zero

    for bundle_id, feats in bundles.items():
        # Calculate stats for all features in the bundle.
        stds = cast(pd.Series, numeric_df[feats].std())
        abs_means = cast(pd.Series, abs(numeric_df[feats].mean()))
        coef_variations = stds.div(abs_means + epsilon)

        if len(feats) == 1:
            representative_feature = feats[0]
            corr_sum = pd.Series({feats[0]: 0.0})
        else:
            # Find the most representative feature using the correlation sum.
            sub_corr = pd.DataFrame(corr.loc[feats, feats].abs())
            np.fill_diagonal(sub_corr.values, 0)  # ignore self-correlation
            corr_sum: pd.Series = sub_corr.sum(axis=1)

            # Identify the top candidate(s) based on the correlation sum.
            max_corr_sum = corr_sum.max()
            top_candidates = corr_sum[corr_sum >= max_corr_sum * 0.999]

            if len(top_candidates) == 1:
                representative_feature = top_candidates.idxmax()
            else:
                # Tie-breaker: select the feature with the highest coef_variation.
                candidate_cv = cast(
                    pd.Series, coef_variations.loc[top_candidates.index]
                )
                representative_feature = candidate_cv.idxmax()

        selected_features[bundle_id] = representative_feature

        for feat in feats:
            is_selected = 1 if feat == representative_feature else 0
            scores_list.append(
                {
                    "bundle_id": bundle_id,
                    "feature_name": feat,
                    "is_selected": is_selected,
                    "corr_sum": corr_sum.loc[feat],
                    "coef_variation": coef_variations.loc[feat],
                }
            )

    bundle_scores_df = pd.DataFrame(scores_list)
    return selected_features, bundles, bundle_scores_df, G, non_numeric_features
