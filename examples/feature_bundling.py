import networkx as nx
import numpy as np
import pandas as pd


def feature_bundling(
    df: pd.DataFrame, threshold: float = 0.7
) -> tuple[dict, dict, dict, nx.Graph]:
    """
    Bundle features by grouping highly correlated features using connected components,
    and select one representative feature from each bundle using a combined score.

    The combined score for a feature is:

    combined_score = (sum of absolute correlations with other features in the bundle)
                        + (normalized variance of the feature)

    where the variance is normalized using z-score standardization per bundle
    to be on the same scale as the correlation sum.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing numerical features.
    threshold : float, optional
        Minimum absolute correlation required to consider two features connected.
        Default is 0.7.

    Returns
    -------
    selected_features : dict
        A dictionary mapping each bundle (component id) to the selected representative
        feature.
    bundles : dict
        A dictionary mapping each bundle (component id) to the list of features in
        that bundle.
    bundle_scores : dict
        A dictionary mapping each bundle (component id) to a pandas Series of
        combined scores for each feature.
    G : networkx.Graph
        The feature correlation graph used for bundling.
    """
    corr = df.corr()
    mask = abs(corr) >= threshold
    # Only consider the upper triangle (excluding the diagonal).
    mask &= np.triu(np.ones(corr.shape, dtype=bool), k=1)
    rows, cols = np.where(mask)
    edges = [(corr.index[i], corr.columns[j]) for i, j in zip(rows, cols)]

    G = nx.Graph()
    G.add_nodes_from(corr.index)
    G.add_edges_from(edges)

    # # Plot the graph
    # plt.figure(figsize=(8, 6))
    # pos = nx.spring_layout(G, seed=42)
    # nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=1500)
    # nx.draw_networkx_edges(G, pos, edge_color="gray")
    # nx.draw_networkx_labels(G, pos, font_size=10)
    # plt.title(f"Feature Correlation Graph (Edges for |corr| >= {threshold})")
    # plt.axis("off")
    # plt.show()

    # Use connected components to group features into bundles.
    connected_components = list(nx.connected_components(G))
    bundles = {i: sorted(list(comp)) for i, comp in enumerate(connected_components)}

    # Compute combined scores and select representative features.
    selected_features = {}
    bundle_scores = {}  # To store the combined score for each feature in each bundle.
    for bundle_id, feats in bundles.items():
        if len(feats) == 1:
            # Only one feature in the bundle: select it.
            selected_features[bundle_id] = feats[0]
            bundle_scores[bundle_id] = pd.Series({feats[0]: 0})
        else:
            # Extract submatrix of absolute correlations for features in the bundle.
            sub_corr = pd.DataFrame(corr.loc[feats, feats].abs())
            # Zero out the diagonal (ignore self-correlation).
            np.fill_diagonal(sub_corr.values, 0)
            # Sum the absolute correlations along rows. This represents how strongly
            # a feature is correlated with *other* features in its bundle.
            corr_sum = sub_corr.sum(axis=1)

            # --- Feature Variances ---
            # Ensure only numeric columns are used for variance calculation.
            numeric_df_subset = df[feats].select_dtypes(include=np.number)

            if numeric_df_subset.empty:
                # Assign zero variance to avoid errors and skip normalization.
                variances = pd.Series(0.0, index=feats)
            else:
                variances = numeric_df_subset.var()

            # --- Z-score Normalization for Variance ---
            mean_variance = variances.mean()
            std_variance = variances.std()

            if std_variance > 0:
                normalized_variance = (variances - mean_variance) / std_variance
            else:
                # If all variances are the same, their normalized contribution is 0.
                normalized_variance = pd.Series(0.0, index=variances.index)

            # Combine the correlation sum and the normalized variance.
            # The normalized variance now represents how "unusual" a feature's variance
            # is within its bundle, relative to the other features in the same bundle.
            # This prevents features with inherently high variances from dominating
            # solely due to their magnitude.
            combined_score = corr_sum + normalized_variance
            bundle_scores[bundle_id] = combined_score
            selected_features[bundle_id] = combined_score.idxmax()

    return selected_features, bundles, bundle_scores, G


# --- Example usage ---
np.random.seed(0)
n_samples = 100

# Generate synthetic data for correlated groups:
# Groceries features: two correlated features
groceries_base = np.random.uniform(20, 100, n_samples)
groceries_spend_avg_30d = groceries_base + np.random.normal(0, 5, n_samples)
groceries_spend_avg_90d = groceries_base * 3 + np.random.normal(0, 10, n_samples)

# Gambling features: two correlated features
gambling_base = np.random.uniform(0, 50, n_samples)
gambling_debit_avg_7d = gambling_base + np.random.normal(0, 2, n_samples)
gambling_debit_avg_180d = gambling_base * 2 + np.random.normal(0, 3, n_samples)

# An independent feature
other_spend = np.random.uniform(0, 200, n_samples)

# Build the DataFrame
df = pd.DataFrame(
    {
        "groceries_spend_avg_30d": groceries_spend_avg_30d,
        "groceries_spend_avg_90d": groceries_spend_avg_90d,
        "gambling_debit_avg_7d": gambling_debit_avg_7d,
        "gambling_debit_avg_180d": gambling_debit_avg_180d,
        "other_spend": other_spend,
    }
)

# Clip to ensure non-negativity
# df = df.clip(lower=0)

# Run the feature bundling function with combined score
selected_features, bundles, bundle_scores, G = feature_bundling(df, threshold=0.9)

print("Bundles:")
for bundle_id, feats in bundles.items():
    print(f"Bundle {bundle_id}: {feats}")

print("\nSelected representative features from each bundle:")
for bundle_id, feature in selected_features.items():
    print(f"Bundle {bundle_id}: {feature}")

print("\nCombined scores for each feature in each bundle:")
for bundle_id, scores in bundle_scores.items():
    print(f"\nBundle {bundle_id} scores:")
    print(scores)
