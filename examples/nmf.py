"""
This script demonstrates the use of Non-negative Matrix Factorization (NMF) for
feature reduction on a synthetic dataset with correlated features.

Interpretation of the results is provided, focusing on how NMF groups correlated
features together:
If one component shows high weights for the groceries features and low for gambling,
and the other component does the opposite, it indicates that NMF has grouped these
correlated features together. You can then choose a representative feature from each
group based on the highest weight.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

# Set random seed and number of samples
np.random.seed(0)
n_samples = 100

# Generate synthetic data for correlated groups

# Groceries features: create a base spend and add noise
groceries_base = np.random.uniform(20, 100, n_samples)
groceries_spend_avg_30d = groceries_base + np.random.normal(0, 5, n_samples)
groceries_spend_avg_90d = groceries_base * 3 + np.random.normal(0, 10, n_samples)

# Gambling features: create a base spend and add noise
gambling_base = np.random.uniform(0, 50, n_samples)
gambling_debit_avg_7d = gambling_base + np.random.normal(0, 2, n_samples)
gambling_debit_avg_180d = gambling_base * 2 + np.random.normal(0, 3, n_samples)

# An independent feature
other_spend = np.random.uniform(0, 200, n_samples)

# Create a DataFrame with these features
df = pd.DataFrame(
    {
        "groceries_spend_avg_30d": groceries_spend_avg_30d,
        "groceries_spend_avg_90d": groceries_spend_avg_90d,
        "gambling_debit_avg_7d": gambling_debit_avg_7d,
        "gambling_debit_avg_180d": gambling_debit_avg_180d,
        "other_spend": other_spend,
    }
)

# NMF requires non-negative data. If any values are negative due to noise, clip them.
df = df.clip(lower=0)

print("DataFrame head:")
print(df.head())

# Apply NMF for feature reduction.
# We choose n_components=2 expecting one component may capture groceries and one gambling.
n_components = 4
nmf_model = NMF(n_components=n_components, init="random", random_state=0)
W = nmf_model.fit_transform(
    df
)  # Reduced feature representation for samples (shape: [n_samples, n_components])
H = (
    nmf_model.components_
)  # Component loadings for original features (shape: [n_components, n_features])

print("\nW (reduced feature representation):")
print(W[:5])
print("\nH (component loadings):")
print(H)

# Examine the loadings for each component to see which features contribute most
feature_names = df.columns
for i, comp in enumerate(H):
    print(f"\nComponent {i}:")
    for feature, weight in zip(feature_names, comp):
        print(f"  {feature}: {weight:.4f}")
