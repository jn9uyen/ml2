"""
This script demonstrates a simple naive classifier using logistic regression
and computes the log loss for the model, a naive expected baseline, and a random
classifier. It also calculates the Log Loss Scoring System (LLSS) for each baseline.
"""

import pyspark

# import databricks_connect as dbc

print(pyspark.__version__)


# Re-import necessary libraries since the execution state was reset
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# Generate synthetic dataset
np.random.seed(42)
n_samples = 1000
X = np.random.rand(n_samples, 5)  # 5 random features
y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1 > 0.99).astype(
    int
)  # Target with some signal

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Model predictions
y_pred_prob = log_reg.predict_proba(X_test)[:, 1]

# Compute log loss for the model
log_loss_model = log_loss(y_test, y_pred_prob)

# Compute expected log loss (naive classifier that predicts class proportion)
p_baseline = np.mean(y_test)
expected_log_loss_baseline = -p_baseline * np.log(p_baseline) - (
    1 - p_baseline
) * np.log(1 - p_baseline)

# Generate random classifier predictions (uniform random between 0 and 1)
random_pred_prob = np.random.rand(len(y_test))

# Compute log loss for the random classifier
log_loss_random = log_loss(y_test, random_pred_prob)

# Simulate a naive classifier that always predicts the class proportion from training data
naive_pred_prob = np.full_like(y_test, np.mean(y_train), dtype=float)

# Compute log loss for the naive classifier
log_loss_naive = log_loss(y_test, naive_pred_prob)

# Compute LLSS for each baseline
llss_model_vs_expected = 1 - (log_loss_model / expected_log_loss_baseline)
llss_model_vs_naive_classifier = 1 - (log_loss_model / log_loss_naive)
llss_model_vs_random = 1 - (log_loss_model / log_loss_random)

# Display results including the naive classifier
results = {
    "Log Loss (Logistic Regression)": log_loss_model,
    "Log Loss (Naive Expected Baseline)": expected_log_loss_baseline,
    "Log Loss (Random Classifier)": log_loss_random,
    "Log Loss (Naive Classifier)": log_loss_naive,
    "LLSS (Model vs. Expected Log Loss)": llss_model_vs_expected,
    "LLSS (Model vs. Naive Classifier)": llss_model_vs_naive_classifier,
    "LLSS (Model vs. Random Classifier)": llss_model_vs_random,
}
print("\nResults:")
for key, value in results.items():
    print(f"{key}: {value:.4f}")
