from typing import List, Optional, Union
import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from sklearn import metrics as sklmetrics


def log_loss_skill_score(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    """
    Compute the Log Loss Skill Score (LLSS) for binary or multi-class models.

    The skill score measures the improvement over a naive baseline model that
    always predicts the marginal class probabilities. The baseline log loss is
    the entropy of the class distribution.
    """
    classes = np.unique(y_true)

    # Calculate the model's log loss
    model_log_loss = sklmetrics.log_loss(y_true, y_pred_prob, labels=classes)

    # Calculate the baseline log loss (entropy of the target variable)
    class_proportions = np.bincount(y_true) / len(y_true)
    # Filter out classes with zero probability to avoid log(0)
    non_zero_proportions = class_proportions[class_proportions > 0]
    baseline_logloss = -np.sum(non_zero_proportions * np.log(non_zero_proportions))

    if baseline_logloss == 0:
        # This can happen in a degenerate case where there is only one class
        return 0.0 if model_log_loss == 0 else -np.inf

    return 1 - (model_log_loss / baseline_logloss)


class ClassifierMetrics:
    def __init__(
        self,
        predictions: pd.DataFrame,
        dataset_col: str,
        target_col: str,
        prediction_cols: Union[str, List[str]],
        objective: Optional[str] = None,
    ):
        """
        A unified class to calculate metrics for binary or multi-class classification.

        Parameters
        ----------
        predictions : pd.DataFrame
            DataFrame with dataset, target, and prediction probability columns.
        dataset_col : str
            Column name specifying dataset groups (e.g., 'train', 'val').
        target_col : str
            Column name for the actual target values.
        prediction_cols : Union[str, List[str]]
            - For binary: The single column name for the positive class probability.
            - For multi-class: A list of column names for class probabilities.
        objective : Optional[str], default=None
            - "binary:logitraw": If binary predictions are raw logits.
            - "multi:softmax": If multi-class predictions are raw logits.
        """
        self.predictions = predictions.copy()
        self.dataset_col = dataset_col
        self.target_col = target_col
        self.prediction_cols = prediction_cols

        self.classes_ = np.sort(self.predictions[self.target_col].unique())
        self.is_binary = len(self.classes_) == 2

        # --- Input Validation and Preprocessing ---
        if self.is_binary:
            if not isinstance(prediction_cols, str):
                raise ValueError(
                    "For binary classification, `prediction_cols` must be a single string."
                )
            self.positive_class_pred_col = prediction_cols
            if objective == "binary:logitraw":
                self.predictions[self.positive_class_pred_col] = expit(
                    self.predictions[self.positive_class_pred_col]
                )
        else:  # Multi-class
            if not isinstance(prediction_cols, list) or len(prediction_cols) != len(
                self.classes_
            ):
                raise ValueError(
                    f"For multi-class, `prediction_cols` must be a list of length {len(self.classes_)}."
                )
            if objective == "multi:softmax":
                raw_preds = self.predictions[self.prediction_cols].values
                self.predictions[self.prediction_cols] = softmax(raw_preds, axis=1)
            elif objective == "multi:softprob":
                # Predictions are already probabilities, no transformation needed.
                pass

        self.metrics_df = None

    def compute_metrics(self) -> pd.DataFrame:
        """
        Compute a consistent set of classification metrics for each dataset group.
        """

        def compute_group_metrics(group):
            y_true = group[self.target_col]

            if self.is_binary:
                y_pred_prob_pos = group[self.positive_class_pred_col].values
                # Create a 2D array for multi-class metric compatibility
                y_pred_prob = np.vstack([1 - y_pred_prob_pos, y_pred_prob_pos]).T
                y_pred_label = (y_pred_prob_pos >= 0.5).astype(int)
            else:  # Multi-class
                y_pred_prob = group[self.prediction_cols].values
                y_pred_label = np.argmax(y_pred_prob, axis=1)

            # --- Compute Consistent Metrics ---
            logloss = sklmetrics.log_loss(y_true, y_pred_prob, labels=self.classes_)
            llss = log_loss_skill_score(y_true, y_pred_prob)
            accuracy = sklmetrics.accuracy_score(y_true, y_pred_label)

            # Use weighted averages for robust, consistent reporting
            f1_avg = sklmetrics.f1_score(
                y_true, y_pred_label, average="weighted", zero_division=0
            )
            precision_avg = sklmetrics.precision_score(
                y_true, y_pred_label, average="weighted", zero_division=0
            )
            recall_avg = sklmetrics.recall_score(
                y_true, y_pred_label, average="weighted", zero_division=0
            )

            # ROC AUC Score
            if self.is_binary:
                roc_auc_avg = sklmetrics.roc_auc_score(y_true, y_pred_prob[:, 1])
                brier_score = sklmetrics.brier_score_loss(y_true, y_pred_prob[:, 1])
            else:
                # ovr: one-vs-rest | ovo: one-vs-one
                roc_auc_avg = sklmetrics.roc_auc_score(
                    y_true, y_pred_prob, multi_class="ovr", average="weighted"
                )
                y_true_one_hot = pd.get_dummies(y_true, columns=self.classes_).values
                brier_score = sklmetrics.brier_score_loss(
                    y_true_one_hot.ravel(), y_pred_prob.ravel()
                )

            return pd.Series(
                {
                    "count": len(y_true),
                    "logloss": logloss,
                    "llss": llss,
                    "accuracy": accuracy,
                    "roc_auc_avg": roc_auc_avg,
                    "f1_avg": f1_avg,
                    "precision_avg": precision_avg,
                    "recall_avg": recall_avg,
                    "brier_score": brier_score,
                }
            )

        self.metrics_df = self.predictions.groupby(
            self.dataset_col, as_index=False
        ).apply(compute_group_metrics)
        return self.metrics_df

    def compare_metrics(self, metric_name: str, group1: str, group2: str) -> float:
        """
        Compute the difference in a selected metric between two dataset groups.
        """
        if self.metrics_df is None:
            self.compute_metrics()

        val1 = self.metrics_df.loc[
            self.metrics_df[self.dataset_col] == group1, metric_name
        ].values
        val2 = self.metrics_df.loc[
            self.metrics_df[self.dataset_col] == group2, metric_name
        ].values

        if len(val1) == 0 or len(val2) == 0:
            raise ValueError(f"Dataset groups '{group1}' or '{group2}' not found.")

        return val1[0] - val2[0]
