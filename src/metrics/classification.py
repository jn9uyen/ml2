from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import auc, average_precision_score, log_loss, roc_auc_score


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_name: Literal["train", "val", "test"],
    y_train_mean: float,
) -> dict:
    """
    Compute classification metrics for a given dataset.

    Parameters
    ----------
    y_true : np.ndarray
        The true target values.
    y_pred : np.ndarray
        The predicted target values.
    dataset_name : Literal["train", "val", "test"]
        The name of the dataset (e.g., "train", "val", "test").
    y_train_mean : float
        The mean of the training target values; used to compute the skill score.

    Returns
    -------
    dict
        A dictionary of model evaluation metrics.

    Examples
    --------
    >>> y_train = pd.Series([0, 1, 0, 1])
    >>> y_pred_train = np.array([0.1, 0.9, 0.2, 0.8])
    >>> compute_model_metrics("train", y_train, y_pred_train, y_train.mean())
    {'size': 4, 'log_loss': 0.3465735902799726, 'log_loss_skill': None, 'roc_auc': 1.0,
    'average_precision': 1.0}
    """
    if dataset_name == "train":
        log_loss_skill = None
    else:
        baseline_pred = [y_train_mean] * len(y_true)
        baseline_log_loss = log_loss(y_true, baseline_pred)
        log_loss_skill = 1 - (log_loss(y_true, y_pred) / baseline_log_loss)

    return {
        "size": len(y_true),
        "log_loss": log_loss(y_true, y_pred),
        "log_loss_skill": log_loss_skill,
        "roc_auc": roc_auc_score(y_true, y_pred),
        "average_precision": average_precision_score(y_true, y_pred),
    }


def margin_auc_score(
    y_true: ArrayLike,
    y_pred_prob: ArrayLike,
    loan_amounts: ArrayLike,
    tp_weight: float,
    tn_weight: float,
    fp_weight: float,
    fn_weight: float,
) -> float:
    """
    Compute a custom "margin AUC" by evaluating expected gain/loss across thresholds,
    similar to ROC AUC, but using business-weighted margins.

    Parameters
    ----------
    y_true : array-like of int
        Binary target values (0 = no default, 1 = default).
    y_pred_prob : array-like of float
        Predicted probabilities of default.
    loan_amounts : array-like of float
        Loan amounts corresponding to each sample.
    tp_weight, tn_weight, fp_weight, fn_weight : float
        Weight multipliers for each outcome.

    Returns
    -------
    float
        Margin AUC: Area under the gain-threshold curve, normalized to [0, 1].
    """
    # Ensure numpy arrays
    y_true = np.asarray(y_true)
    y_pred_prob = np.asarray(y_pred_prob)
    loan_amounts = np.asarray(loan_amounts)

    # Use dense threshold grid if model output is coarse
    unique_preds = np.unique(y_pred_prob)
    if len(unique_preds) < 100:
        thresholds = np.linspace(0.0, 1.0, 1000)
    else:
        thresholds = np.sort(unique_preds)
        thresholds = np.append(thresholds, 1.01)  # include full 1.0 cutoff

    # Preallocate margin array
    margins = np.empty_like(thresholds)

    # Vectorized computation
    for i, thresh in enumerate(thresholds):
        y_pred = y_pred_prob >= thresh

        tp_mask = (y_true == 1) & y_pred
        tn_mask = (y_true == 0) & ~y_pred
        fp_mask = (y_true == 0) & y_pred
        fn_mask = (y_true == 1) & ~y_pred

        margin = (
            loan_amounts[tp_mask].sum() * tp_weight
            + loan_amounts[tn_mask].sum() * tn_weight
            + loan_amounts[fp_mask].sum() * fp_weight
            + loan_amounts[fn_mask].sum() * fn_weight
        )
        margins[i] = margin

    # Normalize margins to [0, 1] and compute AUC.
    margins -= margins.min()
    margins /= margins.ptp() + 1e-12
    normalized_thresholds = np.linspace(0.0, 1.0, len(margins))

    return float(auc(thresholds, margins))
