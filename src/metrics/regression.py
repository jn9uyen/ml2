from typing import Literal

import numpy as np
from scipy.stats import sem
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)


def symmetric_mean_absolute_percentage_error(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """
    Calculate the Symmetric Mean Absolute Percentage Error (sMAPE).

    sMAPE is an alternative to MAPE that is less sensitive to outliers and is
    symmetric, meaning it doesn't penalize negative errors more than positive ones.
    Values range from 0.0 to 2.0.
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    # Create a mask to handle cases where both true and pred are zero.
    mask = denominator > 0

    if not np.any(mask):
        # Handle case where all denominators are zero.
        return 0.0

    smape = np.mean(numerator[mask] / denominator[mask])
    return float(smape)


def median_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Median Absolute Percentage Error (MdAPE).

    MdAPE is a robust alternative to MAPE, using the median instead of the mean, which
    makes it less sensitive to extreme outliers.
    """
    # Create a mask to avoid division by zero.
    mask = y_true != 0

    if not np.any(mask):
        return np.nan  # Or 0.0, depending on desired behavior.

    mdape = np.median(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    return float(mdape)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_name: Literal["train", "val", "test"],
    y_train_mean: float,
) -> dict:
    """
    Compute regression metrics for model evaluation.
    - size: size of data.
    - sample_mean: The mean of the true values. Units: Same as the original data.
    - sample_std: The standard deviation of the true values, indicating data spread.
        Units: Same as the original data.
    - sample_mad: The Mean Absolute Deviation of the true values, a robust measure of
        variability. Units: Same as the original data.
    - sample_mean_se: The standard error of the mean for y_true, indicating the
        precision of the sample mean. Units: Same as the original data.
    - prediction_se: The Standard Error of the Prediction. This metric quantifies the
        uncertainty or confidence around a single forecast made by the model. It
        accounts for both the model's parameter uncertainty and the inherent randomness
        in the data. Units: Same as the original data.
    - rmse: Root Mean Squared Error. Measures the average magnitude of the errors,
        giving higher weight to large errors. Units: Same as the original data.
    - mae: Mean Absolute Error. The average absolute difference between the predicted
        and true values. Units: Same as the original data.
    - mse_skill: The percentage improvement in MSE over a naive model that always
        predicts the mean. A score of 1 is perfect, 0 means no improvement.
        Set to None for the training set.
    - mae_skill: The percentage improvement in MAE over a naive model that always
        predicts the mean. A score of 1 is perfect, 0 means no improvement.
        Set to None for the training set.
    - mape: Mean Absolute Percentage Error. The average absolute error expressed as a
        percentage of the true values. Units: Percentage (%).
    - smape: Symmetric Mean Absolute Percentage Error. A percentage error metric that
        is less biased than MAPE, especially for small true values.
        Units: Percentage (%).
    - mdape: Median Absolute Percentage Error. A robust version of MAPE that uses the
        median, making it less sensitive to outliers. Units: Percentage (%).

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
    """
    if dataset_name == "train":
        mse_skill = None
        mae_skill = None
    else:
        # Model errors.
        mse_model = root_mean_squared_error(y_true, y_pred) ** 2
        mae_model = mean_absolute_error(y_true, y_pred)

        # Naive baseline errors.
        mse_naive = np.mean((y_true - y_train_mean) ** 2)
        mae_naive = np.mean(np.abs(y_true - y_train_mean))  # mean abs deviation.

        mse_skill = 1 - (mse_model / mse_naive) if mse_naive > 0 else 0.0
        mae_skill = 1 - (mae_model / mae_naive) if mae_naive > 0 else 0.0

    return {
        "size": len(y_true),
        "sample_mean": np.mean(y_true),
        "sample_std": np.std(y_true),
        "sample_mad": np.mean(np.abs(y_true - np.mean(y_true))),
        "sample_mean_se": sem(y_true),
        "prediction_se": 0,  # TODO
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mse_skill": mse_skill,
        "mae_skill": mae_skill,
        "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "smape": symmetric_mean_absolute_percentage_error(y_true, y_pred) * 100,
        "mdape": median_absolute_percentage_error(y_true, y_pred) * 100,
    }
