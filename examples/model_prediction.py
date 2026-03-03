import pandas as pd
import xgboost as xgb
from typing import Union


# def generate_predictions_df(
#     model: xgb.XGBClassifier,
#     x: pd.DataFrame,
#     y: Union[pd.Series, pd.DataFrame],
#     dataset_label: str = "train",
# ) -> pd.DataFrame:
#     """
#     Generate a DataFrame containing actual labels and model predictions.
#     """
#     y_df = y.to_frame() if isinstance(y, pd.Series) else y
#     df_pred = pd.DataFrame(model.predict_proba(x), index=y.index)

#     output = y_df.join(df_pred).reset_index()
#     output.insert(1, "dataset", dataset_label)
#     # output.columns = ["id", "dataset", "target", "prediction"]

#     return output


def generate_predictions(
    model: xgb.XGBClassifier,
    x: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    dataset_label: str = "train",
) -> pd.DataFrame:
    """
    Generate a DataFrame containing actual labels and model predictions,
    handling both binary and multi-class classification outputs.

    Parameters
    ----------
    model : xgb.XGBClassifier
        The trained XGBoost classifier model.
    x : pd.DataFrame
        The input features for prediction.
    y : Union[pd.Series, pd.DataFrame]
        The true labels.
    dataset_label : str, default="train"
        A label for the dataset (e.g., 'train', 'validation').

    Returns
    -------
    pd.DataFrame
        A DataFrame ready for the UnifiedClassifierMetrics class.
    """
    # Ensure y is a DataFrame to get the target column name
    y_df = y.to_frame() if isinstance(y, pd.Series) else y
    target_col_name = y_df.columns[0]

    y_pred_proba = model.predict_proba(x)

    if model.n_classes_ == 2:
        df_pred = pd.DataFrame(
            y_pred_proba[:, 1], index=y.index, columns=["prediction"]
        )
    else:
        pred_cols = [f"pred_{i}" for i in range(model.n_classes_)]
        df_pred = pd.DataFrame(y_pred_proba, index=y.index, columns=pred_cols)

    output = y_df.join(df_pred).reset_index()

    if output.columns[0] == "index":
        output = output.rename(columns={"index": "id"})

    output.insert(1, "dataset", dataset_label)

    return output
