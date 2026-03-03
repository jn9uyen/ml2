import warnings

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split


def train_lgbm_with_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring_metric: str | list[str] = "roc_auc",
    eval_metric: str | list[str] | None = None,
    refit_metric: str | None = None,
    scale_pos_weight: float | str | None = 1.0,
    cv_folds: int = 3,
    val_size: float = 0.1,
    early_stopping_rounds: int | None = 20,
    param_dist: dict | None = None,
    random_state: int | None = 42,
    verbose: int = 1,
    importance_type: str = "gain",
) -> GridSearchCV:
    """
    Train a LightGBM classifier with GridSearchCV and early stopping.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    scoring_metric : str, optional
        Scoring metric to optimize for.
        Can be 'roc_auc' or 'average_precision'.
        Defaults to 'roc_auc'.
    eval_metric : str | list[str] | None, optional
        Evaluation metric(s) to be monitored during training for early stopping.
    refit_metric : str | None, optional
        The specific metric from `scoring_metric` to use for refitting the best model.
        If None and `scoring_metric` is a list, a ValueError will occur unless
        `refit` is explicitly set to False. If `scoring_metric` is a single string,
        this is not needed.
    scale_pos_weight : float | str | None, optional
        Class weight adjustment to handle imbalanced datasets.
        If set to "balanced", it is computed as `negative_class / positive_class`.
    cv_folds : int, optional
        Number of cross-validation folds.
        Defaults to 3.
    val_size : float, optional
        Proportion of the training data to use for validation for early stopping.
        Defaults to 0.1.
    early_stopping_rounds : int | None, optional
        Number of boosting rounds with no improvement before early stopping.
    param_dist : dict | None, optional
        Distribution of hyperparameters to sample from.
    random_state : int | None, optional
        Random seed for reproducibility.
    verbose : int, optional
        Controls the verbosity: the higher, the more messages.
        - 0: no messages are printed.
        - >1: the computation time for each fold and parameter candidate is displayed.
        - >2: the score is also displayed.
        - >3: the fold and candidate parameter indexes are also displayed together
        with the starting time of the computation.
        Defaults to 1.
    importance_type : str, optional
        The type of feature importance to be filled into `feature_importances_`.
        If 'split', result contains numbers of times the feature is used in a model.
        If 'gain', result contains total gains of splits which use the feature.
        Defaults to "split".

    Returns
    -------
    GridSearchCV
        The trained GridSearchCV object.
    """
    if isinstance(scale_pos_weight, str) and scale_pos_weight.lower() == "balanced":
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Create a validation set for early stopping.
    # Note: This split is *only* for the `eval_set` in `grid_search.fit`.
    # GridSearchCV itself will perform its own cross-validation on `X_subtrain`.
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        stratify=y_train,
        random_state=random_state,
    )

    lgbm = lgb.LGBMClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        importance_type=importance_type,
    )

    if param_dist is None:
        param_dist = {
            "n_estimators": [2000],
            "max_depth": [3, 5, 7],
            "num_leaves": [8, 32],
            "learning_rate": [0.01, 0.05, 0.1],
        }

    # Determine the refit parameter for GridSearchCV.
    grid_search_refit = True
    if isinstance(scoring_metric, list):
        if refit_metric is None:
            warnings.warn(
                "When using multiple scoring metrics (scoring_metric is a list), "
                "you should specify which metric to use for refitting by setting "
                "`refit_metric`. Alternatively, set `refit=False` in GridSearchCV if "
                "refitting is not needed. Defaulting to first scoring metric: "
                f"{scoring_metric[0]}"
            )
            grid_search_refit = scoring_metric[0]
        else:
            # Check if the specified refit_metric is actually in the scoring list.
            if refit_metric not in scoring_metric:
                raise ValueError(
                    f"refit_metric '{refit_metric}' is not found in the provided "
                    f"scoring_metric list: {scoring_metric}."
                )
            grid_search_refit = refit_metric

    grid_search = GridSearchCV(
        estimator=lgbm,
        param_grid=param_dist,
        scoring=scoring_metric,
        cv=cv_folds,
        n_jobs=-1,
        verbose=verbose,
        return_train_score=True,
        refit=grid_search_refit,
    )

    # Ensure eval_metric is provided if scoring_metric is a list.
    if eval_metric is None:
        if isinstance(scoring_metric, list):
            eval_metric = scoring_metric[0]  # Default to the first metric.
        else:
            eval_metric = scoring_metric

    early_stopping_callback = lgb.early_stopping(
        stopping_rounds=early_stopping_rounds,
        verbose=True,
        min_delta=0.001,
    )
    log_evaluation_callback = lgb.log_evaluation(period=50)

    # Pass the eval_set and eval_metric to the fit method.
    grid_search.fit(
        X_subtrain,
        y_subtrain,
        eval_set=[(X_val, y_val)],
        eval_metric=eval_metric,  # This metric is used by the callbacks.
        callbacks=[early_stopping_callback, log_evaluation_callback],
    )
    return grid_search
