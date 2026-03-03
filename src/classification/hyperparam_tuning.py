from typing import Union, Optional, Dict, List, Literal
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import xgboost as xgb

from helper import utils
from helper import logging_config

logger = logging_config.getLogger(__name__)


class HyperparameterTuner:

    def __init__(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        tuning_iters: int,
        algo: Literal["xgboost", "lightgbm"] = "xgboost",
        class_weight: Optional[
            Union[Dict, List[Dict], Literal["balanced"], None]
        ] = "balanced",
        importance_type: Literal[
            "weight", "gain", "cover", "total_gain", "total_cover"
        ] = "total_gain",
        missing: Optional[float] = np.nan,
        cross_val_method: Literal["cv", "time-series", None] = "cv",
        cross_val_folds: int = 5,
        scoring_metrics: List[str] = ["roc_auc"],
        **kwargs,
    ):
        self.features = features
        self.target = target
        self.tuning_iters = tuning_iters
        self.algo = algo
        self.class_weight = class_weight
        self.importance_type = importance_type
        self.missing = missing
        self.cross_val_method = cross_val_method
        self.cross_val_folds = cross_val_folds
        self.scoring_metrics = scoring_metrics
        self.kwargs = kwargs

        self.param_grid: dict = None
        self.cv: object = None
        self.model: object = None
        self.best_hyperparams: dict = None
        self.best_metric_scores: dict = None

    def _run_xgboost(self) -> None:
        # https://xgboost.readthedocs.io/en/stable/python/python_api.html
        self.param_grid = {
            "n_estimators": [100, 200, 300, 350, 400, 450, 500],
            # "n_estimators": [20, 50, 100, 150, 200],
            "max_depth": [1, 2, 3, 4, 5],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "gamma": [0, 0.1, 0.2, 0.3],
            "min_child_weight": [1, 3, 5],
            "reg_alpha": [0, 0.01, 0.1, 1, 10],  # L1 regularization on weights
            "reg_lambda": [0, 0.01, 0.1, 1, 10],  # L2 regularization on weights
        }
        sample_weights = sklearn.utils.class_weight.compute_sample_weight(
            self.class_weight, self.target
        )
        self.model = xgb.XGBClassifier(
            importance_type=self.importance_type,
            missing=self.missing,
        ).fit(self.features, self.target, sample_weight=sample_weights)

    def tune(self) -> None:
        """
        Tune hyperparameters using RandomizedSearchCV. Scoring methods:
        https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        """
        match self.cross_val_method:
            case "cv":
                self.cv = self.cross_val_folds
            case "time-series":
                self.cv = TimeSeriesSplit(n_splits=self.cross_val_folds)
            case _:
                self.cv = None

        kwargs = utils.filter_kwargs(RandomizedSearchCV.__init__, self.kwargs)
        rscv = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.param_grid,
            n_iter=self.tuning_iters,
            cv=self.cv,
            scoring=self.scoring_metrics,
            refit=self.scoring_metrics[0],
            **kwargs,
            # verbose=2,
            # n_jobs=-1,
            # random_state=42,
        )
        logger.info("Tuning hyperparameters...")
        rscv.fit(self.features, self.target)

        self.model = rscv.best_estimator_
        self.best_hyperparams = rscv.best_params_
        self.best_metric_scores = {
            metric: [
                rscv.cv_results_[f"mean_test_{metric}"][rscv.best_index_],
                rscv.cv_results_[f"std_test_{metric}"][rscv.best_index_],
            ]
            for metric in self.scoring_metrics
        }
        logger.info(f"Best hyperparameters:\n{self.best_hyperparams}")
        for metric, score in self.best_metric_scores.items():
            logger.info(
                f"Best score [mean, SD]: {metric}: [{score[0]:.4f}, {score[1]:.4f}]"
            )

    def run(self) -> None:
        match self.algo:
            case "xgboost":
                self._run_xgboost()
            case "lightgbm":
                raise NotImplementedError
            case _:
                raise ValueError(f"Invalid algorithm: {self.algo}")

        self.tune()
