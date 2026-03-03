from typing import Union, Optional, Dict, List, Literal
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import xgboost as xgb

import evaluation
from helper import logging_config

logger = logging_config.getLogger(__name__)


class FeatureSelector:
    """
    Feature selection using xgboost or lightgbm feature importances. Features are pruned
    based on any of the following criteria:
    1. Top n features
    2. Relative importance threshold
    3. Cumulative relative importance threshold
    """

    def __init__(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        algo: Literal["xgboost", "lightgbm"] = "xgboost",
        class_weight: Union[Dict, List[Dict], Literal["balanced"], None] = "balanced",
        importance_type: Literal[
            "weight", "gain", "cover", "total_gain", "total_cover"
        ] = "total_gain",
        missing: Optional[float] = np.nan,
        top_n_features: Optional[int] = None,
        rel_imp_thres: Optional[float] = None,
        cumulative_thres: Optional[float] = None,
        early_stopping_rounds: int = 10,
        validation_size: float = 0.2,
    ):
        self.features = features
        self.target = target
        self.algo = algo
        self.class_weight = class_weight
        self.importance_type = importance_type
        self.missing = missing
        self.top_n_features = top_n_features
        self.rel_imp_thres = rel_imp_thres
        self.cumulative_thres = cumulative_thres
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_size = validation_size

        self.model: object = None
        self.rel_feat_imp: pd.DataFrame = None
        self.selected_feature_names: list = None

    # TODO: surface hyperparameters
    def _run_xgboost(self) -> None:

        # Split data into training and validation sets
        xtrain, xval, ytrain, yval = train_test_split(
            self.features, self.target, test_size=self.validation_size, random_state=42
        )
        eval_set = [(xval, yval)]
        sample_weights = sklearn.utils.class_weight.compute_sample_weight(
            self.class_weight, ytrain
        )

        # https://xgboost.readthedocs.io/en/stable/python/python_api.html
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=3,
            colsample_bytree=0.9,
            reg_alpha=0.01,  # L1
            reg_lambda=0.01,  # L2
            eval_metric=["logloss", "map", "auc"],  # map: mean avg precision
            importance_type=self.importance_type,
            missing=self.missing,  # cannot be pandas.NA, should be numpy.nan
            early_stopping_rounds=self.early_stopping_rounds,
        ).fit(
            xtrain,
            ytrain,
            sample_weight=sample_weights,
            eval_set=eval_set,
            verbose=False,
        )

    def _run_lightgbm(self) -> None:
        # sample_weights = sklearn.utils.class_weight.compute_sample_weight(
        #     self.class_weight, self.target
        # )

        # # https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
        # self.model = lgb.LGBMClassifier(
        #     n_estimators=200,
        #     max_depth=3,
        #     colsample_bytree=0.9,
        #     reg_alpha=0.01,  # L1
        #     reg_lambda=0.01,  # L2
        #     metric=["binary_logloss", "auc"],  # log loss and AUC
        #     importance_type="gain",  # "split", "gain"
        # ).fit(self.features, self.target, sample_weight=sample_weights)
        raise NotImplementedError

    def _select_features_by_importance(self) -> None:
        self.rel_feat_imp = evaluation.calc_relative_feature_importances(
            self.model, self.features.columns
        )
        logger.info(f"Initial number of features: {len(self.rel_feat_imp)}")

        if self.top_n_features is not None:
            self.rel_feat_imp = self.rel_feat_imp.nlargest(
                self.top_n_features, "importance"
            )
            logger.info(
                f"Number of features after selecting top {self.top_n_features} "
                f"by importance: {len(self.rel_feat_imp)}"
            )

        if self.rel_imp_thres is not None:
            self.rel_feat_imp = self.rel_feat_imp[
                self.rel_feat_imp["importance"] >= self.rel_imp_thres
            ]
            logger.info(
                "Number of features after applying importance threshold of "
                f"{self.rel_imp_thres}: {len(self.rel_feat_imp)}"
            )

        if self.cumulative_thres is not None:
            self.rel_feat_imp = self.rel_feat_imp.sort_values(
                by="importance", ascending=False
            )
            self.rel_feat_imp["cumulative_imp_pct"] = self.rel_feat_imp[
                "importance"
            ].cumsum()
            self.rel_feat_imp["cumulative_imp_pct"] /= self.rel_feat_imp[
                "importance"
            ].sum()
            self.rel_feat_imp = self.rel_feat_imp[
                self.rel_feat_imp["cumulative_imp_pct"] <= self.cumulative_thres
            ]
            logger.info(
                "Number of features after applying cumulative importance threshold of "
                f"{self.cumulative_thres}: {len(self.rel_feat_imp)}"
            )

        self.selected_feature_names = self.rel_feat_imp["feature"].values

    def run(self) -> None:
        """
        Select features
        """
        match self.algo:
            case "xgboost":
                self._run_xgboost()
            case "lightgbm":
                raise NotImplementedError
            case _:
                raise ValueError(f"Invalid algorithm: {self.algo}")

        self._select_features_by_importance()
