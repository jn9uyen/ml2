"""
Modelling pipeline
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import is_classifier

import data_cleaning as cln
from classification.ml_preprocessing import (
    train_test_split_by_time,
    FeatureEngineering,
    FeatureGenerator,
)
from feature_selection import FeatureSelector
from hyperparam_tuning import HyperparameterTuner
from evaluation import Evaluator, predict_probabilities
from helper import utils
from helper import logging_config

logger = logging_config.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
model_folder = f"{current_dir}/models"

if not os.path.exists(model_folder):
    os.makedirs(model_folder)


class ModelPipeline:
    """
    Model pipeline for training and testing. The steps are:
    1. Split train and test data
    2. Preprocess data
        a. Engineer (new) features
        b. Encode and generate features
    3. Select features
    4. Tune hyperparameters
    5. Train model
    6. Calibrate model
    7. Evaluate model
    """

    def __init__(
        self,
        pdf: pd.DataFrame,
        tgt_col: str,
        inference: bool = False,
        col_transformer: ColumnTransformer = None,
        selected_feature_names: list = None,
        model: object = None,
        excl_feature_cols: list = [],
        seasonal_date_cols: list = [],
        split_method: str = "by-time",
        is_only_feature_selection: bool = False,
        is_calibrate_model: bool = False,
        fitted_objects: dict = dict(
            col_transformer=f"{model_folder}/col_transformer.pkl",
            selected_feature_names=f"{model_folder}/selected_feature_names.pkl",
            model=f"{model_folder}/model.pkl",
        ),
        **kwargs,
    ):
        self.pdf = pdf
        self.tgt_col = tgt_col
        self.inference = inference
        self.excl_feature_cols = excl_feature_cols
        self.seasonal_date_cols = seasonal_date_cols
        self.split_method = split_method
        self.is_only_feature_selection = is_only_feature_selection
        self.is_calibrate_model = is_calibrate_model
        self.fitted_objects = fitted_objects
        self.kwargs = kwargs

        if self.inference:
            assert (
                col_transformer is not None
            ), "col_transformer is required for inference"
            assert (
                selected_feature_names is not None
            ), "selected_feature_names is required for inference"
            assert model is not None, "model is required for inference"

            self.col_transformer: ColumnTransformer = col_transformer
            self.selected_feature_names: list = selected_feature_names
            self.model: object = model

        else:
            self.col_transformer: ColumnTransformer = None
            self.selected_feature_names: list = None
            self.model: object = None

        self.xtrain: pd.DataFrame = None
        self.xtest: pd.DataFrame = None
        self.ytrain: pd.Series = None
        self.ytest: pd.Series = None
        self.cv: object = None
        self.predictions: pd.Series = None

    def clean_data(self) -> None:
        self.pdf = cln.clean_missing_values(self.pdf)
        self.pdf = cln.clean_dollar_cols(self.pdf)
        self.pdf = cln.clean_to_binary_cols(self.pdf)
        self.pdf = cln.convert_date_cols(self.pdf)
        self.pdf = cln.clean_special_chars(self.pdf)

    def split_train_test(self) -> None:
        x = self.pdf.drop(columns=self.tgt_col)
        y = self.pdf[self.tgt_col]

        if self.split_method == "by-time":
            if "split_date_col" in self.kwargs:
                self.kwargs["date_col"] = self.kwargs["split_date_col"]

            kwargs = utils.filter_kwargs(train_test_split_by_time, self.kwargs)
            self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split_by_time(
                x, y, **kwargs
            )
        else:
            kwargs = utils.filter_kwargs(train_test_split, self.kwargs)
            self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(
                x, y, **kwargs
            )

    def preprocess_data(self) -> None:
        """
        Feature engineering, categorical encoding and feature generation.
        """

        def process(pdf: pd.DataFrame, inference: bool = True) -> pd.DataFrame:

            feateng = FeatureEngineering(pdf, self.seasonal_date_cols)
            feateng.run()
            pdf = feateng.pdf

            featgen = FeatureGenerator(
                pdf, self.excl_feature_cols, self.col_transformer
            )
            featgen.run(inference, **self.kwargs)
            return featgen.pdf, featgen.col_transformer

        if not self.inference:
            self.xtrain, self.col_transformer = process(self.xtrain, self.inference)
            self.xtest, _ = process(self.xtest, inference=True)
        else:
            self.pdf, _ = process(self.pdf)

    def select_features(self) -> None:
        """
        Select features using XGBoost feature importances
        """
        if not self.inference:
            kwargs = utils.filter_kwargs(FeatureSelector.__init__, self.kwargs)
            feat_sel = FeatureSelector(
                self.xtrain,
                self.ytrain,
                algo="xgboost",
                class_weight="balanced",
                **kwargs,
            )
            feat_sel.run()
            self.selected_feature_names = feat_sel.selected_feature_names
            self.xtrain = self.xtrain[self.selected_feature_names]
            self.xtest = self.xtest[self.selected_feature_names]
        else:
            self.pdf = self.pdf[self.selected_feature_names]

    def tune_hyperparameters(self) -> None:
        """
        Tune hyperparameters
        """
        tuner = HyperparameterTuner(self.xtrain, self.ytrain, **self.kwargs)
        tuner.run()
        self.model = tuner.model
        self.cv = tuner.cv

    def calibrate_model(self) -> None:
        """
        Calibrate classifier model
        TODO: doesn't work with XGBoost because of int32 dtype clash:
        https://notebook.community/ethen8181/machine-learning/model_selection/prob_calibration/prob_calibration
        """
        if self.is_calibrate_model and is_classifier(self.model):
            if "calibration_method" in self.kwargs:
                self.kwargs["method"] = self.kwargs["calibration_method"]

            kwargs = utils.filter_kwargs(CalibratedClassifierCV.__init__, self.kwargs)
            self.model = CalibratedClassifierCV(self.model, cv=self.cv, **kwargs)
            self.model.fit(self.xtrain, self.ytrain)
            logger.info("Model has been calibrated.")
        else:
            logger.info("Calibration is disabled or model is not a classifier.")

def evaluate_model(self) -> None:

    kwargs = utils.filter_kwargs(Evaluator.__init__, self.kwargs)
    dataset_names = {
        "train": [self.xtrain, self.ytrain],
        "test": [self.xtest, self.ytest]
    }

    for dataset_name, (x, y) in dataset_names.items():
        eval = Evaluator(self.model, x, y, **kwargs)
        eval.eval_predictive_performance(dataset_name)

def explain_model(self) -> None:
    kwargs = utils.filter_kwargs(Evaluator.__init__, self.kwargs)
    eval = Evaluator(self.model, self.xtrain, self.ytrain, **kwargs)
    eval.explain_model()

    def run(self):
        """
        Run the model pipeline
        """
        if not self.inference:
            logger.info("Running model training pipeline...")
            self.clean_data()
            self.split_train_test()
            self.preprocess_data()
            self.select_features()

            if self.is_only_feature_selection:
                logger.info(
                    "Model training pipeline completed up to feature selection."
                )
                return

            self.tune_hyperparameters()
            # self.calibrate_model()
            self.evaluate_model()

            # Save column transformer, selected features, model
            with open(self.fitted_objects["col_transformer"], "wb") as f:
                pickle.dump(self.col_transformer, f)
            with open(self.fitted_objects["selected_feature_names"], "wb") as f:
                pickle.dump(self.selected_feature_names, f)
            with open(self.fitted_objects["model"], "wb") as f:
                pickle.dump(self.model, f)

            logger.info("Model training pipeline completed.")
        else:
            logger.info("Running model inference pipeline...")

            self.pdf = self.pdf.drop(columns=self.tgt_col, errors="ignore")
            self.clean_data()
            self.preprocess_data()
            self.select_features()

            if is_classifier(self.model):
                self.predictions = predict_probabilities(self.model, self.pdf)
            logger.info("Model inference pipeline completed.")
