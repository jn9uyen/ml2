import json
import sys
from itertools import cycle
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from cl.model.credit_risk.train import model_utils
from cl.model.credit_risk.train.metrics import ClassifierMetrics
from sklearn.model_selection import ParameterSampler, StratifiedKFold, train_test_split
from tqdm import tqdm
from xgboost import XGBClassifier

pd.set_option("display.max_columns", 20)


def report_best_round(
    model: XGBClassifier, min_boosting_rounds: int = 0, verbose: bool = True
) -> Tuple[int, Optional[float], float]:
    """
    Report the best iteration and scores for both training and validation sets.
    Handles different eval_metrics (e.g., 'auc', 'mlogloss').

    Returns
    -------
    Tuple[int, Optional[float], float]
        Best round, train score (if available), and validation score.
    """
    best_round = model.get_booster().best_iteration + 1
    best_round_continued_model = best_round - min_boosting_rounds
    evals_result = model.evals_result()
    eval_keys = list(evals_result.keys())

    metric_name = model.get_params().get("eval_metric")

    best_train_score = None
    if len(eval_keys) > 1:
        train_key, val_key = eval_keys
        best_train_score = evals_result[train_key][metric_name][
            best_round_continued_model - 1
        ]
        best_val_score = evals_result[val_key][metric_name][
            best_round_continued_model - 1
        ]
    else:
        val_key = eval_keys[0]
        best_val_score = evals_result[val_key][metric_name][
            best_round_continued_model - 1
        ]

    if verbose:
        full_rounds = model.get_params().get("n_estimators", None) + min_boosting_rounds
        msg = (
            f"Early stopping after {best_round} rounds; "
            if best_round < full_rounds
            else ("Training completed for full " f"{full_rounds} rounds; ")
        )
        if best_train_score is not None:
            msg += f"train {metric_name.upper()}: {best_train_score:.4f}, "
        msg += f"val {metric_name.upper()}: {best_val_score:.4f}."
        print(msg, end="")  # Keep output on the same line

    return best_round, best_train_score, best_val_score


def get_num_cores() -> int:
    try:
        import multiprocessing

        n_jobs = multiprocessing.cpu_count()
        print(f"Number of cores: {n_jobs}. Using n_jobs={n_jobs}.")
    except (ImportError, NotImplementedError):
        print("Could not retrieve number of cores. Using default n_jobs=-1.")
        n_jobs = -1

    return n_jobs


class XGBHyperparameterTuner:
    """
    Hyperparameter tuning for XGBoost using randomized search and cross-validation.

    This tuner performs hyperparameter search using a specified parameter grid
    `param_dist` and selects the best model based on cross-validation score.

    Parameters
    ----------
    param_dist : dict
        Hyperparameter search space.
    n_iter : int, default=20
        Number of search iterations.
    n_splits : int, default=3
        Number of splits for cross-validation.
    val_size : float, default=0.2
        Fraction of training data used for validation or early stopping.
    scale_pos_weight : float or str, optional, default=None
        Class weight adjustment to handle imbalanced datasets.
        If set to `"balanced"`, it is computed as `negative_class / positive_class`.
    min_boosting_rounds : int, default=100
        Number of minimum boosting rounds (n_estimators)
    early_stopping_rounds : int, default=50
        Number of boosting rounds with no improvement before early stopping.
    seed : int, default=42
        Random seed for reproducibility.
    verbose : bool, default=True
        If True, prints progress and results.

    Attributes
    ----------
    best_model : XGBClassifier
        Best-trained XGBoost model.
    best_params : dict
        Best hyperparameter set found during tuning.
    best_score : float
        Best score achieved during tuning.
    """

    def __init__(
        self,
        param_dist: Dict[str, list],
        n_iter: int = 20,
        n_splits: int = 3,
        val_size: float = 0.2,
        objective: str | None = "binary:logistic",
        scale_pos_weight: Optional[Union[float, str]] = None,
        min_boosting_rounds: int = 100,
        early_stopping_rounds: int = 50,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.param_dist = param_dist
        self.n_iter = n_iter
        self.n_splits = n_splits
        self.val_size = val_size
        self.objective = objective
        self.scale_pos_weight = scale_pos_weight
        self.min_boosting_rounds = min_boosting_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.seed = seed
        self.verbose = verbose
        self.n_jobs = get_num_cores()

        self.is_binary: bool | None = None
        self.eval_metric: str | None = None

        # TODO: set these scoring_params dynamically based on warmup random search
        self.scoring_params = {
            "binary": {
                "lb": {
                    "roc_auc_diff": 0.01,
                    "gini": 0.2,
                    "ks_stat": 0.2,
                    "scaled_mwu_stat": 0.6,
                },
                "ub": {"roc_auc_diff": 0.1},
            },
            "multiclass": {
                "lb": {
                    "roc_auc_avg": 0.6,
                    "llss": 0.1,
                    "accuracy": 0.4,
                    "roc_auc_diff": 0.01,
                },
                "ub": {"roc_auc_diff": 0.1},
            },
        }
        # self.scoring_params = {
        #     "lb": {
        #         "roc_auc_diff": 0.01,  # 0.01 difference (train minus val) is ideal
        #         "gini": 0.2,  # auc = 0.6
        #         "ks_stat": 0.2,
        #         "scaled_mwu_stat": 0.6,
        #     },
        #     "ub": {
        #         "roc_auc_diff": 0.1,  # 0.1 difference is overfitting
        #     },
        # }
        self.best_model: XGBClassifier = None
        self.best_params: Dict[str, Union[int, float]] = {}
        self.best_score: float = -np.inf
        self.search_results: List[Dict] = []

    def prepare_scoring_data(
        self,
        model: pd.DataFrame,
        *datasets: Tuple[str, np.array, np.array],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for scoring function.
        """
        # Ensure "train" and "val" are in the dataset labels
        dataset_labels = {label for label, _, _ in datasets}
        if "train" not in dataset_labels or "val" not in dataset_labels:
            raise ValueError('Datasets must include both "train" and "val" labels.')

        df_pred_list = [
            model_utils.generate_predictions_df(model, x, y, dataset_label=label)
            for label, x, y in datasets
        ]
        df_pred = pd.concat(df_pred_list, axis=0)

        if self.is_binary:
            prediction_cols_param = "prediction"
            compare_metric = "roc_auc"
            diff_metric_name = "roc_auc_diff"
        else:
            prediction_cols_param = [f"pred_{i}" for i in range(model.n_classes_)]
            compare_metric = "roc_auc_avg"
            diff_metric_name = "roc_auc_diff"

        label, X, y = datasets[0]
        metrics = ClassifierMetrics(
            df_pred,
            dataset_col="dataset",
            target_col=y.name,
            prediction_cols=prediction_cols_param,
            # objective=self.objective,
            # scale_pos_weight=self.scale_pos_weight,
        )
        df_metrics = metrics.compute_metrics()
        df_metrics[diff_metric_name] = metrics.compare_metrics(
            compare_metric, "train", "val"
        )

        return df_metrics, df_pred

    def scoring_fcn(
        self, metrics: pd.DataFrame, dataset_col="dataset", validation_label="val"
    ) -> float:
        """
        Multi-objective scoring function to maximize:
        score = gini - roc_auc_diff + llss + 0.5 * (ks_stat + scaled_mwu_stat)
        where:
        - gini: maximize rankability
        - roc_auc_diff = (auc_train - auc_val): minimize AUC overfit
        - llss: logloss skill score: maximize predicted probability accuracy
        - ks_stat: maximize discrimination between class 0 and 1
        - scaled_mwu_stat: maximize discrimination between class 0 and 1

        All metrics range approx. between in [0, 1] so they are treated equally in the
        scoring function.
        """
        val_metrics = metrics[metrics[dataset_col] == validation_label]
        if val_metrics.empty:
            raise ValueError(f"No metrics found for dataset label: {validation_label}")

        problem_type = "binary" if self.is_binary else "multiclass"
        current_scoring_params = self.scoring_params[problem_type]

        def scale_metrics(metric_values: pd.Series) -> pd.Series:
            metric_values = metric_values.copy()
            metric_names = set(current_scoring_params.get("lb", {}).keys()).union(
                current_scoring_params.get("ub", {}).keys()
            )
            for name in metric_names:
                lb = current_scoring_params.get("lb", {}).get(name, 0)
                ub = current_scoring_params.get("ub", {}).get(name, 1)
                if name in metric_values.index:
                    metric_values[name] = (metric_values[name] - lb) / (ub - lb)
            return metric_values

        scaled = scale_metrics(val_metrics.iloc[0])

        if self.is_binary:
            score = (
                2 * scaled.get("gini", 0)
                + 0.1 * (1 - scaled.get("roc_auc_diff", 0))
                + scaled.get("llss", 0)
                + 1.2 * scaled.get("llss_cls_1", 0)
                + 0.5 * (scaled.get("ks_stat", 0) + scaled.get("scaled_mwu_stat", 0))
            )
            return score / (2 + 0.1 + 1 + 1.2 + 1) * 100
        else:  # Multi-class
            score = (
                2 * scaled.get("roc_auc_avg", 0)
                + 2 * scaled.get("llss", 0)
                + 1 * scaled.get("accuracy", 0)
                + 0.1 * (1 - scaled.get("roc_auc_diff", 0))
            )
            return score / (2 + 2 + 1 + 0.1) * 100

    def train_incremental_model(
        self, params, x_train, x_val, y_train, y_val
    ) -> XGBClassifier:
        """
        Two-phase model training:
        1. Train an initial model for a fixed minimum number of rounds.
        2. Continue training with early stopping.
        """
        params_internal = params.copy()
        params_internal["n_estimators"] = self.min_boosting_rounds

        initial_model = XGBClassifier(
            **params_internal,
            n_jobs=self.n_jobs,
            objective=self.objective,
            eval_metric=self.eval_metric,
            scale_pos_weight=self.scale_pos_weight if self.is_binary else None,
            random_state=self.seed,
        )
        initial_model.fit(x_train, y_train, verbose=False)

        # Resume training from the booster of the initial model
        params_internal["n_estimators"] = (
            params["n_estimators"] - self.min_boosting_rounds
        )
        continued_model = XGBClassifier(
            **params_internal,
            n_jobs=self.n_jobs,
            objective=self.objective,
            eval_metric=self.eval_metric,
            scale_pos_weight=self.scale_pos_weight if self.is_binary else None,
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.seed,
        )
        continued_model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            verbose=False,
            xgb_model=initial_model.get_booster(),
        )

        return continued_model

    def cross_validation(
        self,
        params: dict,
        skf: StratifiedKFold,
        x_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> float:
        """
        Perform cross-validation for a given set of parameters.

        Parameters
        ----------
        params : Dict[str, Union[int, float]]
            Hyperparameters to test.
        skf : StratifiedKFold
            Stratified k-fold cross-validation splitter.
        x_train : pd.DataFrame
            Training feature matrix.
        y_train : pd.Series
            Training target variable.

        Returns
        -------
        float
            Average score across folds.
        """
        cv_scores = []

        for cv_i, (train_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
            x_train_cv, x_val_cv = (x_train.iloc[train_idx], x_train.iloc[val_idx])
            y_train_cv, y_val_cv = (y_train.iloc[train_idx], y_train.iloc[val_idx])
            val_size = self.val_size * (self.n_splits - 1) / self.n_splits

            x_train_cv_sub, x_early_stop, y_train_cv_sub, y_early_stop = (
                train_test_split(
                    x_train_cv,
                    y_train_cv,
                    test_size=val_size,
                    stratify=y_train_cv,
                    random_state=self.seed,
                )
            )
            if self.verbose and cv_i == 0:
                msg = (
                    "| CV train | early stop | CV val | split: "
                    f"{len(y_train_cv_sub)} | {len(y_early_stop)} ({val_size:.4f}) | "
                    f"{len(y_val_cv)}"
                )
                print(msg)

            cv_model = self.train_incremental_model(
                params, x_train_cv_sub, x_early_stop, y_train_cv_sub, y_early_stop
            )

            if self.verbose:
                print(f"CV {cv_i + 1}: ", end="")
                report_best_round(cv_model, self.min_boosting_rounds)

            # Calculate scoring function
            df_metrics = self.prepare_scoring_data(
                cv_model,
                ("train", x_train_cv, y_train_cv),
                ("val", x_val_cv, y_val_cv),
            )[0]
            score = self.scoring_fcn(df_metrics)
            cv_scores.append(score)

            if self.verbose:
                print(f" CV validation score: {score:.4f}; components:\n{df_metrics}")

        avg_score = np.mean(cv_scores)

        if self.verbose:
            print(f"Avg CV validation score: {avg_score:.4f}.")

        return float(avg_score)

    def random_search(
        self,
        param_samples: List[Dict[str, Union[int, float]]],
        skf: StratifiedKFold,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        pbar: tqdm,
    ) -> None:

        for i, params in enumerate(param_samples):

            if self.verbose:
                print(
                    f"\nIteration: {i + 1}/{self.n_iter}\nTrying parameters: {params}"
                )

            cv_score = self.cross_validation(params, skf, x_train, y_train)
            self.search_results.append({"params": params, "score": cv_score})

            if cv_score > self.best_score:
                self.best_score = cv_score
                self.best_params = params

            sys.stdout.flush()
            pbar.update(1)

    def greedy_search(
        self,
        init_params: Dict[str, Union[int, float]],
        skf: StratifiedKFold,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        pbar: tqdm,
    ) -> None:

        param_space = {k: v for k, v in self.param_dist.items() if len(v) > 1}
        total_elements = sum(len(v) for v in param_space.values())

        if self.n_iter < total_elements:
            print(
                f"Warning: Greedy search will not cover all {total_elements} search "
                f"space values in {self.n_iter} iterations."
            )

        current_params = init_params.copy()
        param_cycle = cycle(param_space.keys())  # cycle through keys in order
        i = 0

        while i < self.n_iter:
            param_key = next(param_cycle)  # get next parameter to optimize
            param_values = param_space[param_key]  # get values for the parameter
            best_param_value = param_values[0]
            best_param_score = -np.inf

            for value in param_values:
                if self.verbose:
                    print(
                        f"\nIteration: {i + 1}/{self.n_iter}"
                        f'\nTrying parameter "{param_key}": {value} within:'
                        f"\n{current_params}"
                    )

                # Update only the current parameter
                test_params = current_params.copy()
                test_params[param_key] = value

                cv_score = self.cross_validation(test_params, skf, x_train, y_train)
                self.search_results.append({"params": test_params, "score": cv_score})

                if cv_score > self.best_score:
                    self.best_score = cv_score
                    self.best_params = test_params

                if cv_score >= best_param_score:
                    best_param_score = cv_score
                    best_param_value = value

                i += 1
                sys.stdout.flush()
                pbar.update(1)

                if i >= self.n_iter:
                    return

            # Update params dict after cycling through all values of the param_key
            current_params[param_key] = best_param_value

    def fit(
        self, x_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[
        XGBClassifier, Dict[str, Union[int, float]], int, float, float, List[Dict]
    ]:
        """
        Perform hyperparameter tuning and trains the best XGBoost model.

        Returns
        -------
        Tuple[XGBClassifier, dict, float, float, List[Dict]]
            - Best trained model
            - Best hyperparameters
            - Best boosting round
            - Sub train data indexes
            - Sub val data indexes
            - Train ROC AUC
            - Validation ROC AUC
            - List of search results with parameters and AUC scores
        """
        self.is_binary = y_train.nunique() == 2
        if self.is_binary:
            assert self.objective in [
                "binary:logitraw",
                "binary:logistic",
            ], f"Invalid objective for binary classification: {self.objective}"
            self.eval_metric = "auc"
            if self.scale_pos_weight == "balanced":
                self.scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        else:
            assert self.objective in [
                "multi:softprob",
                "multi:softmax",
            ], f"Invalid objective for multi-class classification: {self.objective}"
            self.eval_metric = "mlogloss"
            if self.scale_pos_weight is not None:
                print(
                    "Warning: `scale_pos_weight` is not used for multi-class "
                    "classification and will be ignored."
                )

        param_samples = list(
            ParameterSampler(
                self.param_dist, n_iter=self.n_iter, random_state=self.seed
            )
        )
        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.seed
        )
        pbar = tqdm(total=self.n_iter + 1, desc="Hyperparameter Tuning")

        # Set aside validation data for final model (used in early stopping)
        x_train_sub, x_val_early_stop, y_train_sub, y_val_early_stop = train_test_split(
            x_train,
            y_train,
            test_size=self.val_size,
            stratify=y_train,
            random_state=self.seed,
        )

        if self.verbose:
            msg = (
                f"\nTrain | val split: {len(y_train_sub)} | {len(y_val_early_stop)} "
                f"(val ratio {self.val_size})"
            )
            print(msg)

        # self.random_search(param_samples, skf, x_train_sub, y_train_sub, pbar)
        self.greedy_search(param_samples[0], skf, x_train_sub, y_train_sub, pbar)

        # Train final model using best parameters
        self.best_params = dict(sorted(self.best_params.items()))
        self.best_model = self.train_incremental_model(
            self.best_params,
            x_train_sub,
            x_val_early_stop,
            y_train_sub,
            y_val_early_stop,
        )

        if self.verbose:
            print("\nFinal model: ", end="")

        best_round, _, _ = report_best_round(
            self.best_model, self.min_boosting_rounds, self.verbose
        )
        df_metrics = self.prepare_scoring_data(
            self.best_model,
            ("train", x_train_sub, y_train_sub),
            ("val", x_val_early_stop, y_val_early_stop),
        )[0]
        score = self.scoring_fcn(df_metrics)

        if self.verbose:
            print(
                f"\nValidation score: {score:.4f}; components:\n{df_metrics}"
                f"\nBest parameters:\n{json.dumps(self.best_params, indent=2)}"
            )

        sys.stdout.flush()
        pbar.update(1)

        return (
            self.best_model,
            self.best_params,
            y_train_sub.index,
            y_val_early_stop.index,
            best_round,
            self.search_results,
        )
