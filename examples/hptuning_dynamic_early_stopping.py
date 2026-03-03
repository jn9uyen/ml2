import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
import ray
from tqdm import tqdm
from typing import Dict, Any, List, Callable


class HyperparameterTuner:
    """
    A parallelized or sequential hyperparameter tuner, designed to be a flexible
    alternative to GridSearchCV.

    It supports different search strategies for exploring the parameter space
    and can run in parallel using Ray or sequentially.

    Attributes:
    -----------
    best_params_ : dict
        The parameter setting that gave the best results on the hold-out data.
    best_score_ : float
        The mean cross-validation score of the best_estimator.
    best_estimator_ : object
        The estimator that was chosen by the search, refitted on the whole data.
    cv_results_ : pd.DataFrame
        A DataFrame containing the detailed results of all trials.
    """

    def __init__(
        self,
        estimator,
        param_grid: Dict[str, List[Any]],
        search_method: str = "grid",
        use_ray: bool = True,
        cv=None,
        scoring: Callable = None,
        eval_metric: str = "auc",
        early_stopping_rounds: int = 10,
    ):
        """
        Initializes the HyperparameterTuner.

        Parameters:
        -----------
        estimator : object
            A scikit-learn compatible estimator object.
        param_grid : dict
            Dictionary with parameter names (str) as keys and lists of
            parameter settings to try as values.
        search_method : str, optional (default='grid')
            The search strategy. Supported methods: 'grid', 'greedy'.
        use_ray : bool, optional (default=True)
            If True, uses Ray for parallel execution. If False, runs sequentially.
        cv : int or cross-validation generator, optional (default=5)
            Determines the cross-validation splitting strategy.
        scoring : callable, optional
            A callable to evaluate the predictions on the test set. Must be
            a function that takes (y_true, y_pred) and returns a float.
        eval_metric : str, optional (default='auc')
            The evaluation metric for early stopping in LightGBM/XGBoost.
        early_stopping_rounds : int, optional (default=10)
            Number of rounds with no improvement to wait before stopping.
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self.search_method = search_method
        self.use_ray = use_ray
        self.cv = cv or StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.scoring = scoring
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds

        self.best_params_ = None
        self.best_score_ = -np.inf
        self.best_estimator_ = None
        self.cv_results_ = pd.DataFrame()

    @staticmethod
    @ray.remote
    def _evaluate_fold_parallel(
        params,
        estimator_cls,
        X_ref,
        y_ref,
        train_idx,
        val_idx,
        scoring,
        eval_metric,
        early_stopping_rounds,
    ):
        """A static remote function to evaluate a single fold for a given parameter set."""
        X, y = ray.get(X_ref), ray.get(y_ref)
        # The core logic is delegated to a common method
        return HyperparameterTuner._evaluate_fold_logic(
            params,
            estimator_cls,
            X,
            y,
            train_idx,
            val_idx,
            scoring,
            eval_metric,
            early_stopping_rounds,
        )

    @staticmethod
    def _evaluate_fold_logic(
        params,
        estimator_cls,
        X,
        y,
        train_idx,
        val_idx,
        scoring,
        eval_metric,
        early_stopping_rounds,
    ):
        """The core logic for fitting and scoring a single fold. Used by both parallel and sequential modes."""
        X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
        X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

        estimator = estimator_cls(**params)
        estimator.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            eval_metric=eval_metric,
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )

        preds = estimator.predict_proba(X_val_fold)[:, 1]
        score = scoring(y_val_fold, preds)
        return score, estimator.best_iteration_

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Run the hyperparameter search.

        Parameters:
        -----------
        X : pd.DataFrame
            Training data.
        y : pd.Series
            Target values.
        """
        if self.use_ray:
            print("Execution mode: Parallel with Ray")
            ray.init(ignore_reinit_error=True)
            try:
                self._execute_search(X, y)
            finally:
                ray.shutdown()
        else:
            print("Execution mode: Sequential")
            self._execute_search(X, y)

        self._refit_best_model(X, y)
        return self

    def _execute_search(self, X, y):
        """Internal method to dispatch to the correct search strategy."""
        if self.search_method == "grid":
            self._grid_search(X, y)
        elif self.search_method == "greedy":
            self._greedy_search(X, y)
        else:
            raise ValueError(f"Unsupported search_method: {self.search_method}")

    def _run_trials(self, param_list, X, y):
        """Helper to run a list of parameter trials, switching between parallel and sequential."""
        tasks = []
        X_ref, y_ref = (ray.put(X), ray.put(y)) if self.use_ray else (None, None)

        for params in param_list:
            for train_idx, val_idx in self.cv.split(X, y):
                if self.use_ray:
                    task = self._evaluate_fold_parallel.remote(
                        params,
                        self.estimator.__class__,
                        X_ref,
                        y_ref,
                        train_idx,
                        val_idx,
                        self.scoring,
                        self.eval_metric,
                        self.early_stopping_rounds,
                    )
                else:
                    task = self._evaluate_fold_logic(
                        params,
                        self.estimator.__class__,
                        X,
                        y,
                        train_idx,
                        val_idx,
                        self.scoring,
                        self.eval_metric,
                        self.early_stopping_rounds,
                    )
                tasks.append({"params": params, "task": task})

        # Collect results
        results_list = []
        pbar = tqdm(total=len(tasks), desc="Evaluating Folds")
        for item in tasks:
            score, best_iter = ray.get(item["task"]) if self.use_ray else item["task"]
            results_list.append(
                {"params": item["params"], "score": score, "best_iteration": best_iter}
            )
            pbar.update(1)
        pbar.close()

        # Aggregate fold results
        df = pd.DataFrame(results_list)
        grouped = (
            df.groupby("params")
            .agg(
                mean_score=("score", "mean"),
                std_score=("score", "std"),
                mean_best_iteration=("best_iteration", "mean"),
            )
            .reset_index()
        )
        return grouped

    def _grid_search(self, X, y):
        print("Starting Grid Search...")
        from sklearn.model_selection import ParameterGrid

        param_list = list(ParameterGrid(self.param_grid))
        self.cv_results_ = self._run_trials(param_list, X, y)

        best_trial = self.cv_results_.loc[self.cv_results_["mean_score"].idxmax()]
        self.best_score_ = best_trial["mean_score"]
        self.best_params_ = best_trial["params"]

    def _greedy_search(self, X, y):
        print("Starting Greedy Search...")
        all_results = []
        # Start with the first value for each parameter as the base
        current_best_params = {p: v[0] for p, v in self.param_grid.items()}

        for param_name in self.param_grid:
            print(f"  Tuning parameter: '{param_name}'")
            params_to_test = []
            for val in self.param_grid[param_name]:
                temp_params = current_best_params.copy()
                temp_params[param_name] = val
                params_to_test.append(temp_params)

            trial_results_df = self._run_trials(params_to_test, X, y)
            all_results.append(trial_results_df)

            best_trial = trial_results_df.loc[trial_results_df["mean_score"].idxmax()]
            current_best_params = best_trial["params"]
            print(
                f"    -> Best value found: {current_best_params[param_name]} (Score: {best_trial['mean_score']:.4f})"
            )

        self.cv_results_ = pd.concat(all_results, ignore_index=True)
        final_best_trial = self.cv_results_.loc[self.cv_results_["mean_score"].idxmax()]
        self.best_score_ = final_best_trial["mean_score"]
        self.best_params_ = final_best_trial["params"]

    def _refit_best_model(self, X, y):
        print("\nRefitting model with best parameters...")
        refit_params = self.best_params_.copy()

        # Get the optimal number of iterations from the CV results
        best_trial_df = self.cv_results_[
            self.cv_results_["params"].apply(lambda p: p == self.best_params_)
        ]
        n_estimators = int(np.ceil(best_trial_df["mean_best_iteration"].iloc[0]))
        refit_params["n_estimators"] = n_estimators

        print(f"Final params: {refit_params}")
        self.best_estimator_ = self.estimator.__class__(**refit_params)
        self.best_estimator_.fit(X, y)
