from typing import cast

import catboost as cb
import dalex
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
import seaborn as sns
import shap
from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMModel, LGBMRegressor
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pdpbox import info_plots, pdp
from scipy.special import expit
from sklearn.base import is_classifier as sklearn_is_classifier
from xgboost import XGBClassifier, XGBModel, XGBRegressor

import visualization as viz

viz.configure_plotting()


def compute_relative_importance(df: pd.DataFrame, importance_col: str):
    """
    Compute relative and cumulative importance given a list of features and their
    importances. Return input df with two additional columns:
    ["relative_importance", "cumulative_importance"].
    """
    df["relative_importance"] = df[importance_col] / df[importance_col].max()
    df["cumulative_importance"] = df[importance_col].cumsum() / df[importance_col].sum()
    return df


def _get_feature_names(model: CatBoost | LGBMModel | XGBModel) -> list[str]:
    """
    Get feature names from a trained model.
    """
    if isinstance(model, CatBoost):
        return model.feature_names_ if model.feature_names_ is not None else []

    if isinstance(model, LGBMModel):
        return model.booster_.feature_name()

    if isinstance(model, XGBModel):
        booster_features = model.get_booster().feature_names
        return cast(list, booster_features) if booster_features is not None else []

    raise TypeError("Unsupported model type for feature name retrieval.")


def clean_df_dtypes_for_dalex(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Pandas Int, Float to standard int, float for dalex."""
    df_cleaned = df.copy()

    for col in df_cleaned.columns:
        # Check if the column's dtype is one of the nullable integer types
        if str(df_cleaned[col].dtype) in ["Int8", "Int16", "Int32", "Int64"]:
            # The column contains pd.NA. We need to handle this.
            # Converting directly to 'int64' will raise an error if pd.NA is present.
            # The safest way is to first replace pd.NA with np.nan and then convert.
            df_cleaned[col] = df_cleaned[col].replace(pd.NA, np.nan)
            try:
                # Try to convert to standard int64. This will only work if there are no NaNs.
                df_cleaned[col] = df_cleaned[col].astype("int64")
            except ValueError:
                # If there were NaNs, convert to float64, which can handle them.
                df_cleaned[col] = df_cleaned[col].astype("float64")

        # Check if the column's dtype is one of the nullable float types
        elif str(df_cleaned[col].dtype) in ["Float32", "Float64"]:
            df_cleaned[col] = df_cleaned[col].astype("float64")

    return df_cleaned


class ShapTreeExplainer:
    """
    A simple wrapper for computing SHAP values for a tree-based model (XGBClassifier,
    LGBMClassifier).

    This class ensures that the input features DataFrame contains all the features
    that the model was trained on, and reorders them accordingly.

    Parameters
    ----------
    model : XGBClassifier | LGBMClassifier
        A trained XGBoost or LightGBM classifier.
    features_df : pd.DataFrame
        Features df that includes all feature names expected by the model.
        The index is expected to be the ID column.
    """

    def __init__(
        self,
        model: CatBoost | LGBMModel | XGBModel,
        features_df: pd.DataFrame,
        exclude_features: list[str] | None = None,
    ):
        if not isinstance(model, (CatBoost | LGBMModel | XGBModel)):
            raise ValueError(
                "The model must be an instance of CatBoost, LGBMModel, or XGBModel."
            )

        self.model = model
        self.exclude_features = exclude_features
        self.feature_names = _get_feature_names(model)

        missing_features = set(self.feature_names) - set(features_df.columns)
        if missing_features:
            raise ValueError(
                f"Input features_df is missing the following {len(missing_features)} "
                f"model features:\n{missing_features}."
            )

        # Reorder features_df to match the order expected by the model.
        self.features_df = features_df[self.feature_names]

        # Attributes
        self.shap_explainer = shap.TreeExplainer(model)
        self.shap_explanation_data, self.shap_values = self.compute_shap_values()

    def compute_shap_values(self) -> tuple[shap.Explanation, pd.DataFrame]:
        """Compute SHAP values for the model's features."""

        # Handle CatBoost categorical features (target) encoding.
        if isinstance(self.model, CatBoost):
            text_features_indices = self.model.get_text_feature_indices()
            cat_features_indices = self.model.get_cat_feature_indices()

            # Create a CatBoost Pool, which is required for its SHAP method.
            pool = cb.Pool(
                self.features_df,
                cat_features=cat_features_indices,
                text_features=text_features_indices,
            )
            shap_values_raw = self.model.get_feature_importance(
                pool, type="ShapValues"  # type: ignore
            )
            assert isinstance(shap_values_raw, np.ndarray)

            # The output needs to be shaped into a shap.Explanation object;
            # the last column of the output is the expected value.
            shap_values_arr = shap_values_raw[:, :-1]
            expected_value = shap_values_raw[0, -1]  # It's the same for all samples.

            shap_explanation_data = shap.Explanation(
                values=shap_values_arr,
                base_values=expected_value,
                data=self.features_df,
                feature_names=self.features_df.columns.tolist(),
            )
        else:
            shap_explanation_data = self.shap_explainer(self.features_df)

        if self.exclude_features:
            features_to_keep = [
                feat for feat in self.feature_names if feat not in self.exclude_features
            ]
            shap_explanation_data = shap_explanation_data[:, features_to_keep]
            self.features_df = self.features_df[features_to_keep]

        shap_values = pd.DataFrame(
            shap_explanation_data.values,
            index=self.features_df.index,
            columns=self.features_df.columns,
        )

        return shap_explanation_data, shap_values

    def compute_shap_feature_importance(self) -> pd.DataFrame:
        shap_feature_importance = (
            self.shap_values.abs()
            .mean()
            .rename("mean_abs_shap")
            .sort_values(ascending=False)
            .to_frame()
            .reset_index()
            .rename(columns={"index": "feature_name"})
        )
        shap_feature_importance = compute_relative_importance(
            shap_feature_importance, "mean_abs_shap"
        )

        return shap_feature_importance

    def plot_shap_summary(
        self,
        max_display: int = 10,
        saveas_filename: str | None = None,
        **kwargs,
    ) -> Figure:
        """Plot a SHAP summary plot for the model's features."""
        features_for_plot = self.features_df.copy()

        # Apply label encoding (factorize) to categorical columns.
        if isinstance(self.model, CatBoost):
            categorical_cols = features_for_plot.select_dtypes(
                include=["object", "category"]
            ).columns
            for col in categorical_cols:
                features_for_plot[col] = pd.factorize(features_for_plot[col])[0]

        fig = plt.figure()
        shap.summary_plot(
            self.shap_explanation_data,
            features_for_plot,
            max_display=max_display,
            show=False,
        )
        ax = plt.gca()
        ax.tick_params(axis="x", colors="lightgray")
        ax.tick_params(axis="y", colors="lightgray")

        if "bbox_inches" not in kwargs:
            kwargs["bbox_inches"] = "tight"

        viz.save_figure(fig, saveas_filename or "shap_summary_plot", **kwargs)
        plt.close(fig)

        return fig


class GlobalModelAgnosticMethods:
    """
    This class includes methods for explaining model predictions based on their feature
    values and contributions. It includes:
    - Univariate feature vs target plots
    - Partial dependence plots (PDP) and Individual Conditional Expectation (ICE)
    - Interaction of two features vs target plots
    - Interaction PD plots

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing id, target, prediction and features.
    """

    def __init__(
        self,
        model: CatBoost | LGBMModel | XGBModel,
        data: pd.DataFrame,
        target_col: str,
        prediction_col: str,
        id_cols: str | list[str] | None = None,
    ):
        self.model = model
        self.data = data
        self.target_col = target_col
        self.prediction_col = prediction_col
        self.id_cols = id_cols

        # Attributes
        self.features_df = data.drop([target_col, prediction_col], axis=1)
        if id_cols is not None:
            self.features_df = self.features_df.set_index(id_cols)
        self.feature_names = _get_feature_names(model)
        self.features_df = self.features_df[self.feature_names]

    @staticmethod
    def _check_missing_values(
        data: pd.DataFrame, feature_names: str | list[str]
    ) -> pd.DataFrame:
        if isinstance(feature_names, str):
            feature_names = [feature_names]

        # Check for missing values in the specified columns.
        if data[feature_names].isna().any().any():
            n_missing = data[feature_names].isna().sum()
            print(
                f"Warning: Found missing values in columns {feature_names}. "
                f"Counts per column:\n{n_missing}. "
                f"Dropping NA rows for plotting."
            )
            plot_data = data.dropna(subset=feature_names)
        else:
            plot_data = data

        return plot_data

    def univariate_target_plot(
        self,
        feature_name: str,
        feature_name_readable: str | None = None,
        categorical_top_n: int = 10,
        **kwargs,
    ) -> tuple[Figure, Axes, pd.DataFrame]:
        """
        Plot the relationship between a feature and the target variable.
        """
        feature_name_readable = feature_name_readable or feature_name
        plot_data = self._check_missing_values(self.data, feature_name)
        saveas_filename = kwargs.pop("saveas_filename", None)
        engine = "matplotlib" if saveas_filename else "plotly"

        # --- Numeric feature plot using pdpbox ---
        if pd.api.types.is_numeric_dtype(plot_data[feature_name]):
            plot_args = {
                "num_grid_points": 10,
                "grid_type": "percentile",
                "percentile_range": None,
                "grid_range": None,
                "cust_grid_points": None,
                "show_outliers": False,
                "endpoint": True,
            }
            plot_args.update(kwargs)

            target_vs_feature = info_plots.TargetPlot(
                df=plot_data,
                feature=feature_name,
                feature_name=feature_name_readable,
                target=self.target_col,
                **plot_args,
            )
            fig, axes, summary_df = target_vs_feature.plot(
                show_percentile=True,
                figsize=None,
                dpi=300,
                ncols=2,
                plot_params=None,
                engine=engine,
                template="plotly_white",
            )
            axes = cast(Axes, axes)

            if engine == "matplotlib":
                # Ensure 'axes' is iterable, even if it's a single object.
                if not isinstance(axes, (list, np.ndarray, dict)):
                    axes_to_modify = [axes]
                # If it's a dictionary, get the axes from its values.
                elif isinstance(axes, dict):
                    axes_to_modify = axes.values()
                else:
                    axes_to_modify = axes

                # Iterate through the list/array of axes and apply settings.
                for ax in axes_to_modify:
                    if hasattr(ax, "tick_params"):  # Final safety check.
                        ax.tick_params(axis="x", colors="lightgray")
                        ax.tick_params(axis="y", colors="lightgray")
        else:
            # --- Categorical Feature Path (using seaborn) ---
            top_categories = (
                plot_data[feature_name].value_counts().nlargest(categorical_top_n).index
            )
            plot_data_cat = plot_data[plot_data[feature_name].isin(top_categories)]

            summary_df = (
                plot_data_cat.groupby(feature_name)[self.target_col]
                .agg(["mean", "count", "std"])
                .sort_values(by="mean", ascending=False)
                .reset_index()
            )

            # Plot: dynamic figure height based on the number of categories.
            dynamic_height = max(viz.HEIGHT, len(top_categories) * 0.5)
            fig, ax = plt.subplots(figsize=(viz.WIDTH, dynamic_height))

            sns.boxplot(
                y=feature_name,
                x=self.target_col,
                data=plot_data_cat,
                ax=ax,
                order=summary_df[feature_name],  # Order by mean target value.
                orient="h",
                showmeans=True,
                linecolor="lightgray",
            )

            ax.set_title(f'"{self.target_col}" by "{feature_name_readable}"')
            ax.set_xlabel(f"Average {self.target_col}")
            ax.set_ylabel(feature_name_readable)
            axes = cast(Axes, ax)  # Maintain consistent return variable name.

        if saveas_filename:
            viz.save_figure(fig, saveas_filename)

        return fig, axes, summary_df

    def prediction_distribution_plot(
        self, feature_name: str, feature_name_readable: str | None = None, **kwargs
    ) -> tuple[Figure, Axes, pd.DataFrame]:
        """
        Plot the target prediction distribution of a feature.
        """
        feature_name_readable = feature_name_readable or feature_name
        plot_data = self._check_missing_values(self.data, feature_name)

        defaults = {
            "pred_func": None,
            "n_classes": None,
            "num_grid_points": 10,
            "grid_type": "percentile",
            "percentile_range": None,
            "grid_range": None,
            "cust_grid_points": None,
            "show_outliers": False,
            "endpoint": True,
            "predict_kwds": {},
            "chunk_size": -1,
        }
        defaults.update(kwargs)

        prediction_distribution = info_plots.PredictPlot(
            model=self.model,
            model_features=self.features_df.columns,
            df=plot_data,
            feature=feature_name,
            feature_name=feature_name_readable,
            **defaults,
        )
        fig, axes, summary_df = prediction_distribution.plot(
            show_percentile=True,
            figsize=None,
            ncols=1,
            plot_params=None,
            engine="plotly",
            template="plotly_white",
        )

        return fig, axes, summary_df

    def partial_dependence_plot(
        self, feature_name: str, feature_name_readable: str | None = None, **kwargs
    ) -> tuple[Figure, Axes]:
        """
        Plot the Partial Dependence curve for a given feature.
        """
        feature_name_readable = feature_name_readable or feature_name
        plot_data = self._check_missing_values(self.data, feature_name)

        defaults = {
            "center": True,
            "plot_lines": True,
            "frac_to_plot": 1,
            "cluster": True,
            "n_cluster_centers": 50,
            "cluster_method": "accurate",
            "plot_pts_dist": True,
            "to_bins": True,
            "show_percentile": True,
            "which_classes": None,
            "figsize": None,
            "dpi": 300,
            "ncols": 2,
            "plot_params": {"pdp_hl": True, "line": {"hl_color": "#f46d43"}},
            "engine": "plotly",
            "template": "plotly_white",
        }
        defaults.update(kwargs)

        pdp_isolate = pdp.PDPIsolate(
            model=self.model,
            df=plot_data,
            model_features=self.features_df.columns,
            feature=feature_name,
            feature_name=feature_name_readable,
        )
        fig, axes = pdp_isolate.plot(**defaults)
        axes = cast(Axes, axes)

        return fig, axes

    def interaction_target_plot(
        self, two_feature_names: list[str], **kwargs
    ) -> tuple[Figure, Axes, pd.DataFrame]:
        """
        Plot the target against two interacting features.
        """
        plot_data = self._check_missing_values(self.data, two_feature_names)

        defaults = {
            "num_grid_points": 10,
            "grid_types": "percentile",
            "percentile_ranges": None,
            "grid_ranges": None,
            "cust_grid_points": None,
            "show_outliers": False,
            "endpoints": True,
        }
        defaults.update(kwargs)

        target_two_features = info_plots.InteractTargetPlot(
            df=plot_data,
            features=two_feature_names,
            feature_names=two_feature_names,
            target=self.target_col,
            **defaults,
        )
        fig, axes, summary_df = target_two_features.plot(
            show_percentile=True,
            figsize=(1200, 700),
            annotate=True,
            engine="plotly",
            template="plotly_white",
        )

        return fig, axes, summary_df

    def interaction_prediction_plot(
        self, two_feature_names: list[str], **kwargs
    ) -> tuple[Figure, Axes, pd.DataFrame]:
        """
        Plot the prediction against two interacting features.
        """
        plot_data = self._check_missing_values(self.data, two_feature_names)

        defaults = {
            "pred_func": None,
            "n_classes": None,
            "num_grid_points": 10,
            "grid_types": "percentile",
            "percentile_ranges": None,
            "grid_ranges": None,
            "cust_grid_points": None,
            "show_outliers": False,
            "endpoints": True,
            "predict_kwds": {},
            "chunk_size": -1,
        }
        defaults.update(kwargs)

        prediction_two_features = info_plots.InteractPredictPlot(
            model=self.model,
            df=plot_data,
            model_features=self.features_df.columns,
            features=two_feature_names,
            feature_names=two_feature_names,
            **defaults,
        )
        fig, axes, summary_df = prediction_two_features.plot(
            show_percentile=True,
            figsize=(1200, 800),
            ncols=2,
            annotate=True,
            plot_params={"subplot_ratio": {"y": [10, 1]}},
            engine="plotly",
            template="plotly_white",
        )

        return fig, axes, summary_df

    def interaction_partial_dependence_plot(
        self,
        two_feature_names: list[str],
        pdp_kwargs: dict | None = None,
        plot_kwargs: dict | None = None,
    ) -> tuple[Figure, Axes]:
        """
        Plot the Partial Dependence surface ("contour" or "grid") for two features.
        """
        plot_data = self._check_missing_values(self.data, two_feature_names)

        defaults = {
            "pred_func": None,
            "n_classes": None,
            "memory_limit": 0.5,
            "chunk_size": -1,
            "n_jobs": 1,
            "predict_kwds": {},
            "data_transformer": None,
            "num_grid_points": 10,
            "grid_types": "percentile",
            "percentile_ranges": None,
            "grid_ranges": None,
            "cust_grid_points": None,
        }
        if pdp_kwargs:
            defaults.update(pdp_kwargs)

        pdp_two_features = pdp.PDPInteract(
            model=self.model,
            df=plot_data,
            model_features=self.features_df.columns,
            features=two_feature_names,
            feature_names=two_feature_names,
            **defaults,
        )

        # Set defaults for the plot method.
        plot_defaults = {
            "plot_type": "contour",
            "plot_pdp": True,
            "to_bins": True,
            "show_percentile": True,
            "which_classes": None,
            "figsize": None,
            "dpi": 300,
            "ncols": 2,
            "plot_params": None,
            "engine": "plotly",
            "template": "plotly_white",
        }
        if plot_kwargs:
            plot_defaults.update(plot_kwargs)

        fig, axes = pdp_two_features.plot(**plot_defaults)
        axes = cast(Axes, axes)

        return fig, axes


class IndividualProfiling:
    """
    This class includes methods for profiling individual predictions based on their
    feature values and contribution.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing id, target, prediction and features.
    """

    def __init__(
        self,
        model: CatBoost | LGBMModel | XGBModel,
        data: pd.DataFrame,
        target_col: str,
        prediction_col: str,
        id_cols: str | list[str] | None = None,
        exclude_features: list[str] | None = None,
    ):
        self.model = model
        self.data = data
        self.target_col = target_col
        self.prediction_col = prediction_col
        self.id_cols = id_cols
        self.exclude_features = exclude_features

        # Attributes
        self.features_df = data.drop([target_col, prediction_col], axis=1)
        if id_cols is not None:
            self.features_df = self.features_df.set_index(id_cols)
        self.feature_names = _get_feature_names(model)
        self.features_df = self.features_df[self.feature_names]

        self.dalex_explainer = dalex.Explainer(
            model, data=self.features_df, y=data[target_col]
        )

    def feature_contributions(
        self,
        id_value: str | tuple[str],
        plot: bool = True,
        plot_max_char_length: int | None = 25,
        saveas_filename: str | None = None,
        **kwargs,
    ) -> pd.DataFrame | tuple[Figure, pd.DataFrame]:
        """
        Compute the feature contributions (SHAP) for a given individual, and optionally
        plot them.
        """
        individual_features, target_value, is_clf, prediction, probability = (
            self._compute_individual_values(id_value)
        )

        if is_clf:
            title_str = (
                f"SHAP Waterfall | ID: {id_value}\n"
                f"Target value: {target_value:.4f}\n"
                f"Model Prediction (Log-odds): {prediction:.4f}\n"
                f"Probability: {probability:.4f}"
            )
        else:
            title_str = (
                f"SHAP Waterfall | ID: {id_value}\n"
                f"Target value: {target_value:.4f}\n"
                f"Model Prediction: {prediction:.4f}"
            )

        individual_shap = ShapTreeExplainer(
            self.model, individual_features, self.exclude_features
        )
        shap_contributions = individual_shap.compute_shap_feature_importance()

        if plot:
            plot_explanation = individual_shap.shap_explanation_data[0]

            if plot_max_char_length:
                # Truncate labels to max length.
                max_len = plot_max_char_length
                truncated_feature_names = [
                    (f"{label[:max_len]}..." if len(label) > max_len else label)
                    for label in plot_explanation.feature_names
                ]
                truncated_values = [
                    (
                        f"{value[:max_len]}..."
                        if isinstance(value, str) and len(value) > max_len
                        else value
                    )
                    for value in plot_explanation.data
                ]
                plot_explanation.feature_names = truncated_feature_names
                plot_explanation.data = truncated_values

            fig = plt.figure()
            shap.plots.waterfall(plot_explanation, show=False)
            ax = plt.gca()
            ax.tick_params(axis="x", colors="lightgray")
            ax.tick_params(axis="y", colors="lightgray")
            ax.set_title(
                title_str,
                fontsize=12,
                color="white",
            )
            # plt.subplots_adjust(left=0.4, top=0.8)

            if "bbox_inches" not in kwargs:
                kwargs["bbox_inches"] = "tight"

            viz.save_figure(fig, saveas_filename or "shap_waterfall", **kwargs)
            plt.close(fig)

            return fig, shap_contributions

        return shap_contributions

    def ceteris_paribus_plot(
        self,
        id_value: str | tuple[str],
        feature_names: str | list[str],
        log_scale: bool = False,
    ) -> Figure:
        """
        Plot the Ceteris Paribus curve(s) for a given individual's feature(s).
        """
        individual_features, target_value, is_clf, prediction, probability = (
            self._compute_individual_values(id_value)
        )

        if is_clf:
            title_str = (
                f"What If? Individual ID: {id_value}\n"
                f"Target value: {target_value:.4f}\n"
                f"Model Prediction (Log-odds): {prediction:.4f}\n"
                f"Probability: {probability:.4f}"
            )
        else:
            title_str = (
                f"What If? Individual ID: {id_value}\n"
                f"Target value: {target_value:.4f}\n"
                f"Model Prediction: {prediction:.4f}"
            )

        cp = self.dalex_explainer.predict_profile(
            self.features_df.loc[[id_value], :], variables=feature_names
        )

        fig = cp.plot(size=3, title=title_str, show=False)

        if log_scale:
            # --- Adjust the x-values by adding minimum negative absolute value ---
            # Separate the data traces from the marker traces.
            data_traces = [t for t in fig.data if len(t.x) > 2]
            marker_traces = [t for t in fig.data if len(t.x) <= 2]

            # Loop through the data traces and apply the transformation
            # to both the data and its corresponding marker.
            for i, trace in enumerate(data_traces):
                if hasattr(trace, "x") and trace.x is not None:
                    min_val = pd.Series(trace.x).min()
                    shift_value = 0

                    if pd.notna(min_val) and min_val <= 0:
                        shift_value = 1 if min_val == 0 else abs(min_val)

                    if shift_value > 0:
                        trace.x = [
                            val + shift_value for val in trace.x if pd.notna(val)
                        ]

                    # Apply the same shift to the corresponding marker trace.
                    if i < len(marker_traces):
                        marker = marker_traces[i]
                        if hasattr(marker, "x") and marker.x is not None:
                            original_marker_x = marker.x[0]
                            # Transform the marker's x-value
                            marker.x = [original_marker_x + shift_value]
                            print(
                                f"Original marker at {original_marker_x:.2f} moved to "
                                f"{marker.x[0]:.2f} (shift: {shift_value:.2f})"
                            )

            fig.update_xaxes(type="log")

        return fig

    def _compute_individual_values(
        self, id_value: str | tuple[str]
    ) -> tuple[pd.DataFrame, float, bool, float, float]:
        """
        Compute the individual feature values and model predictions for a given ID.
        """
        individual_features = pd.DataFrame(self.features_df.loc[[id_value]])
        target_value = self.data[self.target_col].loc[id_value]

        is_clf = sklearn_is_classifier(self.model)
        predictions, probabilities = self._compute_predictions(
            individual_features, is_clf
        )
        prediction, probability = predictions.item(), probabilities.item()

        return individual_features, target_value, is_clf, prediction, probability

    def _compute_predictions(
        self, features_df: pd.DataFrame, is_classifier: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the raw prediction for a given instance. If the model is a classifier,
        return the log-odds and probability. If the model is a regressor, return the raw
        prediction.
        """
        match self.model:
            case CatBoostClassifier():
                predictions = self.model.predict(
                    features_df, prediction_type="RawFormulaVal"
                )
            case LGBMRegressor() | LGBMClassifier():
                predictions = self.model.predict(features_df, raw_score=True)
            case XGBClassifier():
                predictions = self.model.predict(features_df, output_margin=True)
            case _:
                predictions = self.model.predict(features_df)

        match self.model:
            # --- CatBoost ---
            case CatBoostClassifier():
                predictions = self.model.predict(
                    features_df, prediction_type="RawFormulaVal"
                )  # log-odds.
            case CatBoostRegressor():
                predictions = self.model.predict(features_df)

            # --- LightGBM ---
            case LGBMClassifier() | LGBMRegressor():
                predictions = self.model.predict(features_df, raw_score=True)

            # --- XGBoost ---
            case XGBClassifier():
                predictions = self.model.predict(
                    features_df, output_margin=True
                )  # log-odds.
            case XGBRegressor():
                predictions = self.model.predict(features_df)

            case _:
                print("Warning: Using default .predict() for an unknown model type.")
                predictions = self.model.predict(features_df)

        if scipy.sparse.issparse(predictions):
            # Convert to a dense numpy array.
            dense_predictions = predictions.toarray()  # type: ignore
        else:
            dense_predictions = np.asarray(predictions)

        probabilities = np.zeros_like(dense_predictions)

        if is_classifier:
            # Predictions are log odds.
            probabilities = expit(dense_predictions)

        return dense_predictions, probabilities
