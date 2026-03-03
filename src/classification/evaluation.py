"""
Methods for model evaluation
"""

from typing import Optional, Union

import classification.ml_plotting as clplt
import numpy as np
import pandas as pd
import shap
from helper import logging_config
from sklearn import metrics

logger = logging_config.getLogger(__name__)


class Evaluator:
    """
    Evaluate model on data (features, target). Perform two evaluations:
    1. Model explainability
        a. feature importance
        b. SHAP values
        c. partial dependence plots: TODO
    2. Predictive performance
    """

    def __init__(
        self,
        model: object,
        features: pd.DataFrame,
        target: pd.DataFrame,
        shap_sample_size: Optional[int] = None,
        max_feat_display: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.model = model
        self.features = features
        self.target = target
        self.shap_sample_size = shap_sample_size
        self.max_feat_display = max_feat_display
        self.random_state = random_state

        self.rel_feat_imp = calc_relative_feature_importances(
            self.model, self.features.columns
        )
        self.explainer: object = None
        self.shap_values: list = None
        self.target_prob: pd.Series = None

    def explain_model(self) -> None:
        """
        Explain model using training data using two methods:
            a. feature importance
            b. SHAP values
            c. partial dependence plots: TODO
        """
        if self.features.shape[0] > self.shap_sample_size:
            xtrain_smpl = self.features.sample(
                self.shap_sample_size, random_state=self.random_state
            )
        else:
            xtrain_smpl = self.features
        # xtrain_smpl["remainder__loan_raised_date_wk_cos"] = xtrain_smpl[
        #     "remainder__loan_raised_date_wk_cos"
        # ].astype(float)

        self.explainer = shap.Explainer(self.model, xtrain_smpl)
        self.shap_values = self.explainer(xtrain_smpl)

        # Plot feature importance and SHAP beeswarm plot
        clplt.plot_feature_importances(
            self.rel_feat_imp,
            self.model.importance_type,
            self.model.__class__.__name__,
            self.max_feat_display,
            is_relative_feat_imp=True,
        )
        clplt.plot_shap_beeswarm(
            self.shap_values, self.model.__class__.__name__, self.max_feat_display
        )

        logger.info(
            f"\nTop features (count = {len(self.rel_feat_imp)}):\n{self.rel_feat_imp}"
        )

    def eval_predictive_performance(self, dataset_name: str) -> None:
        """
        Evaluate model predictive performance.
        1. Classifier
            a. ROC curve
            b. Precision-recall curve
            c. Confusion matrix metrics
            d. Business metrics: TODO
        2. Regressor: TODO
        """
        y_prob = predict_probabilities(self.model, self.features)
        y = self.target

        fpr, tpr, roc_thresholds, roc_auc, gini = calc_roc_curve(y, y_prob)
        precision, recall, pr_thresholds, ap, ap0, mean_ap = (
            calc_precision_recall_curve(y, y_prob)
        )
        ranked_probabilities = rank_probabilities(y, y_prob, greater_is_better=False)
        lift, baseline_proba_rate = calc_lift(
            y, y_prob, targeted_class=0, by="decile", greater_is_better=False
        )
        business_rpt, business_stats_rpt, confusion_rpt, evaluation_rpt = (
            bld_classification_report(
                self.model,
                self.features,
                self.target,
                thresholds=np.arange(0.1, 1, 0.1),
                # loan_amts=xtest["loan_amount"].astype(float),
                # revenue_pct=0.04,
            )
        )

        clplt.plot_roc_curve(
            fpr,
            tpr,
            roc_thresholds,
            roc_auc,
            gini,
            title=f"ROC ({dataset_name}) [{self.model.__class__.__name__}]",
            saveas_filename=f"roc_curve_{dataset_name}",
        )
        clplt.plot_precision_recall_curve(
            precision,
            recall,
            pr_thresholds,
            ap,
            ap0,
            mean_ap,
            title=f"Precision-Recall ({dataset_name}) [{self.model.__class__.__name__}]",
            saveas_filename=f"pr_curve_{dataset_name}",
        )
        clplt.plot_calibration_curve(
            y,
            y_prob,
            model_name=self.model.__class__.__name__,
            title=f"Calibration Curve ({dataset_name}) [{self.model.__class__.__name__}]",
            saveas_filename=f"calibration_{dataset_name}",
        )
        clplt.plot_ranked_probabilities(
            ranked_probabilities,
            title=f"Ranked probabilities ({dataset_name}) [{self.model.__class__.__name__}]",
            saveas_filename=f"ranked_probas_{dataset_name}",
        )
        clplt.plot_lift_curve(
            lift,
            show_rate=False,
            title=f"Lift Chart ({dataset_name}) [{self.model.__class__.__name__}]",
            saveas_filename=f"lift_chart_{dataset_name}",
        )
        clplt.plot_prediction_distribution(
            pd.DataFrame({"y": y, "y_pred": y_prob}),
            class_mapping={0: "Non-default", 1: "Default"},
            title=f"Predictions ({dataset_name}) [{self.model.__class__.__name__}]",
            saveas_filename=f"pred_distribution_{dataset_name}",
        )

        # Feature distributions by importance for training, test datasets
        df = pd.concat((self.features, self.target), axis=1)
        feature_cols = self.rel_feat_imp["feature"].head(self.max_feat_display)
        titles = [f"{feat} ({dataset_name})" for feat in feature_cols]

        clplt.plot_multifaceted_features(
            df,
            feature_cols=feature_cols,
            target_col="has_defaulted",
            titles=titles,
            saveas_filename=f"multifaceted_features_{dataset_name}",
        )

        logger.info(f"Summary of metrics for dataset: {dataset_name}")
        logger.info(f"Business report ({dataset_name}):\n{business_rpt}")
        logger.info(f"Business stats report ({dataset_name}):\n{business_stats_rpt}")
        logger.info(f"Confusion report ({dataset_name}):\n{confusion_rpt}")
        logger.info(f"Evaluation report ({dataset_name}):\n{evaluation_rpt}")
        logger.info(f"Mean average precision ({dataset_name}): {mean_ap:.4f}")
        logger.info(f"Lift report ({dataset_name}):\n{lift}")
        logger.info(
            f"Baseline probability rate ({dataset_name}): {baseline_proba_rate:.4f}"
        )


def calc_relative_feature_importances(model: object, labels: list) -> pd.DataFrame:
    """
    Calculate relative feature importances ordered by most important feature

    Parameters
    ----------
    mdl : sklearn model
    labels : list or array-like
        list of features

    Returns
    -------
    rel_feat_imp : pd.DataFrame
        relative feature importances are percentages in [0, 1]
    """
    rel_feat_imp = model.feature_importances_ / np.max(model.feature_importances_)
    rel_feat_imp = (
        pd.DataFrame({"feature": labels, "importance": rel_feat_imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return rel_feat_imp


def predict_probabilities(
    model: object, features: pd.DataFrame, return_both: bool = False
) -> np.array:
    y_prob = model.predict_proba(features)

    return y_prob if return_both else y_prob[:, 1]


def calc_roc_curve(
    y: Union[list, np.array, pd.Series], y_prob: Union[list, np.array, pd.Series]
) -> tuple[list, list, list, float, float]:
    """
    Calculate ROC curve metrics: FPR, TRP, thresholds, ROC AUC, Gini

    Parameters
    ----------

    """
    fpr, tpr, thresholds = metrics.roc_curve(y, y_prob)
    roc_auc = metrics.roc_auc_score(y, y_prob)
    gini = 2 * roc_auc - 1

    return fpr, tpr, thresholds, roc_auc, gini


def calc_precision_recall_curve(
    y: Union[list, np.array, pd.Series], y_prob: Union[list, np.array, pd.Series]
) -> tuple[list, list, list, float, float, float]:
    """
    Calculate precision-recall curve metrics
    """
    precision, recall, thresholds = metrics.precision_recall_curve(y, y_prob)
    ap = metrics.average_precision_score(y, y_prob)
    ap0 = metrics.average_precision_score(y, 1 - y_prob)
    mean_ap = np.mean([ap, ap0])

    return precision, recall, thresholds, ap, ap0, mean_ap


def calc_confusion_metrics(
    y: Union[list, np.array, pd.Series],
    y_pred: Union[list, np.array, pd.Series],
    loan_amts: Optional[list[float]] = None,
    revenue_pct: Optional[float] = None,
) -> tuple[tuple, tuple, tuple, tuple, tuple]:
    """
    Calculate confusion matrix metrics and default business metrics
    """
    if loan_amts is None:
        loan_amts = np.ones(len(y), dtype=int)

    tn_ls = ((y == 0) & (y_pred == 0)).astype(int) * loan_amts
    fp_ls = ((y == 0) & (y_pred == 1)).astype(int) * loan_amts
    fn_ls = ((y == 1) & (y_pred == 0)).astype(int) * loan_amts
    tp_ls = ((y == 1) & (y_pred == 1)).astype(int) * loan_amts

    confusion = tuple(np.sum(k) for k in (tn_ls, fp_ls, fn_ls, tp_ls))

    if revenue_pct is not None:
        tn_amts = tn_ls * revenue_pct  # revenue
        fp_amts = fp_ls * revenue_pct * -1  # opportunity cost
        fn_amts = fn_ls * -1  # default cost
        tp_amts = tp_ls * 0  # no revenue, no cost
        confusion_amt = tuple(np.sum(k) for k in (tn_amts, fp_amts, fn_amts, tp_amts))

        ntm_amts = tn_amts + fn_amts
        revenue_amts = tn_amts
        default_loss_amts = fn_amts
        opportunity_loss_amts = fp_amts
        mislabel_loss_amts = default_loss_amts + opportunity_loss_amts

        metrics_ls = (
            ntm_amts,
            revenue_amts,
            default_loss_amts,
            opportunity_loss_amts,
            mislabel_loss_amts,
        )
        n = len(y)
        business_metrics = tuple(np.sum(k) for k in metrics_ls)
        business_metrics_avg = tuple(k / n for k in business_metrics)
        business_metrics_se = tuple(np.std(k) / np.sqrt(n) for k in metrics_ls)
    else:
        confusion_amt = None
        business_metrics = None
        business_metrics_avg = None
        business_metrics_se = None

    return (
        confusion,
        confusion_amt,
        business_metrics,
        business_metrics_avg,
        business_metrics_se,
    )


def bld_classification_report(
    model: object,
    x: pd.DataFrame,
    y: Union[list, np.array, pd.Series],
    thresholds: list,
    classes: list = [0, 1],
    **kwargs,
):
    """
    Build classification report showing metrics by thresholds and classes. Build four
    tables:
    1. business_rpt
        - ntm
        - revenue
        - default_loss
        - opportunity_loss
        - mislabel_loss
    2. business_stats_rpt
        - ntm_avg
        - ntm_se
        - default_loss_avg
        - default_loss_se
        - mislabel_loss_avg
        - mislabel_loss_se
    3. confusion_rpt
        - tn
        - fp
        - fn
        - tp
        - tpr
        - fpr
        - precision
    4. evaluation_rpt
        - roc_auc
        - gini
        - log_loss
        - brier

    Examples
    --------
    business_rpt, business_stats_rpt, confusion_rpt, evaluation_rpt = (
        bld_classification_report(
            clf_lgb,
            xtest_processed,
            ytest,
            thresholds=np.arange(0.1, 1, 0.1),
            loan_amts=xtest["loan_amount"].astype(float),
            revenue_pct=0.04,
        )
    )
    """
    y_prob = predict_probabilities(model, x, return_both=True)

    business_rpt = []
    business_stats_rpt = []
    confusion_rpt = []
    evaluation_rpt = []

    for cls in classes:

        # Class-specific, threshold-agnositic metrics
        roc_auc = metrics.roc_auc_score(y, y_prob[:, cls])
        gini = 2 * roc_auc - 1
        avg_precision = metrics.average_precision_score(y, y_prob[:, cls])
        log_loss = metrics.log_loss(y, y_prob[:, cls])
        brier = metrics.brier_score_loss(y, y_prob[:, cls])
        evaluation_rpt.append(
            {
                "class": cls,
                "roc_auc": roc_auc,
                "gini": gini,
                "avg_precision": avg_precision,
                "log_loss": log_loss,
                "brier": brier,
            }
        )

        for thres in thresholds:
            match cls:
                case 0:
                    y_pred = np.where(y_prob[:, cls] < thres, cls, 1 - cls)
                case 1:
                    y_pred = np.where(y_prob[:, cls] >= thres, cls, 1 - cls)

            confusion = calc_confusion_metrics(y, y_pred)[0]

            if len(kwargs) > 0:
                (
                    _,
                    _,
                    business_metrics,
                    business_metrics_avg,
                    business_metrics_se,
                ) = calc_confusion_metrics(y, y_pred, **kwargs)

                business_rpt.append(
                    {
                        "class": cls,
                        "threshold": thres,
                        "ntm": business_metrics[0],
                        "revenue": business_metrics[1],
                        "default_loss": business_metrics[2],
                        "opportunity_loss": business_metrics[3],
                        "mislabel_loss": business_metrics[4],
                    }
                )
                business_stats_rpt.append(
                    {
                        "class": cls,
                        "threshold": thres,
                        "ntm_avg": business_metrics_avg[0],
                        "ntm_se": business_metrics_se[0],
                        "default_loss_avg": business_metrics_avg[2],
                        "default_loss_se": business_metrics_se[2],
                        "mislabel_loss_avg": business_metrics_avg[4],
                        "mislabel_loss_se": business_metrics_se[4],
                    }
                )

            tn, fp, fn, tp = confusion
            tpr = tp / (fn + tp)
            fpr = fp / (tn + fp)
            precision = tp / (fp + tp)
            confusion_rpt.append(
                {
                    "class": cls,
                    "threshold": thres,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                    "tpr": tpr,
                    "fpr": fpr,
                    "precision": precision,
                }
            )

    business_rpt = pd.DataFrame(business_rpt)
    business_stats_rpt = pd.DataFrame(business_stats_rpt)
    confusion_rpt = pd.DataFrame(confusion_rpt)
    evaluation_rpt = pd.DataFrame(evaluation_rpt)

    return business_rpt, business_stats_rpt, confusion_rpt, evaluation_rpt


def rank_probabilities(
    y: pd.Series,
    y_prob: Union[list, np.array, pd.Series],
    greater_is_better: bool = False,
    cls_map: dict = None,  # {0: "non-default", 1: "default"},
) -> pd.DataFrame:
    """
    Rank predicted probabilities of the target class and calculate their percentile and
    decile.
    """
    pdf = pd.DataFrame(
        index=y.index,
        columns=["y", "y_prob", "rank_pct", "percentile", "decile"],
    )
    pdf["y"] = y.map(cls_map) if cls_map is not None else y
    pdf["y_prob"] = y_prob
    ascending = not greater_is_better
    pdf["rank_pct"] = pdf["y_prob"].rank(pct=True, ascending=ascending)
    pdf["percentile"] = (
        pd.qcut(pdf["rank_pct"], 100, labels=False, duplicates="drop") + 1
    )
    pdf["decile"] = pd.qcut(pdf["rank_pct"], 10, labels=False) + 1

    return pdf.sort_values("rank_pct").reset_index(drop=False)


def calc_lift(
    y: pd.Series,
    y_prob: Union[list, np.array, pd.Series],
    targeted_class: Union[int, str],
    greater_is_better: bool = False,
    by: str = "decile",
    **kwargs,
) -> pd.DataFrame:
    """
    Calculate lift (and cumulative gain) by percentile or decile. A lift chart can help
    decide the cutoff threshold (i.e. y_prob value) for e.g. issuing loans below a
    threshold to reduce the number of defaults.

    Parameters
    ----------
    targeted_class : int, str
        The class label of the desired outcome (e.g. 0 for non-default, 1 for default)
    by: Specify whether to calculate by 'percentile' or 'decile'

    Returns:
    - lift_df: DataFrame containing lift information by percentile or decile
    """
    if by not in ["percentile", "decile"]:
        raise ValueError("Parameter 'by' must be either 'percentile' or 'decile'")

    pdf = rank_probabilities(y, y_prob, greater_is_better, **kwargs)
    target_col = "y"
    proba_col = "y_prob"
    bin_col = "percentile" if by == "percentile" else "decile"

    # Calculate the positive rate in each bin
    grouped = (
        pdf.groupby(bin_col)
        .agg(
            total=(bin_col, "size"),
            targeted=(target_col, lambda x: (x == targeted_class).sum()),
            rate=(target_col, lambda x: (x == targeted_class).mean()),
            min_proba=(proba_col, "min"),
            max_proba=(proba_col, "max"),
        )
        .reset_index()
    )
    grouped["threshold_proba"] = (
        grouped["min_proba"] if greater_is_better else grouped["max_proba"]
    )

    # Calculate the overall targeted class rate
    overall_rate = (pdf[target_col] == targeted_class).mean()

    # Calculate lift for each bin
    grouped["lift"] = grouped["rate"] / overall_rate

    # Cumulative rate and gain
    grouped["cumulative_total"] = grouped["total"].cumsum()
    grouped["cumulative_targeted"] = grouped["targeted"].cumsum()
    grouped["cumulative_rate"] = (
        grouped["cumulative_targeted"] / grouped["cumulative_total"]
    )
    grouped["cumulative_rate_pct_diff"] = grouped["cumulative_rate"] / overall_rate - 1
    total_targeted = grouped["targeted"].sum()
    grouped["cumulative_gain"] = grouped["cumulative_targeted"] / total_targeted

    return grouped, overall_rate


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


def assign_loan_amount_groups(
    loans_df, min_size=1000, group_keys=["model_version", "segment"]
):
    """
    Adds a `loan_amount_group` column with inclusive range labels, ensuring each bin has at least `min_size` rows.
    Groups identical loan_amounts together and avoids small final bins by merging them back.
    """
    df = loans_df.copy()
    results = []

    for group_values, group in df.groupby(group_keys):
        group = group.sort_values("loan_amount").reset_index(drop=True)
        group["loan_amount_group"] = None

        running_bin = []
        running_count = 0
        bin_dfs = []  # store all bins temporarily
        bin_id = 0

        # group identical loan_amounts together
        for value, subdf in group.groupby("loan_amount", sort=True):
            running_bin.append(subdf)
            running_count += len(subdf)

            if running_count >= min_size:
                bin_df = pd.concat(running_bin)
                bin_dfs.append(bin_df)
                running_bin = []
                running_count = 0
                bin_id += 1

        # Handle remainder
        if running_bin:
            remainder_df = pd.concat(running_bin)
            if bin_dfs:
                # Merge with previous bin
                bin_dfs[-1] = pd.concat([bin_dfs[-1], remainder_df])
            else:
                # First bin still under min_size — keep as-is
                bin_dfs.append(remainder_df)

        # Label bins
        for i, bin_df in enumerate(bin_dfs):
            left = bin_df["loan_amount"].min()
            right = bin_df["loan_amount"].max()
            label = f"{i + 1:02d}. [{left:.2f}, {right:.2f}]"
            bin_df["loan_amount_group"] = label
            results.append(bin_df)

    return pd.concat(results).reset_index(drop=True)
