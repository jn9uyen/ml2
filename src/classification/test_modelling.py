import os
import numpy as np
import pandas as pd

from modelling import ModelPipeline
import evaluation as eval
import classification.ml_plotting as clplt
from helper import logging_config

logger = logging_config.getLogger(__name__)
logging_config.overwrite_log()

current_dir = os.path.dirname(os.path.abspath(__file__))
model_folder = f"{current_dir}/models"
if not os.path.exists(model_folder):
    os.makedirs(model_folder)


def main():
    current_dir = os.path.dirname(__file__)
    filepath = os.path.join(current_dir, "data/kiva_loans_cleaned.parquet")
    pdf = pd.read_parquet(filepath)

    # Refinement metrics; first metric is used to tune hyperparameters
    scoring_metrics = [
        "roc_auc",
        "average_precision",
        "neg_log_loss",
        "neg_brier_score",
    ]

    pipeline = ModelPipeline(
        pdf,
        tgt_col="has_defaulted",
        inference=False,
        excl_feature_cols=[
            "id",
            "loan_defaulted_date",
            "loan_disbursed_date",
            "loan_ended_date",
            "first_repayment_due_date",
            "final_repayment_due_date",
            "first_collection_date",
            "last_collection_date",
            "loan_raised_date",
            "loan_stage",
            "first_repayment_status",
            "payments_are_up_to_date",
            "total_collected",
            "total_number_of_days_delinquent",
            "number_of_collections",
            "latest_collection_amount",
            "time_period",
            # Compliance exclusions:
            "borrower_age_loan_raise_date",
            "primary_language",
            "gender",
            "is_bipoc",
            "is_disabled",
            "is_formerly_incarcerated",
            "is_immigrant",
            "is_lgbtq",
            "is_marginalized",
            "is_refugee",
            "is_veteran",
            "has_criminal_conviction",
            "",
        ],
        seasonal_date_cols=["loan_raised_date"],
        split_method=None,  # "by-time",  # None
        split_date_col="loan_disbursed_date",
        split_date="2019-01-01",
        test_size=0.3,
        max_categories=11,
        importance_type="total_gain",  # "weight", "gain", "cover", "total_gain", "total_cover"
        missing=np.nan,  # string columns with missing values are converted to pd.NA and
        # then may be encoded as a category (e.g. "feature1|nan")
        rel_imp_thres=0.03,
        cumulative_thres=0.95,
        is_only_feature_selection=False,
        tuning_iters=100,
        cross_val_method="cv",  # "time-series",
        cross_val_folds=5,
        scoring_metrics=scoring_metrics,
        random_state=123,
        n_jobs=-1,
        is_calibrate_model=True,
        calibration_method="sigmoid",
        shap_sample_size=30000,
        max_feat_display=18,
        fitted_objects=dict(
            col_transformer=f"{model_folder}/col_transformer.pkl",
            selected_feature_names=f"{model_folder}/selected_feature_names.pkl",
            model=f"{model_folder}/model.pkl",
        ),
    )
    pipeline.run()

    if not pipeline.inference:
        # Lift curve showing non-default rates
        y_prob = eval.predict_probabilities(pipeline.model, pipeline.xtest)
        y = pipeline.ytest
        # ranked_probabilities = eval.rank_probabilities(y, y_prob, greater_is_better=False)
        lift, baseline_proba_rate = eval.calc_lift(
            y, y_prob, targeted_class=0, by="decile", greater_is_better=False
        )
        clplt.plot_lift_curve(
            lift,
            baseline_proba_rate,
            show_rate=True,
            title="Rates Lift Chart (test)",
            saveas_filename="lift_chart_rates_test",
        )

        # Default rates
        lift_default, baseline_proba_rate_default = eval.calc_lift(
            y, y_prob, targeted_class=1, by="decile", greater_is_better=False
        )
        clplt.plot_lift_curve(
            lift_default,
            baseline_proba_rate_default,
            show_rate=True,
            title="Default Rates Lift Chart (test)",
            saveas_filename="lift_chart_default_rates_test",
        )

        # Non-default rate of top 6 deciles
        decile = 6
        cond = lift["decile"] <= decile
        non_default_rate = (
            lift.loc[cond, "targeted"].sum() / lift.loc[cond, "total"].sum()
        )
        reach_pct = lift.loc[cond, "total"].sum() / lift["total"].sum()
        logger.info(f"Top {decile} deciles non-default rate: {non_default_rate:.4f}")
        logger.info(f"Top {decile} deciles customer reach pct: {reach_pct:.4f}")

        # Confusion matrix for threshold = 0.38
        clplt.plot_binary_confusion_matrix(
            y, y_prob, thres=0.38, saveas_filename="confusion_thres0_5_test"
        )

        # By percentile
        lift, baseline_proba_rate = eval.calc_lift(
            y, y_prob, targeted_class=0, by="percentile", greater_is_better=False
        )
        clplt.plot_lift_curve(
            lift,
            baseline_proba_rate,
            show_rate=True,
            title="Rates Lift Chart (test)",
            saveas_filename="lift_chart_rates_test_pctl",
        )


if __name__ == "__main__":
    main()

    # TODO: debugging overfitting
    # 1. check predictive performance on in-sample test set but for data > split_date
    # 2. feature-drift analysis
