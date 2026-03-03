import os
import pandas as pd

from modelling import ModelPipeline
from helper import logging_config
import evaluation as eval
import classification.ml_plotting as clplt

logger = logging_config.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
model_folder = f"{current_dir}/models"
if not os.path.exists(model_folder):
    os.makedirs(model_folder)


def main():
    current_dir = os.path.dirname(__file__)
    filepath = os.path.join(current_dir, "data/kiva_loans_cleaned.parquet")
    pdf = pd.read_parquet(filepath)

    pipeline = ModelPipeline(
        pdf,
        tgt_col="has_defaulted",
        inference=True,
        col_transformer=f"{model_folder}/col_transformer.pkl",
        selected_feature_names=f"{model_folder}/selected_feature_names.pkl",
        model=f"{model_folder}/model.pkl",
        seasonal_date_cols=["loan_raised_date"],
    )
    pipeline.run()

    # df = pd.DataFrame(pipeline.predictions, columns=["y_pred"])
    df = pd.DataFrame({"y": pdf["has_defaulted"], "y_pred": pipeline.predictions})
    clplt.plot_prediction_distribution(
        df, saveas_filename="pred_distribution_inference"
    )

    # ADHOC:
    # total $ loaned and defaulted
    # if model was in place, $ defaulted that could have been reinvested
    # based on avg loan size, # of additional ppl who could have received loans
    df = pd.DataFrame(
        {
            "y": pdf["has_defaulted"],
            "y_pred": pipeline.predictions,
            "loan_amt": pipeline.pdf["loan_amount_usd_final"],
        }
    )
    df["default_loss_amt"] = df["y"] * df["loan_amt"]
    df_ranked = eval.rank_probabilities(df["y"], df["y_pred"])
    df_ranked = df_ranked.merge(
        df[["loan_amt", "default_loss_amt"]], left_on="index", right_index=True
    )
    grouped = (
        df_ranked.groupby("decile")
        .agg(
            num_loans_total=("loan_amt", "count"),
            num_loans_defaulted=("y", "sum"),
            loan_amt_avg=("loan_amt", "mean"),
            total_loan_amt=("loan_amt", "sum"),
            total_default_loss_amt=("default_loss_amt", "sum"),
        )
        .reset_index(drop=False)
    )
    grouped["proportion_default_loss"] = (
        grouped["total_default_loss_amt"] / grouped["total_loan_amt"]
    )
    grouped["cumulative_total"] = grouped["total_loan_amt"].cumsum()
    grouped["cumulative_loss"] = grouped["total_default_loss_amt"].cumsum()
    grouped["cumulative_rate"] = (
        grouped["cumulative_loss"] / grouped["cumulative_total"]
    )
    grouped["num_loans_defaulted"] = grouped["num_loans_defaulted"].astype(int)
    grouped["cumulative_rate"] = grouped["cumulative_rate"].round(4)

    baseline_val = grouped.loc[grouped["decile"] == 10, "cumulative_rate"].values[0]
    clplt.plot_lift_curve(
        grouped,
        baseline_val,
        show_rate=True,
        title="Loan Amounts",
        saveas_filename="lift_chart_loan_amt_inference",
        y_str="proportion_default_loss",
        y2_1_str="num_loans_defaulted",
        y2_2_str="cumulative_rate",
    )
    logger.info(f"Loan amount rates:\n{grouped}")
    grouped.to_csv(f"{current_dir}/data/loan_amt_rates.csv", index=False)


if __name__ == "__main__":
    main()

    # TODO: debugging overfitting
    # 1. check predictive performance on in-sample test set but for data > split_date
    # 2. feature-drift analysis
