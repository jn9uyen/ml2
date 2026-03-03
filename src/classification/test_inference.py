import os
import pandas as pd

from modelling import ModelPipeline
from helper import logging_config
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

    col_transformer = pd.read_pickle(f"{model_folder}/col_transformer.pkl")
    selected_feature_names = pd.read_pickle(f"{model_folder}/selected_feature_names.pkl")
    model = pd.read_pickle(f"{model_folder}/model.pkl")

    pipeline = ModelPipeline(
        pdf,
        tgt_col="has_defaulted",
        inference=True,
        col_transformer=col_transformer,
        selected_feature_names=selected_feature_names,
        model=model,
        seasonal_date_cols=["loan_raised_date"],
    )
    pipeline.run()

    df = pd.DataFrame(pipeline.predictions, columns=["y_pred"])
    clplt.plot_prediction_distribution(
        df, saveas_filename="pred_distribution_inference"
    )


if __name__ == "__main__":
    main()

    # TODO: debugging overfitting
    # 1. check predictive performance on in-sample test set but for data > split_date
    # 2. feature-drift analysis
