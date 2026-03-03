"""
Preprocessing Functions
- train-test splitting
- feature engineering
- feature encoding and generation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from helper import utils
from helper import logging_config

logger = logging_config.getLogger(__name__)


def train_test_split_by_time(
    x: pd.DataFrame, y: pd.Series, split_date: str, date_col: str
) -> tuple:
    """
    Split x (features) and y (target variable) by date.
    """
    # Order data for time-series cross validation (during hyperparameter tuning)
    x = x.sort_values(by=date_col)
    y = y.reindex(x.index).reset_index(drop=True)
    x = x.reset_index(drop=True)

    cond_train = x[date_col] < split_date
    xtrain = x[cond_train]
    xtest = x[~cond_train]
    ytrain = y[cond_train]
    ytest = y[~cond_train]

    train_size = len(ytrain)
    test_size = len(ytest)
    train_pct = train_size / len(y)
    test_pct = test_size / len(y)

    logger.info(f"Train, test sizes: [{train_size}, {test_size}]")
    logger.info(f"Train, test pct: [{train_pct:.4f}, {test_pct:.4f}]")
    logger.info(
        f"Train, test target avg: [{np.mean(ytrain):.4f}, {np.mean(ytest):.4f}]"
    )

    return xtrain, xtest, ytrain, ytest


def divide_by_zero(pdf: pd.DataFrame, col1: str, col2: str) -> pd.Series:
    """
    Handle division by zero
    """
    return np.divide(
        pdf[col1],
        pdf[col2],
        out=np.full_like(pdf[col1], np.nan),
        where=(pdf[col2] != 0),
    )


class FeatureEngineering:
    """
    Feature engineering
    """

    def __init__(self, pdf: pd.DataFrame, date_cols: list[str]):
        self.pdf = pdf
        self.date_cols = date_cols

    def create_harmonic_features(
        self,
        date_col: str,
        seasonal_features: list = [
            "dayofweek",
            "dayofyear",
            "week",
            "month",
            "quarter",
        ],
    ) -> None:
        """
        Create harmonic features for seasonal features.
        """
        feat_cols = []
        seasonal_feat_period_name = {
            "dayofweek": [7, "dow"],
            "dayofyear": [365.25, "doy"],
            "week": [52, "wk"],
            "month": [12, "mth"],
            "quarter": [4, "qtr"],
        }
        for feat in seasonal_features:
            try:
                period = seasonal_feat_period_name[feat][0]
                feat_col = f"{date_col}_{seasonal_feat_period_name.get(feat, feat)[1]}"
                feat_cols.append(feat_col)

                if feat in ["weekofyear", "week"]:
                    self.pdf[feat_col] = self.pdf[date_col].dt.isocalendar()[feat]
                else:
                    self.pdf[feat_col] = self.pdf[date_col].dt.__getattribute__(feat)

                self.pdf[f"{feat_col}_sin"] = np.sin(
                    2 * np.pi * self.pdf[feat_col] / period
                )
                self.pdf[f"{feat_col}_cos"] = np.cos(
                    2 * np.pi * self.pdf[feat_col] / period
                )
            except ValueError:
                logger.info(f"Invalid cyclical feature: {feat}")
                continue

        # Drop the original cyclical features
        self.pdf = self.pdf.drop(columns=feat_cols)

        # Convert from Float64 to float64 for compatibility with sklearn, shap
        self.pdf = self.pdf.astype(
            {
                col: "float64"
                for col in self.pdf.select_dtypes(include=["Float64"]).columns
            }
        )

    def run(self) -> None:
        """
        Bespoke feature engineering
        """
        # TODO: move this to a higher-level file
        self.pdf["loan_amount_to_term_ratio"] = divide_by_zero(
            self.pdf,
            "loan_amount_usd_final",
            "lender_term",
        )

        # Seasonality: harmonic regression on date columns
        for date_col in self.date_cols:
            try:
                self.create_harmonic_features(date_col)
            except TypeError:
                logger.info(f"Invalid date column: {date_col}")


class FeatureGenerator:
    """
    Encode categorical variables and generate final features.

    Attributes
    ----------
    numeric: DataFrame
        numeric data
    categorical: DataFrame
        categorical data
    ordenc: OrdinalEncoder
    ordinal_cols: list
    ohe: OneHotEncoder
    nominal_cols: list
    col_transformer: ColumnTransformer
    features: DataFrame
        final transformed features
    """

    def __init__(
        self,
        pdf: pd.DataFrame,
        excl_cols: list,
        col_transformer: ColumnTransformer = None,
        ordinal_cols: list = None,
    ):
        self.pdf = pdf
        self.excl_cols = excl_cols
        self.col_transformer = col_transformer

        self.feature_cols = self.extract_feature_cols()
        self.pdf = self.pdf[self.feature_cols]

        self.numeric: pd.DataFrame = None
        self.categorical: pd.DataFrame = None
        self.ordenc: OrdinalEncoder = None
        self.ordinal_cols = ordinal_cols
        self.ohe: OneHotEncoder = None
        self.nominal_cols: list = None

    def extract_feature_cols(self) -> list:
        """
        Extract feature columns from all columns excluding excl_cols
        """
        return [f for f in self.pdf.columns if f not in self.excl_cols]

    def split_numeric_categorical(self, numeric_dtypes: list = [np.number]) -> None:
        """
        Split data into numeric and categorical data

        Parameters
        ----------
        numeric_dtypes : list
            list of numeric datatypes
        """
        logger.info(f"Input data unique dtypes: {self.pdf.dtypes.unique()}")

        self.numeric = self.pdf.select_dtypes(include=numeric_dtypes)
        self.categorical = self.pdf.select_dtypes(exclude=numeric_dtypes)

        logger.info(f"Numeric features unique dtypes: {self.numeric.dtypes.unique()}")
        logger.info(
            f"Categorical features unique dtypes: {self.categorical.dtypes.unique()}"
        )

    def _filter_by_type(self, ls: list, type_of: type) -> list:
        """
        Filter list by data type

        Parameters
        ----------
        ls : list
        type_of : type
            [int, str]
        """
        ls_of_type = []
        for n in ls:
            try:
                ls_of_type.append(int(n))
            except (ValueError, TypeError):
                if isinstance(n, type_of):
                    ls_of_type.append(n)

        return ls_of_type

    def _extract_ordinal_cols(self) -> list:
        """
        Extract ordinal features from categorical features df. For each feature, get
        unique categories. If all categories (or all except one) are int, flag as
        ordinal; otherwise, non-ordinal
        """
        ordinal_cols = []
        for f in self.categorical.columns:
            categories = self.categorical[f].dropna().unique()
            numeric = self._filter_by_type(categories, int)
            if len(numeric) > 0 and len(numeric) >= len(categories) - 1:
                ordinal_cols.append(f)

        logger.info(f"Number of ordinal features: {len(ordinal_cols)}")

        return ordinal_cols

    def encode_categorical(self, max_categories: int = 11, **kwargs) -> None:
        """
        Perform one-hot encoding or ordinal encoding on categorical features.
        Automatically determine ordinal and non-ordinal features
        """
        if self.ordinal_cols is None:
            self.ordinal_cols = self._extract_ordinal_cols()

        self.nominal_cols = [
            f for f in self.categorical.columns if f not in self.ordinal_cols
        ]
        logger.info(f"Number of non-ordinal features: {len(self.nominal_cols)}")

        if len(self.ordinal_cols) > 0:
            self.ordenc = OrdinalEncoder(
                dtype=int,
                handle_unknown="use_encoded_value",
                unknown_value=99,
                max_categories=max_categories,
                **kwargs,
            )

        if len(self.nominal_cols) > 0:
            self.ohe = OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                max_categories=max_categories,
                sparse_output=False,
                **kwargs,
            )

    def _replace_ohe_feature_separator(
        self, separator: str = "|", ohe_name: str = "OneHotEncoder"
    ) -> list:
        """
        Replace the OneHotEncoder "_" with "|" used to separate the feature name and
        category name
        """
        custom_names = []
        transformer_names = [
            t[1].__class__.__name__ for t in self.col_transformer.transformers_
        ]
        if ohe_name not in transformer_names:
            return self.col_transformer.get_feature_names_out(self.feature_cols)

        for transformer in self.col_transformer.transformers_:
            if transformer[1].__class__.__name__ == ohe_name:
                ohe = transformer[1]
                names_in = ohe.feature_names_in_
                names_out = ohe.get_feature_names_out(names_in)

                for name_out in names_out:
                    for name_in in names_in:
                        if name_out.startswith(name_in):
                            custom_names.append(
                                name_out.replace(name_in + "_", name_in + separator, 1)
                            )
                            break
            else:
                custom_names.extend(transformer[1].get_feature_names_out())

        return custom_names

    def run(
        self,
        inference: bool = False,
        **kwargs,
    ) -> None:
        """
        Generate features:
        - split to numeric, categorical
        - split categorical to ordinal, non-ordinal
        - Encoding categorical features
        - Combine numeric, ordinal, non-ordinal

        Parameters
        ----------
        df : DataFrame
            features data
        inference : bool
            True: use fitted ColumnTransformer for transformation
            False: create and fit ColumnTransformer

        Returns
        -------
        df_out : array | DataFrame
            transformed features
        """
        if not inference:
            self.split_numeric_categorical()
            self.encode_categorical(
                **utils.filter_kwargs(self.encode_categorical, kwargs)
            )

            transformers = []
            if self.ordenc is not None:
                transformers.append(("ordenc", self.ordenc, self.ordinal_cols))
            if self.ohe is not None:
                transformers.append(("ohe", self.ohe, self.nominal_cols))

            self.col_transformer = (
                ColumnTransformer(
                    transformers,
                    remainder="passthrough",
                    verbose_feature_names_out=False,
                )
                .set_output(transform="pandas")
                .fit(self.pdf)
            )

        # Ensure all transformer input columns are present in the input data
        feature_names_in = self.col_transformer.feature_names_in_
        if set(self.pdf.columns) - set(feature_names_in):
            for col in feature_names_in:
                if col not in self.pdf.columns:
                    self.pdf[col] = pd.NA

        self.pdf = self.col_transformer.transform(self.pdf)
        self.pdf.columns = self._replace_ohe_feature_separator()
