from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip numeric columns using train-only IQR bounds."""

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        data = pd.DataFrame(X).astype(float)
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        self.lower_bounds_ = q1 - self.factor * iqr
        self.upper_bounds_ = q3 + self.factor * iqr
        self.feature_names_in_ = np.array(getattr(X, "columns", data.columns), dtype=object)
        return self

    def transform(self, X):
        data = pd.DataFrame(X, columns=getattr(X, "columns", self.feature_names_in_)).astype(float)
        clipped = data.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)
        return clipped.to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.array(input_features if input_features is not None else self.feature_names_in_, dtype=object)


def build_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("outlier_clipper", OutlierClipper()),
            ("scaler", StandardScaler()),
        ]
    )


def optional_pca(variance: float = 0.95) -> PCA:
    return PCA(n_components=variance, svd_solver="full", random_state=RANDOM_STATE)


def build_model_pipeline(estimator, use_pca: bool = False) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("pca", optional_pca() if use_pca else "passthrough"),
            ("classifier", estimator),
        ]
    )
