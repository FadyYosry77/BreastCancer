from __future__ import annotations

from dataclasses import dataclass

from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from .config import RANDOM_STATE


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    param_grid: dict


def _pca_options() -> list:
    return [
        "passthrough",
        PCA(n_components=0.95, svd_solver="full", random_state=RANDOM_STATE),
    ]


def get_model_specs(random_state: int = RANDOM_STATE) -> list[ModelSpec]:
    specs = [
        ModelSpec(
            name="Logistic Regression",
            estimator=LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                solver="liblinear",
                random_state=random_state,
            ),
            param_grid={
                "pca": _pca_options(),
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
            },
        ),
        ModelSpec(
            name="Random Forest",
            estimator=RandomForestClassifier(
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
            param_grid={
                "pca": ["passthrough"],
                "classifier__n_estimators": [200, 400],
                "classifier__max_depth": [None, 4, 8],
                "classifier__min_samples_leaf": [1, 3],
            },
        ),
        ModelSpec(
            name="Gradient Boosting",
            estimator=GradientBoostingClassifier(random_state=random_state),
            param_grid={
                "pca": ["passthrough"],
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.03, 0.1],
                "classifier__max_depth": [2, 3],
            },
        ),
        ModelSpec(
            name="K-Nearest Neighbors",
            estimator=KNeighborsClassifier(),
            param_grid={
                "pca": _pca_options(),
                "classifier__n_neighbors": [3, 5, 9, 15],
                "classifier__weights": ["uniform", "distance"],
                "classifier__p": [1, 2],
            },
        ),
        ModelSpec(
            name="SVM",
            estimator=SVC(class_weight="balanced", probability=True, random_state=random_state),
            param_grid={
                "pca": _pca_options(),
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__kernel": ["linear", "rbf"],
                "classifier__gamma": ["scale", "auto"],
            },
        ),
        ModelSpec(
            name="Neural Network",
            estimator=MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=0.001,
                batch_size=16,
                early_stopping=True,
                learning_rate="adaptive",
                max_iter=500,
                n_iter_no_change=25,
                random_state=random_state,
                validation_fraction=0.15,
            ),
            param_grid={
                "pca": _pca_options(),
                "classifier__hidden_layer_sizes": [(64, 32), (128, 64, 32)],
                "classifier__alpha": [0.0001, 0.001],
                "classifier__learning_rate_init": [0.001, 0.0005],
            },
        ),
    ]

    try:
        from xgboost import XGBClassifier

        specs.insert(
            3,
            ModelSpec(
                name="XGBoost",
                estimator=XGBClassifier(
                    eval_metric="logloss",
                    n_jobs=-1,
                    random_state=random_state,
                    tree_method="hist",
                ),
                param_grid={
                    "pca": ["passthrough"],
                    "classifier__n_estimators": [100, 250],
                    "classifier__max_depth": [2, 3],
                    "classifier__learning_rate": [0.03, 0.1],
                    "classifier__subsample": [0.8, 1.0],
                },
            ),
        )
    except Exception:
        pass

    return specs
