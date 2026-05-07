from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .config import FIGURES_DIR, RANDOM_STATE


def permutation_importance_frame(model, X, y, n_repeats: int = 20) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X,
        y,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def native_importance_frame(model, feature_names: list[str]) -> pd.DataFrame | None:
    pca_step = model.named_steps.get("pca")
    if pca_step != "passthrough":
        return None

    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
        data = {"feature": feature_names, "importance": values}
    elif hasattr(classifier, "coef_"):
        coefficients = classifier.coef_.ravel()
        data = {
            "feature": feature_names,
            "coefficient": coefficients,
            "importance": np.abs(coefficients),
        }
    else:
        return None

    return (
        pd.DataFrame(data)
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_shap_summary_if_available(model, X_sample: pd.DataFrame, output_dir=FIGURES_DIR) -> bool:
    """Save a SHAP beeswarm plot when SHAP is installed."""
    try:
        import matplotlib.pyplot as plt
        import shap
    except Exception:
        return False

    def predict_positive(values):
        frame = pd.DataFrame(values, columns=X_sample.columns)
        return model.predict_proba(frame)[:, 1]

    explainer = shap.Explainer(predict_positive, X_sample)
    shap_values = explainer(X_sample)
    shap.plots.beeswarm(shap_values, show=False, max_display=15)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close()
    return True
