from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from breast_cancer_prediction.config import CLASS_NAMES, MODELS_DIR
from breast_cancer_prediction.evaluation import threshold_predictions


def load_model_bundle(path: str | Path = MODELS_DIR / "best_model.joblib") -> dict:
    return joblib.load(path)


def predict_from_measurements(measurements: dict, model_bundle: dict) -> dict:
    feature_names = model_bundle["feature_names"]
    X = pd.DataFrame([[measurements[name] for name in feature_names]], columns=feature_names)
    probability = float(model_bundle["model"].predict_proba(X)[0, 1])
    threshold = float(model_bundle.get("threshold", 0.5))
    prediction = int(threshold_predictions(np.array([probability]), threshold)[0])
    return {
        "prediction": prediction,
        "class_name": CLASS_NAMES[prediction],
        "malignant_probability": probability,
        "benign_probability": 1.0 - probability,
        "threshold": threshold,
    }
