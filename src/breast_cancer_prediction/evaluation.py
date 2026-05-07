from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import POSITIVE_LABEL


def predict_probabilities(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return (scores - scores.min()) / (scores.max() - scores.min())
    raise TypeError("Model must expose predict_proba or decision_function.")


def threshold_predictions(probas: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (probas >= threshold).astype(int)


def classification_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_pred = threshold_predictions(y_proba, threshold)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def evaluate_model(model, X_test, y_test, threshold: float = 0.5) -> tuple[dict, np.ndarray]:
    start = time.perf_counter()
    y_proba = predict_probabilities(model, X_test)
    inference_time = time.perf_counter() - start
    metrics = classification_metrics(y_test, y_proba, threshold)
    metrics["inference_time_seconds"] = inference_time
    metrics["inference_time_ms_per_sample"] = inference_time * 1000 / len(y_test)
    return metrics, y_proba


def find_best_threshold(
    y_true,
    y_proba,
    recall_floor: float = 0.95,
    thresholds: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    """Find a threshold that prioritizes recall, then F1."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)

    rows = [classification_metrics(y_true, y_proba, float(t)) for t in thresholds]
    table = pd.DataFrame(rows)
    feasible = table[table["recall"] >= recall_floor]
    candidates = feasible if not feasible.empty else table
    best = candidates.sort_values(["f1", "recall", "specificity"], ascending=False).iloc[0]
    return float(best["threshold"]), table


def error_analysis_frame(raw_test: pd.DataFrame, y_true, y_proba, threshold: float) -> pd.DataFrame:
    y_pred = threshold_predictions(y_proba, threshold)
    errors = raw_test.copy()
    errors["actual"] = np.asarray(y_true)
    errors["predicted"] = y_pred
    errors["malignant_probability"] = y_proba
    errors["error_type"] = np.where(
        (errors["actual"] == POSITIVE_LABEL) & (errors["predicted"] == 0),
        "False Negative",
        np.where(
            (errors["actual"] == 0) & (errors["predicted"] == POSITIVE_LABEL),
            "False Positive",
            "Correct",
        ),
    )
    return errors[errors["error_type"] != "Correct"].sort_values("malignant_probability", ascending=False)
