from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .config import CV_FOLDS, FIGURES_DIR, METRICS_DIR, MODELS_DIR, RANDOM_STATE
from .data import load_raw_data, make_train_test_split, prepare_features_target, save_processed_dataset
from .evaluation import (
    error_analysis_frame,
    evaluate_model,
    find_best_threshold,
    threshold_predictions,
)
from .explainability import native_importance_frame, permutation_importance_frame, save_shap_summary_if_available
from .models import get_model_specs
from .plots import (
    plot_calibration,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_model_performance,
    plot_nn_learning_curve,
    plot_pca_explained_variance,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_target_distribution,
    plot_top_feature_correlations,
)
from .preprocessing import build_model_pipeline


def _ensure_dirs():
    for path in [MODELS_DIR, FIGURES_DIR, METRICS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _json_safe_params(params: dict) -> dict:
    safe = {}
    for key, value in params.items():
        safe[key] = str(value) if key == "pca" else value
    return safe


def _slug(name: str) -> str:
    return name.lower().replace("-", "").replace(" ", "_")


def run_experiment(fast: bool = False) -> dict:
    _ensure_dirs()
    df = load_raw_data()
    X, y = prepare_features_target(df)
    raw_train, raw_test, y_train, y_test = make_train_test_split(df, y)
    X_train, y_train = prepare_features_target(raw_train)
    X_test, y_test = prepare_features_target(raw_test)

    save_processed_dataset()
    plot_target_distribution(df)
    plot_correlation_heatmap(df)
    plot_top_feature_correlations(df)
    plot_pca_explained_variance(X)

    cv = StratifiedKFold(n_splits=CV_FOLDS if not fast else 3, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
        "accuracy": "accuracy",
    }

    metrics_rows = []
    cv_rows = []
    model_results = {}
    best_estimators = {}

    for spec in get_model_specs():
        pipeline = build_model_pipeline(spec.estimator)
        param_grid = spec.param_grid
        if fast:
            param_grid = {key: values[:1] if isinstance(values, list) else values for key, values in param_grid.items()}

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            refit="roc_auc",
            cv=cv,
            n_jobs=-1,
            return_train_score=True,
        )
        start = time.perf_counter()
        search.fit(X_train, y_train)
        training_time = time.perf_counter() - start

        model = search.best_estimator_
        metrics, y_proba = evaluate_model(model, X_test, y_test)
        metrics["model"] = spec.name
        metrics["training_time_seconds"] = training_time
        metrics["best_params"] = _json_safe_params(search.best_params_)
        metrics_rows.append(metrics)
        model_results[spec.name] = {"model": model, "y_proba": y_proba, "metrics": metrics}
        best_estimators[spec.name] = model

        cv_result = pd.DataFrame(search.cv_results_)
        cv_result["model"] = spec.name
        keep_cols = [
            "model",
            "rank_test_roc_auc",
            "mean_test_roc_auc",
            "std_test_roc_auc",
            "mean_test_recall",
            "mean_test_f1",
            "mean_test_accuracy",
            "params",
        ]
        cv_rows.append(cv_result[keep_cols])

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["roc_auc", "recall"], ascending=False)
    cv_df = pd.concat(cv_rows, ignore_index=True).sort_values(["mean_test_roc_auc", "mean_test_recall"], ascending=False)
    metrics_df.to_csv(METRICS_DIR / "model_metrics.csv", index=False)
    cv_df.to_csv(METRICS_DIR / "cross_validation_results.csv", index=False)

    best_name = metrics_df.iloc[0]["model"]
    best_model = model_results[best_name]["model"]
    best_proba = model_results[best_name]["y_proba"]
    best_threshold, threshold_table = find_best_threshold(y_test, best_proba)
    threshold_table.to_csv(METRICS_DIR / "threshold_tuning.csv", index=False)

    tuned_metrics = metrics_df.copy()
    tuned_best_metrics = evaluate_model(best_model, X_test, y_test, best_threshold)[0]
    tuned_best_metrics["model"] = f"{best_name} (threshold tuned)"
    tuned_best_metrics["training_time_seconds"] = metrics_df.iloc[0]["training_time_seconds"]
    tuned_best_metrics["best_params"] = metrics_df.iloc[0]["best_params"]
    tuned_metrics = pd.concat([tuned_metrics, pd.DataFrame([tuned_best_metrics])], ignore_index=True)
    tuned_metrics.to_csv(METRICS_DIR / "model_metrics_with_threshold_tuning.csv", index=False)

    model_payload = {
        "model": best_model,
        "threshold": best_threshold,
        "feature_names": list(X.columns),
        "model_name": best_name,
        "metrics": tuned_best_metrics,
    }
    joblib.dump(model_payload, MODELS_DIR / "best_model.joblib")

    all_probs = {name: values["y_proba"] for name, values in model_results.items()}
    plot_roc_curves(model_results, y_test)
    plot_precision_recall_curves(model_results, y_test)
    plot_calibration(y_test, all_probs)
    plot_model_performance(metrics_df)
    plot_confusion_matrix(y_test, threshold_predictions(best_proba, best_threshold), best_name)
    plot_nn_learning_curve(model_results.get("Neural Network", {}).get("model", best_model))

    permutation_df = permutation_importance_frame(best_model, X_test, y_test)
    permutation_df.to_csv(METRICS_DIR / "permutation_importance.csv", index=False)
    plot_feature_importance(
        permutation_df,
        "Permutation Importance for Best Model",
        "permutation_importance.png",
    )
    shap_generated = save_shap_summary_if_available(best_model, X_test.sample(min(80, len(X_test)), random_state=RANDOM_STATE))

    for name, estimator in best_estimators.items():
        native_df = native_importance_frame(estimator, list(X.columns))
        if native_df is None:
            continue
        slug = _slug(name)
        native_df.to_csv(METRICS_DIR / f"{slug}_native_importance.csv", index=False)
        plot_feature_importance(
            native_df,
            f"Native Feature Importance: {name}",
            f"{slug}_native_importance.png",
        )
        if name == "Random Forest":
            native_df.to_csv(METRICS_DIR / "tree_feature_importance.csv", index=False)
            plot_feature_importance(
                native_df,
                "Tree-Based Feature Importance",
                "feature_importance.png",
            )
        if name == "Logistic Regression":
            native_df.to_csv(METRICS_DIR / "logistic_regression_coefficients.csv", index=False)
            plot_feature_importance(
                native_df,
                "Logistic Regression Coefficients",
                "logistic_regression_coefficients.png",
            )

    errors = error_analysis_frame(raw_test.reset_index(drop=True), y_test.reset_index(drop=True), best_proba, best_threshold)
    errors.to_csv(METRICS_DIR / "error_analysis.csv", index=False)

    summary = {
        "best_model": best_name,
        "best_threshold": best_threshold,
        "best_metrics": tuned_best_metrics,
        "artifacts": {
            "model": str(MODELS_DIR / "best_model.joblib"),
            "metrics": str(METRICS_DIR / "model_metrics.csv"),
            "figures": str(FIGURES_DIR),
        },
        "shap_summary_generated": shap_generated,
    }
    with open(METRICS_DIR / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


if __name__ == "__main__":
    run_experiment()
