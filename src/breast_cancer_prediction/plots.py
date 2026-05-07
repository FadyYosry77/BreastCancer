from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CLASS_NAMES, FIGURES_DIR, TARGET_COLUMN


def _save(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_target_distribution(df: pd.DataFrame, output_dir: Path = FIGURES_DIR):
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x=TARGET_COLUMN, hue=TARGET_COLUMN, palette=["#3B82A0", "#D95F59"], legend=False)
    ax.set_title("Target Distribution")
    ax.set_xlabel("Diagnosis")
    ax.set_ylabel("Count")
    _save(output_dir / "target_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path = FIGURES_DIR):
    numeric = df.drop(columns=["id"], errors="ignore").copy()
    numeric[TARGET_COLUMN] = numeric[TARGET_COLUMN].map({"B": 0, "M": 1}).fillna(numeric[TARGET_COLUMN])
    plt.figure(figsize=(13, 10))
    sns.heatmap(numeric.corr(), cmap="vlag", center=0, square=False, cbar_kws={"shrink": 0.7})
    plt.title("Feature Correlation Heatmap")
    _save(output_dir / "correlation_heatmap.png")


def plot_top_feature_correlations(df: pd.DataFrame, output_dir: Path = FIGURES_DIR, top_n: int = 12):
    numeric = df.drop(columns=["id"], errors="ignore").copy()
    numeric[TARGET_COLUMN] = numeric[TARGET_COLUMN].map({"B": 0, "M": 1})
    corr = numeric.corr()[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values(key=abs, ascending=False).head(top_n)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=corr.values, y=corr.index, hue=corr.index, palette="crest", legend=False)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Top Feature Correlations With Malignancy")
    plt.xlabel("Correlation")
    plt.ylabel("")
    _save(output_dir / "top_feature_correlations.png")


def plot_pca_explained_variance(X, output_dir: Path = FIGURES_DIR):
    pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA())])
    pipe.fit(X)
    pca = pipe.named_steps["pca"]
    cumulative = pca.explained_variance_ratio_.cumsum()
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(cumulative) + 1), cumulative, marker="o", linewidth=2)
    plt.axhline(0.95, color="#D95F59", linestyle="--", label="95% variance")
    plt.title("PCA Cumulative Explained Variance")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.legend()
    _save(output_dir / "pca_explained_variance.png")


def plot_roc_curves(results: dict, y_test, output_dir: Path = FIGURES_DIR):
    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    for name, values in results.items():
        RocCurveDisplay.from_predictions(y_test, values["y_proba"], name=name, ax=ax)
    plt.title("ROC Curve Comparison")
    _save(output_dir / "roc_curve_comparison.png")


def plot_precision_recall_curves(results: dict, y_test, output_dir: Path = FIGURES_DIR):
    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    for name, values in results.items():
        PrecisionRecallDisplay.from_predictions(y_test, values["y_proba"], name=name, ax=ax)
    plt.title("Precision-Recall Curve Comparison")
    _save(output_dir / "precision_recall_curve_comparison.png")


def plot_confusion_matrix(y_true, y_pred, model_name: str, output_dir: Path = FIGURES_DIR):
    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(cm, display_labels=[CLASS_NAMES[0], CLASS_NAMES[1]])
    display.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix: {model_name}")
    _save(output_dir / "best_model_confusion_matrix.png")


def plot_model_performance(metrics_df: pd.DataFrame, output_dir: Path = FIGURES_DIR):
    plot_df = metrics_df.melt(
        id_vars="model",
        value_vars=["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(11, 6))
    sns.barplot(data=plot_df, x="score", y="model", hue="metric", palette="tab10")
    plt.xlim(0, 1.05)
    plt.title("Model Performance Comparison")
    plt.xlabel("Score")
    plt.ylabel("")
    plt.legend(loc="lower right")
    _save(output_dir / "model_performance_comparison.png")


def plot_feature_importance(importance_df: pd.DataFrame, title: str, filename: str, output_dir: Path = FIGURES_DIR):
    top = importance_df.head(15).copy()
    plt.figure(figsize=(8, 5))
    sns.barplot(data=top, x="importance", y="feature", hue="feature", palette="mako", legend=False)
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("")
    _save(output_dir / filename)


def plot_calibration(y_true, probabilities_by_model: dict, output_dir: Path = FIGURES_DIR):
    plt.figure(figsize=(7, 5))
    for name, y_proba in probabilities_by_model.items():
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=8, strategy="uniform")
        plt.plot(prob_pred, prob_true, marker="o", label=name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.title("Calibration Curve")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend()
    _save(output_dir / "calibration_curve.png")


def plot_nn_learning_curve(model, output_dir: Path = FIGURES_DIR):
    classifier = model.named_steps.get("classifier")
    if not hasattr(classifier, "loss_curve_"):
        return
    plt.figure(figsize=(7, 4))
    plt.plot(classifier.loss_curve_, label="Training loss")
    if hasattr(classifier, "validation_scores_"):
        plt.plot(classifier.validation_scores_, label="Validation accuracy")
    plt.title("Neural Network Training Curve")
    plt.xlabel("Epoch")
    plt.legend()
    _save(output_dir / "neural_network_training_curve.png")
