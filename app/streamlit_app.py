from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.predict import load_model_bundle, predict_from_measurements
from breast_cancer_prediction.config import METRICS_DIR, MODELS_DIR, RAW_DATA_PATH
from breast_cancer_prediction.data import feature_names, load_raw_data


@st.cache_resource
def cached_model():
    return load_model_bundle(MODELS_DIR / "best_model.joblib")


@st.cache_data
def cached_defaults():
    df = load_raw_data(RAW_DATA_PATH)
    features = feature_names(RAW_DATA_PATH)
    return df[features].median().to_dict(), df[features]


def main():
    st.set_page_config(page_title="Breast Cancer Prediction", page_icon="BC", layout="wide")
    st.title("Breast Cancer Prediction")
    st.warning(
        "Educational project only. This app is not medical advice and must not be used as a clinical diagnosis tool."
    )

    bundle = cached_model()
    defaults, feature_frame = cached_defaults()
    features = bundle["feature_names"]

    st.sidebar.header("Tumor Measurements")
    measurements = {}
    groups = {
        "Mean": [name for name in features if name.endswith("_mean")],
        "Standard Error": [name for name in features if name.endswith("_se")],
        "Worst": [name for name in features if name.endswith("_worst")],
    }

    for group_name, group_features in groups.items():
        with st.sidebar.expander(group_name, expanded=group_name == "Mean"):
            for feature in group_features:
                minimum = float(feature_frame[feature].min())
                maximum = float(feature_frame[feature].max())
                value = float(defaults[feature])
                measurements[feature] = st.number_input(
                    feature,
                    min_value=minimum,
                    max_value=maximum,
                    value=value,
                    step=(maximum - minimum) / 100 if maximum > minimum else 0.01,
                )

    prediction = predict_from_measurements(measurements, bundle)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Prediction")
        st.metric("Predicted class", prediction["class_name"])
        st.metric("Malignant probability", f"{prediction['malignant_probability']:.1%}")
        st.caption(f"Decision threshold: {prediction['threshold']:.2f}")
    with right:
        st.subheader("Model Summary")
        st.write(f"Loaded model: **{bundle.get('model_name', 'Best model')}**")
        metrics = bundle.get("metrics", {})
        if metrics:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "ROC-AUC": metrics.get("roc_auc"),
                            "Recall": metrics.get("recall"),
                            "Precision": metrics.get("precision"),
                            "F1": metrics.get("f1"),
                            "Specificity": metrics.get("specificity"),
                        }
                    ]
                ).round(3),
                use_container_width=True,
            )

    importance_path = METRICS_DIR / "permutation_importance.csv"
    if importance_path.exists():
        st.subheader("Important Features")
        importance = pd.read_csv(importance_path).head(8)
        st.bar_chart(importance.set_index("feature")["importance"])
        st.caption("Feature importance is estimated with permutation importance on the holdout set.")


if __name__ == "__main__":
    main()
