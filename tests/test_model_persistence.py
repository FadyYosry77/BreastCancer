from pathlib import Path

import joblib
import pytest
from sklearn.linear_model import LogisticRegression

from app.predict import load_model_bundle
from breast_cancer_prediction.config import MODELS_DIR
from breast_cancer_prediction.data import load_raw_data, make_train_test_split, prepare_features_target
from breast_cancer_prediction.preprocessing import build_model_pipeline


def test_saved_model_loading(tmp_path: Path):
    df = load_raw_data()
    X, y = prepare_features_target(df)
    X_train, X_test, y_train, _ = make_train_test_split(X, y)
    model = build_model_pipeline(LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)

    payload = {"model": model, "threshold": 0.5, "feature_names": list(X.columns)}
    path = tmp_path / "model.joblib"
    joblib.dump(payload, path)
    loaded = joblib.load(path)

    assert loaded["model"].predict(X_test).shape[0] == X_test.shape[0]
    assert loaded["feature_names"] == list(X.columns)


def test_generated_best_model_can_be_loaded():
    model_path = MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        pytest.skip("Training artifact has not been generated yet.")
    bundle = load_model_bundle(model_path)
    assert {"model", "threshold", "feature_names"}.issubset(bundle)
