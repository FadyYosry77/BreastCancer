from sklearn.linear_model import LogisticRegression

from app.predict import predict_from_measurements
from breast_cancer_prediction.data import load_raw_data, make_train_test_split, prepare_features_target
from breast_cancer_prediction.preprocessing import build_model_pipeline


def test_predict_from_measurements_returns_class_and_probability():
    df = load_raw_data()
    X, y = prepare_features_target(df)
    X_train, X_test, y_train, _ = make_train_test_split(X, y)
    model = build_model_pipeline(LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    bundle = {"model": model, "threshold": 0.5, "feature_names": list(X.columns)}

    result = predict_from_measurements(X_test.iloc[0].to_dict(), bundle)

    assert result["class_name"] in {"Benign", "Malignant"}
    assert 0 <= result["malignant_probability"] <= 1
