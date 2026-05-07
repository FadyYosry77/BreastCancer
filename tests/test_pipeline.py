from sklearn.linear_model import LogisticRegression

from breast_cancer_prediction.data import load_raw_data, make_train_test_split, prepare_features_target
from breast_cancer_prediction.preprocessing import build_model_pipeline, build_preprocessor


def test_preprocessing_pipeline_preserves_row_count():
    df = load_raw_data()
    X, y = prepare_features_target(df)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X, y)
    assert transformed.shape[0] == X.shape[0]
    assert transformed.shape[1] == X.shape[1]


def test_model_prediction_shape():
    df = load_raw_data()
    X, y = prepare_features_target(df)
    X_train, X_test, y_train, _ = make_train_test_split(X, y)
    model = build_model_pipeline(LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    assert predictions.shape[0] == X_test.shape[0]
