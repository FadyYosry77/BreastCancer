import pandas as pd

from breast_cancer_prediction.data import load_raw_data, make_train_test_split, prepare_features_target


def test_load_raw_data_has_expected_shape():
    df = load_raw_data()
    assert df.shape[0] == 569
    assert "diagnosis" in df.columns


def test_prepare_features_target_removes_id_and_encodes_target():
    df = load_raw_data()
    X, y = prepare_features_target(df)
    assert "id" not in X.columns
    assert "diagnosis" not in X.columns
    assert set(y.unique()) == {0, 1}
    assert isinstance(X, pd.DataFrame)


def test_stratified_split_preserves_classes():
    df = load_raw_data()
    X, y = prepare_features_target(df)
    _, _, y_train, y_test = make_train_test_split(X, y)
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}
