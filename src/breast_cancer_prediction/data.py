from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    ID_COLUMN,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TARGET_MAPPING,
    TEST_SIZE,
)


def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the Wisconsin breast cancer diagnostic dataset."""
    df = pd.read_csv(path)
    required_columns = {TARGET_COLUMN}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def prepare_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return numeric features and binary target without leaking identifiers."""
    y = df[TARGET_COLUMN].map(TARGET_MAPPING)
    if y.isna().any():
        unknown = sorted(df.loc[y.isna(), TARGET_COLUMN].dropna().unique())
        raise ValueError(f"Unknown target labels: {unknown}")

    drop_columns = [TARGET_COLUMN]
    if ID_COLUMN in df.columns:
        drop_columns.append(ID_COLUMN)

    X = df.drop(columns=drop_columns)
    return X, y.astype(int)


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible stratified train-test split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def save_processed_dataset(path: str | Path = PROCESSED_DATA_PATH) -> Path:
    """Save a clean processed copy with the encoded target for auditability."""
    df = load_raw_data()
    X, y = prepare_features_target(df)
    processed = X.copy()
    processed[TARGET_COLUMN] = y
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(path, index=False)
    return path


def feature_names(path: str | Path = RAW_DATA_PATH) -> list[str]:
    X, _ = prepare_features_target(load_raw_data(path))
    return list(X.columns)
