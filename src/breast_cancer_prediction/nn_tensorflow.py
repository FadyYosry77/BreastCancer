"""Optional TensorFlow neural network trainer.

The main experiment uses scikit-learn's MLPClassifier so the project can run in
lightweight environments. This module provides a deeper Keras model with
dropout, batch normalization, early stopping, and learning-rate scheduling when
TensorFlow is available.
"""

from __future__ import annotations

from pathlib import Path

import joblib

from .config import MODELS_DIR, RANDOM_STATE
from .data import load_raw_data, make_train_test_split, prepare_features_target
from .preprocessing import build_preprocessor


def train_keras_model(epochs: int = 200, batch_size: int = 16, output_path: Path | None = None):
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras import callbacks, layers, models, optimizers

    tf.keras.utils.set_random_seed(RANDOM_STATE)
    df = load_raw_data()
    X, y = prepare_features_target(df)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)

    preprocessor = build_preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train, y_train)
    X_test_scaled = preprocessor.transform(X_test)

    model = models.Sequential(
        [
            layers.Input(shape=(X_train_scaled.shape[1],)),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.30),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.20),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="roc_auc"), tf.keras.metrics.Recall(name="recall")],
    )

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_roc_auc", mode="max", patience=25, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10),
        ],
        verbose=0,
    )
    metrics = dict(zip(model.metrics_names, model.evaluate(X_test_scaled, y_test, verbose=0), strict=False))

    output_path = output_path or MODELS_DIR / "keras_neural_network.keras"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    joblib.dump(preprocessor, MODELS_DIR / "keras_preprocessor.joblib")

    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("Keras Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Keras Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODELS_DIR.parent / "reports" / "figures" / "keras_training_history.png", dpi=180)
    plt.close()
    return {"model_path": str(output_path), "metrics": metrics}
