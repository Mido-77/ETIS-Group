"""
train_lstm.py

Trains the BiLSTM phishing classifier on the preprocessed text/email subset.

Architecture is the "conservative" option from the parameter-count sanity
check: given only 1,412 training rows, this keeps params-per-training-
sample around ~118 rather than the ~553 the originally-planned larger
architecture would have produced. See CLAUDE.md / thesis Chapter 3.8 for
the reasoning.

Reference point: the TF-IDF + Logistic Regression baseline
(ml/baseline_sanity_check.py) scored 94.68% test accuracy, 0.985 ROC-AUC.
This LSTM should be compared against that number honestly -- on a dataset
this size, a small deep model is not guaranteed to beat a well-tuned
classical baseline, and that's a legitimate finding, not a failure.

Run from the project root (after preprocess_lstm_data.py has been run):
    python ml/train_lstm.py

Outputs:
    ml/models/lstm_phishing_classifier.keras   - best model (by val_loss)
    ml/results/training_history.png            - accuracy + loss curves (Fig 5.1, 5.2)
    ml/results/confusion_matrix.png             - confusion matrix heatmap (Fig 5.3)
    ml/results/roc_curve.png                    - ROC curve with AUC (Fig 5.4)
    ml/results/test_metrics.txt                 - full metrics report
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, Input, LSTM
from tensorflow.keras.models import Sequential

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "ml" / "data"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
RESULTS_DIR = PROJECT_ROOT / "ml" / "results"

VOCAB_SIZE = 5_000
MAX_LEN = 200
EMBED_DIM = 32
LSTM_UNITS = 16
DENSE_UNITS = 8


def load_arrays():
    X_train = np.load(DATA_DIR / "X_train.npy")
    X_val = np.load(DATA_DIR / "X_val.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    y_val = np.load(DATA_DIR / "y_val.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model() -> Sequential:
    model = Sequential([
        Input(shape=(MAX_LEN,)),
        Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True),
        Bidirectional(LSTM(LSTM_UNITS, dropout=0.3, recurrent_dropout=0.2)),
        Dropout(0.4),
        Dense(DENSE_UNITS, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = load_arrays()
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # class_weight computed from the TRAIN split specifically (980 legitimate :
    # 432 phishing, ~2.27:1), not the full subset -- matches what the model
    # actually trains on.
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weight = {0: weights[0], 1: weights[1]}
    print(f"Class weights: {class_weight}")

    model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(
            MODELS_DIR / "lstm_phishing_classifier.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=64,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    # --- Figures 5.1 / 5.2: training curves ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "training_history.png", dpi=150)
    plt.close(fig)

    # --- Test set evaluation ---
    test_proba = model.predict(X_test).ravel()
    test_pred = (test_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, test_pred)
    prec = precision_score(y_test, test_pred)
    rec = recall_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred)
    auc = roc_auc_score(y_test, test_proba)
    report = classification_report(y_test, test_pred, target_names=["legitimate", "phishing"])

    print("\n=== Test set results ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(report)
    print("Compare against baseline (TF-IDF + LogReg): 94.68% accuracy, 0.985 ROC-AUC.")

    with open(RESULTS_DIR / "test_metrics.txt", "w") as f:
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1:        {f1:.4f}\n")
        f.write(f"ROC-AUC:   {auc:.4f}\n\n")
        f.write(report)

    # --- Figure 5.3: confusion matrix ---
    cm = confusion_matrix(y_test, test_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["legitimate", "phishing"]).plot(ax=ax, cmap="Blues")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # --- Figure 5.4: ROC curve ---
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_test, test_proba, ax=ax)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved model to {MODELS_DIR / 'lstm_phishing_classifier.keras'}")
    print(f"Saved figures and metrics to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
