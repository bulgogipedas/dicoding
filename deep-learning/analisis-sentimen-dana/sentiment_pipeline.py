"""Helper pelatihan yang dipakai ulang oleh notebook analisis sentimen DANA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


DATA_PATH = Path("data/dana_reviews_labeled.csv")
MODEL_PATH = Path("models/best_tfidf_svm.joblib")
LABEL_ORDER = ["negatif", "netral", "positif"]


@dataclass
class ExperimentResult:
    name: str
    train_accuracy: float
    test_accuracy: float
    report: str
    confusion: np.ndarray
    model: object


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["clean_text", "sentiment"])
    df = df[df["sentiment"].isin(LABEL_ORDER)].copy()
    df["clean_text"] = df["clean_text"].astype(str)
    return df


def balanced_training_frame(df: pd.DataFrame, min_per_class: int | None = None) -> pd.DataFrame:
    counts = df["sentiment"].value_counts()
    if min_per_class is None:
        min_per_class = int(counts.min())
    parts = []
    for label in LABEL_ORDER:
        label_df = df[df["sentiment"] == label]
        n = min(len(label_df), min_per_class)
        parts.append(label_df.sample(n=n, random_state=42))
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)


def split_data(df: pd.DataFrame):
    return train_test_split(
        df["clean_text"],
        df["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"],
    )


def evaluate_model(name: str, model: Pipeline, x_train, x_test, y_train, y_test) -> ExperimentResult:
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    return ExperimentResult(
        name=name,
        train_accuracy=accuracy_score(y_train, train_pred),
        test_accuracy=accuracy_score(y_test, test_pred),
        report=classification_report(y_test, test_pred, labels=LABEL_ORDER, zero_division=0),
        confusion=confusion_matrix(y_test, test_pred, labels=LABEL_ORDER),
        model=model,
    )


def classical_experiments(x_train, x_test, y_train, y_test) -> list[ExperimentResult]:
    logistic = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    max_features=60_000,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1_000,
                    class_weight="balanced",
                    C=4.0,
                    random_state=42,
                ),
            ),
        ]
    )
    svm = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    max_features=80_000,
                ),
            ),
            ("model", LinearSVC(class_weight="balanced", C=1.2, random_state=42)),
        ]
    )
    return [
        evaluate_model("Logistic Regression + TF-IDF", logistic, x_train, x_test, y_train, y_test),
        evaluate_model("Linear SVM + TF-IDF", svm, x_train, x_test, y_train, y_test),
    ]


def train_deep_learning_experiment(x_train, x_test, y_train, y_test) -> dict[str, object]:
    import tensorflow as tf
    from tensorflow.keras import layers

    label_to_id = {label: idx for idx, label in enumerate(LABEL_ORDER)}
    y_train_id = np.array([label_to_id[label] for label in y_train])
    y_test_id = np.array([label_to_id[label] for label in y_test])

    vectorizer = layers.TextVectorization(max_tokens=30_000, output_sequence_length=80)
    vectorizer.adapt(np.array(x_train))

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(1,), dtype=tf.string),
            vectorizer,
            layers.Embedding(30_000, 64, mask_zero=True),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dropout(0.35),
            layers.Dense(32, activation="relu"),
            layers.Dense(len(LABEL_ORDER), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=2,
        restore_best_weights=True,
    )
    history = model.fit(
        np.array(x_train),
        y_train_id,
        validation_split=0.2,
        epochs=8,
        batch_size=64,
        callbacks=[callback],
        verbose=1,
    )
    train_accuracy = float(model.evaluate(np.array(x_train), y_train_id, verbose=0)[1])
    test_accuracy = float(model.evaluate(np.array(x_test), y_test_id, verbose=0)[1])
    pred_id = model.predict(np.array(x_test), verbose=0).argmax(axis=1)
    pred = [LABEL_ORDER[idx] for idx in pred_id]
    return {
        "name": "BiLSTM + Keras TextVectorization",
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "report": classification_report(y_test, pred, labels=LABEL_ORDER, zero_division=0),
        "confusion": confusion_matrix(y_test, pred, labels=LABEL_ORDER),
        "model": model,
        "history": history.history,
    }


def save_best_classical(results: list[ExperimentResult], path: Path = MODEL_PATH) -> ExperimentResult:
    path.parent.mkdir(exist_ok=True)
    best = max(results, key=lambda item: item.test_accuracy)
    joblib.dump(best.model, path)
    return best


def predict_examples(model: object, examples: list[str]) -> pd.DataFrame:
    labels = model.predict(examples)
    return pd.DataFrame({"teks": examples, "prediksi_sentimen": labels})
