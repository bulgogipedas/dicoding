from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


TARGET = "default_payment_next_month"
DATA_DIR = Path(__file__).resolve().parent / "credit_default_preprocessing"


def load_dataset(data_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]
    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET]
    return X_train, y_train, X_test, y_test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000/"))
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("Credit Scoring Rafli Ardiansyah")
    mlflow.sklearn.autolog()

    X_train, y_train, X_test, y_test = load_dataset(Path(args.data_dir))
    model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)

    with mlflow.start_run(run_name="baseline_autolog_random_forest"):
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        mlflow.log_metric("test_accuracy", accuracy_score(y_test, pred))
        mlflow.log_metric("test_precision", precision_score(y_test, pred, zero_division=0))
        mlflow.log_metric("test_recall", recall_score(y_test, pred, zero_division=0))
        mlflow.log_metric("test_f1", f1_score(y_test, pred, zero_division=0))
        mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, proba))


if __name__ == "__main__":
    main()
