from __future__ import annotations

import json
import os
import argparse
from pathlib import Path

import dagshub
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV


TARGET = "default_payment_next_month"
DATA_DIR = Path(__file__).resolve().parent / "credit_default_preprocessing"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


def configure_tracking() -> None:
    dagshub_owner = os.getenv("DAGSHUB_OWNER", "bulgogipedas")
    dagshub_repo = os.getenv("DAGSHUB_REPO", "smlsml-credit-scoring")

    try:
        dagshub.init(repo_owner=dagshub_owner, repo_name=dagshub_repo, mlflow=True)
    except Exception:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000/"))

    mlflow.set_experiment("Credit Scoring Rafli Ardiansyah Advanced")


def load_dataset(data_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return (
        train.drop(columns=[TARGET]),
        train[TARGET],
        test.drop(columns=[TARGET]),
        test[TARGET],
    )


def save_artifacts(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series, pred, proba) -> dict[str, float]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    (ARTIFACT_DIR / "metric_info.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm = confusion_matrix(y_test, pred)
    ConfusionMatrixDisplay(cm, display_labels=["No default", "Default"]).plot(cmap="Blues")
    plt.title("Training Confusion Matrix")
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "training_confusion_matrix.png", dpi=160)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "roc_curve.png", dpi=160)
    plt.close()

    importances = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).head(15)
    importances.to_csv(ARTIFACT_DIR / "feature_importance.csv", index=False)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=importances, y="feature", x="importance", color="#0f62fe")
    plt.title("Top Feature Importance")
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "feature_importance.png", dpi=160)
    plt.close()

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    args = parser.parse_args()

    configure_tracking()
    X_train, y_train, X_test, y_test = load_dataset(Path(args.data_dir))

    search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid={
            "n_estimators": [150, 250],
            "max_depth": [8, 12, None],
            "min_samples_split": [2, 8],
        },
        scoring="f1",
        cv=3,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="advanced_manual_tuning_random_forest"):
        search.fit(X_train, y_train)
        model = search.best_estimator_
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        metrics = save_artifacts(model, X_test, y_test, pred, proba)

        mlflow.log_params(search.best_params_)
        mlflow.log_metric("best_cv_f1", search.best_score_)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        mlflow.log_artifacts(str(ARTIFACT_DIR))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_test.head(5).astype(float),
            registered_model_name="credit_scoring_random_forest",
        )


if __name__ == "__main__":
    main()
