from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import pandas as pd
import psutil
import requests
from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel
from starlette.responses import Response


MODEL_URI = os.getenv("MODEL_URI")
SERVING_URL = os.getenv("SERVING_URL", "http://127.0.0.1:5001/invocations")
DEFAULT_MODEL_URI_FILE = Path("model_uri.txt")

REQUEST_TOTAL = Counter("credit_request_total", "Total prediction requests")
PREDICTION_TOTAL = Counter("credit_prediction_total", "Total predictions emitted", ["class_name"])
ERROR_TOTAL = Counter("credit_error_total", "Total failed prediction requests")
LATENCY = Histogram("credit_request_latency_seconds", "Prediction request latency")
PREDICTION_CONFIDENCE = Gauge("credit_prediction_confidence_avg", "Average confidence for the latest batch")
DEFAULT_RATIO = Gauge("credit_default_prediction_ratio", "Default prediction ratio for the latest batch")
BATCH_SIZE = Gauge("credit_batch_size", "Latest inference batch size")
CPU_USAGE = Gauge("credit_system_cpu_usage_percent", "System CPU usage percent")
RAM_USAGE = Gauge("credit_system_ram_usage_percent", "System RAM usage percent")
UPTIME = Gauge("credit_exporter_uptime_seconds", "Exporter uptime in seconds")
MODEL_READY = Gauge("credit_model_ready", "Model loaded successfully: 1=yes, 0=no")

STARTED_AT = time.time()
app = FastAPI(title="Credit Scoring Exporter", version="1.0.0")
model = None


class PredictRequest(BaseModel):
    records: list[dict[str, Any]]


def resolve_model_uri() -> str:
    if MODEL_URI:
        return MODEL_URI
    if DEFAULT_MODEL_URI_FILE.exists():
        return DEFAULT_MODEL_URI_FILE.read_text(encoding="utf-8").strip()
    raise RuntimeError("Atur MODEL_URI atau buat model_uri.txt yang berisi URI model MLflow.")


@app.on_event("startup")
def load_model() -> None:
    global model
    try:
        if MODEL_URI or DEFAULT_MODEL_URI_FILE.exists():
            model = mlflow.pyfunc.load_model(resolve_model_uri())
        MODEL_READY.set(1 if model is not None or SERVING_URL else 0)
    except Exception:
        MODEL_READY.set(0)
        model = None


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_ready": model is not None}


@app.post("/predict")
def predict(payload: PredictRequest) -> dict[str, Any]:
    if model is None and not SERVING_URL:
        ERROR_TOTAL.inc()
        raise HTTPException(status_code=503, detail="Model is not loaded. Check MODEL_URI or SERVING_URL.")

    started = time.perf_counter()
    REQUEST_TOTAL.inc()
    try:
        frame = pd.DataFrame(payload.records)
        probabilities = None

        if model is not None:
            predictions = model.predict(frame)
            try:
                probabilities = model._model_impl.sklearn_model.predict_proba(frame)[:, 1]
            except Exception:
                probabilities = None
        else:
            response = requests.post(
                SERVING_URL,
                json={"dataframe_records": payload.records},
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            predictions = result.get("predictions", result)

        predictions_list = [int(value) for value in predictions]
        default_count = sum(predictions_list)
        non_default_count = len(predictions_list) - default_count
        PREDICTION_TOTAL.labels(class_name="default").inc(default_count)
        PREDICTION_TOTAL.labels(class_name="non_default").inc(non_default_count)
        BATCH_SIZE.set(len(predictions_list))
        DEFAULT_RATIO.set(default_count / max(len(predictions_list), 1))

        if probabilities is not None:
            confidence = [max(float(p), 1.0 - float(p)) for p in probabilities]
            PREDICTION_CONFIDENCE.set(sum(confidence) / len(confidence))

        return {"predictions": predictions_list}
    except Exception as exc:
        ERROR_TOTAL.inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        LATENCY.observe(time.perf_counter() - started)


@app.get("/metrics")
def metrics() -> Response:
    CPU_USAGE.set(psutil.cpu_percent(interval=0.1))
    RAM_USAGE.set(psutil.virtual_memory().percent)
    UPTIME.set(time.time() - STARTED_AT)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
