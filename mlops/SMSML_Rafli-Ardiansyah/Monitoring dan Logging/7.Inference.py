from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests


TARGET = "default_payment_next_month"
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "Membangun_model" / "credit_default_preprocessing" / "test.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--rows", type=int, default=5)
    args = parser.parse_args()

    data = pd.read_csv(args.data).drop(columns=[TARGET]).head(args.rows)
    response = requests.post(args.url, json={"records": data.to_dict(orient="records")}, timeout=30)
    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    main()
