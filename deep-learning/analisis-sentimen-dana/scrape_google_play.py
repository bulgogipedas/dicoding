"""Scraping, pembersihan, dan pelabelan ulasan DANA dari Google Play.

File keluaran dibuat agar proses submission analisis sentimen mudah diaudit:

- data/dana_reviews_raw.csv berisi hasil scraping langsung.
- data/dana_reviews_labeled.csv berisi teks bersih dan label sentimen.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from google_play_scraper import Sort, reviews


APP_ID = "id.dana"
APP_NAME = "DANA Dompet Digital Indonesia"
RAW_PATH = Path("data/dana_reviews_raw.csv")
LABELED_PATH = Path("data/dana_reviews_labeled.csv")
POSITIVE_WORDS = {
    "aman",
    "bagus",
    "bagusss",
    "bagussss",
    "baik",
    "cepat",
    "excellent",
    "good",
    "hebat",
    "jos",
    "keren",
    "lancar",
    "love",
    "mantap",
    "membantu",
    "mudah",
    "nyaman",
    "ok",
    "oke",
    "okee",
    "puas",
    "rekomendasi",
    "sip",
    "suka",
    "sukses",
    "terbaik",
    "top",
}
NEGATIVE_WORDS = {
    "blokir",
    "bug",
    "buruk",
    "crash",
    "ditolak",
    "error",
    "force",
    "gagal",
    "gangguan",
    "hang",
    "hilang",
    "jelek",
    "kecewa",
    "kendala",
    "kepotong",
    "lambat",
    "lemot",
    "macet",
    "mahal",
    "masalah",
    "payah",
    "pending",
    "penipuan",
    "potong",
    "ribet",
    "saldo",
    "susah",
    "susahnya",
    "tipu",
}


def clean_text(value: object) -> str:
    """Menormalisasi teks ulasan tanpa mengubah kolom ulasan asli."""
    text = "" if pd.isna(value) else str(value)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^0-9a-zA-ZÀ-ÿ\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def label_from_score(score: int) -> str:
    if score <= 2:
        return "negatif"
    if score == 3:
        return "netral"
    return "positif"


def label_from_text(cleaned_text: str) -> str:
    words = cleaned_text.split()
    positive_hits = sum(word in POSITIVE_WORDS for word in words)
    negative_hits = sum(word in NEGATIVE_WORDS for word in words)
    if positive_hits > negative_hits:
        return "positif"
    if negative_hits > positive_hits:
        return "negatif"
    return "netral"


def scrape_reviews(target: int, batch_size: int, lang: str, country: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    token = None

    while len(rows) < target:
        result, token = reviews(
            APP_ID,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,
            count=min(batch_size, target - len(rows)),
            continuation_token=token,
        )
        if not result:
            break
        rows.extend(result)
        print(f"Berhasil scraping {len(rows):,}/{target:,} ulasan")
        if token is None:
            break

    if not rows:
        raise RuntimeError("Tidak ada ulasan yang berhasil diambil. Periksa koneksi internet dan app ID.")

    df = pd.DataFrame(rows)
    df["source_app_id"] = APP_ID
    df["source_app_name"] = APP_NAME
    return df


def prepare_labeled_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    required = ["reviewId", "userName", "content", "score", "at", "appVersion"]
    for column in required:
        if column not in raw_df.columns:
            raw_df[column] = None

    df = raw_df.copy()
    df["content"] = df["content"].fillna("").astype(str)
    df["clean_text"] = df["content"].map(clean_text)
    df = df[df["clean_text"].str.len() > 0]
    df = df.drop_duplicates(subset=["clean_text"])
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df[df["score"].between(1, 5)]
    df["score"] = df["score"].astype(int)
    df["rating_sentiment"] = df["score"].map(label_from_score)
    df["sentiment"] = df["clean_text"].map(label_from_text)

    columns = [
        "reviewId",
        "userName",
        "content",
        "clean_text",
        "score",
        "rating_sentiment",
        "sentiment",
        "at",
        "appVersion",
        "source_app_id",
        "source_app_name",
    ]
    return df[columns].sort_values("at", ascending=False, na_position="last")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--lang", default="id")
    parser.add_argument("--country", default="id")
    args = parser.parse_args()

    Path("data").mkdir(exist_ok=True)
    raw_df = scrape_reviews(args.target, args.batch_size, args.lang, args.country)
    labeled_df = prepare_labeled_dataset(raw_df)

    raw_df.to_csv(RAW_PATH, index=False)
    labeled_df.to_csv(LABELED_PATH, index=False)

    print(f"Dataset mentah disimpan: {RAW_PATH} ({len(raw_df):,} baris)")
    print(f"Dataset berlabel disimpan: {LABELED_PATH} ({len(labeled_df):,} baris)")
    print(labeled_df["sentiment"].value_counts().to_string())


if __name__ == "__main__":
    main()
