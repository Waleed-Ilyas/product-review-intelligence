"""Generate report figures.  Usage: python -m reviewintel.report"""
from __future__ import annotations

import json

import pandas as pd

from . import config as C
from . import viz


def theme_names() -> dict:
    raw = json.loads((C.ARTIFACTS_DIR / "theme_names.json").read_text())
    return {(k.split("|")[0], int(k.split("|")[1])): v["name"] for k, v in raw.items()}


def main() -> None:
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    F, A = C.FIGURES_DIR, C.ARTIFACTS_DIR
    names = theme_names()
    themes = pd.read_parquet(A / "themes.parquet")
    trend = pd.read_parquet(A / "theme_trend.parquet")
    reviews = pd.read_parquet(C.PROCESSED_DIR / "reviews.parquet")
    sent = json.loads((A / "sentiment_metrics.json").read_text())
    viz.plot_cleaning_waterfall(json.loads((A / "cleaning_report.json").read_text()), F / "01_corpus_waterfall.png")
    viz.plot_eda(reviews, F / "02_eda.png")
    viz.plot_sentiment_models(sent, F / "03_sentiment_models.png")
    viz.plot_confusions(sent, F / "04_confusion.png")
    viz.plot_theme_map(themes, names, F / "05_theme_map.png")
    viz.plot_trends(trend, themes, names, F / "06_complaint_trends.png")
    print("figures written to", F)


if __name__ == "__main__":
    main()
