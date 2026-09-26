"""Small, attributed sample of example reviews for the public repo / deployed app.

The full 120k-review corpus is NOT redistributed (review texts belong to their authors). The repo ships
~500 reviews from 12 products, plus aggregate statistics computed on the full corpus (no text).
Everything else can be rebuilt from the original source with `python -m reviewintel.ingest`.

Usage:  python -m reviewintel.sample     (needs data/processed/reviews_app_full.parquet)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C

PRODUCTS_PER_CATEGORY = 6
MIN_REVIEWS = 35

ATTRIBUTION = (
    "Example reviews are a small sample from the Amazon Reviews 2023 dataset (McAuley Lab, UCSD; "
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023). Hou, Y., Li, J., He, Z., Yan, A., "
    "Chen, X., McAuley, J. (2024). Bridging Language and Items for Retrieval and Recommendation. "
    "arXiv:2403.03952. Review text belongs to the original authors; shown for research / demonstration only."
)


def choose_products(df: pd.DataFrame, per_category: int = PRODUCTS_PER_CATEGORY, seed: int = C.SEED) -> list[str]:
    """Per category: products with enough reviews, spread over the rating range (low / middle / high)."""
    rng = np.random.default_rng(seed)
    picked: list[str] = []
    for _, g in df.groupby("category"):
        stats = g.groupby("parent_asin").agg(n=("rating", "size"), avg=("rating", "mean"))
        stats = stats[stats["n"] >= MIN_REVIEWS].sort_values("avg")
        bins = np.array_split(np.arange(len(stats)), per_category)  # one product from each rating quantile
        picked += [stats.index[int(rng.choice(b))] for b in bins if len(b)]
    return picked


def category_stats(df: pd.DataFrame) -> dict:
    out = {}
    for cat, g in df.groupby("category"):
        out[cat] = {
            "reviews": int(len(g)), "products": int(g["parent_asin"].nunique()),
            "negative_pct": float((g["rating"] <= C.NEG_MAX).mean() * 100),
            "positive_pct": float((g["rating"] >= C.POS_MIN).mean() * 100),
            "avg_rating": float(g["rating"].mean()),
            "date_min": str(g["date"].min())[:10], "date_max": str(g["date"].max())[:10],
        }
    out["_all"] = {"reviews": int(len(df)), "products": int(df["parent_asin"].nunique())}
    return out


def build(full: pd.DataFrame) -> pd.DataFrame:
    sample = full[full["parent_asin"].isin(choose_products(full))].reset_index(drop=True)
    sample.to_parquet(C.ARTIFACTS_DIR / "reviews_sample.parquet")
    (C.ARTIFACTS_DIR / "category_stats.json").write_text(json.dumps(category_stats(full), indent=1))
    (C.ARTIFACTS_DIR / "DATA_ATTRIBUTION.txt").write_text(ATTRIBUTION + "\n", encoding="utf-8")
    print(f"sample: {len(sample)} reviews from {sample['parent_asin'].nunique()} products")
    return sample


if __name__ == "__main__":
    build(pd.read_parquet(C.PROCESSED_DIR / "reviews_app_full.parquet"))
