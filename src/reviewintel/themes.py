"""Aspect / theme discovery: sentence embeddings + BERTopic, then per-theme business statistics."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config as C

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(]?[A-Z0-9])|\s{2,}|\s*[\n;]\s*")
STOP_EXTRA = {"amazon", "product", "item", "one", "just", "really", "would", "also", "get", "got",
              "like", "use", "used", "using", "will", "even", "much", "well", "im", "ive", "dont",
              "didnt", "doesnt", "great", "good", "bad", "love", "loved", "nice", "best", "star", "stars",
              "buy", "bought", "purchase", "purchased", "ordered", "order", "review", "recommend"}


def split_sentences(text: str, min_words: int = 4, max_words: int = 60) -> list[str]:
    out = []
    for s in _SPLIT.split(text):
        s = s.strip()
        n = len(s.split())
        if min_words <= n <= max_words:
            out.append(s)
    return out


def sentence_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in df.itertuples():
        for k, s in enumerate(split_sentences(r.text)):
            rows.append((r.review_id, k, s))
    out = pd.DataFrame(rows, columns=["review_id", "sent_idx", "sentence"])
    meta = df[["review_id", "category", "parent_asin", "rating", "date", "helpful_vote"]]
    return out.merge(meta, on="review_id", how="left")


def embed(sentences: list[str], batch_size: int = 128, show: bool = True) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(C.SENTENCE_ENCODER, device="cpu")
    return model.encode(sentences, batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=show, convert_to_numpy=True).astype(np.float32)


def fit_topics(sentences: list[str], embeddings: np.ndarray, n_topics: int = 30,
               min_topic_size: int = 60, seed: int = C.SEED):
    """BERTopic on precomputed embeddings: UMAP -> HDBSCAN -> c-TF-IDF, merged to ~n_topics, then
    outliers are reassigned to their nearest topic so every sentence has a theme."""
    from bertopic import BERTopic
    from sklearn.cluster import HDBSCAN
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
    from umap import UMAP

    stops = list(ENGLISH_STOP_WORDS | STOP_EXTRA)
    model = BERTopic(
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed),
        hdbscan_model=HDBSCAN(min_cluster_size=min_topic_size, min_samples=10, metric="euclidean"),
        vectorizer_model=CountVectorizer(stop_words=stops, ngram_range=(1, 2), min_df=5),
        nr_topics=n_topics, calculate_probabilities=False, verbose=False)
    topics, _ = model.fit_transform(sentences, embeddings)
    topics = model.reduce_outliers(sentences, topics, strategy="embeddings", embeddings=embeddings)
    return model, np.asarray(topics)


def topic_label(model, tid: int, n: int = 4) -> str:
    return ", ".join(w for w, _ in model.get_topic(tid)[:n])


def theme_stats(sents: pd.DataFrame, reviews: pd.DataFrame, labels: dict[int, str]) -> pd.DataFrame:
    """Per theme: how many reviews mention it, and how it associates with complaints."""
    base_neg = float((reviews["rating"] <= C.NEG_MAX).mean())
    base_pos = float((reviews["rating"] >= C.POS_MIN).mean())
    n_reviews = reviews["review_id"].nunique()
    rows = []
    for tid, g in sents.groupby("topic"):
        rv = reviews[reviews["review_id"].isin(g["review_id"].unique())]
        neg = float((rv["rating"] <= C.NEG_MAX).mean())
        pos = float((rv["rating"] >= C.POS_MIN).mean())
        rows.append({
            "topic": int(tid), "label": labels.get(int(tid), str(tid)), "sentences": len(g),
            "reviews": len(rv), "review_share_pct": len(rv) / n_reviews * 100,
            "avg_rating": float(rv["rating"].mean()), "neg_rate": neg, "pos_rate": pos,
            "neg_lift": neg / base_neg if base_neg else np.nan,
            "pos_lift": pos / base_pos if base_pos else np.nan,
            "sentence_neg_share": float((g["p_pos"] < 0.5).mean()) if "p_pos" in g else np.nan,
        })
    out = pd.DataFrame(rows).sort_values("reviews", ascending=False).reset_index(drop=True)
    out.attrs["base_neg"], out.attrs["base_pos"] = base_neg, base_pos
    return out
