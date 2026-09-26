"""Theme discovery for every review sentence + business statistics + light artifacts for the app.

Usage:  python -m reviewintel.run_themes     (after ingest and train_sentiment)

1. split reviews into sentences and embed them (MiniLM, CPU)
2. per category: BERTopic on a sample (random + extra sentences from <=3-star reviews so that complaint
   themes are well represented) -> ~30 themes; every other sentence is assigned to the nearest
   theme centroid in embedding space
3. per-theme statistics (complaint lift vs the category baseline), quarterly complaint trends,
   representative sentences, and a light TF-IDF theme classifier so the deployed app needs no torch
"""
from __future__ import annotations

import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from . import config as C
from . import themes as T

N_TOPICS = 30
SAMPLE_RANDOM, SAMPLE_COMPLAINT = 18_000, 12_000
MAX_SENTENCES_PER_REVIEW = 8


def embed_all(sents: pd.DataFrame) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(C.SENTENCE_ENCODER, device="cpu")
    model.max_seq_length = 64
    return model.encode(sents["sentence"].tolist(), batch_size=256, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True).astype(np.float16)


def main() -> None:
    warnings.filterwarnings("ignore")
    t0 = time.time()
    C.ARTIFACTS_DIR.mkdir(exist_ok=True)
    reviews = pd.read_parquet(C.PROCESSED_DIR / "reviews.parquet")
    # joblib (pickle) files here are produced by this repository's own training scripts and committed
    # under artifacts/; never load a joblib file from an untrusted source.
    lr = joblib.load(C.ARTIFACTS_DIR / "sentiment_lr.joblib")

    # ---------------- 1. sentences + embeddings ----------------
    sents = T.sentence_table(reviews)
    sents = sents[sents["sent_idx"] < MAX_SENTENCES_PER_REVIEW].reset_index(drop=True)
    print(f"{len(sents):,} sentences from {reviews['review_id'].nunique():,} reviews ({time.time() - t0:.0f}s)")
    emb_path = C.PROCESSED_DIR / "sentence_emb.npy"
    if emb_path.exists() and len(np.load(emb_path, mmap_mode="r")) == len(sents):
        emb = np.load(emb_path)
    else:
        emb = embed_all(sents)
        np.save(emb_path, emb)
    print(f"embedded ({time.time() - t0:.0f}s)")
    sents["p_pos"] = lr.predict_proba(sents["sentence"])[:, 1]  # sentence-level sentiment (light model)

    # ---------------- 2. themes per category ----------------
    rng = np.random.default_rng(C.SEED)
    all_topics = np.full(len(sents), -1)
    theme_rows, examples, topic_labels = [], [], {}
    for cat in C.CATEGORIES:
        idx = np.where(sents["category"].to_numpy() == cat)[0]
        complaint = idx[(sents["rating"].to_numpy()[idx] <= 3)]
        pick = np.unique(np.concatenate([rng.choice(idx, size=min(SAMPLE_RANDOM, len(idx)), replace=False),
                                         rng.choice(complaint, size=min(SAMPLE_COMPLAINT, len(complaint)),
                                                    replace=False)]))
        model, topics = T.fit_topics(sents["sentence"].iloc[pick].tolist(), emb[pick].astype(np.float32),
                                     n_topics=N_TOPICS)
        ids = sorted(set(topics.tolist()) - {-1})
        cent = np.stack([emb[pick][topics == t].astype(np.float32).mean(0) for t in ids])
        cent /= np.linalg.norm(cent, axis=1, keepdims=True)
        sim = emb[idx].astype(np.float32) @ cent.T
        assign = np.array(ids)[sim.argmax(1)]
        all_topics[idx] = assign
        labels = {t: T.topic_label(model, t) for t in ids}
        topic_labels[cat] = labels
        print(f"{cat}: {len(ids)} themes ({time.time() - t0:.0f}s)")

        sub = sents.iloc[idx].assign(topic=assign)
        stats = T.theme_stats(sub, reviews[reviews["category"] == cat], labels)
        stats.insert(0, "category", cat)
        theme_rows.append(stats)
        # representative sentences: nearest to the theme centroid, split by the review's rating
        best = sim.max(1)
        for t in ids:
            m = np.where(assign == t)[0]
            for kind, mask in (("typical", np.ones(len(m), bool)),
                               ("complaint", sub["rating"].to_numpy()[m] <= 2),
                               ("praise", sub["rating"].to_numpy()[m] >= 4)):
                mm = m[mask]
                for j in mm[np.argsort(-best[mm])[:4]]:
                    r = sub.iloc[j]
                    examples.append({"category": cat, "topic": int(t), "kind": kind, "review_id": int(r["review_id"]),
                                     "sentence": r["sentence"], "rating": float(r["rating"])})

    sents["topic"] = all_topics
    theme_df = pd.concat(theme_rows, ignore_index=True)
    theme_df.to_parquet(C.ARTIFACTS_DIR / "themes.parquet")
    pd.DataFrame(examples).to_parquet(C.ARTIFACTS_DIR / "theme_examples.parquet")

    # ---------------- 3. quarterly complaint trends ----------------
    rv = reviews.assign(quarter=reviews["date"].dt.to_period("Q").astype(str))
    n_q = rv.groupby(["category", "quarter"]).size().rename("reviews_in_quarter")
    m = sents[["review_id", "topic"]].drop_duplicates().merge(
        rv[["review_id", "category", "quarter", "rating"]], on="review_id")
    grp = m.groupby(["category", "topic", "quarter"])
    trend = pd.DataFrame({"mentions": grp.size(), "neg_mentions": grp["rating"].apply(lambda r: int((r <= 2).sum()))})
    trend = trend.reset_index().merge(n_q.reset_index(), on=["category", "quarter"])
    trend["complaint_share_pct"] = trend["neg_mentions"] / trend["reviews_in_quarter"] * 100
    trend.to_parquet(C.ARTIFACTS_DIR / "theme_trend.parquet")

    # ---------------- 4. review-level tables for the app ----------------
    m[["review_id", "topic"]].astype({"review_id": "int32", "topic": "int16"}).to_parquet(
        C.ARTIFACTS_DIR / "review_themes.parquet")
    lab_p = lr.predict_proba(reviews["title"].where(reviews["title"].str.len() == 0, reviews["title"] + ". ") +
                             reviews["text"])[:, 1]
    app = reviews[["review_id", "category", "parent_asin", "product_title", "rating", "date", "title", "text",
                   "helpful_vote"]].assign(p_pos_lr=lab_p.astype("float32"))
    app["date"] = app["date"].dt.strftime("%Y-%m-%d")
    app.to_parquet(C.PROCESSED_DIR / "reviews_app_full.parquet")  # NOT committed: see sample.py
    from . import sample
    sample.build(app)

    # ---------------- 5. light theme classifier (no torch needed in the app) ----------------
    clf_report = {}
    for cat in C.CATEGORIES:
        sub = sents[sents["category"] == cat]
        tr, te = train_test_split(sub, test_size=0.15, random_state=C.SEED, stratify=sub["topic"])
        clf = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=80_000,
                                                  sublinear_tf=True)),
                        ("lr", LogisticRegression(C=3.0, max_iter=300))]).fit(tr["sentence"], tr["topic"])
        acc = float((clf.predict(te["sentence"]) == te["topic"]).mean())
        clf_report[cat] = {"held_out_accuracy_vs_embedding_assignment": acc, "n_themes": int(sub["topic"].nunique())}
        joblib.dump(clf, C.ARTIFACTS_DIR / f"theme_clf_{cat}.joblib", compress=3)
        print(f"{cat}: light theme classifier agrees with embedding assignment on {acc * 100:.1f}% of held-out sentences")

    (C.ARTIFACTS_DIR / "theme_labels.json").write_text(json.dumps(
        {c: {str(k): v for k, v in d.items()} for c, d in topic_labels.items()}, indent=1))
    (C.ARTIFACTS_DIR / "theme_meta.json").write_text(json.dumps({
        "n_sentences": len(sents), "n_reviews": int(reviews["review_id"].nunique()), "classifier": clf_report,
        "runtime_min": (time.time() - t0) / 60, "n_topics_requested": N_TOPICS}, indent=1))
    sents[["review_id", "sent_idx", "topic", "p_pos"]].to_parquet(C.PROCESSED_DIR / "sentence_topics.parquet")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
