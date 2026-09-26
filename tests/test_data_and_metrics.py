import numpy as np
import pandas as pd

from reviewintel import config as C
from reviewintel import ingest, themes
from reviewintel import sentiment as S


def _reviews(n_products=40, per=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_products):
        for k in range(per):
            rating = int(rng.choice([1, 2, 3, 4, 5], p=[0.12, 0.05, 0.07, 0.12, 0.64]))
            good = "works great love it" if rating >= 4 else "broke after a week terrible refund"
            rows.append(("Appliances" if p % 2 else "All_Beauty", f"P{p}", f"A{p}", float(rating),
                         f"title {k}", f"{good} number {k} of product {p}", pd.Timestamp("2021-01-01") +
                         pd.Timedelta(days=int(rng.integers(0, 900))), 0, True))
    return pd.DataFrame(rows, columns=["category", "parent_asin", "asin", "rating", "title", "text", "date",
                                       "helpful_vote", "verified_purchase"])


def test_clean_text_strips_html_urls_and_repairs_encoding():
    raw = "It<br />was great!!  Visit http://x.co/abc now. It�s fine� ok"
    out = ingest.clean_text(raw)
    assert "<br" not in out and "http" not in out and "�" not in out
    assert "It's fine" in out and "  " not in out


def test_corpus_waterfall_and_product_cap():
    raw = _reviews()
    raw = pd.concat([raw, raw.iloc[:5], raw.iloc[[0]].assign(text="hi")], ignore_index=True)  # dupes + short
    corpus, rep = ingest.build_corpus(raw)
    steps = {s["step"]: s["removed"] for s in rep["steps"]}
    assert steps["exact duplicate review text on the same product"] == 5
    assert any(k.startswith("review text shorter") and v == 1 for k, v in steps.items())
    assert corpus.groupby("parent_asin").size().max() <= C.MAX_PER_PRODUCT
    assert corpus["review_id"].is_unique and rep["final_reviews"] == len(corpus)


def test_labels_and_group_split_never_share_products():
    df = S.add_labels(ingest.build_corpus(_reviews())[0])
    assert set(df.loc[df.rating <= 2, "label"]) == {0} and set(df.loc[df.rating >= 4, "label"]) == {1}
    assert set(df.loc[df.rating == 3, "label"]) == {-1}
    df = S.split_by_product(df)
    prods = {s: set(df.loc[df.split == s, "parent_asin"]) for s in ("train", "valid", "test")}
    assert not (prods["train"] & prods["test"]) and not (prods["train"] & prods["valid"])
    assert not (prods["valid"] & prods["test"]) and all(len(v) > 0 for v in prods.values())


def test_tfidf_lr_learns_the_signal_and_metrics_are_sane():
    df = S.split_by_product(S.add_labels(ingest.build_corpus(_reviews())[0]))
    lab = df[df.label >= 0]
    tr, te = lab[lab.split == "train"], lab[lab.split == "test"]
    p = S.tfidf_lr(2.0).fit(tr.model_text, tr.label).predict_proba(te.model_text)[:, 1]
    m = S.metrics(te.label, p)
    assert m["roc_auc"] > 0.95 and m["macro_f1"] > 0.9
    assert m["confusion"]["tp"] + m["confusion"]["fn"] == int((te.label == 1).sum())


def test_metrics_perfect_and_threshold_search():
    y = np.array([0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.4, 0.45, 0.8, 0.9, 0.95])
    assert S.metrics(y, p)["negative"]["recall"] == 1.0
    thr = S.best_threshold(y, p)
    assert 0.4 < thr <= 0.45 and S.metrics(y, p, thr)["macro_f1"] == 1.0


def test_sentence_splitter_and_tables():
    text = "The battery died after two days. Shipping was fast! Not bad; would buy again. Ok."
    sents = themes.split_sentences(text)
    assert sents[0].startswith("The battery") and all(len(s.split()) >= 4 for s in sents)
    df = pd.DataFrame({"review_id": [0], "category": ["Appliances"], "parent_asin": ["P"],
                       "rating": [2.0], "date": [pd.Timestamp("2022-01-01")], "helpful_vote": [0], "text": [text]})
    tab = themes.sentence_table(df)
    assert len(tab) == len(sents) and (tab["review_id"] == 0).all()


def test_theme_stats_lift():
    reviews = pd.DataFrame({"review_id": range(10), "rating": [1.0, 1.0, 2.0, 5.0, 5.0, 5.0, 5.0, 4.0, 5.0, 3.0]})
    sents = pd.DataFrame({"review_id": [0, 1, 2, 3, 4, 5], "topic": [0, 0, 0, 1, 1, 1],
                          "p_pos": [0.1, 0.2, 0.3, 0.9, 0.8, 0.95]})
    t = themes.theme_stats(sents, reviews, {0: "broken", 1: "great"}).set_index("label")
    assert t.loc["broken", "neg_lift"] > 2 and t.loc["great", "neg_lift"] == 0
    assert t.loc["broken", "sentence_neg_share"] == 1.0 and t.loc["great", "pos_rate"] == 1.0
