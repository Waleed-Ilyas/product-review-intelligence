"""Sentiment experiments: TF-IDF+LR vs pretrained DistilBERT (zero-shot) vs the same model fine-tuned.

Usage:  python -m reviewintel.train_sentiment
Everything is evaluated on products the models have never seen (group split), and the two
transformer variants + the baseline are compared on the SAME fixed test subset.
"""
from __future__ import annotations

import json
import time
import warnings

import joblib
import mlflow
import pandas as pd

from . import config as C
from . import sentiment as S

TEST_SUBSET = 8000  # transformer inference is ~30 reviews/s on a laptop CPU
VALID_SUBSET = 3000  # for choosing the decision threshold of each model
N_FINETUNE = 6400  # balanced 3,200 + 3,200, one epoch
FT_MAX_LEN = 96


def main() -> None:
    warnings.filterwarnings("ignore")
    t0 = time.time()
    C.ARTIFACTS_DIR.mkdir(exist_ok=True)
    C.MODELS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{C.ROOT / 'mlflow.db'}")
    mlflow.set_experiment("review-sentiment")
    df = S.split_by_product(S.add_labels(pd.read_parquet(C.PROCESSED_DIR / "reviews.parquet")))
    df[["review_id", "split"]].to_parquet(C.PROCESSED_DIR / "splits.parquet")
    lab = df[df["label"] >= 0].reset_index(drop=True)
    tr, va, te = (lab[lab["split"] == s] for s in ("train", "valid", "test"))
    print(f"labelled {len(lab):,}: train {len(tr):,} valid {len(va):,} test {len(te):,} "
          f"| positive share {lab['label'].mean() * 100:.1f}%")

    with mlflow.start_run(run_name="sentiment"):
        # ---------------- 1. TF-IDF + logistic regression ----------------
        best_c, best_f1, grid = None, -1.0, []
        for c in (0.5, 1, 2, 4, 8):
            m = S.tfidf_lr(c).fit(tr["model_text"], tr["label"])
            f1 = S.metrics(va["label"], m.predict_proba(va["model_text"])[:, 1])["macro_f1"]
            grid.append({"C": c, "valid_macro_f1": f1})
            print(f"LR C={c}: valid macro-F1 {f1:.4f}")
            if f1 > best_f1:
                best_c, best_f1 = c, f1
        lr = S.tfidf_lr(best_c).fit(tr["model_text"], tr["label"])
        joblib.dump(lr, C.ARTIFACTS_DIR / "sentiment_lr.joblib", compress=3)
        p_te_lr = lr.predict_proba(te["model_text"])[:, 1]
        p_va_lr = lr.predict_proba(va["model_text"])[:, 1]
        thr_lr = S.best_threshold(va["label"], p_va_lr)
        results = {"tfidf_logreg": {"full_test": S.metrics(te["label"], p_te_lr),
                                    "full_test_tuned_threshold": S.metrics(te["label"], p_te_lr, thr_lr),
                                    "best_C": best_c, "C_grid": grid}}

        # temporal robustness: train on <= 2020, test on 2021+ (any product)
        early, late = lab[lab["date"] < "2021-01-01"], lab[lab["date"] >= "2021-01-01"]
        mt = S.tfidf_lr(best_c).fit(early["model_text"], early["label"])
        results["tfidf_logreg"]["temporal_2021plus"] = S.metrics(
            late["label"], mt.predict_proba(late["model_text"])[:, 1])

        # ---------------- 2. the shared fixed test / valid subsets ----------------
        te_s = te.sample(n=min(TEST_SUBSET, len(te)), random_state=C.SEED).reset_index(drop=True)
        va_s = va.sample(n=min(VALID_SUBSET, len(va)), random_state=C.SEED).reset_index(drop=True)
        results["tfidf_logreg"]["subset_test"] = S.metrics(te_s["label"], lr.predict_proba(te_s["model_text"])[:, 1])

        # ---------------- 3. pretrained transformer, zero-shot ----------------
        model, tok = S.load_pretrained()
        t1 = time.time()
        p_va_pre = S.transformer_predict(model, tok, va_s["model_text"].tolist(), log_every=40)
        p_te_pre = S.transformer_predict(model, tok, te_s["model_text"].tolist(), log_every=40)
        thr_pre = S.best_threshold(va_s["label"], p_va_pre)
        results["pretrained_distilbert_sst2"] = {
            "subset_test": S.metrics(te_s["label"], p_te_pre),
            "subset_test_tuned_threshold": S.metrics(te_s["label"], p_te_pre, thr_pre),
            "reviews_per_second": len(te_s) / (time.time() - t1) * 1.0}
        print(f"pretrained done ({time.time() - t0:.0f}s)")

        # ---------------- 4. fine-tune on a small balanced sample ----------------
        half = N_FINETUNE // 2
        ft = pd.concat([g.sample(n=min(half, len(g)), random_state=C.SEED) for _, g in tr.groupby("label")])
        ft = ft.sample(frac=1.0, random_state=C.SEED)
        model = S.finetune(model, tok, ft["model_text"].tolist(), ft["label"].to_numpy(), epochs=1,
                           batch_size=16, max_len=FT_MAX_LEN)
        p_va_ft = S.transformer_predict(model, tok, va_s["model_text"].tolist())
        p_te_ft = S.transformer_predict(model, tok, te_s["model_text"].tolist(), log_every=40)
        thr_ft = S.best_threshold(va_s["label"], p_va_ft)
        results["finetuned_distilbert"] = {
            "subset_test": S.metrics(te_s["label"], p_te_ft),
            "subset_test_tuned_threshold": S.metrics(te_s["label"], p_te_ft, thr_ft),
            "train_examples": int(len(ft)), "epochs": 1, "max_len": FT_MAX_LEN}
        model.save_pretrained(C.MODELS_DIR / "distilbert_finetuned")
        tok.save_pretrained(C.MODELS_DIR / "distilbert_finetuned")

        # ---------------- 5. predictions for error analysis / the app ----------------
        out = te_s[["review_id", "category", "parent_asin", "rating", "label", "title", "text", "date", "n_chars"]].copy()
        out["p_lr"] = lr.predict_proba(te_s["model_text"])[:, 1]
        out["p_pretrained"], out["p_finetuned"] = p_te_pre, p_te_ft
        out.to_parquet(C.PROCESSED_DIR / "sentiment_test_predictions.parquet")
        results["thresholds"] = {"tfidf_logreg": thr_lr, "pretrained": thr_pre, "finetuned": thr_ft}

        # breakdowns on the shared subset (fine-tuned vs LR)
        def by(col_fn):
            rows = {}
            for name, g in out.groupby(col_fn):
                rows[str(name)] = {"n": len(g), **{k: S.metrics(g["label"], g[c])["macro_f1"]
                                                   for k, c in (("lr", "p_lr"), ("pretrained", "p_pretrained"),
                                                                ("finetuned", "p_finetuned"))}}
            return rows
        results["by_category"] = by("category")
        results["by_length"] = by(pd.cut(out["n_chars"], [0, 80, 200, 500, 10**6],
                                         labels=["<80", "80-200", "200-500", ">500"]))
        results["by_rating"] = {str(int(r)): {k: float(((g[c] >= 0.5).astype(int) == g["label"]).mean())
                                              for k, c in (("lr", "p_lr"), ("pretrained", "p_pretrained"),
                                                           ("finetuned", "p_finetuned"))}
                                for r, g in out.groupby("rating")}
        results["data"] = {"labelled": len(lab), "train": len(tr), "valid": len(va), "test": len(te),
                           "subset_test": len(te_s), "positive_share_pct": float(lab["label"].mean() * 100),
                           "runtime_min": (time.time() - t0) / 60}
        (C.ARTIFACTS_DIR / "sentiment_metrics.json").write_text(json.dumps(results, indent=1))
        mlflow.log_metrics({
            "lr_macro_f1": results["tfidf_logreg"]["subset_test"]["macro_f1"],
            "pretrained_macro_f1": results["pretrained_distilbert_sst2"]["subset_test"]["macro_f1"],
            "finetuned_macro_f1": results["finetuned_distilbert"]["subset_test"]["macro_f1"]})
        print(f"done in {(time.time() - t0) / 60:.1f} min")
        for k in ("tfidf_logreg", "pretrained_distilbert_sst2", "finetuned_distilbert"):
            m = results[k]["subset_test"]
            print(f"{k:28s} acc {m['accuracy']:.4f} macroF1 {m['macro_f1']:.4f} AUC {m['roc_auc']:.4f} "
                  f"neg-F1 {m['negative']['f1']:.3f}")


def learning_curve() -> None:
    """Fair data-efficiency comparison: the LR baseline trained on the SAME 6,400 balanced reviews the
    transformer was fine-tuned on, and on larger samples. Merged into sentiment_metrics.json."""
    warnings.filterwarnings("ignore")
    df = S.split_by_product(S.add_labels(pd.read_parquet(C.PROCESSED_DIR / "reviews.parquet")))
    lab = df[df["label"] >= 0].reset_index(drop=True)
    tr, te = lab[lab["split"] == "train"], lab[lab["split"] == "test"]
    te_s = te.sample(n=min(TEST_SUBSET, len(te)), random_state=C.SEED).reset_index(drop=True)
    path = C.ARTIFACTS_DIR / "sentiment_metrics.json"
    res = json.loads(path.read_text())
    c = res["tfidf_logreg"]["best_C"]
    half = N_FINETUNE // 2
    same = pd.concat([g.sample(n=min(half, len(g)), random_state=C.SEED) for _, g in tr.groupby("label")])
    curve = {}
    for name, sub in (("same_6400_balanced_as_transformer", same),
                      ("random_20000", tr.sample(n=20000, random_state=C.SEED)),
                      ("all_train", tr)):
        m = S.tfidf_lr(c).fit(sub["model_text"], sub["label"])
        curve[name] = {"n_train": int(len(sub)),
                       **S.metrics(te_s["label"], m.predict_proba(te_s["model_text"])[:, 1])}
        print(name, len(sub), round(curve[name]["macro_f1"], 4), round(curve[name]["roc_auc"], 4))
    res["lr_learning_curve"] = curve
    path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    import sys

    learning_curve() if sys.argv[1:] == ["extra"] else main()
