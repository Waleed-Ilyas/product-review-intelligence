"""Sentiment models: TF-IDF + logistic regression baseline, pretrained transformer, fine-tuned transformer.

Task: binary sentiment from the star rating (1-2 = negative, 4-5 = positive; 3 stars excluded).
Splits are by PRODUCT so the test set measures generalisation to products the model has never seen.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from . import config as C


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["label"] = np.where(out["rating"] >= C.POS_MIN, 1, np.where(out["rating"] <= C.NEG_MAX, 0, -1))
    out["model_text"] = np.where(out["title"].str.len() > 0, out["title"] + ". " + out["text"], out["text"])
    return out


def split_by_product(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `split` in {train, valid, test} (70/10/20), grouped by product within each category."""
    df = df.copy()
    df["split"] = "train"
    for _cat, g in df.groupby("category"):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=C.SEED)
        rest, test = next(gss.split(g, groups=g["parent_asin"]))
        df.loc[g.index[test], "split"] = "test"
        g2 = g.iloc[rest]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=C.SEED)
        _, valid = next(gss2.split(g2, groups=g2["parent_asin"]))
        df.loc[g2.index[valid], "split"] = "valid"
    return df


def tfidf_lr(c: float = 4.0) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=120_000, sublinear_tf=True,
                                  strip_accents="unicode")),
        ("lr", LogisticRegression(C=c, max_iter=2000, class_weight="balanced")),
    ])


def metrics(y, p, threshold: float = 0.5) -> dict:
    """y in {0 (negative), 1 (positive)}, p = P(positive)."""
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")), "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "negative": {"precision": float(prec[0]), "recall": float(rec[0]), "f1": float(f1[0])},
        "positive": {"precision": float(prec[1]), "recall": float(rec[1]), "f1": float(f1[1])},
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": float(threshold),
    }


def best_threshold(y, p) -> float:
    """Threshold maximising macro-F1 (chosen on the validation split only)."""
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y, (p >= t).astype(int), average="macro") for t in grid]
    return float(grid[int(np.argmax(scores))])


# ------------------------------------------------------------------ transformers
def _device():
    import torch
    return torch.device("cpu")


def transformer_predict(model, tokenizer, texts: list[str], batch_size: int = 32,
                        max_len: int = C.MAX_LEN, log_every: int = 0) -> np.ndarray:
    """P(positive) for each text (softmax over the two classes). Sorted by length for speed."""
    import torch
    order = np.argsort([len(t) for t in texts])
    out = np.zeros(len(texts), dtype=np.float32)
    model.eval()
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            idx = order[i:i + batch_size]
            enc = tokenizer([texts[j] for j in idx], truncation=True, max_length=max_len, padding=True,
                            return_tensors="pt")
            logits = model(**enc).logits
            out[idx] = torch.softmax(logits, dim=-1)[:, 1].numpy()
            if log_every and (i // batch_size) % log_every == 0:
                print(f"   {i + len(idx):,}/{len(texts):,} ({time.time() - t0:.0f}s)")
    return out


def load_pretrained(name: str = C.PRETRAINED_SENTIMENT):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    return model, tok


def finetune(model, tok, texts: list[str], labels: np.ndarray, epochs: int = 1, batch_size: int = 16,
             lr: float = 2e-5, max_len: int = C.MAX_LEN, seed: int = C.SEED, log_every: int = 50):
    """Plain PyTorch fine-tuning loop (CPU). Labels: 1 = positive (SST-2 id 1 = POSITIVE)."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n = len(texts)
    steps = epochs * ((n + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.06 * steps)) *
                                              max(0.0, (steps - s) / steps))
    model.train()
    step, t0, losses = 0, time.time(), []
    for _ in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=max_len, padding=True,
                      return_tensors="pt")
            out = model(**enc, labels=torch.tensor(labels[idx], dtype=torch.long))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(float(out.loss))
            step += 1
            if log_every and step % log_every == 0:
                print(f"   step {step}/{steps} loss {np.mean(losses[-log_every:]):.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    return model
