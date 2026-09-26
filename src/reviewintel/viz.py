"""Report figures (matplotlib, dark/gold theme matching the portfolio site)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BG, PANEL, IVORY, GOLD, BRONZE, CYAN, RED = (
    "#0A0A0C", "#15110F", "#F2EDE4", "#D9B26A", "#8B5E3C", "#4FD1C5", "#ff5a4f",
)


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
        "axes.edgecolor": "#3a332d", "axes.labelcolor": IVORY, "text.color": IVORY,
        "xtick.color": IVORY, "ytick.color": IVORY, "grid.color": "#2a2521", "grid.linewidth": 0.8,
        "axes.grid": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "legend.frameon": False,
        "figure.dpi": 110, "font.family": "DejaVu Sans",
    })


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cleaning_waterfall(report: dict, path) -> None:
    apply_style()
    steps = report["steps"]
    labels = ["Raw reviews (2 categories)"] + [s["step"].split(" (")[0][:58] for s in steps]
    vals = [report["raw_reviews"]] + [s["removed"] for s in steps]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    bars = ax.barh(labels[::-1], vals[::-1], color=[BRONZE] * len(steps) + [CYAN])
    ax.bar_label(bars, labels=[f"{v:,}" for v in vals[::-1]], color=IVORY, padding=4, fontsize=9)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_xlabel("reviews")
    ax.set_title(f"Corpus build: {report['raw_reviews']:,} raw -> {report['final_reviews']:,} analysed reviews")
    _save(fig, path)


def plot_eda(reviews: pd.DataFrame, path) -> None:
    apply_style()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    r = reviews["rating"].value_counts(normalize=True).sort_index() * 100
    ax[0].bar(r.index.astype(int), r.values, color=[RED, RED, BRONZE, GOLD, GOLD])
    ax[0].set_title("Star ratings are J-shaped")
    ax[0].set_xlabel("stars")
    ax[0].set_ylabel("% of reviews")
    for x, v in zip(r.index.astype(int), r.values, strict=True):
        ax[0].text(x, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    ax[1].hist(np.log10(reviews["n_chars"]), bins=40, color=GOLD)
    ax[1].set_xticks([1.5, 2, 2.5, 3, 3.5], ["32", "100", "316", "1k", "3k"])
    ax[1].set_title("Review length (characters, log scale)")
    yr = reviews.groupby(reviews["date"].dt.year).size()
    yr = yr[yr.index >= 2012]
    ax[2].bar(yr.index, yr.values, color=CYAN)
    ax[2].set_title("Reviews per year (analysed corpus)")
    _save(fig, path)


def plot_sentiment_models(res: dict, path) -> None:
    apply_style()
    keys = {"tfidf_logreg": ("TF-IDF + logistic regression", BRONZE),
            "pretrained_distilbert_sst2": ("DistilBERT SST-2 (zero-shot)", CYAN),
            "finetuned_distilbert": ("DistilBERT fine-tuned (6.4k reviews)", GOLD)}
    metrics = [("macro_f1", "Macro-F1"), ("roc_auc", "ROC-AUC"), ("neg_f1", "Negative-class F1")]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    w = 0.26
    for i, (k, (label, color)) in enumerate(keys.items()):
        m = res[k]["subset_test"]
        vals = [m["macro_f1"], m["roc_auc"], m["negative"]["f1"]]
        bars = ax.bar(np.arange(3) + (i - 1) * w, vals, w, color=color, label=label)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, color=IVORY, padding=2)
    ax.set_xticks(range(3), [m[1] for m in metrics])
    ax.set_ylim(0.6, 1.02)
    ax.set_title("Sentiment on unseen products (8,000-review test subset)")
    ax.legend(loc="lower center", ncol=1, fontsize=9)
    _save(fig, path)


def plot_confusions(res: dict, path) -> None:
    apply_style()
    keys = [("tfidf_logreg", "LR"), ("pretrained_distilbert_sst2", "DistilBERT zero-shot"),
            ("finetuned_distilbert", "DistilBERT fine-tuned")]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for axis, (k, title) in zip(ax, keys, strict=True):
        c = res[k]["subset_test"]["confusion"]
        cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]], float)
        norm = cm / cm.sum(1, keepdims=True)
        axis.imshow(norm, cmap="YlOrBr", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                axis.text(j, i, f"{norm[i, j] * 100:.1f}%\n({int(cm[i, j]):,})", ha="center", va="center",
                          color="#0A0A0C" if norm[i, j] > 0.5 else IVORY, fontsize=10)
        axis.set_xticks([0, 1], ["pred neg", "pred pos"])
        axis.set_yticks([0, 1], ["actual neg", "actual pos"])
        axis.set_title(title)
        axis.grid(False)
    _save(fig, path)


def plot_theme_map(themes: pd.DataFrame, names: dict, path) -> None:
    apply_style()
    cats = list(themes["category"].unique())
    fig, ax = plt.subplots(1, len(cats), figsize=(7.5 * len(cats), 6.2))
    ax = np.atleast_1d(ax)
    for axis, cat in zip(ax, cats, strict=True):
        t = themes[themes["category"] == cat]
        axis.scatter(t["review_share_pct"], t["neg_lift"], s=np.sqrt(t["sentences"]) * 4,
                     c=np.where(t["neg_lift"] > 1.4, RED, np.where(t["neg_lift"] < 0.7, CYAN, GOLD)),
                     alpha=0.75, edgecolor="none")
        axis.axhline(1, color=IVORY, ls=":", lw=1)
        axis.set_xscale("log")
        for r in t.itertuples():
            if r.neg_lift > 1.8 or r.neg_lift < 0.45 or r.review_share_pct > 12:
                axis.annotate(names.get((cat, r.topic), r.label)[:34], (r.review_share_pct, r.neg_lift),
                              xytext=(5, 4), textcoords="offset points", fontsize=8, color=IVORY)
        axis.set_xlabel("% of reviews mentioning the theme (log)")
        axis.set_ylabel("complaint lift (1.0 = average review)")
        axis.set_title(f"{cat.replace('_', ' ')}: what customers talk about")
    _save(fig, path)


def plot_trends(trend: pd.DataFrame, themes: pd.DataFrame, names: dict, path, top_n: int = 4) -> None:
    apply_style()
    cats = list(themes["category"].unique())
    fig, ax = plt.subplots(1, len(cats), figsize=(7.5 * len(cats), 4.8))
    ax = np.atleast_1d(ax)
    palette = [RED, GOLD, CYAN, "#c9b8e8", "#ff9f5a"]
    for axis, cat in zip(ax, cats, strict=True):
        t = themes[(themes["category"] == cat) & (themes["review_share_pct"] > 1.5)]
        t = t[[("generic" not in names.get((cat, x), "")) for x in t["topic"]]]  # skip catch-all themes
        top = t.assign(volume=t["reviews"] * t["neg_rate"]).nlargest(top_n, "volume")
        for color, r in zip(palette, top.itertuples(), strict=False):
            d = trend[(trend["category"] == cat) & (trend["topic"] == r.topic) & (trend["quarter"] >= "2016Q1")
                      & (trend["reviews_in_quarter"] >= 150)].sort_values("quarter")
            axis.plot(d["quarter"], d["complaint_share_pct"], color=color, lw=2,
                      label=names.get((cat, r.topic), r.label)[:34])
        axis.set_xticks(axis.get_xticks()[::4])
        axis.tick_params(axis="x", rotation=45)
        axis.set_ylabel("% of the quarter's reviews that are 1-2 stars AND mention it")
        axis.set_title(f"{cat.replace('_', ' ')}: biggest complaint themes over time")
        axis.legend(fontsize=8)
    _save(fig, path)
