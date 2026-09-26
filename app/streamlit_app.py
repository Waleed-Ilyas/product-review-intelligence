"""Product Review Intelligence - Streamlit app (sentiment, themes, complaint trends)."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ART = ROOT / "artifacts"
MODELS = ROOT / "models"

from reviewintel import themes as T  # noqa: E402

GOLD, IVORY, PANEL, CYAN, BRONZE, RED = "#D9B26A", "#F2EDE4", "#15110F", "#4FD1C5", "#8B5E3C", "#ff5a4f"
CAT_LABEL = {"Appliances": "Home appliances", "All_Beauty": "Beauty"}

st.set_page_config(page_title="Review Intelligence", page_icon="💬", layout="wide")
st.markdown(
    f"""
    <style>
    .block-container {{padding-top: 2rem; max-width: 1250px;}}
    h1, h2, h3 {{font-family: Georgia, 'Times New Roman', serif; letter-spacing: -0.01em;}}
    h1 {{font-weight: 400; font-size: 2.6rem;}}
    .kpi {{background: {PANEL}; border: 1px solid rgba(242,237,228,.08); border-radius: 12px;
          padding: 1rem 1.2rem; height: 100%;}}
    .kpi .label {{font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; opacity: .6;}}
    .kpi .value {{font-size: 1.9rem; font-family: Georgia, serif; line-height: 1.25;}}
    .kpi .sub {{font-size: .85rem; opacity: .75;}}
    .quote {{border-left: 3px solid {GOLD}; padding: .25rem .9rem; margin: .35rem 0; opacity: .92;}}
    .quote.bad {{border-color: {RED};}}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load():
    names_path = ART / "theme_names.json"
    names = json.loads(names_path.read_text()) if names_path.exists() else {}
    themes = pd.read_parquet(ART / "themes.parquet")
    themes["name"] = [names.get(f"{r.category}|{r.topic}", {}).get("name", r.label) for r in themes.itertuples()]
    themes["kind"] = [names.get(f"{r.category}|{r.topic}", {}).get("kind", "") for r in themes.itertuples()]
    # Only a small attributed sample of reviews is shipped (see artifacts/DATA_ATTRIBUTION.txt); category
    # statistics were computed on the full corpus and stored without any review text.
    full = ROOT / "data" / "processed" / "reviews_app_full.parquet"
    reviews = pd.read_parquet(full if (os.environ.get("REVIEWINTEL_FULL") and full.exists()) else ART / "reviews_sample.parquet")
    return {
        "themes": themes, "reviews": reviews, "trend": pd.read_parquet(ART / "theme_trend.parquet"),
        "examples": pd.read_parquet(ART / "theme_examples.parquet"),
        "review_themes": pd.read_parquet(ART / "review_themes.parquet"),
        "stats": json.loads((ART / "category_stats.json").read_text()),
        "attribution": (ART / "DATA_ATTRIBUTION.txt").read_text(encoding="utf-8"),
        "sent": json.loads((ART / "sentiment_metrics.json").read_text()),
        "clean": json.loads((ART / "cleaning_report.json").read_text()),
        "meta": json.loads((ART / "theme_meta.json").read_text()),
    }


@st.cache_resource
def load_models():
    # joblib files are produced by this repository's own training scripts (trusted artifacts).
    lr = joblib.load(ART / "sentiment_lr.joblib")
    clf = {c: joblib.load(ART / f"theme_clf_{c}.joblib") for c in CAT_LABEL}
    return lr, clf


@st.cache_resource
def load_transformer():
    """Optional: the fine-tuned DistilBERT, loaded from the Hugging Face Hub (or ./models locally).

    Needs torch + transformers (see requirements-hf.txt). Returns None in the light deployment."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from reviewintel import hub
        cfg = hub.model_settings()
        source = cfg["repo_id"] or (str(MODELS / "distilbert_finetuned") if (MODELS / "distilbert_finetuned").exists() else None)
        if source is None:
            return None
        tok = AutoTokenizer.from_pretrained(source)
        model = AutoModelForSequenceClassification.from_pretrained(source).eval()
        return model, tok, torch, cfg["threshold"], source
    except Exception:  # noqa: BLE001 - torch / transformers / network not available: fall back to LR only
        return None


D = load()
LR, CLF = load_models()
THEMES, REV = D["themes"], D["reviews"]


def kpi(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
            f'<div class="sub">{sub}</div></div>')


def style(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=IVORY), margin=dict(l=10, r=10, t=40, b=10))
    fig.update_xaxes(gridcolor="rgba(242,237,228,.06)")
    fig.update_yaxes(gridcolor="rgba(242,237,228,.06)")
    return fig


def esc(s: str) -> str:
    return s.replace("<", "&lt;").replace(">", "&gt;")


st.title("Product Review Intelligence")
st.caption("What do customers love and hate - without reading thousands of reviews · Amazon Reviews 2023 "
           f"({D['stats']['_all']['reviews']:,} reviews analysed, {D['stats']['_all']['products']:,} products, 2 categories)")

tab_over, tab_prod, tab_trend, tab_try, tab_model = st.tabs(
    ["Category themes", "Product analyzer", "Complaint trends", "Analyze a review", "Model accuracy"])

# ------------------------------------------------------------------ category themes
with tab_over:
    cat = st.radio("Category", list(CAT_LABEL), format_func=CAT_LABEL.get, horizontal=True, key="cat_over")
    t = THEMES[THEMES.category == cat]
    cs = D["stats"][cat]
    c = st.columns(4)
    c[0].markdown(kpi("Reviews analysed", f"{cs['reviews']:,}", f"{cs['products']:,} products"), unsafe_allow_html=True)
    c[1].markdown(kpi("Negative reviews (1-2★)", f"{cs['negative_pct']:.1f}%",
                      f"positive (4-5★) {cs['positive_pct']:.1f}%"), unsafe_allow_html=True)
    c[2].markdown(kpi("Themes discovered", f"{len(t)}", "BERTopic on sentence embeddings"), unsafe_allow_html=True)
    worst = t[t.review_share_pct > 1.5].sort_values("neg_lift", ascending=False).iloc[0]
    c[3].markdown(kpi("Most complaint-heavy theme", worst["name"][:26], f"{worst.neg_lift:.1f}× the average complaint rate"),
                  unsafe_allow_html=True)
    st.markdown("")
    fig = px.scatter(t, x="review_share_pct", y="neg_lift", size="sentences", color="neg_lift", log_x=True,
                     hover_name="name", hover_data={"review_share_pct": ":.1f", "neg_lift": ":.2f", "avg_rating": ":.2f",
                                                    "sentences": True},
                     color_continuous_scale=[[0, CYAN], [0.35, GOLD], [1, RED]],
                     labels={"review_share_pct": "% of reviews mentioning the theme (log)", "neg_lift": "complaint lift"})
    fig.add_hline(y=1, line_dash="dot", line_color=IVORY, opacity=0.5)
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(tickvals=[1, 2, 5, 10, 20], ticktext=["1%", "2%", "5%", "10%", "20%"])
    st.plotly_chart(style(fig, 470), width="stretch")
    st.caption("Complaint lift = share of 1-2★ reviews among reviews mentioning the theme ÷ share among all reviews. "
               "Above 1 = the theme is disproportionately raised by unhappy customers.")
    left_col, right_col = st.columns(2)
    cols = ["name", "review_share_pct", "avg_rating", "neg_lift"]
    cfg = {"name": "theme", "review_share_pct": st.column_config.NumberColumn("% reviews", format="%.1f"),
           "avg_rating": st.column_config.NumberColumn("avg ★", format="%.2f"),
           "neg_lift": st.column_config.NumberColumn("complaint lift", format="%.2f")}
    big = t[(t.review_share_pct > 1.5) & ~t.name.str.contains("generic", case=False)]
    left_col.markdown("#### Top complaint themes")
    left_col.dataframe(big.sort_values("neg_lift", ascending=False).head(8)[cols], hide_index=True, column_config=cfg, width="stretch")
    right_col.markdown("#### Top praise themes")
    right_col.dataframe(big.sort_values("pos_lift", ascending=False).head(8)[cols], hide_index=True, column_config=cfg, width="stretch")

    st.markdown("#### Read the voice of the customer")
    opts = t.sort_values("reviews", ascending=False)["name"].tolist()
    pick = st.selectbox("Theme", opts, index=next(i for i, n in enumerate(opts) if "generic" not in n))
    row = t[t.name == pick].iloc[0]
    ex = D["examples"]
    ex = ex[(ex.category == cat) & (ex.topic == row.topic)]
    a, b = st.columns(2)
    a.markdown("**Complaints**")
    for s in ex[ex.kind == "complaint"].sentence.head(4):
        a.markdown(f'<div class="quote bad">{esc(s)}</div>', unsafe_allow_html=True)
    b.markdown("**Praise**")
    for s in ex[ex.kind == "praise"].sentence.head(4):
        b.markdown(f'<div class="quote">{esc(s)}</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ product analyzer
with tab_prod:
    cat = st.radio("Category", list(CAT_LABEL), format_func=CAT_LABEL.get, horizontal=True, key="cat_prod")
    rv = REV[REV.category == cat]
    cs = D["stats"][cat]
    if len(REV) < 5000:
        st.info(f"Demo sample: {REV.parent_asin.nunique()} products / {len(REV)} example reviews are included here. The full "
                "corpus is not redistributed (review text belongs to its authors); category themes and trends were "
                "computed on all reviews. Rebuild everything with the scripts in the README.", icon="ℹ️")
    prods = rv.groupby(["parent_asin", "product_title"]).size().reset_index(name="n").sort_values("n", ascending=False)
    prods["label"] = prods["product_title"].fillna(prods["parent_asin"]).str.slice(0, 90) + "  (" + prods["n"].astype(str) + " reviews)"
    q = st.text_input("Search product title", "")
    shown = prods[prods["label"].str.contains(re.escape(q), case=False)] if q else prods
    if shown.empty:
        st.warning("No product matches.")
    else:
        choice = st.selectbox("Product", shown["label"].head(300).tolist())
        asin = shown[shown["label"] == choice].iloc[0]["parent_asin"]
        pr = rv[rv.parent_asin == asin].copy()
        pt = D["review_themes"][D["review_themes"].review_id.isin(pr.review_id)].merge(
            pr[["review_id", "rating"]], on="review_id")
        tm = THEMES[THEMES.category == cat].set_index("topic")
        c = st.columns(4)
        c[0].markdown(kpi("Reviews", f"{len(pr)}", f"{pr.date.min()[:4]}-{pr.date.max()[:4]}"), unsafe_allow_html=True)
        c[1].markdown(kpi("Average rating", f"{pr.rating.mean():.2f}★", f"category {cs['avg_rating']:.2f}★"), unsafe_allow_html=True)
        c[2].markdown(kpi("Negative (1-2★)", f"{(pr.rating <= 2).mean() * 100:.0f}%", f"category {cs['negative_pct']:.0f}%"),
                      unsafe_allow_html=True)
        c[3].markdown(kpi("Model: positive sentiment", f"{(pr.p_pos_lr >= 0.5).mean() * 100:.0f}%", "TF-IDF + LR on review text"),
                      unsafe_allow_html=True)
        st.markdown("")
        left, right = st.columns([1, 1.3])
        dist = pr.rating.value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
        fig = go.Figure(go.Bar(x=[f"{i}★" for i in dist.index], y=dist.values, marker_color=[RED, RED, BRONZE, GOLD, GOLD]))
        left.plotly_chart(style(fig.update_layout(title="Rating distribution"), 300), width="stretch")

        neg = pt[pt.rating <= 2].groupby("topic").size().rename("neg_mentions")
        pos = pt[pt.rating >= 4].groupby("topic").size().rename("pos_mentions")
        s = pd.concat([neg, pos], axis=1).fillna(0).join(tm[["name", "neg_lift", "pos_lift", "review_share_pct"]], how="left")
        generic = s["name"].str.contains("generic|praise", case=False, na=False)
        complaints = s[(s.neg_lift > 1.15) & (s.neg_mentions >= 2) & ~generic].sort_values("neg_mentions", ascending=False).head(4)
        strengths = s[(s.pos_lift > 1.0) & (s.pos_mentions >= 3) & ~generic].sort_values("pos_mentions", ascending=False).head(4)
        with right:
            st.markdown("**What customers complain about** (themes over-represented in 1-2★ reviews)")
            if complaints.empty:
                st.write("No recurring complaint theme (fewer than 2 negative reviews on the same theme).")
            for _tid, r in complaints.iterrows():
                st.markdown(f"- **{r['name']}**: {int(r.neg_mentions)} negative reviews")
            st.markdown("**What customers like**")
            if strengths.empty:
                st.write("No recurring praise theme.")
            for _tid, r in strengths.iterrows():
                st.markdown(f"- **{r['name']}**: {int(r.pos_mentions)} positive reviews")
        st.markdown("#### Most helpful negative and positive reviews")
        a, b = st.columns(2)
        for col, cond, title in ((a, pr.rating <= 2, "Negative"), (b, pr.rating >= 4, "Positive")):
            col.markdown(f"**{title}**")
            for r in pr[cond].sort_values("helpful_vote", ascending=False).head(3).itertuples():
                col.markdown(f'<div class="quote {"bad" if title == "Negative" else ""}"><b>{esc(r.title or "")}</b> '
                             f'({int(r.rating)}★, {r.date[:4]})<br>{esc(r.text[:420])}{"…" if len(r.text) > 420 else ""}</div>',
                             unsafe_allow_html=True)

# ------------------------------------------------------------------ trends
with tab_trend:
    cat = st.radio("Category", list(CAT_LABEL), format_func=CAT_LABEL.get, horizontal=True, key="cat_trend")
    t = THEMES[(THEMES.category == cat) & (THEMES.review_share_pct > 1.5)].copy()
    t["volume"] = t.reviews * t.neg_rate
    default = t[~t.name.str.contains("generic|praise", case=False, na=False)].nlargest(4, "volume")["name"].tolist()
    chosen = st.multiselect("Themes", t.sort_values("volume", ascending=False)["name"].tolist(), default=default)
    tr = D["trend"][(D["trend"].category == cat) & (D["trend"].quarter >= "2016Q1") & (D["trend"].reviews_in_quarter >= 150)]
    fig = go.Figure()
    for name in chosen:
        tid = t[t.name == name].iloc[0].topic
        d = tr[tr.topic == tid].sort_values("quarter")
        fig.add_trace(go.Scatter(x=d.quarter, y=d.complaint_share_pct, name=name, mode="lines+markers"))
    fig.update_yaxes(title="% of the quarter's reviews that are 1-2★ and mention the theme")
    fig.update_layout(legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(style(fig, 460), width="stretch")
    st.caption("Quarters with fewer than 150 reviews are hidden. The corpus is a sample, so use the trend "
               "direction, not exact percentages.")

# ------------------------------------------------------------------ analyze a review
with tab_try:
    st.markdown("#### Paste any review")
    ex_text = ("The blender is powerful and very easy to clean. However the lid cracked after two weeks and "
               "customer service never answered my emails. Delivery was quick though.")
    text = st.text_area("Review text", ex_text, height=130)
    cat = st.radio("Closest category", list(CAT_LABEL), format_func=CAT_LABEL.get, horizontal=True, key="cat_try")
    if text.strip():
        p_lr = float(LR.predict_proba([text])[:, 1][0])
        trf = load_transformer()
        cols = st.columns(3)
        cols[0].markdown(kpi("Sentiment (TF-IDF + LR)", "Positive" if p_lr >= 0.5 else "Negative", f"P(positive) = {p_lr:.2f}"),
                         unsafe_allow_html=True)
        if trf:
            model, tok, torch, thr, source = trf
            with torch.inference_mode():
                enc = tok([text], truncation=True, max_length=128, return_tensors="pt")
                p_ft = float(torch.softmax(model(**enc).logits, dim=-1)[0, 1])
            # prior correction: shift the logit so the validated threshold (chosen on natural class
            # frequencies) becomes 0.5; the decision is identical to "p_ft >= thr"
            logit = lambda x: float(np.log(x / (1 - x)))  # noqa: E731
            p_adj = 1 / (1 + np.exp(-(logit(min(max(p_ft, 1e-6), 1 - 1e-6)) - logit(thr))))
            cols[1].markdown(kpi("Sentiment (fine-tuned DistilBERT)", "Positive" if p_adj >= 0.5 else "Negative",
                                 f"prior-corrected P(positive) = {p_adj:.2f} (raw {p_ft:.2f})"), unsafe_allow_html=True)
        else:
            cols[1].markdown(kpi("Fine-tuned DistilBERT", "not loaded", "light deployment: needs torch (see README)"),
                             unsafe_allow_html=True)
        sents = T.split_sentences(text, min_words=3) or [text]
        theme_ids = CLF[cat].predict(sents)
        p_s = LR.predict_proba(sents)[:, 1]
        names = THEMES[THEMES.category == cat].set_index("topic")["name"].to_dict()
        rows = pd.DataFrame({"sentence": sents, "theme": [names.get(int(i), str(i)) for i in theme_ids],
                             "sentiment": np.where(p_s >= 0.5, "positive", "negative"), "P(pos)": p_s.round(2)})
        cols[2].markdown(kpi("Aspects found", str(rows.theme.nunique()), "themes across the sentences"), unsafe_allow_html=True)
        st.markdown("")
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption("Aspects: each sentence is assigned to the nearest discovered theme (TF-IDF classifier that agrees "
                   "with the embedding-based assignment on most sentences), and scored with the LR sentiment model. "
                   "Sentence-level sentiment is noisier than review-level sentiment.")

# ------------------------------------------------------------------ model accuracy
with tab_model:
    m = D["sent"]
    st.markdown(f"#### Sentiment (positive vs negative) on products the models never saw · "
                f"{m['data']['subset_test']:,}-review test subset")
    names = {"tfidf_logreg": "TF-IDF + logistic regression", "pretrained_distilbert_sst2": "DistilBERT SST-2 (zero-shot)",
             "finetuned_distilbert": "DistilBERT fine-tuned on 6.4k reviews"}
    df = pd.DataFrame([{"model": n, "accuracy": m[k]["subset_test"]["accuracy"], "macro-F1": m[k]["subset_test"]["macro_f1"],
                        "ROC-AUC": m[k]["subset_test"]["roc_auc"],
                        "negative F1": m[k]["subset_test"]["negative"]["f1"],
                        "negative recall": m[k]["subset_test"]["negative"]["recall"]} for k, n in names.items()])
    st.dataframe(df, hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="%.3f") for c in ["accuracy", "macro-F1", "ROC-AUC", "negative F1", "negative recall"]})
    st.caption(f"Positive = 4-5★, negative = 1-2★ (3★ excluded). {m['data']['positive_share_pct']:.0f}% of labelled reviews are "
               "positive, so macro-F1 and the negative-class F1 are the honest numbers, not accuracy.")
    st.markdown("#### Data source and attribution")
    st.caption(D["attribution"])
    st.markdown("#### How the corpus was built")
    st.dataframe(pd.DataFrame(D["clean"]["steps"]), hide_index=True, width="stretch")
