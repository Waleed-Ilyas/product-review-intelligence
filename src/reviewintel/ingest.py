"""Download Amazon Reviews 2023 (Hugging Face) and build a cleaned, sampled review corpus.

Usage:  python -m reviewintel.ingest

Data: McAuley-Lab/Amazon-Reviews-2023 (Hou et al., 2024), categories Appliances and All_Beauty.
The raw files are large JSON-lines; the Appliances review file is read only up to a byte budget.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from . import config as C


def _content_length(url: str) -> int:
    r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
    r.raise_for_status()
    return int(r.headers["Content-Range"].split("/")[-1])


def _fetch_range(url: str, lo: int, hi: int, retries: int = 4) -> bytes:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=120)
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            time.sleep(2**attempt)
    raise RuntimeError(f"range download failed: {url} {lo}-{hi}")


def _download(url: str, dest, max_bytes: int | None, workers: int = 8, seg: int = 16 << 20) -> None:
    """Parallel ranged download (a single HF connection is throttled to a few MB/min)."""
    if dest.exists():
        return
    total = _content_length(url)
    end = min(total, max_bytes) if max_bytes else total
    tmp = dest.with_suffix(".part")
    with open(tmp, "wb") as f:
        f.truncate(end)
    ranges = [(lo, min(lo + seg, end) - 1) for lo in range(0, end, seg)]

    def work(rng):
        data = _fetch_range(url, *rng)
        with open(tmp, "r+b") as f:
            f.seek(rng[0])
            f.write(data)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, ranges))
    tmp.rename(dest)


def download() -> None:
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for cat, cfg in C.CATEGORIES.items():
        _download(f"{C.HF_BASE}/review_categories/{cat}.jsonl", C.RAW_DIR / f"reviews_{cat}.jsonl",
                  cfg["review_bytes"])
        _download(f"{C.HF_BASE}/meta_categories/meta_{cat}.jsonl", C.RAW_DIR / f"meta_{cat}.jsonl", None)


def read_reviews(path, category: str) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # the last line of a byte-limited download is cut off
                continue
            rows.append((category, r["parent_asin"], r["asin"], r["rating"], r["title"], r["text"],
                         r["timestamp"], r["helpful_vote"], r["verified_purchase"]))
    df = pd.DataFrame(rows, columns=["category", "parent_asin", "asin", "rating", "title", "text",
                                     "timestamp", "helpful_vote", "verified_purchase"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.drop(columns="timestamp")


def read_meta(path, asins: set[str]) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["parent_asin"] in asins:
                rows.append((r["parent_asin"], r.get("title"), r.get("main_category"), r.get("store"),
                             r.get("average_rating"), r.get("rating_number")))
    return pd.DataFrame(rows, columns=["parent_asin", "product_title", "main_category", "store",
                                       "avg_rating", "rating_number"])


_TAGS = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+")
_WS = re.compile(r"\s+")
_APOS = re.compile(r"(?<=[A-Za-z])�(?=[A-Za-z])")  # "it�s" -> "it's" (encoding damage in the source)


def clean_text(s: str) -> str:
    """Strip HTML, URLs and encoding damage; collapse whitespace. Keeps case and punctuation."""
    s = _TAGS.sub(" ", s.replace("<br />", " ").replace("<br/>", " "))
    s = _URL.sub(" ", s)
    s = _APOS.sub("'", s).replace("�", " ")
    return _WS.sub(" ", s).strip()


def _sample_category(g: pd.DataFrame) -> pd.DataFrame:
    """Keep whole review sets of products with >= MIN_PRODUCT_REVIEWS_FOR_APP reviews (needed for
    product-level analysis), then fill the remaining budget with random reviews of other products."""
    counts = g.groupby("parent_asin")["review_id"].transform("size") if "review_id" in g else         g.groupby("parent_asin")["text"].transform("size")
    rich = g[counts >= C.MIN_PRODUCT_REVIEWS_FOR_APP]
    if len(rich) >= C.REVIEWS_PER_CATEGORY:
        keep = rich["parent_asin"].drop_duplicates().sample(frac=1.0, random_state=C.SEED)
        cum = rich.groupby("parent_asin").size().reindex(keep).cumsum()
        return rich[rich["parent_asin"].isin(cum[cum <= C.REVIEWS_PER_CATEGORY].index)]
    rest = g.drop(rich.index)
    fill = rest.sample(n=min(len(rest), C.REVIEWS_PER_CATEGORY - len(rich)), random_state=C.SEED)
    return pd.concat([rich, fill])


def build_corpus(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Cleaning waterfall + per-product cap + per-category sample. Returns (corpus, report)."""
    report = {"raw_reviews": len(raw), "steps": []}
    df = raw
    def step(name, before, after):
        report["steps"].append({"step": name, "removed": len(before) - len(after), "left": len(after)})

    n = df
    df = df.assign(text=df["text"].fillna("").map(clean_text), title=df["title"].fillna("").map(clean_text))
    df = df[df["rating"].between(1, 5)]
    step("rating missing / outside 1-5", n, df)
    n = df
    df = df[df["text"].str.len() >= C.MIN_REVIEW_CHARS]
    step(f"review text shorter than {C.MIN_REVIEW_CHARS} characters after cleaning", n, df)
    n = df
    df = df.drop_duplicates(subset=["category", "parent_asin", "text"])
    step("exact duplicate review text on the same product", n, df)
    n = df
    latin = df["text"].str.count(r"[A-Za-z]") / df["text"].str.len().clip(lower=1)
    df = df[latin >= 0.6]
    step("mostly non-Latin text (not English)", n, df)
    n = df
    df = df.groupby(["category", "parent_asin"], group_keys=False).sample(
        frac=1.0, random_state=C.SEED).groupby(["category", "parent_asin"]).head(C.MAX_PER_PRODUCT)
    step(f"per-product cap ({C.MAX_PER_PRODUCT} reviews)", n, df)
    n = df
    df = pd.concat([_sample_category(g) for _, g in df.groupby("category")])
    step(f"sample to {C.REVIEWS_PER_CATEGORY:,} reviews per category (whole review sets of products with >= {C.MIN_PRODUCT_REVIEWS_FOR_APP} reviews first)", n, df)
    df = df.reset_index(drop=True)
    df["review_id"] = df.index
    df["n_chars"] = df["text"].str.len()
    report["final_reviews"] = len(df)
    report["by_category"] = df["category"].value_counts().to_dict()
    report["products"] = int(df["parent_asin"].nunique())
    return df, report


def run() -> None:
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    C.ARTIFACTS_DIR.mkdir(exist_ok=True)
    download()
    raw = pd.concat([read_reviews(C.RAW_DIR / f"reviews_{c}.jsonl", c) for c in C.CATEGORIES],
                    ignore_index=True)
    corpus, report = build_corpus(raw)
    meta = pd.concat([read_meta(C.RAW_DIR / f"meta_{c}.jsonl", set(corpus.loc[corpus.category == c, "parent_asin"]))
                      for c in C.CATEGORIES], ignore_index=True).drop_duplicates("parent_asin")
    corpus = corpus.merge(meta, on="parent_asin", how="left")
    report["reviews_with_product_title"] = int(corpus["product_title"].notna().sum())
    corpus.to_parquet(C.PROCESSED_DIR / "reviews.parquet")
    (C.ARTIFACTS_DIR / "cleaning_report.json").write_text(json.dumps(report, indent=1, default=int))
    print(json.dumps(report, indent=1, default=int))


if __name__ == "__main__":
    run()
