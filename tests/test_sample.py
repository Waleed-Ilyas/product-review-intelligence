import numpy as np
import pandas as pd

from reviewintel import sample


def _full():
    rng = np.random.default_rng(0)
    rows = []
    for cat in ("Appliances", "All_Beauty"):
        for p in range(15):
            n = 50 if p < 10 else 10  # 5 products per category are too small
            for k in range(n):
                rows.append((cat, f"{cat[:3]}{p}", f"title {p}", float(rng.choice([1, 2, 4, 5])) if p % 3 else 5.0,
                             f"text {k}", "2021-05-01", 0, 0.9))
    return pd.DataFrame(rows, columns=["category", "parent_asin", "product_title", "rating", "text", "date",
                                       "helpful_vote", "p_pos_lr"])


def test_choose_products_is_small_deterministic_and_respects_minimum():
    df = _full()
    a, b = sample.choose_products(df), sample.choose_products(df)
    assert a == b and len(a) == 2 * sample.PRODUCTS_PER_CATEGORY
    counts = df.groupby("parent_asin").size()
    assert all(counts[p] >= sample.MIN_REVIEWS for p in a)


def test_category_stats_has_no_text_and_correct_shares():
    df = _full()
    st = sample.category_stats(df)
    assert st["_all"]["reviews"] == len(df) and set(st) == {"Appliances", "All_Beauty", "_all"}
    assert 0 <= st["Appliances"]["negative_pct"] <= 100
    assert "text" not in str(st)


def test_attribution_names_the_source():
    assert "Amazon-Reviews-2023" in sample.ATTRIBUTION and "arXiv:2403.03952" in sample.ATTRIBUTION
