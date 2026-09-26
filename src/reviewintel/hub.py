"""Hugging Face Hub helpers: publish the fine-tuned model, and tell the app where to load it from.

Usage (after `hf auth login` in YOUR terminal - this code never reads or prints your token):
    python -m reviewintel.hub publish        # upload models/distilbert_finetuned to the Hub
    python -m reviewintel.hub deploy-space   # create/update a Streamlit Space that uses the Hub model

The app reads `artifacts/hf_model.json` (written by `publish`) or the env var REVIEWINTEL_HF_MODEL.
"""
from __future__ import annotations

import json
import os
import sys

from . import config as C

GITHUB_USER = "Waleed-Ilyas"  # GitHub account (differs from the Hugging Face user name)
MODEL_NAME = "distilbert-amazon-review-sentiment"
SPACE_NAME = "product-review-intelligence"
CONFIG_PATH = C.ARTIFACTS_DIR / "hf_model.json"


def model_settings() -> dict:
    """{'repo_id': str | None, 'threshold': float}. Env var wins over the committed json."""
    cfg = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    repo = os.environ.get("REVIEWINTEL_HF_MODEL") or cfg.get("repo_id")
    if repo and "/" not in repo:
        repo = None  # ignore placeholders such as "TODO"
    thr = cfg.get("threshold")
    if thr is None:
        metrics = C.ARTIFACTS_DIR / "sentiment_metrics.json"
        thr = json.loads(metrics.read_text())["thresholds"]["finetuned"] if metrics.exists() else 0.5
    return {"repo_id": repo, "threshold": float(thr)}


def model_card(username: str, metrics: dict) -> str:
    ft = metrics["finetuned_distilbert"]
    t, lr, pre = ft["subset_test_tuned_threshold"], metrics["tfidf_logreg"], metrics["pretrained_distilbert_sst2"]
    return f"""---
language: en
license: apache-2.0
base_model: distilbert-base-uncased-finetuned-sst-2-english
datasets:
- McAuley-Lab/Amazon-Reviews-2023
tags:
- text-classification
- sentiment-analysis
- amazon-reviews
- distilbert
pipeline_tag: text-classification
---

# {MODEL_NAME}

DistilBERT ([`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english))
fine-tuned for **positive vs negative sentiment of Amazon product reviews** (Appliances + Beauty).
Label 1 = positive (4-5 stars), label 0 = negative (1-2 stars). Part of the
[product-review-intelligence](https://github.com/{GITHUB_USER}/product-review-intelligence) project.

## Training
* 6,400 reviews (3,200 positive / 3,200 negative), **one epoch on a laptop CPU**, max length 96, AdamW lr 2e-5, batch 16.
* Reviews come from products disjoint from the validation/test products.

## Results (8,000 held-out reviews from products never seen in training)
| Model | Macro-F1 | ROC-AUC |
|---|---|---|
| This model, threshold 0.5 | {ft['subset_test']['macro_f1']:.3f} | {ft['subset_test']['roc_auc']:.3f} |
| **This model, threshold {t['threshold']:.2f} (recommended)** | **{t['macro_f1']:.3f}** | {t['roc_auc']:.3f} |
| DistilBERT SST-2 zero-shot | {pre['subset_test']['macro_f1']:.3f} | {pre['subset_test']['roc_auc']:.3f} |
| TF-IDF + logistic regression (78k training reviews) | {lr['subset_test']['macro_f1']:.3f} | {lr['subset_test']['roc_auc']:.3f} |

**Use the threshold {t['threshold']:.2f} on P(positive).** Fine-tuning on a balanced sample shifts the model toward
"negative" relative to the natural population (82% of labelled reviews are positive); the threshold was chosen on a
validation subset to maximise macro-F1.

## Limitations
* Trained on only two product categories and 6.4k examples; a longer fine-tune would probably improve it.
* Labels are star ratings, which are noisy; mixed 2-star and 4-star reviews are the main error source.
* English only; not evaluated for other domains. Not for automated decisions about people.

## Usage
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
tok = AutoTokenizer.from_pretrained("{username}/{MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained("{username}/{MODEL_NAME}").eval()
enc = tok(["Works great but the lid cracked after two weeks."], return_tensors="pt", truncation=True, max_length=128)
p_pos = torch.softmax(model(**enc).logits, dim=-1)[0, 1].item()
print("positive" if p_pos >= {t['threshold']:.2f} else "negative", round(p_pos, 3))
```
"""


def publish() -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    username = api.whoami()["name"]  # raises if you are not logged in; the token itself is never touched
    repo_id = f"{username}/{MODEL_NAME}"
    local = C.MODELS_DIR / "distilbert_finetuned"
    if not local.exists():
        sys.exit(f"{local} not found - run `python -m reviewintel.train_sentiment` first")
    metrics = json.loads((C.ARTIFACTS_DIR / "sentiment_metrics.json").read_text())
    (local / "README.md").write_text(model_card(username, metrics), encoding="utf-8")
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(local), repo_id=repo_id, repo_type="model",
                      commit_message="Upload fine-tuned DistilBERT for Amazon review sentiment")
    CONFIG_PATH.write_text(json.dumps({"repo_id": repo_id, "threshold": metrics["thresholds"]["finetuned"]}, indent=1))
    print(f"published https://huggingface.co/{repo_id}\nwrote {CONFIG_PATH}")


SPACE_README = """---
title: Product Review Intelligence
emoji: 💬
colorFrom: yellow
colorTo: gray
sdk: streamlit
sdk_version: "{streamlit_version}"
app_file: app/streamlit_app.py
pinned: false
license: mit
short_description: Sentiment, themes and complaint trends from Amazon reviews
---

Interactive review analyzer. Source code: https://github.com/{GITHUB_USER}/product-review-intelligence
"""


def deploy_space() -> None:
    """Create/update a Streamlit Space with the app, artifacts and the heavier requirements (torch)."""
    import importlib.metadata as md
    import shutil
    import tempfile
    from pathlib import Path

    from huggingface_hub import HfApi

    api = HfApi()
    username = api.whoami()["name"]
    if not model_settings()["repo_id"]:
        sys.exit("run `python -m reviewintel.hub publish` first so the app knows the model repo")
    space_id = f"{username}/{SPACE_NAME}"
    api.create_repo(space_id, repo_type="space", space_sdk="streamlit", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("app", "src", "artifacts", ".streamlit"):
            shutil.copytree(C.ROOT / name, root / name)
        shutil.copy(C.ROOT / "requirements-hf.txt", root / "requirements.txt")
        (root / "README.md").write_text(SPACE_README.format(GITHUB_USER=GITHUB_USER,
                                                             streamlit_version=md.version("streamlit")), encoding="utf-8")
        api.upload_folder(folder_path=str(root), repo_id=space_id, repo_type="space",
                          commit_message="Deploy Streamlit app")
    print(f"deployed https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"publish": publish, "deploy-space": deploy_space}.get(cmd, lambda: sys.exit(__doc__))()
