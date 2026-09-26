"""Project-wide constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"  # small committed files the app reads
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

# McAuley-Lab/Amazon-Reviews-2023 on Hugging Face (public, no login).
HF_BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw"
# category -> (bytes of the review file to read (None = all), whole meta file)
CATEGORIES = {
    "Appliances": {"review_bytes": 320_000_000},
    "All_Beauty": {"review_bytes": None},
}
LABEL = {"Appliances": "Home appliances", "All_Beauty": "Beauty"}

# Sampling: cap reviews per product so a few blockbuster items cannot dominate the corpus.
MAX_PER_PRODUCT = 40
REVIEWS_PER_CATEGORY = 60_000
MIN_REVIEW_CHARS = 30
MIN_PRODUCT_REVIEWS_FOR_APP = 25
SEED = 42

# Sentiment target from the star rating: 1-2 = negative, 4-5 = positive; 3 stars are ambiguous and
# are excluded from the binary task (but kept for theme analysis).
POS_MIN, NEG_MAX = 4, 2

PRETRAINED_SENTIMENT = "distilbert-base-uncased-finetuned-sst-2-english"
SENTENCE_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LEN = 128
