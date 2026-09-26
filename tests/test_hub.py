import json

from reviewintel import config as C
from reviewintel import hub


def test_placeholder_repo_id_is_ignored(monkeypatch, tmp_path):
    cfg = tmp_path / "hf_model.json"
    cfg.write_text(json.dumps({"repo_id": "TODO-set-by-publish", "threshold": 0.06}))
    monkeypatch.setattr(hub, "CONFIG_PATH", cfg)
    monkeypatch.delenv("REVIEWINTEL_HF_MODEL", raising=False)
    assert hub.model_settings() == {"repo_id": None, "threshold": 0.06}


def test_env_var_overrides_and_real_repo_is_kept(monkeypatch, tmp_path):
    cfg = tmp_path / "hf_model.json"
    cfg.write_text(json.dumps({"repo_id": "alice/model", "threshold": 0.1}))
    monkeypatch.setattr(hub, "CONFIG_PATH", cfg)
    monkeypatch.delenv("REVIEWINTEL_HF_MODEL", raising=False)
    assert hub.model_settings()["repo_id"] == "alice/model"
    monkeypatch.setenv("REVIEWINTEL_HF_MODEL", "bob/other")
    assert hub.model_settings()["repo_id"] == "bob/other"


def test_model_card_states_the_recommended_threshold():
    metrics = json.loads((C.ARTIFACTS_DIR / "sentiment_metrics.json").read_text())
    card = hub.model_card("alice", metrics)
    assert card.startswith("---") and "alice/distilbert-amazon-review-sentiment" in card
    assert "threshold 0.06" in card and "base_model:" in card
