"""REST API smoke test against a small randomly initialised checkpoint."""
import importlib

import pytest
from fastapi.testclient import TestClient

from src.gpt import GPTConfig, MiniGPT
from src.tokenizer import CharTokenizer
from src.utils import save_checkpoint


@pytest.fixture()
def client(tmp_path, monkeypatch):
    tok = CharTokenizer.from_text("abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ:\n,.!?'")
    model = MiniGPT(GPTConfig(vocab_size=tok.vocab_size, context_length=32, d_model=32, n_heads=2, n_layers=1))
    path = tmp_path / "api.pt"
    save_checkpoint(path, model, tok, {}, {"step": 0, "val_loss": 4.0})
    monkeypatch.setenv("CHECKPOINT_PATH", str(path))
    import api.app as app_module

    importlib.reload(app_module)  # re-read CHECKPOINT_PATH
    with TestClient(app_module.app) as c:
        yield c


def test_health_and_info(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    info = client.get("/info").json()
    assert info["model_config"]["context_length"] == 32
    assert info["parameters"] > 0


def test_generate(client):
    r = client.post("/generate", json={"prompt": "ROMEO: 42", "max_new_tokens": 50, "temperature": 0.8, "seed": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["generated_text"].startswith("ROMEO: ")
    assert body["dropped_characters"] == ["2", "4"]
    assert len(body["completion"]) == 50
    # same seed -> same text
    r2 = client.post("/generate", json={"prompt": "ROMEO: 42", "max_new_tokens": 50, "temperature": 0.8, "seed": 0})
    assert r2.json()["generated_text"] == body["generated_text"]


def test_generate_validation(client):
    assert client.post("/generate", json={"prompt": "a", "max_new_tokens": 0}).status_code == 422
    assert client.post("/generate", json={"prompt": "a", "max_new_tokens": 10**6}).status_code == 422
    assert client.post("/generate", json={"prompt": "a", "temperature": -1}).status_code == 422
