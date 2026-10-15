"""FastAPI service exposing MiniGPT generation.

    uvicorn api.app:app --host 0.0.0.0 --port 8000

Environment variables
    CHECKPOINT_PATH   path to the .pt file (default checkpoints/tiny_gpt.pt)
    CHECKPOINT_URL    optional URL to download the checkpoint from if it is missing
    MAX_NEW_TOKENS_LIMIT  upper bound accepted by /generate (default 1000)
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.generate import generate_text
from src.utils import ROOT, count_parameters, load_checkpoint

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", str(ROOT / "checkpoints" / "tiny_gpt.pt"))
MAX_NEW_TOKENS_LIMIT = int(os.environ.get("MAX_NEW_TOKENS_LIMIT", "1000"))
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Small model + small shared CPU on free hosting: extra threads only add overhead.
    # Override with TORCH_NUM_THREADS if the host has dedicated cores.
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
    model, tok, ckpt = load_checkpoint(CHECKPOINT_PATH, "cpu")
    STATE.update(model=model, tokenizer=tok, ckpt=ckpt)
    yield
    STATE.clear()


app = FastAPI(title="MiniGPT", description="Character-level GPT trained from scratch on Tiny Shakespeare", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field("ROMEO:", max_length=2000)
    max_new_tokens: int = Field(200, ge=1)
    temperature: float = Field(0.8, ge=0.0, le=5.0, description="0 = greedy")
    top_k: int | None = Field(None, ge=1)
    top_p: float | None = Field(None, gt=0.0, le=1.0)
    seed: int | None = None


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    completion: str
    dropped_characters: list[str]
    new_tokens: int
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in STATE}


@app.get("/info")
def info():
    model, ckpt = STATE["model"], STATE["ckpt"]
    return {
        "parameters": count_parameters(model),
        "model_config": ckpt["model_config"],
        "vocab_size": STATE["tokenizer"].vocab_size,
        "checkpoint_metrics": ckpt["metrics"],
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if req.max_new_tokens > MAX_NEW_TOKENS_LIMIT:
        raise HTTPException(422, f"max_new_tokens must be <= {MAX_NEW_TOKENS_LIMIT}")
    t0 = time.perf_counter()
    text, dropped = generate_text(
        STATE["model"], STATE["tokenizer"], req.prompt, req.max_new_tokens,
        req.temperature, req.top_k, req.top_p, req.seed,
    )
    prompt_used = "".join(ch for ch in req.prompt if ch not in dropped)
    return GenerateResponse(
        prompt=req.prompt,
        generated_text=text,
        completion=text[len(prompt_used):] if prompt_used else text,
        dropped_characters=dropped,
        new_tokens=req.max_new_tokens,
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


@app.get("/", response_class=HTMLResponse)
def index():
    """A minimal browser UI for the live demo."""
    return """<!doctype html><html><head><meta charset="utf-8"><title>MiniGPT</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem}
textarea{width:100%;font-family:monospace}pre{white-space:pre-wrap;background:#f4f4f4;padding:1rem}
label{margin-right:1rem}</style></head><body>
<h1>MiniGPT &mdash; Tiny Shakespeare</h1>
<p>A character-level GPT trained from scratch. Docs: <a href="/docs">/docs</a></p>
<textarea id="p" rows="3">ROMEO:</textarea><br>
<label>tokens <input id="n" type="number" value="300" min="1" max="1000"></label>
<label>temperature <input id="t" type="number" value="0.8" step="0.1" min="0" max="5"></label>
<button onclick="go()">Generate</button><pre id="o"></pre>
<script>async function go(){const o=document.getElementById('o');o.textContent='...';
const r=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({prompt:document.getElementById('p').value,
max_new_tokens:+document.getElementById('n').value,temperature:+document.getElementById('t').value})});
const j=await r.json();o.textContent=r.ok?j.generated_text+'\\n\\n['+j.latency_ms+' ms]':JSON.stringify(j);}</script>
</body></html>"""
