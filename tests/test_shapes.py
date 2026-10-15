"""Tensor-shape checks through every component."""
import pytest
import torch

from src.attention import MultiHeadAttention
from src.dataset import TokenStream
from src.gpt import GPTConfig, MiniGPT
from src.positional_encoding import LearnedPositionalEncoding, SinusoidalPositionalEncoding, sinusoidal_table
from src.tokenizer import CharTokenizer
from src.transformer_block import FeedForward, TransformerBlock

B, T, C, H = 2, 10, 32, 4


def test_split_and_merge_heads_are_inverse():
    mha = MultiHeadAttention(C, H)
    x = torch.randn(B, T, C)
    split = mha.split_heads(x)
    assert split.shape == (B, H, T, C // H)
    # head h holds channels [h*D, (h+1)*D) of the input
    assert torch.equal(split[:, 1], x[:, :, C // H : 2 * C // H])
    assert torch.equal(mha.merge_heads(split), x)


def test_multihead_trace():
    mha = MultiHeadAttention(C, H)
    mha.trace = []
    out = mha(torch.randn(B, T, C))
    shapes = dict(mha.trace)
    assert shapes["input x"] == (B, T, C)
    assert shapes["Q split into heads"] == (B, H, T, C // H)
    assert shapes["attention weights per head"] == (B, H, T, T)
    assert shapes["per-head outputs"] == (B, H, T, C // H)
    assert shapes["heads concatenated"] == (B, T, C)
    assert out.shape == (B, T, C)


def test_heads_must_divide_d_model():
    with pytest.raises(ValueError):
        MultiHeadAttention(30, 4)


def test_ffn_and_block_preserve_shape():
    x = torch.randn(B, T, C)
    assert FeedForward(C, 4 * C)(x).shape == (B, T, C)
    assert TransformerBlock(C, H, 4 * C)(x).shape == (B, T, C)


def test_sinusoidal_encoding():
    pe = sinusoidal_table(50, C)
    assert pe.shape == (50, C)
    # position 0: sin(0) = 0 on even dims, cos(0) = 1 on odd dims
    assert torch.allclose(pe[0, 0::2], torch.zeros(C // 2))
    assert torch.allclose(pe[0, 1::2], torch.ones(C // 2))
    assert torch.all(pe.abs() <= 1.0)
    # every position gets a distinct vector
    assert torch.cdist(pe, pe).add(torch.eye(50)).min() > 0
    assert SinusoidalPositionalEncoding(50, C)(T).shape == (T, C)
    assert LearnedPositionalEncoding(50, C)(T).shape == (T, C)


@pytest.mark.parametrize("pos", ["sinusoidal", "learned", "none"])
@pytest.mark.parametrize("tie", [False, True])
def test_gpt_forward_shapes(pos, tie):
    cfg = GPTConfig(vocab_size=11, context_length=16, d_model=C, n_heads=H, n_layers=2, pos_encoding=pos, tie_weights=tie)
    model = MiniGPT(cfg)
    x = torch.randint(0, 11, (B, T))
    logits, loss = model(x, x)
    assert logits.shape == (B, T, 11)
    assert loss.ndim == 0
    out = model.generate(x[:, :3], max_new_tokens=20, temperature=1.0, top_k=5)
    assert out.shape == (B, 23)  # longer than context_length: generation crops the window


def test_gpt_rejects_too_long_input():
    model = MiniGPT(GPTConfig(vocab_size=5, context_length=8, d_model=16, n_heads=2, n_layers=1))
    with pytest.raises(ValueError):
        model(torch.zeros(1, 9, dtype=torch.long))


def test_targets_are_inputs_shifted_by_one():
    ids = torch.arange(100)
    stream = TokenStream(ids, context_length=8)
    x, y = stream.get_window(5)
    assert torch.equal(x, torch.arange(5, 13))
    assert torch.equal(y, torch.arange(6, 14))
    xb, yb = stream.get_batch(4)
    assert xb.shape == yb.shape == (4, 8)
    assert torch.equal(xb[:, 1:], yb[:, :-1])


def test_tokenizer_roundtrip_and_determinism():
    text = "hello world\nHELLO!"
    tok = CharTokenizer.from_text(text)
    assert tok.decode(tok.encode(text)) == text
    assert CharTokenizer.from_text(text[::-1]).chars == tok.chars  # order-independent
    with pytest.raises(ValueError):
        tok.encode("hello 7")
    assert tok.decode(tok.encode("hello 7", errors="ignore")) == "hello "
