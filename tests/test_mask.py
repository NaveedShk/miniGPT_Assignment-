"""Causal correctness: position t must never use information from positions > t."""
import torch

from src.attention import MultiHeadAttention, causal_mask, scaled_dot_product_attention
from src.gpt import GPTConfig, MiniGPT


def test_causal_mask_is_lower_triangular():
    m = causal_mask(4)
    expected = torch.tensor(
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]], dtype=torch.bool)
    assert torch.equal(m, expected)


def test_future_positions_receive_zero_weight():
    torch.manual_seed(0)
    q, k, v = torch.randn(3, 6, 8), torch.randn(3, 6, 8), torch.randn(3, 6, 8)
    _, w = scaled_dot_product_attention(q, k, v, causal_mask(6))
    future = ~causal_mask(6)
    assert torch.all(w[:, future] == 0)
    assert torch.allclose(w.sum(-1), torch.ones(3, 6))


def test_multihead_attention_weights_are_causal_in_every_head():
    mha = MultiHeadAttention(32, 4, causal=True)
    mha.store_attention = True
    mha(torch.randn(2, 9, 32))
    w = mha.last_attention  # (B, H, T, T)
    assert w.shape == (2, 4, 9, 9)
    assert torch.all(w[..., ~causal_mask(9)] == 0)


def _model(causal=True):
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=20, context_length=16, d_model=32, n_heads=4, n_layers=2, dropout=0.0, causal=causal)
    return MiniGPT(cfg).eval()


def test_changing_future_tokens_does_not_change_past_logits():
    model = _model()
    x = torch.randint(0, 20, (2, 16))
    t = 7
    x2 = x.clone()
    x2[:, t + 1:] = torch.randint(0, 20, (2, 16 - t - 1))  # rewrite the future
    logits1, _ = model(x)
    logits2, _ = model(x2)
    assert torch.allclose(logits1[:, : t + 1], logits2[:, : t + 1], atol=1e-6)


def test_gradient_of_past_output_wrt_future_input_is_zero():
    model = _model()
    x = torch.randint(0, 20, (1, 12))
    emb = (model.token_emb(x) * model.config.d_model ** 0.5).detach().requires_grad_(True)
    h = emb + model.pos_enc(12)
    for block in model.blocks:
        h = block(h)
    t = 5
    h[0, t].sum().backward()
    grad_per_position = emb.grad[0].abs().sum(-1)
    assert torch.all(grad_per_position[t + 1:] == 0)    # future: no influence
    assert torch.all(grad_per_position[: t + 1] > 0)    # past and present: influence


def test_bidirectional_model_does_see_the_future():
    """Contrast case (BERT-style attention): the same check must FAIL without the mask."""
    model = _model(causal=False)
    x = torch.randint(0, 20, (1, 16))
    x2 = x.clone()
    x2[0, -1] = (x[0, -1] + 1) % 20
    logits1, _ = model(x)
    logits2, _ = model(x2)
    assert not torch.allclose(logits1[:, 0], logits2[:, 0])


def test_generation_matches_full_forward():
    """Generating token by token gives the same next-token distribution as one full pass."""
    model = _model()
    x = torch.randint(0, 20, (1, 10))
    full_logits, _ = model(x)
    for t in range(1, 10):
        prefix_logits, _ = model(x[:, :t])
        assert torch.allclose(prefix_logits[0, -1], full_logits[0, t - 1], atol=1e-5)
