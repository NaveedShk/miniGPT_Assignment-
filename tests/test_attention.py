"""Scaled dot-product attention: numeric hand example + reference comparison."""
import math

import torch
import torch.nn.functional as F

from src.attention import SelfAttention, causal_mask, scaled_dot_product_attention

# 3 tokens, d_k = d_v = 2 (the same sizes as the hand calculation in the report)
Q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
K = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
V = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


def manual_attention(q, k, v, causal=False):
    """Pure-Python attention, written exactly like the hand calculation."""
    n, d = len(q), len(q[0])
    out, all_w = [], []
    for i in range(n):
        scores = [sum(q[i][c] * k[j][c] for c in range(d)) / math.sqrt(d) for j in range(n)]
        if causal:
            scores = [s if j <= i else -math.inf for j, s in enumerate(scores)]
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        w = [e / sum(exps) for e in exps]
        all_w.append(w)
        out.append([sum(w[j] * v[j][c] for j in range(n)) for c in range(len(v[0]))])
    return torch.tensor(out), torch.tensor(all_w)


def test_hand_example_matches_manual_computation():
    out, w = scaled_dot_product_attention(Q, K, V)
    exp_out, exp_w = manual_attention(Q.tolist(), K.tolist(), V.tolist())
    assert torch.allclose(w, exp_w, atol=1e-6)
    assert torch.allclose(out, exp_out, atol=1e-6)


def test_hand_example_known_values():
    # scores = QK^T / sqrt(2); row 0 = [1, 0, 1] / sqrt(2)
    _, w = scaled_dot_product_attention(Q, K, V)
    a = math.exp(1 / math.sqrt(2))
    expected_row0 = torch.tensor([a, 1.0, a]) / (2 * a + 1)
    assert torch.allclose(w[0], expected_row0, atol=1e-6)


def test_weights_are_a_distribution():
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 5, 8), torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    _, w = scaled_dot_product_attention(q, k, v)
    assert torch.all(w >= 0)
    assert torch.allclose(w.sum(-1), torch.ones(2, 5), atol=1e-6)


def test_matches_pytorch_reference():
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 3, 7, 16), torch.randn(2, 3, 7, 16), torch.randn(2, 3, 7, 16)
    for causal in (False, True):
        mask = causal_mask(7) if causal else None
        ours, _ = scaled_dot_product_attention(q, k, v, mask)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        assert torch.allclose(ours, ref, atol=1e-5)


def test_causal_hand_example():
    out, w = scaled_dot_product_attention(Q, K, V, causal_mask(3))
    exp_out, exp_w = manual_attention(Q.tolist(), K.tolist(), V.tolist(), causal=True)
    assert torch.allclose(w, exp_w, atol=1e-6)
    # token 1 can only see itself, so its output is exactly v_1
    assert torch.allclose(out[0], V[0])
    assert torch.allclose(out, exp_out, atol=1e-6)


def test_single_head_module_shapes():
    attn = SelfAttention(d_model=16, d_k=8)
    out, w = attn(torch.randn(4, 10, 16))
    assert out.shape == (4, 10, 8)
    assert w.shape == (4, 10, 10)
