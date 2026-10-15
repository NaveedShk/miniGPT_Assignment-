# Hand-calculated attention (Figure 1)

> **TODO before submitting:** photograph or scan your handwritten calculation and save it in this
> folder as `hand_calculation_page1.jpg` (and `_page2.jpg`, … if needed). Then delete this TODO line.

![Hand calculation, page 1](hand_calculation_page1.jpg)

## The example worked by hand

Three tokens, d_k = 2:

```
Q = K = [[1, 0],        V = [[1, 2],
         [0, 1],             [3, 4],
         [1, 1]]             [5, 6]]
```

Steps to show on paper: (1) Q Kᵀ, (2) divide by √2, (3) row-wise softmax, (4) multiply by V,
then repeat with the causal mask: (5) set the upper triangle to −∞, (6) softmax, (7) multiply by V.

## Check against the code

`python -m experiments.attention_numeric_example` computes the same example step by step and asserts
that `src.attention.scaled_dot_product_attention` gives the same answer. The output (from the Colab run,
also in `outputs/logs/attention_numeric_and_shape_trace.txt`):

| Step | Result |
|---|---|
| 1) Q Kᵀ | `[[1, 0, 1], [0, 1, 1], [1, 1, 2]]` |
| 2) ÷ √2 | `[[0.7071, 0, 0.7071], [0, 0.7071, 0.7071], [0.7071, 0.7071, 1.4142]]` |
| 3) softmax | `[[0.4011, 0.1978, 0.4011], [0.1978, 0.4011, 0.4011], [0.2483, 0.2483, 0.5035]]` |
| 4) · V | `[[3.0000, 4.0000], [3.4067, 4.4067], [3.5105, 4.5105]]` |
| 5) masked scores | `[[0.7071, −∞, −∞], [0, 0.7071, −∞], [0.7071, 0.7071, 1.4142]]` |
| 6) masked softmax | `[[1, 0, 0], [0.3302, 0.6698, 0], [0.2483, 0.2483, 0.5035]]` |
| 7) masked output | `[[1.0000, 2.0000], [2.3395, 3.3395], [3.5105, 4.5105]]` |

What the mask changes: token 1 can now only see itself, so its output is exactly V₁ = [1, 2].
Token 2 sees tokens 1–2 only. Token 3 already sat at the end of the sequence, so its row is unchanged.

The causal-mask figures are `outputs/figures/fig2_causal_mask_3tok.png` and `fig2_causal_mask_12.png`.
