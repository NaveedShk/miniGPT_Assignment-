# Building a GPT from Scratch: An 810K-Parameter Shakespeare Model

*Technical report — draft for Medium. Paste into Medium, upload the figures listed in each section,
fill in the `<…>` placeholders, and add the published link to the README.*

**Code:** `<GitHub repo URL>` · **Live demo:** `<Render URL>` · **Author(s):** `<names>`

---

## 1. Goal

We built a small decoder-only Transformer (a "GPT") without any pretrained weights or `transformers`
model classes, and trained it at the character level on Tiny Shakespeare. Every piece is our own PyTorch
code: tokenizer, scaled dot-product attention, causal mask, multi-head split and merge, sinusoidal
positional encoding, pre-LayerNorm blocks, language-model head, training loop, sampling, and a REST API.

## 2. Data and tokenizer

Tiny Shakespeare has 1,115,394 characters. The vocabulary is the sorted set of characters in the file,
which gives 65 tokens. Sorting makes the mapping deterministic. There are no special tokens: training uses
fixed windows (no padding) and generation stops at a length limit (no EOS).

The first 90% of the text is training data and the last 10% is validation. The split is contiguous so
overlapping windows cannot leak between them. For a window starting at `i`, the input is
`x = ids[i:i+T]` and the target is `y = ids[i+1:i+T+1]`, so one window gives T next-character examples.

## 3. Attention by hand

> *Insert photo: `docs/hand_calculation/hand_calculation_page1.jpg`*

We first computed attention for three tokens with d_k = 2 on paper, then checked it against the code.
The scores are Q Kᵀ / √d_k, softmax turns each row into weights that sum to 1, and the output is the
weighted sum of V. With the causal mask the upper triangle is set to −∞ before the softmax. Because
exp(−∞) = 0, future positions get exactly zero weight. Token 1 can then only attend to itself, so its
output becomes exactly V₁ = [1, 2] instead of [3, 4].

> *Insert figure: `outputs/figures/fig2_causal_mask_3tok.png`*

We divide by √d_k because dot products grow with dimension. Without the scaling, the softmax saturates
and the gradients vanish.

## 4. Architecture

> *Insert figure: `docs/architecture.png`*

| Setting | Value |
|---|---|
| vocabulary | 65 characters |
| context length | 128 |
| d_model / heads / head dim | 128 / 4 / 32 |
| layers | 4 (pre-LayerNorm) |
| feed-forward | 128 → 512 → 128, GELU |
| dropout | 0.1 |
| positions | sinusoidal (no parameters) |
| **parameters** | **809,984** |

Most parameters are in the feed-forward layers (526,848) and the attention projections (264,192).

The multi-head shape trace for B = 2, T = 8:

```
input x                    (2, 8, 128)
Q split into heads         (2, 4, 8, 32)
attention weights          (2, 4, 8, 8)
per-head outputs           (2, 4, 8, 32)
heads concatenated         (2, 8, 128)
output projection W_o      (2, 8, 128)
LM head logits             (2, 8, 65)
```

Design choices: pre-LN blocks keep the residual stream an identity path, which trains stably at a high
learning rate. Token embeddings are multiplied by √d_model so they enter at the same scale as the ±1
sinusoids. The output projection W_o mixes information across heads.

> *Insert figure: `outputs/figures/fig4_positional_encoding.png`*

Self-attention by itself is permutation-invariant: without positions it cannot tell "dog bites man" from
"man bites dog". The sinusoidal encoding gives each position a unique pattern, with wavelengths that grow
across dimensions, so nearby positions have similar encodings (cosine similarity 0.97 at distance 1,
0.48 at distance 64).

## 5. Training

AdamW (β = 0.9, 0.99; weight decay 0.1 on weight matrices only), peak learning rate 3e-3 with 100 warm-up
steps and cosine decay to 3e-4, gradient clipping at 1.0, batch 32 × 128 characters, 5000 steps. On a
Colab T4 GPU this took 95 seconds (~234k tokens/s). Before the full run we checked that the model can
memorise one small batch (loss 4.20 → 0.001), which shows the model and loss are wired correctly.

> *Insert figure: `outputs/figures/tiny_gpt_loss.png`*

The loss starts at 4.22, which matches uniform guessing over 65 characters (ln 65 = 4.17). The final
validation loss is 1.493 (perplexity 4.45, 2.15 bits per character). Train and validation stay close,
so the model is not overfitting.

## 6. Tests

33 unit tests pass. The two most important check causality: changing future tokens does not change the
logits of earlier positions, and the gradient of output *t* with respect to any input after *t* is exactly
zero. The same test fails for a model built without the mask, which shows that the test can catch a leak.

## 7. Generated text

Prompt `JULIET:\nO Romeo,`, temperature 0.8:

```
JULIET:
O Romeo, what is the loss of thine own conjure?

CLARENCE:
My lord, do I see, prepare you for this man.
```

The model learned the format of a play: speaker names in capitals, line breaks, verse-like rhythm and
archaic words. It does not keep meaning beyond a line or two. Greedy decoding gets stuck in loops
("That the state of the state of the sea…"); temperature 0.8 gives the best balance between variety and
real words (95% real words).

> *Add the other four samples from `outputs/samples/evaluation.md`*

## 8. What attention learned

> *Insert figure: `outputs/figures/fig7_attention_layer0.png`*

In layer 0, head 1 puts 93% of its weight on the previous character. It acts as a "previous character"
head that lets the next layers read bigrams. Deeper layers look further back: layer 3 heads look back
6–10 characters on average, and one layer-2 head about 17.

## 9. Ablations

Each ablation changes one factor and trains for 2000 steps with the same seed and data order.

| Study | Values | Best val loss |
|---|---|---|
| depth | 2 / 4 / 6 layers | 1.684 / 1.604 / 1.569 |
| heads | 2 / 4 / 8 | 1.613 / 1.604 / 1.610 |
| context | 64 / 128 / 256 | 1.664 / 1.604 / 1.563 |
| positions | sinusoidal / learned / none | 1.604 / 1.688 / 1.816 |

> *Insert figures: `outputs/ablation/ablation_depth.png`, `ablation_heads.png`, `ablation_context.png`*

**Observation:** longer context and more layers lowered validation loss; the number of heads made almost
no difference at a fixed d_model. **Interpretation:** at the character level, seeing more of the line
matters more than splitting the same 128 dimensions into more heads. The cost is memory: the attention
matrix grows with T², from 8 MB at T = 64 to 128 MB at T = 256.

## 10. Why GPT needs the causal mask

We trained the same model with full (bidirectional) attention on the next-token objective. The loss fell
to 0.04 because the model can simply copy the next character, which is visible in its input. When
generating, that character is not there, and the model produced `ROMEO::::::::`. BERT avoids this by
keeping full attention and changing the objective to predicting masked tokens, which is why BERT cannot
generate left to right.

## 11. Deployment

A FastAPI service (`/generate`, `/health`, `/info`, a small web page at `/`) loads the checkpoint with
`weights_only=True` and is deployed on Render's free tier from `render.yaml`. CLI:
`python -m src.generate --prompt "ROMEO:" --max-new-tokens 300`.

## 12. Limitations

- 128 characters is only about 25 words of context, so the text loses meaning after a line or two.
- With 0.8 M parameters and 1 M characters, a larger model would overfit quickly.
- The vocabulary is fixed to the corpus; digits in prompts are dropped.
- No KV-cache, so generation recomputes the window for every character.
- Generated text is an experimental artifact, not reliable content.

## References

1. Vaswani et al., *Attention Is All You Need*, 2017.
2. Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2), 2019.
3. Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers*, 2019.
4. Ba et al., *Layer Normalization*, 2016; Hendrycks & Gimpel, *GELUs*, 2016.
5. Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, 2019.
6. Holtzman et al., *The Curious Case of Neural Text Degeneration*, 2020.
7. A. Karpathy, char-rnn Tiny Shakespeare dataset; nanoGPT and "Let's build GPT" (consulted as learning material).
