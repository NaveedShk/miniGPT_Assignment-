# Building a GPT from Scratch: An 810K-Parameter Shakespeare Model, One Tensor at a Time

*A technical report on MiniGPT, a character-level, decoder-only Transformer written in plain PyTorch, trained on Tiny Shakespeare and deployed as a live API.*

**Code:** [https://github.com/NaveedShk/miniGPT_Assignment-](https://github.com/NaveedShk/miniGPT_Assignment-) · **Live demo:** [https://minigpt-assignment.onrender.com](https://minigpt-assignment.onrender.com) · **Demo video:** `<your video link>`

---

## 1. Why build a GPT from scratch?

Calling `AutoModelForCausalLM.from_pretrained(...)` teaches you how to use a language model, not how one works. The goal of this project was to write every moving part myself and test each one: tokenizer, scaled dot-product attention, causal mask, multi-head split and merge, positional encoding, Transformer block, training loop, sampling and serving. No pretrained weights and no `transformers` model classes are used.

The result is **MiniGPT**, a 4-layer, 4-head, 128-dimensional decoder-only Transformer with **809,984 parameters**. It reaches a validation perplexity of **4.45** on Tiny Shakespeare after 95 seconds of training on a free Colab T4 GPU.

Transformers replaced RNNs for a simple reason. An RNN has to process token *t−1* before token *t*, so its sequential depth is O(T). Self-attention computes all positions at once, so its depth is O(1). The cost is memory: the attention matrix is T × T. A quick benchmark in the repo (`experiments/why_transformers.py`) shows this on a T4. At T = 512, a Python-loop RNN takes 51 ms per forward pass and attention takes 4.6 ms, but the attention scores need 64 MB against 4 MB for the RNN hidden states.

## 2. Data and tokenizer

The dataset is Tiny Shakespeare (Karpathy's char-rnn corpus): 1,115,394 characters.

- **Tokenizer:** character-level. The vocabulary is the *sorted* set of characters, giving **65 tokens**. Sorting makes the id mapping deterministic across machines.
- **No special tokens.** Training uses fixed-length windows, so no padding is needed, and generation stops at a length limit, so no EOS is needed.
- **Split:** the first 90% is used for training and the last 10% for validation. The split is *contiguous*, not random, so overlapping windows can't leak text between the two sets.
- **Targets:** for a window starting at *i*, `x = ids[i : i+T]` and `y = ids[i+1 : i+T+1]`. One window of length T gives T next-character prediction problems at once.

## 3. Attention, by hand and in code

Scaled dot-product attention is

`Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) · V`

To be sure the code matches the maths, I worked a 3-token example with d_k = 2 by hand: Q = K = [[1,0],[0,1],[1,1]] and V = [[1,2],[3,4],[5,6]].

1. Q Kᵀ = [[1,0,1],[0,1,1],[1,1,2]]
2. Dividing by √2 gives [[0.707,0,0.707],[0,0.707,0.707],[0.707,0.707,1.414]]
3. Row-wise softmax gives [[0.401,0.198,0.401],[0.198,0.401,0.401],[0.248,0.248,0.504]]
4. Multiplying by V gives [[3.00,4.00],[3.41,4.41],[3.51,4.51]]

> 📷 **[Insert photo of handwritten calculation: docs/hand_calculation/hand_calculation_page1.jpg]**

`experiments/attention_numeric_example.py` prints the same steps from `src/attention.py`, and `tests/test_attention.py` checks the function against a pure-Python version and against PyTorch's built-in `scaled_dot_product_attention`.

**Why divide by √d_k?** If the entries of q and k are roughly unit-variance, their dot product has variance d_k. Large scores push the softmax into a near one-hot regime where its gradients vanish. Dividing by √d_k brings the variance back to about 1.

## 4. The causal mask

A GPT predicts the next character, so position *t* must not see positions after *t*. The mask is a lower-triangular boolean matrix. Blocked scores are set to **−∞** rather than a large negative number, because e^(−∞) is *exactly* 0.

Applied to the hand example, row 1 becomes [1, 0, 0] and its output is just v₁ = (1, 2). Row 2 becomes [0.330, 0.670, 0] with output (2.34, 3.34). Row 3 is unchanged, because the last token already "sees" everything.

> 🖼️ **[Insert images/fig2_causal_mask_12.png]**

The tests go further than checking the shape of the matrix:

- future attention weights are exactly 0 in every head;
- changing a future token does not change any earlier logit;
- the **gradient** of output *t* with respect to every input after *t* is exactly zero, and a bidirectional model fails the same check.

All 33 unit tests pass (`python -m pytest`).

### What happens without the mask?

This was the most instructive experiment in the project (`experiments/mask_comparison.py`, 1500 steps). I trained three small models:

> 🖼️ **[On Medium: insert images/table1_mask_comparison.png here instead of the table below]**

| Variant | Train loss | Val loss | Val loss (last position only) | Sample |
|---|---|---|---|---|
| Causal + next-token (GPT) | 1.58 | 1.76 | 1.72 | Shakespeare-like |
| Bidirectional + next-token | **0.04** | **0.04** | 2.26 | `ROMEO::::::::::` |
| Bidirectional + masked-LM (BERT) | 2.03 | 1.99 | n/a | can't generate |

Without the mask, the "next character" is already visible in the input, so the model learns to copy it and the loss collapses to almost zero. At generation time the future isn't there, and the model emits a stream of colons. BERT keeps full attention but changes the *objective* to predicting masked tokens, which is why BERT is not a left-to-right generator.

> 🖼️ **[Insert images/fig_mlm_vs_clm_losses.png]**

## 5. Multi-head attention and its shapes

With d_model = 128 and 4 heads, each head works in 32 dimensions. The shape trace from `experiments/shape_trace.py` for B = 2 and T = 8:

```
input x                      (2, 8, 128)
Q = x W_q                    (2, 8, 128)
Q split into heads        (2, 4, 8, 32)    # view (B,T,H,D) then transpose -> (B,H,T,D)
attention weights per head (2, 4, 8, 8)
per-head outputs          (2, 4, 8, 32)
heads concatenated           (2, 8, 128)   # transpose back, .contiguous().view(...)
output projection W_o        (2, 8, 128)
```

The `.contiguous()` call is needed because `transpose` only changes strides, and `.view()` requires contiguous memory. The output projection W_o mixes information across heads. Without it, each head's 32 dimensions would stay isolated.

## 6. Positional encoding

Attention is permutation-invariant: with no position signal, "dog bites man" and "man bites dog" contain the same set of tokens. MiniGPT adds **sinusoidal** encodings, where PE(pos, 2i) = sin(pos / 10000^(2i/d)) and PE(pos, 2i+1) = cos(…). The wavelengths range from 2π to about 10000·2π across dimensions, so low dimensions encode fine position and high dimensions encode coarse position. The cosine similarity between encodings falls smoothly with distance: 0.97 at distance 1, 0.76 at 4, 0.62 at 16 and 0.48 at 64.

> 🖼️ **[Insert images/fig4_positional_encoding.png]**

Token embeddings are multiplied by √d_model so they enter at the same scale as the ±1 sinusoids. Otherwise the position signal would drown out the token signal at initialisation.

## 7. The full architecture

> 🖼️ **[Insert images/architecture.png]**

Each of the 4 blocks is **pre-LayerNorm** (GPT-2 style):
`x = x + Attn(LN(x))`, then `x = x + FFN(LN(x))`. The FFN expands 128 → 512 → 128 with GELU. Pre-LN keeps the residual stream an identity path, which gives gradients a direct route and makes training with a high learning rate stable.

> 🖼️ **[On Medium: insert images/table2_parameters.png here instead of the table below]**

| Component | Parameters |
|---|---|
| Token embedding (65 × 128) | 8,320 |
| Attention (4 blocks × Q, K, V, O) | 264,192 |
| Feed-forward (4 blocks) | 526,848 |
| LayerNorms | 2,304 |
| LM head (128 → 65) | 8,320 |
| **Total** | **809,984** |

Two-thirds of the parameters sit in the feed-forward layers, which is typical for Transformers.

## 8. Training

> 🖼️ **[On Medium: insert images/table3_training_config.png here instead of the table below]**

| Setting | Value |
|---|---|
| Context length | 128 characters |
| Batch | 32 × 128 = 4,096 tokens/step |
| Steps | 5,000 (≈ 20 epochs) |
| Optimizer | AdamW, β = (0.9, 0.99), weight decay 0.1 on weight matrices only |
| Learning rate | 3e-3 peak, 100 warm-up steps, cosine decay to 3e-4 |
| Gradient clipping | 1.0 |
| Dropout | 0.1 |
| Hardware | Tesla T4 (Colab), 94.6 s, ~234k tokens/s, 249 MB peak memory |

Before the real run, an **overfit gate** checks that the model can memorise one tiny batch. The loss goes from 4.196 (≈ ln 65 = 4.174, uniform guessing) to 0.0013. This catches most wiring bugs in seconds.

> 🖼️ **[Insert images/tiny_gpt_loss.png]**

The loss starts at ln 65, drops quickly in the first 1000 steps, then slowly improves. The best validation loss of **1.4928** came at step 4750, and that checkpoint is the one saved. On the full validation split this is **perplexity 4.45 (2.15 bits per character)**. Validation loss was still improving (slowly) until step 4750 and never turned upward, so the model is not overfitting yet. The train/val gap does widen to about 0.23 nats by the end, which is the thing to watch: with more steps or a bigger model, a 1M-character corpus would start to overfit.

## 9. Generation

At each step, only the logits for the last position are used. They are divided by a temperature, optionally filtered with top-k or top-p, and sampled. Here are the five fixed-prompt samples (temperature 0.8, seed 42), shortened to the first few lines:

```
ROMEO:
As look and that I make him down that noble
And they fear her place of quite, and that's then?
```

```
JULIET:
O Romeo, what is the loss of thine own conjure?

CLARENCE:
My lord, do I see, prepare you for this man.
```

```
KING HENRY:
What news ling only that invocase thee?

BENVOLIO:
Who shall think you do come?
```

```
First Citizen:
We think the desire of so much make me my son,
That I think the subjects thou very well.
```

```
To be, or not to be the prince.

SICINIUS:
Is that he proclaim'd more crown: why, is't the glock,
```

The full 300-character versions are in `outputs/samples/evaluation.md`.

The model has learned the format (speaker names in capitals, line breaks, verse rhythm) and plenty of real words, but it loses meaning after a line or two. Decoding settings matter a lot:

> 🖼️ **[On Medium: insert images/table4_decoding.png here instead of the table below]**

| Decoding | Distinct words | Repeated 4-grams | Real words |
|---|---|---|---|
| Greedy (T = 0) | 0.22 | 73% | 100% |
| T = 0.5 | 0.69 | 20% | 98% |
| T = 0.8 | 0.77 | 12% | 95% |
| T = 1.5 | 0.94 | 5% | 75% |

Greedy decoding falls into a loop: *"That the state of the state of the state of the sea…"*.

## 10. What the attention heads learned

> 🖼️ **[Insert images/fig7_attention_layer0.png]**

On the prompt *"ROMEO:\nBut soft, what light through yonder window breaks?"*, **layer 0 head 1** puts **93%** of its weight on the previous character. It is a "previous-token head", which makes sense as the first step in building bigram statistics. Deeper layers look further back. Layer 3 heads average about 6–10 characters of look-back, and layer 2 head 1 averages 17 characters, roughly the length of the current word or phrase. This is an observation from one prompt. Calling these heads "word" or "line" detectors would need more evidence.

## 11. Ablations

Each ablation changes one factor from the baseline, for 2000 steps with the same seed and data order.

> 🖼️ **[On Medium: insert images/table5_ablations.png here instead of the table below]**

| Study | Values | Best val loss | Cost |
|---|---|---|---|
| Depth | 2 / **4** / 6 layers | 1.684 / 1.604 / **1.569** | 20 s / 38 s / 55 s |
| Heads | 2 / **4** / 8 | 1.613 / **1.604** / 1.610 | 33 s / 37 s / 46 s |
| Context | 64 / **128** / 256 | 1.664 / 1.604 / **1.563** | attention memory 8 / 32 / 128 MB |
| Position | **sinusoidal** / learned / none | **1.604** / 1.688 / 1.816 | — |

**Observations.** More depth and more context both help. Head count at a fixed width barely matters (a 0.01-nat spread), but more heads cost time and memory. Removing positional encoding hurts the most.

**Interpretation.** Characters need context to form words, so longer context and more layers to combine it pay off. Splitting the same 128 dimensions into more heads gives no new capacity. These are single-seed runs, so differences of about 0.01 are within noise.

> 🖼️ **[Insert images/ablation_depth.png and ablation_context.png]**

## 12. Deployment

The model is served with **FastAPI** (`api/app.py`): `POST /generate`, `GET /info`, `GET /health`, a small web page at `/` and interactive docs at `/docs`. It runs on Render's free plan using `render.yaml` (CPU-only PyTorch). The 3 MB checkpoint is loaded with `torch.load(weights_only=True)`, so a downloaded checkpoint can't execute arbitrary code.

> 🖼️ **[Insert docs/screenshots/render_live_demo.png]**

```bash
curl -X POST https://minigpt-assignment.onrender.com/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "ROMEO:", "max_new_tokens": 200, "seed": 0}'
```

## 13. Limitations

- **Short memory.** 128 characters is only about 25 words, so the text loses meaning after a line or two.
- **Character-level.** The model must learn spelling, and it still invents non-words ("commplor", "invocase").
- **Small data.** With 1M characters, the train/val gap grows quickly with more steps or a bigger model.
- **Fixed vocabulary.** Digits and non-ASCII characters in prompts are dropped.
- **No KV-cache.** Each new character recomputes the whole window. That's fine at this size but would not scale.
- Generated text is an experimental artifact, not factual or reliable content.

## 14. What I learned

The tests taught me the most. Shapes can line up perfectly while the model quietly cheats by looking at the future, and only the gradient-causality test and the "no mask" experiment made that visible. Second, small details such as the √d_k scaling, the embedding scale and pre-LN decide whether a high learning rate trains smoothly or diverges.

## References

1. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
2. Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2), 2019.
3. Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL 2019.
4. A. Karpathy, *char-rnn* (Tiny Shakespeare dataset), and the *nanoGPT* / "Let's build GPT" material, used as learning references. The code in this project was written for the assignment.
5. Hendrycks & Gimpel, *Gaussian Error Linear Units (GELUs)*, 2016.
6. Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW), ICLR 2019.
7. Ba, Kiros & Hinton, *Layer Normalization*, 2016.
8. Holtzman et al., *The Curious Case of Neural Text Degeneration* (top-p / nucleus sampling), ICLR 2020.
