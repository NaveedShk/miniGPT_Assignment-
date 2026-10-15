# Data

The corpus is **Tiny Shakespeare** (Karpathy, char-rnn): 1,115,394 characters, ~40,000 lines
from a variety of Shakespeare plays. It is the same text as the Hugging Face dataset
[`tiny_shakespeare`](https://huggingface.co/datasets/karpathy/tiny_shakespeare).

You do not need to download it manually. The first call to `src.dataset.load_text()`
(any training or experiment script) downloads it to `data/tinyshakespeare.txt` from

    https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

The text file is git-ignored.

Preprocessing: none (no filtering, lower-casing or normalisation). The vocabulary is the
65 distinct characters of the file; the first 90% of characters are training data and the
last 10% validation data (contiguous split).
