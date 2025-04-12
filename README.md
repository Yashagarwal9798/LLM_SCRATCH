# GPT from Scratch

This project is a minimalist implementation of a GPT-style language model built entirely from scratch in Python using PyTorch. The design closely follows the architecture described in the *"Attention is All You Need"* research paper, and incorporates tokenization using OpenAI's `tiktoken` library. This is intended as an educational, yet functional base for understanding and experimenting with transformer-based LLMs.

## Features

- GPT architecture (customized GPT-2 style)
- Single transformer block for simplicity and efficiency
- Token and positional embeddings
- Multi-head self-attention layers
- Feed-forward layers
- Layer normalization
- Dropout regularization
- Training loop with validation
- Supports custom datasets

## Dataset

This model was trained on the text of a Harry Potter book, consisting of:
- **66,569 characters**
- **14,708 tokens** (after tokenization)

This relatively compact dataset allows for quick experimentation while still providing rich language patterns for the model to learn.

## Dataset Examples

You can train this model on:

- Public domain books (e.g., from Project Gutenberg)
- Small dialogue datasets like `blended_skill_talk`
- Movie quotes or other curated text files

## Installation

```bash
pip install torch tiktoken datasets tqdm
```

## Training

Modify `p1.py` to load and preprocess your dataset, and then run:

```bash
python main.py
```

You can save the model using:

```python
torch.save(model.state_dict(), "gpt_model_124M.pth")
```

## Running Inference

Create a script like `test.py` to load the model weights and generate text based on a starting prompt.

## Git Ignore

Make sure to include the following in your `.gitignore` to avoid pushing large or unnecessary files:

```
__pycache__/
*.pyc
*.pth
*.pkl
*.pt
.env
venv/
datasets/
*.log
.cache/
.ipynb_checkpoints
```

## Transformer Architecture

Below is the transformer block architecture used in this implementation:

![Untitled design](https://github.com/user-attachments/assets/63cfb777-f931-403a-8a47-2db426e5e194)

- This model uses a **single transformer block** rather than 12, as in the original GPT-2, to reduce computational overhead.
- Each block contains masked multi-head attention, layer norms, and feed-forward networks.
- Input is tokenized and passed through token & positional embeddings.


## Acknowledgments

- [Attention is All You Need (Vaswani et al. 2017)](https://arxiv.org/abs/1706.03762)
- [GPT-2 OpenAI Paper](https://openai.com/research/language-unsupervised)
- [tiktoken Tokenizer](https://github.com/openai/tiktoken)

---

**Note**: This project is intended for educational purposes and is optimized for training on small to medium datasets on limited hardware.

