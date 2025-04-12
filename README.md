# 🧠 LLM From Scratch — A Minimal GPT-Style Language Model

This project is an educational implementation of a GPT-style Language Model built from scratch using PyTorch. It includes everything from tokenization to model training and inference.

---

## 🚀 Features

- Byte Pair Encoding (BPE) Tokenizer using [tiktoken](https://github.com/openai/tiktoken)
- Transformer-based language model (GPT-like)
- Clean training and validation loop
- Support for modern datasets (e.g., Blended Skill Talk)
- Model checkpointing and inference
- Trains on any `.txt` file or Hugging Face dataset

---

## 🛠️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/llm-from-scratch.git
cd llm-from-scratch
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
