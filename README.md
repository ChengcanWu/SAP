# SAP — Reference Implementation

This repository contains a reference implementation of the **SAP** method, built on **Llama-2-7B-Chat**, **LoRA**, and a **contrastive-gradient**-style training procedure. Please read the published paper for the full algorithm and notation.

> **Note:** Training expects local data paths and downloaded weights. After cloning, prepare `dataset/` and the model directory as described below, and configure your GPU and Python environment.

---

## Requirements

| Item | Recommendation |
|------|----------------|
| **Python** | 3.10+ (3.10 or 3.11 is a safe default) |
| **GPU** | NVIDIA GPU; memory depends on batch size and sequence length. The scripts use `bfloat16` and `device_map="auto"`; **CPU training is not supported out of the box**. |
| **CUDA** | Must match the installed `torch` build. Pick the install command from the [PyTorch “Get Started”](https://pytorch.org/get-started/locally/) page. |

---

## Setup

### 1. Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 2. Install PyTorch

Install PyTorch for your **CUDA** version from the official site, for example (check the site for the exact command):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install the remaining dependencies

From the repository root (this README’s directory):

```bash
pip install -r requirements.txt
```

---

## Data

The code uses fixed paths. Place the files there or **change** the paths in `main.py` to match your setup.

| Path | Purpose | Notes |
|------|---------|--------|
| `./dataset/alpaca/alpaca-cleaned-train.arrow` | Alpaca SFT | Must be **Arrow** format readable with HuggingFace `Dataset.from_file`. You can build it from a public cleaned-Alpaca export. |
| `./dataset/justinphan/train-00000-of-00001.parquet` | Contrastive (chosen / rejected) | **Parquet** columns must match `load_contrastive_data` in `datas.py`: `prompt`, `llama3_output`, `response`. If your schema differs, edit `ic_preprocess_function`. |

Example layout:

```text
SAP_code/
  dataset/
    alpaca/
      alpaca-cleaned-train.arrow
    justinphan/
      train-00000-of-00001.parquet
```

---

## Model and licensing

- **Base model:** In `main.py`, `model_path = "Llama-2-7b-chat-hf"` is a **local directory name**. Download [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) under Meta’s terms, then place it in this directory as `Llama-2-7b-chat-hf`, **or** set `model_path` to `meta-llama/Llama-2-7b-chat-hf` and run `huggingface-cli login` first.
- Llama 2 is governed by the **Meta Llama 2 Community License**. This repository **does not** ship any model weights.

---

## Run

After data and model paths are configured:

```bash
python main.py
```

This runs `test(...)` with the hyperparameters at the end of `main.py` (e.g. `w_lr`, `v_lr`, `v_register_layer`, `num_epochs`). To run other experiments, edit that call, or add `argparse` / a config file.

---

## Layout

```text
main.py     # Entry: data loading, epochs, and `test` hyperparameters
train.py    # Training step: contrastive gradients, LoRA/bias updates, validation loss
model.py    # LoRA load, trainable bias, tokenizer, chat template
datas.py    # Alpaca / AdvBench / contrastive DataLoaders
utils.py    # LoRA gradient merge, norms, zeroing optimizer-side tensors, etc.
```

`load_advbench_data` is also in `datas.py` but is **not** used in `main.py` today. If your paper includes AdvBench, wire it up with your own paths and columns.

---

## Known caveats

1. Training expects **CUDA** and enough memory; on OOM, reduce `batch_size`, `max_token`, etc. (see `load_alpaca_data` / `load_contrastive_data` in `main.py` and `datas.py`).

---

## Citation

If you use this code or the **SAP** paper, please cite the final publication (update the BibTeX when the camera-ready is available):

---

## Contact

For questions, reach the corresponding author(s) of the paper, or use [GitHub Issues](https://github.com/your-username/your-repo) (replace with your real URL after publishing).
