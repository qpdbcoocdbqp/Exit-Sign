# Exit-Sign
Fine-tune scripts. Playing with [Exit Sign](https://www.youtube.com/watch?v=uIm7njrSZak)

* **About Exit Sign**

    > Hilltop Hoods
    >
    > The Great Expanse

## Fable-5-traces x Qwen3-0.6B — SFT Fine-tuning

Supervised Fine-Tuning (SFT) of Qwen3-0.6B on the
[Glint-Research/Fable-5-traces](https://huggingface.co/datasets/Glint-Research/Fable-5-traces) dataset.
This is the first step of an RLHF pipeline (Behavior Cloning).

---

## Requirements

- Python 3.10+
- CUDA 13.0 compatible GPU (≥ 8 GB VRAM recommended)
- [uv](https://docs.astral.sh/uv/) package manager

---

## Installation

### 1. Create virtual environment

```bash
uv venv .venv
.venv\Scripts\activate   # Windows
```

### 2. Install PyTorch (CUDA 13.0)

```bash
uv pip install torch==2.12.1 torchvision \
    --index-url https://download.pytorch.org/whl/cu130
```

### 3. Install Python packages

```bash
uv pip install transformers datasets trl peft accelerate bitsandbytes
```

---

## Dataset

Download the dataset locally before training to avoid repeated network fetches:

```bash
huggingface-cli download Glint-Research/Fable-5-traces --repo-type dataset
```

---

## Fine-tuning: `finetune_fable5_qwen3.py`

### What the script does

| Step | Description |
|------|-------------|
| **1. Load dataset** | Loads `Glint-Research/Fable-5-traces` and filters rows where `type == "message"` |
| **2. Extract text** | Parses each message's `content` block list — handles `thinking` (CoT), `text`, and `toolCall` blocks |
| **3. Build pairs** | Matches each assistant message to its parent user message via `parentId`, producing `(prompt, completion)` pairs in TRL conversational format |
| **4. Load model** | Loads `Qwen/Qwen3-0.6B` in 4-bit NF4 quantisation (~2 GB VRAM) via `BitsAndBytesConfig` |
| **5. LoRA config** | Attaches LoRA adapters (r=16, alpha=32) to all attention and MLP projection layers (~1–2% trainable parameters) |
| **6. SFTConfig** | Configures training: cosine LR schedule, `paged_adamw_8bit`, `bf16`, `assistant_only_loss=True` to mask prompt tokens from the loss |
| **7. Train** | Runs `SFTTrainer.train()` — chat template application and tokenisation are handled automatically |
| **8. Save** | Saves the LoRA adapter to `./qwen3-fable5-sft/final` |
| **9. Inference** | Merges the adapter into the base model and runs a smoke-test generation |

### Key design choices

- **`assistant_only_loss=True`** — loss is computed on the assistant's completion only; TRL auto-patches the Qwen3 chat template with `{% generation %}` markers
- **Conversational prompt-completion format** — no manual tokenisation or label masking needed; SFTTrainer handles it end-to-end
- **`peft_config` passed to SFTTrainer** — no manual `get_peft_model()` call required

### Run

```bash
python finetune_fable5_qwen3.py
```

Output adapter is saved to `./qwen3-fable5-sft/final`.

### Dry-run vs full training

The script defaults to 1% of data for a quick smoke-test.
For full training, change the split in the script:

```python
# dry-run (default)
split = hf_dataset.train_test_split(train_size=0.01, test_size=0.01, seed=42)

# full training
split = hf_dataset.train_test_split(train_size=0.9, test_size=0.1, seed=42)
```

---

## Output

```
qwen3-fable5-sft/
├── checkpoint-200/        # intermediate checkpoints
├── checkpoint-400/
└── final/                 # final merged adapter
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
```
