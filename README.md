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

## Project layout

```
src/
├── train.py, inference.py, quantization.py, evaluate.py   # CLI scripts (python -m src.<name>)
└── module/
    ├── utils.py          # shared paths, stage timer, device helper
    ├── train.py          # dataset pairs, LoRA/SFT config, training
    ├── inference.py      # load 4-bit base + adapter, generate
    ├── quantization.py   # NF4 base/adapter export and loading
    └── evaluate.py       # logprob comparison, base vs fine-tuned generation
```

Run every command from the repository root.

---

## Fine-tuning: `src/train.py`

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

### Key design choices

- **`assistant_only_loss=True`** — loss is computed on the assistant's completion only; TRL auto-patches the Qwen3 chat template with `{% generation %}` markers
- **Conversational prompt-completion format** — no manual tokenisation or label masking needed; SFTTrainer handles it end-to-end
- **`peft_config` passed to SFTTrainer** — no manual `get_peft_model()` call required

### Run

```bash
python -m src.train
```

Output adapter is saved to `./qwen3-fable5-sft/final`.

### Dry-run vs full training

The script defaults to 10% train / 10% eval for a quick smoke-test.
For full training, pass the split sizes:

```bash
python -m src.train --train-size 0.9 --test-size 0.1
```

To compare the base and fine-tuned model side by side after training:

```bash
python -m src.evaluate generation
```

---

## Output

```
qwen3-fable5-sft/
├── checkpoint-200/        # intermediate checkpoints
├── checkpoint-400/
└── final/                 # LoRA adapter, not merged into the base
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
```

## Separate 4-bit export and inference

Export the original base and trained LoRA weights independently:

```bash
python -m src.quantization --local-files-only
```

The default output is `qwen3-fable5-sft/separate-4bit/`:

- `base-4bit/`: standard Transformers/bitsandbytes NF4 base model and tokenizer.
- `adapter-4bit/`: custom NF4 LoRA weights, complete quantization state,
  `adapter_quantization.json`, PEFT configuration, and training tokenizer.

The adapter is **4-bit on disk and restored to FP32 at runtime** by
`src.module.quantization`. It cannot be loaded directly with
`PeftModel.from_pretrained`; use `src.inference` or `attach_adapter` from that
module. Quantization is lossy and can change model responses. The exporter
reports the weight error and verifies that serialization adds no further change.
The original `final` directory is preserved.
Existing output directories are rejected; use a new `--output-dir` to export again.

```bash
# Default: saved base + restored adapter (no repeated base quantization)
python -m src.inference --prompt "What is 17 * 23?"
python -m src.inference --use-adapter --thinking --max-new-tokens 512

# Same saved base, without reading any adapter files
python -m src.inference --no-use-adapter --prompt "What is 17 * 23?"

# Original floating-point PEFT adapter is still supported
python -m src.inference --adapter-path ./qwen3-fable5-sft/final

# Format validation and round-trip tests; no downloads required
python -m unittest discover -s tests -v
```

For a custom separate output, pass its two subdirectories via
`--base-model` and `--adapter-path`. Inference uses local/cached files only.

## Compare logprobs before and after fine-tuning

Compare **P(next token | full input)** with the adapter disabled/enabled.
Defaults: `qwen3-fable5-sft/separate-4bit/base-4bit` and `adapter-4bit`.
Both passes use identical input tokens, with no generation or sampling.

```bash
# Next-token distribution + per-layer hidden-state differences; save full report
python -m src.evaluate logprobs --text "this is a pen" --hidden-states --output result.json

# Show each model's top 20 candidates (display only; metrics use the full vocabulary)
python -m src.evaluate logprobs --text "this is a pen" --top-k 20

# Read input from a UTF-8 file
python -m src.evaluate logprobs --text-file input.txt --output result.json

# Optional: score the supplied text itself instead of the next-token distribution
python -m src.evaluate logprobs --mode text --text "this is a pen"

# Score a fixed answer to a chat prompt
python -m src.evaluate logprobs --mode text --prompt "What is 17 * 23?" --text "391."
```

Use `--base-model` / `--adapter-path` for custom models and `--device cpu` for CPU.
`--max-tokens` defaults to 4096 including any prompt; longer inputs are rejected.
Run `python -m src.evaluate logprobs --help` for all options.

**Reading the output:**

- **Delta** = adapter enabled minus disabled. Positive logprob delta means a
  token becomes more likely; probability deltas are in percentage points.
- **TV / KL / JS** summarize the full distribution difference; zero means
  identical distributions. These are not accuracy metrics.
- **Hidden states** (`--hidden-states`): each decoder layer reports
  `100 * mean(abs(enabled - disabled)) / mean(abs(disabled))` across all input
  tokens and hidden dimensions. The overall value is the mean across layers.
- **JSON** includes full vocabulary logprob arrays under `distribution`
  (array index = token ID), top candidates, and optional hidden-state statistics.
