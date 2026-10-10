"""SFT fine-tuning of Qwen3 on Fable-5-traces with LoRA (no work at import time).

The dataset contains reasoning traces from claude-fable-5, including:
  - context    : conversation history (USER messages)
  - cot        : Chain-of-Thought reasoning process
  - output     : final action (tool_use or text)
  - completion : full output (<think>...</think> + output)

No chosen/rejected pairs exist, so SFT (Supervised Fine-Tuning) is the
most straightforward approach. SFT is the first step of the RLHF pipeline
(Behavior Cloning).
"""

import json
from pathlib import Path

from .utils import DATASET_ID, FINAL_ADAPTER, MODEL_ID, OUTPUT_ROOT, load_env, stage_timer, to_local_path


MAX_LEN = 4096


# ─────────────────────────────────────────────
# Extract text from content blocks
# ─────────────────────────────────────────────
def extract_user_text(content: list) -> str:
    """content is a list of {type, text, ...} blocks."""
    return "\n".join(
        block["text"] for block in content if block.get("type") == "text"
    ).strip()


def extract_assistant_text(content: list) -> str:
    """
    content is a list that may contain:
      - {type: thinking, thinking: "..."}   <- CoT reasoning
      - {type: toolCall, name, arguments}   <- tool invocation
      - {type: text, text: "..."}           <- plain text response
    All blocks are serialised into a single string for the model to learn.
    """
    parts = []
    for block in content:
        t = block.get("type")
        if t == "thinking":
            parts.append(f"<think>\n{block['thinking']}\n</think>")
        elif t == "text":
            parts.append(block["text"])
        elif t == "toolCall":
            args = json.dumps(block.get("arguments", {}), ensure_ascii=False)
            parts.append(f"<tool_call>\n{block['name']}({args})\n</tool_call>")
    return "\n".join(parts).strip()


# ─────────────────────────────────────────────
# Build pairs in TRL conversational prompt-completion format
#
#    SFTTrainer natively supports:
#      {"prompt": [{"role": "user", "content": "..."}],
#       "completion": [{"role": "assistant", "content": "..."}]}
#
#    Combined with SFTConfig(assistant_only_loss=True) this automatically:
#      - applies the chat template (tokenisation)
#      - masks prompt tokens so loss is computed on completion only
# ─────────────────────────────────────────────
def resolve_trace_dir(data_dir: str | Path | None = None, dataset_id: str = DATASET_ID) -> Path:
    """Locate the directory holding the Fable-5-traces *.jsonl session files.

    Order: explicit data_dir, then the DATASET_PATH variable (environment or the
    project's .env), then a Hugging Face download/cache lookup. Accepts the
    traces folder itself, a snapshot folder, or the hub cache folder
    (datasets--<org>--<name>); the snapshot that refs/main points to wins.
    """
    given = data_dir or load_env().get("DATASET_PATH")
    if given:
        root = to_local_path(given)
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset path does not exist: {root}")
        found = _find_trace_dir(root)
        if found is None:
            raise FileNotFoundError(f"No *.jsonl trace files found under {root}")
        return found
    from huggingface_hub import snapshot_download

    print(f"📦 Locating {dataset_id} in the Hugging Face cache (downloads if missing)...")
    snapshot = Path(snapshot_download(dataset_id, repo_type="dataset", allow_patterns=["pi-traces/*.jsonl"]))
    found = _find_trace_dir(snapshot)
    if found is None:
        raise FileNotFoundError(f"No *.jsonl trace files found in {snapshot}")
    return found


def _find_trace_dir(root: Path) -> Path | None:
    snapshots = root / "snapshots"
    if snapshots.is_dir():
        ref = root / "refs" / "main"
        preferred = [snapshots / ref.read_text(encoding="utf-8").strip()] if ref.is_file() else []
        for snapshot in preferred + sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            found = _find_trace_dir(snapshot)
            if found is not None:
                return found
        return None
    for candidate in (root / "pi-traces", root):
        if any(candidate.glob("*.jsonl")):
            return candidate
    return None


def read_pairs(trace_dir: str | Path):
    """Pair every assistant message with its parent user message (same session file).

    Reads the raw trace files directly: one JSON object per line, with
    type == "message", an id/parentId chain, and message.role/content blocks.
    Returns (prompts, completions, assistant_count).
    """
    prompts, completions, assistants = [], [], 0
    for path in sorted(Path(trace_dir).glob("*.jsonl")):
        messages = {}
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("type") == "message":
                    messages[row["id"]] = row
        for row in messages.values():
            if row["message"]["role"] != "assistant":
                continue
            assistants += 1
            parent = messages.get(row.get("parentId"))
            if parent is None or parent["message"]["role"] != "user":
                continue
            user_text = extract_user_text(parent["message"]["content"])
            asst_text = extract_assistant_text(row["message"]["content"])
            if not user_text or not asst_text:
                continue
            prompts.append([{"role": "user", "content": user_text}])
            completions.append([{"role": "assistant", "content": asst_text}])
    return prompts, completions, assistants


def load_pairs(data_dir: str | Path | None = None, dataset_id: str = DATASET_ID):
    """Load the trace files and return a Dataset of prompt/completion pairs."""
    from datasets import Dataset

    with stage_timer("1. Locate trace files"):
        trace_dir = resolve_trace_dir(data_dir, dataset_id)
        print(f"   Trace dir  : {trace_dir}")

    with stage_timer("2. Build prompt/completion pairs"):
        prompts, completions, assistants = read_pairs(trace_dir)
        if not prompts:
            raise ValueError(f"No valid prompt/completion pairs found in {trace_dir}")
        pairs = Dataset.from_dict({"prompt": prompts, "completion": completions})
    print(f"Valid pairs: {len(pairs)}  Skipped: {assistants - len(pairs)}")
    print("\n=== First example ===")
    print("PROMPT:", pairs[0]["prompt"][0]["content"][:200])
    print("\nCOMPLETION:", pairs[0]["completion"][0]["content"][:300])
    return pairs


def split_pairs(pairs, train_size: float = 0.1, test_size: float = 0.1, seed: int = 42):
    """Dry-run fractions by default; use train_size=0.9, test_size=0.1 for full training."""
    with stage_timer("3. Train/test split"):
        split = pairs.train_test_split(train_size=train_size, test_size=test_size, seed=seed)
    print(f"\nTrain: {len(split['train'])}  |  Eval: {len(split['test'])}")
    return split["train"], split["test"]


# ─────────────────────────────────────────────
# Model with 4-bit quantisation (~2 GB VRAM)
#    Tokenizer is loaded automatically by SFTTrainer from the model ID
# ─────────────────────────────────────────────
def load_base_model(model_id: str = MODEL_ID):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    print(f"\n🤖 Loading model: {model_id}")
    with stage_timer("4. Load model (4-bit + sdpa)"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        model.config.use_cache = False
    return model


# ─────────────────────────────────────────────
# LoRA config (Parameter-Efficient Fine-Tuning)
#    Passed directly to SFTTrainer; no manual get_peft_model() needed
# ─────────────────────────────────────────────
def make_lora_config():
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj",
            "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def make_sft_config(output_dir: str | Path = OUTPUT_ROOT, max_length: int = MAX_LEN):
    from trl import SFTConfig

    return SFTConfig(
        output_dir=str(output_dir),

        # Hyperparameters
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,   # effective batch size = 16
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        weight_decay=0.01,

        # Sequence length (Fable-5 completions can reach ~73k tokens; truncate)
        max_length=max_length,

        # Compute loss on assistant completion only;
        # TRL auto-patches the Qwen3 chat template with generation markers
        assistant_only_loss=True,
        packing=False,                   # keep conversations intact

        # Evaluation & checkpointing
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,

        # Performance
        bf16=True,
        tf32=True,
        dataloader_num_workers=4,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        optim="paged_adamw_8bit",        # 8-bit AdamW to save VRAM

        # Logging
        logging_steps=20,
        report_to="none",                # set to "wandb" to enable W&B logging
    )


# ─────────────────────────────────────────────
# Build SFTTrainer and train
#
#    SFTTrainer automatically:
#      - loads the tokenizer from the model ID
#      - applies the Qwen3 chat template (with assistant_only_loss markers)
#      - wraps the model with LoRA via peft_config
# ─────────────────────────────────────────────
def train(
    model_id: str = MODEL_ID,
    dataset_id: str = DATASET_ID,
    data_dir: str | Path | None = None,
    output_dir: str | Path = OUTPUT_ROOT,
    final_dir: str | Path = FINAL_ADAPTER,
    train_size: float = 0.1,
    test_size: float = 0.1,
    max_length: int = MAX_LEN,
):
    """Run the full SFT pipeline and save the LoRA adapter to final_dir."""
    from trl import SFTTrainer

    train_ds, eval_ds = split_pairs(load_pairs(data_dir, dataset_id), train_size=train_size, test_size=test_size)
    model = load_base_model(model_id)

    with stage_timer("5. Build SFTTrainer"):
        trainer = SFTTrainer(
            model=model,
            args=make_sft_config(output_dir, max_length=max_length),
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            peft_config=make_lora_config(),
        )

    with stage_timer("6. Train"):
        print("\n🚀 Starting training...")
        trainer.train()

    with stage_timer("7. Save model"):
        print("\n💾 Saving model...")
        trainer.save_model(str(final_dir))
        print(f"✅ Done! Model saved to {final_dir}")
    return Path(final_dir)
