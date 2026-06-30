"""
Fable-5-traces x Qwen3-0.6B -- SFT Fine-tuning Example
========================================================

The dataset contains reasoning traces from claude-fable-5, including:
  - context    : conversation history (USER messages)
  - cot        : Chain-of-Thought reasoning process
  - output     : final action (tool_use or text)
  - completion : full output (<think>...</think> + output)

No chosen/rejected pairs exist, so SFT (Supervised Fine-Tuning) is the
most straightforward approach. SFT is the first step of the RLHF pipeline
(Behavior Cloning).

Install dependencies:
    pip install transformers datasets trl peft accelerate bitsandbytes
"""

import json
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType



# ─────────────────────────────────────────────
# Profiling helper
# ─────────────────────────────────────────────
_stage_times = []
 
class stage_timer:
    """A simple context manager that records and prints the elapsed time for each phase."""
    def __init__(self, name: str):
        self.name = name
 
    def __enter__(self):
        self.t0 = time.perf_counter()
        print(f"\n⏱️  [{self.name}] start...")
        return self
 
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.t0
        _stage_times.append((self.name, elapsed))
        print(f"⏱️  [{self.name}] done in {elapsed:.2f}s")
        return False
 
def print_profile_summary():
    print("\n" + "=" * 50)
    print("📊 Stage timing summary")
    print("=" * 50)
    total = sum(t for _, t in _stage_times)
    for name, t in _stage_times:
        pct = (t / total * 100) if total > 0 else 0
        print(f"  {name:<30s} {t:8.2f}s  ({pct:5.1f}%)")
    print(f"  {'TOTAL':<30s} {total:8.2f}s")
    print("=" * 50)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen3-0.6B"
MAX_LEN  = 4096


# ─────────────────────────────────────────────
# 1. Load dataset
# ─────────────────────────────────────────────
with stage_timer("1. Load dataset"):
    print("📦 Loading Fable-5-traces dataset...")
    dataset = load_dataset("Glint-Research/Fable-5-traces", split="train")

    print(f"   Total rows : {len(dataset)}")
    print(f"   Columns    : {dataset.column_names}")

# Split into user / assistant rows and build an id -> row lookup
# Use batched filter dataset;
with stage_timer("2. Filter message/user/assistant rows"):
    message_rows = dataset.filter(
        lambda batch: [t == "message" for t in batch["type"]],
        batched=True,
        num_proc=None,
    )

    user_rows = message_rows.filter(
        lambda batch: [m["role"] == "user" for m in batch["message"]],
        batched=True,
        num_proc=None,
    )
    assistant_ds = message_rows.filter(
        lambda batch: [m["role"] == "assistant" for m in batch["message"]],
        batched=True,
        num_proc=None,
    )

    user_by_id     = {r["id"]: r for r in user_rows}
    assistant_msgs = list(assistant_ds)

print(f"user: {len(user_by_id)}  assistant: {len(assistant_msgs)}")


# ─────────────────────────────────────────────
# 2. Extract text from content blocks
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
# 3. Build pairs in TRL conversational prompt-completion format
#
#    SFTTrainer natively supports:
#      {"prompt": [{"role": "user", "content": "..."}],
#       "completion": [{"role": "assistant", "content": "..."}]}
#
#    Combined with SFTConfig(assistant_only_loss=True) this automatically:
#      - applies the chat template (tokenisation)
#      - masks prompt tokens so loss is computed on completion only
# ─────────────────────────────────────────────
def build_pairs_batch(batch):
    """Batched version: Processes a batch of assistant rows at a time to reduce Python function call overhead."""
    out_prompts, out_completions = [], []
    for parent_id, content in zip(batch["parentId"], batch["message"]):
        u = user_by_id.get(parent_id)
        if u is None:
            continue
        user_text = extract_user_text(u["message"]["content"])
        asst_text = extract_assistant_text(content["content"])
        if not user_text or not asst_text:
            continue
        out_prompts.append([{"role": "user", "content": user_text}])
        out_completions.append([{"role": "assistant", "content": asst_text}])
    return {"prompt": out_prompts, "completion": out_completions}

# batched=True reduces per-row overhead; num_proc=None keeps it single-process to avoid Windows process spawning issues.
with stage_timer("3. Build prompt/completion pairs"):
    hf_dataset = assistant_ds.map(
        build_pairs_batch,
        batched=True,
        num_proc=None,
        remove_columns=assistant_ds.column_names,
    )

    skipped = len(assistant_msgs) - len(hf_dataset)

print(f"Valid pairs: {len(hf_dataset)}  Skipped: {skipped}")
print("\n=== First example ===")
print("PROMPT:", hf_dataset[0]["prompt"][0]["content"][:200])
print("\nCOMPLETION:", hf_dataset[0]["completion"][0]["content"][:300])


# Train / eval split
# Dry-run fractions; change to train_size=0.9, test_size=0.1 for full training
with stage_timer("4. Train/test split"):
    split = hf_dataset.train_test_split(train_size=0.1, test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

print(f"\nTrain: {len(train_ds)}  |  Eval: {len(eval_ds)}")


# ─────────────────────────────────────────────
# 4. Load model with 4-bit quantisation (~2 GB VRAM)
#    Tokenizer is loaded automatically by SFTTrainer from MODEL_ID
# ─────────────────────────────────────────────
print(f"\n🤖 Loading model: {MODEL_ID}")

with stage_timer("5. Load model (4-bit + sdpa)"):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False


# ─────────────────────────────────────────────
# 5. LoRA config (Parameter-Efficient Fine-Tuning)
#    Passed directly to SFTTrainer; no manual get_peft_model() needed
# ─────────────────────────────────────────────
lora_config = LoraConfig(
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


# ─────────────────────────────────────────────
# 6. Training config
# ─────────────────────────────────────────────
sft_config = SFTConfig(
    output_dir="./qwen3-fable5-sft",

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
    max_length=MAX_LEN,

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
# 7. Build SFTTrainer and train
#
#    SFTTrainer automatically:
#      - loads the tokenizer from MODEL_ID
#      - applies the Qwen3 chat template (with assistant_only_loss markers)
#      - wraps the model with LoRA via peft_config
# ─────────────────────────────────────────────
with stage_timer("6. Build SFTTrainer"):
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
    )

with stage_timer("7. Train"):
    print("\n🚀 Starting training...")
    trainer.train()

with stage_timer("8. Save model"):
    print("\n💾 Saving model...")
    trainer.save_model("./qwen3-fable5-sft/final")
    print("✅ Done! Model saved to ./qwen3-fable5-sft/final")


# ─────────────────────────────────────────────
# 8. Inference test
# ─────────────────────────────────────────────
def inference_test(prompt: str, max_new_tokens: int = 512):
    """Run inference with the fine-tuned model."""
    from transformers import AutoTokenizer, pipeline, GenerationConfig
    from peft import PeftModel

    print("\n🔍 Running inference test...")
    base   = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")
    merged = PeftModel.from_pretrained(base, "./qwen3-fable5-sft/final").merge_and_unload()

    tok = AutoTokenizer.from_pretrained(
        "./qwen3-fable5-sft/final",
        clean_up_tokenization_spaces=False,  # BPE tokenizers should not apply this post-processing
    )

    # Consolidate generation parameters into GenerationConfig to avoid conflicts
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tok.eos_token_id,
    )

    pipe = pipeline("text-generation", model=merged, tokenizer=tok, generation_config=gen_config)

    messages = [{"role": "user", "content": prompt}]
    response = pipe(messages)
    print(f"Prompt: {prompt}\n\nResponse:\n{response[0]['generated_text'][-1]['content']}")
    return response


if __name__ == "__main__":
    with stage_timer("9. Inference test"):
        inference_test("Write a simple Python HTTP server with logging.")
