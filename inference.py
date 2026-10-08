"""Generation smoke test for the NF4 model exported by model_quantization.py.

Install: pip install torch transformers peft accelerate bitsandbytes
Run:     python inference.py --prompt "What is 17 * 23?"
         python inference.py --thinking --temperature 0.6 --max-new-tokens 512
         python inference.py --use-adapter
         python inference.py --no-use-adapter

Default: load separate-4bit/base-4bit and separate-4bit/adapter-4bit.
--no-use-adapter loads only the base; --use-adapter explicitly enables LoRA.
NF4 adapters are 4-bit on disk and restored to FP32 at runtime. Original PEFT
adapters are also supported.
All model/tokenizer files are loaded locally or from the Hugging Face cache.
"""

import argparse
import math
from pathlib import Path
import time


DEFAULT_ROOT = Path(__file__).resolve().parent / "qwen3-fable5-sft" / "separate-4bit"
DEFAULT_ADAPTER = DEFAULT_ROOT / "adapter-4bit"
DEFAULT_BASE = DEFAULT_ROOT / "base-4bit"


def load_inference_model(
    device: str = "cuda:0",
    use_adapter: bool = True,
    adapter_path: str | Path | None = None,
    base_model: str | Path | None = None,
):
    """Load the shared local 4-bit base, optional adapter, and tokenizer."""
    if base_model is None:
        base_model = DEFAULT_BASE
        if not (base_model / "config.json").is_file():
            raise FileNotFoundError(f"Missing saved base model in {base_model}; run model_quantization.py first.")
    if use_adapter:
        adapter_path = Path(adapter_path or DEFAULT_ADAPTER).expanduser().resolve()
        if not (adapter_path / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Missing adapter_config.json in {adapter_path}")

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    target_device = torch.device(device)
    if target_device.type not in ("cuda", "cpu"):
        raise ValueError("device must be cpu or a CUDA device such as cuda:0.")
    if target_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Install CUDA-enabled PyTorch or use --device cpu.")
        torch.cuda.set_device(target_device)

    load_kwargs = {"device_map": {"": str(target_device)}, "local_files_only": True}
    model_source = tokenizer_source = base_model
    if use_adapter and (adapter_path / "tokenizer.json").is_file():
        tokenizer_source = adapter_path
    print(f"Mode: 4-bit base; adapter {'enabled' if use_adapter else 'disabled'}.", flush=True)

    config = AutoConfig.from_pretrained(model_source, local_files_only=True)
    quantization = getattr(config, "quantization_config", {}) or {}
    if getattr(config, "exit_sign_export_mode", None) == "merged":
        raise ValueError("--base-model must be an unmerged base model, such as separate-4bit/base-4bit.")
    if quantization:
        if quantization.get("quant_method") != "bitsandbytes" or not quantization.get("load_in_4bit"):
            raise ValueError("Expected a bitsandbytes 4-bit model.")
        # Use the stored quantization settings, without quantizing a second time.
    else:
        # Preserve support for an explicitly selected cached, unquantized base.
        compute_dtype = torch.float32
        if target_device.type == "cuda":
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        load_kwargs.update(
            dtype=compute_dtype,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            ),
        )

    print(f"Loading 4-bit model: {model_source}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, local_files_only=True, clean_up_tokenization_spaces=False,
    )
    # Saved bases use their stored settings; unquantized bases are quantized on load.
    model = AutoModelForCausalLM.from_pretrained(model_source, **load_kwargs)
    if not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("The model did not load in 4-bit mode.")
    if use_adapter:
        from model_quantization import attach_adapter

        model = attach_adapter(model, adapter_path, target_device)
    model.eval()
    print(f"Model loaded on {target_device}; footprint: {model.get_memory_footprint() / 1024**2:.1f} MiB")
    return model, tokenizer, target_device


def inference_test(
    prompt: str,
    max_new_tokens: int = 256,
    device: str = "cuda:0",
    temperature: float = 0.0,
    enable_thinking: bool = False,
    use_adapter: bool = True,
    adapter_path: str | Path | None = None,
    base_model: str | Path | None = None,
) -> str:
    """Generate with a saved 4-bit base and optional separate LoRA adapter."""
    if not prompt.strip():
        raise ValueError("Prompt must not be empty.")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive.")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative (0 = greedy decoding).")

    import torch
    from transformers import GenerationConfig

    model, tokenizer, target_device = load_inference_model(
        device=device, use_adapter=use_adapter, adapter_path=adapter_path, base_model=base_model,
    )

    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer(chat, add_special_tokens=False, return_tensors="pt").to(target_device)
    prompt_length = inputs["input_ids"].shape[1]
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = model.generation_config.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    # Preserve Qwen's multiple EOS IDs, without inheriting sampling-only defaults
    # when running the deterministic smoke test (temperature=0).
    sampling = {"temperature": temperature, "top_p": 0.95, "top_k": 20} if temperature > 0 else {}
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=pad_token_id,
        use_cache=True,
        **sampling,
    )
    # Replace model defaults as well: Transformers 5 fills unset values in a
    # passed GenerationConfig from model.generation_config.
    model.generation_config = generation_config
    print(f"\nPrompt: {prompt}\nGenerating...", flush=True)
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs)
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    elapsed = time.perf_counter() - started
    new_tokens = output[0, prompt_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"\nResponse:\n{response}")
    print(f"\nGenerated {new_tokens.numel()} tokens in {elapsed:.2f}s.")
    if new_tokens.numel() == max_new_tokens:
        print("Token limit reached; increase --max-new-tokens if the response is incomplete.")
    return response


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--use-adapter", action=argparse.BooleanOptionalAction, default=True,
        help="Enable/disable LoRA on the saved 4-bit base (default: enabled)",
    )
    parser.add_argument("--adapter-path", type=Path, help=f"Default: {DEFAULT_ADAPTER}")
    parser.add_argument("--base-model", help=f"Saved base path or cached model ID; default: {DEFAULT_BASE}")
    parser.add_argument("--prompt", default="What is 17 * 23? Answer briefly.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda:0", help="cuda:0 (default) or cpu")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy; positive = sampling")
    parser.add_argument("--thinking", action="store_true", help="Enable Qwen3's thinking mode")
    args = parser.parse_args()
    inference_test(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        temperature=args.temperature,
        enable_thinking=args.thinking,
        use_adapter=args.use_adapter,
        adapter_path=args.adapter_path,
        base_model=args.base_model,
    )


if __name__ == "__main__":
    main()
