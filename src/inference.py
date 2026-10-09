"""Generation smoke test for the NF4 model exported by src.quantization.

Run:  python -m src.inference --prompt "What is 17 * 23?"
      python -m src.inference --thinking --temperature 0.6 --max-new-tokens 512
      python -m src.inference --no-use-adapter

Default: load separate-4bit/base-4bit and separate-4bit/adapter-4bit.
--no-use-adapter loads only the base; --use-adapter explicitly enables LoRA.
"""

import argparse
from pathlib import Path

from src.module.inference import generate
from src.module.utils import DEFAULT_ADAPTER, DEFAULT_BASE


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
    generate(
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
