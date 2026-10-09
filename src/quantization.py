"""Export the Fable-5 base and LoRA adapter separately in 4-bit NF4.

Run:  python -m src.quantization --local-files-only
Test: python -m src.quantization --prompt "What is 17 * 23?"

Input defaults to ./qwen3-fable5-sft/final.
Default output: ./qwen3-fable5-sft/separate-4bit/{base-4bit,adapter-4bit}.
Existing output directories are rejected; use a new --output-dir to export again.
"""

import argparse
from pathlib import Path

from src.module.quantization import quantize_model
from src.module.utils import FINAL_ADAPTER, SEPARATE_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter-path", type=Path, default=FINAL_ADAPTER)
    parser.add_argument("--output-dir", type=Path, default=SEPARATE_ROOT,
                        help="Output directory containing base-4bit and adapter-4bit.")
    parser.add_argument("--base-model", help="Override base_model_name_or_path from adapter_config.json.")
    parser.add_argument("--device", default="cuda:0", help="Quantization/inference device: cuda:0 or cpu.")
    parser.add_argument("--local-files-only", action="store_true", help="Use cached/local files only.")
    parser.add_argument("--prompt", help="Optional chat prompt to test the saved, reloaded model.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    quantize_model(
        adapter_path=args.adapter_path, output_dir=args.output_dir, base_model=args.base_model,
        device=args.device, local_files_only=args.local_files_only,
        prompt=args.prompt, max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
