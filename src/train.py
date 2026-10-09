"""Fable-5-traces x Qwen3-0.6B -- SFT fine-tuning.

Run:  python -m src.train                                   # dry run: 10% train / 10% eval
      python -m src.train --train-size 0.9 --test-size 0.1  # full training

Saves the LoRA adapter (not merged) to ./qwen3-fable5-sft/final.
"""

import argparse
from pathlib import Path

from src.module.train import MAX_LEN, train
from src.module.utils import DATASET_ID, FINAL_ADAPTER, MODEL_ID, OUTPUT_ROOT, print_profile_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT, help="Checkpoint directory")
    parser.add_argument("--final-dir", type=Path, default=FINAL_ADAPTER, help="Where the final adapter is saved")
    parser.add_argument("--train-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=MAX_LEN)
    args = parser.parse_args()
    train(
        model_id=args.model_id, dataset_id=args.dataset_id,
        output_dir=args.output_dir, final_dir=args.final_dir,
        train_size=args.train_size, test_size=args.test_size, max_length=args.max_length,
    )
    print_profile_summary()


if __name__ == "__main__":
    main()
