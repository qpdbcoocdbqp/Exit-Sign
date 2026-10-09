"""Evaluate the fine-tuned adapter against its base.

logprobs   -- compare next-token distributions (or fixed-text scores), adapter disabled vs enabled:
    python -m src.evaluate logprobs --text "this is a pen" --hidden-states --output result.json
    python -m src.evaluate logprobs --mode text --prompt "What is 17 * 23?" --text "391."
generation -- side-by-side generation with the base and merged LoRA model:
    python -m src.evaluate generation --prompt "Think step by step: what is 17 * 23?"
"""

import argparse
import json
from pathlib import Path

from src.module.evaluate import (
    compare_logprobs, compare_models, print_hidden_states_report, print_next_token_report, print_text_report,
)
from src.module.utils import DEFAULT_ADAPTER, DEFAULT_BASE, FINAL_ADAPTER, MODEL_ID, stage_timer


DEFAULT_PROMPTS = [
    "Write a simple Python HTTP server with logging.",
    "Use a tool to search for the latest news about climate change.",
    "Think step by step: what is 17 * 23?",
]


def run_logprobs(parser, args):
    try:
        text = args.text_file.read_text(encoding="utf-8") if args.text_file else args.text
        report = compare_logprobs(
            text, prompt=args.prompt, device=args.device, adapter_path=args.adapter_path,
            base_model=args.base_model, enable_thinking=args.thinking, max_tokens=args.max_tokens,
            include_hidden_states=args.hidden_states,
            mode=args.mode, top_k=args.top_k,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.mode == "next-token":
        print_next_token_report(report)
    else:
        print_text_report(report)
    if "hidden_states" in report:
        print_hidden_states_report(report["hidden_states"])
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"\nSaved: {args.output}")


def run_generation(args):
    with stage_timer("9. Inference comparison"):
        compare_models(
            args.prompt or DEFAULT_PROMPTS, finetuned_path=args.adapter_path,
            max_new_tokens=args.max_new_tokens, base_model_id=args.base_model,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    logprobs = commands.add_parser("logprobs", help="Compare logprobs with the adapter disabled/enabled")
    source = logprobs.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Fixed input context (or text to score with --mode text)")
    source.add_argument("--text-file", type=Path, help="UTF-8 file containing fixed input text")
    logprobs.add_argument("--mode", choices=("next-token", "text"), default="next-token",
                          help="next-token (default): distribution after the full input; text: score supplied tokens")
    logprobs.add_argument("--top-k", type=int, default=10, help="Display union of each model's top-k next tokens; JSON keeps full distributions")
    logprobs.add_argument("--prompt", help="Optional user chat prompt preceding --text as a fixed assistant continuation")
    logprobs.add_argument("--thinking", action="store_true", help="Enable thinking in the chat prefix")
    logprobs.add_argument("--base-model", help=f"Unmerged local/cached base; default: {DEFAULT_BASE}")
    logprobs.add_argument("--adapter-path", type=Path, help=f"PEFT or NF4 adapter; default: {DEFAULT_ADAPTER}")
    logprobs.add_argument("--device", default="cuda:0")
    logprobs.add_argument("--max-tokens", type=int, default=4096, help="Reject longer inputs (including prompt); no truncation")
    logprobs.add_argument("--hidden-states", action="store_true", help="Also compare each decoder layer over all input tokens")
    logprobs.add_argument("--output", type=Path, help="Save full report and per-token scores as UTF-8 JSON")

    generation = commands.add_parser("generation", help="Generate side-by-side with base and merged LoRA model")
    generation.add_argument("--prompt", action="append", help="Repeatable; defaults to three built-in prompts")
    generation.add_argument("--adapter-path", type=Path, default=FINAL_ADAPTER, help="Floating-point PEFT adapter")
    generation.add_argument("--base-model", default=MODEL_ID)
    generation.add_argument("--max-new-tokens", type=int, default=512)

    args = parser.parse_args()
    if args.command == "logprobs":
        run_logprobs(logprobs, args)
    else:
        run_generation(args)


if __name__ == "__main__":
    main()
