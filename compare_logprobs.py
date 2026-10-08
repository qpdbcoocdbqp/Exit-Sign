"""Compare next-token distributions with the same input, with/without LoRA.

python -m compare_logprobs --text "this is a pen" --hidden-states
python -m compare_logprobs --mode text --prompt "What is 17 * 23?" --text "391."
"""

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path

from inference import DEFAULT_ADAPTER, DEFAULT_BASE, load_inference_model


def prepare_tokens(tokenizer, text, prompt=None, enable_thinking=False, mode="text"):
    """Encode once; return token IDs and the first position to score.

    Raw text has no implicit BOS/EOS; its first token supplies context. In chat
    mode the rendered generation prefix and fixed continuation are encoded
    separately, matching the token boundary used during generation.
    """
    if not text or not text.strip():
        raise ValueError("Text must not be empty.")
    target = tokenizer.encode(text, add_special_tokens=False)
    if not target:
        raise ValueError("Text must contain at least one token.")
    if prompt is None:
        if mode == "text" and len(target) < 2:
            raise ValueError("Raw text needs at least two tokens; use --prompt to score a single token.")
        return target, 1
    if not prompt.strip():
        raise ValueError("Prompt must not be empty.")
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=enable_thinking,
    )
    context = tokenizer.encode(prefix, add_special_tokens=False)
    if not context:
        raise ValueError("Chat template produced no context tokens.")
    return context + target, len(context)


@contextmanager
def capture_layer_states(model, destination):
    """Capture decoder block outputs before the final model normalization."""
    if destination is None:
        yield
        return
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    layers = getattr(base.get_decoder(), "layers", None)
    if layers is None or not len(layers):
        raise ValueError("Hidden-state comparison requires a decoder with a nonempty layers list.")
    handles = []

    def capture(index):
        def hook(module, args, output):
            state = output[0] if isinstance(output, tuple) else output
            # Keep an independent CPU copy; do not retain all layers on the GPU.
            destination[index] = state.detach().to(device="cpu", copy=True)
        return hook

    try:
        for index, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(capture(index)))
        yield
        if set(destination) != set(range(len(layers))):
            raise ValueError("Not all decoder layers produced hidden states.")
    finally:
        for handle in handles:
            handle.remove()


def summarize_hidden_states(before, after):
    """Normalized MAE per layer over all input positions and hidden dimensions."""
    import torch

    if not before or set(before) != set(after):
        raise ValueError("Hidden states must cover the same nonempty set of layers.")
    rows = []
    for index in sorted(before):
        base, tuned = before[index].float(), after[index].float()
        if base.shape != tuned.shape or base.ndim != 3 or base.numel() == 0:
            raise ValueError("Hidden states must have matching nonempty [batch, tokens, hidden] shapes.")
        if not torch.isfinite(base).all() or not torch.isfinite(tuned).all():
            raise ValueError("Model produced non-finite hidden states.")
        magnitude = base.abs().mean(dtype=torch.float64).item()
        difference = (tuned - base).abs().mean(dtype=torch.float64).item()
        # A zero baseline with a nonzero change has no finite relative percentage.
        percent = 100.0 * difference / magnitude if magnitude else (0.0 if difference == 0 else None)
        rows.append({
            "layer": index + 1, "module_index": index,
            "token_count": base.shape[0] * base.shape[1], "hidden_size": base.shape[2],
            "base_mean_absolute_activation": magnitude,
            "mean_absolute_difference": difference, "relative_mae_percent": percent,
        })
    percentages = [row["relative_mae_percent"] for row in rows]
    return {
        "metric": "relative_mae_percent",
        "formula": "100 * mean(abs(finetuned - base)) / mean(abs(base))",
        "capture_point": "decoder block output after residual additions, before final model norm",
        "token_scope": "all input tokens, including the first token and any chat prompt/prefix",
        "aggregation": "arithmetic mean of layer percentages; embeddings and final model norm excluded",
        "zero_baseline_policy": "both zero => 0%; nonzero difference => null (also null overall mean)",
        "layer_count": len(rows),
        "mean_relative_mae_percent": (
            math.fsum(percentages) / len(percentages) if None not in percentages else None
        ),
        "layers": rows,
    }


def score_tokens(model, token_ids, score_start, device, hidden_states=None):
    """Return log P(x[t] | x[:t]) in nats, with correct causal alignment."""
    import torch
    import torch.nn.functional as F

    if not 1 <= score_start < len(token_ids):
        raise ValueError("score_start must select tokens with preceding context.")
    inputs = torch.tensor([token_ids], dtype=torch.long, device=device)
    model.eval()
    with torch.inference_mode(), capture_layer_states(model, hidden_states):
        logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).logits
        values = []
        # Limit the temporary FP32 vocabulary matrix for long inputs.
        for start in range(score_start, len(token_ids), 128):
            end = min(start + 128, len(token_ids))
            loss = F.cross_entropy(
                logits[0, start - 1:end - 1].float(), inputs[0, start:end], reduction="none",
            )
            values.extend((-loss).double().cpu().tolist())
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Model produced non-finite logprobs.")
    return values


def next_token_logprobs(model, token_ids, device, hidden_states=None):
    """Log P(v | entire input) for every vocabulary ID, without sampling filters."""
    import torch

    if not token_ids:
        raise ValueError("Next-token comparison needs at least one input token.")
    inputs = torch.tensor([token_ids], dtype=torch.long, device=device)
    model.eval()
    with torch.inference_mode(), capture_layer_states(model, hidden_states):
        output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False)
        logits = output.logits[0, -1].detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(logits).all():
            raise ValueError("Model produced non-finite next-token logits.")
        return logits.log_softmax(dim=-1)


def make_next_token_report(tokenizer, token_ids, before, after, top_k=10):
    """Full-vocabulary metrics plus a union of each model's top-k candidates."""
    import torch

    if top_k < 1:
        raise ValueError("top_k must be positive.")
    before, after = before.detach().cpu().double(), after.detach().cpu().double()
    if before.ndim != 1 or before.numel() == 0 or before.shape != after.shape:
        raise ValueError("Next-token distributions must have the same nonempty vocabulary.")
    if not torch.isfinite(before).all() or not torch.isfinite(after).all():
        raise ValueError("Next-token logprobs must be finite.")
    if abs(before.logsumexp(0).item()) > 1e-6 or abs(after.logsumexp(0).item()) > 1e-6:
        raise ValueError("Next-token logprobs must be normalized over the full vocabulary.")
    p, q = before.exp(), after.exp()
    log_midpoint = torch.logaddexp(before, after) - math.log(2.)
    count = min(top_k, before.numel())
    base_top = torch.argsort(before, descending=True, stable=True)[:count].tolist()
    adapter_top = torch.argsort(after, descending=True, stable=True)[:count].tolist()
    candidates = sorted(set(base_top) | set(adapter_top), key=lambda i: (-max(p[i].item(), q[i].item()), i))
    rows = [{
        "token_id": i, "token": tokenizer.convert_ids_to_tokens(i),
        "base_logprob": before[i].item(), "adapter_logprob": after[i].item(),
        "delta_logprob": (after[i] - before[i]).item(),
        "base_probability": p[i].item(), "adapter_probability": q[i].item(),
        "delta_probability_percentage_points": 100. * (q[i] - p[i]).item(),
    } for i in candidates]
    return {
        "log_base": "e", "delta_definition": "adapter_enabled - adapter_disabled",
        "input_token_ids": token_ids, "prediction_position": len(token_ids),
        "summary": {
            "input_tokens": len(token_ids), "vocab_size": before.numel(),
            "kl_base_to_adapter_nats": max(0., (p * (before - after)).sum().item()),
            "kl_adapter_to_base_nats": max(0., (q * (after - before)).sum().item()),
            "js_divergence_nats": max(0., (.5 * (p * (before - log_midpoint)).sum()
                                         + .5 * (q * (after - log_midpoint)).sum()).item()),
            "total_variation_percent": 50. * (p - q).abs().sum().item(),
            "base_entropy_nats": -(p * before).sum().item(),
            "adapter_entropy_nats": -(q * after).sum().item(),
            "base_top1_token_id": base_top[0], "adapter_top1_token_id": adapter_top[0],
            "top1_changed": base_top[0] != adapter_top[0],
        },
        "top_k": count, "base_top_token_ids": base_top, "adapter_top_token_ids": adapter_top,
        "tokens": rows,
        "distribution": {
            "index_definition": "array index is token_id; includes every model vocabulary ID, including special/unmapped IDs",
            "base_logprobs": before.tolist(), "adapter_logprobs": after.tolist(),
            "delta_logprobs": (after - before).tolist(),
        },
    }


def make_report(tokenizer, token_ids, score_start, before, after):
    count = len(token_ids) - score_start
    if count < 1 or len(before) != count or len(after) != count:
        raise ValueError("Both scores must cover the same nonempty token sequence.")
    rows = [
        {"position": position, "token_id": token_id,
         "token": tokenizer.convert_ids_to_tokens(token_id),
         "base_logprob": base, "finetuned_logprob": tuned, "delta_logprob": tuned - base}
        for position, token_id, base, tuned in zip(
            range(score_start, len(token_ids)), token_ids[score_start:], before, after,
        )
    ]
    base_sum, tuned_sum = math.fsum(before), math.fsum(after)
    return {
        "log_base": "e", "delta_definition": "finetuned - base",
        "input_token_ids": token_ids, "score_start": score_start,
        "summary": {
            "scored_tokens": count,
            "base_total_logprob": base_sum, "finetuned_total_logprob": tuned_sum,
            "delta_total_logprob": tuned_sum - base_sum,
            "base_mean_logprob": base_sum / count, "finetuned_mean_logprob": tuned_sum / count,
            "delta_mean_logprob": (tuned_sum - base_sum) / count,
        },
        "tokens": rows,
    }


def compare_logprobs(text, prompt=None, device="cuda:0", adapter_path=None,
                     base_model=None, enable_thinking=False, max_tokens=4096,
                     include_hidden_states=False, mode="next-token", top_k=10):
    if mode not in ("next-token", "text"):
        raise ValueError("mode must be next-token or text.")
    if max_tokens < (2 if mode == "text" else 1):
        raise ValueError("max_tokens must be positive (at least two in text mode).")
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    if not text or not text.strip():
        raise ValueError("Text must not be empty.")
    if prompt is not None and not prompt.strip():
        raise ValueError("Prompt must not be empty.")
    if enable_thinking and prompt is None:
        raise ValueError("--thinking requires --prompt.")
    model, tokenizer, target_device = load_inference_model(
        device=device, use_adapter=True, adapter_path=adapter_path, base_model=base_model,
    )
    # PEFT cannot restore original base biases when disabling a bias-trained LoRA.
    for config in model.peft_config.values():
        if config.peft_type != "LORA" or config.bias != "none":
            raise ValueError("Comparison requires a LoRA adapter with bias='none'.")
    token_ids, score_start = prepare_tokens(tokenizer, text, prompt, enable_thinking, mode=mode)
    context_limit = getattr(model.config, "max_position_embeddings", max_tokens)
    limit = min(max_tokens, context_limit or max_tokens)
    if len(token_ids) > limit:
        raise ValueError(f"Input has {len(token_ids)} tokens, exceeding limit {limit}; no truncation was applied.")
    print(f"Comparing next-token distributions after all {len(token_ids)} input tokens." if mode == "next-token"
          else f"Scoring {len(token_ids) - score_start} fixed tokens (natural-log units).", flush=True)
    base_states = {} if include_hidden_states else None
    tuned_states = {} if include_hidden_states else None
    def score(states):
        if mode == "next-token":
            return next_token_logprobs(model, token_ids, target_device, hidden_states=states)
        return score_tokens(model, token_ids, score_start, target_device, hidden_states=states)

    print("Forward pass: adapter disabled (base).", flush=True)
    with model.disable_adapter():
        before = score(base_states)
    print("Forward pass: adapter enabled.", flush=True)
    after = score(tuned_states)
    report = (make_next_token_report(tokenizer, token_ids, before, after, top_k=top_k)
              if mode == "next-token" else make_report(tokenizer, token_ids, score_start, before, after))
    if include_hidden_states:
        report["hidden_states"] = summarize_hidden_states(base_states, tuned_states)
    report.update(
        text=text, prompt=prompt, mode="raw_text" if prompt is None else "chat_completion",
        comparison=mode,
        enable_thinking=enable_thinking, base_model=str(base_model or DEFAULT_BASE),
        adapter_path=str(adapter_path or DEFAULT_ADAPTER),
        adapter_format="nf4" if (Path(adapter_path or DEFAULT_ADAPTER) / "adapter_quantization.json").is_file() else "peft",
        base_quantization="4bit", tokenizer_source=str(getattr(tokenizer, "name_or_path", "")),
    )
    return report


def print_next_token_report(report):
    summary = report["summary"]
    print(f"\nNext token after {summary['input_tokens']} input tokens; full vocabulary: {summary['vocab_size']}")
    print("Delta = adapter enabled - disabled; raw logits, no sampling filters.")
    print(f"Total variation: {summary['total_variation_percent']:.6f}%")
    print(f"KL(base || adapter): {summary['kl_base_to_adapter_nats']:.6f} nats")
    print(f"KL(adapter || base): {summary['kl_adapter_to_base_nats']:.6f} nats")
    print(f"JS divergence: {summary['js_divergence_nats']:.6f} nats")
    print(f"Top-1 token ID: {summary['base_top1_token_id']} -> {summary['adapter_top1_token_id']}")
    print(f"\nUnion of top {report['top_k']} from each distribution (probabilities normalized over full vocabulary):")
    print(" Token ID     Base %  Adapter %   Delta pp    Base logp Adapter logp   Delta logp  Token")
    for row in report["tokens"]:
        print(f"{row['token_id']:>9} {row['base_probability'] * 100:>10.5f} "
              f"{row['adapter_probability'] * 100:>10.5f} {row['delta_probability_percentage_points']:>+10.5f} "
              f"{row['base_logprob']:>12.6f} {row['adapter_logprob']:>12.6f} {row['delta_logprob']:>+12.6f}  "
              f"{json.dumps(row['token'], ensure_ascii=True)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Fixed input context (or text to score with --mode text)")
    source.add_argument("--text-file", type=Path, help="UTF-8 file containing fixed input text")
    parser.add_argument("--mode", choices=("next-token", "text"), default="next-token",
                        help="next-token (default): distribution after the full input; text: score supplied tokens")
    parser.add_argument("--top-k", type=int, default=10, help="Display union of each model's top-k next tokens; JSON keeps full distributions")
    parser.add_argument("--prompt", help="Optional user chat prompt preceding --text as a fixed assistant continuation")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking in the chat prefix")
    parser.add_argument("--base-model", help=f"Unmerged local/cached base; default: {DEFAULT_BASE}")
    parser.add_argument("--adapter-path", type=Path, help=f"PEFT or NF4 adapter; default: {DEFAULT_ADAPTER}")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Reject longer inputs (including prompt); no truncation")
    parser.add_argument("--hidden-states", action="store_true", help="Also compare each decoder layer over all input tokens")
    parser.add_argument("--output", type=Path, help="Save full report and per-token scores as UTF-8 JSON")
    args = parser.parse_args()
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
    summary = report["summary"]
    if args.mode == "next-token":
        print_next_token_report(report)
    else:
        print(f"\nScored tokens: {summary['scored_tokens']} | delta = fine-tuned - base (nats)")
        print(f"{'Metric':<18} {'Base':>14} {'Fine-tuned':>14} {'Delta':>14}")
        for metric in ("total", "mean"):
            print(f"{metric + ' logprob':<18} {summary[f'base_{metric}_logprob']:>14.6f} "
                  f"{summary[f'finetuned_{metric}_logprob']:>14.6f} {summary[f'delta_{metric}_logprob']:>+14.6f}")
        print("\nPosition   Token ID         Base   Fine-tuned        Delta  Token")
        for row in report["tokens"]:
            print(f"{row['position']:>8} {row['token_id']:>10} {row['base_logprob']:>12.6f} "
                  f"{row['finetuned_logprob']:>12.6f} {row['delta_logprob']:>+12.6f}  "
                  f"{json.dumps(row['token'], ensure_ascii=True)}")
    if "hidden_states" in report:
        hidden = report["hidden_states"]
        print("\nHidden states: 100 * mean(abs(fine-tuned - base)) / mean(abs(base))")
        print("All input tokens; decoder block outputs before final model norm; layers numbered from 1.")
        print(f"{'Layer':>8} {'Base mean |h|':>16} {'Mean |delta h|':>16} {'Difference %':>16}")
        for row in hidden["layers"]:
            percent = row["relative_mae_percent"]
            formatted = f"{percent:.6f}%" if percent is not None else "undefined"
            print(f"{row['layer']:>8} {row['base_mean_absolute_activation']:>16.6f} "
                  f"{row['mean_absolute_difference']:>16.6f} {formatted:>16}")
        mean = hidden["mean_relative_mae_percent"]
        print(f"Mean across layers: {mean:.6f}%" if mean is not None else "Mean across layers: undefined")
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
