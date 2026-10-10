"""Export the Fable-5 base and LoRA adapter separately in 4-bit NF4.

Base uses standard Transformers/bitsandbytes storage. Adapter uses a custom
NF4 format, restored to FP32 by attach_adapter at runtime.
Both use double quantization; base embeddings/norm/head remain floating point.
Source files are never changed.
"""

import gc
import json
import math
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from .utils import FINAL_ADAPTER, SEPARATE_ROOT, resolve_device


MANIFEST = "adapter_quantization.json"
WEIGHTS = "adapter_model.nf4.safetensors"
FORMAT = "exit-sign-lora-nf4"
FORMAT_VERSION = 1
STATE_KEYS = {"absmax", "quant_map", "nested_absmax", "nested_quant_map",
              "quant_state.bitsandbytes__nf4"}


def _read_config(directory):
    config = json.loads((Path(directory) / "adapter_config.json").read_text(encoding="utf-8"))
    if (config.get("peft_type") != "LORA" or config.get("bias", "none") != "none"
            or config.get("modules_to_save") or config.get("use_dora")
            or config.get("lora_bias") or config.get("target_parameters")
            or config.get("trainable_token_indices") or config.get("alora_invocation_tokens")):
        raise ValueError("NF4 adapter storage supports only ordinary LoRA A/B weights without biases or extra modules.")
    return config


def _validate_weight(name, tensor):
    import torch

    if not name.endswith((".lora_A.weight", ".lora_B.weight")):
        raise ValueError(f"Unsupported adapter weight: {name}; expected LoRA A/B weights only.")
    if tensor.ndim != 2 or tensor.numel() == 0 or tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"Unsupported shape/dtype for {name}: {tuple(tensor.shape)}, {tensor.dtype}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Non-finite adapter weight: {name}")


def export_quantized_adapter(source, destination, device, base_model_id=None, base_revision=None):
    """Save NF4 tensors and state, then verify exact serialization round trips."""
    import torch
    from bitsandbytes import functional as bnb
    from safetensors.torch import load_file, save_file

    source, destination = Path(source), Path(destination)
    config = _read_config(source)
    if destination.exists():
        raise FileExistsError(f"Adapter output already exists: {destination}")
    if (source / "adapter_model.safetensors").is_file():
        weights = load_file(str(source / "adapter_model.safetensors"))
    else:
        weights = torch.load(source / "adapter_model.bin", map_location="cpu", weights_only=True)
    if not weights:
        raise ValueError("Adapter checkpoint contains no weights.")
    payload, entries, reference = {}, {}, {}
    error_squared, elements, max_error = 0.0, 0, 0.0
    for index, (name, tensor) in enumerate(sorted(weights.items())):
        _validate_weight(name, tensor)
        prefix = f"tensor_{index:04d}"
        original = tensor.to(device).contiguous()
        packed, state = bnb.quantize_4bit(
            original, blocksize=64, compress_statistics=True, quant_type="nf4", quant_storage=torch.uint8,
        )
        # Clone the state tensors: quantization maps may share storage.
        payload[f"{prefix}.data"] = packed.cpu().contiguous().clone()
        state_dict = state.as_dict(packed=True)
        for key, value in state_dict.items():
            payload[f"{prefix}.state.{key}"] = value.cpu().contiguous().clone()
        restored = bnb.dequantize_4bit(packed, quant_state=state).float()
        if not torch.isfinite(restored).all():
            raise ValueError(f"Quantization produced non-finite values: {name}")
        reference[name] = restored.cpu()
        error = restored - original.float()
        squared = error.double().square().sum().item()
        error_squared += squared
        elements += tensor.numel()
        max_error = max(max_error, error.abs().max().item())
        entries[name] = {
            "prefix": prefix, "shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch."),
            "state_keys": sorted(state_dict), "rmse": math.sqrt(squared / tensor.numel()),
        }
    manifest = {
        "format": FORMAT, "format_version": FORMAT_VERSION,
        "quant_type": "nf4", "blocksize": 64, "double_quant": True,
        "runtime_dtype": "float32", "base_model_name_or_path": base_model_id or config.get("base_model_name_or_path"),
        "base_revision": base_revision if base_revision is not None else config.get("revision"),
        "versions": {package: version(package) for package in ("torch", "transformers", "peft", "bitsandbytes", "safetensors")},
        "tensors": entries,
        "metrics": {"tensor_count": len(entries), "elements": elements,
                    "rmse": math.sqrt(error_squared / elements), "max_abs_error": max_error,
                    "original_weight_bytes": sum(t.numel() * t.element_size() for t in weights.values())},
    }
    destination.mkdir(parents=True)
    (destination / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    save_file(payload, str(destination / WEIGHTS), metadata={"format": FORMAT})
    manifest["metrics"]["quantized_weight_bytes"] = (destination / WEIGHTS).stat().st_size
    (destination / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    reloaded = load_quantized_adapter_state(destination, device)
    for name, expected in reference.items():
        if not torch.equal(expected, reloaded[name]):
            raise RuntimeError(f"Adapter serialization round trip changed {name}")
    metrics = manifest["metrics"]
    print(f"Verified {len(entries)} adapter tensors; round trip exact; NF4 RMSE={metrics['rmse']:.6g}, "
          f"max error={max_error:.6g}; weights {metrics['original_weight_bytes'] / 1024**2:.2f} -> "
          f"{metrics['quantized_weight_bytes'] / 1024**2:.2f} MiB.", flush=True)
    return manifest


def load_quantized_adapter_state(directory, device):
    """Read and validate the custom checkpoint; return ordinary FP32 CPU tensors."""
    import torch
    from bitsandbytes import functional as bnb
    from safetensors.torch import load_file

    directory = Path(directory)
    _read_config(directory)
    manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT or manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported adapter quantization format/version in {directory / MANIFEST}")
    if (manifest.get("quant_type") != "nf4" or manifest.get("blocksize") != 64
            or manifest.get("double_quant") is not True or manifest.get("runtime_dtype") != "float32"):
        raise ValueError("Unsupported adapter quantization settings; expected double-quantized NF4, blocksize 64.")
    entries = manifest.get("tensors")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Adapter quantization manifest has no tensors.")
    payload = load_file(str(directory / WEIGHTS))
    expected_keys, prefixes = set(), set()
    for name, entry in entries.items():
        prefix = entry.get("prefix")
        if not isinstance(prefix, str) or prefix in prefixes:
            raise ValueError(f"Missing or duplicate tensor prefix for {name}")
        prefixes.add(prefix)
        if set(entry.get("state_keys", [])) != STATE_KEYS:
            raise ValueError(f"Missing or unsupported quantization state for {name}")
        expected_keys.add(f"{prefix}.data")
        expected_keys.update(f"{prefix}.state.{key}" for key in STATE_KEYS)
    if set(payload) != expected_keys:
        raise ValueError(f"Incomplete adapter weights/quantization state: missing={sorted(expected_keys - set(payload))[:3]}, "
                         f"unexpected={sorted(set(payload) - expected_keys)[:3]}")
    restored_weights = {}
    device = torch.device(device)
    for name, entry in entries.items():
        prefix = entry["prefix"]
        try:
            state = bnb.QuantState.from_dict(
                {key: payload[f"{prefix}.state.{key}"] for key in STATE_KEYS}, device=device,
            )
            if (list(state.shape) != entry.get("shape") or str(state.dtype).removeprefix("torch.") != entry.get("dtype")
                    or state.quant_type != "nf4" or state.blocksize != 64 or state.state2 is None):
                raise ValueError("shape, dtype, or NF4 state does not match manifest")
            packed = payload[f"{prefix}.data"]
            if packed.dtype != torch.uint8 or packed.numel() != (math.prod(state.shape) + 1) // 2:
                raise ValueError("packed weight dtype or size is invalid")
            tensor = bnb.dequantize_4bit(packed.to(device), quant_state=state).float().cpu()
            _validate_weight(name, tensor)
            restored_weights[name] = tensor
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            raise ValueError(f"Invalid quantized adapter tensor {name}: {exc}") from exc
    return restored_weights


def attach_adapter(base_model, directory, device):
    """Attach either this custom NF4 format or an existing standard PEFT adapter."""
    import torch
    from peft import PeftConfig, PeftModel, get_peft_model
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    directory = Path(directory)
    if not (directory / MANIFEST).exists():
        if (directory / WEIGHTS).exists():
            raise ValueError(f"Missing {MANIFEST} for NF4 adapter in {directory}")
        model = PeftModel.from_pretrained(base_model, str(directory), local_files_only=True)
        print(f"Loaded standard PEFT adapter: {directory}", flush=True)
        return model

    weights = load_quantized_adapter_state(directory, device)
    manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    base_identity = getattr(base_model.config, "exit_sign_base_model", None)
    if base_identity and base_identity != manifest.get("base_model_name_or_path"):
        raise ValueError(f"Adapter base model mismatch: {manifest.get('base_model_name_or_path')} != {base_identity}")
    config = PeftConfig.from_pretrained(str(directory), local_files_only=True)
    config.inference_mode = True
    # This in-memory config refers to the local model being attached. The source
    # identity remains in the manifest; do not leave a staging directory on disk.
    config.base_model_name_or_path = base_model.name_or_path
    model = get_peft_model(base_model, config, autocast_adapter_dtype=True)
    expected = get_peft_model_state_dict(model, save_embedding_layers=False)
    if set(expected) != set(weights):
        raise ValueError(f"Adapter names do not match base model: missing={sorted(set(expected) - set(weights))[:3]}, "
                         f"unexpected={sorted(set(weights) - set(expected))[:3]}")
    for name, tensor in weights.items():
        if tensor.shape != expected[name].shape:
            raise ValueError(f"Adapter shape mismatch for {name}: {tuple(tensor.shape)} != {tuple(expected[name].shape)}")
    result = set_peft_model_state_dict(model, weights)
    if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
        raise ValueError(f"Failed to install all adapter weights: {result}")
    installed = get_peft_model_state_dict(model, save_embedding_layers=False)
    for name, tensor in installed.items():
        if tensor.dtype != torch.float32 or not torch.equal(tensor.cpu(), weights[name]):
            raise RuntimeError(f"Adapter weight was not installed exactly as FP32: {name}")
    print(f"Loaded NF4 adapter: {directory} (4-bit on disk; FP32 at runtime, {len(weights)} tensors).", flush=True)
    return model


def quantize_model(
    adapter_path: str | Path = FINAL_ADAPTER,
    output_dir: str | Path = SEPARATE_ROOT,
    base_model: str | None = None,
    device: str = "cuda:0",
    local_files_only: bool = False,
):
    """Export base-4bit and adapter-4bit under output_dir; return output_dir."""
    adapter_path = Path(adapter_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing adapter_config.json in {adapter_path}")
    if not any((adapter_path / name).is_file() for name in
               ("adapter_model.safetensors", "adapter_model.bin")):
        raise FileNotFoundError(f"Missing LoRA adapter weights in {adapter_path}")
    if output_dir == adapter_path or output_dir in adapter_path.parents or adapter_path in output_dir.parents:
        raise ValueError("Output and adapter directories must not overlap.")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}. Choose a new output directory.")

    import torch
    from bitsandbytes.nn import Linear4bit
    from peft import PeftConfig
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    device, compute_dtype = resolve_device(device)

    peft_config = PeftConfig.from_pretrained(str(adapter_path), local_files_only=True)
    if peft_config.peft_type != "LORA":
        raise ValueError("This exporter expects the LoRA adapter produced by src.train.")
    base_model_id = base_model or peft_config.base_model_name_or_path
    if not base_model_id:
        raise ValueError("Adapter has no base model ID; specify --base-model.")
    base_kwargs = {"local_files_only": local_files_only}
    if peft_config.revision and not base_model:
        base_kwargs["revision"] = peft_config.revision
    base_config = AutoConfig.from_pretrained(base_model_id, **base_kwargs)
    if getattr(base_config, "quantization_config", None):
        raise ValueError("Use the original unquantized base model for export (--base-model).")

    # Keep the training tokenizer and its chat template whenever available.
    if any((adapter_path / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), local_files_only=True)
    else:
        print("No tokenizer found in adapter directory; using the base model tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, **base_kwargs)
        template_path = adapter_path / "chat_template.jinja"
        if template_path.is_file():
            tokenizer.chat_template = template_path.read_text(encoding="utf-8")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Stage both artifacts on the output filesystem; publish only after reload succeeds.
    with TemporaryDirectory(prefix=".quantization-", dir=output_dir.parent) as temp_dir:
        export_dir = Path(temp_dir) / "export"
        model_dir = export_dir / "base-4bit"
        print("[1/4] Quantize adapter independently (NF4 on disk, FP32 at runtime).", flush=True)
        export_quantized_adapter(
            adapter_path, export_dir / "adapter-4bit", device,
            base_model_id=base_model_id, base_revision=base_kwargs.get("revision"),
        )
        tokenizer.save_pretrained(export_dir / "adapter-4bit")
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_id, **base_kwargs)

        print(f"[2/4] Quantize to NF4 on {device} (compute dtype: {compute_dtype}).", flush=True)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            config=base_config,
            quantization_config=quantization_config,
            dtype=compute_dtype,
            device_map={"": str(device)},
            **base_kwargs,
        )
        model.eval()
        if not getattr(model, "is_loaded_in_4bit", False):
            raise RuntimeError("Model was not loaded in 4-bit mode.")
        print("[3/4] Save quantized model and tokenizer.", flush=True)
        model.config.use_cache = True
        model.config.exit_sign_export_mode = "base"
        model.config.exit_sign_base_model = base_model_id
        model.save_pretrained(model_dir, safe_serialization=True)
        base_tokenizer.save_pretrained(model_dir)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print("[4/4] Reload saved model and verify 4-bit weights.", flush=True)
        # Read quantization settings from the saved config, just as inference will.
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, device_map={"": str(device)}, local_files_only=True,
        )
        model.eval()
        quantized_layers = [module for module in model.modules() if isinstance(module, Linear4bit)]
        if not getattr(model, "is_loaded_in_4bit", False) or not quantized_layers:
            raise RuntimeError("Saved model did not reload as a 4-bit model.")
        if any(module.weight.quant_state is None or module.weight.quant_state.quant_type != "nf4"
               or module.weight.quant_state.state2 is None for module in quantized_layers):
            raise RuntimeError("Saved model is missing 4-bit quantization state.")
        if any("lora_" in name for name, _ in model.named_parameters()):
            raise RuntimeError("Saved base model unexpectedly contains LoRA parameters.")
        print(f"Verified {len(quantized_layers)} NF4 linear layers.", flush=True)
        attach_adapter(model, export_dir / "adapter-4bit", device)
        del quantized_layers, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        export_dir.rename(output_dir)

    size_mb = sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file()) / 1024**2
    print(f"Done: {output_dir} ({size_mb:.1f} MiB)")
    return output_dir
