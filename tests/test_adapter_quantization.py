"""Small real NF4 round-trip and corruption tests; no model downloads required.

Run: python -m unittest discover -s tests -v
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from peft import get_peft_model_state_dict
from safetensors.torch import load_file, save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.module.quantization import MANIFEST, WEIGHTS, attach_adapter
from src.module.quantization import export_quantized_adapter, load_quantized_adapter_state, quantize_model


class AdapterQuantizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "source"
        self.output = Path(self.temp.name) / "output"
        self.source.mkdir()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.config = {
            "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 4,
            "lora_alpha": 8, "lora_dropout": 0.0, "bias": "none",
            "inference_mode": True, "target_modules": ["q_proj"],
            "base_model_name_or_path": "Qwen/Qwen3-0.6B",
        }
        (self.source / "adapter_config.json").write_text(json.dumps(self.config), encoding="utf-8")
        prefix = "base_model.model.model.layers.0.self_attn.q_proj"
        generator = torch.Generator().manual_seed(42)
        self.weights = {
            prefix + ".lora_A.weight": torch.randn(4, 32, generator=generator) * 0.01,
            prefix + ".lora_B.weight": torch.zeros(32, 4),
        }
        save_file(self.weights, str(self.source / "adapter_model.safetensors"))

    def export(self):
        return export_quantized_adapter(self.source, self.output, self.device)

    def edit_manifest(self, edit):
        path = self.output / MANIFEST
        data = json.loads(path.read_text(encoding="utf-8"))
        edit(data)
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_roundtrip_and_fp32_installation(self):
        report = self.export()
        self.assertEqual(report["metrics"]["tensor_count"], 2)
        restored = load_quantized_adapter_state(self.output, self.device)
        self.assertEqual(set(restored), set(self.weights))
        for name, tensor in restored.items():
            self.assertEqual(tensor.shape, self.weights[name].shape)
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertTrue(torch.isfinite(tensor).all())
        self.assertTrue(torch.equal(restored[next(name for name in restored if "lora_B" in name)], torch.zeros(32, 4)))
        base = self.make_base()
        model = attach_adapter(base, self.output, self.device)
        installed = get_peft_model_state_dict(model, save_embedding_layers=False)
        for name, tensor in installed.items():
            self.assertTrue(torch.equal(tensor.cpu(), restored[name]))
            self.assertEqual(tensor.dtype, torch.float32)
        with torch.inference_mode():
            self.assertTrue(torch.isfinite(model(input_ids=torch.tensor([[1, 2, 3]])).logits).all())

    def make_base(self):
        return Qwen3ForCausalLM(Qwen3Config(
            vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        ))

    def test_missing_state_rejected(self):
        self.export()
        payload = load_file(str(self.output / WEIGHTS))
        del payload[next(key for key in payload if key.endswith(".state.absmax"))]
        save_file(payload, str(self.output / WEIGHTS))
        with self.assertRaisesRegex(ValueError, "Incomplete.*quantization state"):
            load_quantized_adapter_state(self.output, self.device)

    def test_version_rejected(self):
        self.export()
        self.edit_manifest(lambda data: data.update(format_version=999))
        with self.assertRaisesRegex(ValueError, "format/version"):
            load_quantized_adapter_state(self.output, self.device)

    def test_manifest_shape_rejected(self):
        self.export()
        self.edit_manifest(lambda data: next(iter(data["tensors"].values())).update(shape=[1, 1]))
        with self.assertRaisesRegex(ValueError, "shape"):
            load_quantized_adapter_state(self.output, self.device)

    def test_base_adapter_shape_mismatch_rejected(self):
        self.export()
        self.config["r"] = 8
        (self.output / "adapter_config.json").write_text(json.dumps(self.config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            attach_adapter(self.make_base(), self.output, self.device)

    def test_unsupported_weights_rejected(self):
        save_file({"lm_head.weight": torch.ones(32, 32)}, str(self.source / "adapter_model.safetensors"))
        with self.assertRaisesRegex(ValueError, "Unsupported adapter weight"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_existing_output_protected(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            quantize_model(adapter_path=self.source, output_dir=self.output)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

if __name__ == "__main__":
    unittest.main()
