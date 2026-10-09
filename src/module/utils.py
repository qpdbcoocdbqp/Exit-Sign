"""Shared paths, profiling, and device helpers used by several modules."""

import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_ID = "Qwen/Qwen3-0.6B"
DATASET_ID = "Glint-Research/Fable-5-traces"

OUTPUT_ROOT = PROJECT_ROOT / "qwen3-fable5-sft"
FINAL_ADAPTER = OUTPUT_ROOT / "final"
SEPARATE_ROOT = OUTPUT_ROOT / "separate-4bit"
DEFAULT_ADAPTER = SEPARATE_ROOT / "adapter-4bit"
DEFAULT_BASE = SEPARATE_ROOT / "base-4bit"


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
# Device helper
# ─────────────────────────────────────────────
def resolve_device(device):
    """Validate a cpu/cuda device, select it, and return (device, compute dtype)."""
    import torch

    target_device = torch.device(device)
    if target_device.type not in ("cuda", "cpu"):
        raise ValueError("device must be cpu or a CUDA device such as cuda:0.")
    if target_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Install CUDA-enabled PyTorch or use --device cpu.")
        torch.cuda.set_device(target_device)
        return target_device, torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return target_device, torch.float32
