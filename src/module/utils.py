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
# .env helper
# ─────────────────────────────────────────────
def load_env(path: Path | None = None) -> dict:
    """Read KEY=VALUE pairs from the project's .env; real environment variables win.

    Values are taken literally (no escape processing), so Windows paths work as written.
    """
    import os

    values = {}
    env_file = path or PROJECT_ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("\"'")
    values.update({key: os.environ[key] for key in values if key in os.environ})
    if "DATASET_PATH" in os.environ:
        values["DATASET_PATH"] = os.environ["DATASET_PATH"]
    return values


# ─────────────────────────────────────────────
# Path helper
# ─────────────────────────────────────────────
def to_local_path(path) -> Path:
    """Expand ~ and, under WSL/Linux, map a Windows path like C:\\Users\\me\\x to /mnt/c/Users/me/x."""
    import os
    import re

    text = str(path)
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", text)
    if match and os.name != "nt":
        text = f"/mnt/{match.group(1).lower()}/" + match.group(2).replace("\\", "/")
    return Path(text).expanduser()


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
