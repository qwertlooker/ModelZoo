"""Shared Canary-1B inference/evaluation helpers.

This module is intentionally scoped to model execution helpers used by
``infer.py`` and ``eval_canary.py``.  It imports the required NeMo / torch / NPU
runtime dependencies at module import time so missing execution dependencies fail
promptly instead of silently falling back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch_npu
from nemo.collections.asr.models import EncDecMultiTaskModel


def resolve_device(device_name: str) -> torch.device:
    """Return a torch.device without hard-coding card indices."""
    if device_name not in {"npu", "cpu", "cuda"}:
        raise ValueError("--device must be one of: npu, cpu, cuda")
    if device_name == "npu" and not torch_npu.npu.is_available():
        raise RuntimeError("NPU device requested but torch_npu reports NPU unavailable")
    return torch.device(device_name)


def resolve_compute_dtype(
    dtype_name: str,
    device: torch.device | None = None,
    auto_bf16: bool = False,
) -> torch.dtype | None:
    """Return requested model compute dtype; None preserves checkpoint/default dtype."""
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "auto":
        if auto_bf16:
            if device is None:
                raise ValueError("device is required when auto_bf16=True")
            if device.type in {"npu", "cuda"}:
                return torch.bfloat16
        return None
    raise ValueError("--compute_dtype must be one of: auto, float32, float16, bfloat16")


def load_canary_model(
    model: str,
    device_name: str,
    compute_dtype: str = "auto",
    auto_bf16: bool = False,
    beam_size: int | None = None,
    decoding_strategy: str | None = None,
    performance_mode: bool = False,
) -> Any:
    """Load Canary-1B from a .nemo file, local directory, or HF model id."""
    device = resolve_device(device_name)
    model_path = Path(model).expanduser()
    if model_path.is_file() and model_path.suffix == ".nemo":
        loaded_model = EncDecMultiTaskModel.restore_from(str(model_path), map_location=device)
    elif model_path.is_dir() and (model_path / "canary-1b.nemo").is_file():
        loaded_model = EncDecMultiTaskModel.restore_from(str(model_path / "canary-1b.nemo"), map_location=device)
    else:
        loaded_model = EncDecMultiTaskModel.from_pretrained(model, map_location=device)

    loaded_model.eval()
    loaded_model.to(device)
    dtype = resolve_compute_dtype(compute_dtype, device=device, auto_bf16=auto_bf16)
    if dtype is not None:
        loaded_model.to(dtype)
    if beam_size is not None:
        configure_decoding(
            loaded_model,
            beam_size=beam_size,
            decoding_strategy=decoding_strategy,
            performance_mode=performance_mode,
        )
    return loaded_model


def configure_decoding(
    model: Any,
    beam_size: int,
    decoding_strategy: str | None = None,
    performance_mode: bool = False,
) -> None:
    """Apply Canary AED decoding settings in-place."""
    decode_cfg = model.cfg.decoding
    decode_cfg.beam.beam_size = beam_size
    if decoding_strategy is not None and decoding_strategy != "auto":
        decode_cfg.strategy = decoding_strategy
    elif decoding_strategy == "auto" and performance_mode and beam_size == 1:
        decode_cfg.strategy = "greedy_batch"
    model.change_decoding_strategy(decode_cfg)


def synchronize_device(device_name: str) -> None:
    """Synchronize asynchronous accelerator work before/after timed sections."""
    if device_name == "cuda":
        torch.cuda.synchronize()
    elif device_name == "npu":
        torch.npu.synchronize()


def extract_text(item: Any) -> str:
    """Return the expected NeMo transcription text field."""
    return str(item.text)
