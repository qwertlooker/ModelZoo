#!/usr/bin/env python3
"""Canary-1B inference entry for CPU/NPU adaptation.

This script is maintained in the current model directory and is not part of the
upstream NeMo patch set.  The default device is NPU; use ``--device cpu`` for
CPU validation.  The actual NPU card is selected by ASCEND_RT_VISIBLE_DEVICES.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import torch
from nemo.collections.asr.models import EncDecMultiTaskModel


def _resolve_device(device_name: str) -> torch.device:
    """Return a torch.device without hard-coding card indices."""
    if device_name == "npu":
        import torch_npu  # noqa: F401 - registers the NPU backend with PyTorch

    if device_name not in {"npu", "cpu", "cuda"}:
        raise ValueError("--device must be one of: npu, cpu, cuda")
    return torch.device(device_name)


def _extract_text(item: Any) -> str:
    """Return the expected NeMo transcription text field."""
    return str(item.text)


def _resolve_compute_dtype(dtype_name: str) -> torch.dtype | None:
    """Return requested model compute dtype; None preserves checkpoint/default dtype."""
    if dtype_name == "auto":
        return None
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError("--compute_dtype must be one of: auto, float32, float16, bfloat16")


def _build_manifest(args: argparse.Namespace) -> str:
    """Build a temporary Canary manifest so ASR/AST language tokens are explicit."""
    manifest = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    with manifest:
        for audio_path in args.audio:
            item = {
                "audio_filepath": str(Path(audio_path).expanduser()),
                "duration": args.duration,
                "taskname": args.task,
                "source_lang": args.source_lang,
                "target_lang": args.target_lang,
                "pnc": args.pnc,
            }
            manifest.write(json.dumps(item, ensure_ascii=False) + "\n")
    return manifest.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NVIDIA NeMo Canary-1B on CPU/NPU")
    parser.add_argument("--model", default="nvidia/canary-1b", help="HF model id or local .nemo/path")
    parser.add_argument("--audio", nargs="+", required=True, help="One or more input wav/flac files")
    parser.add_argument("--device", default="npu", help="npu, cpu, or cuda; default: npu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--task", choices=["asr", "ast", "s2t_translation"], default="asr")
    parser.add_argument("--source_lang", default="en", choices=["en", "de", "es", "fr"])
    parser.add_argument("--target_lang", default="en", choices=["en", "de", "es", "fr"])
    parser.add_argument("--pnc", default="yes", choices=["yes", "no"])
    parser.add_argument("--duration", type=float, default=100000.0, help="Manifest duration value")
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument(
        "--compute_dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model compute dtype. auto preserves the checkpoint/default dtype.",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)

    model_path = Path(args.model).expanduser()
    if model_path.is_file() and model_path.suffix == ".nemo":
        model = EncDecMultiTaskModel.restore_from(str(model_path), map_location=device)
    elif model_path.is_dir() and (model_path / "canary-1b.nemo").is_file():
        model = EncDecMultiTaskModel.restore_from(str(model_path / "canary-1b.nemo"), map_location=device)
    else:
        model = EncDecMultiTaskModel.from_pretrained(args.model, map_location=device)
    model.eval()
    model.to(device)
    compute_dtype = _resolve_compute_dtype(args.compute_dtype)
    if compute_dtype is not None:
        model.to(compute_dtype)

    decode_cfg = model.cfg.decoding
    decode_cfg.beam.beam_size = args.beam_size
    model.change_decoding_strategy(decode_cfg)

    manifest_path = _build_manifest(args)
    outputs = model.transcribe(audio=manifest_path, batch_size=args.batch_size)

    for idx, item in enumerate(outputs):
        print(f"[{idx}] {_extract_text(item)}")


if __name__ == "__main__":
    main()
