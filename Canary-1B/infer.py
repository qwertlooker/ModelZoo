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
from utils import extract_text, load_canary_model


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

    model = load_canary_model(
        args.model,
        device_name=args.device,
        compute_dtype=args.compute_dtype,
        beam_size=args.beam_size,
    )

    manifest_path = _build_manifest(args)
    outputs = model.transcribe(audio=manifest_path, batch_size=args.batch_size)

    for idx, item in enumerate(outputs):
        print(f"[{idx}] {extract_text(item)}")


if __name__ == "__main__":
    main()
