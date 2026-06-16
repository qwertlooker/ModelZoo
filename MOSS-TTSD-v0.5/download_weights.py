#!/usr/bin/env python3
"""Download the pinned MOSS-TTSD-v0.5 model and codec snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_REPO = "OpenMOSS-Team/MOSS-TTSD-v0.5"
CODEC_REPO = "OpenMOSS-Team/XY_Tokenizer_TTSD_V0_hf"
MODEL_REVISION = "8527b9136b6afefe2252ae597cecea2e80e7ebeb"
CODEC_REVISION = "c884072fd69ed00b72cd0d43355c06341c4f51a6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MOSS-TTSD-v0.5 and codec snapshots")
    parser.add_argument("--output_dir", default="MOSS-TTSD-v0.5/weights")
    parser.add_argument("--model_repo", default=MODEL_REPO)
    parser.add_argument("--codec_repo", default=CODEC_REPO)
    parser.add_argument("--model_revision", default=MODEL_REVISION)
    parser.add_argument("--codec_revision", default=CODEC_REVISION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    model_dir = snapshot_download(
        args.model_repo,
        revision=args.model_revision,
        local_dir=str(root / "MOSS-TTSD-v0.5"),
    )
    codec_dir = snapshot_download(
        args.codec_repo,
        revision=args.codec_revision,
        local_dir=str(root / "XY_Tokenizer_TTSD_V0_hf"),
    )
    result = {
        "model_repo": args.model_repo,
        "model_revision": args.model_revision,
        "model_dir": model_dir,
        "codec_repo": args.codec_repo,
        "codec_revision": args.codec_revision,
        "codec_dir": codec_dir,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
