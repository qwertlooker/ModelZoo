#!/usr/bin/env python3
"""Validate MOSS-TTSD-v0.5 output files structurally.

This is not a substitute for MOS/CMOS, speaker similarity, ASR-CER/WER, DNSMOS,
or TTSD-eval.  It only checks that generated WAV files referenced by an output
manifest exist, are readable PCM WAV files, and have non-zero duration.
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return float(handle.getnframes()) / float(handle.getframerate())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check structural validity of generated WAV outputs")
    parser.add_argument("--manifest", required=True, help="manifest.jsonl emitted by infer.py")
    parser.add_argument("--min_total_seconds", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest).expanduser()
    total_seconds = 0.0
    total_files = 0
    with manifest.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            files = row["audio_files"]
            if not files:
                raise ValueError(f"{manifest}:{line_no} has no audio_files")
            for file_name in files:
                path = Path(file_name).expanduser()
                if not path.is_file():
                    raise FileNotFoundError(path)
                seconds = wav_seconds(path)
                if seconds <= 0.0:
                    raise ValueError(f"zero-duration WAV: {path}")
                total_seconds += seconds
                total_files += 1
    if total_files == 0:
        raise ValueError(f"no rows found in {manifest}")
    if total_seconds < args.min_total_seconds:
        raise ValueError(f"total audio duration {total_seconds:.3f}s < {args.min_total_seconds:.3f}s")
    print(json.dumps({"files": total_files, "total_seconds": total_seconds}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
