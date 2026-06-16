#!/usr/bin/env python3
"""Prepare a tiny JSONL schema sample for MOSS-TTSD-v0.5.

The default synthetic prompt is only for pipeline smoke testing and JSONL shape
validation.  It is not a speech-quality or voice-cloning evaluation sample.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import wave
from pathlib import Path


def write_tone(path: Path, frequency: float, seconds: float, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    amplitude = 0.20
    frames = bytearray()
    total = int(seconds * sample_rate)
    for idx in range(total):
        value = amplitude * math.sin(2.0 * math.pi * frequency * idx / sample_rate)
        frames.extend(struct.pack("<h", int(max(-1.0, min(1.0, value)) * 32767)))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(frames)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create MOSS-TTSD-v0.5 smoke-test JSONL")
    parser.add_argument("--output_dir", default="MOSS-TTSD-v0.5/test_data")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--prompt_seconds", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    s1 = out / "synthetic_speaker1.wav"
    s2 = out / "synthetic_speaker2.wav"
    write_tone(s1, frequency=220.0, seconds=args.prompt_seconds, sample_rate=args.sample_rate)
    write_tone(s2, frequency=330.0, seconds=args.prompt_seconds, sample_rate=args.sample_rate)

    item = {
        "base_path": str(out.resolve()),
        "text": "[S1]这是一个用于检查推理链路的中文短句。[S2]This is a short English sentence for pipeline validation.",
        "prompt_audio_speaker1": s1.name,
        "prompt_text_speaker1": "这是说话人一的参考文本。",
        "prompt_audio_speaker2": s2.name,
        "prompt_text_speaker2": "This is the reference text for speaker two.",
    }
    jsonl = out / "smoke.jsonl"
    jsonl.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
    print(jsonl)


if __name__ == "__main__":
    main()
