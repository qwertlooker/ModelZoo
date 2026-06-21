#!/usr/bin/env python3
"""Validate WAV files and create a deterministic DNSMOS evaluation manifest."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a DNSMOS WAV manifest.")
    parser.add_argument("--audio_dir", nargs="+", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="unspecified")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    roots = [Path(value).resolve() for value in args.audio_dir]
    for root in roots:
        if not root.is_dir():
            raise NotADirectoryError(root)
    audio_paths = sorted(
        {
            path.resolve()
            for root in roots
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() == ".wav"
        }
    )
    if not audio_paths:
        raise ValueError("No WAV files found.")
    if args.limit and len(audio_paths) > args.limit:
        random.Random(args.seed).shuffle(audio_paths)
        audio_paths = sorted(audio_paths[: args.limit])

    output_path = Path(args.output_manifest).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    total_seconds = 0.0
    sample_rates = set()
    channels = set()
    for index, audio_path in enumerate(audio_paths):
        info = sf.info(audio_path)
        if info.frames <= 0 or info.samplerate <= 0:
            raise ValueError(f"Unreadable or empty audio: {audio_path}")
        duration = info.frames / info.samplerate
        total_seconds += duration
        sample_rates.add(info.samplerate)
        channels.add(info.channels)
        try:
            relative_path = audio_path.relative_to(output_path.parent)
            stored_path = str(relative_path)
        except ValueError:
            stored_path = str(audio_path)
        rows.append(
            {
                "id": f"{index:06d}",
                "audio_path": stored_path,
                "duration": duration,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "sha256": sha256(audio_path),
            }
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "source_directories": [str(root) for root in roots],
        "limit": args.limit,
        "seed": args.seed,
        "sample_count": len(rows),
        "total_audio_seconds": total_seconds,
        "sample_rates": sorted(sample_rates),
        "channels": sorted(channels),
        "manifest_sha256": sha256(output_path),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
