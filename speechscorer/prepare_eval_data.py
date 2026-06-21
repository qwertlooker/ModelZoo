#!/usr/bin/env python3
"""Prepare SpeechOcean762 test audio and a fixed evaluation manifest."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SpeechOcean762 test split.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_wav_scp(path: Path) -> list[tuple[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_number}: expected '<id> <path>'")
            rows.append((parts[0], parts[1]))
    if not rows:
        raise ValueError(f"Empty wav.scp: {path}")
    return rows


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    wav_scp = dataset_dir / "test" / "wav.scp"
    labels_path = dataset_dir / "test" / "all-info.json"
    if not wav_scp.is_file():
        raise FileNotFoundError(wav_scp)
    if not labels_path.is_file():
        raise FileNotFoundError(labels_path)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    rows = read_wav_scp(wav_scp)
    if args.limit:
        rows = rows[: args.limit]

    output_dir = Path(args.output_dir).resolve()
    audio_dir = output_dir / "wavs"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    manifest_rows = []
    total_seconds = 0.0
    for utterance_id, relative_path in rows:
        source = dataset_dir / relative_path
        if not source.is_file():
            raise FileNotFoundError(source)
        target = audio_dir / f"{utterance_id}{source.suffix.lower()}"
        shutil.copy2(source, target)
        info = sf.info(target)
        if info.frames <= 0 or info.samplerate <= 0:
            raise ValueError(f"Unreadable audio: {target}")
        if utterance_id not in labels:
            raise KeyError(f"Missing label for {utterance_id}")
        duration = info.frames / info.samplerate
        total_seconds += duration
        manifest_rows.append(
            {
                "utterance_id": utterance_id,
                "audio_path": str(target.relative_to(output_dir)),
                "duration": duration,
                "sample_rate": info.samplerate,
                "sha256": sha256(target),
                "scores": {
                    key: value
                    for key, value in labels[utterance_id].items()
                    if key not in {"words", "text"}
                },
            }
        )
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "dataset": "SpeechOcean762",
        "split": "test",
        "source_wav_scp": str(wav_scp),
        "source_labels": str(labels_path),
        "limit": args.limit,
        "sample_count": len(manifest_rows),
        "total_audio_seconds": total_seconds,
        "manifest_sha256": sha256(manifest_path),
    }
    metadata_path = manifest_path.with_suffix(".jsonl.meta.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
