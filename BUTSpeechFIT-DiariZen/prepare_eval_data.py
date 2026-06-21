#!/usr/bin/env python3
"""Validate diarization audio/reference inputs and create a fixed manifest."""

import argparse
import hashlib
import json
from pathlib import Path

import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a DiariZen evaluation manifest.")
    parser.add_argument("--wav_scp", required=True)
    parser.add_argument("--reference_rttm")
    parser.add_argument("--uem")
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rttm_ids(path: Path) -> set[str]:
    ids = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 8 or parts[0] != "SPEAKER":
                raise ValueError(f"{path}:{line_number}: invalid RTTM row")
            ids.add(parts[1])
    return ids


def main() -> None:
    args = parse_args()
    wav_scp = Path(args.wav_scp).resolve()
    if not wav_scp.is_file():
        raise FileNotFoundError(wav_scp)
    output_path = Path(args.output_manifest).resolve()
    rows = []
    with wav_scp.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"{wav_scp}:{line_number}: expected '<id> <wav>'")
            audio_path = Path(parts[1])
            if not audio_path.is_absolute():
                audio_path = wav_scp.parent / audio_path
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            info = sf.info(audio_path)
            if info.frames <= 0:
                raise ValueError(f"Empty audio: {audio_path}")
            resolved_audio = audio_path.resolve()
            try:
                stored_audio = str(resolved_audio.relative_to(output_path.parent))
            except ValueError:
                stored_audio = str(resolved_audio)
            rows.append(
                {
                    "id": parts[0],
                    "audio_path": stored_audio,
                    "duration": info.frames / info.samplerate,
                    "sample_rate": info.samplerate,
                    "channels": info.channels,
                    "sha256": sha256(audio_path),
                }
            )
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"Empty wav.scp: {wav_scp}")
    ids = {row["id"] for row in rows}
    if len(ids) != len(rows):
        raise ValueError(f"Duplicate session ids in {wav_scp}")

    reference_path = Path(args.reference_rttm).resolve() if args.reference_rttm else None
    if reference_path:
        if not reference_path.is_file():
            raise FileNotFoundError(reference_path)
        missing_references = sorted(ids - rttm_ids(reference_path))
        if missing_references:
            raise ValueError(f"Sessions missing from reference RTTM: {missing_references}")
    uem_path = Path(args.uem).resolve() if args.uem else None
    if uem_path and not uem_path.is_file():
        raise FileNotFoundError(uem_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "sample_count": len(rows),
        "total_audio_seconds": sum(row["duration"] for row in rows),
        "wav_scp": str(wav_scp),
        "wav_scp_sha256": sha256(wav_scp),
        "reference_rttm": str(reference_path) if reference_path else None,
        "reference_rttm_sha256": sha256(reference_path) if reference_path else None,
        "uem": str(uem_path) if uem_path else None,
        "uem_sha256": sha256(uem_path) if uem_path else None,
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
