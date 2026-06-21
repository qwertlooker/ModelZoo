#!/usr/bin/env python3
"""Prepare deterministic MOSS-TTSD JSONL subsets and evaluator manifests."""

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MOSS-TTSD evaluation data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subset = subparsers.add_parser("subset")
    subset.add_argument("--input_jsonl", required=True)
    subset.add_argument("--output_jsonl", required=True)
    subset.add_argument("--limit", type=int, required=True)
    subset.add_argument("--dataset", required=True)
    subset.add_argument("--split", required=True)

    attach = subparsers.add_parser("attach-output")
    attach.add_argument("--input_jsonl", required=True)
    attach.add_argument("--output_jsonl", required=True)
    attach.add_argument("--output_dir", required=True)
    attach.add_argument(
        "--path_root",
        help="Directory used to validate relative prompt-audio paths.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "text" not in row:
                raise ValueError(f"{path}:{line_number}: missing text")
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty JSONL: {path}")
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def subset(args: argparse.Namespace) -> None:
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    input_path = Path(args.input_jsonl).resolve()
    output_path = Path(args.output_jsonl).resolve()
    rows = read_rows(input_path)[: args.limit]
    write_rows(output_path, rows)
    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "source": str(input_path),
        "source_sha256": sha256(input_path),
        "sample_count": len(rows),
        "manifest_sha256": sha256(output_path),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def attach_output(args: argparse.Namespace) -> None:
    input_path = Path(args.input_jsonl).resolve()
    output_path = Path(args.output_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()
    path_root = Path(args.path_root).resolve() if args.path_root else Path.cwd()
    rows = read_rows(input_path)
    attached = []
    for index, row in enumerate(rows):
        audio_path = output_dir / f"output_{index}.wav"
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        result = dict(row)
        result["output_audio"] = str(audio_path)
        for field in ("prompt_audio_speaker1", "prompt_audio_speaker2"):
            if field not in result:
                raise ValueError(f"{input_path}: missing {field}")
            prompt_path = Path(result[field]).expanduser()
            if not prompt_path.is_absolute():
                prompt_path = path_root / prompt_path
            if not prompt_path.is_file():
                raise FileNotFoundError(prompt_path)
        attached.append(result)
    write_rows(output_path, attached)
    metadata = {
        "source": str(input_path),
        "source_sha256": sha256(input_path),
        "output_dir": str(output_dir),
        "path_root": str(path_root),
        "sample_count": len(attached),
        "manifest_sha256": sha256(output_path),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "subset":
        subset(args)
    else:
        attach_output(args)


if __name__ == "__main__":
    main()
