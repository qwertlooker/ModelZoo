#!/usr/bin/env python3
"""Run the pinned dscore implementation with explicit overlap semantics."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score RTTM output with dscore.")
    parser.add_argument("--dscore_dir", required=True)
    parser.add_argument("--reference_rttm", nargs="+", required=True)
    parser.add_argument("--system_rttm", nargs="+", required=True)
    parser.add_argument("--uem")
    parser.add_argument("--collar", type=float, default=0.0)
    parser.add_argument("--ignore_overlaps", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    score_script = Path(args.dscore_dir) / "score.py"
    if not score_script.is_file():
        raise FileNotFoundError(score_script)
    references = [Path(value) for value in args.reference_rttm]
    systems = [Path(value) for value in args.system_rttm]
    for path in references + systems:
        if not path.is_file():
            raise FileNotFoundError(path)
    command = [
        sys.executable,
        str(score_script),
        "-r",
        *[str(path) for path in references],
        "-s",
        *[str(path) for path in systems],
        "--collar",
        str(args.collar),
    ]
    if args.uem:
        uem = Path(args.uem)
        if not uem.is_file():
            raise FileNotFoundError(uem)
        # dscore e02f949 exposes this historical option string literally.
        command.extend(["-u,--uem", str(uem)])
    if args.ignore_overlaps:
        command.append("--ignore_overlaps")
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(completed.stdout, encoding="utf-8")
    metadata = {
        "command": command,
        "score_script": str(score_script),
        "score_script_sha256": sha256(score_script),
        "reference_rttm": [
            {"path": str(path), "sha256": sha256(path)} for path in references
        ],
        "system_rttm": [
            {"path": str(path), "sha256": sha256(path)} for path in systems
        ],
        "uem": str(args.uem) if args.uem else None,
        "uem_sha256": sha256(Path(args.uem)) if args.uem else None,
        "collar": args.collar,
        "ignore_overlaps": args.ignore_overlaps,
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(completed.stdout, end="")
    print(f"output={output_path}")
    print(f"metadata={metadata_path}")


if __name__ == "__main__":
    main()
