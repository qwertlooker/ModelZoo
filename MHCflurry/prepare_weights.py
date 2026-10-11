# Licensed under the Apache License, Version 2.0.
"""Download with curl, verify the fixed official archive, then safely extract."""

import argparse
import subprocess
import tarfile
from pathlib import Path

from adapter_utils import WEIGHT_NAME, WEIGHT_SHA256, WEIGHT_URL, sha256


def extract_verified(archive, output):
    if sha256(archive) != WEIGHT_SHA256:
        raise ValueError("Official archive SHA256 mismatch")
    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError(target)
    with tarfile.open(archive, "r:bz2") as stream:
        for member in stream.getmembers():
            destination = (target / member.name).resolve()
            if not destination.is_relative_to(target) or not (
                member.isfile() or member.isdir()
            ):
                raise ValueError("Unsafe archive member: " + member.name)
        target.mkdir(parents=True)
        stream.extractall(target, filter="data")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="weights")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    archive = Path(args.directory) / WEIGHT_NAME
    if args.dry_run:
        print({"url": WEIGHT_URL, "archive": str(archive), "sha256": WEIGHT_SHA256})
        return
    if args.check_only:
        if args.offline:
            if sha256(archive) != WEIGHT_SHA256:
                raise ValueError("Official archive SHA256 mismatch")
        else:
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--head",
                    "--max-time",
                    "30",
                    WEIGHT_URL,
                ],
                check=True,
            )
        return
    if not archive.exists():
        if args.offline:
            raise FileNotFoundError(archive)
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(archive.suffix + ".tmp")
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "2",
                "--output",
                str(temporary),
                WEIGHT_URL,
            ],
            check=True,
        )
        if sha256(temporary) != WEIGHT_SHA256:
            raise ValueError("Downloaded archive SHA256 mismatch")
        temporary.rename(archive)
    extract_verified(archive, archive.parent / "extracted")
    print("Verified weights extracted to", archive.parent / "extracted")


if __name__ == "__main__":
    main()
