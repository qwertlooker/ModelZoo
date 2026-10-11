# Licensed under the Apache License, Version 2.0.
"""Strict, dependency-free input and provenance helpers."""

import csv
import hashlib
import json
import math
from pathlib import Path

SOURCE_REVISION = "20f03d45358dd104df9df0e28e2e0ccea11c9bfb"
WEIGHT_NAME = "models_class1_presentation.20260928.tar.bz2"
WEIGHT_SHA256 = "768e360832b3d45dfe9d5e8cde4ffcb90abdb87773ca65f3a86ba7cc20f614db"
WEIGHT_URL = (
    "https://github.com/openvax/mhcflurry/releases/download/2.3.0/" + WEIGHT_NAME
)
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite numeric value")
    return result


def positive_int(value):
    result = int(value)
    if result < 1:
        raise ValueError("Expected positive integer")
    return result


def read_inputs(path, task="presentation"):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        if not {"id", "peptide"} <= fields:
            raise ValueError("CSV requires id,peptide")
        if ("n_flank" in fields) != ("c_flank" in fields):
            raise ValueError("Provide both flank columns or neither")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty input CSV")
    ids = set()
    for row in rows:
        if not row["id"] or row["id"] in ids:
            raise ValueError("Empty or duplicate id: " + row["id"])
        ids.add(row["id"])
        peptide = row["peptide"]
        if not 8 <= len(peptide) <= 15 or not set(peptide) <= AMINO_ACIDS:
            raise ValueError("Expected canonical 8-15 residue peptide: " + peptide)
        for name in ("n_flank", "c_flank"):
            if name in fields and not set(row[name]) <= AMINO_ACIDS:
                raise ValueError("Noncanonical flank: " + name)
        if task != "processing":
            alleles = row["alleles"].split(";")
            if not 1 <= len(alleles) <= 6 or any(not item for item in alleles):
                raise ValueError("Expected 1-6 semicolon-separated alleles")
            if task == "affinity" and len(alleles) != 1:
                raise ValueError("Affinity task requires exactly one allele per row")
        if "label" in fields and row["label"] not in ("0", "1"):
            raise ValueError("Presentation label must be 0 or 1")
    return rows


def weights_inventory(directory):
    root = Path(directory)
    required = [
        root / "weights.csv",
        root / "affinity_predictor" / "manifest.csv",
        root / "processing_predictor_with_flanks" / "manifest.csv",
        root / "processing_predictor_without_flanks" / "manifest.csv",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Weight symlinks are not accepted")
        if path.is_file():
            files.append(
                {"path": path.relative_to(root).as_posix(), "sha256": sha256(path)}
            )
    if not any(item["path"].endswith(".npz") for item in files):
        raise ValueError("No NPZ model weights")
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return files, identity
