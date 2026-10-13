# Licensed under the Apache License, Version 2.0.
"""Strict, dependency-free input and provenance helpers."""

import csv
import hashlib
import json
import math
import tarfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

SOURCE_REVISION = "20f03d45358dd104df9df0e28e2e0ccea11c9bfb"
WEIGHT_NAME = "models_class1_presentation.20260928.tar.bz2"
WEIGHT_SHA256 = "768e360832b3d45dfe9d5e8cde4ffcb90abdb87773ca65f3a86ba7cc20f614db"
WEIGHT_URL = (
    "https://github.com/openvax/mhcflurry/releases/download/2.3.0/" + WEIGHT_NAME
)
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
RESULT_COLUMNS = {
    "affinity": ("affinity",),
    "processing": ("processing_score",),
    "presentation": (
        "affinity",
        "processing_score",
        "presentation_score",
        "presentation_percentile",
    ),
}


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


def write_report(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(text)


def read_csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        fields = reader.fieldnames
        if not fields or any(not name.strip() for name in fields):
            raise ValueError("CSV requires nonempty column names")
        if len(set(fields)) != len(fields):
            raise ValueError("Duplicate CSV column names")
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(
                    "CSV row width differs from header at line %d" % reader.line_num
                )
            rows.append(row)
    if not rows:
        raise ValueError("Empty input CSV")
    return fields, rows


def validate_predictions(rows, task):
    ids = set()
    for row in rows:
        identity = row["id"]
        if not identity.strip() or identity in ids:
            raise ValueError("Empty or duplicate result id: " + identity)
        ids.add(identity)
        for column in RESULT_COLUMNS[task]:
            value = finite_float(row[column])
            if column == "affinity":
                valid = value > 0
            else:
                upper = 100 if column == "presentation_percentile" else 1
                valid = 0 <= value <= upper
            if not valid:
                raise ValueError("Out-of-range result for %s: %s" % (column, identity))
        if task == "presentation" and not row["best_allele"].strip():
            raise ValueError("Missing presentation best_allele")


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
    names, rows = read_csv_rows(path)
    fields = set(names)
    if not {"id", "peptide"} <= fields:
        raise ValueError("CSV requires id,peptide")
    if ("n_flank" in fields) != ("c_flank" in fields):
        raise ValueError("Provide both flank columns or neither")
    if task != "processing" and "alleles" not in fields:
        raise ValueError("Affinity/presentation CSV requires alleles")
    ids = set()
    for row in rows:
        if not row["id"].strip() or row["id"] in ids:
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
            if not 1 <= len(alleles) <= 6 or any(not item.strip() for item in alleles):
                raise ValueError("Expected 1-6 semicolon-separated alleles")
            if task == "affinity" and len(alleles) != 1:
                raise ValueError("Affinity task requires exactly one allele per row")
        if "sample_id" in fields and not (row["sample_id"] or "").strip():
            raise ValueError("Empty patient sample_id")
        if "label" in fields and row["label"] not in ("0", "1"):
            raise ValueError("Presentation label must be 0 or 1")
    return rows


def presentation_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        if not (row["sample_id"] or "").strip():
            raise ValueError("Empty patient sample_id")
        if row["label"] not in ("0", "1"):
            raise ValueError("Presentation label must be 0 or 1")
        groups[row["sample_id"]].append(row)
    for patient_rows in groups.values():
        if {row["label"] for row in patient_rows} != {"0", "1"}:
            raise ValueError("Each patient requires both labels")
    return groups


def weights_inventory(directory, archive):
    # Re-derive the trusted inventory from the pinned archive on every run.
    # A locally editable sidecar is not proof of official weight identity.
    if sha256(archive) != WEIGHT_SHA256:
        raise ValueError("Official archive SHA256 mismatch")
    expected = {}
    with tarfile.open(archive, "r:bz2") as stream:
        members = stream.getmembers()
        roots = [
            PurePosixPath(member.name).parent.parent
            for member in members
            if member.isfile()
            and PurePosixPath(member.name).parts[-2:]
            == ("affinity_predictor", "manifest.csv")
        ]
        if len(roots) != 1:
            raise ValueError("Expected exactly one presentation bundle in archive")
        for member in members:
            name = PurePosixPath(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError("Unsafe archive member: " + member.name)
            if member.isfile() and name.is_relative_to(roots[0]):
                relative = name.relative_to(roots[0]).as_posix()
                if relative in expected:
                    raise ValueError("Duplicate archive member: " + relative)
                with stream.extractfile(member) as source:
                    expected[relative] = hashlib.file_digest(
                        source, "sha256"
                    ).hexdigest()

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
    if {item["path"]: item["sha256"] for item in files} != expected:
        raise ValueError("Model directory differs from the fixed official archive")
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return files, identity
