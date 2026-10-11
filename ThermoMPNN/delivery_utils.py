"""Strict, dependency-free manifest and result validation."""
import hashlib
import json
import math
from pathlib import Path

ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
UPSTREAM_COMMIT = '370f76ec62bd929f7425e311d8df04a0d094990f'
WEIGHT_SHA = 'af449118ccfb4e34971d802321907ed82a8a035ae80f055bf8fc56009a4a838a'
BACKBONE_SHA = 'c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if not rows:
        raise ValueError('Empty input')
    ids = [row['id'] for row in rows]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError('IDs must be nonempty strings')
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate IDs')
    return rows


def finite(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Nonfinite number')
    return result


def validate_sample(row):
    if type(row['position']) is not int or row['position'] < 0:
        raise ValueError('position must be a zero-based nonnegative integer')
    for key in ('wildtype', 'mutation'):
        if row[key] not in ALPHABET or len(row[key]) != 1:
            raise ValueError(f'Invalid {key}')
    if not row['chain'] or len(row['chain']) != 1:
        raise ValueError('Single PDB chain required')
    if 'ddG_true' in row:
        finite(row['ddG_true'])


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
