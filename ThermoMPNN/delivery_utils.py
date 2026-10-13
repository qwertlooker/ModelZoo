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
    if 'wt_sequence' in row:
        seq = row['wt_sequence']
        if not isinstance(seq, str) or not seq or any(aa not in ALPHABET for aa in seq):
            raise ValueError('Invalid WT sequence')
        if row['position'] >= len(seq) or seq[row['position']] != row['wildtype']:
            raise ValueError(f'WT sequence/index mismatch: {row["id"]}')


def validate_protein(pdb, rows):
    """Validate selected chain and all mutations before changing official seq."""
    chain = rows[0]['chain']
    if len(pdb) != 1 or pdb[0]['num_of_chains'] != 1:
        raise ValueError('Exactly one parsed chain required')
    seq = pdb[0][f'seq_chain_{chain}']
    has_wt = ['wt_sequence' in row for row in rows]
    if any(has_wt):
        if not all(has_wt) or len({row['wt_sequence'] for row in rows}) != 1:
            raise ValueError('Inconsistent WT sequence within protein')
        seq = rows[0]['wt_sequence']
        if len(seq) != len(pdb[0][f'seq_chain_{chain}']):
            raise ValueError('Megascale WT sequence/structure length mismatch')
    coords = pdb[0][f'coords_chain_{chain}']
    for row in rows:
        validate_sample(row)
        if (row['chain'] != chain or row['position'] >= len(seq)
                or seq[row['position']] != row['wildtype']):
            raise ValueError(f'Wildtype/index mismatch: {row["id"]}')
        for atom in ('N', 'CA', 'C', 'O'):
            positions = coords[f'{atom}_chain_{chain}']
            if len(positions) != len(seq) or len(positions[row['position']]) != 3:
                raise ValueError(f'Invalid backbone coordinates: {row["id"]}/{atom}')
            for coordinate in positions[row['position']]:
                finite(coordinate)
    if any(has_wt):
        # Preserve seq_chain_A, as in the pinned official MegaScaleDataset.
        pdb[0]['seq'] = seq


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
