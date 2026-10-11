"""Freeze an explicitly curated single-chain mutation CSV into a manifest."""
import argparse
import csv
import json
from pathlib import Path
from delivery_utils import finite, sha256, validate_sample, write_json


def prepare(csv_path, pdb_dir, output, dataset, split):
    rows = []
    with Path(csv_path).open(newline='') as stream:
        for raw in csv.DictReader(stream):
            row = {key: raw[key] for key in ('id', 'pdb', 'chain', 'wildtype', 'mutation')}
            row['position'] = int(raw['position'])
            if 'ddG_true' in raw:
                row['ddG_true'] = finite(raw['ddG_true'])
            if 'wt_sequence' in raw:
                row['wt_sequence'] = raw['wt_sequence']
            validate_sample(row)
            root = Path(pdb_dir).resolve()
            pdb = (root / row['pdb']).resolve()
            if not pdb.is_relative_to(root):
                raise ValueError('PDB path escapes pdb-dir')
            if not any(line.startswith('ATOM  ') for line in pdb.read_text().splitlines()):
                raise ValueError(f'No ATOM records: {pdb}')
            row['pdb'] = str(pdb)
            row['pdb_sha256'] = sha256(pdb)
            rows.append(row)
    if not rows or any(not row['id'] for row in rows) or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty input or duplicate/empty IDs')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    write_json(str(output) + '.meta.json', {
        'dataset': dataset, 'split': split, 'samples': len(rows),
        'csv_sha256': sha256(csv_path), 'manifest_sha256': sha256(output),
        'position': 'zero-based parsed chain sequence index; NOT PDB residue number',
        'ddG_convention': 'kcal/mol; positive destabilizing',
        'official_split_verified': False,
    })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--pdb-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--split', required=True)
    args = parser.parse_args()
    prepare(args.csv, args.pdb_dir, args.output, args.dataset, args.split)


if __name__ == '__main__':
    main()
