"""Normalize the pinned upstream Megascale test CSV without loading a model."""
import argparse
import csv
import hashlib
from pathlib import Path
import zipfile
from delivery_utils import finite, sha256, write_json

PDB_ARCHIVE_MD5 = 'c29317a75f28d008b2718316995902ef'


def prepare_structures(rows, archive, output):
    """Select required PDBs from the fixed Zenodo archive; never extract paths."""
    archive = Path(archive)
    if hashlib.md5(archive.read_bytes()).hexdigest() != PDB_ARCHIVE_MD5:
        raise ValueError('AlphaFold archive MD5 mismatch')
    names = sorted({row['pdb'] for row in rows})
    if any(Path(name).name != name or '/' in name or '\\' in name for name in names):
        raise ValueError('Invalid PDB filename')
    with zipfile.ZipFile(archive) as source:
        selected = {}
        for name in names:
            matches = [item for item in source.infolist()
                       if not item.is_dir() and item.filename.split('/')[-1] == name
                       and not item.filename.startswith('__MACOSX/')]
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one archived PDB: {name}; found {len(matches)}')
            data = source.read(matches[0])
            if not any(line.startswith(b'ATOM  ') for line in data.splitlines()):
                raise ValueError(f'No ATOM records: {name}')
            selected[name] = data
    output = Path(output)
    for name, data in selected.items():
        target = output / name
        if target.exists() and target.read_bytes() != data:
            raise ValueError(f'Existing PDB differs: {target}')
    output.mkdir(parents=True, exist_ok=True)
    for name, data in selected.items():
        target = output / name
        if not target.exists():
            with target.open('xb') as stream:
                stream.write(data)
    return {'archive_md5': PDB_ARCHIVE_MD5, 'archive_sha256': sha256(archive),
            'pdb_sha256': {name: sha256(output / name) for name in names}}


def normalize(path):
    with Path(path).open(newline='') as stream:
        raw_rows = list(csv.DictReader(stream))
    rows = []
    excluded = 0
    for index, raw in enumerate(raw_rows):
        mutation = raw['mut_type']
        if raw['ddG_ML'] == '-' or mutation == 'wt' or any(s in mutation for s in ('ins', 'del', ':')):
            excluded += 1
            continue
        wt, mut = mutation[0], mutation[-1]
        position = int(mutation[1:-1]) - 1
        seq = raw['wt_seq']
        if len(seq) != len(raw['aa_seq']) or seq[position] != wt or raw['aa_seq'][position] != mut:
            raise ValueError(f'Megascale sequence/mutation mismatch: row {index}')
        rows.append({'id': f'mega-test:{index}', 'pdb': raw['WT_name'].split('.pdb')[0].replace('|', ':') + '.pdb',
                     'chain': 'A', 'position': position, 'wildtype': wt, 'mutation': mut,
                     'ddG_true': -finite(raw['ddG_ML']), 'wt_sequence': seq})
    return rows, {'raw_rows': len(raw_rows), 'mutations': len(rows), 'excluded_rows': excluded,
                  'proteins': len({r['pdb'] for r in rows}), 'source_sha256': sha256(path),
                  'label_transform': '-ddG_ML (matches official MegaScaleDataset)',
                  'dataset': 'pinned upstream data_all/testing/mega_test.csv',
                  'official_split_verified': False,
                  'note': 'CSV inventory only; validate structures and original split before S3'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--output', help='Normalized CSV; omitted for inventory-only')
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--pdb-archive', help='Local Zenodo 7992926 AlphaFold_model_PDBs.zip')
    parser.add_argument('--pdb-dir', help='Destination for required structures; matching files reused')
    args = parser.parse_args()
    rows, inventory = normalize(args.csv)
    if bool(args.pdb_archive) != bool(args.pdb_dir):
        parser.error('--pdb-archive and --pdb-dir must be supplied together')
    if args.pdb_archive:
        inventory['structures'] = prepare_structures(rows, args.pdb_archive, args.pdb_dir)
    if args.output:
        with Path(args.output).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(args.inventory, inventory)
    print(inventory)


if __name__ == '__main__':
    main()
