"""Normalize the pinned upstream Megascale test CSV without loading a model."""
import argparse
import csv
from pathlib import Path
from delivery_utils import finite, sha256, write_json


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
    args = parser.parse_args()
    rows, inventory = normalize(args.csv)
    if args.output:
        with Path(args.output).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(args.inventory, inventory)
    print(inventory)


if __name__ == '__main__':
    main()
