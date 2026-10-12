"""Independent CPU baseline using the pinned official MegaScaleDataset/model."""
import argparse
from collections import defaultdict, deque
import json
import math
from pathlib import Path
import subprocess
import sys

SOURCE = Path(__file__).resolve().parent / 'source'
sys.path.insert(0, str(SOURCE))
import numpy as np
import torch
from omegaconf import OmegaConf
from datasets import MegaScaleDataset
from train_thermompnn import TransferModelPL
from delivery_utils import (BACKBONE_SHA, UPSTREAM_COMMIT, WEIGHT_SHA,
                            finite, read_jsonl, sha256, validate_sample, write_json)

CSV_SHA = '3adaa13aae93a575704e50cbb9bec4b84516789558204124dacf43925fb9b68f'
SPLITS_SHA = '9e06230fe8febd8f07f8fab374f153328d4bdde20b78de15ead93b39e61ca1eb'


def align_mutations(pdb, mutations, rows):
    """Bind original dataset outputs to IDs; do not construct model inputs here."""
    if len(pdb) != 1 or pdb[0]['num_of_chains'] != 1 or 'seq_chain_A' not in pdb[0]:
        raise ValueError('Official dataset requires a single chain A structure')
    queues = defaultdict(deque)
    for row in rows:
        if row['chain'] != 'A' or row['wt_sequence'] != pdb[0]['seq']:
            raise ValueError('Official WT sequence/chain differs from manifest')
        queues[row['position'], row['wildtype'], row['mutation']].append(row)
    aligned = []
    for mutation in mutations:
        key = (mutation.position, mutation.wildtype, mutation.mutation)
        if not queues[key]:
            raise ValueError(f'Official mutation absent from manifest: {key}')
        row = queues[key].popleft()
        # The official dataset materializes labels as float32 tensors.
        if not math.isclose(finite(row['ddG_true']), finite(mutation.ddG.item()),
                            rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError(f'Official label differs: {row["id"]}')
        aligned.append(row)
    if any(queues.values()):
        raise ValueError('Manifest contains mutations absent from official dataset')
    return aligned


def run_baseline(dataset, model, groups):
    predictions = []
    seen = set()
    with torch.no_grad():
        for index, name in enumerate(dataset.wt_names):
            filename = name.split('.pdb')[0].replace('|', ':') + '.pdb'
            if filename in seen or filename not in groups:
                raise ValueError(f'Official protein differs from manifest: {filename}')
            seen.add(filename)
            # No infer.py preparation helpers or normalized CSV are used.
            pdb, mutations = dataset[index]
            rows = align_mutations(pdb, mutations, groups[filename])
            outputs, _ = model(pdb, mutations)
            if len(outputs) != len(rows):
                raise ValueError('Official prediction count mismatch')
            for row, result in zip(rows, outputs):
                if result['ddG'].device.type != 'cpu':
                    raise RuntimeError('Official baseline output must be on CPU')
                predictions.append({**row, 'ddG_pred': finite(result['ddG'].item())})
    if seen != set(groups):
        raise ValueError('Manifest protein set differs from official test split')
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--pdb-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != UPSTREAM_COMMIT:
        raise ValueError('Upstream revision mismatch')
    subprocess.run(['git', '-C', str(SOURCE), 'diff', '--exit-code', 'HEAD'], check=True)
    csv_path = SOURCE / 'data_all/testing/mega_test.csv'
    splits = SOURCE / 'dataset_splits/mega_splits.pkl'
    backbone = SOURCE / 'vanilla_model_weights/v_48_020.pt'
    for path, digest in ((csv_path, CSV_SHA), (splits, SPLITS_SHA),
                         (backbone, BACKBONE_SHA), (args.checkpoint, WEIGHT_SHA)):
        if sha256(path) != digest:
            raise ValueError(f'Pinned artifact SHA mismatch: {path}')
    samples = read_jsonl(args.manifest)
    pdb_dir = Path(args.pdb_dir).resolve()
    groups = defaultdict(list)
    for row in samples:
        validate_sample(row)
        pdb_path = Path(row['pdb']).resolve()
        if pdb_path.parent != pdb_dir or sha256(pdb_path) != row['pdb_sha256']:
            raise ValueError('Manifest structure path/SHA differs from official pdb-dir')
        groups[pdb_path.name].append(row)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Use official config; only replace environment/data locations and disable reduction.
    cfg = OmegaConf.load(SOURCE / 'config.yaml')
    cfg.platform = {'thermompnn_dir': str(SOURCE), 'cache_dir': str(output / 'cache')}
    cfg.data_loc = {'megascale_csv': str(csv_path), 'megascale_splits': str(splits),
                    'megascale_pdbs': str(pdb_dir)}
    cfg.reduce = ''
    torch.manual_seed(0)
    np.random.seed(0)
    dataset = MegaScaleDataset(cfg, 'test')
    model = TransferModelPL.load_from_checkpoint(args.checkpoint, cfg=cfg, map_location='cpu').model
    model = model.eval().to('cpu')
    predictions = run_baseline(dataset, model, groups)
    path = output / 'predictions.jsonl'
    with path.open('x') as stream:
        for row in predictions:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    write_json(output / 'baseline.json', {
        'device': 'cpu', 'simulated': False, 'python': sys.version, 'torch': torch.__version__,
        'command': sys.argv, 'upstream_commit': revision, 'dataset': 'MegaScaleDataset',
        'split': 'test', 'samples': len(predictions), 'proteins': len(groups),
        'csv_sha256': sha256(csv_path), 'splits_sha256': sha256(splits),
        'config_sha256': sha256(SOURCE / 'config.yaml'),
        'manifest_sha256': sha256(args.manifest), 'checkpoint_sha256': sha256(args.checkpoint),
        'backbone_sha256': sha256(backbone), 'predictions_sha256': sha256(path),
        'purpose': 'Official CPU accuracy baseline; not NPU or performance acceptance',
    })


if __name__ == '__main__':
    main()
