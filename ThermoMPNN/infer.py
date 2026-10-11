"""ThermoMPNN FP32 single-chain inference; NPU is the explicit default."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parent / 'source'
sys.path.insert(0, str(SOURCE))
import numpy as np
import torch
from omegaconf import OmegaConf
from datasets import Mutation
from train_thermompnn import TransferModelPL
from protein_mpnn_utils import alt_parse_PDB
from delivery_utils import (BACKBONE_SHA, UPSTREAM_COMMIT, WEIGHT_SHA,
                            finite, read_jsonl, sha256, validate_sample, write_json)


def select_device(name):
    if name == 'npu':
        import torch_npu
        if not torch.npu.is_available():
            raise RuntimeError('NPU unavailable')
    elif name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    return torch.device(name)


def synchronize(name):
    if name == 'npu':
        torch.npu.synchronize()
    elif name == 'cuda':
        torch.cuda.synchronize()


def runtime_metadata(name):
    metadata = {'python': sys.version, 'torch': torch.__version__}
    if name == 'npu':
        import torch_npu
        metadata['torch_npu'] = torch_npu.__version__
        metadata['logical_device'] = torch.npu.current_device()
        metadata['ascend_home_path'] = os.environ.get('ASCEND_HOME_PATH')
    elif name == 'cuda':
        metadata['logical_device'] = torch.cuda.current_device()
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['npu', 'cpu', 'cuda'], default='npu')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--runs', type=int, default=3)
    args = parser.parse_args()
    if args.runs < 3 or args.warmup < 0:
        raise ValueError('At least 3 measured runs; nonnegative warmup')
    revision = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != UPSTREAM_COMMIT:
        raise ValueError('Upstream revision mismatch')
    subprocess.run(['git', '-C', str(SOURCE), 'diff', '--exit-code', 'HEAD'], check=True)
    if sha256(args.checkpoint) != WEIGHT_SHA:
        raise ValueError('Checkpoint SHA mismatch')
    backbone = SOURCE / 'vanilla_model_weights/v_48_020.pt'
    if sha256(backbone) != BACKBONE_SHA:
        raise ValueError('ProteinMPNN backbone SHA mismatch')
    samples = read_jsonl(args.manifest)
    groups = defaultdict(list)
    for row in samples:
        validate_sample(row)
        if sha256(row['pdb']) != row['pdb_sha256']:
            raise ValueError('PDB SHA mismatch')
        groups[row['pdb'], row['chain']].append(row)
    device = select_device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = OmegaConf.create({
        'platform': {'thermompnn_dir': str(SOURCE)},
        'training': {'learn_rate': 0.001, 'lr_schedule': True},
        'model': {'hidden_dims': [64, 32], 'subtract_mut': True,
                  'num_final_layers': 2, 'freeze_weights': True,
                  'load_pretrained': True, 'lightattn': True},
    })
    # Official Lightning checkpoint loader, trusted pinned checkpoint only.
    model = TransferModelPL.load_from_checkpoint(args.checkpoint, cfg=cfg, map_location='cpu').model
    model = model.eval().to(device)
    prepared = []
    for (pdb_path, chain), rows in groups.items():
        pdb = alt_parse_PDB(pdb_path, chain)
        seq = pdb[0][f'seq_chain_{chain}']
        if 'wt_sequence' in rows[0]:
            wt_seq = rows[0]['wt_sequence']
            if len(seq) != len(wt_seq) or any(r['wt_sequence'] != wt_seq for r in rows):
                raise ValueError('Megascale WT sequence/structure length mismatch')
            # Official MegaScaleDataset uses experimental WT sequence with AF coordinates.
            pdb[0]['seq'] = wt_seq
            pdb[0][f'seq_chain_{chain}'] = wt_seq
            seq = wt_seq
        for row in rows:
            if seq[row['position']] != row['wildtype']:
                raise ValueError(f'Wildtype mismatch: {row["id"]}')
        mutations = [Mutation(r['position'], r['wildtype'], r['mutation'], None, pdb[0]['name']) for r in rows]
        prepared.append((pdb, mutations, rows))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    durations = []
    predictions = []
    with torch.no_grad():
        for run in range(args.warmup + args.runs):
            synchronize(args.device)
            start = time.perf_counter()
            current = []
            for pdb, mutations, rows in prepared:
                outputs, _ = model(pdb, mutations)
                if len(outputs) != len(rows):
                    raise ValueError('Prediction count mismatch')
                for row, result in zip(rows, outputs):
                    if result['ddG'].device.type != args.device:
                        raise RuntimeError('Output device mismatch')
                    current.append({**row, 'ddG_pred': finite(result['ddG'].cpu().item())})
            synchronize(args.device)
            elapsed = time.perf_counter() - start
            if run >= args.warmup:
                durations.append(elapsed)
                predictions = current
    with (output / 'predictions.jsonl').open('x') as stream:
        for row in predictions:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    write_json(output / 'run.json', {
        **runtime_metadata(args.device),
        'schema': 1, 'simulated': False, 'device': str(device), 'dtype': 'float32',
        'command': sys.argv, 'upstream_commit': revision, 'torch': torch.__version__,
        'manifest_sha256': sha256(args.manifest), 'checkpoint_sha256': sha256(args.checkpoint),
        'backbone_sha256': sha256(backbone), 'predictions_sha256': sha256(output / 'predictions.jsonl'),
        'visible_devices': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
        'samples': len(samples), 'proteins': len(groups), 'runs': args.runs, 'warmup': args.warmup,
        'seconds': durations, 'median_seconds': statistics.median(durations),
        'mutations_per_second': len(samples) / statistics.median(durations),
        'timing_scope': 'model + upstream featurization + result transfer; excludes PDB parse/load',
    })


if __name__ == '__main__':
    main()
