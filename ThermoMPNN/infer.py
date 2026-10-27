"""ThermoMPNN FP32 single-chain inference; NPU is the explicit default."""
import argparse
from collections import defaultdict
import json
import math
import os
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import uuid

SOURCE = Path(__file__).resolve().parent / 'source'
sys.path.insert(0, str(SOURCE))
import numpy as np
import torch
from omegaconf import OmegaConf
from datasets import Mutation
from train_thermompnn import TransferModelPL
from protein_mpnn_utils import alt_parse_PDB, tied_featurize
from transfer_model import ALPHABET
from delivery_utils import (BACKBONE_SHA, UPSTREAM_COMMIT, WEIGHT_SHA,
                            finite, read_jsonl, sha256, validate_protein, validate_sample, write_json)


class _AccurateGELU(torch.nn.Module):
    """Exact GELU expressed with erf, avoiding the NPU GELU approximation."""

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def use_accurate_gelu(model):
    """Replace GELUs in this model instance only; no global/backend mutation."""
    for name, child in model.named_children():
        if isinstance(child, torch.nn.GELU):
            if child.approximate != 'none':
                raise ValueError('Expected exact upstream GELU')
            model.add_module(name, _AccurateGELU())
        else:
            use_accurate_gelu(child)


def light_attention_length1(model, lin_input):
    """Eval-only length-one LightAttention: zero padding leaves the center tap."""
    attention = model.light_attention
    convolution = attention.feature_convolution
    if attention.training or model.training:
        raise ValueError('Length-one optimization requires eval mode')
    if (lin_input.ndim != 2 or convolution.kernel_size != (9,)
            or convolution.padding != (4,) or convolution.stride != (1,)
            or convolution.dilation != (1,) or convolution.groups != 1
            or convolution.padding_mode != 'zeros'):
        raise ValueError('Unsupported LightAttention configuration')
    return torch.nn.functional.linear(lin_input, convolution.weight[:, :, 4], convolution.bias)


def _batched_head(model, lin_input, mutations):
    if model.lightattn:
        lin_input = light_attention_length1(model, lin_input)
    scores = model.ddg_out(model.both_out(lin_input).unsqueeze(-1)).squeeze(-1).squeeze(-1)
    aa = torch.tensor([ALPHABET.index(m.mutation) for m in mutations], device=scores.device)
    ddg = scores.gather(1, aa[:, None]).squeeze(1)
    if model.subtract_mut:
        wt = torch.tensor([ALPHABET.index(m.wildtype) for m in mutations], device=scores.device)
        ddg = ddg - scores.gather(1, wt[:, None]).squeeze(1)
    return ddg


def _forward_proteins(model, prepared):
    """Reuse upstream featurization/GNN, then gather all mutation vectors at once."""
    device = next(model.parameters()).device
    proteins = [pdb[0] for pdb, _, _ in prepared]
    lengths = [len(protein['seq']) for protein in proteins]
    indices = (0, 1, 2, 4, 5, 12)
    if len(set(lengths)) == 1:
        features = tied_featurize(proteins, device, None, None, None, None, None, None, ca_only=False)
        tensors = [features[index] for index in indices]
    else:
        # Pinned tied_featurize pads auxiliary 2-D arrays on BOTH axes for B>1.
        # Featurize separately, then pad only the six GNN inputs along length.
        maximum = max(lengths)
        columns = [[] for _ in indices]
        for protein, length in zip(proteins, lengths):
            features = tied_featurize([protein], device, None, None, None, None, None, None, ca_only=False)
            for column, index in zip(columns, indices):
                padding = (0, 0, 0, 0, 0, maximum - length) if index == 0 else (0, maximum - length)
                column.append(torch.nn.functional.pad(features[index], padding))
        tensors = [torch.cat(column, dim=0) for column in columns]
    X, S, mask, chain_M, chain_encoding_all, residue_idx = tensors
    # Upstream's implicit decoding_order has batch=1; gather does not broadcast.
    decoding_order = torch.arange(X.shape[1], device=device).unsqueeze(0).expand(X.shape[0], -1)
    hidden, embedding, _ = model.prot_mpnn(
        X, S, mask, chain_M, residue_idx, chain_encoding_all, None,
        use_input_decoding_order=True, decoding_order=decoding_order)
    mutations = [mutation for _, group, _ in prepared for mutation in group]
    protein_index = torch.tensor([i for i, (_, group, _) in enumerate(prepared) for _ in group],
                                 device=device)
    position = torch.tensor([mutation.position for mutation in mutations], device=device)
    inputs = []
    if model.num_final_layers > 0:
        mpnn_hidden = torch.cat(hidden[:model.num_final_layers], dim=-1)
        inputs.append(mpnn_hidden[protein_index, position])
    inputs.append(embedding[protein_index, position])
    ddg = _batched_head(model, torch.cat(inputs, dim=-1), mutations)
    return list(ddg.split([len(group) for _, group, _ in prepared]))


def batched_forward(model, pdb, mutations):
    """One protein, one head call; return a device tensor in mutation order."""
    return _forward_proteins(model, [(pdb, mutations, None)])[0]


def batched_forward_all(model, prepared):
    """Pad variable-length proteins and run one GNN and one mutation head."""
    return _forward_proteins(model, prepared)


def batched_forward_grouped(model, prepared):
    """Batch equal lengths only and restore the original protein order."""
    buckets = defaultdict(list)
    for index, item in enumerate(prepared):
        buckets[len(item[0][0]['seq'])].append((index, item))
    outputs = [None] * len(prepared)
    for bucket in buckets.values():
        values = _forward_proteins(model, [item for _, item in bucket])
        for (index, _), value in zip(bucket, values):
            outputs[index] = value
    return outputs


def validate_source():
    revision = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != UPSTREAM_COMMIT:
        raise ValueError('Upstream revision mismatch')
    patch_file = SOURCE.parent / 'diff.patch'
    if not patch_file.is_file():
        subprocess.run(['git', '-C', str(SOURCE), 'diff', '--exit-code', 'HEAD'], check=True)
        return revision, None
    filename = 'protein_mpnn_utils.py'
    subprocess.run(['git', '-C', str(SOURCE), 'diff', '--exit-code', 'HEAD', '--',
                    f':(exclude){filename}'], check=True)
    subprocess.run(['git', '-C', str(SOURCE), 'apply', '--reverse', '--check', str(patch_file)], check=True)
    # Reverse-check alone accepts unrelated changes elsewhere in the same file.
    baseline = subprocess.check_output(['git', '-C', str(SOURCE), 'show', f'HEAD:{filename}'])
    with tempfile.TemporaryDirectory(prefix='thermompnn-source-') as directory:
        expected = Path(directory) / filename
        expected.write_bytes(baseline)
        subprocess.run(['git', '-C', directory, 'apply', str(patch_file)], check=True)
        if expected.read_bytes() != (SOURCE / filename).read_bytes():
            raise ValueError('Upstream source differs from the exact approved patch')
    return revision, sha256(patch_file)


def validate_coordinate_span(pdb, chain):
    """Bound all valid CA distances below the patch's 1e4 padding sentinel."""
    coords = pdb[0][f'coords_chain_{chain}']
    backbone = np.stack([np.asarray(coords[f'{atom}_chain_{chain}']) for atom in ('N', 'CA', 'C', 'O')], axis=1)
    valid_ca = backbone[np.isfinite(backbone).all(axis=(1, 2)), 1]
    if len(valid_ca) and np.linalg.norm(np.ptp(valid_ca, axis=0)) >= 1e4:
        raise ValueError('Protein coordinate span exceeds the padding distance bound')


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
        metadata['cann_version'] = torch.version.cann
        metadata['device_name'] = torch.npu.get_device_name(metadata['logical_device'])
        metadata['hostname'] = platform.node()
        for key in ('ascend_home_path', 'cann_version', 'device_name', 'hostname'):
            if not isinstance(metadata[key], str) or metadata[key].strip() in ('', 'not known', 'N/A'):
                raise ValueError(f'Missing NPU environment evidence: {key}')
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
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--upstream-head', action='store_true', help='Per-protein upstream mutation loop (GELU transform still applied)')
    modes.add_argument('--gnn-bucket', action='store_true', help='Batch equal-length proteins without length padding')
    modes.add_argument('--per-protein', dest='batch_all', action='store_false', help='One protein per GNN; batch mutation heads')
    parser.set_defaults(batch_all=True)
    args = parser.parse_args()
    if args.upstream_head or args.gnn_bucket:
        args.batch_all = False
    if args.runs < 3 or args.warmup < 0:
        raise ValueError('At least 3 measured runs; nonnegative warmup')
    revision, patch_sha = validate_source()
    if args.batch_all and patch_sha is None:
        raise ValueError('Full batching requires diff.patch and the patched upstream source')
    if sha256(args.checkpoint) != WEIGHT_SHA:
        raise ValueError('Checkpoint SHA mismatch')
    backbone = SOURCE / 'vanilla_model_weights/v_48_020.pt'
    if sha256(backbone) != BACKBONE_SHA:
        raise ValueError('ProteinMPNN backbone SHA mismatch')
    samples = read_jsonl(args.manifest)
    groups = defaultdict(list)
    pdb_hashes = {}
    for row in samples:
        validate_sample(row)
        if row['pdb'] not in pdb_hashes:
            pdb_hashes[row['pdb']] = sha256(row['pdb'])
        if pdb_hashes[row['pdb']] != row['pdb_sha256']:
            raise ValueError('PDB SHA mismatch')
        groups[row['pdb'], row['chain']].append(row)
    device = select_device(args.device)
    environment = runtime_metadata(args.device)
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
    use_accurate_gelu(model)
    prepared = []
    for (pdb_path, chain), rows in groups.items():
        pdb = alt_parse_PDB(pdb_path, chain)
        validate_protein(pdb, rows)
        validate_coordinate_span(pdb, chain)
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
            if args.batch_all:
                grouped = batched_forward_all(model, prepared)
            elif args.gnn_bucket:
                grouped = batched_forward_grouped(model, prepared)
            else:
                grouped = None
            for index, (pdb, mutations, rows) in enumerate(prepared):
                if grouped is not None:
                    ddg = grouped[index]
                elif args.upstream_head:
                    outputs, _ = model(pdb, mutations)
                    ddg = torch.cat([result['ddG'].reshape(-1) for result in outputs])
                else:
                    ddg = batched_forward(model, pdb, mutations)
                if ddg.ndim != 1 or ddg.numel() != len(rows):
                    raise ValueError('Prediction count mismatch')
                if ddg.device.type != args.device:
                    raise RuntimeError('Output device mismatch')
                current.append((rows, [finite(value) for value in ddg.cpu().tolist()]))
            synchronize(args.device)
            elapsed = time.perf_counter() - start
            if run >= args.warmup:
                durations.append(elapsed)
                predictions = current
    with (output / 'predictions.jsonl').open('x') as stream:
        for rows, values in predictions:
            for row, value in zip(rows, values):
                stream.write(json.dumps({**row, 'ddG_pred': value}, allow_nan=False) + '\n')
    write_json(output / 'run.json', {
        **environment,
        'schema': 3, 'run_id': str(uuid.uuid4()), 'simulated': False, 'device': str(device), 'dtype': 'float32',
        'command': sys.argv, 'upstream_commit': revision, 'torch': torch.__version__,
        'patch_sha256': patch_sha, 'infer_sha256': sha256(__file__),
        'inference_mode': ('batch-all' if args.batch_all else 'gnn-bucket' if args.gnn_bucket
                           else 'upstream-head' if args.upstream_head else 'per-protein'),
        'gelu': 'exact-erf',
        'manifest_sha256': sha256(args.manifest), 'checkpoint_sha256': sha256(args.checkpoint),
        'backbone_sha256': sha256(backbone), 'predictions_sha256': sha256(output / 'predictions.jsonl'),
        'visible_devices': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
        'samples': len(samples), 'proteins': len(groups), 'runs': args.runs, 'warmup': args.warmup,
        'seconds': durations, 'median_seconds': statistics.median(durations),
        'mutations_per_second': len(samples) / statistics.median(durations),
        'timing_scope': 'model + upstream featurization + batched result transfer/finite check; excludes PDB parse/load, row merge and file writes',
    })


if __name__ == '__main__':
    main()
