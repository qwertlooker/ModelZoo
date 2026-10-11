"""Offline regression. Mocks are not model or NPU acceptance evidence."""
import ast
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import statistics
from collections import defaultdict
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from delivery_utils import read_jsonl, sha256, validate_sample, write_json
from prepare_eval_data import prepare
from compare_results import compare
from report_acceptance import collect
from prepare_megascale import normalize


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.pdb = self.root / 'demo.pdb'
        self.pdb.write_text('ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n')
        self.row = {'id': 'demo:A:0:C', 'pdb': str(self.pdb), 'pdb_sha256': sha256(self.pdb),
                    'chain': 'A', 'position': 0, 'wildtype': 'A', 'mutation': 'C', 'ddG_pred': 1.0}

    def tearDown(self):
        self.tmp.cleanup()

    def jsonl(self, name, rows):
        path = self.root / name
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return path

    def test_pair_reordering(self):
        other = {**self.row, 'id': 'second'}
        self.assertTrue(compare(self.jsonl('a', [self.row, other]),
                                self.jsonl('b', [other, self.row]), 0, 0)['passed'])

    def test_threshold_failure(self):
        a = self.jsonl('a', [self.row])
        b = self.jsonl('b', [{**self.row, 'ddG_pred': 2}])
        self.assertFalse(compare(a, b, 0.001, 0.001)['passed'])
        proc = subprocess.run([sys.executable, str(ROOT / 'compare_results.py'),
                               '--baseline', str(a), '--candidate', str(b), '--atol', '0.001',
                               '--rtol', '0.001', '--report', str(self.root / 'report.json')], capture_output=True)
        self.assertEqual(proc.returncode, 1)

    def test_identity_mismatch(self):
        for field, value in [('position', 1), ('chain', 'B'), ('pdb_sha256', 'changed'),
                             ('wildtype', 'C'), ('mutation', 'D'), ('id', 'other')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare(self.jsonl('a', [self.row]),
                        self.jsonl('b', [{**self.row, field: value}]), 0, 0)

    def test_nonfinite_and_negative_tolerance(self):
        a = self.jsonl('a', [self.row])
        for value in [float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                compare(a, self.jsonl('b', [{**self.row, 'ddG_pred': value}]), 0, 0)
        with self.assertRaises(ValueError):
            compare(a, a, -1, 0)

    def test_duplicate_and_empty(self):
        for rows in [[], [self.row, self.row]]:
            with self.assertRaises(ValueError):
                read_jsonl(self.jsonl('bad', rows))

    def test_invalid_sample(self):
        for change in [{'position': -1}, {'position': True}, {'wildtype': 'X'},
                       {'mutation': 'CC'}, {'chain': 'AB'}, {'ddG_true': 'nan'}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_sample({**self.row, **change})

    def test_prepare_hash_and_no_overwrite(self):
        csv = self.root / 'input.csv'
        csv.write_text('id,pdb,chain,position,wildtype,mutation,ddG_true\na,demo.pdb,A,0,A,C,1\n')
        output = self.root / 'manifest.jsonl'
        rows = prepare(csv, self.root, output, 'fixture', 'mock')
        self.assertEqual(rows[0]['pdb_sha256'], sha256(self.pdb))
        meta = json.loads(Path(str(output) + '.meta.json').read_text())
        self.assertFalse(meta['official_split_verified'])
        self.assertEqual(meta['samples'], 1)
        with self.assertRaises(FileExistsError):
            prepare(csv, self.root, output, 'fixture', 'mock')

    def test_prepare_path_escape(self):
        csv = self.root / 'input.csv'
        csv.write_text('id,pdb,chain,position,wildtype,mutation\na,../outside.pdb,A,0,A,C\n')
        with self.assertRaises(ValueError):
            prepare(csv, self.root, self.root / 'manifest', 'fixture', 'mock')

    def test_megascale_sign_index_filter(self):
        path = self.root / 'mega.csv'
        path.write_text('mut_type,ddG_ML,wt_seq,aa_seq,WT_name\nwt,0,AC,AC,demo.pdb\nA1C,2,AC,CC,demo.pdb\nA1C,-,AC,CC,demo.pdb\n')
        rows, meta = normalize(path)
        self.assertEqual(rows[0]['position'], 0)
        self.assertEqual(rows[0]['ddG_true'], -2)
        self.assertEqual(meta['excluded_rows'], 2)
        self.assertEqual(meta['mutations'], 1)
        path.write_text('mut_type,ddG_ML,wt_seq,aa_seq,WT_name\nC1A,2,AC,CC,demo.pdb\n')
        with self.assertRaises(ValueError):
            normalize(path)

    def test_mock_device_routes_and_unavailable(self):
        # Execute the actual device helpers without importing model dependencies.
        tree = ast.parse((ROOT / 'infer.py').read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in ('select_device', 'synchronize')]
        fake_torch = Mock()
        fake_torch.device.side_effect = lambda value: value
        namespace = {'torch': fake_torch}
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'infer.py', 'exec'), namespace)
        with patch.dict(sys.modules, {'torch_npu': types.ModuleType('torch_npu')}):
            for name in ('npu', 'cpu', 'cuda'):
                self.assertEqual(namespace['select_device'](name), name)
                namespace['synchronize'](name)
            fake_torch.npu.synchronize.assert_called_once()
            fake_torch.cuda.synchronize.assert_called_once()
            fake_torch.npu.is_available.return_value = False
            with self.assertRaises(RuntimeError):
                namespace['select_device']('npu')
            fake_torch.cuda.is_available.return_value = False
            with self.assertRaises(RuntimeError):
                namespace['select_device']('cuda')

    def test_mock_inference_orchestration(self):
        tree = ast.parse((ROOT / 'infer.py').read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        import argparse
        source = self.root / 'source'
        (source / 'vanilla_model_weights').mkdir(parents=True)
        backbone = source / 'vanilla_model_weights/v_48_020.pt'
        backbone.write_bytes(b'fixture backbone')
        checkpoint = self.root / 'checkpoint.pt'
        checkpoint.write_bytes(b'fixture checkpoint')
        manifest = self.jsonl('manifest', [self.row])
        output = self.root / 'outputs'
        class FakeDevice:
            def __init__(self, name):
                self.type = name

            def __str__(self):
                return self.type

        fake_torch = Mock()
        fake_torch.__version__ = 'mock-only'
        fake_torch.npu.current_device.return_value = 0
        fake_torch.device.side_effect = FakeDevice
        fake_torch.no_grad.side_effect = nullcontext
        tensor = Mock()
        tensor.device.type = 'npu'
        tensor.cpu.return_value.item.return_value = 1.25
        model = Mock(return_value=([{'ddG': tensor}], None))
        model.eval.return_value.to.return_value = model
        loader = Mock()
        loader.load_from_checkpoint.return_value.model = model
        fake_process = Mock()
        fake_process.check_output.return_value = 'pinned\n'
        from delivery_utils import finite
        namespace = {'argparse': argparse, 'defaultdict': defaultdict, 'json': json,
                     'os': __import__('os'), 'Path': Path, 'statistics': statistics,
                     'subprocess': fake_process, 'sys': sys, 'time': time, 'SOURCE': source,
                     'torch': fake_torch, 'np': Mock(), 'OmegaConf': Mock(), 'TransferModelPL': loader,
                     'Mutation': lambda *args: args,
                     'alt_parse_PDB': Mock(return_value=[{'seq_chain_A': 'A', 'seq': 'A', 'name': 'demo'}]),
                     'UPSTREAM_COMMIT': 'pinned', 'WEIGHT_SHA': sha256(checkpoint),
                     'BACKBONE_SHA': sha256(backbone), 'sha256': sha256, 'finite': finite,
                     'read_jsonl': read_jsonl, 'validate_sample': validate_sample, 'write_json': write_json}
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'infer.py', 'exec'), namespace)
        argv = ['infer.py', '--manifest', str(manifest), '--checkpoint', str(checkpoint), '--output-dir', str(output)]
        backend = types.ModuleType('torch_npu')
        backend.__version__ = 'mock-only'
        with patch.object(sys, 'argv', argv), patch.dict(sys.modules, {'torch_npu': backend}):
            namespace['main']()
        self.assertEqual(model.call_count, 4)  # one warmup + three measured runs
        self.assertEqual(read_jsonl(output / 'predictions.jsonl')[0]['ddG_pred'], 1.25)
        metadata = json.loads((output / 'run.json').read_text())
        self.assertEqual(metadata['runs'], 3)
        self.assertEqual(metadata['device'], 'npu')
        self.assertEqual(metadata['torch_npu'], 'mock-only')
        self.assertEqual(model.eval.return_value.to.call_args.args[0].type, 'npu')
        self.assertEqual(loader.load_from_checkpoint.call_args.kwargs['map_location'], 'cpu')

    def test_report_rejects_mock_cpu_and_tampering(self):
        predictions = self.jsonl('predictions.jsonl', [self.row])
        digest = sha256(predictions)
        run = {'device': 'npu', 'simulated': False, 'runs': 3, 'seconds': [1, 2, 3],
               'samples': 1, 'predictions_sha256': digest}
        comparison = {'passed': True, 'samples': 1, 'candidate_sha256': digest}
        metrics = {'samples': 1, 'predictions_sha256': digest}
        c, m = self.root / 'compare.json', self.root / 'metrics.json'
        write_json(c, comparison)
        write_json(m, metrics)
        write_json(self.root / 'run.json', run)
        self.assertIn('S3 requires', collect(self.root, c, m)['status'])
        for change in [{'simulated': True}, {'device': 'cpu'}, {'runs': 1},
                       {'seconds': [0, 2, 3]}, {'predictions_sha256': 'bad'}, {'samples': 2}]:
            write_json(self.root / 'run.json', {**run, **change})
            with self.subTest(change=change), self.assertRaises(ValueError):
                collect(self.root, c, m)


if __name__ == '__main__':
    unittest.main()
