"""Offline regression. Mocks are not model or NPU acceptance evidence."""
import ast
import hashlib
import uuid
import zipfile
from contextlib import nullcontext
import json
import platform
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
from delivery_utils import read_jsonl, sha256, validate_protein, validate_sample, write_json
from prepare_eval_data import prepare
from compare_results import compare
from report_acceptance import collect, performance
from prepare_megascale import normalize, prepare_structures


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

    def run_record(self, directory, seconds, samples=1, proteins=1):
        median = statistics.median(seconds)
        return {'schema': 3, 'run_id': str(uuid.uuid4()), 'device': 'npu', 'simulated': False,
                'runs': len(seconds), 'warmup': 1, 'seconds': seconds,
                'median_seconds': median, 'mutations_per_second': samples / median,
                'samples': samples, 'proteins': proteins, 'dtype': 'float32', 'timing_scope': 'fixture',
                'infer_sha256': 'fixture-infer', 'patch_sha256': 'fixture-patch',
                'inference_mode': 'batch-all', 'gelu': 'exact-erf',
                'manifest_sha256': 'manifest', 'checkpoint_sha256': 'checkpoint',
                'backbone_sha256': 'backbone', 'upstream_commit': 'source',
                'torch': 'fixture', 'torch_npu': 'fixture', 'visible_devices': '0',
                'python': '3.10 fixture', 'logical_device': 0, 'ascend_home_path': '/fixture/cann',
                'cann_version': '8.3.RC1', 'device_name': 'Ascend910B', 'hostname': 'fixture-host',
                'predictions_sha256': sha256(directory / 'predictions.jsonl')}

    def quality_metrics(self, samples, digest):
        return {'samples': samples, 'predictions_sha256': digest, 'mse': 1.0, 'rmse': 1.0,
                'r2': 0.0, 'pearson': 0.5, 'spearman': 0.5}

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

    def test_wt_sequence_bounds_and_identity(self):
        for change in [{'wt_sequence': ''}, {'wt_sequence': 'X'},
                       {'wt_sequence': 'C'}, {'wt_sequence': 'A', 'position': 1}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_sample({**self.row, **change})

    def test_parsed_protein_rejects_mixed_sequences_bounds_and_missing_atoms(self):
        pdb = [{'seq_chain_A': 'M', 'seq': 'M', 'num_of_chains': 1,
                'coords_chain_A': {f'{atom}_chain_A': [[0, 1, 2]] for atom in ('N', 'CA', 'C', 'O')}}]
        for rows in [[self.row, {**self.row, 'wt_sequence': 'A'}],
                     [{**self.row, 'wt_sequence': 'A'}, self.row],
                     [{**self.row, 'position': 1}], [{**self.row, 'chain': 'B'}]]:
            with self.subTest(rows=rows), self.assertRaises((ValueError, KeyError)):
                validate_protein(pdb, rows)
        validate_protein(pdb, [{**self.row, 'wt_sequence': 'A'}])
        self.assertEqual(pdb[0]['seq'], 'A')
        self.assertEqual(pdb[0]['seq_chain_A'], 'M')
        pdb[0]['coords_chain_A']['CA_chain_A'][0][0] = float('nan')
        with self.assertRaises(ValueError):
            validate_protein(pdb, [{**self.row, 'wt_sequence': 'A'}])

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

    def test_prepare_reads_each_structure_once(self):
        csv = self.root / 'input.csv'
        csv.write_text('id,pdb,chain,position,wildtype,mutation\na,demo.pdb,A,0,A,C\nb,demo.pdb,A,0,A,D\n')
        with patch('prepare_eval_data.sha256', wraps=sha256) as digest:
            rows = prepare(csv, self.root, self.root / 'manifest', 'fixture', 'mock')
        self.assertEqual(len(rows), 2)
        self.assertEqual(sum(Path(call.args[0]) == self.pdb for call in digest.call_args_list), 1)

    def test_megascale_rejects_invalid_single_mutations(self):
        path = self.root / 'invalid.csv'
        for mutation, wt_seq, aa_seq in [('A0C', 'AA', 'AC'), ('A3C', 'AA', 'AA'),
                                         ('A1C', 'AA', 'CC'), ('A1X', 'AA', 'XA'),
                                         ('A1A', 'AA', 'AA')]:
            path.write_text(f'mut_type,ddG_ML,wt_seq,aa_seq,WT_name\n{mutation},1,{wt_seq},{aa_seq},demo.pdb\n')
            with self.subTest(mutation=mutation, aa_seq=aa_seq), self.assertRaises(ValueError):
                normalize(path)
        path.write_text('mut_type,ddG_ML,wt_seq,aa_seq,WT_name\nwt,0,AA,AA,demo.pdb\n')
        with self.assertRaises(ValueError):
            normalize(path)

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
        manifest = self.jsonl('manifest', [{**self.row, 'wt_sequence': 'A'}])
        output = self.root / 'outputs'
        class FakeDevice:
            def __init__(self, name):
                self.type = name

            def __str__(self):
                return self.type

        fake_torch = Mock()
        fake_torch.__version__ = 'mock-only'
        fake_torch.npu.current_device.return_value = 0
        fake_torch.npu.get_device_name.return_value = 'Ascend910B'
        fake_torch.version.cann = '8.3.RC1'
        fake_torch.device.side_effect = FakeDevice
        fake_torch.no_grad.side_effect = nullcontext
        tensor = Mock()
        tensor.device.type = 'npu'
        tensor.ndim = 1
        tensor.numel.return_value = 1
        tensor.cpu.return_value.tolist.return_value = [1.25]
        fake_torch.cat.return_value = tensor
        model = Mock(return_value=([{'ddG': tensor}], None))
        model.eval.return_value.to.return_value = model
        loader = Mock()
        loader.load_from_checkpoint.return_value.model = model
        fake_process = Mock()
        fake_process.check_output.return_value = 'pinned\n'
        from delivery_utils import finite
        namespace = {'argparse': argparse, 'defaultdict': defaultdict, 'json': json,
                     'os': __import__('os'), 'Path': Path, 'statistics': statistics,
                     'uuid': uuid, 'platform': platform, 'subprocess': fake_process, 'sys': sys, 'time': time, 'SOURCE': source,
                     '__file__': str(ROOT / 'infer.py'),
                     'torch': fake_torch, 'np': Mock(), 'OmegaConf': Mock(), 'TransferModelPL': loader,
                     'Mutation': lambda *args: args,
                     'alt_parse_PDB': Mock(return_value=[{'seq_chain_A': 'M', 'seq': 'M', 'name': 'demo',
                         'num_of_chains': 1, 'coords_chain_A': {
                             f'{atom}_chain_A': [[0, 1, 2]] for atom in ('N', 'CA', 'C', 'O')}}]),
                     'UPSTREAM_COMMIT': 'pinned', 'WEIGHT_SHA': sha256(checkpoint),
                     'BACKBONE_SHA': sha256(backbone), 'sha256': sha256, 'finite': finite,
                     'read_jsonl': read_jsonl, 'validate_protein': validate_protein,
                     'validate_sample': validate_sample, 'write_json': write_json}
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'infer.py', 'exec'), namespace)
        namespace['use_accurate_gelu'] = Mock()
        namespace['validate_coordinate_span'] = Mock()
        argv = ['infer.py', '--upstream-head', '--manifest', str(manifest), '--checkpoint', str(checkpoint), '--output-dir', str(output)]
        backend = types.ModuleType('torch_npu')
        backend.__version__ = 'mock-only'
        with patch.object(sys, 'argv', argv), patch.dict(sys.modules, {'torch_npu': backend}), \
                patch.dict(__import__('os').environ, {'ASCEND_HOME_PATH': '/fixture/cann'}):
            namespace['main']()
        self.assertEqual(model.call_args.args[0][0]['seq_chain_A'], 'M')
        self.assertEqual(model.call_args.args[0][0]['seq'], 'A')
        self.assertTrue(metadata_id := json.loads((output / 'run.json').read_text())['run_id'])
        uuid.UUID(metadata_id)
        self.assertEqual(model.call_count, 4)  # one warmup + three measured runs
        self.assertEqual(read_jsonl(output / 'predictions.jsonl')[0]['ddG_pred'], 1.25)
        metadata = json.loads((output / 'run.json').read_text())
        self.assertEqual(metadata['runs'], 3)
        self.assertEqual(metadata['device'], 'npu')
        self.assertEqual(metadata['torch_npu'], 'mock-only')
        self.assertEqual(metadata['cann_version'], '8.3.RC1')
        self.assertEqual(metadata['device_name'], 'Ascend910B')
        self.assertEqual(metadata['schema'], 3)
        self.assertEqual(model.eval.return_value.to.call_args.args[0].type, 'npu')
        self.assertEqual(loader.load_from_checkpoint.call_args.kwargs['map_location'], 'cpu')

    def test_report_rejects_mock_cpu_and_tampering(self):
        predictions = self.jsonl('predictions.jsonl', [self.row])
        digest = sha256(predictions)
        run = self.run_record(self.root, [1, 2, 3])
        comparison = {'passed': True, 'samples': 1, 'candidate_sha256': digest}
        metrics = self.quality_metrics(1, digest)
        c, m = self.root / 'compare.json', self.root / 'metrics.json'
        write_json(c, comparison)
        write_json(m, metrics)
        write_json(self.root / 'run.json', run)
        self.assertIn('S3 requires', collect(self.root, c, m)['status'])
        self.assertTrue(any('independent-process' in item for item in collect(self.root, c, m)['pending']))
        for change in [{'simulated': True}, {'device': 'cpu'}, {'runs': 1},
                       {'seconds': [0, 2, 3]}, {'predictions_sha256': 'bad'}, {'samples': 2},
                       {'proteins': 2}, {'samples': True}, {'runs': 3.0}, {'warmup': -1},
                       {'schema': 2}, {'run_id': ''}, {'cann_version': 'not known'},
                       {'logical_device': True}, {'median_seconds': 7}, {'mutations_per_second': 100}]:
            write_json(self.root / 'run.json', {**run, **change})
            with self.subTest(change=change), self.assertRaises(ValueError):
                collect(self.root, c, m)
        write_json(self.root / 'run.json', run)
        for key in ('schema', 'python', 'torch', 'torch_npu', 'logical_device', 'ascend_home_path', 'cann_version', 'device_name', 'hostname'):
            write_json(self.root / 'run.json', {k: v for k, v in run.items() if k != key})
            with self.subTest(missing=key), self.assertRaises(KeyError):
                collect(self.root, c, m)
        write_json(self.root / 'run.json', run)
        for change in [{'mse': -1}, {'pearson': 2}, {'rmse': float('nan')}]:
            m.write_text(json.dumps({**metrics, **change}))
            with self.subTest(metrics=change), self.assertRaises(ValueError):
                collect(self.root, c, m)
        write_json(m, metrics)
        write_json(c, {**comparison, 'passed': 'false'})
        with self.assertRaises(ValueError):
            collect(self.root, c, m)
        write_json(c, comparison)
        for row in [{**self.row, 'ddG_pred': float('nan')}, {**self.row, 'position': -1}]:
            predictions.write_text(json.dumps(row) + '\n')
            changed = sha256(predictions)
            write_json(self.root / 'run.json', {**run, 'predictions_sha256': changed})
            write_json(c, {**comparison, 'candidate_sha256': changed})
            write_json(m, {**metrics, 'predictions_sha256': changed})
            with self.subTest(prediction=row), self.assertRaises(ValueError):
                collect(self.root, c, m)
        self.jsonl('predictions.jsonl', [self.row])
        write_json(self.root / 'run.json', {k: v for k, v in run.items() if k != 'cann_version'})
        write_json(c, comparison)
        write_json(m, metrics)
        proc = subprocess.run([sys.executable, str(ROOT / 'report_acceptance.py'),
                               '--run-dir', str(self.root), '--comparison', str(c),
                               '--metrics', str(m), '--output', str(self.root / 'report.json')], capture_output=True)
        self.assertNotEqual(proc.returncode, 0)


    def test_structure_archive_selection_and_reuse(self):
        archive = self.root / 'pdb.zip'
        with zipfile.ZipFile(archive, 'w') as stream:
            stream.writestr('nested/demo.pdb', self.pdb.read_bytes())
            stream.writestr('../unneeded.pdb', b'ignored')
        output = self.root / 'structures'
        with patch('prepare_megascale.PDB_ARCHIVE_MD5', hashlib.md5(archive.read_bytes()).hexdigest()):
            result = prepare_structures([{'pdb': 'demo.pdb'}], archive, output)
            self.assertEqual(result['pdb_sha256']['demo.pdb'], sha256(self.pdb))
            prepare_structures([{'pdb': 'demo.pdb'}], archive, output)
            self.assertEqual(list(output.iterdir()), [output / 'demo.pdb'])
            with self.assertRaises(ValueError):
                prepare_structures([{'pdb': 'missing.pdb'}], archive, output)
            (output / 'demo.pdb').write_text('changed')
            with self.assertRaises(ValueError):
                prepare_structures([{'pdb': 'demo.pdb'}], archive, output)
        with self.assertRaises(ValueError):
            prepare_structures([{'pdb': 'demo.pdb'}], archive, output)

    def test_independent_performance_runs(self):
        dirs = []
        for i, seconds in enumerate([[1, 2, 3], [3, 4, 5], [5, 6, 7]]):
            directory = self.root / str(i)
            directory.mkdir()
            rows = [{**self.row, 'id': str(j), 'pdb': str(self.pdb if j < 6 else self.root / 'other.pdb')}
                    for j in range(12)]
            (directory / 'predictions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            write_json(directory / 'run.json', {**self.run_record(directory, seconds, 12, 2), 'run_id': str(i)})
            dirs.append(directory)
        base = json.loads((dirs[0] / 'run.json').read_text())
        result = performance(dirs, base)
        self.assertEqual(result['process_median_seconds'], [2, 4, 6])
        self.assertEqual(result['median_seconds'], 4)
        self.assertEqual(result['mutations_per_second'], 3)
        first = json.loads((dirs[0] / 'run.json').read_text())
        c, m = self.root / 'comparison.json', self.root / 'metrics.json'
        write_json(c, {'passed': True, 'samples': 12, 'candidate_sha256': first['predictions_sha256']})
        write_json(m, self.quality_metrics(12, first['predictions_sha256']))
        command = [sys.executable, str(ROOT / 'report_acceptance.py'), '--run-dir', str(dirs[0]),
                   '--comparison', str(c), '--metrics', str(m), '--output', str(self.root / 'report.json'),
                   '--performance-dirs']
        proc = subprocess.run(command + list(map(str, dirs)), capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads((self.root / 'report.json').read_text())['performance']['processes'], 3)
        proc = subprocess.run(command + list(map(str, dirs[:2])), capture_output=True)
        self.assertNotEqual(proc.returncode, 0)
        for bad in [dirs[:2], [dirs[0]] * 3]:
            with self.assertRaises(ValueError):
                performance(bad, base)
        path = dirs[2] / 'run.json'
        original = json.loads(path.read_text())
        for change in [{'run_id': '0'}, {'manifest_sha256': 'other'}, {'device': 'cpu'},
                       {'seconds': [0, 1, 2]}, {'predictions_sha256': 'wrong'}, {'simulated': True},
                       {'python': '3.12 fixture'}, {'logical_device': 1},
                       {'ascend_home_path': '/other/cann'}, {'cann_version': '9.1.0'},
                       {'device_name': 'Ascend310P'}, {'hostname': 'other-host'},
                       {'cann_version': 'not known'}, {'ascend_home_path': None},
                       {'logical_device': True}, {'infer_sha256': 'other'},
                       {'patch_sha256': 'other'}, {'inference_mode': 'per-protein'},
                       {'gelu': 'approximate'}]:
            write_json(path, {**original, **change})
            with self.subTest(change=change), self.assertRaises(ValueError):
                performance(dirs, base)
        for key in ('python', 'logical_device', 'ascend_home_path', 'cann_version', 'device_name', 'hostname'):
            legacy = {k: v for k, v in original.items() if k != key}
            write_json(path, legacy)
            with self.subTest(missing=key), self.assertRaises(KeyError):
                performance(dirs, base)
        write_json(path, {**original, 'python': 'different'})
        proc = subprocess.run(command + list(map(str, dirs)), capture_output=True)
        self.assertNotEqual(proc.returncode, 0)

    def test_runtime_metadata_requires_actual_npu_environment(self):
        tree = ast.parse((ROOT / 'infer.py').read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'runtime_metadata']
        fake_torch = Mock()
        fake_torch.__version__ = 'fixture'
        fake_torch.npu.current_device.return_value = 0
        fake_torch.npu.get_device_name.return_value = 'Ascend910B'
        fake_torch.version.cann = '8.3.RC1'
        namespace = {'torch': fake_torch, 'platform': Mock(node=lambda: 'fixture-host'),
                     'sys': sys, 'os': __import__('os')}
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'infer.py', 'exec'), namespace)
        backend = types.ModuleType('torch_npu')
        backend.__version__ = 'fixture'
        with patch.dict(sys.modules, {'torch_npu': backend}), \
                patch.dict(__import__('os').environ, {'ASCEND_HOME_PATH': '/fixture/cann'}):
            self.assertEqual(namespace['runtime_metadata']('npu')['cann_version'], '8.3.RC1')
            fake_torch.version.cann = 'not known'
            with self.assertRaises(ValueError):
                namespace['runtime_metadata']('npu')
        self.assertNotIn('cann_version', namespace['runtime_metadata']('cpu'))


if __name__ == '__main__':
    unittest.main()
