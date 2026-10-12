"""Mock/stdlib contracts for the independent baseline, not model evidence."""
import argparse
import ast
from collections import defaultdict, deque
from contextlib import nullcontext
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from delivery_utils import finite, read_jsonl, sha256, validate_sample, write_json


class UpstreamBaselineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        tree = ast.parse((ROOT / 'baseline_upstream.py').read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        self.torch = Mock()
        self.torch.no_grad.side_effect = nullcontext
        self.namespace = dict(argparse=argparse, defaultdict=defaultdict, deque=deque,
                              json=json, math=math, Path=Path, sys=sys, torch=self.torch,
                              finite=finite, read_jsonl=read_jsonl, sha256=sha256,
                              validate_sample=validate_sample, write_json=write_json)
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'baseline_upstream.py', 'exec'), self.namespace)
        self.pdb = [{'seq': 'A', 'seq_chain_A': 'M', 'num_of_chains': 1}]
        self.row = dict(id='first', chain='A', position=0, wildtype='A', mutation='C',
                        ddG_true=1.2, wt_sequence='A')
        self.mutation = types.SimpleNamespace(position=0, wildtype='A', mutation='C',
                                              ddG=Mock(item=lambda: 1.2000000476837158))

    def tearDown(self):
        self.tmp.cleanup()

    def test_duplicate_measurements_retain_distinct_ids(self):
        rows = [self.row, {**self.row, 'id': 'second', 'ddG_true': 2.0}]
        second = types.SimpleNamespace(**{**vars(self.mutation), 'ddG': Mock(item=lambda: 2.0)})
        aligned = self.namespace['align_mutations'](self.pdb, [self.mutation, second], rows)
        self.assertEqual([row['id'] for row in aligned], ['first', 'second'])
        self.assertEqual(self.pdb[0]['seq_chain_A'], 'M')

    def test_alignment_rejects_labels_sequences_and_missing_samples(self):
        for change in ({'ddG_true': -1.2}, {'wt_sequence': 'M'}, {'chain': 'B'}, {'position': 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.namespace['align_mutations'](self.pdb, [self.mutation], [{**self.row, **change}])
        with self.assertRaises(ValueError):
            self.namespace['align_mutations'](self.pdb, [], [self.row])
        with self.assertRaises(ValueError):
            self.namespace['align_mutations'](self.pdb, [self.mutation], [])

    def test_official_inputs_pass_directly_to_model(self):
        mutations = [self.mutation]
        dataset = Mock(wt_names=['demo.pdb'])
        dataset.__getitem__ = Mock(return_value=(self.pdb, mutations))
        tensor = Mock(device=types.SimpleNamespace(type='cpu'), item=lambda: 1.25)
        model = Mock(return_value=([{'ddG': tensor}], None))
        predictions = self.namespace['run_baseline'](dataset, model, {'demo.pdb': [self.row]})
        self.assertEqual(predictions[0]['ddG_pred'], 1.25)
        self.assertIs(model.call_args.args[0], self.pdb)
        self.assertIs(model.call_args.args[1], mutations)
        tensor.device.type = 'npu'
        with self.assertRaises(RuntimeError):
            self.namespace['run_baseline'](dataset, model, {'demo.pdb': [self.row]})
        tensor.device.type = 'cpu'
        with self.assertRaises(ValueError):
            self.namespace['run_baseline'](dataset, model, {'other.pdb': [self.row]})
        model.return_value = ([], None)
        with self.assertRaises(ValueError):
            self.namespace['run_baseline'](dataset, model, {'demo.pdb': [self.row]})

    def test_main_uses_fixed_original_dataset_and_private_cache(self):
        source = self.root / 'source'
        files = ['data_all/testing/mega_test.csv', 'dataset_splits/mega_splits.pkl',
                 'vanilla_model_weights/v_48_020.pt', 'config.yaml']
        for filename in files:
            path = source / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('trusted fixture ' + filename)
        checkpoint = self.root / 'checkpoint.pt'
        checkpoint.write_bytes(b'fixture')
        pdb_dir = self.root / 'pdbs'
        pdb_dir.mkdir()
        pdb_file = pdb_dir / 'demo.pdb'
        pdb_file.write_text('ATOM  fixture')
        manifest = self.root / 'manifest.jsonl'
        manifest.write_text(json.dumps({**self.row, 'pdb': str(pdb_file), 'pdb_sha256': sha256(pdb_file)}) + '\n')
        output = self.root / 'out'
        cfg = types.SimpleNamespace()
        dataset = Mock(wt_names=['demo.pdb'])
        dataset.__getitem__ = Mock(return_value=(self.pdb, [self.mutation]))
        factory = Mock(return_value=dataset)
        model = Mock(return_value=([{'ddG': Mock(device=types.SimpleNamespace(type='cpu'), item=lambda: 1.25)}], None))
        model.eval.return_value.to.return_value = model
        loader = Mock()
        loader.load_from_checkpoint.return_value.model = model
        self.torch.__version__ = 'mock-only'
        process = Mock()
        process.check_output.return_value = 'pinned\n'
        self.namespace.update(SOURCE=source, CSV_SHA=sha256(source / files[0]),
                              SPLITS_SHA=sha256(source / files[1]), BACKBONE_SHA=sha256(source / files[2]),
                              WEIGHT_SHA=sha256(checkpoint), UPSTREAM_COMMIT='pinned',
                              subprocess=process, np=Mock(), OmegaConf=Mock(load=lambda path: cfg),
                              MegaScaleDataset=factory, TransferModelPL=loader)
        argv = ['baseline_upstream.py', '--checkpoint', str(checkpoint), '--manifest', str(manifest),
                '--pdb-dir', str(pdb_dir), '--output-dir', str(output)]
        with patch.object(sys, 'argv', argv):
            self.namespace['main']()
        factory.assert_called_once_with(cfg, 'test')
        self.assertEqual(cfg.data_loc['megascale_csv'], str(source / files[0]))
        self.assertEqual(cfg.platform['cache_dir'], str(output / 'cache'))
        self.assertEqual(loader.load_from_checkpoint.call_args.kwargs['map_location'], 'cpu')
        metadata = json.loads((output / 'baseline.json').read_text())
        self.assertEqual(metadata['samples'], 1)
        self.assertEqual(metadata['device'], 'cpu')
        self.assertEqual(metadata['predictions_sha256'], sha256(output / 'predictions.jsonl'))
        with patch.object(sys, 'argv', argv), self.assertRaises(FileExistsError):
            self.namespace['main']()
        (source / files[0]).write_text('changed CSV')
        with patch.object(sys, 'argv', argv), self.assertRaisesRegex(ValueError, 'artifact SHA'):
            self.namespace['main']()


if __name__ == '__main__':
    unittest.main()
