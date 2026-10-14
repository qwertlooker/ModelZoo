"""CPU mathematical regressions on the fixed real model; not NPU acceptance."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import infer


def protein(length, seed=0, missing=None):
    generator = np.random.default_rng(seed)
    ca = generator.normal(size=(length, 3)) + np.arange(length)[:, None] * [1.3, 0.2, 0.1]
    coords = {f'{atom}_chain_A': (ca + shift).tolist()
              for atom, shift in [('N', [-1, 0.2, 0]), ('CA', [0, 0, 0]),
                                  ('C', [1, 0.3, 0.1]), ('O', [1.2, 1, 0.2])]}
    if missing is not None:
        coords['O_chain_A'][missing] = [float('nan')] * 3
    sequence = (infer.ALPHABET[:20] * (length // 20 + 1))[:length]
    pdb = [{'name': f'fixture-{seed}-{length}', 'seq': sequence, 'seq_chain_A': sequence,
            'num_of_chains': 1, 'coords_chain_A': coords}]
    positions = sorted({0, length // 2, length - 1} - {missing})
    mutations = [infer.Mutation(position, sequence[position], 'Y' if sequence[position] != 'Y' else 'A')
                 for position in positions]
    return pdb, mutations, [{}] * len(mutations)


class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        infer.validate_source()
        cfg = OmegaConf.create({
            'platform': {'thermompnn_dir': str(infer.SOURCE)},
            'training': {'learn_rate': 0.001, 'lr_schedule': True},
            'model': {'hidden_dims': [64, 32], 'subtract_mut': True,
                      'num_final_layers': 2, 'freeze_weights': True,
                      'load_pretrained': True, 'lightattn': True},
        })
        cls.model = infer.TransferModelPL.load_from_checkpoint(
            infer.SOURCE / 'models/thermoMPNN_default.pt', cfg=cfg, map_location='cpu').model.eval()
        infer.use_accurate_gelu(cls.model)

    def test_exact_gelu_and_local_replacement(self):
        x = torch.linspace(-10, 10, 1001)
        torch.testing.assert_close(infer._AccurateGELU()(x), torch.nn.functional.gelu(x), atol=2e-6, rtol=2e-6)
        self.assertFalse(any(isinstance(module, torch.nn.GELU) for module in self.model.modules()))
        self.assertIsInstance(torch.nn.GELU(), torch.nn.GELU)

    def test_center_tap_single_and_multiple_mutations(self):
        for count in (1, 7):
            x = torch.randn(count, 384)
            with torch.no_grad():
                reference = self.model.light_attention(x.unsqueeze(-1), torch.ones(count, 1)).reshape(count, 384)
                candidate = infer.light_attention_length1(self.model, x)
            torch.testing.assert_close(candidate, reference, atol=2e-6, rtol=2e-6)
        with patch.object(self.model.light_attention, 'training', True), self.assertRaises(ValueError):
            infer.light_attention_length1(self.model, torch.randn(1, 384))

    def test_all_modes_short_variable_and_equal_lengths(self):
        prepared = [protein(length, seed) for seed, length in enumerate((1, 2, 17, 48, 53, 53))]
        with torch.no_grad():
            all_values = infer.batched_forward_all(self.model, prepared)
            grouped = infer.batched_forward_grouped(self.model, prepared)
            for item, candidate, bucket in zip(prepared, all_values, grouped):
                pdb, mutations, _ = item
                reference, _ = self.model(pdb, mutations)
                reference = torch.cat([entry['ddG'].reshape(-1) for entry in reference])
                single = infer.batched_forward(self.model, pdb, mutations)
                for value in (candidate, bucket, single):
                    self.assertEqual(value.shape, (len(mutations),))
                    torch.testing.assert_close(value, reference, atol=1e-5, rtol=1e-5)

    def test_missing_nonmutation_coordinate_stays_batch_invariant(self):
        prepared = [protein(17, 42, missing=5), protein(53, 8)]
        with torch.no_grad():
            all_values = infer.batched_forward_all(self.model, prepared)
            for (pdb, mutations, _), candidate in zip(prepared, all_values):
                torch.testing.assert_close(candidate, infer.batched_forward(self.model, pdb, mutations),
                                           atol=1e-5, rtol=1e-5)

    def test_non_subtract_head(self):
        item = protein(9, 20)
        with patch.object(self.model, 'subtract_mut', False), torch.no_grad():
            reference, _ = self.model(item[0], item[1])
            torch.testing.assert_close(infer.batched_forward(self.model, item[0], item[1]),
                                       torch.cat([r['ddG'].reshape(-1) for r in reference]), atol=1e-5, rtol=1e-5)

    def test_coordinate_sentinel_bound(self):
        pdb, _, _ = protein(3)
        infer.validate_coordinate_span(pdb, 'A')
        pdb[0]['coords_chain_A']['CA_chain_A'][1][0] = 2e4
        with self.assertRaises(ValueError):
            infer.validate_coordinate_span(pdb, 'A')

    def test_cli_modes_are_mutually_exclusive(self):
        for flags in (['--gnn-bucket', '--upstream-head'], ['--gnn-bucket', '--per-protein'],
                      ['--upstream-head', '--per-protein']):
            with patch.object(sys, 'argv', ['infer.py', '--manifest', 'unused', '--checkpoint', 'unused',
                                           '--output-dir', 'unused', *flags]), self.assertRaises(SystemExit) as raised:
                infer.main()
            self.assertEqual(raised.exception.code, 2)


class SourceGuardTests(unittest.TestCase):
    def test_guard_rejects_extra_changes_even_when_reverse_check_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            original = subprocess.check_output(['git', '-C', str(infer.SOURCE), 'show', 'HEAD:protein_mpnn_utils.py'])
            filename = source / 'protein_mpnn_utils.py'
            filename.write_bytes(original)
            (source / 'other.py').write_text('x = 1\n')
            subprocess.run(['git', '-C', str(source), 'init', '-q'], check=True)
            subprocess.run(['git', '-C', str(source), 'add', '.'], check=True)
            subprocess.run(['git', '-C', str(source), '-c', 'user.name=Test', '-c', 'user.email=test@example.com',
                            'commit', '-qm', 'fixed source'], check=True)
            revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
            patch_file = root / 'diff.patch'
            patch_file.write_bytes((ROOT / 'diff.patch').read_bytes())
            subprocess.run(['git', '-C', str(source), 'apply', str(patch_file)], check=True)
            with patch.object(infer, 'SOURCE', source), patch.object(infer, 'UPSTREAM_COMMIT', revision):
                self.assertEqual(infer.validate_source(), (revision, infer.sha256(patch_file)))
                approved = filename.read_bytes()
                filename.write_bytes(approved + b'\n# unrelated change\n')
                with self.assertRaisesRegex(ValueError, 'exact approved patch'):
                    infer.validate_source()
                filename.write_bytes(approved)
                (source / 'other.py').write_text('x = 2\n')
                with self.assertRaises(subprocess.CalledProcessError):
                    infer.validate_source()


if __name__ == '__main__':
    unittest.main()
