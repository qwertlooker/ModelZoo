# Licensed under the Apache License, Version 2.0.
"""Offline regression: no simulated output is saved as NPU evidence."""

import ast
import csv
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import tarfile
import io
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
utils = importlib.import_module("adapter_utils")
read_inputs, sha256, write_json = utils.read_inputs, utils.sha256, utils.write_json
compare = importlib.import_module("compare_results").compare
ppv_at_n = importlib.import_module("eval_metrics").ppv_at_n
extract_verified = importlib.import_module("prepare_weights").extract_verified


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def csv(self, text):
        path = self.directory / "input.csv"
        path.write_text(text, encoding="utf-8")
        return path

    def test_demo_contract(self):
        self.assertEqual(len(read_inputs(ROOT / "test_data/smoke.csv")), 4)

    def test_invalid_inputs(self):
        for text in (
            "id,peptide,alleles\nx,SIINFEKL,A\nx,SIINFEKL,A\n",
            "id,peptide,alleles\nx,XXX,A\n",
            "id,peptide,alleles,n_flank\nx,SIINFEKL,A,AAA\n",
            "id,peptide,alleles\nx,SIINFEKL,A;\n",
            "id,peptide,alleles,label\nx,SIINFEKL,A,2\n",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                read_inputs(self.csv(text))

    def test_multiallele_is_presentation_only(self):
        path = self.csv("id,peptide,alleles\nx,SIINFEKL,A;B\n")
        self.assertEqual(len(read_inputs(path)), 1)
        with self.assertRaises(ValueError):
            read_inputs(path, "affinity")

    def test_processing_does_not_require_alleles(self):
        self.assertEqual(
            len(read_inputs(self.csv("id,peptide\nx,SIINFEKL\n"), "processing")), 1
        )

    def test_cli_dry_run_no_dependencies_or_output(self):
        destination = self.directory / "output"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "infer.py"),
                "--manifest",
                str(ROOT / "test_data/smoke.csv"),
                "--model-dir",
                "nonexistent",
                "--output-dir",
                str(destination),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("simulation", result.stdout)
        self.assertFalse(destination.exists())

    def test_cli_invalid_runs_fail(self):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "infer.py"),
                "--manifest",
                str(ROOT / "test_data/smoke.csv"),
                "--model-dir",
                "missing",
                "--output-dir",
                str(self.directory / "out"),
                "--runs",
                "2",
                "--dry-run",
            ],
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)

    def test_preparation_writes_metadata(self):
        destination = self.directory / "prepared"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "prepare_eval_data.py"),
                "--input",
                str(ROOT / "test_data/smoke.csv"),
                "--output-dir",
                str(destination),
                "--dataset",
                "fixture",
                "--split",
                "demo",
                "--purpose",
                "smoke",
            ],
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        meta = json.loads((destination / "manifest.meta.json").read_text())
        self.assertEqual(meta["manifest_sha256"], sha256(destination / "manifest.csv"))
        self.assertEqual(meta["purpose"], "smoke")

    def test_smoke_cannot_be_declared_l2(self):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "prepare_eval_data.py"),
                "--input",
                str(ROOT / "test_data/smoke.csv"),
                "--output-dir",
                str(self.directory / "prepared"),
                "--dataset",
                "fixture",
                "--split",
                "demo",
                "--purpose",
                "l2",
            ],
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)

    def result(self, name, device, values=(0.3, 0.6)):
        directory = self.directory / name
        directory.mkdir()
        with (directory / "predictions.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "processing_score"])
            writer.writerows(zip(["a", "b"], values, strict=True))
        meta = {
            "simulation": False,
            "device": device,
            "rows": 2,
            "predictions_sha256": sha256(directory / "predictions.csv"),
            "manifest_sha256": "same",
            "weights_sha256": "same",
            "source_revision": "same",
            "mhcflurry": "2.3.14",
            "torch": "2.10.0",
            "task": "processing",
            "batch_size": 128,
            "optimization_level": 0,
            "compile": False,
            "network_devices": [[device]],
            "run_seconds": [1, 2, 3],
        }
        write_json(directory / "run.json", meta)
        return directory

    def test_compare_id_order_and_numeric_failure(self):
        baseline = self.result("cpu", "cpu")
        candidate = self.result("npu", "npu", (0.3, 0.7))
        self.assertFalse(compare(baseline, candidate, 0.001, 0.001)["numeric_pass"])
        self.assertEqual(
            compare(baseline, candidate, 0.2, 0.001)["npu_median_seconds"], 2
        )

    def test_compare_rejects_simulation_and_provenance_mismatch(self):
        baseline = self.result("cpu", "cpu")
        candidate = self.result("npu", "npu")
        meta = json.loads((candidate / "run.json").read_text())
        for key, value in (
            ("simulation", True),
            ("weights_sha256", "different"),
            ("network_devices", [["cpu"]]),
            ("run_seconds", [1]),
            ("device", "cpu"),
        ):
            changed = dict(meta)
            changed[key] = value
            write_json(candidate / "run.json", changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                compare(baseline, candidate, 0.001, 0.001)

    def test_compare_nonfinite_and_negative_tolerance(self):
        baseline = self.result("cpu", "cpu")
        candidate = self.result("npu", "npu", (float("nan"), 0.6))
        with self.assertRaises(ValueError):
            compare(baseline, candidate, 0.001, 0.001)
        with self.assertRaises(ValueError):
            compare(baseline, candidate, -1, 0)

    def test_compare_cli_threshold_failure_is_nonzero(self):
        baseline = self.result("cpu", "cpu")
        candidate = self.result("npu", "npu", (0.3, 0.8))
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "compare_results.py"),
                "--reference",
                str(baseline),
                "--candidate",
                str(candidate),
                "--report",
                str(self.directory / "report.json"),
            ],
            capture_output=True,
        )
        self.assertEqual(result.returncode, 1)

    def test_ppv_ties_are_order_independent(self):
        self.assertEqual(ppv_at_n([1, 0, 1, 0], [0.8, 0.8, 0.2, 0.2]), 0.5)
        self.assertEqual(ppv_at_n([0, 1, 0, 1], [0.8, 0.8, 0.2, 0.2]), 0.5)
        self.assertEqual(ppv_at_n([1, 1, 0], [0.9, 0.8, 0.1]), 1)

    def test_corrupt_weights_fail_before_extraction(self):
        archive = self.directory / "weights.tar.bz2"
        archive.write_text("HTML proxy error")
        with self.assertRaises(ValueError):
            extract_verified(archive, self.directory / "extracted")
        self.assertFalse((self.directory / "extracted").exists())

    def test_weight_inventory_matches_official_archive(self):
        archive = self.directory / "bundle.tar.bz2"
        model = self.directory / "model"
        names = [
            "weights.csv",
            "affinity_predictor/manifest.csv",
            "processing_predictor_with_flanks/manifest.csv",
            "processing_predictor_without_flanks/manifest.csv",
            "weights.npz",
        ]
        with tarfile.open(archive, "w:bz2") as stream:
            for name in names:
                data = ("fixture:" + name).encode()
                member = tarfile.TarInfo("bundle/" + name)
                member.size = len(data)
                stream.addfile(member, io.BytesIO(data))
                target = model / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "archive SHA256"):
            utils.weights_inventory(model, archive)
        with patch.object(utils, "WEIGHT_SHA256", sha256(archive)):
            self.assertEqual(
                len(utils.weights_inventory(model, archive)[0]), len(names)
            )
            for name in ("weights.npz", "affinity_predictor/manifest.csv"):
                target = model / name
                original = target.read_bytes()
                target.write_text("changed")
                with self.assertRaisesRegex(ValueError, "differs"):
                    utils.weights_inventory(model, archive)
                target.unlink()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    utils.weights_inventory(model, archive)
                target.write_bytes(original)
            (model / "extra.npz").write_text("extra")
            with self.assertRaisesRegex(ValueError, "differs"):
                utils.weights_inventory(model, archive)

    def test_patient_validation(self):
        good = [
            dict(sample_id="p1", label="0"),
            dict(sample_id="p1", label="1"),
            dict(sample_id="p2", label="1"),
            dict(sample_id="p2", label="0"),
        ]
        self.assertEqual(set(utils.presentation_groups(good)), {"p1", "p2"})
        for bad in (
            [dict(sample_id="", label="1")],
            [dict(sample_id="  ", label="1")],
            [dict(sample_id="p1", label="1")],
            [dict(sample_id="p1", label="2")],
        ):
            with self.assertRaises(ValueError):
                utils.presentation_groups(bad)
        with self.assertRaises(KeyError):
            utils.presentation_groups([dict(label="1")])

    def test_l2_preparation_rejects_missing_patient_labels(self):
        source = self.csv(
            "id,peptide,alleles\n"
            + "".join(f"{i},SIINFEKL,HLA-A*02:01\n" for i in range(1000))
        )
        destination = self.directory / "l2"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "prepare_eval_data.py"),
                "--input",
                str(source),
                "--output-dir",
                str(destination),
                "--dataset",
                "fixture",
                "--split",
                "test",
                "--purpose",
                "l2",
            ],
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(destination.exists())

    def test_runtime_routes_and_presentation_id_alignment(self):
        torch = types.ModuleType("torch")
        mhcflurry = types.ModuleType("mhcflurry")
        mhcflurry.Class1PresentationPredictor = Mock()
        common = types.ModuleType("mhcflurry.common")
        common.configure_pytorch = Mock()
        common.get_pytorch_device = Mock()
        with patch.dict(
            sys.modules,
            {"torch": torch, "mhcflurry": mhcflurry, "mhcflurry.common": common},
        ):
            sys.modules.pop("runtime", None)
            runtime = importlib.import_module("runtime")
            self.addCleanup(lambda: sys.modules.pop("runtime", None))
            rows = read_inputs(ROOT / "test_data/smoke.csv")[:2]
            predictor = Mock()
            predictor.affinity_predictor.predict.return_value = [100, 200]
            predictor.predict_processing.return_value = [0.1, 0.2]
            affinity = runtime.predict(predictor, rows, "affinity", 16)
            self.assertEqual(affinity[0]["affinity"], 100)
            self.assertEqual(
                predictor.affinity_predictor.predict.call_args.kwargs["model_kwargs"],
                {"batch_size": 16},
            )
            self.assertEqual(
                runtime.predict(predictor, rows, "processing", 16)[1][
                    "processing_score"
                ],
                0.2,
            )
            records = [
                {
                    "peptide_num": i,
                    "peptide": row["peptide"],
                    "sample_name": "genotype_0",
                    "affinity": 100,
                    "processing_score": 0.2,
                    "presentation_score": 0.5,
                    "presentation_percentile": 5,
                }
                for i, row in enumerate(rows)
            ]
            predictor.predict.return_value.to_dict.return_value = list(
                reversed(records)
            )
            self.assertEqual(
                [
                    row["id"]
                    for row in runtime.predict(predictor, rows, "presentation", 16)
                ],
                [row["id"] for row in rows],
            )
            grouped_rows = [dict(rows[0], id=str(i), alleles="A;B") for i in range(128)]
            grouped_rows.append(dict(rows[0], id="128", alleles="B;A"))
            grouped_records = [
                dict(
                    peptide_num=i,
                    peptide=row["peptide"],
                    sample_name="genotype_0" if i < 128 else "genotype_1",
                    affinity=i + 1,
                    processing_score=0.2,
                    presentation_score=0.5,
                    presentation_percentile=5,
                )
                for i, row in enumerate(grouped_rows)
            ]
            predictor.predict.return_value.to_dict.return_value = list(
                reversed(grouped_records)
            )
            grouped_output = runtime.predict(
                predictor, grouped_rows, "presentation", 128
            )
            kwargs = predictor.predict.call_args.kwargs
            self.assertEqual(kwargs["sample_names"].count("genotype_0"), 128)
            self.assertEqual(
                kwargs["alleles"], {"genotype_0": ["A", "B"], "genotype_1": ["B", "A"]}
            )
            self.assertEqual(
                [row["affinity"] for row in grouped_output], list(range(1, 130))
            )
            predictor.predict.return_value.to_dict.return_value = records
            records[0]["sample_name"] = "wrong"
            with self.assertRaises(ValueError):
                runtime.predict(predictor, rows, "presentation", 16)


@unittest.skipUnless(
    os.environ.get("MHCFLURRY_UPSTREAM"),
    "Set MHCFLURRY_UPSTREAM to fixed patched source for backend tests",
)
class BackendTests(unittest.TestCase):
    def function(self, filename, name, extra=None):
        source = Path(os.environ["MHCFLURRY_UPSTREAM"]) / "mhcflurry" / filename
        tree = ast.parse(source.read_text(encoding="utf-8"))
        node = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        namespace = dict(extra or {})
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"),
            namespace,
        )
        return namespace[name]

    def test_explicit_npu_and_failure_no_cpu_fallback(self):
        torch = types.ModuleType("torch")
        torch.device = Mock(side_effect=lambda value: value)
        torch.cuda = Mock()
        extension = types.ModuleType("torch_npu")
        extension.npu = Mock()
        with patch.dict(sys.modules, {"torch": torch, "torch_npu": extension}):
            function = self.function(
                "common.py", "get_pytorch_device", {"_pytorch_backend": "npu"}
            )
            extension.npu.is_available.return_value = True
            self.assertEqual(function(), "npu")
            extension.npu.is_available.return_value = False
            with self.assertRaises(RuntimeError):
                function()
            torch.cuda.is_available.assert_not_called()

    def test_cpu_cuda_unchanged(self):
        torch = types.ModuleType("torch")
        torch.device = lambda value: value
        torch.cuda = Mock()
        torch.cuda.is_available.return_value = True
        with patch.dict(sys.modules, {"torch": torch, "torch_npu": None}):
            self.assertEqual(
                self.function(
                    "common.py", "get_pytorch_device", {"_pytorch_backend": "cpu"}
                )(),
                "cpu",
            )
            self.assertEqual(
                self.function(
                    "common.py", "get_pytorch_device", {"_pytorch_backend": "gpu"}
                )(),
                "cuda",
            )
            self.assertEqual(
                self.function(
                    "common.py", "get_pytorch_device", {"_pytorch_backend": "auto"}
                )(),
                "cuda",
            )

    def test_missing_backend_dependency_propagates(self):
        torch = types.ModuleType("torch")
        with patch.dict(sys.modules, {"torch": torch, "torch_npu": None}):
            with self.assertRaises(ModuleNotFoundError):
                self.function(
                    "common.py", "get_pytorch_device", {"_pytorch_backend": "npu"}
                )()

    def test_synchronization_routes(self):
        torch = types.ModuleType("torch")
        torch.npu = Mock()
        torch.cuda = Mock()
        function = self.function("pytorch_sizing.py", "synchronize_device")
        with patch.dict(sys.modules, {"torch": torch}):
            function(types.SimpleNamespace(type="npu"))
            torch.npu.synchronize.assert_called_once_with()
            function(types.SimpleNamespace(type="cuda"))
            self.assertEqual(torch.cuda.synchronize.call_count, 1)
            function(types.SimpleNamespace(type="cpu"))
            self.assertEqual(torch.npu.synchronize.call_count, 1)

    def test_installed_source_contract_matches_patched_checkout(self):
        root = Path(os.environ["MHCFLURRY_UPSTREAM"]) / "mhcflurry"
        manifest = json.loads((ROOT / "source_manifest.json").read_text())
        for relative, expected in manifest.items():
            actual = hashlib.sha256(
                (root / relative).read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            self.assertEqual(actual, expected, relative)


if __name__ == "__main__":
    unittest.main()
