import json
import subprocess
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np
import common
import model

KIND = "DINOv2"


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / "input.npz"
        if KIND == "DINOv2":
            self.x = np.zeros((2, 3, 224, 224), dtype=np.float32)
            self.y = np.array([1, 2], dtype=np.int64)
            self.pred = np.zeros((2, 1000), dtype=np.float32)
            self.pred[0, 1] = self.pred[1, 2] = 2
            prep = {"preprocessor_sha256": model.ASSETS["preprocessor_config.json"], "count": 2}
        elif KIND == "TTM-R2":
            self.x = np.zeros((2, 512, 7), dtype=np.float32)
            self.y = np.ones((2, 96, 7), dtype=np.float32)
            self.pred = self.y.copy()
            prep = {"model": "TTM-R2", "task": "forecast", "scaling": "etth1_train_standard"}
        elif KIND == "Toto-2.0":
            self.x = np.zeros((2, 7, 512), dtype=np.float32)
            self.y = np.ones((2, 7, 96), dtype=np.float32)
            self.pred = self.y.copy()
            prep = {"model": "Toto-2.0", "task": "forecast", "scaling": "none"}
        else:
            self.x = np.zeros((2, 7, 512), dtype=np.float32)
            self.y = self.x.copy()
            self.pred = self.y.copy()
            prep = {"model": "MOMENT", "task": "reconstruction", "scaling": "etth1_train_standard"}
        np.savez_compressed(self.data, inputs=self.x, targets=self.y, preparation=json.dumps(prep))

    def tearDown(self):
        self.tmp.cleanup()

    def cli(self, *args, ok=True):
        command = [sys.executable, str(Path(model.__file__)), *map(str, args)]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode == 0, ok, result.stderr)
        return result

    def result(self, path, offset=0, input_sha=None):
        meta = {"model": model.MODEL, "model_sha256": model.ASSETS["model.safetensors"],
                "config_sha256": model.ASSETS["config.json"], "head_sha256": None,
                "input_sha256": input_sha or common.digest(self.data), "dtype": "float32",
                "device": "cpu", "task": "reconstruction"}
        fields = {"prediction": self.pred + offset, "metadata": json.dumps(meta)}
        if KIND == "Toto-2.0":
            fields["quantiles"] = np.stack([self.pred + offset] * 9)
        np.savez_compressed(path, **fields)

    def test_evaluation_and_comparison_fail_closed(self):
        ref = self.root / "reference.npz"
        other = self.root / "other.npz"
        report = self.root / "report.json"
        self.result(ref)
        self.result(other, offset=0.25)
        metric = "classification" if KIND == "DINOv2" else "regression"
        self.cli("evaluate", "--input", self.data, "--results", ref, "--output", report, "--metric", metric)
        score = json.loads(report.read_text())
        self.assertEqual(score["top1"] if KIND == "DINOv2" else score["mse"], 1.0 if KIND == "DINOv2" else 0.0)
        if KIND == "Toto-2.0":
            self.assertEqual(score["mean_wql"], 0.0)
        self.cli("compare", "--reference", ref, "--candidate", other, "--max-absolute", 0.1,
                 "--output", report, ok=False)
        self.cli("compare", "--reference", ref, "--candidate", other, "--max-absolute", 0.3,
                 "--output", report)
        self.result(other, input_sha="incorrect")
        self.cli("compare", "--reference", ref, "--candidate", other, "--max-absolute", 1,
                 "--output", report, ok=False)

    def test_truncated_predictions_and_scalar_inputs_fail(self):
        result = self.root / "truncated.npz"
        self.result(result)
        with np.load(result, allow_pickle=False) as data:
            fields = {k: data[k] for k in data.files}
        fields["prediction"] = fields["prediction"][:1]
        np.savez_compressed(result, **fields)
        self.cli("evaluate", "--input", self.data, "--results", result,
                 "--output", self.root / "score.json", "--metric",
                 "classification" if KIND == "DINOv2" else "regression", ok=False)
        np.savez_compressed(self.data, inputs=np.array(0, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            common.input_arrays(self.data)

    def test_fixture_inference_persists_performance(self):
        import torch

        class FixtureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))

        def predict(args, net, batch, target):
            shape = (len(batch), 1000) if KIND == "DINOv2" else target.shape
            prediction = torch.zeros(shape, dtype=torch.float32)
            if KIND == "Toto-2.0":
                return prediction, prediction.unsqueeze(0).expand(9, *shape)
            return prediction

        for filename in ("config.json", "model.safetensors"):
            (self.root / filename).write_bytes(b"fixture, not model weights")
        fixture = types.SimpleNamespace(
            MODEL="fixture-only", __file__=__file__,
            ASSETS={name: common.digest(self.root / name) for name in ("config.json", "model.safetensors")},
            validate_input=lambda *args: None,
            load_model=lambda *args: FixtureModel().eval(), predict=predict)
        output = self.root / "output.npz"
        args = Namespace(input=str(self.data), model_dir=str(self.root), output=str(output),
                         batch_size=1, warmup=1, device="cpu", head=None, head_sha256=None, task="fixture")
        common.infer(args, fixture)
        report = json.loads(Path(str(output) + ".performance.json").read_text())
        self.assertEqual(report["samples"], 2)
        self.assertEqual(len(report["batch_durations_ms"]), 2)
        self.assertEqual(report["result_sha256"], common.digest(output))

    def test_model_input_contract(self):
        task = "classification" if KIND == "DINOv2" else ("reconstruction" if KIND == "MOMENT" else "forecast")
        args = Namespace(input=str(self.data), head=None, head_sha256=None, task=task)
        if KIND == "DINOv2":
            args.head, args.head_sha256 = "head.pth", "a" * 64
        model.validate_input(args, self.x, self.y)
        if KIND == "MOMENT":
            args.task = "forecast"
            with self.assertRaisesRegex(ValueError, "trained head"):
                model.validate_input(args, self.x, np.zeros((2, 7, 192), dtype=np.float32))
        else:
            with self.assertRaises(ValueError):
                model.validate_input(args, self.x[:, :, :-1], self.y)

    def test_npu_unavailable_does_not_fallback(self):
        torch = types.ModuleType("torch")
        torch.npu = types.SimpleNamespace(is_available=lambda: False)
        with mock.patch.dict(sys.modules, {"torch": torch, "torch_npu": types.ModuleType("torch_npu")}):
            with self.assertRaisesRegex(RuntimeError, "NPU requested"):
                common.device_and_sync("npu")

    def test_cli_help_does_not_require_npu(self):
        self.cli("--help")

    def test_source_integrity_mismatch_fails(self):
        common.require_source(OfflineTests, common.digest(__file__))
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            common.require_source(OfflineTests, "0" * 64)

    def test_manifest_requires_exact_count_and_valid_class_map(self):
        root = self.root / "images"
        (root / "class_a").mkdir(parents=True)
        (root / "class_b").mkdir()
        (root / "class_a" / "a.jpg").write_bytes(b"fixture")
        (root / "class_b" / "b.jpg").write_bytes(b"fixture")
        labels = self.root / "labels.json"
        labels.write_text(json.dumps({"class_a": 3, "class_b": 9}))
        output = self.root / "manifest.csv"
        command = [sys.executable, str(Path(model.__file__).parent / "make_manifest.py"), "--root", str(root),
                   "--class-map", str(labels), "--output", str(output), "--expected-count", "2"]
        self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertEqual(json.loads(Path(str(output) + ".json").read_text())["count"], 2)
        shards = self.root / "shards"
        split_cmd = [sys.executable, str(Path(model.__file__).parent / "split_manifest.py"),
                     "--manifest", str(output), "--output-dir", str(shards), "--shard-size", "1"]
        self.assertEqual(subprocess.run(split_cmd, capture_output=True).returncode, 0)
        work = self.root / "work"
        for sub in ("inputs", "results", "scores"):
            (work / sub).mkdir(parents=True)
        for idx, label in enumerate((3, 9)):
            stem = f"shard-{idx:05d}"
            input_path = work / "inputs" / (stem + ".npz")
            result_path = work / "results" / (stem + ".npz")
            prep = {"manifest_sha256": common.digest(shards / (stem + ".csv"))}
            np.savez_compressed(input_path, inputs=np.zeros((1, 3, 224, 224), dtype=np.float32),
                                targets=np.array([label]), preparation=json.dumps(prep))
            logits = np.zeros((1, 1000), dtype=np.float32)
            logits[0, label] = 1
            meta = {"model": model.MODEL, "model_sha256": model.ASSETS["model.safetensors"],
                    "config_sha256": model.ASSETS["config.json"], "head_sha256": "a" * 64,
                    "input_sha256": common.digest(input_path), "dtype": "float32",
                    "device": "cpu", "task": "classification"}
            np.savez_compressed(result_path, prediction=logits, metadata=json.dumps(meta))
            self.cli("evaluate", "--input", input_path, "--results", result_path,
                     "--output", work / "scores" / (stem + ".json"), "--metric", "classification")
        aggregate = [sys.executable, str(Path(model.__file__).parent / "aggregate_metrics.py"),
                     "--manifest", str(output), "--shard-dir", str(shards), "--work-dir", str(work),
                     "--expected-count", "2", "--output", str(self.root / "aggregate.json")]
        self.assertEqual(subprocess.run(aggregate, capture_output=True).returncode, 0)
        self.assertEqual(json.loads((self.root / "aggregate.json").read_text())["top1"], 1.0)
        score_path = work / "scores" / "shard-00001.json"
        valid_score = score_path.read_text()
        tampered = json.loads(valid_score)
        tampered["top1"] = 0.0
        score_path.write_text(json.dumps(tampered))
        rejected = subprocess.run(aggregate, capture_output=True, text=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("scores differ from saved logits", rejected.stderr)
        score_path.write_text(valid_score)
        result_path = work / "results" / "shard-00001.npz"
        with np.load(result_path, allow_pickle=False) as data:
            fields = {k: data[k] for k in data.files}
        meta = json.loads(str(fields["metadata"].item()))
        meta["input_sha256"] = "other-input"
        fields["metadata"] = json.dumps(meta)
        np.savez_compressed(result_path, **fields)
        tampered = json.loads(valid_score)
        tampered["results_sha256"] = common.digest(result_path)
        score_path.write_text(json.dumps(tampered))
        self.assertNotEqual(subprocess.run(aggregate, capture_output=True).returncode, 0)
        (work / "scores" / "shard-00001.json").unlink()
        self.assertNotEqual(subprocess.run(aggregate, capture_output=True).returncode, 0)
        labels.write_text(json.dumps({"class_a": 3, "class_b": 3}))
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_predict_contract_with_mock_model(self):
        import torch
        batch = torch.from_numpy(self.x)
        args = Namespace(task="reconstruction")
        if KIND == "TTM-R2":
            def fake(**kwargs):
                return types.SimpleNamespace(prediction_outputs=torch.zeros((2, 96, 7)))
            output = model.predict(args, fake, batch, self.y)
            self.assertEqual(tuple(output.shape), (2, 96, 7))
        elif KIND == "Toto-2.0":
            class FakeToto:
                def forecast(self, inputs, horizon, **kwargs):
                    self.assertions = (inputs["target_mask"].all().item(), inputs["series_ids"].shape, kwargs)
                    return torch.arange(9, dtype=torch.float32).reshape(9, 1, 1, 1).expand(9, 2, 7, horizon)
            fake = FakeToto()
            median, quantiles = model.predict(args, fake, batch, self.y)
            self.assertEqual(float(median[0, 0, 0]), 4.0)
            self.assertEqual(tuple(quantiles.shape), (9, 2, 7, 96))
            self.assertTrue(fake.assertions[0])
            self.assertFalse(fake.assertions[2]["has_missing_values"])
        elif KIND == "MOMENT":
            class FakeMoment:
                def reconstruct(self, **kwargs):
                    return types.SimpleNamespace(reconstruction=kwargs["x_enc"])
            output = model.predict(args, FakeMoment(), batch, self.y)
            self.assertEqual(tuple(output.shape), (2, 7, 512))
        else:
            output = model.predict(args, lambda value: torch.zeros((len(value), 1000)), batch, self.y)
            self.assertEqual(tuple(output.shape), (2, 1000))


if __name__ == "__main__":
    unittest.main()
