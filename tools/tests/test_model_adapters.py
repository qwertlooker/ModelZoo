"""Offline regression for the five standalone model deliveries; no NPU certification."""

import importlib.util
import csv
import json
import subprocess
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_entry(folder, filename=None, model_module=None):
    if filename is None:
        filename = "timesfm_npu.py" if folder == "TimesFM-2.5" else "model.py"
    spec = importlib.util.spec_from_file_location("delivery_" + folder + "_" + filename, ROOT / folder / filename)
    module = importlib.util.module_from_spec(spec)
    if model_module is None:
        spec.loader.exec_module(module)
    else:
        with mock.patch.dict(sys.modules, {"model": model_module}):
            spec.loader.exec_module(module)
    return module


class TimesFMTests(unittest.TestCase):
    adapter = load_entry("TimesFM-2.5")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "etth1.csv"
        with self.data.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", *self.adapter.CHANNELS])
            start = datetime(2020, 1, 1)
            for i in range(20):
                writer.writerow([(start + timedelta(hours=i)).isoformat(sep=" "), *[i + n for n in range(7)]])
        self.manifest = self.root / "manifest.json"
        self.run_cli(
            "prepare",
            "--dataset",
            self.data,
            "--output",
            self.manifest,
            "--expected-sha256",
            self.adapter.sha256(self.data),
            "--test-start",
            10,
            "--test-end",
            16,
            "--context",
            4,
            "--horizon",
            2,
            "--stride",
            2,
        )

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, *args, success=True):
        cmd = [sys.executable, str(Path(self.adapter.__file__).resolve()), *map(str, args)]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode == 0, success, result.stderr)
        return result

    def write_results(self, path, delta=0, alter_truth=False):
        manifest = self.adapter.load_manifest(self.manifest)
        columns = self.adapter.read_series(self.data)
        meta = {
            "type": "metadata",
            "manifest_sha256": self.adapter.sha256(self.manifest),
            "dataset_sha256": manifest["dataset_sha256"],
            "model_sha256": self.adapter.MODEL_SHA,
            "config_sha256": self.adapter.CONFIG_SHA,
            "device": "cpu",
            "dtype": "float32",
            "context": manifest["context"],
            "horizon": manifest["horizon"],
        }
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(meta) + "\n")
            for batch in self.adapter.batches(manifest, columns, 5):
                for sample_id, _, truth in batch:
                    pred = [v + delta for v in truth]
                    if alter_truth:
                        truth[0] += 1
                    f.write(json.dumps({"id": sample_id, "truth": truth, "prediction": pred}) + "\n")

    def test_prepare_windows_and_metrics(self):
        manifest = self.adapter.load_manifest(self.manifest)
        self.assertEqual(manifest["window_count"], 3)
        self.assertEqual(manifest["series_count"], 21)
        predictions = self.root / "predictions.jsonl"
        self.write_results(predictions, delta=2)
        report = self.root / "report.json"
        self.run_cli("evaluate", "--manifest", self.manifest, "--results", predictions, "--output", report)
        result = json.loads(report.read_text())
        self.assertEqual((result["mse"], result["mae"], result["values"]), (4.0, 2.0, 42))

    def test_mismatched_data_and_truth_fail(self):
        predictions = self.root / "bad.jsonl"
        self.write_results(predictions, alter_truth=True)
        self.run_cli(
            "evaluate",
            "--manifest",
            self.manifest,
            "--results",
            predictions,
            "--output",
            self.root / "report.json",
            success=False,
        )
        self.data.write_text(self.data.read_text() + "\n")
        self.run_cli(
            "infer",
            "--model-dir",
            self.root,
            "--manifest",
            self.manifest,
            "--output",
            self.root / "out.jsonl",
            success=False,
        )

    def test_compare_threshold_and_horizon_failure(self):
        ref, candidate = (self.root / "ref.jsonl", self.root / "candidate.jsonl")
        self.write_results(ref)
        self.write_results(candidate, delta=0.25)
        self.run_cli(
            "compare",
            "--reference",
            ref,
            "--candidate",
            candidate,
            "--max-absolute",
            0.2,
            "--output",
            self.root / "diff.json",
            success=False,
        )
        self.run_cli(
            "compare",
            "--reference",
            ref,
            "--candidate",
            candidate,
            "--max-absolute",
            0.3,
            "--output",
            self.root / "diff.json",
        )
        self.run_cli(
            "prepare",
            "--dataset",
            self.data,
            "--output",
            self.root / "invalid.json",
            "--expected-sha256",
            self.adapter.sha256(self.data),
            "--horizon",
            129,
            success=False,
        )

    def test_requested_npu_unavailable_fails_without_cpu_fallback(self):
        torch = types.ModuleType("torch")
        torch.npu = types.SimpleNamespace(is_available=lambda: False)
        torch_npu = types.ModuleType("torch_npu")
        args = Namespace(batch_size=1, warmup=0, manifest="manifest", model_dir="model", device="npu")

        def asset_sha(path):
            return self.adapter.MODEL_SHA if path.name == "model.safetensors" else self.adapter.CONFIG_SHA

        with (
            mock.patch.object(self.adapter, "load_manifest", return_value={"dataset_path": "data", "dataset_rows": 1}),
            mock.patch.object(self.adapter, "read_series", return_value={"HUFL": [1]}),
            mock.patch.object(self.adapter, "sha256", side_effect=asset_sha),
            mock.patch.dict(sys.modules, {"torch": torch, "torch_npu": torch_npu}),
        ):
            with self.assertRaisesRegex(RuntimeError, "NPU requested"):
                self.adapter.infer(args)

    def test_evaluation_rejects_wrong_model_or_horizon_metadata(self):
        predictions = self.root / "predictions.jsonl"
        self.write_results(predictions)
        original = predictions.read_text()
        for key, value in (("model_sha256", "wrong"), ("context", 5), ("horizon", 3)):
            lines = original.splitlines()
            metadata = json.loads(lines[0])
            metadata[key] = value
            predictions.write_text("\n".join([json.dumps(metadata), *lines[1:]]) + "\n")
            self.run_cli(
                "evaluate",
                "--manifest",
                self.manifest,
                "--results",
                predictions,
                "--output",
                self.root / "report.json",
                success=False,
            )

    def test_transformers_version_is_enforced(self):
        with mock.patch("importlib.metadata.version", return_value="5.3.0"):
            self.adapter.require_transformers_version()
        with mock.patch("importlib.metadata.version", return_value="5.2.0"):
            with self.assertRaisesRegex(RuntimeError, "transformers==5.3.0"):
                self.adapter.require_transformers_version()


class OfflineModelChecks:
    metric = "regression"
    task = "forecast"
    quantiles = False

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / "input.npz"
        prep = self.make_fixture()
        np.savez_compressed(self.data, inputs=self.x, targets=self.y, preparation=json.dumps(prep))

    def tearDown(self):
        self.tmp.cleanup()

    def cli(self, *args, ok=True):
        result = subprocess.run(
            [sys.executable, str(self.adapter.__file__), *map(str, args)],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertEqual(result.returncode == 0, ok, result.stderr)
        return result

    def result(self, path, offset=0, input_sha=None):
        meta = {
            "model": self.adapter.MODEL,
            "model_sha256": self.adapter.ASSETS["model.safetensors"],
            "config_sha256": self.adapter.ASSETS["config.json"],
            "head_sha256": None,
            "input_sha256": input_sha or self.adapter.digest(self.data),
            "dtype": "float32",
            "device": "cpu",
            "task": self.task,
        }
        fields = {"prediction": self.pred + offset, "metadata": json.dumps(meta)}
        if self.quantiles:
            fields["quantiles"] = np.stack([self.pred + offset] * 9)
        np.savez_compressed(path, **fields)

    def test_evaluation_and_comparison_fail_closed(self):
        ref, other, report = self.root / "reference.npz", self.root / "other.npz", self.root / "report.json"
        self.result(ref)
        self.result(other, offset=0.25)
        self.cli("evaluate", "--input", self.data, "--results", ref, "--output", report, "--metric", self.metric)
        score = json.loads(report.read_text())
        key, expected = ("top1", 1.0) if self.metric == "classification" else ("mse", 0.0)
        self.assertEqual(score[key], expected)
        if self.quantiles:
            self.assertEqual(score["mean_wql"], 0.0)
        for threshold, ok in ((0.1, False), (0.3, True)):
            self.cli(
                "compare",
                "--reference",
                ref,
                "--candidate",
                other,
                "--max-absolute",
                threshold,
                "--output",
                report,
                ok=ok,
            )
        self.result(other, input_sha="incorrect")
        self.cli("compare", "--reference", ref, "--candidate", other, "--max-absolute", 1, "--output", report, ok=False)

    def test_truncated_predictions_and_scalar_inputs_fail(self):
        result = self.root / "truncated.npz"
        self.result(result)
        with np.load(result, allow_pickle=False) as data:
            fields = {k: data[k] for k in data.files}
        fields["prediction"] = fields["prediction"][:1]
        np.savez_compressed(result, **fields)
        self.cli(
            "evaluate",
            "--input",
            self.data,
            "--results",
            result,
            "--output",
            self.root / "score.json",
            "--metric",
            self.metric,
            ok=False,
        )
        np.savez_compressed(self.data, inputs=np.array(0, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.adapter.input_arrays(self.data)

    def test_fixture_inference_persists_performance(self):
        import torch

        class FixtureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))

        def predict(args, net, batch, target):
            prediction = torch.zeros((len(batch), *self.pred.shape[1:]), dtype=torch.float32)
            return (prediction, prediction.unsqueeze(0).expand(9, *prediction.shape)) if self.quantiles else prediction

        for filename in ("config.json", "model.safetensors"):
            (self.root / filename).write_bytes(b"fixture, not model weights")
        assets = {name: self.adapter.digest(self.root / name) for name in ("config.json", "model.safetensors")}
        output = self.root / "output.npz"
        args = Namespace(
            input=str(self.data),
            model_dir=str(self.root),
            output=str(output),
            batch_size=1,
            warmup=1,
            device="cpu",
            head=None,
            head_sha256=None,
            task="fixture",
        )
        with mock.patch.multiple(
            self.adapter,
            MODEL="fixture-only",
            ASSETS=assets,
            validate_input=lambda *args: None,
            load_model=lambda *args: FixtureModel().eval(),
            predict=predict,
        ):
            self.adapter.infer(args)
        report = json.loads(Path(str(output) + ".performance.json").read_text())
        self.assertEqual(report["samples"], 2)
        self.assertEqual(len(report["batch_durations_ms"]), 2)
        self.assertEqual(report["result_sha256"], self.adapter.digest(output))

    def test_model_input_contract(self):
        args = Namespace(input=str(self.data), head=None, head_sha256=None, task=self.task)
        if self.metric == "classification":
            args.head, args.head_sha256 = "head.pth", "a" * 64
        self.adapter.validate_input(args, self.x, self.y)
        with self.assertRaises(ValueError):
            self.adapter.validate_input(args, self.x[:, :, :-1], self.y)

    def test_npu_unavailable_does_not_fallback(self):
        torch = types.ModuleType("torch")
        torch.npu = types.SimpleNamespace(is_available=lambda: False)
        with mock.patch.dict(sys.modules, {"torch": torch, "torch_npu": types.ModuleType("torch_npu")}):
            with self.assertRaisesRegex(RuntimeError, "NPU requested"):
                self.adapter.device_and_sync("npu")

    def test_cli_help_does_not_require_npu(self):
        self.cli("--help")

    def test_source_integrity_mismatch_fails(self):
        self.adapter.require_source(type(self), self.adapter.digest(__file__))
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.adapter.require_source(type(self), "0" * 64)


class TTMTests(OfflineModelChecks, unittest.TestCase):
    adapter = load_entry("TTM-R2")

    def make_fixture(self):
        self.x = np.zeros((2, 512, 7), dtype=np.float32)
        self.y = np.ones((2, 96, 7), dtype=np.float32)
        self.pred = self.y.copy()
        prep = {"model": "TTM-R2", "task": "forecast", "scaling": "etth1_train_standard"}
        return prep

    def test_checkpoint_configuration_mismatch_fails(self):
        self.adapter.verify_config({"length": 512}, {"length": 512})
        self.adapter.verify_config(
            {"torch_dtype": "float32", "transformers_version": "4.37.2"},
            {"dtype": "float32", "transformers_version": "4.57.6"},
        )
        with self.assertRaisesRegex(ValueError, "config"):
            self.adapter.verify_config({"length": 512, "unknown": 1}, {"length": 512})
        with self.assertRaisesRegex(ValueError, "config"):
            self.adapter.verify_config({"torch_dtype": "float32"}, {"dtype": "float16"})

    def test_predict_contract_with_mock_model(self):
        import torch

        batch = torch.from_numpy(self.x)
        args = Namespace(task="reconstruction")

        def fake(**kwargs):
            return types.SimpleNamespace(prediction_outputs=torch.zeros((2, 96, 7)))

        output = self.adapter.predict(args, fake, batch, self.y)
        self.assertEqual(tuple(output.shape), (2, 96, 7))


class DINOTests(OfflineModelChecks, unittest.TestCase):
    adapter = load_entry("DINOv2")
    metric = "classification"
    task = "classification"

    def make_fixture(self):
        self.x = np.zeros((2, 3, 224, 224), dtype=np.float32)
        self.y = np.array([1, 2], dtype=np.int64)
        self.pred = np.zeros((2, 1000), dtype=np.float32)
        self.pred[0, 1] = self.pred[1, 2] = 2
        prep = {"preprocessor_sha256": self.adapter.ASSETS["preprocessor_config.json"], "count": 2}
        return prep

    def test_unified_data_cli_and_image_preparation_fixture(self):
        from PIL import Image

        prepare_data = load_entry("DINOv2", "prepare_data.py", self.adapter)
        entry = str(Path(self.adapter.__file__).parent / "prepare_data.py")
        for command in ("manifest", "split", "images"):
            result = subprocess.run([sys.executable, entry, command, "--help"], capture_output=True)
            self.assertEqual(result.returncode, 0)
        invalid = subprocess.run(
            [
                sys.executable,
                entry,
                "split",
                "--manifest",
                str(self.root / "missing.csv"),
                "--output-dir",
                str(self.root / "shards"),
                "--shard-size",
                "0",
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("shard-size must be positive", invalid.stderr)
        image = self.root / "sample.png"
        Image.new("RGB", (8, 8)).save(image)
        manifest = self.root / "images.csv"
        manifest.write_text("path,label\nsample.png,3\n", encoding="utf-8")
        output = self.root / "pixels.npz"

        def processor(**kwargs):
            return {"pixel_values": np.zeros((1, 3, 224, 224), dtype=np.float32)}

        transformers = types.ModuleType("transformers")
        transformers.AutoImageProcessor = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor)
        args = Namespace(model_dir=str(self.root), manifest=str(manifest), output=str(output))
        with (
            mock.patch.dict(sys.modules, {"transformers": transformers}),
            mock.patch.object(self.adapter, "require_package"),
            mock.patch.object(self.adapter, "checked_file"),
        ):
            prepare_data.prepare_images(args)
        with np.load(output, allow_pickle=False) as data:
            self.assertEqual(data["inputs"].shape, (1, 3, 224, 224))
            self.assertEqual(data["targets"].tolist(), [3])
            self.assertEqual(json.loads(str(data["preparation"].item()))["count"], 1)

    def test_manifest_requires_exact_count_and_valid_class_map(self):
        root = self.root / "images"
        (root / "class_a").mkdir(parents=True)
        (root / "class_b").mkdir()
        (root / "class_a" / "a.jpg").write_bytes(b"fixture")
        (root / "class_b" / "b.jpg").write_bytes(b"fixture")
        labels = self.root / "labels.json"
        labels.write_text(json.dumps({"class_a": 3, "class_b": 9}))
        output = self.root / "manifest.csv"
        command = [
            sys.executable,
            str(Path(self.adapter.__file__).parent / "prepare_data.py"),
            "manifest",
            "--root",
            str(root),
            "--class-map",
            str(labels),
            "--output",
            str(output),
            "--expected-count",
            "2",
        ]
        self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertEqual(json.loads(Path(str(output) + ".json").read_text())["count"], 2)
        shards = self.root / "shards"
        split_cmd = [
            sys.executable,
            str(Path(self.adapter.__file__).parent / "prepare_data.py"),
            "split",
            "--manifest",
            str(output),
            "--output-dir",
            str(shards),
            "--shard-size",
            "1",
        ]
        self.assertEqual(subprocess.run(split_cmd, capture_output=True).returncode, 0)
        work = self.root / "work"
        for sub in ("inputs", "results", "scores"):
            (work / sub).mkdir(parents=True)
        for idx, label in enumerate((3, 9)):
            stem = f"shard-{idx:05d}"
            input_path = work / "inputs" / (stem + ".npz")
            result_path = work / "results" / (stem + ".npz")
            prep = {"manifest_sha256": self.adapter.digest(shards / (stem + ".csv"))}
            np.savez_compressed(
                input_path,
                inputs=np.zeros((1, 3, 224, 224), dtype=np.float32),
                targets=np.array([label]),
                preparation=json.dumps(prep),
            )
            logits = np.zeros((1, 1000), dtype=np.float32)
            logits[0, label] = 1
            meta = {
                "model": self.adapter.MODEL,
                "model_sha256": self.adapter.ASSETS["model.safetensors"],
                "config_sha256": self.adapter.ASSETS["config.json"],
                "head_sha256": "a" * 64,
                "input_sha256": self.adapter.digest(input_path),
                "dtype": "float32",
                "device": "cpu",
                "task": "classification",
            }
            np.savez_compressed(result_path, prediction=logits, metadata=json.dumps(meta))
            self.cli(
                "evaluate",
                "--input",
                input_path,
                "--results",
                result_path,
                "--output",
                work / "scores" / (stem + ".json"),
                "--metric",
                "classification",
            )
        aggregate = [
            sys.executable,
            str(Path(self.adapter.__file__).parent / "aggregate_metrics.py"),
            "--manifest",
            str(output),
            "--shard-dir",
            str(shards),
            "--work-dir",
            str(work),
            "--expected-count",
            "2",
            "--output",
            str(self.root / "aggregate.json"),
        ]
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
        tampered["results_sha256"] = self.adapter.digest(result_path)
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
        output = self.adapter.predict(args, lambda value: torch.zeros((len(value), 1000)), batch, self.y)
        self.assertEqual(tuple(output.shape), (2, 1000))


class TotoTests(OfflineModelChecks, unittest.TestCase):
    adapter = load_entry("Toto-2.0")
    quantiles = True

    def make_fixture(self):
        self.x = np.zeros((2, 7, 512), dtype=np.float32)
        self.y = np.ones((2, 7, 96), dtype=np.float32)
        self.pred = self.y.copy()
        prep = {"model": "Toto-2.0", "task": "forecast", "scaling": "none"}
        return prep

    def test_quantile_median_must_match_point_forecast(self):
        result = self.root / "bad-median.npz"
        self.result(result)
        with np.load(result, allow_pickle=False) as data:
            fields = {k: data[k] for k in data.files}
        fields["quantiles"] = fields["quantiles"].copy()
        fields["quantiles"][4] += 1
        np.savez_compressed(result, **fields)
        self.cli(
            "evaluate",
            "--input",
            self.data,
            "--results",
            result,
            "--output",
            self.root / "score.json",
            "--metric",
            "regression",
            ok=False,
        )
        self.cli(
            "compare",
            "--reference",
            result,
            "--candidate",
            result,
            "--max-absolute",
            0,
            "--output",
            self.root / "compare.json",
            ok=False,
        )

    def test_wql_uses_float64_accumulation(self):
        self.y = np.full_like(self.y, 3e38)
        self.pred = np.zeros_like(self.y)
        np.savez_compressed(self.data, inputs=self.x, targets=self.y)
        result, report = (self.root / "large.npz", self.root / "score.json")
        self.result(result)
        self.cli("evaluate", "--input", self.data, "--results", result, "--output", report, "--metric", "regression")
        self.assertAlmostEqual(json.loads(report.read_text())["mean_wql"], 1.0)

    def test_checkpoint_configuration_mismatch_fails(self):
        self.adapter.verify_config({"patch_size": 32}, {"patch_size": 32})
        with self.assertRaisesRegex(ValueError, "config"):
            self.adapter.verify_config({"patch_size": 32}, {"patch_size": 64})

    def test_predict_contract_with_mock_model(self):
        import torch

        batch = torch.from_numpy(self.x)
        args = Namespace(task="reconstruction")

        class FakeToto:
            def forecast(self, inputs, horizon, **kwargs):
                self.assertions = (inputs["target_mask"].all().item(), inputs["series_ids"].shape, kwargs)
                return torch.arange(9, dtype=torch.float32).reshape(9, 1, 1, 1).expand(9, 2, 7, horizon)

        fake = FakeToto()
        median, quantiles = self.adapter.predict(args, fake, batch, self.y)
        self.assertEqual(float(median[0, 0, 0]), 4.0)
        self.assertEqual(tuple(quantiles.shape), (9, 2, 7, 96))
        self.assertTrue(fake.assertions[0])
        self.assertFalse(fake.assertions[2]["has_missing_values"])


class MomentTests(OfflineModelChecks, unittest.TestCase):
    adapter = load_entry("MOMENT")
    task = "reconstruction"

    def make_fixture(self):
        self.x = np.zeros((2, 7, 512), dtype=np.float32)
        self.y = self.x.copy()
        self.pred = self.y.copy()
        prep = {"model": "MOMENT", "task": "reconstruction", "scaling": "etth1_train_standard"}
        return prep

    def test_training_rejects_nonfinite_learning_rate_before_loading(self):
        command = [
            sys.executable,
            str(Path(self.adapter.__file__).parent / "train_head.py"),
            "--model-dir",
            str(self.root),
            "--train",
            str(self.data),
            "--output",
            str(self.root / "head.safetensors"),
            "--learning-rate",
            "nan",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid output or training parameters", result.stderr)

    def test_model_input_contract(self):
        task = "reconstruction"
        args = Namespace(input=str(self.data), head=None, head_sha256=None, task=task)
        self.adapter.validate_input(args, self.x, self.y)
        args.task = "forecast"
        with self.assertRaisesRegex(ValueError, "trained head"):
            self.adapter.validate_input(args, self.x, np.zeros((2, 7, 192), dtype=np.float32))
        args.head = str(self.root / "head.safetensors")
        args.head_sha256 = "a" * 64
        Path(args.head + ".json").write_text(
            json.dumps(
                {
                    "head_sha256": args.head_sha256,
                    "steps": 0,
                    "train_split": "train",
                    "horizon": 192,
                    "upstream_model_sha256": self.adapter.ASSETS["model.safetensors"],
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "metadata"):
            self.adapter.validate_input(args, self.x, np.zeros((2, 7, 192), dtype=np.float32))

    def test_predict_contract_with_mock_model(self):
        import torch

        batch = torch.from_numpy(self.x)
        args = Namespace(task="reconstruction")

        class FakeMoment:
            def reconstruct(self, **kwargs):
                return types.SimpleNamespace(reconstruction=kwargs["x_enc"])

        output = self.adapter.predict(args, FakeMoment(), batch, self.y)
        self.assertEqual(tuple(output.shape), (2, 7, 512))


class ModelCLITests(unittest.TestCase):
    def test_unsupported_tasks_and_metrics_fail_during_argument_parsing(self):
        for folder in ("TTM-R2", "Toto-2.0", "MOMENT", "DINOv2"):
            entry = ROOT / folder / "model.py"
            metric = "regression" if folder == "DINOv2" else "classification"
            command = [
                sys.executable,
                str(entry),
                "evaluate",
                "--input",
                "missing.npz",
                "--results",
                "missing.npz",
                "--output",
                "unused.json",
                "--metric",
                metric,
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("invalid choice", result.stderr)
        for folder in ("TTM-R2", "Toto-2.0"):
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / folder / "prepare_series.py"),
                    "--dataset",
                    "missing.csv",
                    "--output",
                    "unused.npz",
                    "--task",
                    "reconstruction",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("invalid choice", result.stderr)


class ETTh1PreparationTests(unittest.TestCase):
    def test_layout_split_and_training_scaler_with_synthetic_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "fixture.csv"
            raw = np.arange(14400, dtype=np.float64)[:, None] + np.arange(7)
            with dataset.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"])
                for i, row in enumerate(raw):
                    writer.writerow([(datetime(2020, 1, 1) + timedelta(hours=i)).isoformat(), *row])
            for folder in ("TTM-R2", "Toto-2.0", "MOMENT"):
                prepare = load_entry(folder, "prepare_series.py")
                horizon = 192 if folder == "MOMENT" else 96
                scaled = raw if folder == "Toto-2.0" else (raw - raw[:8640].mean(axis=0)) / raw[:8640].std(axis=0)
                for split, start, end in (("train", 512, 8640), ("test", 11520, 14400)):
                    with self.subTest(model=folder, split=split):
                        output = root / f"{folder}-{split}.npz"
                        prepare.prepare(
                            dataset,
                            output,
                            split,
                            1024,
                            "forecast",
                            horizon,
                            prepare.hashlib.sha256(dataset.read_bytes()).hexdigest(),
                        )
                        with np.load(output, allow_pickle=False) as data:
                            expected = scaled[start - 512 : start].astype(np.float32)
                            expected_targets = scaled[start : start + horizon].astype(np.float32)
                            if folder != "TTM-R2":
                                expected, expected_targets = (expected.T, expected_targets.T)
                            np.testing.assert_array_equal(data["inputs"][0], expected)
                            np.testing.assert_array_equal(data["targets"][0], expected_targets)
                            self.assertEqual(len(data["inputs"]), len(range(start, end - horizon + 1, 1024)))
                if folder == "MOMENT":
                    output = root / "reconstruction.npz"
                    prepare.prepare(
                        dataset,
                        output,
                        "test",
                        1024,
                        "reconstruction",
                        512,
                        prepare.hashlib.sha256(dataset.read_bytes()).hexdigest(),
                    )
                    with np.load(output, allow_pickle=False) as data:
                        np.testing.assert_array_equal(data["inputs"], data["targets"])
                        self.assertEqual(json.loads(str(data["preparation"].item()))["split_start"], 12032)


if __name__ == "__main__":
    unittest.main()
