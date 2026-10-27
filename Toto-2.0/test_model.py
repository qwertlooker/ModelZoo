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

KIND = "Toto-2.0"


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

    def test_quantile_median_must_match_point_forecast(self):
        result = self.root / "bad-median.npz"
        self.result(result)
        with np.load(result, allow_pickle=False) as data:
            fields = {k: data[k] for k in data.files}
        fields["quantiles"] = fields["quantiles"].copy()
        fields["quantiles"][4] += 1
        np.savez_compressed(result, **fields)
        self.cli("evaluate", "--input", self.data, "--results", result,
                 "--output", self.root / "score.json", "--metric", "regression", ok=False)
        self.cli("compare", "--reference", result, "--candidate", result,
                 "--max-absolute", 0, "--output", self.root / "compare.json", ok=False)

    def test_wql_uses_float64_accumulation(self):
        self.y = np.full_like(self.y, 3e38)
        self.pred = np.zeros_like(self.y)
        np.savez_compressed(self.data, inputs=self.x, targets=self.y)
        result, report = self.root / "large.npz", self.root / "score.json"
        self.result(result)
        self.cli("evaluate", "--input", self.data, "--results", result,
                 "--output", report, "--metric", "regression")
        self.assertAlmostEqual(json.loads(report.read_text())["mean_wql"], 1.0)

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

    def test_checkpoint_configuration_mismatch_fails(self):
        model.verify_config({"patch_size": 32}, {"patch_size": 32})
        with self.assertRaisesRegex(ValueError, "config"):
            model.verify_config({"patch_size": 32}, {"patch_size": 64})

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
