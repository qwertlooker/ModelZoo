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

import timesfm_npu as adapter


class OfflineWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "etth1.csv"
        with self.data.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", *adapter.CHANNELS])
            start = datetime(2020, 1, 1)
            for i in range(20):
                writer.writerow([(start + timedelta(hours=i)).isoformat(sep=" "), *[i + n for n in range(7)]])
        self.manifest = self.root / "manifest.json"
        self.run_cli("prepare", "--dataset", self.data, "--output", self.manifest,
                     "--expected-sha256", adapter.sha256(self.data), "--test-start", 10,
                     "--test-end", 16, "--context", 4, "--horizon", 2, "--stride", 2)

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, *args, success=True):
        cmd = [sys.executable, str(Path(adapter.__file__).resolve()), *map(str, args)]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode == 0, success, result.stderr)
        return result

    def write_results(self, path, delta=0, alter_truth=False):
        manifest = adapter.load_manifest(self.manifest)
        columns = adapter.read_series(self.data)
        meta = {"type": "metadata", "manifest_sha256": adapter.sha256(self.manifest),
                "dataset_sha256": manifest["dataset_sha256"], "model_sha256": adapter.MODEL_SHA,
                "config_sha256": adapter.CONFIG_SHA, "device": "cpu", "dtype": "float32",
                "context": manifest["context"], "horizon": manifest["horizon"]}
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(meta) + "\n")
            for batch in adapter.batches(manifest, columns, 5):
                for sample_id, _, truth in batch:
                    pred = [v + delta for v in truth]
                    if alter_truth:
                        truth[0] += 1
                    f.write(json.dumps({"id": sample_id, "truth": truth, "prediction": pred}) + "\n")

    def test_prepare_windows_and_metrics(self):
        manifest = adapter.load_manifest(self.manifest)
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
        self.run_cli("evaluate", "--manifest", self.manifest, "--results", predictions,
                     "--output", self.root / "report.json", success=False)
        self.data.write_text(self.data.read_text() + "\n")
        self.run_cli("infer", "--model-dir", self.root, "--manifest", self.manifest,
                     "--output", self.root / "out.jsonl", success=False)

    def test_compare_threshold_and_horizon_failure(self):
        ref, candidate = self.root / "ref.jsonl", self.root / "candidate.jsonl"
        self.write_results(ref)
        self.write_results(candidate, delta=0.25)
        self.run_cli("compare", "--reference", ref, "--candidate", candidate,
                     "--max-absolute", 0.2, "--output", self.root / "diff.json", success=False)
        self.run_cli("compare", "--reference", ref, "--candidate", candidate,
                     "--max-absolute", 0.3, "--output", self.root / "diff.json")
        self.run_cli("prepare", "--dataset", self.data, "--output", self.root / "invalid.json",
                     "--expected-sha256", adapter.sha256(self.data), "--horizon", 129, success=False)

    def test_requested_npu_unavailable_fails_without_cpu_fallback(self):
        torch = types.ModuleType("torch")
        torch.npu = types.SimpleNamespace(is_available=lambda: False)
        torch_npu = types.ModuleType("torch_npu")
        args = Namespace(batch_size=1, warmup=0, manifest="manifest", model_dir="model", device="npu")

        def asset_sha(path):
            return adapter.MODEL_SHA if path.name == "model.safetensors" else adapter.CONFIG_SHA

        with mock.patch.object(adapter, "load_manifest", return_value={"dataset_path": "data"}), \
             mock.patch.object(adapter, "read_series", return_value={}), \
             mock.patch.object(adapter, "sha256", side_effect=asset_sha), \
             mock.patch.dict(sys.modules, {"torch": torch, "torch_npu": torch_npu}):
            with self.assertRaisesRegex(RuntimeError, "NPU requested"):
                adapter.infer(args)


if __name__ == "__main__":
    unittest.main()
