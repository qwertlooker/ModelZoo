import json
import tempfile
import unittest
from pathlib import Path

import importlib.util

ROOT = Path(__file__).resolve().parents[2]


class PerformanceTests(unittest.TestCase):
    def test_independent_npu_runs_and_failure_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, timings in enumerate(
                ([100.0, 200.0], [200.0, 400.0], [300.0, 600.0])
            ):
                p = Path(tmp) / f"run{i}.json"
                p.write_text(
                    json.dumps(
                        {
                            "schema": "synchronized-model-timing-v1",
                            "run_id": str(i),
                            "identity": {
                                "device": "npu",
                                "input_sha256": "fixture",
                                "batch_size": 8,
                            },
                            "samples": 9,
                            "batch_durations_ms": timings,
                            "model_seconds": sum(timings) / 1000,
                            "samples_per_second": 9000 / sum(timings),
                            "scope": "fixture only",
                        }
                    )
                )
                paths.append(p)
            report = self.adapter.summarize_performance(paths)
            self.assertEqual(report["median_model_seconds"], 0.6)
            self.assertEqual(report["median_samples_per_second"], 15)
            for invalid in (paths[:2], [paths[0]] * 3):
                with self.assertRaises(ValueError):
                    self.adapter.summarize_performance(invalid)
            original = paths[2].read_text()
            for key, value in (
                ("device", "cpu"),
                ("input_sha256", "other"),
                ("batch_size", 1),
            ):
                data = json.loads(original)
                data["identity"][key] = value
                paths[2].write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    self.adapter.summarize_performance(paths)
            data = json.loads(original)
            data["batch_durations_ms"] = [float("nan")]
            paths[2].write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                self.adapter.summarize_performance(paths)

    def test_cpu_sidecar_is_recorded_but_cannot_certify_npu(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output.bin"
            output.write_bytes(b"fixture")
            report = self.adapter.write_report(
                output, {"device": "cpu"}, [10.0, 20.0], 3
            )
            self.assertEqual(report["samples_per_second"], 100)
            self.assertEqual(report["result_sha256"], self.digest(output))
            self.assertTrue(Path(str(output) + ".performance.json").is_file())
            with self.assertRaises(ValueError):
                self.adapter.write_report(output, {"device": "cpu"}, [], 3)


def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    for folder in ("TimesFM-2.5", "TTM-R2", "DINOv2", "Toto-2.0", "MOMENT"):
        entry = "timesfm_npu.py" if folder == "TimesFM-2.5" else "model.py"
        spec = importlib.util.spec_from_file_location(
            "performance_contract_" + folder, ROOT / folder / entry
        )
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        for method in loader.getTestCaseNames(PerformanceTests):
            case = PerformanceTests(method)
            case.adapter = adapter
            case.digest = adapter.sha256 if folder == "TimesFM-2.5" else adapter.digest
            suite.addTest(case)
    return suite


if __name__ == "__main__":
    unittest.main()
