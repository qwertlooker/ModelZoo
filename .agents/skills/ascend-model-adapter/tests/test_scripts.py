from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_DIR / "scripts"


def load_script(name: str):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ModelZooSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_script("modelzoo_sampler.py")

    def test_filter_and_non_destructive_clone(self):
        entry = self.module.Entry(
            "audio", "DemoASR", "/x", "1 天前", 1, 1, "vLLM route"
        )
        self.assertEqual(
            [entry], self.module.filter_entries([entry], {"audio"}, ["vllm"])
        )
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "existing"
            destination.mkdir()
            marker = destination / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                self.module.clone_sparse([entry], destination)
            self.assertEqual("keep", marker.read_text(encoding="utf-8"))


class ReviewSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_script("gitcode_pr_review_sampler.py")

    def test_exact_pr_path_and_noise(self):
        mapping = self.module.parse_pr_path(["12=ACL_PyTorch/built-in/cv/Demo"])
        self.assertEqual({12: ["ACL_PyTorch/built-in/cv/Demo"]}, mapping)
        self.assertEqual(
            ["ACL_PyTorch/built-in/cv/Demo"],
            self.module.match_paths(
                ["ACL_PyTorch/built-in/cv/Demo/README.md"], mapping[12]
            ),
        )
        self.assertTrue(self.module.is_noise("changed the description"))
        self.assertTrue(self.module.is_noise("AI 代码审查执行失败，请稍后重试"))

    def test_fetches_full_discussion_page_and_sorts_newest_first(self):
        module = self.module

        class FakeSession:
            def __init__(self):
                self.urls = []

            def get(self, url, headers, timeout):
                self.urls.append(url)
                if url.endswith("/pull/12"):
                    return module.HttpResponse(200, "{}")
                if url.endswith("/isource/merge_requests/12"):
                    return module.HttpResponse(
                        200, json.dumps({"iid": 12, "title": "Demo", "sha": "abc"})
                    )
                return module.HttpResponse(
                    200,
                    json.dumps(
                        {
                            "total": 2,
                            "content": {
                                "data": [
                                    {
                                        "notes": [
                                            {
                                                "body": "old review",
                                                "created_at": "2026-07-13T10:00:00",
                                                "author": {"username": "bot"},
                                            }
                                        ]
                                    },
                                    {
                                        "notes": [
                                            {
                                                "body": "new review",
                                                "created_at": "2026-07-13T11:00:00",
                                                "author": {"username": "bot"},
                                            },
                                        ]
                                    }
                                ]
                            },
                        }
                    ),
                )

        session = FakeSession()
        detail, notes = module.fetch_pr(
            session, "https://gitcode.example", "Ascend/ModelZoo-PyTorch", 12, {}
        )
        self.assertIn("page=1&per_page=100", session.urls[-1])
        self.assertEqual(2, detail["_discussion_total"])
        self.assertEqual(2, detail["_discussion_fetched"])
        self.assertEqual(
            ["new review", "old review"],
            [note.body for note in module.newest_notes_first(notes)],
        )
        duplicate_statuses = [
            module.Note(
                12, "Demo", "robot", "pipeline passed", "2026-07-13T10:00:00"
            ),
            module.Note(
                12, "Demo", "robot", "pipeline passed", "2026-07-13T12:00:00"
            ),
        ]
        deduped = module.dedupe_notes(duplicate_statuses)
        self.assertEqual(1, len(deduped))
        self.assertEqual("2026-07-13T12:00:00", deduped[0].created_at)

    def test_truncated_discussions_fail_closed(self):
        module = self.module

        class FakeSession:
            def get(self, url, headers, timeout):
                if url.endswith("/pull/12"):
                    return module.HttpResponse(200, "{}")
                if url.endswith("/isource/merge_requests/12"):
                    return module.HttpResponse(200, json.dumps({"iid": 12}))
                return module.HttpResponse(
                    200,
                    json.dumps({"total": 101, "content": {"data": [{}]}}),
                )

        detail, _ = module.fetch_pr(
            FakeSession(),
            "https://gitcode.example",
            "Ascend/ModelZoo-PyTorch",
            12,
            {},
        )
        self.assertIn("truncated", detail["_discussion_error"])


class QuickcheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_script("modelzoo_pr_quickcheck.py")

    def test_setext_heading_is_not_conflict_and_compile_is_clean(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "ACL_PyTorch/built-in/cv/Demo"
            target.mkdir(parents=True)
            readme = target / "README.md"
            readme.write_text("Title\n=======\n", encoding="utf-8")
            script = target / "infer.py"
            script.write_text("print('ok')\n", encoding="utf-8")
            self.assertEqual([], self.module.check_conflict_markers(root, [target]))
            self.assertEqual([], self.module.check_py_compile(target))
            self.assertFalse(any(target.rglob("__pycache__")))
            readme.write_text(
                "<<<<<<< ours\na\n=======\nb\n>>>>>>> theirs\n", encoding="utf-8"
            )
            self.assertEqual(1, len(self.module.check_conflict_markers(root, [target])))

    def test_review_patterns_catch_false_success_and_blob_download(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "ACL_PyTorch/built-in/cv/Demo"
            target.mkdir(parents=True)
            (target / "README.md").write_text(
                "https://huggingface.co/org/repo/blob/main/model.bin\n",
                encoding="utf-8",
            )
            (target / "run.py").write_text(
                "import subprocess\nsubprocess.run(['false'])\n", encoding="utf-8"
            )
            messages = [
                finding.message
                for finding in self.module.check_review_patterns(root, target)
            ]
            self.assertTrue(any("child failure" in message for message in messages))
            self.assertTrue(any("download HTML" in message for message in messages))

    def test_catches_runtime_cleanup_and_markdown_gate_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "ACL_PyTorch/built-in/audio/Demo"
            target.mkdir(parents=True)
            (target / "README.md").write_text(
                "# 模型推理性能与精度\n[结果](#模型推理性能精度)\n```\ntext\n```",
                encoding="utf-8",
            )
            (target / "verify.py").write_text(
                "om = InferSession(0, 'demo.om')\n"
                "ort_session = ort.InferenceSession('demo.onnx')\n",
                encoding="utf-8",
            )
            findings = [
                *self.module.check_readme_markdown(root, target),
                *self.module.check_review_patterns(root, target),
            ]
            messages = [finding.message for finding in findings]
            self.assertTrue(any("final newline" in message for message in messages))
            self.assertTrue(any("lacks a language" in message for message in messages))
            self.assertTrue(
                any("anchor does not match" in message for message in messages)
            )
            self.assertTrue(any("free_resource" in message for message in messages))
            self.assertTrue(any("scope-exit cleanup" in message for message in messages))

            (target / "verify.py").write_text(
                "with ExitStack() as cleanup:\n"
                "    sessions = {}\n"
                "    sessions['om'] = InferSession(0, 'demo.om')\n"
                "    cleanup.callback(sessions['om'].free_resource)\n"
                "    sessions['ort'] = ort.InferenceSession('demo.onnx')\n"
                "    cleanup.callback(sessions.clear)\n",
                encoding="utf-8",
            )
            fixed_messages = [
                finding.message
                for finding in self.module.check_review_patterns(root, target)
            ]
            self.assertFalse(any("free_resource" in message for message in fixed_messages))
            self.assertFalse(
                any("scope-exit cleanup" in message for message in fixed_messages)
            )

    def test_modelzoo_constraints_catch_device_and_runtime_download(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "ACL_PyTorch/built-in/cv/Demo"
            target.mkdir(parents=True)
            (target / "infer.py").write_text(
                'device = "npu:0"\nfrom datasets import load_dataset\n'
                'data = datasets.load_dataset("demo")\n',
                encoding="utf-8",
            )
            messages = [
                finding.message
                for finding in self.module.check_modelzoo_constraints(root, target)
            ]
            self.assertTrue(any("hard-coded device" in message for message in messages))
            self.assertTrue(
                any("embeds an online data download" in message for message in messages)
            )


class ScaffoldTests(unittest.TestCase):
    def test_collision_does_not_partially_write(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "out"
            output.mkdir()
            requirements = output / "requirements.txt"
            requirements.write_text("existing\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "scaffold_adapter.py"),
                    "https://example.com/demo.git",
                    str(output),
                    "--category",
                    "cv",
                    "--route",
                    "onnx-om",
                    "--layout",
                    "target",
                ],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(0, result.returncode)
            self.assertFalse((output / "README.md").exists())
            self.assertEqual("existing\n", requirements.read_text(encoding="utf-8"))

    def test_workspace_layout_includes_three_main_documents(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "workspace"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "scaffold_adapter.py"),
                    "https://example.com/demo.git",
                    str(output),
                    "--category",
                    "cv",
                    "--route",
                    "torch-npu",
                    "--layout",
                    "workspace",
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            for name in ("README.md", "NPU_ADAPTATION.md", "ACCEPTANCE_PLAN.md"):
                self.assertTrue((output / name).is_file(), name)


class ProjectLogTests(unittest.TestCase):
    def run_log(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPTS / "project_log.py"), *args],
            text=True,
            capture_output=True,
        )

    def test_project_specific_log_and_status_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            common = ["--workspace", temp, "--project-id", "cv__Demo"]
            result = self.run_log(
                "init",
                *common,
                "--model-url",
                "https://example.com/demo",
                "--target-path",
                "ACL_PyTorch/built-in/cv/Demo",
                "--source-revision",
                "abc123",
                "--checkpoint",
                "not-applicable",
                "--target-repo-commit",
                "target123",
                "--target-path-exists",
                "no",
                "--route",
                "torch-npu",
                "--hardware-model",
                "Atlas 800I A2",
            )
            self.assertEqual(0, result.returncode, result.stderr)
            path = Path(temp) / ".ascend-adaptation/cv__Demo/worklog.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertIsNone(data["status"])
            self.assertFalse(data["target_ready"])
            second = self.run_log(
                "init",
                "--workspace",
                temp,
                "--project-id",
                "cv__Other",
                "--model-url",
                "https://example.com/other",
                "--target-path",
                "ACL_PyTorch/built-in/cv/Other",
            )
            self.assertEqual(0, second.returncode, second.stderr)
            self.assertTrue(
                (Path(temp) / ".ascend-adaptation/cv__Other/worklog.json").exists()
            )
            blocked = self.run_log("set-status", *common, "--status", "S0")
            self.assertNotEqual(0, blocked.returncode)
            for evidence_type in ("target_baseline", "upstream_audit"):
                evidence = self.run_log(
                    "add-evidence",
                    *common,
                    "--type",
                    evidence_type,
                    "--command",
                    evidence_type,
                    "--exit-code",
                    "0",
                    "--claim",
                    f"{evidence_type} passed",
                )
                self.assertEqual(0, evidence.returncode, evidence.stderr)
            advanced = self.run_log("set-status", *common, "--status", "S0")
            self.assertEqual(0, advanced.returncode, advanced.stderr)
            frozen = self.run_log(
                "update-context", *common, "--source-revision", "different"
            )
            self.assertNotEqual(0, frozen.returncode)

            for evidence_type, status in (
                ("static_check", "S1"),
                ("functional_inference", "S2"),
            ):
                evidence = self.run_log(
                    "add-evidence",
                    *common,
                    "--type",
                    evidence_type,
                    "--command",
                    evidence_type,
                    "--exit-code",
                    "0",
                    "--claim",
                    f"{evidence_type} passed",
                )
                self.assertEqual(0, evidence.returncode, evidence.stderr)
                status_result = self.run_log("set-status", *common, "--status", status)
                self.assertEqual(0, status_result.returncode, status_result.stderr)

            for evidence_type in (
                "npu_accuracy",
                "npu_performance",
                "comparison_report",
            ):
                evidence = self.run_log(
                    "add-evidence",
                    *common,
                    "--type",
                    evidence_type,
                    "--command",
                    evidence_type,
                    "--exit-code",
                    "0",
                    "--claim",
                    f"{evidence_type} passed",
                )
                self.assertEqual(0, evidence.returncode, evidence.stderr)
            s3 = self.run_log("set-status", *common, "--status", "S3")
            self.assertEqual(0, s3.returncode, s3.stderr)
            not_ready = self.run_log("set-target-ready", *common)
            self.assertNotEqual(0, not_ready.returncode)
            for evidence_type in ("target_audit", "clean_room_replay"):
                evidence = self.run_log(
                    "add-evidence",
                    *common,
                    "--type",
                    evidence_type,
                    "--command",
                    evidence_type,
                    "--exit-code",
                    "0",
                    "--claim",
                    f"{evidence_type} passed",
                )
                self.assertEqual(0, evidence.returncode, evidence.stderr)
            ready = self.run_log("set-target-ready", *common)
            self.assertEqual(0, ready.returncode, ready.stderr)


if __name__ == "__main__":
    unittest.main()
