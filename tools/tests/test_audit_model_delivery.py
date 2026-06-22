import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import audit_model_delivery


class AuditModelDeliveryTest(unittest.TestCase):
    def test_required_readme_terms_must_be_headings(self) -> None:
        text = """# Demo 推理指导

正文提到概述和快速上手，但没有对应章节。

## 推理环境准备
"""

        missing = audit_model_delivery.missing_heading_groups(
            text,
            (("概述",), ("推理环境",), ("快速上手",)),
        )

        self.assertEqual(missing, [("概述",), ("快速上手",)])

    def test_level_one_title_ignores_shell_comments(self) -> None:
        text = """# Demo 推理指导

```bash
# 这是 shell 注释，不是 Markdown 标题
python3 infer.py
```
"""

        self.assertEqual(audit_model_delivery.level_one_heading_count(text), 1)

    def test_tree_references_only_candidate_files(self) -> None:
        text = """```text
Demo
├── README.md
├── patches
│   └── adapt.patch
├── upstream-npu
│   ├── inference.py
│   └── examples
│       └── sample.jsonl
└── weights
    └── model.bin  # 下载后生成
```
"""

        self.assertEqual(
            audit_model_delivery.tree_referenced_files(text),
            {"README.md", "patches/adapt.patch"},
        )

    def test_internal_files_are_excluded_from_target_candidate(self) -> None:
        excluded = (
            "NPU_ADAPTATION.md",
            "ACCEPTANCE_PLAN.md",
            "README_old.md",
            "upstream/model.py",
            "weights/model.bin",
            "results/npu.json",
            "patches/README.md",
        )
        included = (
            "README.md",
            "infer.py",
            "patches/adapt.patch",
            "requirements.txt",
            "test_data/demo.wav",
        )

        for value in excluded:
            with self.subTest(value=value):
                self.assertTrue(
                    audit_model_delivery.is_internal_only(Path(value))
                )
        for value in included:
            with self.subTest(value=value):
                self.assertFalse(
                    audit_model_delivery.is_internal_only(Path(value))
                )

    def test_target_readiness_accepts_complete_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            model_dir = repo_root / "Demo"
            model_dir.mkdir()
            (model_dir / "modelzoo_level.txt").write_text(
                "FuncStatus:OK\nPerfStatus:OK\nPrecisionStatus:OK\n",
                encoding="utf-8",
            )
            documents = {
                "README.md": """# Demo 推理指导

commit_id=0123456789abcdef0123456789abcdef01234567
""",
                "NPU_ADAPTATION.md": """## 目标仓快照
## 拟合入路径
## 最新参考目录
最后实质变更 commit/date
## 上库文件清单
## 许可证
## PR 门禁
modelzoo_level.txt
当前状态: S3
""",
            }
            candidate_files = [
                Path("README.md"),
                Path("infer.py"),
                Path("modelzoo_level.txt"),
            ]

            with (
                mock.patch.object(
                    audit_model_delivery,
                    "target_candidate_files",
                    return_value=candidate_files,
                ),
                mock.patch.object(
                    audit_model_delivery,
                    "tracked_model_files",
                    return_value=candidate_files,
                ),
            ):
                failures, actual_candidates = (
                    audit_model_delivery.target_readiness_failures(
                        repo_root,
                        model_dir,
                        "ACL_PyTorch/built-in/audio/Demo",
                        documents,
                    )
                )

            self.assertEqual(failures, [])
            self.assertEqual(actual_candidates, candidate_files)


if __name__ == "__main__":
    unittest.main()
