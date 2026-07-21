#!/usr/bin/env python3
"""MechVL-4B-RL 一键批量测试（对接 vLLM-Ascend OpenAI-compatible 服务）。

本脚本仿照 cad_prompt--openrouter/batch_analyze_mechvl.ps1 + mechvl_cpu.py 的
输出契约（每图一份 Markdown 特征表 + 总账 CSV/Markdown），但推理后端由本地 CPU
transformers 改为 ModelZoo/MechVL-4B-RL 部署的 vLLM-Ascend HTTP 服务（serve.sh）。

测试用例与脚本均自包含于本目录：
  - 默认图片目录：tests/fixtures/images（由 tests/make_fixtures.py 生成，无联网）
  - 默认提示词：  tests/fixtures/prompt.txt
  - 也可指向 runtime/MechVQA/benchmark_data 等真实数据
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

# 复用部署侧 infer.py 的 RL 后缀、HTTP 与图片编码实现，避免逻辑分叉。
from infer import (
    RL_SUFFIX,
    extract_answer,
    get_json,
    image_data_url,
    post_json,
    sha256,
)

MODEL_ID = "MechVL-4B-RL"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


@dataclass
class RunRecord:
    attempt_id: str
    started_at: str
    completed_at: str
    requested_model: str
    actual_model: str
    image: str
    image_sha256: str
    status: str
    output: str
    raw_output: str
    seconds: float
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    error: str


def safe_segment(value: str) -> str:
    result = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._-")
    return result or "default"


def extract_markdown_table(text: str) -> str:
    """与 mechvl_cpu.py 保持一致的 answer 提取与表格截取。"""
    answer = extract_answer(text)
    lines = answer.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if re.match(r"^\s*\|\s*序号\s*\|", line)),
        -1,
    )
    if start < 0:
        return answer
    table: list[str] = []
    for line in lines[start:]:
        if line.lstrip().startswith("|"):
            table.append(line.strip())
        elif line.strip():
            break
    return "\n".join(table).strip()


def list_images(images_dir: Path) -> list[Path]:
    return sorted(
        (p.resolve() for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.name.lower(),
    )


class MockEngine:
    """无 NPU 时校验端到端管线：返回固定表格。"""

    def generate(self, image: Path, prompt: str, max_tokens: int) -> tuple[str, dict[str, Any]]:
        text = (
            "<think>mock</think><answer>## 零件尺寸特征（按几何顺序）\n\n"
            "| 序号 | 特征分类 | 标准特征名（中/英） | 特征值 | 备注 |\n"
            "| ---- | -------- | ------------------- | ------ | ---- |\n"
            "| — | 总体 | 总长/最大外径 Overall | $100$ mm | mock |\n"
            f"| 1 | 轮廓类 | 圆柱外圆 Cylinder OD | $Ø{image.stem[-2:] or 30}$ mm | mock |\n"
            "</answer>"
        )
        usage = {"prompt_tokens": len(prompt), "completion_tokens": len(text), "total_tokens": len(prompt) + len(text)}
        return text, {"usage": usage, "finish_reason": "stop"}


class VllmAscendEngine:
    """对接 serve.sh 启动的 vLLM-Ascend OpenAI-compatible 服务。"""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        served = get_json(
            f"{args.base_url.rstrip('/')}/models", args.api_key, args.timeout
        )
        served_models = {str(item["id"]) for item in served.get("data", [])}
        if args.model not in served_models:
            raise ValueError(
                f"served model mismatch: expected {args.model}, got {sorted(served_models)}"
            )
        self.actual_model = args.model

    def generate(self, image: Path, prompt: str, max_tokens: int) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": self.args.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url(image)}},
                        {"type": "text", "text": prompt.strip() + RL_SUFFIX},
                    ],
                }
            ],
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "top_k": self.args.top_k,
            "max_tokens": max_tokens,
            "stream": False,
        }
        response = post_json(
            f"{self.args.base_url.rstrip('/')}/chat/completions",
            payload,
            self.args.api_key,
            self.args.timeout,
        )
        choice = response["choices"][0]
        text = choice["message"].get("content") or ""
        return text, {"usage": response.get("usage", {}), "finish_reason": choice.get("finish_reason")}


def write_overall(records: Iterable[RunRecord], csv_path: Path, markdown_path: Path) -> None:
    rows = [asdict(record) for record in records]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RunRecord.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(rows)
    ok_count = sum(row["status"] == "OK" for row in rows)
    has_mock = any(r.requested_model != r.actual_model and str(r.actual_model).endswith(":mock") for r in records)
    device_line = "CPU/Mock（未连接 NPU 服务）" if has_mock else "NPU（vLLM-Ascend OpenAI-compatible）"
    lines = [
        "# MechVL-4B-RL 批量测试总体记录",
        "",
        f"- 模型：`{MODEL_ID}`",
        f"- 成功：{ok_count}/{len(rows)}",
        f"- 设备：{device_line}",
        "",
        "| 图片 | 状态 | 用时（秒） | 输入 Token | 输出 Token | finish_reason | 输出文件 | 错误 |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        error = str(row["error"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {row['image']} | {row['status']} | {row['seconds']} | {row['prompt_tokens']} | "
            f"{row['completion_tokens']} | {row['finish_reason']} | {row['output']} | {error} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pic", type=Path)
    source.add_argument("--images-dir", type=Path)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("MODEL", MODEL_ID))
    parser.add_argument("--api-key", default=os.environ.get("VQA_TARGET_API_KEY", "EMPTY"))
    parser.add_argument("--device", choices=("npu",), default="npu")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock", action="store_true", help="不连接服务，用 mock 表格校验管线")
    args = parser.parse_args(argv)
    if args.max_tokens < 1:
        parser.error("max-tokens 必须大于 0")
    if not (0.0 <= args.temperature):  # vLLM 允许 0；上限由服务侧约束
        parser.error("temperature 不能为负")
    if not (0.0 < args.top_p <= 1.0):
        parser.error("top-p 必须在 (0, 1]")
    if args.top_k < 1:
        parser.error("top-k 必须大于 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prompt_file = args.prompt_file.resolve()
    output_dir = args.output_dir.resolve()
    if not prompt_file.is_file():
        print(f"[ERROR] 提示词文件不存在：{prompt_file}", file=sys.stderr)
        return 1
    if args.pic:
        images = [args.pic.resolve()]
    else:
        images_dir = args.images_dir.resolve()
        if not images_dir.is_dir():
            print(f"[ERROR] 图片目录不存在：{images_dir}", file=sys.stderr)
            return 1
        images = list_images(images_dir)
    if not images or any(not image.is_file() for image in images):
        print("[ERROR] 没有可分析的图片，或图片不存在。", file=sys.stderr)
        return 1
    prompt = prompt_file.read_text(encoding="utf-8").strip()
    if not prompt:
        print("[ERROR] 提示词为空。", file=sys.stderr)
        return 1

    print(f"模型：{args.model}")
    print(f"服务：{args.base_url}")
    print(f"设备：{args.device}（vLLM-Ascend OpenAI-compatible）")
    print(f"图片数：{len(images)}")
    print(f"最大输出 Token：{args.max_tokens}")
    for index, image in enumerate(images, 1):
        print(f"[{'DryRun ' if args.dry_run else ''}{index}/{len(images)}] {image}")
    if args.dry_run:
        print("[DryRun] 计划校验完成；未连接服务、未执行推理。")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    engine: VllmAscendEngine | MockEngine = MockEngine() if args.mock else VllmAscendEngine(args)
    actual_model = getattr(engine, "actual_model", args.model + ":mock")
    records: list[RunRecord] = []
    for index, image in enumerate(images, 1):
        started = datetime.now()
        stamp = started.strftime("%Y%m%d_%H%M%S_%f")
        output_path = output_dir / f"dimension_features_mechvl_rl_pic_{safe_segment(image.stem)}_{stamp}.md"
        raw_path = output_path.with_name(output_path.stem + "_raw.txt")
        print(f"[{index}/{len(images)}] 正在分析：{image.name}", flush=True)
        before = time.perf_counter()
        raw_text = ""
        prompt_tokens = 0
        completion_tokens = 0
        finish_reason = ""
        try:
            raw_text, meta = engine.generate(image, prompt, args.max_tokens)
            seconds = round(time.perf_counter() - before, 3)
            usage = meta.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            finish_reason = str(meta.get("finish_reason") or "")
            report = extract_markdown_table(raw_text)
            if not report or not re.search(r"\|\s*序号\s*\|", report):
                raise RuntimeError("模型输出未包含有效的尺寸特征表头")
            output_path.write_text(report.rstrip() + "\n", encoding="utf-8")
            if args.keep_raw:
                raw_path.write_text(raw_text, encoding="utf-8")
            record = RunRecord(
                uuid.uuid4().hex,
                started.strftime("%Y-%m-%d %H:%M:%S"),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                args.model, actual_model,
                image.name, sha256(image), "OK", str(output_path),
                str(raw_path) if args.keep_raw else "",
                seconds, prompt_tokens, completion_tokens, finish_reason, "",
            )
            print(f"OUTPUT_FILE={output_path}")
            print(f"RESULT_STATUS=OK")
        except Exception as exc:  # 批处理需要将单图失败写入总账后继续
            seconds = round(time.perf_counter() - before, 3)
            if args.keep_raw and raw_text:
                raw_path.write_text(raw_text, encoding="utf-8")
            record = RunRecord(
                uuid.uuid4().hex,
                started.strftime("%Y-%m-%d %H:%M:%S"),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                args.model, actual_model,
                image.name, sha256(image), "FAILED", "",
                str(raw_path) if args.keep_raw and raw_text else "",
                seconds, prompt_tokens, completion_tokens, finish_reason, str(exc),
            )
            print(f"[ERROR] {image.name}: {exc}", file=sys.stderr)
        records.append(record)
        write_overall(records, output_dir / "mechvl_test_overall.csv", output_dir / "mechvl_test_overall.md")
        if record.status == "FAILED" and args.stop_on_failure:
            break

    summary_path = output_dir / f"batch_summary_mechvl_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
    write_overall(records, summary_path, output_dir / "mechvl_test_overall.md")
    print(f"SUMMARY_FILE={summary_path}")
    print(f"OVERALL_MD={output_dir / 'mechvl_test_overall.md'}")
    return 1 if any(record.status == "FAILED" for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
