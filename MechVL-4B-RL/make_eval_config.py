#!/usr/bin/env python3
"""Generate the fixed three-judge MechVQA evaluator configuration."""

import argparse
import json
from pathlib import Path
from typing import Any

from infer import RL_SUFFIX


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a MechVQA config matching the paper's three-judge protocol."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument(
        "--output", required=True, help="Generated evaluator config JSON."
    )
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--target-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--target-model", default="MechVL-4B-RL")
    parser.add_argument("--gpt-oss-base-url", required=True)
    parser.add_argument("--deepseek-base-url", required=True)
    parser.add_argument("--kimi-base-url", required=True)
    parser.add_argument("--gpt-oss-model", default="GPT-OSS-120B")
    parser.add_argument("--deepseek-model", default="DeepSeek-V3.2")
    parser.add_argument("--kimi-model", default="Kimi-k2")
    parser.add_argument("--max-workers", type=positive_int, default=8)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def judge(name: str, model: str, base_url: str, api_key_env: str) -> dict[str, Any]:
    return {
        "name": name,
        "model": model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "temperature": 0.1,
        "max_tokens": 1024,
    }


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest).resolve()
    image_root = Path(args.image_root).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"image root does not exist: {image_root}")

    results_dir = Path(args.results_dir).resolve()
    config = {
        "input_file": str(manifest),
        "image_root": str(image_root),
        "phase1_output": str(results_dir / "responses_MechVL-4B-RL.jsonl"),
        "evaluated_output": str(results_dir / "evaluated_MechVL-4B-RL.jsonl"),
        "stats_output": str(results_dir / "stats_MechVL-4B-RL.json"),
        "max_workers": args.max_workers,
        "append_answer_suffix": True,
        "answer_suffix": RL_SUFFIX,
        "target_models": [
            {
                "name": args.target_model,
                "model": args.target_model,
                "base_url": args.target_base_url,
                "api_key_env": "VQA_TARGET_API_KEY",
                "temperature": 0.6,
                "max_tokens": 4096,
                "extra_body": {"top_p": 0.95, "top_k": 20},
            }
        ],
        "judge_models": [
            judge(
                "GPT-OSS-120B",
                args.gpt_oss_model,
                args.gpt_oss_base_url,
                "GPT_OSS_API_KEY",
            ),
            judge(
                "DeepSeek-V3.2",
                args.deepseek_model,
                args.deepseek_base_url,
                "DEEPSEEK_API_KEY",
            ),
            judge("Kimi-k2", args.kimi_model, args.kimi_base_url, "KIMI_API_KEY"),
        ],
    }
    output_path = Path(args.output)
    write_json(output_path, config)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
