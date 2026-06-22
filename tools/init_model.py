#!/usr/bin/env python3
"""Create the standard document skeleton for a new NPU adaptation."""

import argparse
from pathlib import Path


TEMPLATES = {
    "README.md": "TEMPLATE_README.md",
    "NPU_ADAPTATION.md": "TEMPLATE_NPU_ADAPTATION.md",
    "ACCEPTANCE_PLAN.md": "TEMPLATE_ACCEPTANCE_PLAN.md",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize README/NPU_ADAPTATION/ACCEPTANCE_PLAN skeletons."
    )
    parser.add_argument("model_dir", help="Model directory to create or update.")
    parser.add_argument("--name", help="Display model name. Defaults to directory name.")
    parser.add_argument("--domain", default="<DOMAIN>", help="Target built-in domain.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing generated document files.",
    )
    return parser.parse_args()


def render_template(text: str, model_dir: Path, model_name: str, domain: str) -> str:
    target_path = f"ACL_PyTorch/built-in/{domain}/{model_dir.name}"
    replacements = {
        "<MODEL_NAME>": model_name,
        "<MODEL_DIR>": model_dir.name,
        "<DOMAIN>": domain,
        "ACL_PyTorch/built-in/<DOMAIN>/<MODEL_DIR>": target_path,
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    template_dir = repo_root / "tools"
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = repo_root / model_dir
    model_dir = model_dir.resolve()
    try:
        relative_model_dir = model_dir.relative_to(repo_root)
    except ValueError as exc:
        raise SystemExit("model_dir must be inside this repository") from exc
    model_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.name or model_dir.name
    created = []
    skipped = []
    for output_name, template_name in TEMPLATES.items():
        output_path = model_dir / output_name
        if output_path.exists() and not args.force:
            skipped.append(output_path)
            continue
        template_text = (template_dir / template_name).read_text(encoding="utf-8")
        output_path.write_text(
            render_template(template_text, model_dir, model_name, args.domain),
            encoding="utf-8",
        )
        created.append(output_path)

    for path in created:
        print(f"created {path.relative_to(repo_root)}")
    for path in skipped:
        print(f"skipped existing {path.relative_to(repo_root)}")
    print(
        "next: replace placeholders, add runnable entries, then run "
        f"python3 tools/audit_model_delivery.py {relative_model_dir}"
    )


if __name__ == "__main__":
    main()
