#!/usr/bin/env python3
"""Manage one private Ascend adaptation work log per project."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

STATUS_ORDER = ["S0", "S1", "S2", "S3", "S4"]
STATUS_EVIDENCE = {
    "S0": {"target_baseline", "upstream_audit"},
    "S1": {"static_check"},
    "S2": {"functional_inference"},
    "S3": {"npu_accuracy", "npu_performance", "comparison_report"},
    "S4": {"extended_validation"},
}
EVIDENCE_TYPES = sorted(
    {
        "upstream_audit",
        "target_baseline",
        "upstream_smoke",
        "dependency_smoke",
        "patch_check",
        "onnx_check",
        "atc_compile",
        "npu_inference",
        "functional_inference",
        "npu_accuracy",
        "npu_performance",
        "comparison_report",
        "extended_validation",
        "static_check",
        "target_audit",
        "clean_room_replay",
        "data_dry_run",
        "service_smoke",
        "other",
    }
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    if not normalized:
        raise ValueError("project id must contain at least one safe character")
    return normalized


def log_path(workspace: Path, project_id: str) -> Path:
    return (
        workspace.resolve() / ".ascend-adaptation" / slug(project_id) / "worklog.json"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def add_git_exclude(workspace: Path) -> None:
    result = subprocess.run(
        ["git", "-C", str(workspace), "rev-parse", "--show-toplevel"],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return
    root = Path(result.stdout.strip()).resolve()
    try:
        relative = workspace.resolve().relative_to(root)
    except ValueError:
        return
    pattern = str(relative / ".ascend-adaptation") + "/"
    if pattern == ".ascend-adaptation/":
        pattern = "/.ascend-adaptation/"
    git_path = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--git-path", "info/exclude"],
        text=True,
        capture_output=True,
    )
    if git_path.returncode != 0:
        return
    exclude = Path(git_path.stdout.strip())
    if not exclude.is_absolute():
        exclude = root / exclude
    exclude.parent.mkdir(parents=True, exist_ok=True)
    existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
    if pattern not in existing.splitlines():
        with exclude.open("a", encoding="utf-8") as handle:
            if existing and not existing.endswith("\n"):
                handle.write("\n")
            handle.write(pattern + "\n")


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"project work log does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def init(args: argparse.Namespace) -> int:
    path = log_path(args.workspace, args.project_id)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite project work log: {path}")
    data = {
        "schema_version": 2,
        "project_id": slug(args.project_id),
        "model_url": args.model_url,
        "source_revision": args.source_revision,
        "checkpoint": args.checkpoint,
        "target_repo_commit": args.target_repo_commit,
        "target_path_exists": args.target_path_exists,
        "patch_required": args.patch_required,
        "target_path": args.target_path,
        "mode": args.mode,
        "route": args.route,
        "hardware_model": args.hardware_model,
        "status": None,
        "target_ready": False,
        "created_at": now(),
        "updated_at": now(),
        "decisions": [],
        "issues": [],
        "evidence": [],
    }
    write_json(path, data)
    add_git_exclude(args.workspace)
    print(path)
    return 0


def add_evidence(args: argparse.Namespace) -> int:
    path = log_path(args.workspace, args.project_id)
    data = load(path)
    artifacts = []
    for raw_path in args.artifact:
        artifact = Path(raw_path).expanduser().resolve()
        if not artifact.is_file():
            raise FileNotFoundError(f"evidence artifact does not exist: {artifact}")
        artifacts.append({"path": str(artifact), "sha256": sha256(artifact)})
    data["evidence"].append(
        {
            "type": args.type,
            "timestamp": now(),
            "command": args.command,
            "exit_code": args.exit_code,
            "claim": args.claim,
            "log_path": args.log_path,
            "environment": args.environment,
            "artifacts": artifacts,
        }
    )
    data["updated_at"] = now()
    write_json(path, data)
    return 0


def set_status(args: argparse.Namespace) -> int:
    path = log_path(args.workspace, args.project_id)
    data = load(path)
    current = data["status"]
    expected_index = 0 if current is None else STATUS_ORDER.index(current) + 1
    target_index = STATUS_ORDER.index(args.status)
    if target_index != expected_index:
        expected = (
            STATUS_ORDER[expected_index]
            if expected_index < len(STATUS_ORDER)
            else "none"
        )
        raise ValueError(
            f"status transitions must be sequential: current={current}, expected={expected}"
        )
    successful = {
        item["type"] for item in data["evidence"] if item.get("exit_code") == 0
    }
    missing = STATUS_EVIDENCE.get(args.status, set()) - successful
    if missing:
        raise ValueError(
            f"missing successful evidence for {args.status}: {sorted(missing)}"
        )
    if args.status == "S0":
        unresolved = [
            key
            for key in (
                "target_repo_commit",
                "target_path_exists",
                "source_revision",
                "checkpoint",
            )
            if data.get(key) == "unconfirmed"
        ]
        if unresolved:
            raise ValueError(f"confirm project context before S0: {unresolved}")
    if (
        args.status == "S1"
        and data.get("patch_required")
        and "patch_check" not in successful
    ):
        raise ValueError(
            "patch_required=true but successful patch_check evidence is missing"
        )
    if args.status == "S2":
        unresolved = [
            key
            for key in ("route", "hardware_model")
            if data.get(key) in {"undecided", "unconfirmed"}
        ]
        if unresolved:
            raise ValueError(f"confirm project context before S2: {unresolved}")
    data["status"] = args.status
    data["updated_at"] = now()
    write_json(path, data)
    return 0


def add_decision(args: argparse.Namespace) -> int:
    path = log_path(args.workspace, args.project_id)
    data = load(path)
    data["decisions"].append(
        {
            "timestamp": now(),
            "decision": args.decision,
            "rationale": args.rationale,
            "alternatives": args.alternatives,
            "impact": args.impact,
        }
    )
    data["updated_at"] = now()
    write_json(path, data)
    return 0


def add_issue(args: argparse.Namespace) -> int:
    path = log_path(args.workspace, args.project_id)
    data = load(path)
    data["issues"].append(
        {
            "timestamp": now(),
            "phase": args.phase,
            "symptom": args.symptom,
            "attempts": args.attempts,
            "root_cause": args.root_cause,
            "fix": args.fix,
            "reusable": args.reusable,
        }
    )
    data["updated_at"] = now()
    write_json(path, data)
    return 0


def update_context(args: argparse.Namespace) -> int:
    path = log_path(args.workspace, args.project_id)
    data = load(path)
    status = data.get("status")
    baseline_keys = {
        "source_revision",
        "checkpoint",
        "target_repo_commit",
        "target_path_exists",
    }
    patch_keys = {"patch_required"}
    route_keys = {"route", "hardware_model"}
    for key in (
        "source_revision",
        "checkpoint",
        "target_repo_commit",
        "target_path_exists",
        "patch_required",
        "route",
        "hardware_model",
    ):
        value = getattr(args, key)
        if value is not None:
            if status is not None and key in baseline_keys and value != data.get(key):
                raise ValueError(
                    f"{key} is frozen after S0; start a new project log for a new baseline"
                )
            if (
                status in {"S1", "S2", "S3", "S4"}
                and key in patch_keys
                and value != data.get(key)
            ):
                raise ValueError("patch_required is frozen after S1")
            if (
                status in {"S2", "S3", "S4"}
                and key in route_keys
                and value != data.get(key)
            ):
                raise ValueError(
                    f"{key} is frozen after S2; start a new project log for a new route/hardware"
                )
            data[key] = value
    data["target_ready"] = False
    data["updated_at"] = now()
    write_json(path, data)
    return 0


def set_target_ready(args: argparse.Namespace) -> int:
    path = log_path(args.workspace, args.project_id)
    data = load(path)
    if data.get("status") not in {"S3", "S4"}:
        raise ValueError("target readiness requires technical status S3 or S4")
    successful = {
        item["type"] for item in data["evidence"] if item.get("exit_code") == 0
    }
    missing = {"target_audit", "clean_room_replay"} - successful
    if missing:
        raise ValueError(f"missing target-readiness evidence: {sorted(missing)}")
    data["target_ready"] = True
    data["updated_at"] = now()
    write_json(path, data)
    return 0


def show(args: argparse.Namespace) -> int:
    path = log_path(args.workspace, args.project_id)
    print(json.dumps(load(path), ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    init_parser = subparsers.add_parser(
        "init", help="Create a new project-specific work log"
    )
    init_parser.add_argument("--workspace", type=Path, required=True)
    init_parser.add_argument("--project-id", required=True)
    init_parser.add_argument("--model-url", required=True)
    init_parser.add_argument("--source-revision", default="unconfirmed")
    init_parser.add_argument("--checkpoint", default="unconfirmed")
    init_parser.add_argument("--target-repo-commit", default="unconfirmed")
    init_parser.add_argument(
        "--target-path-exists",
        choices=["yes", "no", "unconfirmed"],
        default="unconfirmed",
    )
    init_parser.add_argument("--patch-required", action="store_true")
    init_parser.add_argument("--target-path", required=True)
    init_parser.add_argument(
        "--mode", choices=["adapt", "review", "resume"], default="adapt"
    )
    init_parser.add_argument(
        "--route",
        choices=[
            "undecided",
            "onnx-om",
            "torch-npu",
            "torchair",
            "vllm-ascend",
            "hybrid",
        ],
        default="undecided",
    )
    init_parser.add_argument("--hardware-model", default="unconfirmed")
    init_parser.set_defaults(func=init)

    evidence_parser = subparsers.add_parser(
        "add-evidence", help="Append command/result evidence"
    )
    evidence_parser.add_argument("--workspace", type=Path, required=True)
    evidence_parser.add_argument("--project-id", required=True)
    evidence_parser.add_argument("--type", choices=EVIDENCE_TYPES, required=True)
    evidence_parser.add_argument("--command", required=True)
    evidence_parser.add_argument("--exit-code", type=int, required=True)
    evidence_parser.add_argument("--claim", required=True)
    evidence_parser.add_argument("--log-path", default="")
    evidence_parser.add_argument("--environment", default="")
    evidence_parser.add_argument("--artifact", action="append", default=[])
    evidence_parser.set_defaults(func=add_evidence)

    status_parser = subparsers.add_parser(
        "set-status", help="Advance one validated status"
    )
    status_parser.add_argument("--workspace", type=Path, required=True)
    status_parser.add_argument("--project-id", required=True)
    status_parser.add_argument("--status", choices=STATUS_ORDER, required=True)
    status_parser.set_defaults(func=set_status)

    decision_parser = subparsers.add_parser(
        "add-decision", help="Append an adaptation decision"
    )
    decision_parser.add_argument("--workspace", type=Path, required=True)
    decision_parser.add_argument("--project-id", required=True)
    decision_parser.add_argument("--decision", required=True)
    decision_parser.add_argument("--rationale", required=True)
    decision_parser.add_argument("--alternatives", default="")
    decision_parser.add_argument("--impact", default="")
    decision_parser.set_defaults(func=add_decision)

    issue_parser = subparsers.add_parser(
        "add-issue", help="Append a troubleshooting record"
    )
    issue_parser.add_argument("--workspace", type=Path, required=True)
    issue_parser.add_argument("--project-id", required=True)
    issue_parser.add_argument("--phase", required=True)
    issue_parser.add_argument("--symptom", required=True)
    issue_parser.add_argument("--attempts", default="")
    issue_parser.add_argument("--root-cause", required=True)
    issue_parser.add_argument("--fix", required=True)
    issue_parser.add_argument("--reusable", action="store_true")
    issue_parser.set_defaults(func=add_issue)

    context_parser = subparsers.add_parser(
        "update-context", help="Update frozen project context"
    )
    context_parser.add_argument("--workspace", type=Path, required=True)
    context_parser.add_argument("--project-id", required=True)
    context_parser.add_argument("--source-revision")
    context_parser.add_argument("--checkpoint")
    context_parser.add_argument("--target-repo-commit")
    context_parser.add_argument(
        "--target-path-exists", choices=["yes", "no", "unconfirmed"]
    )
    context_parser.add_argument(
        "--patch-required", action=argparse.BooleanOptionalAction, default=None
    )
    context_parser.add_argument(
        "--route",
        choices=[
            "undecided",
            "onnx-om",
            "torch-npu",
            "torchair",
            "vllm-ascend",
            "hybrid",
        ],
    )
    context_parser.add_argument("--hardware-model")
    context_parser.set_defaults(func=update_context)

    ready_parser = subparsers.add_parser(
        "set-target-ready",
        help="Mark target readiness after S3/S4, audit, and clean-room replay",
    )
    ready_parser.add_argument("--workspace", type=Path, required=True)
    ready_parser.add_argument("--project-id", required=True)
    ready_parser.set_defaults(func=set_target_ready)

    show_parser = subparsers.add_parser("show", help="Print a project work log")
    show_parser.add_argument("--workspace", type=Path, required=True)
    show_parser.add_argument("--project-id", required=True)
    show_parser.set_defaults(func=show)

    args = parser.parse_args()
    try:
        return args.func(args)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
