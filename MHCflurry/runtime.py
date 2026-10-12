# Licensed under the Apache License, Version 2.0.
"""Model execution dependencies are required at import of this execution module."""

import csv
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
from importlib.metadata import version
from pathlib import Path

import torch
from mhcflurry import Class1PresentationPredictor
from mhcflurry.common import configure_pytorch, get_pytorch_device

from adapter_utils import (
    SOURCE_REVISION,
    finite_float,
    sha256,
    weights_inventory,
    write_json,
)


def synchronize(device):
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def predict(predictor, rows, task, batch_size):
    peptides = [row["peptide"] for row in rows]
    flanks = {}
    if "n_flank" in rows[0]:
        flanks = {
            "n_flanks": [row["n_flank"] for row in rows],
            "c_flanks": [row["c_flank"] for row in rows],
        }
    if task == "affinity":
        values = predictor.affinity_predictor.predict(
            peptides=peptides,
            alleles=[row["alleles"] for row in rows],
            throw=True,
            model_kwargs={"batch_size": batch_size},
        )
        output = [
            {"id": row["id"], "affinity": finite_float(value)}
            for row, value in zip(rows, values, strict=True)
        ]
    elif task == "processing":
        values = predictor.predict_processing(
            peptides=peptides, batch_size=batch_size, throw=True, verbose=0, **flanks
        )
        output = [
            {"id": row["id"], "processing_score": finite_float(value)}
            for row, value in zip(rows, values, strict=True)
        ]
    else:
        genotype_names = {}
        sample_names = []
        for row in rows:
            genotype = tuple(row["alleles"].split(";"))
            if genotype not in genotype_names:
                genotype_names[genotype] = "genotype_%d" % len(genotype_names)
            sample_names.append(genotype_names[genotype])
        frame = predictor.predict(
            peptides=peptides,
            alleles={name: list(genotype) for genotype, name in genotype_names.items()},
            sample_names=sample_names,
            throw=True,
            verbose=0,
            affinity_model_kwargs={"batch_size": batch_size},
            processing_batch_size=batch_size,
            **flanks,
        )
        by_id = {}
        for record in frame.to_dict("records"):
            index = int(record["peptide_num"])
            if index < 0 or index >= len(rows):
                raise ValueError("Unexpected presentation peptide index")
            row = rows[index]
            if (
                record["sample_name"] != sample_names[index]
                or record["peptide"] != row["peptide"]
                or row["id"] in by_id
            ):
                raise ValueError("Unexpected presentation row identity")
            best_allele = record["best_allele"]
            if not isinstance(best_allele, str) or not best_allele.strip():
                raise ValueError("Missing presentation best_allele")
            by_id[row["id"]] = {
                "id": row["id"],
                "best_allele": best_allele,
                **{
                    name: finite_float(record[name])
                    for name in (
                        "affinity",
                        "processing_score",
                        "presentation_score",
                        "presentation_percentile",
                    )
                },
            }
        if set(by_id) != {row["id"] for row in rows}:
            raise ValueError("Missing presentation rows")
        output = [by_id[row["id"]] for row in rows]
    return output


def active_networks(predictor, task, flanks):
    networks = []
    if task != "processing":
        networks.extend(predictor.affinity_predictor.neural_networks)
    if task != "affinity":
        component = (
            predictor.processing_predictor_with_flanks
            if flanks
            else predictor.processing_predictor_without_flanks
        )
        networks.extend(component.models)
    return networks


def run(args, rows):
    if version("mhcflurry") != "2.3.14":
        raise ValueError("Expected fixed mhcflurry 2.3.14 source checkout")
    package_root = Path(
        importlib.util.find_spec("mhcflurry").submodule_search_locations[0]
    )
    source_manifest = json.loads(
        (Path(__file__).parent / "source_manifest.json").read_text(encoding="utf-8")
    )
    for relative, expected in source_manifest.items():
        actual = hashlib.sha256(
            (package_root / relative).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        if actual != expected:
            raise ValueError(
                "Installed source does not match fixed patched checkout: " + relative
            )
    files, weight_identity = weights_inventory(args.model_dir, args.weight_archive)
    configure_pytorch(backend="gpu" if args.device == "cuda" else args.device)
    device = get_pytorch_device()
    if device.type != args.device:
        raise RuntimeError("Requested device does not match resolved device")
    probe = torch.ones(1, device=device) + 1
    synchronize(device)
    if probe.device.type != args.device or probe.cpu().item() != 2:
        raise RuntimeError("Device tensor probe failed")
    start = time.perf_counter()
    predictor = Class1PresentationPredictor.load(models_dir=args.model_dir)
    output = predict(predictor, rows, args.task, args.batch_size)
    synchronize(device)
    cold_seconds = time.perf_counter() - start
    for _ in range(args.warmup):
        predict(predictor, rows, args.task, args.batch_size)
        synchronize(device)
    times = []
    for _ in range(args.runs):
        synchronize(device)
        start = time.perf_counter()
        output = predict(predictor, rows, args.task, args.batch_size)
        synchronize(device)
        times.append(time.perf_counter() - start)
    network_devices = []
    for network in active_networks(predictor, args.task, "n_flank" in rows[0]):
        types = {tensor.device.type for tensor in network.network().parameters()}
        types.update(tensor.device.type for tensor in network.network().buffers())
        if types != {args.device}:
            raise RuntimeError("Neural network device mismatch: " + str(types))
        network_devices.append(sorted(types))
    if not network_devices:
        raise RuntimeError("No neural networks were verified")
    target = Path(args.output_dir)
    target.mkdir(parents=True, exist_ok=False)
    with (target / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    elapsed = statistics.median(times)
    metadata = {
        "simulation": False,
        "device": args.device,
        "network_devices": network_devices,
        "command": sys.argv,
        "source_revision": SOURCE_REVISION,
        "mhcflurry": version("mhcflurry"),
        "torch": torch.__version__,
        "python": sys.version,
        "torch_npu": version("torch-npu") if args.device == "npu" else None,
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        if args.device == "npu"
        else os.environ.get("CUDA_VISIBLE_DEVICES"),
        "task": args.task,
        "batch_size": args.batch_size,
        "rows": len(rows),
        "manifest_sha256": sha256(args.manifest),
        "weights_sha256": weight_identity,
        "predictions_sha256": sha256(target / "predictions.csv"),
        "cold_load_and_predict_seconds": cold_seconds,
        "warmup": args.warmup,
        "run_seconds": times,
        "median_seconds": elapsed,
        "peptides_per_second": len(rows) / elapsed,
        "timing_scope": "end-to-end predict including encoding, transfer, ensemble and CPU postprocessing; excludes load and CSV write",
        "optimization_level": 0,
        "compile": False,
    }
    write_json(target / "weights_inventory.json", files)
    write_json(target / "run.json", metadata)
    print(metadata)
