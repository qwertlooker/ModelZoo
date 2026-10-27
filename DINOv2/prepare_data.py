"""Prepare DINOv2 manifests, bounded shards and fixed image tensors."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import model


def make_manifest(args):
    if args.expected_count < 1:
        raise ValueError("expected-count must be positive")
    root = Path(args.root).resolve(strict=True)
    mapping_path = Path(args.class_map).resolve(strict=True)
    mapping_raw = mapping_path.read_bytes()
    mapping = json.loads(mapping_raw)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("class-map must be a nonempty JSON object")
    if any((type(v) is not int or not 0 <= v < 1000 for v in mapping.values())):
        raise ValueError("class indices must be integers in 0..999")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("class indices must be unique")
    rows = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            raise ValueError(f"unexpected non-directory in validation root: {folder}")
        if folder.name not in mapping:
            raise ValueError(f"class missing from map: {folder.name}")
        for image in sorted(folder.iterdir()):
            if not image.is_file() or image.suffix.lower() not in (
                ".jpg",
                ".jpeg",
                ".png",
            ):
                raise ValueError(f"unexpected entry in class directory: {image}")
            rows.append((str(image.resolve()), mapping[folder.name]))
    if len(rows) != args.expected_count:
        raise ValueError(f"found {len(rows)} images; expected {args.expected_count}")
    with Path(args.output).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["path", "label"])
        writer.writerows(rows)
    report = {
        "count": len(rows),
        "class_map_sha256": hashlib.sha256(mapping_raw).hexdigest(),
        "root": str(root),
        "manifest_sha256": hashlib.sha256(Path(args.output).read_bytes()).hexdigest(),
    }
    Path(args.output + ".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))


def split_manifest(args):
    if args.shard_size < 1:
        raise ValueError("shard-size must be positive")
    original = Path(args.manifest).resolve(strict=True)
    with original.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["path", "label"]:
            raise ValueError("manifest must have path,label columns")
        rows = list(reader)
    for row in rows:
        path = Path(row["path"])
        row["path"] = str(
            (path if path.is_absolute() else original.parent / path).resolve()
        )
    if not rows or len({r["path"] for r in rows}) != len(rows):
        raise ValueError("manifest is empty or has duplicate images")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    shards = []
    for offset in range(0, len(rows), args.shard_size):
        chunk = rows[offset : offset + args.shard_size]
        name = f"shard-{len(shards):05d}.csv"
        path = output / name
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["path", "label"], lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(chunk)
        shards.append({"name": name, "count": len(chunk), "sha256": model.digest(path)})
    index = {
        "manifest_sha256": model.digest(args.manifest),
        "count": len(rows),
        "shards": shards,
    }
    (output / "index.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"count": len(rows), "shards": len(shards)}))


def prepare_images(args):
    import numpy as np
    from PIL import Image

    if Path(args.output).suffix != ".npz":
        raise ValueError("output must end in .npz")
    model.require_package("transformers", "4.57.6")
    model.checked_file(
        Path(args.model_dir) / "preprocessor_config.json",
        model.ASSETS["preprocessor_config.json"],
    )
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(
        args.model_dir, local_files_only=True
    )
    arrays, labels, image_hashes = ([], [], [])
    manifest = Path(args.manifest).resolve()
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["path", "label"]:
            raise ValueError("manifest must have path,label columns")
        for row in reader:
            image_path = Path(row["path"])
            if not image_path.is_absolute():
                image_path = manifest.parent / image_path
            label = int(row["label"])
            if not 0 <= label < 1000:
                raise ValueError("labels must be ImageNet indices 0..999")
            image_hashes.append(model.digest(image_path))
            with Image.open(image_path) as img:
                pixels = processor(images=img.convert("RGB"), return_tensors="np")[
                    "pixel_values"
                ]
            if pixels.shape != (1, 3, 224, 224):
                raise ValueError("processor did not produce [1,3,224,224]")
            arrays.append(pixels[0].astype(np.float32))
            labels.append(label)
    if not arrays:
        raise ValueError("empty image manifest")
    info = {
        "manifest_sha256": model.digest(manifest),
        "image_sha256": image_hashes,
        "preprocessor_sha256": model.ASSETS["preprocessor_config.json"],
        "count": len(labels),
    }
    np.savez_compressed(
        args.output,
        inputs=np.stack(arrays),
        targets=np.asarray(labels, dtype=np.int64),
        preparation=json.dumps(info, sort_keys=True),
    )
    print(
        json.dumps({"count": len(labels), "manifest_sha256": info["manifest_sha256"]})
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("manifest")
    p.add_argument(
        "--root", required=True, help="validation images in class-ID subdirectories"
    )
    p.add_argument(
        "--class-map",
        required=True,
        help="JSON mapping directory names to ImageNet class indices",
    )
    p.add_argument("--output", required=True, help="CSV image manifest")
    p.add_argument("--expected-count", type=int, default=50000)
    p.set_defaults(run=make_manifest)
    p = sub.add_parser("split")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--shard-size", type=int, default=256)
    p.set_defaults(run=split_manifest)
    p = sub.add_parser("images")
    p.add_argument("--model-dir", required=True)
    p.add_argument(
        "--manifest",
        required=True,
        help="CSV columns path,label; labels use ImageNet class indices",
    )
    p.add_argument("--output", required=True)
    p.set_defaults(run=prepare_images)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
