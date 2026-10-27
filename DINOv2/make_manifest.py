"""Create a labeled image manifest from ImageNet class folders and a verified index map."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="validation images in class-ID subdirectories")
    p.add_argument("--class-map", required=True, help="JSON mapping directory names to ImageNet class indices")
    p.add_argument("--output", required=True, help="CSV image manifest")
    p.add_argument("--expected-count", type=int, default=50000)
    args = p.parse_args()
    if args.expected_count < 1:
        raise ValueError("expected-count must be positive")
    root = Path(args.root).resolve(strict=True)
    mapping_path = Path(args.class_map).resolve(strict=True)
    mapping_raw = mapping_path.read_bytes()
    mapping = json.loads(mapping_raw)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("class-map must be a nonempty JSON object")
    if any(type(v) is not int or not 0 <= v < 1000 for v in mapping.values()):
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
            if not image.is_file() or image.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                raise ValueError(f"unexpected entry in class directory: {image}")
            rows.append((str(image.resolve()), mapping[folder.name]))
    if len(rows) != args.expected_count:
        raise ValueError(f"found {len(rows)} images; expected {args.expected_count}")
    with Path(args.output).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["path", "label"])
        writer.writerows(rows)
    report = {"count": len(rows), "class_map_sha256": hashlib.sha256(mapping_raw).hexdigest(),
              "root": str(root), "manifest_sha256": hashlib.sha256(Path(args.output).read_bytes()).hexdigest()}
    Path(args.output + ".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
