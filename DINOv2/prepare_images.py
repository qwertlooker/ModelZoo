"""Prepare an explicit image/label CSV using the fixed DINOv2 image processor."""
import argparse
import csv
import json
from pathlib import Path

import common
import model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--manifest", required=True, help="CSV columns path,label; labels use ImageNet class indices")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    import numpy as np
    from PIL import Image
    if Path(args.output).suffix != ".npz":
        raise ValueError("output must end in .npz")
    common.require_package("transformers", "4.57.6")
    common.checked_file(Path(args.model_dir) / "preprocessor_config.json", model.ASSETS["preprocessor_config.json"])
    from transformers import AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained(args.model_dir, local_files_only=True)
    arrays, labels, image_hashes = [], [], []
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
            image_hashes.append(common.digest(image_path))
            with Image.open(image_path) as img:
                pixels = processor(images=img.convert("RGB"), return_tensors="np")["pixel_values"]
            if pixels.shape != (1, 3, 224, 224):
                raise ValueError("processor did not produce [1,3,224,224]")
            arrays.append(pixels[0].astype(np.float32))
            labels.append(label)
    if not arrays:
        raise ValueError("empty image manifest")
    info = {"manifest_sha256": common.digest(manifest), "image_sha256": image_hashes,
            "preprocessor_sha256": model.ASSETS["preprocessor_config.json"], "count": len(labels)}
    np.savez_compressed(args.output, inputs=np.stack(arrays), targets=np.asarray(labels, dtype=np.int64),
                        preparation=json.dumps(info, sort_keys=True))
    print(json.dumps({"count": len(labels), "manifest_sha256": info["manifest_sha256"]}))


if __name__ == "__main__":
    main()
