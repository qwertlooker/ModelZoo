"""Granite TTM-R2 main checkpoint, fixed 512 to 96 forecasting."""
import json
import sys

import common

MODEL = "ibm-granite/granite-timeseries-ttm-r2@d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f"
ASSETS = {"model.safetensors": "a706726a7eb01bbcb42994b7dcb3c06ea9557898dbae8d480eb04fe8ccb89710",
          "config.json": "5e2367547c103e92cb8ebc63cbd5ad4d7bf83facad5a2a2ec261fa770ed659d5"}


def verify_config(raw, actual):
    bad = []
    for key, value in raw.items():
        if key == "transformers_version":
            continue
        actual_key = "dtype" if key == "torch_dtype" else key
        if actual_key not in actual or actual[actual_key] != value:
            bad.append(key)
    if bad:
        raise ValueError(f"TTM config values did not load unchanged: {bad}")


def validate_input(args, x, targets):
    import numpy as np
    if x.ndim != 3 or x.shape[1] != 512:
        raise ValueError("TTM inputs require [batch,512,channels]")
    if targets is None or targets.shape != (len(x), 96, x.shape[2]):
        raise ValueError("TTM targets require [batch,96,channels]")
    with np.load(args.input, allow_pickle=False) as data:
        info = json.loads(str(data["preparation"].item()))
    if info.get("model") != "TTM-R2" or info.get("task") != "forecast" or info.get("scaling") != "etth1_train_standard":
        raise ValueError("TTM requires training-split standardization")
    if args.head or args.task != "forecast":
        raise ValueError("TTM does not use a separate head or task switch")


def load_model(args, model_dir, device):
    if not (3, 11) <= sys.version_info[:2] < (3, 14):
        raise RuntimeError("granite-tsfm 0.3.8 requires Python >=3.11,<3.14")
    common.require_package("granite-tsfm", "0.3.8")
    common.require_package("transformers", "4.57.6")
    common.require_torch_range("2.10", "2.11")
    from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
    common.require_source(TinyTimeMixerForPrediction, "6c4cbcaa417ea09f643ee83918524e7bca6ac369b6b3500a5073c7ccc868dc14")
    model, info = TinyTimeMixerForPrediction.from_pretrained(
        str(model_dir), local_files_only=True, output_loading_info=True
    )
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if info.get(key):
            raise ValueError(f"checkpoint {key}: {info[key]}")
    verify_config(json.loads((model_dir / "config.json").read_text(encoding="utf-8")), model.config.to_dict())
    return model.to(device).eval()


def predict(args, model, batch, target):
    prediction = model(past_values=batch).prediction_outputs
    if tuple(prediction.shape) != (len(batch), 96, batch.shape[2]):
        raise ValueError("unexpected TTM output shape")
    return prediction


if __name__ == "__main__":
    common.main(sys.modules[__name__])
