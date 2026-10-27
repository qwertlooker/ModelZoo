"""Datadog Toto 2.0 22m median forecast using the official implementation."""
import sys

import common

MODEL = "Datadog/Toto-2.0-22m@685e4ae3e2be8d8998025e53dd98e7fdcb296a89"
ASSETS = {"model.safetensors": "9cd503d82df3aa71747862688f47a31c1d0a4b80f898df6e046189016eaa21dd",
          "config.json": "abeaf0fcd54aaac66757fde69ec3ddb4d3bfdcf96e0c8f767aefd03ab4c9e8d9"}


def verify_config(raw, actual):
    bad = [key for key, value in raw.items() if key not in actual or actual[key] != value]
    if bad:
        raise ValueError(f"Toto config values did not load unchanged: {bad}")


def validate_input(args, x, targets):
    import json
    import numpy as np
    if x.ndim != 3 or x.shape[-1] == 0 or x.shape[-1] % 32:
        raise ValueError("Toto inputs require [batch,variates,time], time divisible by 32")
    if targets is None or targets.ndim != 3 or targets.shape[:2] != x.shape[:2] or targets.shape[-1] == 0:
        raise ValueError("Toto targets require [batch,variates,horizon]")
    if args.head or args.task != "forecast":
        raise ValueError("Toto does not use a separate head or task switch")
    with np.load(args.input, allow_pickle=False) as data:
        info = json.loads(str(data["preparation"].item()))
    if info.get("model") != "Toto-2.0" or info.get("task") != "forecast" or info.get("scaling") != "none":
        raise ValueError("Toto requires raw ETTh1 forecasting data")


def load_model(args, model_dir, device):
    import dataclasses
    import json
    if sys.version_info < (3, 12):
        raise RuntimeError("Toto 2.0 requires Python >=3.12")
    common.require_package("toto-2", "2.0.0")
    common.require_package("dd-unit-scaling", "0.1.0")
    common.require_torch_range("2.14")
    from toto2 import Toto2Model
    from toto2.configuration import Toto2ModelConfig
    common.require_source(Toto2Model, "62614814939484b05aac1423d6ebf09977543fda75a53e3dadb5a54f8ec6fdbf")
    common.require_source(Toto2ModelConfig, "1e699fcbbc38a7f64071ee98a298f8e8a810369556775bf229613d62820b33e8")
    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    known = {field.name for field in dataclasses.fields(Toto2ModelConfig)}
    if set(raw) - known:
        raise ValueError(f"unsupported Toto config keys: {sorted(set(raw) - known)}")
    model = Toto2Model.from_pretrained(str(model_dir), local_files_only=True, strict=True)
    verify_config(raw, dataclasses.asdict(model.config))
    return model.to(device).eval()


def predict(args, model, batch, target):
    import torch
    inputs = {"target": batch, "target_mask": torch.ones_like(batch, dtype=torch.bool),
              "series_ids": torch.zeros(batch.shape[:2], dtype=torch.long, device=batch.device)}
    output = model.forecast(inputs, horizon=target.shape[-1], decode_block_size=None, has_missing_values=False)
    if tuple(output.shape) != (9, len(batch), batch.shape[1], target.shape[-1]):
        raise ValueError("unexpected Toto quantile output shape")
    return output[4], output


if __name__ == "__main__":
    common.main(sys.modules[__name__])
