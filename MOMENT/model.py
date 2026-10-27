"""MOMENT-1-large reconstruction or forecasting with an explicit trained head."""
import json
import sys

import common

MODEL = "AutonLab/MOMENT-1-large@ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc"
ASSETS = {"model.safetensors": "a56928052ac6f5d09b97c3834bea6ce3aef9f02b513b5fac98954e7377801572",
          "config.json": "b0111650e8718e31d7e89575666aa3e140747d35279ec1a7c9f6ed7616c29433"}


def validate_input(args, x, targets):
    import numpy as np
    from pathlib import Path
    if args.task not in ("forecast", "reconstruction"):
        raise ValueError("MOMENT task must be forecast or reconstruction")
    if x.ndim != 3 or x.shape[-1] != 512:
        raise ValueError("MOMENT inputs require [batch,channels,512]")
    if targets is None or targets.ndim != 3 or targets.shape[:2] != x.shape[:2]:
        raise ValueError("targets must have shape [batch,channels,horizon]")
    if args.task == "forecast" and (not args.head or not args.head_sha256):
        raise ValueError("forecast requires a trained head and its SHA256")
    if args.task == "forecast":
        head_meta = json.loads(Path(args.head + ".json").read_text(encoding="utf-8"))
        if (head_meta.get("head_sha256") != args.head_sha256 or head_meta.get("steps", 0) < 1
                or head_meta.get("train_split") != "train" or head_meta.get("horizon") != targets.shape[-1]
                or head_meta.get("upstream_model_sha256") != ASSETS["model.safetensors"]):
            raise ValueError("trained head metadata does not match model or forecast task")
    if args.task == "reconstruction" and (args.head or targets.shape[-1] != 512):
        raise ValueError("reconstruction requires 512 targets and no forecast head")
    with np.load(args.input, allow_pickle=False) as data:
        info = json.loads(str(data["preparation"].item()))
    if info.get("model") != "MOMENT" or info.get("task") != args.task or info.get("scaling") != "etth1_train_standard":
        raise ValueError("MOMENT requires its standardized ETTh1 data and matching task")


def load_model(args, model_dir, device):
    common.require_package("momentfm", "0.1.5")
    common.require_package("transformers", "4.57.6")
    common.require_torch_range("2.0.1")
    from momentfm import MOMENTPipeline
    common.require_source(MOMENTPipeline, "70510f443376c9e15bc41326cf7ed3374b1f9e43d9cda2ac448e6004c0943820")
    from safetensors.torch import load_file, load_model as load_safetensors
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    options = {"task_name": "reconstruction"}
    if args.task == "forecast":
        import numpy as np
        with np.load(args.input, allow_pickle=False) as data:
            horizon = int(data["targets"].shape[-1])
        options = {"task_name": "forecasting", "forecast_horizon": horizon}
    model = MOMENTPipeline(config, model_kwargs=options)
    load_safetensors(model, str(model_dir / "model.safetensors"), strict=True)
    model.init()
    if args.task == "forecast":
        model.head.load_state_dict(load_file(args.head, device="cpu"), strict=True)
    return model.to(device).eval()


def predict(args, model, batch, target):
    import torch
    mask = torch.ones((len(batch), batch.shape[-1]), dtype=torch.bool, device=batch.device)
    if args.task == "forecast":
        result = model.forecast(x_enc=batch, input_mask=mask).forecast
    else:
        result = model.reconstruct(x_enc=batch, input_mask=mask).reconstruction
    if tuple(result.shape) != tuple(target.shape):
        raise ValueError("unexpected MOMENT output shape")
    return result


if __name__ == "__main__":
    common.main(sys.modules[__name__])
