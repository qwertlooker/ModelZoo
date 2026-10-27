"""Train only the MOMENT forecast head on a prepared ETTh1 training split."""
import argparse
import json
import math
from pathlib import Path

import model as moment_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--output", required=True, help="trained head, .safetensors")
    p.add_argument("--device", choices=("npu", "cpu", "cuda"), default="npu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=0.0001)
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args()
    if (Path(args.output).suffix != ".safetensors" or args.batch_size <= 0 or args.epochs <= 0
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError("invalid output or training parameters")
    import numpy as np
    import torch
    moment_model.require_package("momentfm", "0.1.5")
    moment_model.require_package("transformers", "4.57.6")
    moment_model.require_torch_range("2.0.1")
    from momentfm import MOMENTPipeline
    from safetensors.torch import load_model, save_file
    moment_model.require_source(MOMENTPipeline, "70510f443376c9e15bc41326cf7ed3374b1f9e43d9cda2ac448e6004c0943820")
    x, y = moment_model.input_arrays(args.train)
    with np.load(args.train, allow_pickle=False) as data:
        prep = json.loads(str(data["preparation"].item()))
    if (prep.get("model") != "MOMENT" or prep.get("split") != "train" or prep.get("task") != "forecast"
            or prep.get("scaling") != "etth1_train_standard"):
        raise ValueError("training requires prepared, standardized ETTh1 training split")
    if x.ndim != 3 or x.shape[-1] != 512 or y is None or y.ndim != 3 or y.shape[:2] != x.shape[:2]:
        raise ValueError("invalid MOMENT training tensors")
    model_dir = Path(args.model_dir)
    for filename, expected in moment_model.ASSETS.items():
        moment_model.checked_file(model_dir / filename, expected)
    device, synchronize = moment_model.device_and_sync(args.device)
    torch.manual_seed(args.seed)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    net = MOMENTPipeline(config, model_kwargs={"task_name": "forecasting", "forecast_horizon": y.shape[-1]})
    load_model(net, str(model_dir / "model.safetensors"), strict=True)
    net.init()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    for parameter in net.head.parameters():
        parameter.requires_grad_(True)
    net.to(device)
    moment_model.check_model_device(net, device)
    optimizer = torch.optim.Adam(net.head.parameters(), lr=args.learning_rate)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    steps = len(loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, total_steps=steps, pct_start=0.3)
    net.train()
    for _ in range(args.epochs):
        for inputs, truth in loader:
            inputs, truth = inputs.to(device), truth.to(device)
            mask = torch.ones((len(inputs), inputs.shape[-1]), dtype=torch.bool, device=device)
            optimizer.zero_grad(set_to_none=True)
            forecast = net.forecast(x_enc=inputs, input_mask=mask).forecast
            if forecast.shape != truth.shape:
                raise ValueError("forecast and training labels differ in shape")
            loss = torch.nn.functional.mse_loss(forecast, truth)
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.head.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
    synchronize()
    save_file({k: v.detach().cpu().contiguous() for k, v in net.head.state_dict().items()}, args.output)
    report = {"head_sha256": moment_model.digest(args.output), "training_data_sha256": moment_model.digest(args.train),
              "upstream_model_sha256": moment_model.ASSETS["model.safetensors"], "device": args.device,
              "train_split": "train", "source_dataset_sha256": prep["dataset_sha256"], "horizon": y.shape[-1],
              "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
              "learning_rate": args.learning_rate, "steps": steps, "precision": "float32"}
    Path(args.output + ".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
