"""DINOv2 ViT-L/14 with a separately supplied official linear4 classifier."""
import sys

import common

MODEL = "facebook/dinov2-large@47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"
ASSETS = {"model.safetensors": "399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23",
          "config.json": "12df51c069a2dc1305e34ba71ef58bc2407ea553b75f4722a1715c1bce3bbed0",
          "preprocessor_config.json": "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828"}


def validate_input(args, x, targets):
    import json
    import numpy as np
    if x.ndim != 4 or x.shape[1:] != (3, 224, 224):
        raise ValueError("DINO inputs require preprocessed float32 [batch,3,224,224]")
    if targets is None or targets.shape != (len(x),) or targets.dtype.kind not in "iu":
        raise ValueError("DINO requires one integer class label per input")
    if not args.head or not args.head_sha256:
        raise ValueError("DINO requires the official linear4 head and its SHA256")
    if args.task != "classification":
        raise ValueError("DINO requires the classification task")
    with np.load(args.input, allow_pickle=False) as data:
        info = json.loads(str(data["preparation"].item()))
    if info.get("preprocessor_sha256") != ASSETS["preprocessor_config.json"] or info.get("count") != len(x):
        raise ValueError("DINO input preparation metadata mismatch")


def load_model(args, model_dir, device):
    import torch
    common.require_package("transformers", "4.57.6")
    from transformers import Dinov2Model
    common.require_source(Dinov2Model, "e1376e0decc4cf3fd60042a17464a4118f1802a4e1ca2af4496c211fabb6d34b")

    class Classifier(torch.nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            output = self.backbone(pixel_values=x, output_hidden_states=True)
            layers = [self.backbone.layernorm(h) for h in output.hidden_states[-4:]]
            features = torch.cat([h[:, 0] for h in layers] + [layers[-1][:, 1:].mean(dim=1)], dim=1)
            return self.head(features)

    backbone, info = Dinov2Model.from_pretrained(
        str(model_dir), local_files_only=True, output_loading_info=True
    )
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if info.get(key):
            raise ValueError(f"checkpoint {key}: {info[key]}")
    head = torch.nn.Linear(5120, 1000)
    head.load_state_dict(torch.load(args.head, map_location="cpu", weights_only=True), strict=True)
    return Classifier(backbone, head).to(device).eval()


def predict(args, model, batch, target):
    output = model(batch)
    if tuple(output.shape) != (len(batch), 1000):
        raise ValueError("unexpected classifier output shape")
    return output


if __name__ == "__main__":
    common.main(sys.modules[__name__])
