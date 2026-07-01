# ModelZoo output contract

Use this contract when finalizing an Ascend ModelZoo-PyTorch adaptation.

## Directory layout

Prefer a flat, ModelZoo-style directory under `ACL_PyTorch/built-in/<category>/<model>`. Provide only necessary files; prefer patching upstream code over adding duplicate scripts. Some ModelZoo projects are close to README + patch + requirements; others keep only a few helper/fix scripts.

```text
<ModelName>/
├── README.md                         # required
├── requirements.txt                  # required; business deps only
├── diff.patch or <model>_NPU.patch   # required if upstream code is modified
└── optional, only when needed
    ├── export_onnx.py / pth2onnx.py  # ONNX/OM route
    ├── convert_om.sh / atc.sh        # ONNX/OM route
    ├── infer.py / ascend_infer.py    # only if upstream lacks an inference entry
    ├── validate_acc.py / eval_accuracy.py
    ├── validate_perf.py / benchmark.sh
    └── helper/fix files needed by the above
```

Key principles:

- If upstream already has inference/evaluation entries (`inference.py`, `infer.py`, `test.py`, `demo.py`, shell commands), patch them to support NPU rather than adding duplicate scripts.
- Use a single `--device npu/cpu` parameter or environment variable where practical; do not default to separate `infer_cpu.py` and `infer_npu.py`.
- Do not include agent-internal files as submission artifacts: `env_check.py`, `docker_run.sh`, `collect_report.py`, `adaptation_config.yaml`. Put environment checks, Docker commands, and evidence collection commands in README.
- For user-provided checkpoint/weights, document the provided artifact path, expected directory tree, and any config/tokenizer/label-map pairing; do not silently switch to another checkpoint.

## README sections

Include these sections or equivalent names:

1. Title: `<ModelName>-推理指导` or `<ModelName>(路线)-推理指导`.
2. 概述: task, upstream link, fixed commit/revision, user-provided or official checkpoint info, license, adaptation scope, supported chip(s).
3. 输入输出数据: tensor names, shapes, dtypes, layouts; mandatory for OM route.
4. 推理环境准备: firmware/driver, CANN, Python, PyTorch, torch_npu, torchvision/torchaudio, extra SDKs, vLLM/TorchAir/ais_bench/msit versions. State not to reinstall image-provided `torch/torch_npu` unless justified.
5. 镜像启动: docker pull/run, NPU device mounts, env vars, and `source /usr/local/Ascend/ascend-toolkit/set_env.sh`.
6. 快速上手: clone ModelZoo, clone upstream at fixed commit, apply patch, install business deps, prepare weights/data.
7. 准备权重和数据: weight list, sources or user-provided path, directory tree, offline cache configuration.
8. 模型导出/转换或服务启动:
   - ONNX/OM: export ONNX, validate ONNX, ATC with `--soc_version`, sample OM inference.
   - TorchAir: graph compile settings, cache location, first-run compile note, NPU ID.
   - vLLM-Ascend: image tag, server launch, memory/env variables, client command.
9. 推理: exact command on NPU; patch upstream command when possible.
10. 精度验证: dataset, original/upstream metric, command, CPU/official result, NPU result, tolerance/delta.
11. 性能验证: command/tool, warmup, loop, batch/concurrency, precision, latency/FPS/QPS/RTF, chip.
12. FAQ/已知问题: unsupported ops, long ATC compile, dependency conflicts, offline downloads, CPU fallback reasons, patch troubleshooting.
13. 公网地址说明 when external URLs are referenced.

Optional for complex pipelines:

- Pipeline 组件部署: list each component, upstream backend, chosen backend, NPU feasibility, and CPU fallback reason. This is recommended for diarization/OCR/VLM/TTS pipelines but should not become a generic extra deliverable for simple models.
- 交付件清单: list submitted files with descriptions when helpful.

## Metric selection

- Accuracy: use the original upstream metric, dataset split, preprocessing, postprocessing, and threshold whenever feasible. If original metric cannot be reproduced, compare NPU against CPU/upstream baseline on identical inputs and document the replacement.
- Benchmark: use original or same-task ModelZoo performance metric when meaningful. Otherwise use route conventions: `ais_bench` latency/FPS for OM, QPS/tokens/s/latency for vLLM services, RTF/RTFx for audio, and both pure-model and end-to-end latency for pipelines.
- Always state warmup, loop count, batch/concurrency, input shape, precision, chip, and whether first compile or CPU fallback is included.

## Container command template

```bash
export IMAGE=<ascend-image-tag>
docker pull ${IMAGE}
docker run -it --rm --net=host --privileged=true --shm-size=256g \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v $PWD:/workspace -w /workspace \
  ${IMAGE} bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
python3 -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available())"
```

Adjust device count for Atlas A2/A3 servers and selected NPU ID. Add `/usr/local/bin/npu-smi` mount if the host uses that path.

## Review-ready self-check

Before considering the directory ready for PR, make these true by default:

- PR/README text has no template placeholders, duplicated setup sections, debug code, stale comments, unclear variable names, or residual imports.
- Upstream source is pinned to a commit/revision; README includes checkout and patch application instructions.
- User-provided checkpoint/weights are honored and paired with the correct config/tokenizer/label map; any substitution is explicit.
- README includes chip/host information and a command to obtain chip name before setting `SOC_VERSION` or `chip_name`.
- Accuracy is not represented by screenshots or output files alone; task metric commands and results are included.
- Performance metric and unit match the task and are consistent across README, scripts, and PR text.
- Pipeline CPU fallback has a concrete technical reason; it is not merely “upstream default backend”.
- Local lint/import/help checks pass; Antipoison, CodeCheck, SCA, and PR pipeline are expected to pass.

## Validation evidence checklist

- [ ] Upstream URL and commit/revision are fixed.
- [ ] License/redistribution constraints are checked.
- [ ] Container image and host driver/CANN compatibility are stated.
- [ ] CPU/upstream baseline output or metric is recorded.
- [ ] ONNX export succeeds and ONNX checker/simplifier result is recorded, if applicable.
- [ ] ATC succeeds and `.om` artifact path is recorded, if applicable.
- [ ] NPU single-sample inference succeeds.
- [ ] Accuracy metric is within tolerance or delta is justified.
- [ ] Performance command and result table are recorded.
- [ ] CPU-only limitations are marked `待 NPU 验证`, not passed.
