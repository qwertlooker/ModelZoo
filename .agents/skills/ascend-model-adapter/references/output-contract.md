# ModelZoo output contract

Use this contract when finalizing an Ascend ModelZoo-PyTorch adaptation.

## Directory layout

Prefer a flat, ModelZoo-style directory under `ACL_PyTorch/built-in/<category>/<model>`:

```text
<ModelName>/
├── README.md
├── requirements.txt
├── diff.patch or <model>_NPU.patch        # if upstream code is modified
├── export_onnx.py or pth2onnx.py          # ONNX/OM route
├── optimize_onnx.py or fix_onnx.py        # only when needed
├── convert_om.sh or export_om.py          # ATC wrapper
├── infer.py or infer_npu.py               # NPU/OM/TorchAir/vLLM inference
├── infer_cpu.py                           # CPU/upstream baseline when useful
├── eval_accuracy.py or task metric script
├── eval_performance.py or benchmark.sh
└── helper files needed by the above
```

Avoid extra documentation files unless the upstream project truly requires them. Keep user-facing instructions in `README.md`.

## README sections

Include these sections or equivalent names:

1. Title: `<ModelName>-推理指导` or `<ModelName>(OM/TorchAir/vLLM)-推理指导`.
2. Overview: task, upstream link, fixed commit/revision, adaptation scope, supported chip(s).
3. Input/output data: tensor names, shapes, dtypes, layouts, dynamic axes, sample input.
4. Inference environment: firmware/driver, CANN, Python, PyTorch, torch_npu, torch/torchvision/torchaudio, extra SDKs, vLLM/TorchAir/ais_bench/msit versions.
5. Image startup: default docker image, `docker pull`, `docker run`, mounted NPU devices, environment variables, and `source /usr/local/Ascend/ascend-toolkit/set_env.sh`.
6. Quick start: clone ModelZoo, clone upstream at fixed commit, apply patch, install business dependencies, prepare weights/data.
7. Model export/conversion or service launch:
   - ONNX/OM: export ONNX, validate ONNX, run ATC with `--soc_version`, run sample OM inference.
   - TorchAir: graph compile settings, cache location, first-run compile note, NPU ID.
   - vLLM-Ascend: image tag, server launch, memory/env variables, client script.
8. Accuracy validation: dataset, metric, command, expected upstream/CPU result, NPU result, tolerance/delta. Default to the upstream/original task metric and evaluation protocol; if unavailable, use CPU/upstream baseline comparison and state why.
9. Performance validation: command/tool, warmup, loop count, batch/concurrency, precision, latency/FPS/QPS/RTF, chip. Default to the upstream or same-task ModelZoo benchmark metric; otherwise use the route-specific ModelZoo standard and separate pure inference from end-to-end.
10. FAQ/known issues: unsupported ops, long ATC compile, dependency version conflicts, offline downloads, CPU fallback limitations.
11. Submission checklist.


## Metric selection

- Accuracy: use the original upstream metric, dataset split, preprocessing, postprocessing, and threshold whenever feasible. If the original metric cannot be reproduced, compare NPU against CPU/upstream baseline on the same inputs and document the replacement. Do not compare with official numbers when metric implementations differ.
- Benchmark: use the original performance metric when it is reproducible and meaningful for Ascend. Otherwise use ModelZoo route conventions: `ais_bench` latency/FPS for OM, QPS/tokens/s/latency for vLLM services, RTF/RTFx for audio, and both pure-model and end-to-end latency for pipelines. Always state warmup, loop count, batch/concurrency, input shape, precision, chip, and whether first compile or CPU fallback is included.

## Container command template

```bash
export IMAGE=<ascend-image-tag>
docker pull ${IMAGE}
docker run -it --rm --net=host --privileged=true --shm-size=256g \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v $PWD:/workspace \
  -w /workspace \
  ${IMAGE} bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
python3 env_check.py
```

Adjust device count for Atlas A2/A3 servers and the selected NPU ID.

## ATC template

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
SOC_VERSION=${SOC_VERSION:-Ascend910B}
atc --framework=5 \
  --model=model.onnx \
  --output=model_bs1 \
  --soc_version=${SOC_VERSION} \
  --input_shape="input:1,3,224,224" \
  --input_format=NCHW \
  --precision_mode_v2=mixed_float16
```

Use `npu-smi info` to derive the actual `soc_version`. Document any dynamic shape, `--optypelist_for_implmode`, `--modify_mixlist`, fusion switch, or precision flags.


## Review-ready self-check

Before considering the directory ready for PR, make these true by default:

- PR/README text has no template placeholders, duplicated setup sections, debug code, stale comments, unclear variable names, or residual imports from removed files.
- Upstream source is pinned to a commit/revision; README includes `git checkout`/`git reset --hard` and patch application instructions.
- README includes chip/host information and a command to obtain chip name (`npu-smi info`) before setting `SOC_VERSION` or `chip_name`.
- Weight, config, and model variant are explicitly paired; example versions do not contradict the model title.
- External data files used by accuracy scripts are explained with source or generation steps.
- Accuracy is not represented by screenshots or output files alone; task metric commands and results are included.
- Performance metric and unit match the task and are consistent across README, scripts, and PR text.
- Local lint/import/help checks pass; any lint suppression comment has a reason.
- Antipoison, CodeCheck, SCA, and the PR pipeline are expected to pass; copied third-party code and license-sensitive snippets are avoided or justified.

## Validation evidence checklist

- [ ] Upstream URL and commit/revision are fixed.
- [ ] License/redistribution constraints are checked.
- [ ] Container image and host driver/CANN compatibility are stated.
- [ ] `env_check.py` output is recorded.
- [ ] CPU/upstream baseline output or metric is recorded.
- [ ] ONNX export succeeds and ONNX checker/simplifier result is recorded, if applicable.
- [ ] ATC succeeds and `.om` artifact path is recorded, if applicable.
- [ ] NPU single-sample inference succeeds.
- [ ] Accuracy metric is within tolerance or delta is justified.
- [ ] Performance command and result table are recorded.
- [ ] CPU-only limitations are marked `待 NPU 验证`, not passed.
