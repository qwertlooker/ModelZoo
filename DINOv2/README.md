# DINOv2 ViT-L/14 无 register + ImageNet linear4 分类头

## 概述

本工作区是离线适配准备，默认 NPU，不自动回退 CPU。**尚未运行真实模型或 NPU，未达到 S3。** 固定模型与源代码：<https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8>；权重 facebook/dinov2-large revision `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`，许可证 Apache-2.0。

参考实现 `commit_id=7764ea0f912e53c92e82eb78a2a1631e92725fc8`（固定源码或资料审计快照；实际运行依赖版本仍以上述约束为准）。

## 输入输出数据

输入图像/标签清单 CSV（列名 `path,label`；相对路径相对于清单目录，label 必须是 ImageNet 类别索引）；预处理为 `[N,3,224,224]`，输出 logits `[N,1000]`。仅当清单覆盖完整 ImageNet 验证集且标签映射正确时，top1/top5 才是全量结果。自行提供 ImageNet 授权图片及经过核实的类别索引；工具按分片加载并检查汇总来源，不得将子集分数称为全量。

## 推理环境准备

代码按 Transformers 4.57.6 的 Dinov2Model 结构编写；必须在选定 CANN 配套的 torch/torch_npu 环境中确认 Transformers、Pillow 和 numpy。当前未安装或验证。 使用匹配设备的 CANN、PyTorch 和 torch_npu；未确定硬件和配套版本，不安装 CPU-only PyTorch 再追加 torch_npu。模型代码、numpy、safetensors 等依赖须按固定上游版本及目标栈配置；缺失时直接报错。非必要不创建 venv。

确认配套栈后，可安装业务依赖；安装前核对解析过程不能替换配套 torch：

```bash
set -e
python3 -c 'import torch, torch_npu'
python3 -m pip install 'transformers==4.57.6' Pillow safetensors
python3 -m pip check
```

入口校验 Transformers 中 DINOv2 模型源码 SHA `e1376e0decc4cf3fd60042a17464a4118f1802a4e1ca2af4496c211fabb6d34b`。尚未在目标栈运行安装或推理。

## 文件目录

```text
README.md
LICENSE
model.py
prepare_data.py
aggregate_metrics.py
```

## 快速上手

以下命令从本目录执行。准备模型和数据的固定文件，校验 SHA；若网络在 NPU 机不可用，在联网机执行后将文件传入本地目录。

```bash
(
set -e
export MODEL_WORK_DIR="$PWD/work"
mkdir -p "$MODEL_WORK_DIR/model" "$MODEL_WORK_DIR/data"
curl -fL 'https://huggingface.co/facebook/dinov2-large/resolve/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c/model.safetensors' -o "$MODEL_WORK_DIR/model/model.safetensors"
curl -fL 'https://huggingface.co/facebook/dinov2-large/resolve/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c/config.json' -o "$MODEL_WORK_DIR/model/config.json"
printf '%s  %s\n' '399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23' "$MODEL_WORK_DIR/model/model.safetensors" | sha256sum -c -
printf '%s  %s\n' '12df51c069a2dc1305e34ba71ef58bc2407ea553b75f4722a1715c1bce3bbed0' "$MODEL_WORK_DIR/model/config.json" | sha256sum -c -
curl -fL 'https://huggingface.co/facebook/dinov2-large/resolve/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c/preprocessor_config.json' -o "$MODEL_WORK_DIR/model/preprocessor_config.json"
curl -fL 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_linear4_head.pth' -o "$MODEL_WORK_DIR/linear4.pth"
printf '%s  %s\n' '14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828' "$MODEL_WORK_DIR/model/preprocessor_config.json" | sha256sum -c -
export IMAGE_MANIFEST="$MODEL_WORK_DIR/images.csv"
export IMAGENET_VAL_DIR="$MODEL_WORK_DIR/data/imagenet_val_by_class"
export IMAGENET_CLASS_MAP="$MODEL_WORK_DIR/data/imagenet_class_indices.json"
test -d "$IMAGENET_VAL_DIR" && test -f "$IMAGENET_CLASS_MAP"
python3 prepare_data.py manifest --root "$IMAGENET_VAL_DIR" --class-map "$IMAGENET_CLASS_MAP" --output "$IMAGE_MANIFEST" --expected-count 50000
python3 prepare_data.py split --manifest "$IMAGE_MANIFEST" --output-dir "$MODEL_WORK_DIR/shards" --shard-size 256
mkdir -p "$MODEL_WORK_DIR/inputs" "$MODEL_WORK_DIR/results" "$MODEL_WORK_DIR/scores"
export HEAD_SHA256="$(sha256sum "$MODEL_WORK_DIR/linear4.pth" | cut -d ' ' -f 1)"
for SHARD in "$MODEL_WORK_DIR"/shards/shard-*.csv; do
  STEM="$(basename "$SHARD" .csv)"
  python3 prepare_data.py images --model-dir "$MODEL_WORK_DIR/model" --manifest "$SHARD" --output "$MODEL_WORK_DIR/inputs/$STEM.npz"
  python3 model.py infer --model-dir "$MODEL_WORK_DIR/model" --input "$MODEL_WORK_DIR/inputs/$STEM.npz" --output "$MODEL_WORK_DIR/results/$STEM.npz" --device npu --task classification --head "$MODEL_WORK_DIR/linear4.pth" --head-sha256 "$HEAD_SHA256"
  python3 model.py evaluate --input "$MODEL_WORK_DIR/inputs/$STEM.npz" --results "$MODEL_WORK_DIR/results/$STEM.npz" --output "$MODEL_WORK_DIR/scores/$STEM.json" --metric classification
done
python3 aggregate_metrics.py --manifest "$IMAGE_MANIFEST" --shard-dir "$MODEL_WORK_DIR/shards" --work-dir "$MODEL_WORK_DIR" --expected-count 50000 --output "$MODEL_WORK_DIR/npu-metrics.json"
)
```

其中 NPU 推理或训练须在硬件及依赖配套后执行。对照运行可显式使用 CPU；比较前按任务确定可接受的阈值。下例的 0.01 仅说明参数，不是验收门槛。

```bash
export MODEL_WORK_DIR="$PWD/work"
export HEAD_SHA256="$(sha256sum "$MODEL_WORK_DIR/linear4.pth" | cut -d ' ' -f 1)"
python3 model.py infer --model-dir "$MODEL_WORK_DIR/model" --input "$MODEL_WORK_DIR/inputs/shard-00000.npz" --output "$MODEL_WORK_DIR/cpu.npz" --device cpu --task classification --head "$MODEL_WORK_DIR/linear4.pth" --head-sha256 "$HEAD_SHA256"
export MAX_ABSOLUTE_ERROR=0.01
python3 model.py compare --reference "$MODEL_WORK_DIR/cpu.npz" --candidate "$MODEL_WORK_DIR/results/shard-00000.npz" --max-absolute "$MAX_ABSOLUTE_ERROR" --output "$MODEL_WORK_DIR/compare.json"
```

## 性能与精度

官方 ViT-L/14 无 register 的 ImageNet kNN 83.5%、linear 86.3%；复用匹配 head、预处理、评估范围及标签映射后才能比较。当前 HF backbone + Meta head 组合还未验证等价。分类头尚未取得官方固定 SHA；运行前记录实际 SHA 只保证后续文件一致，不证明来源。HF backbone 的中间层归一化与 Meta 原始路径仍需逐层/数据集比较；分片工具校验原始清单、各分片、数据和结果 SHA 后按样本数汇总。固定输入 SHA、模型 SHA、任务及 head 后才允许比较。同步计时覆盖模型调用，排除输入准备和结果传回；单次 batch 中位数不是三次独立运行中位数。完整 NPU 精度、三次独立性能、端到端时间和显存待上机测量。

## NPU 独立性能报告

每次 `infer` 成功后额外生成 推理输出文件旁的 `.performance.json`，保存逐批同步耗时、总模型调用时间、吞吐、结果 SHA、命令和运行 ID。元信息含 Python、PyTorch、torch_npu、设备名称、进程内逻辑卡号及 `ASCEND_RT_VISIBLE_DEVICES` 原值；物理卡映射仍须结合 `npu-smi info` 核对。`model.py performance` 校验至少三次不同运行 ID、相同模型/数据/设备/参数，再汇总独立运行中位数；CPU 报告不能用于 NPU 性能结论。

以下性能命令只测固定首个分片，不能作为完整 ImageNet 精度验收；完整精度仍按前述所有分片执行。

```bash
set -e
for RUN in 1 2 3; do
  python3 model.py infer --model-dir "$MODEL_WORK_DIR/model" --input "$MODEL_WORK_DIR/inputs/shard-00000.npz" --output "$MODEL_WORK_DIR/npu-run-$RUN.npz" --device npu --task classification --head "$MODEL_WORK_DIR/linear4.pth" --head-sha256 "$HEAD_SHA256"
done
python3 model.py performance --runs "$MODEL_WORK_DIR"/npu-run-*.npz.performance.json --output "$MODEL_WORK_DIR/performance-median.json"
```

统计包括末尾不足 batch 的批次；吞吐单位为输入样本/秒（TimesFM 为通道序列/秒）。这是纯模型调用口径，端到端时间、首次编译和峰值 HBM 仍需在目标 NPU 单独实测。报告校验不能认证人工编辑的文件或证明设备证据真实，发布前须核对原始日志。

本目录 `LICENSE` 保留固定上游参考实现的许可原文；正式自测、完整依赖安装与 NPU 验收仍待完成。

## 公网地址

| 内容 | 具体地址 |
|---|---|
| 上游源码 | https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8 |
| 固定权重 | https://huggingface.co/facebook/dinov2-large/tree/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c |
| 公开参考 | https://raw.githubusercontent.com/facebookresearch/dinov2/7764ea0f912e53c92e82eb78a2a1631e92725fc8/README.md |
| 官方 linear4 头 | https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_linear4_head.pth |
| 下载或依赖地址 | https://huggingface.co/facebook/dinov2-large/resolve/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c/config.json |
| 下载或依赖地址 | https://huggingface.co/facebook/dinov2-large/resolve/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c/model.safetensors |
| 下载或依赖地址 | https://huggingface.co/facebook/dinov2-large/resolve/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c/preprocessor_config.json |
| 参考许可证 | https://raw.githubusercontent.com/facebookresearch/dinov2/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE |
