# MOSS-TTSD-v0.5 验证记录

## 1. 当前环境验证结果

检查日期：2026-06-16。

| 项 | 结果 |
|---|---|
| 工作目录 | `/home/pei/ModelZoo` |
| 原项目 tag | `OpenMOSS/MOSS-TTSD` tag `v0.5` |
| tag commit | `0e078c62389922d3aa873ce182daf31142860b18` |
| patch | `MOSS-TTSD-v0.5/patches/0001-adapt-v0.5-inference-to-npu.patch` |
| 当前系统 Python | `Python 3.12.3` |
| 当前环境依赖 | 未安装 `torch`、`torch-npu`、模型权重和 `xy_tokenizer.ckpt` |
| CPU 实推 | 未执行，原因：当前环境缺少依赖和权重 |
| NPU 实推 | 未执行，原因：当前环境无 Ascend NPU/CANN 运行条件 |

已完成的本地校验：

```bash
git -C MOSS-TTSD-v0.5/upstream reset --hard v0.5
git -C MOSS-TTSD-v0.5/upstream apply --check ../patches/0001-adapt-v0.5-inference-to-npu.patch
```

## 2. 提交前必跑检查

### 2.1 patch apply 检查

```bash
git -C MOSS-TTSD-v0.5/upstream reset --hard v0.5
git -C MOSS-TTSD-v0.5/upstream apply --check ../patches/0001-adapt-v0.5-inference-to-npu.patch
```

### 2.2 语法检查

```bash
git -C MOSS-TTSD-v0.5/upstream reset --hard v0.5
git -C MOSS-TTSD-v0.5/upstream apply ../patches/0001-adapt-v0.5-inference-to-npu.patch
python -m py_compile \
  MOSS-TTSD-v0.5/upstream/inference.py \
  MOSS-TTSD-v0.5/upstream/generation_utils.py \
  MOSS-TTSD-v0.5/upstream/XY_Tokenizer/inference.py \
  MOSS-TTSD-v0.5/upstream/XY_Tokenizer/xy_tokenizer/model.py \
  MOSS-TTSD-v0.5/upstream/XY_Tokenizer/xy_tokenizer/nn/quantizer.py
git -C MOSS-TTSD-v0.5/upstream reset --hard v0.5
```

### 2.3 权重校验

```bash
sha256sum /path/to/fnlp/MOSS-TTSD-v0.5/*
sha256sum /path/to/XY_Tokenizer/weights/xy_tokenizer.ckpt
```

把精确文件名、来源 URL/revision 和 SHA256 补入验收报告。

## 3. CPU 验证命令

```bash
cd MOSS-TTSD-v0.5/upstream
python inference.py \
  --jsonl examples/examples.jsonl \
  --output_dir outputs_cpu \
  --device cpu \
  --dtype float32 \
  --attn_implementation sdpa \
  --model_path /path/to/fnlp/MOSS-TTSD-v0.5 \
  --spt_config_path XY_Tokenizer/config/xy_tokenizer_config.yaml \
  --spt_checkpoint_path /path/to/XY_Tokenizer/weights/xy_tokenizer.ckpt \
  --seed 42 \
  --use_normalize
```

通过条件：

- 退出码为 0；
- 输出目录生成 `output_*.wav`；
- WAV 可读且时长大于 0；
- 无缺权重、缺字段、设备不一致错误。

## 4. NPU 验证命令

```bash
cd MOSS-TTSD-v0.5/upstream
ASCEND_RT_VISIBLE_DEVICES=0 python inference.py \
  --jsonl examples/examples.jsonl \
  --output_dir outputs_npu \
  --device npu \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --model_path /path/to/fnlp/MOSS-TTSD-v0.5 \
  --spt_config_path XY_Tokenizer/config/xy_tokenizer_config.yaml \
  --spt_checkpoint_path /path/to/XY_Tokenizer/weights/xy_tokenizer.ckpt \
  --seed 42 \
  --use_normalize
```

通过条件：

- 无 `Expected all tensors to be on the same device`；
- 无 CUDA-only / `.cuda()` / 硬编码 CUDA 设备导致的错误；
- 无静默切到 CPU；
- 输出 WAV 可读且非零时长。

如目标 torch-npu 组合不支持 `sdpa`，可显式改为 `--attn_implementation eager` 复测，并在报告中记录；不要在代码中自动降级。

## 5. 验收报告模板

```text
模型：MOSS-TTSD-v0.5
源码：OpenMOSS/MOSS-TTSD tag v0.5 / 0e078c62389922d3aa873ce182daf31142860b18
patch：0001-adapt-v0.5-inference-to-npu.patch
模型权重来源/revision/SHA256：
XY Tokenizer checkpoint 来源/SHA256：
日期：
硬件：NPU 型号 / 数量 / HBM
驱动/固件/CANN：
Python / torch / torch-npu / transformers / torchaudio：
命令：
输入 JSONL：
样本数：
输出 WAV 数量和总时长：
耗时 / RTF / RTFx：
峰值 HBM/RSS：
ASR 回识别 CER/WER：
说话人相似度：
DNSMOS/UTMOS：
人工 MOS/CMOS 或 A/B 偏好：
结论：通过/不通过/需复测
问题与日志：
```
