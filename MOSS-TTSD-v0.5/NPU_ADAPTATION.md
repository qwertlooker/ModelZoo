# MOSS-TTSD-v0.5 NPU 适配说明

## 1. 适配目标

在不新增独立代码文件的前提下，基于原项目 `OpenMOSS/MOSS-TTSD` tag `v0.5` 的已有推理链路完成 NPU 适配：

- 默认 `--device npu`；
- CPU 验证显式 `--device cpu`；
- 不使用 `auto/use_gpu` 作为默认设备选择；
- 不写死 `npu:0` / `cuda:0`，实际卡号由环境变量控制；
- 必要代码改动通过 patch 交付。

## 2. patch 策略

当前 patch：`MOSS-TTSD-v0.5/patches/0001-adapt-v0.5-inference-to-npu.patch`。

基准源码：

```bash
git -C MOSS-TTSD-v0.5/upstream checkout v0.5
git -C MOSS-TTSD-v0.5/upstream rev-parse HEAD
# 0e078c62389922d3aa873ce182daf31142860b18
```

应用：

```bash
git -C MOSS-TTSD-v0.5/upstream apply ../patches/0001-adapt-v0.5-inference-to-npu.patch
```

补丁修改原项目已有文件：

- `inference.py`
- `generation_utils.py`
- `XY_Tokenizer/inference.py`
- `XY_Tokenizer/xy_tokenizer/model.py`
- `XY_Tokenizer/xy_tokenizer/nn/quantizer.py`

## 3. 环境准备

NPU 环境中先安装与 CANN 匹配的 `torch` / `torch-npu`，再使用原项目 requirements：

```bash
cd MOSS-TTSD-v0.5/upstream
pip install torch torch-npu
pip install -r requirements.txt
pip install -r XY_Tokenizer/requirements.txt
```

原 README 中的 Ascend 版本约束可作为目标环境参考：驱动/固件 `>=25.0.RC1.1`，CANN Toolkit/Kernel/NNAL `>=8.2.RC1`，PyTorch/torch-npu `>=2.6.0`。最终以目标 CANN 对应的 torch-npu 官方匹配表为准。

## 4. 权重准备

原 v0.5 推理代码默认：

```text
MODEL_PATH = fnlp/MOSS-TTSD-v0.5
SPT_CONFIG_PATH = XY_Tokenizer/config/xy_tokenizer_config.yaml
SPT_CHECKPOINT_PATH = XY_Tokenizer/weights/xy_tokenizer.ckpt
```

patch 后可通过命令行覆盖：

```bash
--model_path /path/to/fnlp/MOSS-TTSD-v0.5 \
--spt_config_path /path/to/XY_Tokenizer/config/xy_tokenizer_config.yaml \
--spt_checkpoint_path /path/to/XY_Tokenizer/weights/xy_tokenizer.ckpt
```

正式验收前记录：模型权重来源、HF/ModelScope revision、`model.safetensors` 或等效权重 SHA256、`xy_tokenizer.ckpt` SHA256。

## 5. 推理命令

NPU：

```bash
cd MOSS-TTSD-v0.5/upstream
ASCEND_RT_VISIBLE_DEVICES=0 python inference.py \
  --jsonl examples/examples.jsonl \
  --output_dir outputs_npu \
  --device npu \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --model_path fnlp/MOSS-TTSD-v0.5 \
  --spt_config_path XY_Tokenizer/config/xy_tokenizer_config.yaml \
  --spt_checkpoint_path XY_Tokenizer/weights/xy_tokenizer.ckpt \
  --seed 42 \
  --use_normalize
```

CPU：

```bash
cd MOSS-TTSD-v0.5/upstream
python inference.py \
  --jsonl examples/examples.jsonl \
  --output_dir outputs_cpu \
  --device cpu \
  --dtype float32 \
  --attn_implementation sdpa \
  --model_path fnlp/MOSS-TTSD-v0.5 \
  --spt_config_path XY_Tokenizer/config/xy_tokenizer_config.yaml \
  --spt_checkpoint_path XY_Tokenizer/weights/xy_tokenizer.ckpt \
  --seed 42 \
  --use_normalize
```

输出仍沿用原项目逻辑：`output_*.wav` 保存到指定 `--output_dir`。

## 6. 与旧手工修改说明的关系

旧说明要求手工改多处 `cuda` 字符串。本次将这些改动收敛为可复现 patch；后续如果需要适配 Gradio、podcast 生成或其他路径，也应继续基于原项目已有文件生成新的 patch，而不是新增旁路脚本。
