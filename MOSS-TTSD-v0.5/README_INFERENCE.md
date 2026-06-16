# MOSS-TTSD-v0.5 推理快速指南（patch 方式）

原则：不改原始 `README.md`，不新增独立推理代码文件；使用原项目 `OpenMOSS/MOSS-TTSD` tag `v0.5` 的已有 `inference.py`，通过 patch 适配 NPU。

## 1. 准备原项目代码

```bash
# 如已存在 MOSS-TTSD-v0.5/upstream，可跳过 clone
# git clone https://github.com/OpenMOSS/MOSS-TTSD.git MOSS-TTSD-v0.5/upstream

git -C MOSS-TTSD-v0.5/upstream fetch --depth 1 origin tag v0.5
git -C MOSS-TTSD-v0.5/upstream checkout v0.5
git -C MOSS-TTSD-v0.5/upstream apply ../patches/0001-adapt-v0.5-inference-to-npu.patch
```

## 2. 准备环境与权重

NPU 环境先安装与 CANN 匹配的 `torch` / `torch-npu`，再在原项目目录安装原项目依赖：

```bash
cd MOSS-TTSD-v0.5/upstream
pip install torch torch-npu
pip install -r requirements.txt
pip install -r XY_Tokenizer/requirements.txt
```

权重按原项目 v0.5 方式准备：

- MOSS-TTSD-v0.5 权重：`fnlp/MOSS-TTSD-v0.5` 或本地同等 snapshot。
- XY Tokenizer 配置：默认 `XY_Tokenizer/config/xy_tokenizer_config.yaml`。
- XY Tokenizer checkpoint：默认 `XY_Tokenizer/weights/xy_tokenizer.ckpt`，可从已验证的一键包或官方发布物复制到该路径。

正式验收前记录权重和 checkpoint 的来源与 SHA256。

## 3. NPU 推理

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

实际 NPU 卡号由 `ASCEND_RT_VISIBLE_DEVICES` 控制，不在代码中写死 `npu:0`。

## 4. CPU 验证

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

CPU 仅用于功能/质量基线，不代表 NPU 性能。
