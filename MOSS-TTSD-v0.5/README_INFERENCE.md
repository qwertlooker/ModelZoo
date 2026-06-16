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

NPU 环境先安装与 CANN 匹配的 `torch` / `torch-npu`，再在原项目目录安装原项目依赖中的非 `flash-attn` 部分：

```bash
cd MOSS-TTSD-v0.5/upstream
pip install torch torch-npu
grep -vE '^flash-attn([<>= ].*)?$' requirements.txt > /tmp/moss-ttsd-v0.5-requirements-npu.txt
pip install -r /tmp/moss-ttsd-v0.5-requirements-npu.txt
pip install -r XY_Tokenizer/requirements.txt
```

说明：`flash-attn` 官方包面向 CUDA/ROCm GPU kernel，当前不作为 Ascend NPU 依赖安装。NPU 路径使用本适配默认的 `--attn_implementation sdpa`；如目标 torch-npu 组合不支持 `sdpa`，显式改为 `eager` 复测并记录。只有 CUDA/ROCm GPU 路径显式使用 `flash_attention_2` 时才安装 `flash-attn`。

说明：如果环境中 `torchaudio.load` 报 `TorchCodec is required for load_with_torchcodec`，请确认已应用最新 patch。本适配不会要求额外安装 `torchcodec`；prompt 音频文件读取已改为原依赖中的 `soundfile`，避免 TorchAudio 2.9+ 对 TorchCodec 的强依赖。

权重按原项目 v0.5 方式准备。官方下载 URL：

- MOSS-TTSD-v0.5 权重：HF <https://huggingface.co/fnlp/MOSS-TTSD-v0.5>，同内容镜像/组织别名为 <https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v0.5>；ModelScope <https://modelscope.cn/models/openmoss/MOSS-TTSD-v0.5>。本次记录 HF HEAD `8527b9136b6afefe2252ae597cecea2e80e7ebeb`、ModelScope HEAD `2633fdb794b9b6acd2a0c80dae6c2961f7db9d59`，HF 核心权重文件固定 URL <https://huggingface.co/fnlp/MOSS-TTSD-v0.5/resolve/8527b9136b6afefe2252ae597cecea2e80e7ebeb/model.safetensors>。
- XY Tokenizer checkpoint：HF <https://huggingface.co/fnlp/XY_Tokenizer_TTSD_V0>；ModelScope <https://modelscope.cn/models/openmoss/XY_Tokenizer_TTSD_V0>，文件 `xy_tokenizer.ckpt`；本次记录 HF HEAD `c83433728e698ed0698e88cb5096bc221fb8f8c5`、ModelScope HEAD `79082154409f5e883d9487c4d4b4be363323b039`。
- XY Tokenizer 配置：原项目自带 `XY_Tokenizer/config/xy_tokenizer_config.yaml`。

推荐下载命令（在 `MOSS-TTSD-v0.5/upstream/` 下执行）：

```bash
python -m pip install -U "huggingface_hub[cli]"
mkdir -p weights/MOSS-TTSD-v0.5 XY_Tokenizer/weights

hf download fnlp/MOSS-TTSD-v0.5 \
  --revision 8527b9136b6afefe2252ae597cecea2e80e7ebeb \
  --local-dir weights/MOSS-TTSD-v0.5

hf download fnlp/XY_Tokenizer_TTSD_V0 xy_tokenizer.ckpt \
  --revision c83433728e698ed0698e88cb5096bc221fb8f8c5 \
  --local-dir XY_Tokenizer/weights
```

ModelScope 可选下载命令（国内镜像；在同一目录下执行）：

```bash
python -m pip install -U modelscope
mkdir -p weights/MOSS-TTSD-v0.5 XY_Tokenizer/weights

modelscope download --model openmoss/MOSS-TTSD-v0.5 \
  --local_dir weights/MOSS-TTSD-v0.5

modelscope download --model openmoss/XY_Tokenizer_TTSD_V0 xy_tokenizer.ckpt \
  --local_dir XY_Tokenizer/weights
```

如果只需要直接下载 codec checkpoint，也可使用固定 URL：

```bash
mkdir -p XY_Tokenizer/weights
wget -O XY_Tokenizer/weights/xy_tokenizer.ckpt \
  https://huggingface.co/fnlp/XY_Tokenizer_TTSD_V0/resolve/c83433728e698ed0698e88cb5096bc221fb8f8c5/xy_tokenizer.ckpt
```

正式验收前记录权重和 checkpoint 的来源、revision 与 SHA256。

## 3. NPU 推理

```bash
cd MOSS-TTSD-v0.5/upstream
ASCEND_RT_VISIBLE_DEVICES=0 python inference.py \
  --jsonl examples/examples.jsonl \
  --output_dir outputs_npu \
  --device npu \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --model_path weights/MOSS-TTSD-v0.5 \
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
  --model_path weights/MOSS-TTSD-v0.5 \
  --spt_config_path XY_Tokenizer/config/xy_tokenizer_config.yaml \
  --spt_checkpoint_path XY_Tokenizer/weights/xy_tokenizer.ckpt \
  --seed 42 \
  --use_normalize
```

CPU 仅用于功能/质量基线，不代表 NPU 性能。
