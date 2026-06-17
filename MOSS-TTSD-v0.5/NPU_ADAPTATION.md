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
- `modeling_asteroid.py`
- `XY_Tokenizer/inference.py`
- `XY_Tokenizer/utils/helpers.py`
- `XY_Tokenizer/xy_tokenizer/model.py`
- `XY_Tokenizer/xy_tokenizer/nn/quantizer.py`

其中 `modeling_asteroid.py` 修正 v0.5 自定义 `GenerationMixin._sample` 中的长度状态：原逻辑先记录原始 shifted 输入长度，再裁掉 `channels - 1` 个位置用于初始前向，但没有同步 `cur_len`。NPU `sdpa` 会下发到 `aclnnFlashAttentionScore`，该算子严格校验 attention mask 的 query/key 长度；不同步时可能产生 `[B, 1, L+7, L]` 形状的 mask（例如 `[2, 1, 1584, 1577]`），与实际 query 长度不一致并报错。补丁在裁剪 `input_ids` / `attention_mask` 后重置 `cur_len = input_ids.shape[1]`。

## 3. 环境准备

NPU 环境中先安装与 CANN 匹配的 `torch` / `torch-npu`，再安装原项目依赖中的非 `flash-attn` 部分：

```bash
cd MOSS-TTSD-v0.5/upstream
pip install torch torch-npu
grep -vE '^flash-attn([<>= ].*)?$' requirements.txt > /tmp/moss-ttsd-v0.5-requirements-npu.txt
pip install -r /tmp/moss-ttsd-v0.5-requirements-npu.txt
pip install -r XY_Tokenizer/requirements.txt
```

`flash-attn` 官方包面向 CUDA/ROCm GPU kernel，当前不作为 Ascend NPU 依赖安装；NPU 推理使用本适配补丁默认的 `--attn_implementation sdpa`，必要时显式切到 `eager` 复测。只有在 CUDA/ROCm GPU 路径且显式使用 `--attn_implementation flash_attention_2` 时，才按原项目要求安装 `flash-attn`。

原 README 中的 Ascend 版本约束可作为目标环境参考：驱动/固件 `>=25.0.RC1.1`，CANN Toolkit/Kernel/NNAL `>=8.2.RC1`，PyTorch/torch-npu `>=2.6.0`。最终以目标 CANN 对应的 torch-npu 官方匹配表为准。

音频读取说明：TorchAudio 2.9 起 `torchaudio.load` 依赖 TorchCodec，缺少 `torchcodec` 会报 `TorchCodec is required for load_with_torchcodec`。本补丁参考 Ascend MMAudio 的 CANN/torch-npu/torchaudio 版本约束思路，不把 `torchcodec` 作为 NPU 新增依赖；而是在原项目已有 `generation_utils.py` 与 `XY_Tokenizer/utils/helpers.py` 中把文件读取改为 `soundfile`，继续保留 `torchaudio.functional.resample` / `torchaudio.save` 用于重采样和写文件。`soundfile` 已在原项目 `requirements.txt` 中声明。

## 4. 权重准备

原 v0.5 推理代码默认：

```text
MODEL_PATH = fnlp/MOSS-TTSD-v0.5
SPT_CONFIG_PATH = XY_Tokenizer/config/xy_tokenizer_config.yaml
SPT_CHECKPOINT_PATH = XY_Tokenizer/weights/xy_tokenizer.ckpt
```

官方下载来源与当前记录 revision：

| 资产 | URL | revision / HEAD | 目标路径 |
|---|---|---|---|
| MOSS-TTSD-v0.5 模型权重 | HF <https://huggingface.co/fnlp/MOSS-TTSD-v0.5>；同内容别名 <https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v0.5>；ModelScope <https://modelscope.cn/models/openmoss/MOSS-TTSD-v0.5>；HF 核心权重文件 <https://huggingface.co/fnlp/MOSS-TTSD-v0.5/resolve/8527b9136b6afefe2252ae597cecea2e80e7ebeb/model.safetensors> | HF `8527b9136b6afefe2252ae597cecea2e80e7ebeb`；ModelScope `2633fdb794b9b6acd2a0c80dae6c2961f7db9d59` | `weights/MOSS-TTSD-v0.5/` |
| XY Tokenizer checkpoint | HF <https://huggingface.co/fnlp/XY_Tokenizer_TTSD_V0>；ModelScope <https://modelscope.cn/models/openmoss/XY_Tokenizer_TTSD_V0> | HF `c83433728e698ed0698e88cb5096bc221fb8f8c5`；ModelScope `79082154409f5e883d9487c4d4b4be363323b039` | `XY_Tokenizer/weights/xy_tokenizer.ckpt` |
| XY Tokenizer config | 原项目 tag `v0.5` 自带 | `0e078c62389922d3aa873ce182daf31142860b18` | `XY_Tokenizer/config/xy_tokenizer_config.yaml` |

下载命令（在 `MOSS-TTSD-v0.5/upstream/` 下执行）：

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

patch 后可通过命令行覆盖：

```bash
--model_path weights/MOSS-TTSD-v0.5 \
--spt_config_path XY_Tokenizer/config/xy_tokenizer_config.yaml \
--spt_checkpoint_path XY_Tokenizer/weights/xy_tokenizer.ckpt
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
  --model_path weights/MOSS-TTSD-v0.5 \
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
  --model_path weights/MOSS-TTSD-v0.5 \
  --spt_config_path XY_Tokenizer/config/xy_tokenizer_config.yaml \
  --spt_checkpoint_path XY_Tokenizer/weights/xy_tokenizer.ckpt \
  --seed 42 \
  --use_normalize
```

输出仍沿用原项目逻辑：`output_*.wav` 保存到指定 `--output_dir`。

## 6. 与旧手工修改说明的关系

旧说明要求手工改多处 `cuda` 字符串。本次将这些改动收敛为可复现 patch；后续如果需要适配 Gradio、podcast 生成或其他路径，也应继续基于原项目已有文件生成新的 patch，而不是新增旁路脚本。
