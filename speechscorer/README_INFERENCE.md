# speechscorer 推理指导

## 概述

speechscorer 根据语音模型预测分布的 entropy/perplexity 为 utterance 评分。本适配的正式验收对象是默认 `whisper-clm` + `openai/whisper-base.en`，不将 HuBERT/WavLM 变体的结果混为同一指标。

```text
upstream=https://github.com/yaya-sy/speechscorer.git
branch=main
commit=bbe0be772b37f472994d5a97f809214fd67a2c8e
reference=https://gitcode.com/Ascend-SACT/speechscorer
reference_commit=f1d6e3ee3d0f113c610a969e6fde4a29af3216d1
checkpoint=https://huggingface.co/openai/whisper-base.en
checkpoint_head=911407f4214e0e1d82085af863093ec0b66f9cd6
```

## 环境与安装

参考环境为 Python 3.10、CANN 8.2.RC1、PyTorch/torch-npu 2.5.1。实际安装必须采用 CANN 配套版本。

```bash
git clone https://github.com/yaya-sy/speechscorer.git upstream
git -C upstream checkout bbe0be772b37f472994d5a97f809214fd67a2c8e
git -C upstream apply ../patches/0001-add-explicit-device-selection.patch

pip install torch==2.5.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cpu
pip install torch-npu==2.5.1 \
  -i https://mirrors.huaweicloud.com/repository/pypi/simple
pip install -r requirements.txt
pip install -e upstream --no-deps
```

不要直接安装 upstream `requirements.txt`：其中固定 PyTorch 2.0.1，与本适配
torch-npu 2.5.1 不匹配。当前目录的 `requirements.txt` 保留推理入口直接需要的
非框架依赖；`main.py` 在模块加载时导入全部 scorer，因此 `fairseq` 仍是必需依赖，
不能通过缺包回退规避。

下载并固定权重：

```bash
huggingface-cli download openai/whisper-base.en \
  --revision 911407f4214e0e1d82085af863093ec0b66f9cd6 \
  --local-dir weights/whisper-base.en
```

正式评测准备 SpeechOcean762 test 2,500 条：

```bash
git clone https://github.com/jimbozhang/speechocean762.git \
  eval_data/speechocean762
```

upstream demo 读取 `test/wav.scp` 和人工句级标签。数据集、指标和关联分析口径见
[ACCEPTANCE_PLAN.md](ACCEPTANCE_PLAN.md)。

## 推理

NPU：

```bash
cd upstream
ASCEND_RT_VISIBLE_DEVICES=0 speechscore \
  --audio /path/to/wavs \
  --model_checkpoint ../weights/whisper-base.en \
  --processor_checkpoint ../weights/whisper-base.en \
  --scorer whisper-clm \
  --device npu \
  --batch_size 8
```

CPU/CUDA 对齐只改变 `--device`：

```bash
speechscore --audio /path/to/wavs \
  --model_checkpoint ../weights/whisper-base.en \
  --scorer whisper-clm --device cpu --batch_size 8
```

输出为 `upstream/results/results.csv`，包含 `utterance_id`、`entropy` 和 `perplexity`。验收见 [ACCEPTANCE_PLAN.md](ACCEPTANCE_PLAN.md)，实现边界见 [NPU_ADAPTATION.md](NPU_ADAPTATION.md)。
