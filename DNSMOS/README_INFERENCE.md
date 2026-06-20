# DNSMOS 推理指导

## 概述

本目录适配 Microsoft DNSMOS P.835 本地 ONNX 模型，输出 `SIG`、`BAK`、`OVRL` 和 `P808_MOS`。NPU 路径使用 ONNX Runtime `CANNExecutionProvider`，CPU 路径用于同模型对齐。

版本边界：

```text
upstream=https://github.com/microsoft/DNS-Challenge.git
branch=master
commit=591184a9fcb2cbdec02520fed81a32bbbf9d73ff
reference=https://gitcode.com/Ascend-SACT/DNSMOS
reference_commit=d1e4c2c14df9cb935d61dc5f448e655772b12379
```

当前适配的是 `DNSMOS/sig_bak_ovr.onnx`、`pDNSMOS/sig_bak_ovr.onnx` 和 `DNSMOS/model_v8.onnx`，不是 DNSMOS API 服务。

## 环境

参考适配环境为 Python 3.10、CANN 8.2.RC1。安装与 CANN 匹配、且包含 `CANNExecutionProvider` 的 ONNX Runtime，然后安装：

```bash
pip install -r requirements.txt
```

确认 provider：

```bash
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

输出必须包含 `CANNExecutionProvider`。脚本不会静默回退 CPU。

## 准备源码、权重和数据

```bash
git clone https://github.com/microsoft/DNS-Challenge.git upstream
mkdir -p weights
cp -r upstream/DNSMOS/DNSMOS weights/
cp -r upstream/DNSMOS/pDNSMOS weights/
```

权重来自上述固定 commit，正式验收前记录三个 ONNX 文件的 SHA256：

```bash
sha256sum weights/DNSMOS/model_v8.onnx \
  weights/DNSMOS/sig_bak_ovr.onnx \
  weights/pDNSMOS/sig_bak_ovr.onnx
```

当前固定 commit 的校验值：

```text
model_v8.onnx             9246480c58567bc6affd4200938e77eef49468c8bc7ed3776d109c07456f6e91
DNSMOS/sig_bak_ovr.onnx   269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd
pDNSMOS/sig_bak_ovr.onnx  9e3a197449ca2177f0997afec3bd6b890117ce2f17b89d6eea7fa0d47272c81c
```

输入为 WAV。VCC2018 可用于功能和公开语音集合回归，但它不是 DNSMOS 论文发布的官方隐藏测试集：

```bash
wget https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_submitted_systems_converted_speech.tar.gz
tar -xzf vcc2018_submitted_systems_converted_speech.tar.gz
```

## 推理

NPU：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python3 infer.py \
  --audio /path/to/wavs \
  --model_root weights \
  --device npu \
  --output_csv results_npu.csv
```

CPU 对齐：

```bash
python3 infer.py \
  --audio /path/to/wavs \
  --model_root weights \
  --device cpu \
  --output_csv results_cpu.csv
```

个性化 DNSMOS：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python3 infer.py \
  --audio /path/to/wavs \
  --model_root weights \
  --device npu \
  --personalized \
  --output_csv results_personalized.csv
```

脚本保持官方 9.01 秒窗口、1 秒 hop、全部有效窗口平均和官方多项式校正。验收方法见 [ACCEPTANCE_PLAN.md](ACCEPTANCE_PLAN.md)，适配分析和实际验证事实见 [NPU_ADAPTATION.md](NPU_ADAPTATION.md)。
