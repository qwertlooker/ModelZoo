---
license: apache-2.0
hardware: NPU
---
# 引言

本案例给出语音对话大模型MOSS-Speech在NPU环境部署，并基于torch_npu执行推理任务的迁移实践。

**使用约束**
|依赖软件|版本|
| ----------- | ----------- |
|昇腾NPU驱动|>=25.0.RC1.1商发版本|
|昇腾NPU固件|>=25.0.RC1.1商发版本|
|CANN Toolkit|>=8.2.RC1商发版本|
|CANN Kernel|>=8.2.RC1商发版本|
|CANN NNAL|>=8.2.RC1商发版本|

 **硬件设备**
|设备型号|NPU配置|
|---|---|
|Atlas 800I A2 910B|	1卡| 

# 一、环境准备
安装依赖包：
pip install -r requirements.txt

# 二、下载官方代码和权重

MOSS-Speech推理依赖MOSS-Speech和MOSS-Speech-Codec

## 2.1 下载MOSS-Speech权重
https://modelscope.cn/models/openmoss/MOSS-Speech

## 2.2 下载MOSS-Speech-Codec相关代码
https://modelscope.cn/models/AI-ModelScope/MOSS-Speech-Codec

## 2.3 下载MOSS-Speech Space相关代码
https://huggingface.co/spaces/OpenMOSS-Team/MOSS-Speech/tree/main


# 三、运行指导

## 3.1 huggingface下载方式修改
在环境依赖中，对diffusers/utils/dynamic_modules_utils.py进行修改

由于对cached_download不支持，对第28行注释，并做以下修改：
```
#from huggingface_hub import cached_download, hf_hub_download, model_info
from huggingface_hub import hf_hub_download, model_info
```
此外，将288行的cached_download修改为hf_hub_download

## 3.2 Whisper特征提取修改
在环境依赖中，对transformers/models/whisper/feature_extraction_whisper.py进行修改

将第319行修改为:
```
#input_features = extract_fbank_features(input_features[0], device)
input_features = extract_fbank_features(input_features[0], 'cpu')
```
## 3.3 Matcha-TTS Transformer模型推理修改
MOSS-Speech的Space包含TTS必备代码，并对MOSS-Speech/Matcha-TTS/matcha/models/components/transformer.py进行修改

第267行增加对bf16的转换：
```
norm_hidden_states.to(torch.bfloat16),
```
## 3.4 反向傅里叶变换修改
对MOSS-Speech/cosyvoice/hifigan/generator.py进行修改，
```
#inverse_transform = torch.istft(torch.complex(real, img), self.istft_params["n_fft"], self.istft_params["hop_len"],
#                                self.istft_params["n_fft"], window=self.stft_window.to(magnitude.device))
 inverse_transform = torch.istft(torch.complex(real, img).to("cpu"), self.istft_params["n_fft"], self.istft_params["hop_len"], self.i    stft_params["n_fft"], window=self.stft_window.to("cpu"))
```

## 3.5推理参数改进
infer.py是推理主入口，将Moss-Speech和Moss-Speech对TTS的引入加入到path中，示例：
```
sys.path.append("./MOSS-Speech-space/MOSS-Speech/")
sys.path.append("./MOSS-Speech-space/MOSS-Speech/Matcha-TTS/")
```
此外，需要修改prompt_audio音频路径，model_path模型路径和codec_path路径。


## 3.6 执行推理
```shell
python infer.py
```

