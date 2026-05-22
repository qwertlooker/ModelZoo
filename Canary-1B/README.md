---
license: apache-2.0
hardware: NPU
---
# 引言

本案例给出NeMo系列的语音识别模型Canary-1B在NPU环境部署，并基于torch_npu执行推理任务的迁移实践。

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


## 2.1 下载开源代码
git clone https://github.com/NVIDIA-NeMo/NeMo

## 2.2 下载开源模型权重
https://huggingface.co/nvidia/canary-1b


# 三、运行指导

## 3.1 把infer.py移动到NeMo官方代码仓
mv infer.py  \<your-path-to-NeMo\>

## 3.2 修改模型路径
对infer.py中模型路径进行修改，修改为自己的路径
```
os.system("ln -s models--nvidia--canary-1b     ~/.cache/huggingface/hub/models--nvidia--canary-1b")
```

## 3.3 指定任务进行修改
在infer.py中以此执行了三个任务，包括默认语音（英语）的ASR推理、指定语言的ASR推理和指定语言的语音到文本的翻译。

可以选择执行其中的任务，并删除不要的任务，同时需要对音频路径进行修改，例如：
```
# Transcribe
transcript = canary_model.transcribe(audio=["2902-9008-0000_01.wav"])
```

## 3.4 执行推理
```
python infer.py
```