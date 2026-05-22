---
license: apache-2.0
hardware: NPU
---
# 引言

本案例给出FireRedASR系列的语音识别模型FireRedASR-AED在NPU环境部署，并基于torch_npu执行推理任务的迁移实践。

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
git clone https://github.com/FireRedTeam/FireRedASR

## 2.2 下载开源模型权重
https://modelscope.cn/models/FireRedTeam/FireRedASR-AED-L


# 三、运行指导

## 3.1 把本[infer.py](infer.py)移动到FireRedASR官方代码仓
mv infer.py \<your-path-to-FireRedASR\>

## 3.2 修改推理音频id和路径
对infer.py中要推理的音频id和路径进行修改
```
batch_uttid = ["BAC009S0764W0121"]
batch_wav_path = ["examples/wav/BAC009S0764W0121.wav"]
```

## 3.3 执行推理
```
python infer.py
```