---
license: apache-2.0
language:
  - zh
hardware: NPU
---
# 引言

本案例给出微软语音模型Beats在NPU环境部署，并基于torch_npu执行推理任务的迁移实践。

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

## 2.1 下载官方代码
git clone https://github.com/microsoft/unilm.git

## 2.2 下载官方权重
https://github.com/microsoft/unilm/tree/master/beats 
官方代码仓开源了不同版本模型的权重文件，选择适配自己任务的模型下载

# 三、运行指导

## 3.1 修改推理脚本和项目代码
建议优先使用 patch 修改官方代码：

```shell
cd <your-path-to-UniLM>
git apply <path-to-this-repo>/BEATs/patches/0001-add-npu-fbank-device-support.patch
```

然后拷贝本目录下的 `infer.py` 到官方代码 `beats/` 目录下：

```shell
cp <path-to-this-repo>/BEATs/infer.py <your-path-to-UniLM>/beats/infer.py
```

## 3.2 执行推理
```shell
cd <your-path-to-UniLM>/beats
python infer.py --checkpoint /path/to/model.pt --wav /path/to/audio.wav --device npu
```