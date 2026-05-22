---
license: apache-2.0
language:
  - zh
pipeline_tag: audio-to-audio
tags:
  - 部署指南
hardware: NPU
---

# 引言
ClearerVoice-Studio 是一个开源的、基于 AI 的语音处理工具包，旨在为研究人员、开发人员及终端用户提供语音增强、语音分离、语音超分辨率、目标说话人提取等功能。该工具包包含了先进的预训练模型，以及用于训练和推理的脚本。

MossFormer2_SE_48模型是ClearerVoice-Studio中的一个模型，用于 48 kHz 的语音增强，通过它去除背景噪音来增强语音音频。

# 一、运行环境准备
### 表 1 版本配套表
|配套|版本 |环境准备指导 |
|--|--|--|
|Python|3.10.12|>=3.8即可|
|torch|2.5.1+cpu|>=2.0.1|
|torch_npu|2.5.1|>=2.0.1|

## 1	环境安装
克隆仓库，MossFormer2_SE_48模型是基于ClearerVoice-Studio仓库的，需要先安装ClearerVoice-Studio仓库。
```
git clone https://github.com/modelscope/ClearerVoice-Studio.git
```
如果github.com下载失败，可以使用githubfast.com下载。
```
git clone https://githubfast.com/modelscope/ClearerVoice-Studio.git
```

## 2	安装Conda虚拟环境
为了不与现有环境冲突，可以创建一个Conda虚拟环境运行MossFormer2_SE_48。
先下载并安装conda软件：

```
bash Miniconda3-py311_24.1.2-0-Linux-x86_64.sh
source ~/.bashrc
cd ClearerVoice-Studio
conda create -n ClearerVoice-Studio python=3.10
conda activate ClearerVoice-Studio
pip install -r requirements.txt
```
最好安装python=3.10以上版本，减少出现错误。


### 3	其他依赖安装
```
pip install cloudpickle ml-dtypes psutil tornado 
```

### 4	说明
如何不考虑环境冲突，也可以不创建conda虚拟环境，直接在Docker镜像上运行。
如果仅仅是运行MossFormer2_SE_48模型，可以不需要安装 requirements.txt中的全部依赖，可安装部分依赖。
```
pip install yamlargparse librosa pydub torchinfo rotary-embedding-torch attrs absl-py torchaudio
```
如果仅仅是运行MossFormer2_SE_48模型，可以不需要安装 ffmpeg。如果需要安装ffmpeg，则：
```
apt update && apt install ffmpeg
```

# 二、下载模型权重
## 1 安装modelscope
```
pip install modelscope
```
## 2 进入并创建目录checkpoints
```
cd ClearerVoice-Studio
cd clearvoice
mkdir checkpoints
```
注意：将checkpoints放在ClearerVoice-Studio/clearvoice目录下，后面运行时，将自动从本地加载模型。
## 3 下载模型
从魔塔社区拷贝的下载命令行，并执行：
```
modelscope download --model iic/ClearerVoice-Studio  --local_dir ./checkpoints
```
这里会下载SE_48K，SS_16K，SR_48K等全部模型。如果仅需要SE_48K，请参考modelscope命令修改参数。

# 三、运行指导
## 1	自带demo.py启动测试
原demo.py脚本如下：
```
if False:
    myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])
    ...
if True:
    myClearVoice = ClearVoice(task='speech_separation', model_names=['MossFormer2_SS_16K'])
```
修改demo.py，打开SE_48K模型测试打开，关闭SS_16K模型测试开关：
```
if True:
    myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])
    ...
if False:
    myClearVoice = ClearVoice(task='speech_separation', model_names=['MossFormer2_SS_16K'])
```
再执行测试：
```
python demo.py
```

## 2	运行自己的脚本
在ClearerVoice-Studio/clearvoice目录下，编写自己的运行脚本
```
from clearvoice import ClearVoice
import os
import torch
import torch_npu

if torch.npu.is_available():
    device = torch.device("npu:0")
else:
    device = torch.device("cpu")
torch.device(device)

# 初始化语音增强模型
cv_se = ClearVoice(
    task='speech_enhancement',
    model_names=['MossFormer2_SE_48K']
)

# 处理单个音频文件
input_path = 'samples/input.wav'
print("-----input_path:-----",input_path)
output_wav = cv_se(
    input_path=input_path,
    online_write=False
)

# 保存增强后的音频
output_dir = 'samples/enhanced'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'input_enhanced.wav')
print("-----output_path:-----",output_path)
cv_se.write(output_wav, output_path=output_path)
```
再通过pythony命令运行。