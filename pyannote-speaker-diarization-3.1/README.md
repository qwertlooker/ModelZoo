---
license: mit
language:
  - zh
pipeline_tag: voice-activity-detection
tags:
  - 部署指南
hardware: NPU
---

# 引言
该流水线由 Séverin Baroudi 使用 pyannote.audio 3.0.0 结合 AISHELL、AliMeeting、AMI、AVA-AVD、DIHARD、Ego4D、MSDWild、REPERE 和 VoxConverse 的训练集进行训练。
它处理采样率为 16kHz 的单声道音频，并输出作为 Annotation 实例的说话人对齐结果：

  立体声或多声道音频文件通过平均通道自动混音为单声道。
  
  采样率不同的音频文件在加载时会自动重采样到 16kHz。
  
此模型会关联使用到segmentation-3.0、wespeaker-voxceleb-resnet34-LM模型；

speaker-diarization-3.1：运行在纯 PyTorch 上，它需要 pyannote.audio 3.1 或更高版本。
segmentation-3.0：该模型接收10秒、采样率为16kHz的单声道音频，并输出一个(num_frames, num_classes)矩阵形式的发言者对白分割结果，其中7个类别分别是_非语音_, 发言者 #1, 发言者 #2, 发言者 #3, 发言者 #1 和 #2, 发言者 #1 和 #3, 以及_发言者 #2 和 #3_。

wespeaker-voxceleb-resnet34-LM：围绕 WeSpeaker 的 wespeaker-voxceleb-resnet34-LM 预训练说话人嵌入模型的封装，用于 pyannote.audio 中。

# 一、运行环境准备
### 表 1 版本配套表
|配套|版本 |环境准备指导 |
|--|--|--|
|Python|3.10.12|-|
|torch|2.5.1+cpu|-|
|torch_npu|2.5.1|-|

 **硬件设备**
|设备型号|NPU配置|
|---|---|
|Atlas 800I A2 910B|	1卡| 

可以通过下面的命令安装torch2.5.1+cpu版本
```
pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch-npu==2.5.1 -i https://mirrors.huaweicloud.com/repository/pypi/simple --no-cache-dir
```
通过pip list | grep torch查看信息如下：
```
[root:speaker-diarization-3.1]$ pip list | grep torch
pytorch-lightning                        2.6.0
pytorch-metric-learning                  2.9.0
torch                                    2.5.1+cpu
torch-audiomentations                    0.12.0
torch-npu                                2.5.1
torch_pitch_shift                        1.2.5
torchaudio                               2.5.1+cpu
torchdata                                0.11.0
torchmetrics                             1.8.2
torchvision                              0.20.1+cpu
```

# 二、下载模型权重
## 1安装modelscope
```
pip install modelscope
```
## 2	进入并创建目录speaker-diarization-3.1
```
mkdir pyannote
cd pyannote
mkdir speaker-diarization-3.1
modelscope download --model pyannote/speaker-diarization-3.1 --local_dir ./speaker-diarization-3.1
```
## 3	下载segmentation-3.0
```
mkdir segmentation-3.0
modelscope download --model pyannote/segmentation-3.0 --local_dir ./segmentation-3.0
```

## 3	下载wespeaker-voxceleb-resnet34-LM
```
mkdir wespeaker-voxceleb-resnet34-LM
modelscope download --model pyannote/wespeaker-voxceleb-resnet34-LM --local_dir ./wespeaker-voxceleb-resnet34-LM
```

# 三、安装依赖
## 1安装paynnote.audio
paynnote.audio是语音识别和分割的基础软件；要求版本>3.1。
```
pip install pyannote.audio==3.1.1
```
查看pyannote软件版本信息，缺省为：
```
[root:speaker-diarization-3.1]$ pip list | grep pyannote
pyannote.audio                           3.1.1
pyannote-core                            6.0.1
pyannote-database                        6.1.0
pyannote-metrics                         4.0.0
pyannote-pipeline                        4.0.0
```

再安装指定版本的pyannote.pipeline和pyannote.metrics版本：
```
pip install -y pyannote.pipeline==3.0.1
pip install -y pyannote.metrics==3.2
```

通过pip list | grep pyannote查询信息：
```
[root:speaker-diarization-3.1]$ pip list | grep pyannote
pyannote.audio                           3.1.1
pyannote.core                            5.0.0
pyannote.database                        5.1.3
pyannote.metrics                         3.2
pyannote.pipeline                        3.0.1
```

## 2安装ffmpeg
ffmpeg在语音处理时也会被使用；
```
apt update && apt install ffmpeg
```
## 3安装其他依赖
安装一些其他需要的依赖：
```
pip install transformers==4.55
pip install numpy==1.26.4
pip install matplotlib pydub ml-dtypes
```
如果transformers版本过低，会出现一些其他错误。

# 三、运行指导
## 1	修改与配置config.yaml
进入speaker-diarization-3.1目录：
```
cd speaker-diarization-3.1
```
config.yaml修改如下：
```
embedding: pyannote/wespeaker-voxceleb-resnet34-LM
...
segmentation: pyannote/segmentation-3.0
修改为（路径下的pytorch_model.bin文件）：
embedding: /pyannote/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin
...
segmentation:/pyannote/segmentation-3.0/pytorch_model.bin

```

## 2	进行推理脚本
在speaker-diarization-3.1目录，编写自己的运行脚本 infer.py:
```
from pyannote.audio import Pipeline
from pyannote.audio import Inference
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torchaudio

if torch.npu.is_available():
    device = torch.device("npu:0") 
else:
    device = torch.device("cpu")

# 读取配置文件
pipeline = Pipeline.from_pretrained("/inspire/sj-ssd/project/embodied-multimodality-ascend/public/models/pyannote/speaker-diarization-3.1/config.yaml")
pipeline = pipeline.to(device)

waveform, sample_rate = torchaudio.load("/inspire/sj-ssd/project/embodied-multimodality-ascend/public/models/pyannote/speaker-diarization-3.1/R8005_segment_01.wav")
print("音频采样率:",sample_rate)
print("音频张量形状:",waveform.dim())
diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

# 声纹识别
#diarization = pipeline("audio_sample/speech_mixure1.wav")

# 输出结果
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
```
再通过python  infer.py命令运行。
输出结果：
```
音频采样率: 16000
音频张量形状: 2
start=7.9s stop=9.6s speaker_SPEAKER_03
start=9.9s stop=15.8s speaker_SPEAKER_03
start=16.1s stop=16.2s speaker_SPEAKER_02
start=16.2s stop=17.2s speaker_SPEAKER_03
start=17.2s stop=25.0s speaker_SPEAKER_02
start=17.3s stop=17.4s speaker_SPEAKER_03
start=17.4s stop=17.7s speaker_SPEAKER_00
start=17.7s stop=17.8s speaker_SPEAKER_03
start=17.8s stop=18.1s speaker_SPEAKER_01
```

# 三、其他说明
## 1 出现错误的文件修改
错误1：
```
File "/usr/local/python3.11.13/lib/python3.11/site-packages/torchaudio/compliance/kaldi.py", line 616, in fbank
    spectrum = torch.fft.rfft(strided_input).abs()
 ```
修改为：
```
 #spectrum = torch.fft.rfft(strided_input).abs()
c = torch.fft.rfft(strided_input)
spectrum = torch.hypot(c.real, c.imag)
```
## 2 numpy中出现np.NaN不支持的错误
修改方式1：安装numpy==1.26.4版本
```
 pip install ==1.26.4
```
修改方式2：
在对应的安装目录pyannote/audio/pipelines/speaker_diarization.py 和inference.py中，修改为：
```
np_version=version.parse(np.__version__)
if np_version >= version.parse("2.0"):
    np_value = np.nan
else:
    np_value = np.NaN
...
```
用np_value替换np.nan.
