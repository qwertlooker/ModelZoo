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
DiariZen 是一款由 AudioZen 与 Pyannote 3.1 驱动的说话人分轨工具包。
github中的地址为：https://github.com/BUTSpeechFIT/DiariZen

# 一、运行环境准备

## 表 1 版本配套表
|配套|版本 |环境准备指导 |
|--|--|--|
|Python|3.10.12|-|
|torch|2.5.1+cpu|-|
|torch_npu|2.5.1|-|

 **硬件设备**
|设备型号|NPU配置|
|---|---|
|Atlas 800I A2 910B|	1卡| 

# 二、下载模型权重
## 1 从github中下载DiariZen文件
```
git clone https://github.com/BUTSpeechFIT/DiariZen.git
```
或者
```
git clone https://githubfast.com/BUTSpeechFIT/DiariZen.git
```

## 2 从huggingface.co中下载diarizen-wavlm-large-s80-md文件
```
export HF_ENDPOINT=https://hf-mirror.com
git clone https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md.git
```

## 3 下载dscore子目录
```
git submodule init
git submodule update
```
如果下载不成功，则使用下列命令：
```
rm -rf dscore
git clone https://githubfast.com/nryant/dscore.git dscore
```

# 三、安装python虚拟环境和pyannote软件
## 1	conda创建python虚拟环境
```
conda create --name diarizen python=3.10
conda activate diarizen
```
如果conda软件不存在，则下载Miniconda3-py311_24.1.2-0-Linux-x86_64.sh，并安装
```
bash Miniconda3-py311_24.1.2-0-Linux-x86_64.sh
```

## 2	进入虚拟环境diarizen,并安装基础torch软件
```
pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt && pip install -e .
pip install ml-dtypes cloudpickle
```

## 3	安装torch_npu软件以及其他依赖
```
pip install torch-npu==2.5.1 -i https://mirrors.huaweicloud.com/repository/pypi/simple --no-cache-dir
```
注意，这里一定要保证torch-npu与torch版本匹配，否则后面就容易出现各种错误；
```
(diarizen) [root:DiariZen]$ pip list | grep torch
torch                     2.5.1+cpu
torch-npu                 2.5.1
torchaudio                2.5.1+cpu
torchinfo                 1.8.0
torchvision               0.20.1+cpu
```

## 4	下载并安装pyannote所有相关的软件
1，安装缺省的pyannote.audio=3.1.1版本
```
cd pyannote-audio && pip install -e .[dev,testing]
cd ../
```

2，下载除pyannote.audio外其他pyannote软件(与pyannote.audio=3.1.1配套的版本)
```
wget https://githubfast.com/pyannote/pyannote-core/archive/refs/tags/5.0.0.tar.gz
wget https://githubfast.com/pyannote/pyannote-database/archive/refs/tags/5.0.1.tar.gz
wget https://githubfast.com/pyannote/pyannote-metrics/archive/refs/tags/3.2.tar.gz
wget https://githubfast.com/pyannote/pyannote-pipeline/archive/refs/tags/3.0.1.tar.gz
```

3，修改名称并解压文件：
```
mv 5.0.0.tar.gz pyannote-core5.0.0.tar.gz
mv 5.0.1.tar.gz pyannote-database5.0.1.tar.gz
mv 3.2.tar.gz pyannote-metrics3.2.tar.gz
mv 3.0.1.tar.gz pyannote.pipeline3.0.1.tar.gz
tar -zxvf pyannote-core5.0.0.tar.gz
tar -zxvf pyannote-database5.0.1.tar.gz
tar -zxvf pyannote-metrics3.2.tar.gz
tar -zxvf pyannote.pipeline3.0.1.tar.gz

 ```
 
 4，安装其他pyannote软件
```
pip uninstall pyannote-database pyannote-metrics pyannote-pipeline pyannote-core
cd pyannote-metrics-3.2 && pip install -e .
cd ../pyannote-pipeline-3.0.1 && pip install -e .
cd ../pyannote-database-5.0.1 && pip install -e .
cd ../pyannote-core-5.0.0 && pip install -e .
cd ../

 ```
 说明：如果不安装pyannote-core、pyannote-database等软件，推理运行会出现错误；

# 四、安装其他依赖
## 1安装ffmpeg
```
apt update && apt install ffmpeg
```
## 2安装numpy==1.26.4
```
pip install numpy==1.26.4
```

# 五、运行指导
## 1 代码修改
File "/root/miniconda3/envs/diarizen/lib/python3.10/site-packages/torchaudio/compliance/kaldi.py", line 616，修改为：
```
 #spectrum = torch.fft.rfft(strided_input).abs()
c = torch.fft.rfft(strided_input)
spectrum = torch.hypot(c.real, c.imag)
```
不修改会出现torch.fft.rfft(strided_input).abs()不支持DT_COMPLEX64的错误

## 2	创建infer.py进行推理
回到目录(diarizen) [root:DiariZen]下，参考https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md使用说明，创建infer.py进行推理
```
from diarizen.pipelines.inference import DiariZenPipeline


# load pre-trained model
diar_pipeline = DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-wavlm-large-s80-md")
# apply diarization pipeline
diar_results = diar_pipeline('./example/EN2002a_30s.wav')

# print results
for turn, _, speaker in diar_results.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

# load pre-trained model and save RTTM result
diar_pipeline = DiariZenPipeline.from_pretrained(
        "BUT-FIT/diarizen-wavlm-large-s80-md",
        rttm_out_dir='.'
)
# apply diarization pipeline
diar_results = diar_pipeline('./example/EN2002a_30s.wav', sess_name='session_name')
```

## 3	其他问题
### 1 推理时出现连接huggingface.co失败， 则设置镜像
```
export HF_ENDPOINT=https://hf-mirror.com
```
### 2 加载本地模型出现“多余1个/”失败， 则创建一个软链接
```
ln -s /inspire/sj-ssd/project/embodied-multimodality-ascend/public/xxx/DiariZen/but-fit/models--but-fit--diarizen-wavlm-large-s80-md ~/.cache/huggingface/hub/models--but-fit-- diarizen-wavlm-large-s80-md
```
## 3 numpy中出现np.NaN不支持的错误
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
