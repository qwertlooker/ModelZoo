---
license: apache-2.0
library_name: pytorch
hardware: NPU
---
### 硬件设备
|设备型号|NPU配置|
|---|---|
|Ascend 910B|	1卡| 

### 软件版本配套表
**基础镜像（供参考）** ：使用 cann:8.3.rc1 或者 vllm-ascend:0.10.2rc1
```
docker pull quay.io/ascend/cann:8.3.rc1
or
docker pull quay.io/ascend/vllm-ascend:v0.10.2rc1
```

|配套|版本| 环境准备指导|
|---|---|---|
|CANN | 8.3.RC1 | - |
|Python| 3.10.12 | - |
|torch| 2.5.1 | - |
|torch_npu|	2.5.1| - |
|transformers| 4.52.1 | - |
|torchaudio| 2.5.1 | - |

### 模型介绍
IndexTTS2模型提出了一种新颖、通用且适合自回归模型使用的语音时长控制方法。该方法支持两种生成模式：一种明确指定生成token的数量，以精确控制语音时长；另一种则以自回归的方式自由生成语音，不指定token数量，同时忠实再现输入提示的韵律特征。此外，IndexTTS2实现了情感表达与说话人身份的解耦，从而能够独立控制音色和情感。在零样本设置下，模型能够准确地重建目标音色（来自音色提示），同时完美地再现指定的情感语调（来自风格提示）。为了增强高情感表达下的语音清晰度，我们引入了GPT潜在表示，并设计了一种新颖的三阶段训练范式，以提高生成语音的稳定性。此外，为了降低情感控制的门槛，我们基于文本描述设计了一种软指令机制，通过微调Qwen3模型，有效引导生成具有所需情感倾向的语音。

###  离线适配

[Index-TTS——GitHub仓库](https://github.com/index-tts/index-tts/tree/main)
#### 1. 源码下载
```
git lfs install
git clone https://github.com/index-tts/index-tts.git
cd index-tts
git lfs pull  # download large repository files
```

#### 2. 安装依赖：
①可以先尝试官方文档，用uv安装依赖包
```
pip install -U uv
uv sync --all-extras
```
②如果uv安装失败，则用pip安装，手动添加以下requirement.txt，再pip install -r；（建议先按照下面方法手动安装pynini再执行）
```
#requirement.txt
accelerate==1.8.1
cn2an==0.5.22
cython==3.0.7
descript-audiotools==0.7.2
ffmpeg-python==0.2.0
g2p-en==2.1.0
jieba==0.42.1
json5==0.10.0
keras==2.9.0
librosa==0.10.2.post1
matplotlib==3.8.2
modelscope==1.27.0
munch==4.0.0
numba==0.58.1
numpy==1.26.2
omegaconf>=2.3.0
opencv-python==4.9.0.80
pandas==2.3.2
safetensors==0.5.2
sentencepiece>=0.2.1
tensorboard==2.9.1
textstat>=0.7.10
tokenizers==0.21.0
tqdm>=4.67.1
transformers==4.52.1
wetext>=0.0.9; sys_platform != 'linux'
WeTextProcessing; sys_platform == 'linux'
```
**安装WeTextProcessing的pynini依赖时大概率会报错，建议先按照以下步骤手动安装pynini 2.1.6**
```
（1）下载openfst1.8.3源码
cd /tmp
wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.8.3.tar.gz --no-check-certificate
tar -zxvf openfst-1.8.3.tar.gz && cd openfst-1.8.3
（2）执行configure
./configure \
--prefix=/usr/local \
--enable-shared \
--disable-static \
--enable-far \
--enable-grm \
--enable-mpdt \
--build=aarch64-unknown-linux-gnu
（3）编译并安装
make -j4 && make install
（4）检查关键库是否生成
ls /usr/local/lib/libfst.so*          # 基础库
ls /usr/local/lib/libfstfarscript.so* # FAR 模块库
ls /usr/local/lib/libfstscript.so*    # Script 模块库（默认启用）
ls /usr/local/lib/libfstmpdtscript.so* # MPDT 模块库
（5）下载pynini2.1.6源码
cd /tmp
wget https://files.pythonhosted.org/packages/9b/69/4b59968b0fd351a153d7a3c2feaa6e7514c903c063bbd7f7e1e3f1c079b0/pynini-2.1.6.tar.gz --no-check-certificate
tar -zxvf pynini-2.1.6.tar.gz && cd pynini-2.1.6
（6）编译安装pynini
# 指定 OpenFST 头文件和库路径
export CFLAGS="-I/usr/local/include"
export LDFLAGS="-L/usr/local/lib"
python setup.py install
```
如果镜像里没有ffmpeg可能还需要另外安装。

#### 3. 下载模型权重
若官方的huggingface下载速度慢，可以在modelscope上下载
```
modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
modelscope download --model facebook/w2v-bert-2.0 --local_dir models/facebook/w2v-bert-2.0
modelscope download --model amphion/MaskGCT semantic_codec/model.safetensors --local_dir models/amphion/MaskGCT
modelscope download --model iic/speech_campplus_sv_zh-cn_16k-common campplus_cn_common.bin --local_dir models/iic/speech_campplus_sv_zh-cn_16k-common
modelscope download --model nv-community/bigvgan_v2_22khz_80band_256x bigvgan_generator.pt --local_dir models/nv-community/bigvgan_v2_22khz_80band_256x
modelscope download --model nv-community/bigvgan_v2_22khz_80band_256x config.json --local_dir models/nv-community/bigvgan_v2_22khz_80band_256x
```
#### 4. 修改torchaudio
修改torchaudio的fbank.py，使fbank支持在npu上运行，否则会报错

[fbank_npu.patch文件获取](https://ai.gitcode.com/Ascend-SACT/WeNet/blob/main/fbank_npu.patch)

```
# 将patch放到torchaudio的路径下，如果路径不一致请自行修改
pip show torchaudio
cp fbank_npu.patch /usr/local/lib/python3.11/site-packages/torchaudio/

# 在torchaudio路径下执行补丁
patch -p1 < fbank_npu.patch
```

#### 5. 修改infer_v2.py
修改./indextts/infer_v2.py文件,修改实际权重路径，以及添加import torch_npu；

```
### 添加torch_npu库
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
······
torch_npu.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format=False
torch.npu.set_device(0)
```

```
### 修改本地权重路径
# self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained("./models/facebook/w2v-bert-2.0", local_files_only=True)

······

# semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
semantic_code_ckpt = "./models/amphion/MaskGCT/semantic_codec/model.safetensors"

······

# load campplus_model
# campplus_ckpt_path = hf_hub_download(
#     "funasr/campplus", filename="campplus_cn_common.bin"
# )
campplus_ckpt_path = "./models/iic/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"

······

# bigvgan_name = self.cfg.vocoder.name
bigvgan_name = './models/nv-community/bigvgan_v2_22khz_80band_256x'

```
[具体修改见infer_v2.py](https://ai.gitcode.com/Ascend-SACT/Index-TTS-2/blob/main/infer_v2.py)

#### 6. 运行webui.py

执行 bash run_web.sh 来拉起webui.py；

![image.png](https://raw.gitcode.com/user-images/assets/8428614/4d220a62-1d65-45fe-bf40-530408d2eadb/image.png 'image.png')

![image.png](https://raw.gitcode.com/user-images/assets/8428614/459d6c1c-4895-4bab-8bf5-942a8811034c/image.png 'image.png')

#### 7. 性能数据

| 硬件环境  | RTF（实时率）  |
| ------------ | ------------ |
|  CPU |  45 |
| NPU  | 1.6  |
| GPU | 1.4 |

Index-TTS-2 （CPU）：
```
>> gpt_gen_time: 239.82 seconds
>> gpt_forward_time: 1.42 seconds
>> s2mel_time: 35.52 seconds
>> bigvgan_time: 11.87 seconds
>> Total inference time: 312.05	seconds
>> Generated audio length: 6.83	seconds
>> RTF: 45.7110
```

Index-TTS-2 （NPU）：
```
>> gpt gen time: 12.68 seconds
>> gpt forward_time: 0.02 seconds
>> s2mel time: 0.78 seconds
>> bigvgan time: 0.06 seconds
>> Total inference time: 14.69 secondse
>> Generated audio length: 9.10 secondsce
>> RTF: 1.6142
```

Index-TTS-2 （GPU）：
```
>> gpt_gen_time: 26.93 seconds
>> gpt forward time: 0.04 seconds
>> s2mel time: 3.48 seconds
>> bigvgan_time: 0.55 seconds
>> Total inference time: 31.92 secondsc
>> Genenated audio length: 22.51 secondssr
>> RTF: 1.4178
```