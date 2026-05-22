---
library_name: pytorch
language:
  - zh
pipeline_tag: text-to-speech
hardware: NPU
---
## 1. 模型介绍
IndexTTS2模型提出了一种新颖、通用且适合自回归模型使用的语音时长控制方法。该方法支持两种生成模式：一种明确指定生成token的数量，以精确控制语音时长；另一种则以自回归的方式自由生成语音，不指定token数量，同时忠实再现输入提示的韵律特征。此外，IndexTTS2实现了情感表达与说话人身份的解耦，从而能够独立控制音色和情感。在零样本设置下，模型能够准确地重建目标音色（来自音色提示），同时完美地再现指定的情感语调（来自风格提示）。为了增强高情感表达下的语音清晰度，我们引入了GPT潜在表示，并设计了一种新颖的三阶段训练范式，以提高生成语音的稳定性。此外，为了降低情感控制的门槛，我们基于文本描述设计了一种软指令机制，通过微调Qwen3模型，有效引导生成具有所需情感倾向的语音。

## 2. 准备环境
### 2.1 硬件设备
硬件：NPU
型号：Ascend 910B3

### 2.2 软件配套版本
组件：版本
CANN:8.3.RC1
Python:3.11.13
torch :2.8.0
torch_npu:v2.8.0-7.2.0

### 2.3 启动容器

以 CANN 8.3.RC1，910b 机器，ubuntu22.04 系统，py3.11 为例：
```
#  拉取镜像
docker pull quay.io/ascend/cann:8.3.rc1
#  启动容器
docker run -itd --privileged --name=${container_name}  --net=host \
   --shm-size 500g \
   --device=/dev/davinci0 \
   --device=/dev/davinci_manager \
   --device=/dev/hisi_hdc \
   --device /dev/devmm_svm \
   -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
   -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
   -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
   -v /usr/local/sbin:/usr/local/sbin \
   -v /etc/hccn.conf:/etc/hccn.conf \
   -v /var/run/docker.sock:/var/run/docker.sock \
   -v /usr/bin/docker:/usr/bin/docker \
   -v /etc/docker:/etc/docker \
   -v /var/lib/docker:/var/lib/docker \
   --entrypoint /bin/bash quay.io/ascend/cann:8.3.rc1
   #  进入容器
   docker exec -it ${container_name} /bin/bash
```
### 2.4 源码下载
```
git clone https://github.com/triomino/index-tts.git && cd index-tts
git lfs pull #  如果需要项目中的样例音频
```
## 3. 安装依赖
### 3.1 WeTextProcessing安装

在pip直接安装可能会出错，单独进行安装：
```
#  下载安装包并解压
wget https://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.8.3.tar.gz
tar -zxvf openfst-1.8.3.tar.gz
#  进入目录后编译安装
cd openfst-1.8.3
./configure --enable-far --enable-mpdt --enable-pdt
make -j$(nproc)
make install
#  确认动态库文件存在：
ls /usr/local/lib/libfstmpdtscript.so.26
#  配置动态库路径
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
sudo ldconfig
#  安装WeTextProcessing
pip3 install WeTextProcessing==1.0.4.1
```
### 3.2 安装仓库依赖
```
pip install -e .
```
### 3.3 安装系统依赖
```
apt-get update
apt-get install -y patch build-essential libbz2-dev libreadline-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev liblzma-dev m4 dos2unix libopenblas-dev git
```
### 3.4 安装GCC 13.3.0
```
#  v2.8.0-7.2.0 要求gcc=13.3.0 cmake>=3.31.0
#  gcc安装（耗时约1h）
wget https://repo.huaweicloud.com/gnu/gcc/gcc-13.3.0/gcc-13.3.0.tar.gz
apt-get install bzip2
tar -zxvf gcc-13.3.0.tar.gz
cd gcc-13.3.0
./contrib/download_prerequisites    # 这步可能有网络问题，需联网
./configure --enable-languages=c,c++ --disable-multilib --with-system-zlib --prefix=/usr/local/gcc13.3.0
make -j15    #  通过grep -w processor /proc/cpuinfo|wc -l查看cpu数，示例为15，用户可自行设置相应参数。
make install

export LD_LIBRARY_PATH=/usr/local/gcc13.3.0/lib64:${LD_LIBRARY_PATH}
export CC=/usr/local/gcc13.3.0/bin/gcc
export CXX=/usr/local/gcc13.3.0/bin/g++
export PATH=/usr/local/gcc13.3.0/bin:${PATH}
gcc --version  # 回显gcc (GCC) 13.3.0表示安装成功，将上面的环境变量添加到~/.bashrc避免每次进容器都执行
```
### 3.5 安装CMake 3.31.0
```
#  cmake安装
wget https://cmake.org/files/v3.31/cmake-3.31.0.tar.gz
tar -xf cmake-3.31.0.tar.gz
cd cmake-3.31.0/
./configure --prefix=/usr/local/cmake
make && make install
ln -s /usr/local/cmake/bin/cmake /usr/bin/cmake
cmake --version   # 回显cmake version 3.31.0则成功
```
### 3.6 安装torch_npu v2.8.0-7.2.0
```
#  安装torch_npu
git clone https://gitcode.com/ascend/pytorch.git -b v2.8.0-7.2.0 --depth 1
cd pytorch
bash ci/build.sh --python=3.11
pip3 install --upgrade dist/torch_npu-2.8.0.post1+gitc7b6b32-cp311-cp311-linux_aarch64.whl
pip list|grep torch_npu # 查看是否安装成功
```
**安装完成后比对pip包版本，是否是torchaudio==2.8.0和torchvision==0.23.0**
### 3.7 拉取模型
```
modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
modelscope download --model facebook/w2v-bert-2.0 --local_dir models/facebook/w2v-bert-2.0
modelscope download --model amphion/MaskGCT semantic_codec/model.safetensors --local_dir models/amphion/MaskGCT
modelscope download --model iic/speech_campplus_sv_zh-cn_16k-common campplus_cn_common.bin --local_dir models/iic/speech_campplus_sv_zh-cn_16k-common
modelscope download --model nv-community/bigvgan_v2_22khz_80band_256x bigvgan_generator.pt --local_dir models/nv-community/bigvgan_v2_22khz_80band_256x
modelscope download --model nv-community/bigvgan_v2_22khz_80band_256x config.json --local_dir models/nv-community/bigvgan_v2_22khz_80band_256x
```
 
## 4. 代码适配
### 4.1 修改infer_v2.py
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
### 4.2 修改infer_v2_npu.py
ps:非本仓库的，是前面triomino/index-tts仓库的文件
```
def get_fixed_embedding_with_tensor_input(self, ind):
    #  新增以下判断，避免torch组图抛出ind Nonetype异常
    if ind is None:
        return torch.zeros(1, 1, self.emb.embedding_dim, device=self.emb.weight.device)
    return self.emb(ind).unsqueeze(0)
  ```
### 4.3 修改webui.py
```
#  51行
#  from indextts.infer_v2 import IndexTTS2
from indextts.infer_v2_npu import IndexTTS2NPU
 
#  56行
tts = IndexTTS2NPU(
    cfg_path="checkpoints/config.yaml", 
    model_dir="checkpoints", 
    use_cuda_kernel=True, 
    use_fp16=True, 
    static=True
)
# tts = IndexTTS2(model_dir=cmd_args.model_dir,
#                 cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
#                 use_fp16=cmd_args.fp16,
#                 use_deepspeed=cmd_args.deepspeed,
#                 use_cuda_kernel=cmd_args.cuda_kernel,
#                 )
```
## 5. 性能测试
### 5.1 启动webui
```
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export CPU_AFFINITY_CONF=1
export TASK_QUEUE_ENABLE=1
ASCEND_RT_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1 PYTHONPATH="$PYTHONPATH:."  python ./webui.py
```
### 5.2 测试结果

启动webui后，在浏览器访问 {服务器ip}:7860
选择任意example进行测试，在warmup完成后，index-tts-2的推理RTF在0.6左右，性能基本可用。

 