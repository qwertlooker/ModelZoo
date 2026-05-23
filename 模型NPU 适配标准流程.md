# 模型 NPU 适配标准流程

## 一句话指令

> 先克隆 upstream，确认远端最新 commit，区分上游源码改动和当前适配脚本。上游已有文件的修改必须生成 patch；新增 `infer.py` 不放进 patch，直接放当前模型目录。`infer.py` 只保留一个，默认 `--device npu`，CPU 验证用 `--device cpu`，不要使用 `auto/use_gpu`，不要写死 `npu:0/cuda:0`，实际设备由环境变量控制。必须补全环境搭建、权重下载、测试数据下载、CPU 当前环境验证、NPU 验证说明。最后生成 `ANALYSIS.md`、`NPU_ADAPTATION.md`、`NPU_VALIDATION.md`，并验证 `git apply --check`、`py_compile`、当前环境 CPU 推理。

---

## 详细流程

### Step 1：确认目标目录和已有文件

```bash
find <model_dir> -maxdepth 3 -type f | sort
git status --short
```

确认：

- 是否已有 `README.md`；
- 是否已有 `infer.py` 或多个推理脚本；
- 是否已有 `requirements.txt` / `environment.yml` / 安装说明；
- 是否已有上游源码副本；
- 是否已有权重、测试数据、示例音频/图片/文本。

---

### Step 2：确认上游仓库和最新 commit

```bash
git clone <upstream_repo> <model_dir>/upstream
git -C <model_dir>/upstream rev-parse HEAD
git -C <model_dir>/upstream ls-remote origin <default_branch>
```

必须记录：

- upstream repo URL；
- default branch；
- 本地 clone commit；
- 远端最新 commit；
- 检查日期；
- 当前适配是否匹配远端最新版本。

如果上游有更新，必须先分析差异，不能直接套旧 patch。

---

### Step 3：区分三类文件

#### A. 上游已有文件

例如：

```text
BEATs.py
fireredasr/models/fireredasr.py
speech2text.py
nemo/collections/asr/...
```

如果要改，必须在 `<model_dir>/upstream` 中修改并生成 patch。

#### B. 当前适配新增文件

例如：

```text
infer.py
ANALYSIS.md
NPU_ADAPTATION.md
NPU_VALIDATION.md
patches/README.md
scripts/download_weights.sh
scripts/download_test_data.sh
```

这些不进入 patch，直接放当前模型目录维护。

#### C. upstream 克隆目录

只用于对比和验证，不提交。应加入根目录 `.gitignore`：

```text
<model_dir>/upstream/
```

---

### Step 4：扫描设备相关代码

重点搜索：

```bash
grep -RIn "cuda\|gpu\|npu\|to(device)\|torch.load\|map_location\|nccl" <model_dir>/upstream <model_dir> --exclude-dir=.git
```

重点处理：

```python
.cuda()
torch.cuda.*
device="cuda"
backend="nccl"
torch.load(...)
map_location="cuda"
```

要求：

- 找到所有硬编码 CUDA/GPU 的节点；
- 判断是否属于推理链路；
- 对上游已有文件的修改生成 patch；
- 对新增 `infer.py` 直接维护在模型目录。

---

### Step 5：设备适配原则

推荐：

```python
import torch

if args.device == "npu":
    import torch_npu  # 注册 NPU backend

device = torch.device(args.device)
model.to(device)
tensor = tensor.to(device)
```

默认：

```bash
--device npu
```

CPU 验证：

```bash
--device cpu
```

不要在代码里绑定：

```text
npu:0
cuda:0
auto
use_gpu
```

实际卡号用环境变量控制：

```bash
ASCEND_RT_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=0
```

---

### Step 6：环境搭建必须补全

每个模型必须在 `README.md` 和 `NPU_ADAPTATION.md` 中说明环境搭建方式。

至少包含：

- Python 版本；
- PyTorch 版本；
- `torch-npu` 版本；
- CANN / 驱动 / 固件要求；
- 上游项目安装方式；
- 最小推理依赖；
- 当前 `requirements.txt` 是否为最小依赖，若不是必须说明；
- CPU 验证环境是否可跳过 `torch-npu`，或是否需要条件导入。

推荐结构：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# CPU 验证最小依赖
pip install -r requirements_cpu.txt

# NPU 推理依赖，版本以 CANN 对应版本为准
pip install torch torch-npu
pip install -r requirements.txt
```

如果无法拆分 `requirements_cpu.txt` / `requirements.txt`，必须在文档中写明最小依赖建议。

---

### Step 7：权重下载必须补全

每个模型必须说明权重来源和下载方式。

至少包含：

- 官方权重 URL，例如 Hugging Face、ModelScope、GitHub Release、官方网盘；
- 权重文件名或目录结构；
- 下载命令；
- 离线部署方式；
- `infer.py` 如何通过 `--model` / `--checkpoint` / `--model_dir` 指定权重；
- 权重不要写死到用户本地路径；
- 大文件不要直接提交，必要时使用 Git LFS 或下载脚本。

示例：

```bash
mkdir -p <model_dir>/weights
huggingface-cli download <org/model> \
  --local-dir <model_dir>/weights/<model_name> \
  --local-dir-use-symlinks False

python infer.py --model <model_dir>/weights/<model_name> --device cpu ...
```

如果需要登录：

```bash
huggingface-cli login
```

必须在 `NPU_VALIDATION.md` 记录实际验证使用的权重路径或权重来源。

---

### Step 8：测试数据集 / 测试样例必须补全

每个模型必须提供可复现的 CPU 和 NPU 验证输入。

至少包含：

- 测试数据来源 URL；
- 下载命令；
- 数据保存路径；
- 最小样例文件名；
- 预处理命令；
- 预期输入格式；
- 预期输出类型：文本、分类 top-k、shape、音频文件、图像文件等。

示例：

```bash
mkdir -p <model_dir>/test_data
wget -O <model_dir>/test_data/sample.wav <sample_url>

python infer.py \
  --audio <model_dir>/test_data/sample.wav \
  --device cpu
```

如果官方没有小样例，必须提供以下之一：

- 从公开数据集中下载 1 条样本；
- 用脚本生成一个最小可运行的假输入；
- 在文档中清楚说明用户需要准备的数据格式。

---

### Step 9：patch 生成原则

只对上游已有文件生成 patch：

```bash
mkdir -p <model_dir>/patches
git -C <model_dir>/upstream diff -- <upstream_existing_file> \
  > <model_dir>/patches/0001-xxx.patch
```

验证：

```bash
git -C <model_dir>/upstream apply --check ../patches/0001-xxx.patch
```

如果没有修改上游已有文件：

- 不生成空 patch；
- 建议写 `patches/README.md`，说明“本次适配未修改上游源码，因此无 patch”。

---

### Step 10：infer.py 处理原则

- 当前适配目录尽量只保留一个 `infer.py`；
- NPU/CPU 融合；
- 默认 `--device npu`；
- CPU 验证显式使用：
  ```bash
  --device cpu
  ```
- 不使用 `auto`；
- 不使用 `use_gpu`；
- 不指定 `npu:0` / `cuda:0`；
- 所有路径参数化：模型权重、输入数据、输出目录；
- 支持打印可验证结果，例如识别文本、top-k、输出 shape、保存文件路径；
- `torch_npu` 必须条件导入，避免 CPU-only 环境无法 import 脚本；
- 脚本必须通过：
  ```bash
  python3 -m py_compile <model_dir>/infer.py
  ```

---

### Step 11：当前环境 CPU 验证（必须执行）

即使没有 NPU，也必须尽量使用当前环境执行 CPU 验证流程。

#### 11.1 先做轻量静态验证

```bash
python3 -m py_compile <model_dir>/infer.py
```

#### 11.2 准备 CPU 可运行依赖

优先使用最小依赖，不要盲目安装超大训练环境：

```bash
pip install -r <model_dir>/requirements.txt
# 或按文档安装最小 CPU 依赖
```

如果当前环境无法安装依赖，必须在 `NPU_VALIDATION.md` 写明阻塞原因，例如：

- 缺少系统库；
- Python 版本不兼容；
- 网络无法下载权重；
- 权重需要授权；
- 当前机器磁盘/内存不足。

#### 11.3 下载或准备权重与测试数据

按照文档中的下载命令执行，或使用已有本地路径：

```bash
# 权重
<download_weights_command>

# 测试数据
<download_test_data_command>
```

必须记录：

- 权重路径；
- 测试数据路径；
- 是否为官方样例；
- 是否为 dummy 输入。

#### 11.4 执行 CPU 推理

必须尝试执行：

```bash
python <model_dir>/infer.py --device cpu <model_args> <input_args>
```

CPU 验证至少记录：

- 命令；
- 是否成功；
- 输出文本 / top-k / shape / 文件路径；
- 运行耗时（如方便）；
- 如果失败，完整错误摘要和下一步处理建议。

#### 11.5 CPU 验证判定

- 成功：说明 `infer.py` 参数、权重加载、数据读取、主推理链路基本正确；
- 失败但可解释：记录为“当前环境 CPU 验证阻塞”，不可简单省略；
- 不能因为无 NPU 而跳过 CPU 验证。

---

### Step 12：NPU 验证

有 NPU 环境时执行：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python <model_dir>/infer.py --device npu <model_args> <input_args>
```

必要时对比 CPU：

```bash
python <model_dir>/infer.py --device cpu <model_args> <input_args>
```

对比内容：

- 输出 shape；
- top-k label 或识别文本；
- 输出文件是否生成；
- 是否有设备不匹配错误；
- NPU 运行耗时；
- 首次编译/加载耗时和稳定推理耗时尽量分开记录。

---

### Step 13：文档必须包含

每个模型至少生成：

```text
ANALYSIS.md
NPU_ADAPTATION.md
NPU_VALIDATION.md
README.md
```

#### ANALYSIS.md 必须包含

- upstream repo；
- upstream commit；
- 是否匹配远端最新版本；
- 当前目录原有文件分析；
- 扫描到的设备相关节点；
- 修改了哪些上游源码节点；
- 是否生成 patch；
- 已知风险和限制；
- 上游更新时如何处理。

#### NPU_ADAPTATION.md 必须包含

- 环境搭建；
- 权重下载；
- 测试数据下载；
- patch 如何应用；
- `infer.py` 参数说明；
- CPU 推理命令；
- NPU 推理命令；
- 常见问题。

#### NPU_VALIDATION.md 必须包含

- upstream clone / commit 验证；
- `git apply --check` 结果，或说明无 patch；
- `py_compile` 结果；
- 当前环境 CPU 验证命令和结果；
- NPU 验证命令和结果，若无 NPU 则说明未执行原因；
- 权重路径；
- 测试数据路径；
- 输出摘要；
- 已知限制。

#### README.md 必须包含

- 模型简介；
- 硬件/软件约束；
- 环境安装；
- 权重下载；
- 测试数据下载；
- CPU 验证；
- NPU 推理；
- 文件说明。

---

### Step 14：最终验证清单

提交前至少执行：

```bash
# 1. 文件检查
find <model_dir> -maxdepth 3 -type f | sort

# 2. patch 检查；如果没有 patch，文档必须说明
for p in <model_dir>/patches/*.patch; do
  git -C <model_dir>/upstream apply --check "../patches/$(basename "$p")"
done

# 3. Python 语法检查
python3 -m py_compile <model_dir>/infer.py

# 4. 当前环境 CPU 验证
python <model_dir>/infer.py --device cpu <model_args> <input_args>

# 5. NPU 验证，有 NPU 环境时执行
ASCEND_RT_VISIBLE_DEVICES=0 python <model_dir>/infer.py --device npu <model_args> <input_args>
```

如果某一步无法执行，不能删除该步骤，必须在 `NPU_VALIDATION.md` 中记录：

- 未执行命令；
- 未执行原因；
- 需要用户提供什么；
- 后续如何补验。

---
