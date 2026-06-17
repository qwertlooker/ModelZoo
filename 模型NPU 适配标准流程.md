# 模型 NPU 适配标准流程

## 一句话指令

> 先克隆 upstream，确认远端最新 commit，并明确“当前适配的精确版本边界”：源码 repo/分支/commit、模型权重 repo/文件/commit 或校验值、辅助模型版本，以及明确排除同系列其他变体。区分上游源码改动和当前适配脚本。上游已有文件的修改必须生成 patch；新增 `infer.py` 不放进 patch，直接放当前模型目录。`infer.py` 只保留一个，默认 `--device npu`，CPU 验证用 `--device cpu`，不要使用 `auto/use_gpu`，不要写死 `npu:0/cuda:0`，实际设备由环境变量控制。适配/评测脚本必须按项目级“严格失败”原则实现：必需依赖统一前置 import，缺依赖、缺官方预期字段或版本不匹配时直接暴露原始错误，不添加不必要的 `try/except`、`hasattr/getattr`、regex/basic 替代、CPU/远端 fallback 等静默兼容。必须补全环境搭建、权重下载、测试数据下载、CPU 当前环境验证、NPU 验证说明。还必须参考原始模型的功能、性能、精度和公开评测，生成 `ACCEPTANCE_PLAN.md`；验收主线必须先写清“原始测试集是什么、原始指标是多少、NPU 如何对齐原始模型结果”，同 checkpoint、同测试集/manifest、同评测脚本、同参数比较 CPU/CUDA 原始路径与 NPU 结果；官方未发布测试集或指标时必须明确写“官方未发布”，不得编造。再按数据集大小、获取难度、验证难度设计 L0/L1/L2/L3 分层验收、通过条件和报告模板。最后生成 `ANALYSIS.md`、`NPU_ADAPTATION.md`、`NPU_VALIDATION.md`、`ACCEPTANCE_PLAN.md`，并验证 `git apply --check`、`py_compile`、下载 URL/脚本可用性、测试数据可用性、当前环境 CPU 推理；不能只补文档不做验证，不能只用 dummy smoke test 代替完整验收方案。

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

### Step 2.5：明确适配版本边界

每个模型在开始适配、补验或提交前，都必须明确“当前适配的到底是哪一个版本”。同一模型系列常见有多个变体，例如 `canary-1b` / `canary-1b-flash` / `canary-1b-v2`、`whisper-large-v3` / `large-v3-turbo`、`MossFormer2_SE_48K` / `SS_16K` / `SR_48K`。不能只写模型系列名。

必须记录到 `NPU_ADAPTATION_ANALYSIS.md` 的“参考原始仓库与适配版本边界”章节，并同步写入该模型的 `README.md` / `ANALYSIS.md` / `NPU_ADAPTATION.md` / `NPU_VALIDATION.md` 中合适位置：

- 源码来源：repo URL、默认分支、commit；如果使用子目录，写明子目录；
- 权重来源：Hugging Face / ModelScope / GitHub Release / 网盘 URL、repo HEAD 或发布版本、具体文件名或目录；
- 权重校验：优先记录 SHA256；大权重未下载时至少记录 metadata / HEAD 检查结果和未下载原因；
- 变体边界：明确“当前适配的是 X，不是 Y/Z”；
- 辅助模型：如 tokenizer、codec、vocoder、embedding、segmentation 等，也要记录 repo / 文件 / commit；
- 检查日期：记录当前检查日期，后续适配前重新执行远端检查。

推荐检查命令：

```bash
# GitHub / Hugging Face / ModelScope Git 端点
git ls-remote --symref <repo_url> HEAD

# 已克隆 upstream
git -C <model_dir>/upstream rev-parse HEAD
git -C <model_dir>/upstream remote -v

# 已下载权重，尽量记录校验值
sha256sum <weight_file>
```

如果暂时无法固定具体权重（例如 BEATs 需要用户选择某个 OneDrive checkpoint），必须在文档中标注“源码适配已固定，checkpoint 尚未固定”，并要求后续下载/验证时补充 checkpoint 名称、来源和 SHA256。

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

#### 5.0 项目级脚本严格失败原则

该原则适用于整个 ModelZoo 的所有适配脚本、评测脚本和数据准备脚本，不是某个模型目录的局部要求。除非本标准流程在具体步骤中明确允许，否则不要为了“跑通”而添加静默兼容层。

必须遵守：

- **必需依赖统一前置 import**：模型入口类、评测库、官方 normalizer / tokenizer / processor 等必需依赖应放在文件顶部导入；缺依赖时脚本启动阶段直接报原始 `ImportError` / `ModuleNotFoundError`。
- **设备后端可条件导入**：仅按设备选择才需要的后端注册模块可以条件导入，例如只有 `--device npu` 时导入 `torch_npu`，只有 `--device cuda` 时触发 CUDA 专用依赖。
- **官方评测路径不可替代**：公开指标要求的 normalizer、tokenizer、decode 参数、metric 实现必须使用官方或明确等价路径；不得用 regex/basic normalizer、其他同名包、简化 metric、CPU fallback 或远端自动 fallback 生成看似可用但不可对齐官方口径的结果。
- **官方预期字段直接访问**：对模型配置、解码配置、版本字段、推理输出结构等官方预期字段，直接访问并让缺字段报错；不要用 `hasattr/getattr`、宽泛 `try/except`、dict/string 兜底来掩盖环境或上游版本不匹配。
- **禁止吞错继续**：不要捕获宽泛 `Exception` 后继续执行；如必须捕获异常用于补充上下文，必须重新抛出，且不得切换到非官方替代实现。
- **兼容处理必须有依据**：如果确实需要兼容多个官方版本或多个公开权重变体，必须在文档中列出版本边界、触发条件、验证命令和指标影响；不能把未验证的兼容逻辑混入默认路径。

提交前应检查新增/修改脚本中是否存在不必要的 `try/except`、`hasattr/getattr`、`pass`、`fallback`、`auto`、`use_gpu`、硬编码设备、静默下载远端替代等模式；发现后要么删除，要么在文档中说明其必要性和验证结果。


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

#### 6.1 多 requirements / extras 的依赖闭环必须说明

如果上游项目提供多组 requirements 或 pip extras，适配文档必须说明目标任务所需的完整依赖闭环，不能只写触发当前报错的单个包。

必须记录：

- 每个 requirements 文件或 extra 大致对应的功能域；
- 目标任务需要安装哪些 requirements / extras；
- 某个 requirements 是否包含另一个 requirements；
- 推荐的一步到位安装命令；
- 如果 NPU 环境中的 `torch` / `torch-npu` 已经按 CANN 配好，如何避免被 pip 覆盖。

以 NeMo / Canary-1B 为例：

- Canary-1B 的 ASR、AST、PnC 都走 `nemo.collections.asr.models.EncDecMultiTaskModel`；
- ASR / AST / PnC 统一使用 NeMo 的 ASR extra；
- `requirements_asr.txt` 不包含 `requirements_lightning.txt`；
- `requirements_lightning.txt` 只解决 `lightning.pytorch`、Hydra、OmegaConf 等 NeMo core/lightning 依赖；
- `requirements_asr.txt` 解决 `lhotse`、`librosa`、`soundfile`、`jiwer`、`sacrebleu` 等 ASR/AST/PnC 依赖；

#### 6.2 上库推理指导文档编写规范

面向 ModelZoo 上库的推理指导文档应与模型目录中的可执行入口、数据准备方式和性能结果保持一致。生成或修改 `README_INFERENCE.md` / 上库 README 时遵守以下通用规则：

**标题和章节**

- 标题使用“`<ModelName> 推理指导`”，不要混入“模型-推理指导”等不一致格式。
- Markdown 层级必须连续且清晰：文档标题用一级 `#`；主要章节用二级 `##`；主要章节内的子项用三级 `###`。不要在中途把“模型推理性能”“公网地址说明”等主要章节写成一级标题。
- 推荐章节顺序：
  1. 概述
  2. 输入输出数据
  3. 推理环境准备
  4. 文件目录
  5. 快速上手（获取源码、准备权重、准备数据集、模型推理）
  6. 模型推理性能
  7. 公网地址说明

**概述和输入输出**

- 概述只写模型来源、核心架构、支持任务/语言/模态，以及“本文档介绍该模型基于昇腾 NPU 推理指导”。不要把性能测试、精度测试流程等细节堆到概述中。
- 输入输出数据以推理为主：输入写模型推理可接受的数据形式，输出写模型推理结果。若文档包含评测流程，只额外简短说明“评测使用 JSONL manifest”，不要在输入输出章节展开 manifest 字段、metric 文件或评测产物细节。

**推理环境准备**

- 该章节只保留版本配套表和必要说明，不写 pip 安装命令，不解释依赖安装细节。
- 表格按“配套 / 版本”两列写清楚固件与驱动、CANN、Python、PyTorch / torch_npu、torchaudio、核心 I/O 依赖（如 soundfile）等上库环境版本。
- 硬件写具体产品形态，例如 `Atlas 800T A2, Atlas 800I A2`，不要只写笼统的 `昇腾 910 / 910B`。
- 如果固件与驱动版本依赖 CANN，写一句“Atlas 800I A2 推理卡请以 CANN 版本选择实际固件与驱动版本”之类的说明即可。
- pip 安装命令放在“快速上手 / 获取源码”下，直接给命令；不要在环境章节引用 `requirements.txt`。`requirements.txt` 容易包含历史完整环境，可能引入与上库版本不一致的依赖；上库文档应优先通过上游仓库指定 commit / extra 安装必要依赖。

**源码和依赖**

- “参考实现”中的“通过 Git 获取对应代码的方法如下”保留通用模板写法：
  ```bash
  git clone {repository_url}
  cd {repository_name}
  git checkout {branch/tag}
  git reset --hard {commit_id}
  cd {code_path}
  ```
- “快速上手 / 获取源码”中给出当前模型可直接执行的具体命令。
- 如果运行依赖可通过 pip 从上游仓指定 commit 安装，优先写 pip 安装命令，不要求用户手动 clone 上游源码；离线场景可另行准备源码包或 wheel，但不要把离线说明混入默认路径。
- 安装命令直接给出，不写大段解释；版本号应与推理环境准备表一致。

**文件目录**

- 单独设置“文件目录”章节，目录树必须与文档中的命令入口一致。
- 不要列出与上库推理无关的目录，例如 patch 说明目录、历史日志、本地虚拟环境、upstream clone、缓存目录等。
- 如果文档命令使用根目录脚本名（如 `python eval_xxx.py`），仓库中必须提供对应入口脚本；否则文档必须写真实路径（如 `python scripts/eval_xxx.py`）。文档和脚本入口必须一致，不能只在目录树里省略 `scripts/`，但命令里仍使用 `scripts/`。
- 新增根目录 wrapper 可以用于统一上库入口，但 wrapper 只能转发到真实脚本，不应复制大量逻辑。

**权重和测试数据**

- “准备权重”优先给出原始权重 URL 和直接下载命令（如 `wget -O ... <url>`），不要默认依赖项目内部 `download_weights.sh` 包装脚本。
- 上库推理文档中不需要写镜像 URL、脚本环境变量分支、SHA256 校验等适配过程细节，除非上库模板明确要求。
- “准备数据集”中的 smoke / demo 输入优先使用公网可下载的通用测试文件，并给出直接下载命令；不要默认依赖 `download_test_data.sh` 生成或下载。
- 如果模型评测需要公开数据集 manifest，数据准备命令必须写正式准备方式，并列出生成的 manifest 路径。

**数据准备和评测组织**

- ASR、AST、TTS、SE、diarization 等不同任务的数据准备和执行命令应分开写；不要把多任务合并成一个含混的“全部评测”命令作为唯一说明。
- 对 ASR/AST 等多任务模型，分别写：
  - 准备 ASR 评测数据及生成的 ASR manifest 路径；
  - 准备 AST 评测数据及生成的 AST manifest 路径；
  - 执行 ASR 评测命令；
  - 执行 AST 评测命令。
- manifest 路径要完整列出，便于用户复制到 `--manifest` 参数。

**命令风格**

- 文档命令中不写 `ASCEND_RT_VISIBLE_DEVICES=0` / `CUDA_VISIBLE_DEVICES=0` 前缀；设备卡选择属于运行环境设置，不混入上库基础命令。脚本应通过 `--device npu/cpu/cuda` 表达设备类型。
- 命令应从文档声明的模型目录执行，路径相对该目录保持一致。
- 不要使用不存在的任务名或脚本参数；文档命令必须能与当前脚本 `--help` 对齐。

**模型推理性能**

- 性能章节保持简洁，避免重复前文已经说明的性能模式、warmup、dtype、batch、decode 参数等长段解释。
- 推荐只保留必要的性能表和精度表，列出 Model、Card、数据集、Batch Size、Beam Size、RTF/RTFx、WER 等关键字段。
- 公开参考值可简短说明一句，不要把公开榜单环境、限制、对比注意事项在性能章节反复展开；详细对齐分析可放到 `ACCEPTANCE_PLAN.md` 或 `NPU_VALIDATION.md`。
- 一步到位推荐命令：

```bash
cd /path/to/NeMo
python -m pip install -e ".[asr]"
```

如果不能使用 editable extra，则手工安装至少应覆盖：

```bash
python -m pip install -r requirements/requirements.txt
python -m pip install -r requirements/requirements_common.txt
python -m pip install -r requirements/requirements_lightning.txt
python -m pip install -r requirements/requirements_asr.txt
```

NPU 环境还必须额外验证 `torch_npu`：

```bash
python - <<'PY'
import torch
import torch_npu
print("torch:", torch.__version__)
print("torch_npu ok")
print(torch.randn(1).to("npu").device)
PY
```

如已安装匹配 CANN 的 `torch` / `torch-npu`，安装模型依赖时应避免 pip 自动升级或替换 PyTorch。必要时可使用 `--no-deps` 安装源码包，再手工安装除 torch 外的依赖。

#### 6.2 依赖文件含义记录模板

模型适配文档中建议加入如下表格：

| 文件/extra | 功能域 | 目标模型是否需要 | 说明 |
|---|---|---:|---|
| `requirements.txt` | 基础依赖 | 是 | 框架基础包，如 `torch`、`numpy`、`huggingface_hub` 等 |
| `requirements_common.txt` | 通用数据/文本依赖 | 视模型而定 | 如 `datasets`、`sentencepiece`、`pandas` |
| `requirements_lightning.txt` | Lightning/Core 依赖 | NeMo 类模型通常需要 | 如 `lightning`、`hydra-core`、`omegaconf` |
| `requirements_asr.txt` | ASR/AST/PnC 依赖 | 语音识别/翻译类需要 | 如 `lhotse`、`librosa`、`soundfile`、`jiwer`、`sacrebleu` |
| `requirements_audio.txt` | 通用音频处理/评估 | 视模型而定 | 不一定等同于完整 ASR 依赖 |
| `requirements_tts.txt` | TTS | TTS 模型需要 | ASR 模型通常不需要 |
| `requirements_test.txt` | 测试开发 | 非推理必需 | 单元测试、格式化、CI |
| `requirements_docs.txt` | 文档构建 | 非推理必需 | Sphinx 等 |
| `requirements_cu*.txt` | CUDA 附加依赖 | NPU 通常不需要 | NPU 环境不要误装 CUDA 专用 extra |

#### 6.3 依赖验收必须包含导入测试

依赖安装完成后，必须在 `NPU_VALIDATION.md` 中给出最小导入测试。不要只写“安装完成”。

以 Canary-1B 为例：

```bash
python - <<'PY'
import torch
import torch_npu
import lightning.pytorch
import lhotse
import librosa
import soundfile
from nemo.collections.asr.models import EncDecMultiTaskModel

print("torch:", torch.__version__)
print("NPU:", torch.randn(1).to("npu").device)
print("Canary ASR/AST/PnC deps ok")
PY
```

该导入测试至少应覆盖：

- 上游模型入口类；
- 任务域关键依赖；
- 官方评测/后处理依赖（如 normalizer、metric、tokenizer）；
- NPU 后端注册；
- 一个最小 NPU tensor 迁移。

导入测试不得用宽泛 `try/except` 吞错；缺依赖、缺字段或版本不匹配必须在验证记录中体现为失败并说明修复方式。

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
- 大文件不要直接提交，必要时使用 Git LFS 或下载脚本；
- 下载脚本必须有可验证路径：能真实下载，或提供 `CHECK_ONLY=1` / `--check-only` 之类轻量检查模式。

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

#### 7.1 权重下载 URL 必须验证

不能只把官方页面写进文档。提交前必须至少做以下一种验证，并把命令和结果写入 `NPU_VALIDATION.md`：

1. **小权重/可接受大小**：实际运行下载脚本，确认目标文件存在、大小合理，最好记录 SHA256。
2. **大权重**：不强制完整下载，但必须检查仓库 metadata 和必需文件 URL：
   - Hugging Face / Gitee HF：调用 `HfApi.model_info()` 或 API，确认 repo 存在、commit sha、`siblings` 包含必需文件；对关键文件执行 HEAD，记录 HTTP status、`X-Linked-Size` / `Content-Length`。
   - ModelScope：检查模型页面/API 可访问，若使用 SDK/CLI，至少 dry-run 或列文件。
   - GitHub Release / 普通 URL：对最终下载 URL 执行 `curl -I -L --fail`，记录 status 和 size。
   - 网盘/OneDrive/Google Drive：必须验证链接是否能在当前环境非浏览器访问；如果不能稳定直连，下载脚本不得假装可自动下载，必须说明需要浏览器下载或用户提供稳定直链。
3. **需要鉴权/授权**：明确写出登录命令、权限要求，以及当前环境是否因鉴权未完成而阻塞。

推荐在下载脚本中提供轻量检查模式，例如：

```bash
MODEL_CHECK_ONLY=1 ./<model_dir>/scripts/download_weights.sh <target_dir>
```

检查模式应验证：repo/model id、必需文件名、关键文件 URL、文件大小；不能下载多 GiB 大文件。

若 URL 检查失败，必须修正下载源或在文档中记录真实失败原因和可执行替代方案，不能简单写“从官网下载”。

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

#### 8.1 测试数据脚本必须验证

提交前必须实际运行测试数据下载/生成脚本，并验证文件可读：

- 音频：用 `wave` / `soundfile` / `torchaudio.info` 检查声道数、采样率、帧数/时长；
- 图片：用 PIL/OpenCV 检查尺寸和通道；
- 文本/JSON/manifest：检查行数、字段名、路径是否存在；
- dummy 输入：必须明确记录“只验证链路，不验证准确率”。

验证命令和输出必须写入 `NPU_VALIDATION.md`。

#### 8.2 评测数据准备必须与评测脚本解耦

Canary-1B/FLEURS 适配暴露出几个常见坑：`--task all` 会先触发不相关数据集下载；`datasets.load_dataset(..., split="test")` 也可能在构建缓存时下载同 config 的 train/validation 文件；streaming/parquet/range request 会产生大量底层 HTTP 日志；只看命令行参数而不检查实际下载文件，容易误判是否真的只用了 test split。后续模型必须按以下规则处理 L1/L2/L3 数据。

1. **准备数据和评测分开**
   - 数据准备脚本只做下载、抽样、转码、manifest/metadata 生成，不加载模型、不计算指标。
   - 评测脚本只读取已准备的 manifest/索引，复用模型已有推理入口或官方 eval 机制，避免重复下载和重新抽样。
   - CPU/CUDA/NPU 对比必须使用同一份 manifest 和同一批本地文件。

2. **显式记录 split/config/limit**
   - 所有数据准备输出旁边必须生成 metadata，例如 `*.meta.json`，记录 dataset id、config、split、subset limit、抽样 seed、样本数、总时长、下载日期。
   - 文档命令必须显式写 `--split test` / `--config xxx` / `--limit N`，不要只依赖默认值。
   - 输出日志应打印实际加载的数据文件或 URL，例如 `.../test-00000-of-00001.parquet`。

3. **验证“只下载需要的 split”**
   - 使用 HF `datasets` 前，先用 `HfApi.list_repo_files()` 或等价方式检查数据仓库文件布局。
   - 对 parquet/webdataset/tar 格式，优先直接指定 `data_files` 到目标 split 文件；不要默认调用可能准备全量 builder cache 的 `load_dataset(dataset, config, split=...)`。
   - 提交前用 `--limit 1` 或 mock/fake `load_dataset` 做轻量测试，断言 data_files 不包含 `train`、`validation`，并记录测试结果。
   - 如果数据格式决定了即使抽 50 条也必须下载完整 shard/tar/parquet，必须在文档中明确说明预计下载大小和原因。

4. **降低下载日志噪声，保留关键可审计日志**
   - 建议文档给出：`HF_HUB_VERBOSITY=error`、`DATASETS_VERBOSITY=error`、`HF_HUB_DISABLE_PROGRESS_BARS=1`。
   - 不能把底层带签名 URL 的长日志作为正常输出；如出现反复同一 range request，要提示可能是网络重试。

5. **避免混合任务互相阻塞**
   - `--task all` 只适合数据源都已验证后使用；文档必须给出单独准备 ASR/AST/分类等子任务数据的命令。
   - 当某个数据源失败时，不应阻塞其他任务数据准备；必要时用多个脚本或多个命令分开执行。

#### 8.3 数据集在线/离线混合准备要求

Canary-1B 的 FLEURS 与 LibriSpeech 数据准备进一步明确了一个通用要求：**评测数据脚本不应只依赖 Hugging Face cache 或用户浏览器下载；必须支持“指定项目目录、在线自动下载、离线复用本地文件”的混合模式**。后续模型的数据准备脚本优先按以下规范设计。

1. **显式本地数据目录参数**
   - 为每个外部数据源提供独立路径参数，例如 `--fleurs_parquet_dir`、`--librispeech_dir`、`--dataset_dir`、`--manifest_dir`。
   - 目录语义必须清楚：脚本下载/解压/复用都发生在该目录下，不依赖 `~/.cache/huggingface`、系统临时目录或用户浏览器默认下载目录。
   - 文档必须列出该目录下的目标结构，例如：

     ```text
     <data_dir>/fleurs_parquet/en_us/test-00000-of-00001.parquet
     <data_dir>/fleurs_parquet/de_de/test-00000-of-00001.parquet
     <data_dir>/librispeech_raw/test-clean.tar.gz
     <data_dir>/librispeech_raw/LibriSpeech/test-clean/
     ```

2. **存在即复用，缺失才下载**
   - 脚本启动时先检查目标文件/解压目录是否已存在；存在则打印 `using existing ...` 并直接使用。
   - 如果压缩包已存在但解压目录不存在，允许自动解压；如果文件和目录都不存在，在线模式才下载。
   - 下载到临时文件，例如 `*.tmp`，完成后原子 rename 到目标路径，避免中断后留下半文件被误复用。

3. **离线模式必须严格禁止联网**
   - 提供统一 `--offline` 或等价参数。
   - `--offline` 下缺文件必须立即报出具体缺失路径，不得 fallback 到 HF hub、OpenSLR、HTTP URL 或其他远端。
   - 离线命令应写入 README / NPU_VALIDATION，例如：

     ```bash
     python prepare_eval_data.py --task ast \
       --data_dir Canary-1B/eval_data \
       --fleurs_parquet_dir Canary-1B/eval_data/fleurs_parquet \
       --offline

     python prepare_eval_data.py --task asr \
       --data_dir Canary-1B/eval_data \
       --librispeech_dir Canary-1B/eval_data/librispeech_raw \
       --offline
     ```

4. **在线模式下载到指定目录，不把 cache 当交付路径**
   - 在线命令必须能把数据下载到项目指定目录，便于打包迁移到 NPU 离线环境。
   - 可以使用 `urllib` / `wget` / `curl` / 官方 SDK 下载，但最终产物必须落到脚本参数指定目录。
   - 如果因某些库版本问题导致远程 URL 可用但 `datasets.load_dataset(data_files=...)` 解析失败，应提供本地文件 fallback：先下载到指定目录，再从本地文件加载。

5. **手动下载与脚本下载必须等价**
   - 文档必须给出命令行手动下载方式，并保证下载到同一目标结构后脚本不会重复下载。
   - 对普通 URL 推荐给出 `curl -L -o ...` 或 `wget -O ...`；对压缩包给出解压目标目录；对 HF 单文件数据给出最终文件名。
   - 手动下载示例：

     ```bash
     mkdir -p Canary-1B/eval_data/fleurs_parquet/en_us
     curl -L -o Canary-1B/eval_data/fleurs_parquet/en_us/test-00000-of-00001.parquet \
       https://huggingface.co/datasets/google/fleurs/resolve/main/parquet-data/en_us/test-00000-of-00001.parquet

     mkdir -p Canary-1B/eval_data/librispeech_raw
     curl -L -o Canary-1B/eval_data/librispeech_raw/test-clean.tar.gz \
       https://www.openslr.org/resources/12/test-clean.tar.gz
     tar -xzf Canary-1B/eval_data/librispeech_raw/test-clean.tar.gz \
       -C Canary-1B/eval_data/librispeech_raw
     ```

6. **记录数据来源与实际本地路径**
   - metadata 必须记录远端 URL / repo id、config、split、目标本地文件、是否复用已有文件、是否离线、样本数和总时长。
   - `NPU_VALIDATION.md` 必须记录实际使用的本地数据目录和一次可读性检查结果。

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
- `torch_npu` 必须条件导入，避免 CPU-only 环境因 NPU 后端缺失而无法运行 CPU 路径；
- 除 `torch_npu` 这类设备后端注册模块外，模型入口类和推理必需依赖应前置 import；缺依赖应及时报错，不为了 `--help` 延迟暴露依赖问题；
- 对官方预期的模型输出字段、decode 配置字段、版本字段直接访问；字段缺失表示环境/版本不匹配，应立即失败，不添加静默兼容兜底；
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
python3 <model_dir>/infer.py --help
```

要求：`--help` 不应因为缺少 `torch_npu`、权重文件或非必要推理依赖而失败。若上游包导入不可避免，必须在 `NPU_VALIDATION.md` 记录失败原因并说明如何安装最小依赖。

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
# 权重：小权重实际下载；大权重至少 check-only
<download_weights_command>
# 或
MODEL_CHECK_ONLY=1 <download_weights_command>

# 测试数据：必须实际下载/生成
<download_test_data_command>

# 检查测试数据可读性
<inspect_test_data_command>
```

必须记录：

- 权重路径；
- 测试数据路径；
- 是否为官方样例；
- 是否为 dummy 输入；
- 权重 URL 检查结果：repo commit、必需文件列表、HTTP status、文件大小或失败原因；
- 测试数据可读性检查结果：采样率/尺寸/字段等。

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

### Step 12.4：验收主线必须对齐原始模型结果

当前是模型 NPU 适配迁移，不是重新定义模型能力。每个模型的验收计划和验收报告都必须把下面三件事放在最前面：

1. **原始模型测试集是什么**：记录官方/论文/模型卡/README 使用的数据集、config、split、样本数或时长、manifest 生成方式、测试数据下载来源。
2. **原始模型指标是多少**：记录官方公开指标、metric、normalizer/后处理、decode/推理参数、checkpoint/版本、硬件和来源链接。
3. **NPU 如何对齐原始结果**：同 checkpoint、同测试集或同一固定 manifest、同官方或等价评测脚本、同推理参数下，比较 CPU/CUDA 原始路径与 NPU 结果；NPU 通过线应优先定义为“相对原始 CPU/CUDA 不退化”以及“在可复现条件下接近官方公开指标”。

如果官方没有发布正式测试集或指标，必须在 `ACCEPTANCE_PLAN.md` 中明确写“官方未发布测试集/指标”，并说明当前只能使用官方示例、公开替代集或内部固定集做迁移对齐；不得编造官方指标，也不得把内部集、人工抽检、第三方榜单或 smoke test 说成原始模型官方结果。

性能、稳定性、长稳压测、内部业务集和人工 MOS/CMOS 可以保留为补充验收，但不能替代“原始测试集 + 原始指标 + NPU 对齐原始结果”这条主线。dummy / 随机输入 / 1 条样例只能作为 L0 链路验证，不得作为精度、质量或性能验收结论。

### Step 12.5：完整验收方案（必须补充）

每个模型必须新增 `ACCEPTANCE_PLAN.md`。该文件不是当前环境验证日志，而是正式交付/上线验收设计，必须参考原始模型的公开功能、性能和精度。不能只写 “smoke test 通过”。

`ACCEPTANCE_PLAN.md` 至少包含：

- **验收目标与版本边界**：明确当前适配的是哪个模型/权重/变体，排除哪些同系列变体；
- **原始模型能力**：列出原始模型支持的任务、语言、输入输出、batch、解码参数或其他关键功能；
- **原始测试集与官方/公开精度评测数据**：优先记录原始模型官方使用的测试集、split、样本规模、manifest/数据准备方式，以及模型卡、论文、README 或官方 benchmark 的关键精度指标，例如 WER/CER/BLEU/mAP/Accuracy；必须记录 normalizer/后处理、decode 参数、checkpoint/版本和来源链接；如果官方未发布，必须明确写“官方未发布”，不得编造；
- **官方/公开性能评测数据**：记录模型卡、论文、README 或官方 benchmark 的速度/资源指标，例如 latency、throughput、RTF/RTFx、tokens/s、最大 batch、显存/内存、测试硬件、驱动/框架版本、batch 策略和来源链接；如果官方没有发布数值，必须明确写“官方未发布硬件性能数值”，并记录可替代参考（例如官方示例 batch、Leaderboard RTFx、论文训练/推理硬件）和不可直接对齐的原因；
- **数据集选择**：按数据集大小、获取难度、授权/登录要求、验证难度和覆盖能力进行分级；
- **分层验收**：
  - L0 smoke：极小样本，只验证链路；
  - L1 功能回归：小样本覆盖所有关键功能开关；
  - L2 推荐正式验收：可获取公开数据或内部固定集，计算主要精度和性能指标；
  - L3 完整复现：尽量对齐原始公开 benchmark 全量数据和官方评测配置；
- **功能矩阵**：覆盖所有核心任务、语言/模态、batch、长输入、异常输入；
- **精度/质量验收**：优先围绕原始测试集和原始指标设计；指标、normalizer/后处理、CPU/CUDA 原始路径 vs NPU 对齐阈值、对官方/公开精度指标的允许差异必须写清楚；
- **性能验收**：加载时间、延迟、吞吐、RTF/RTFx、最大 batch、峰值 HBM/RSS、稳定性，并说明与官方/公开性能指标是否可比；
- **最低正式验收清单**：资源受限时也必须执行的最小集合；
- **报告模板**：环境、功能、精度、性能、稳定性、结论。

`NPU_VALIDATION.md` 中必须说明现有 smoke test 的局限，并引用 `ACCEPTANCE_PLAN.md` 作为后续完整验收入口。`README.md` 的文件说明中也必须列出 `ACCEPTANCE_PLAN.md`。

示例判定原则：

- NPU 适配本身优先要求同 checkpoint、同数据、同脚本下相对 CPU/CUDA 不退化；
- 只有使用原始公开数据全量、官方或等价评测脚本、匹配解码/后处理配置时，才可宣称复现原始公开指标；
- 官方精度指标和官方/公开性能指标必须分开记录；如果性能指标来自第三方 Leaderboard 或与官方 model card 测试硬件不同，必须标明“仅作参考，不作为 NPU 通过线”；
- dummy / 随机输入只能作为 L0 链路验证，不得作为精度或性能验收结论。

---

### Step 13：文档必须包含

每个模型至少生成，并同步维护根目录 `NPU_ADAPTATION_ANALYSIS.md` 的“参考原始仓库与适配版本边界”章节：

```text
ANALYSIS.md
NPU_ADAPTATION.md
NPU_VALIDATION.md
ACCEPTANCE_PLAN.md
README.md
NPU_ADAPTATION_ANALYSIS.md（参考原始仓库与适配版本边界章节）
```

#### ANALYSIS.md 必须包含

- upstream repo；
- upstream commit；
- 是否匹配远端最新版本；
- 当前适配版本边界：源码、权重、辅助模型、明确排除的同系列变体；
- 当前目录原有文件分析；
- 扫描到的设备相关节点；
- 修改了哪些上游源码节点；
- 是否生成 patch；
- 已知风险和限制；
- 上游更新时如何处理。

#### NPU_ADAPTATION.md 必须包含

- 环境搭建；
- 适配版本边界；
- 权重下载；
- 测试数据下载；
- patch 如何应用；
- `infer.py` 参数说明；
- CPU 推理命令；
- NPU 推理命令；
- 常见问题。

#### NPU_VALIDATION.md 必须包含

- upstream clone / commit 验证；
- 权重 repo / 文件 / SHA256 或 metadata 检查结果；
- `git apply --check` 结果，或说明无 patch；
- `py_compile` 结果；
- 当前环境 CPU 验证命令和结果；
- NPU 验证命令和结果，若无 NPU 则说明未执行原因；
- 权重路径；
- 测试数据路径；
- 输出摘要；
- 已知限制；
- 已说明现有 smoke test 的局限，并引用 `ACCEPTANCE_PLAN.md`。

#### ACCEPTANCE_PLAN.md 必须包含

- 原始模型功能/性能/精度参考；
- 数据集大小、获取难度、验证难度分析；
- L0/L1/L2/L3 分层验收；
- 功能矩阵、精度指标、性能指标、稳定性场景；
- 最低正式验收清单和报告模板。

#### README.md 必须包含

- 模型简介；
- 当前适配的精确版本及非目标变体说明；
- 硬件/软件约束；
- 环境安装；
- 权重下载；
- 测试数据下载；
- CPU 验证；
- NPU 推理；
- 文件说明；
- `ACCEPTANCE_PLAN.md` 链接/说明。

---

### Step 14：最终验证清单

提交前至少执行：

```bash
# 1. 文件检查
find <model_dir> -maxdepth 3 -type f | sort

# 1.5 版本边界检查
git ls-remote --symref <upstream_repo> HEAD
# 如有权重文件，记录校验值
sha256sum <weight_file>

# 2. patch 检查；如果没有 patch，文档必须说明
for p in <model_dir>/patches/*.patch; do
  git -C <model_dir>/upstream apply --check "../patches/$(basename "$p")"
done

# 3. Python 语法检查
python3 -m py_compile <model_dir>/infer.py

# 4. infer.py help 检查
python <model_dir>/infer.py --help

# 5. 权重下载 URL / metadata 检查；小权重应实际下载，大权重可 check-only
MODEL_CHECK_ONLY=1 <model_dir>/scripts/download_weights.sh <weights_dir>
# 或实际下载：<model_dir>/scripts/download_weights.sh <weights_dir>

# 6. 测试数据脚本和可读性检查
<model_dir>/scripts/download_test_data.sh <test_data_dir>
<inspect_test_data_command>

# 7. 当前环境 CPU 验证
python <model_dir>/infer.py --device cpu <model_args> <input_args>

# 8. NPU 验证，有 NPU 环境时执行
ASCEND_RT_VISIBLE_DEVICES=0 python <model_dir>/infer.py --device npu <model_args> <input_args>

# 9. 完整验收方案检查
test -f <model_dir>/ACCEPTANCE_PLAN.md
grep -E "L0|L1|L2|L3|精度|性能|数据集" <model_dir>/ACCEPTANCE_PLAN.md
```

如果某一步无法执行，不能删除该步骤，必须在 `NPU_VALIDATION.md` 中记录：

- 未执行命令；
- 未执行原因；
- 已执行的替代轻量验证，例如 URL HEAD、metadata、`--help`、测试数据可读性；
- 需要用户提供什么，例如授权 token、浏览器下载文件、NPU 机器；
- 后续如何补验。

---

## 版本边界核对清单（提交前必填）

提交前逐项确认：

- [ ] `NPU_ADAPTATION_ANALYSIS.md` 的“参考原始仓库与适配版本边界”章节已记录该模型源码 repo、默认分支、HEAD commit；
- [ ] 已记录模型权重来源、具体文件/目录、repo HEAD 或 release/tag；
- [ ] 已记录本地实际验证权重 SHA256；如未下载，已记录 metadata 检查结果和原因；
- [ ] 已记录 tokenizer / codec / vocoder / embedding / segmentation 等辅助模型版本；
- [ ] 已明确排除同系列其他变体；
- [ ] `README.md`、`ANALYSIS.md`、`NPU_ADAPTATION.md`、`NPU_VALIDATION.md`、`ACCEPTANCE_PLAN.md` 中的版本边界一致；
- [ ] `ACCEPTANCE_PLAN.md` 已参考原始模型功能/性能/精度，列出数据集大小、获取难度、验证难度和分层验收标准。
