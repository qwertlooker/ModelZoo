# 模型 NPU 适配标准流程

## 一句话指令

> 先克隆 upstream，确认远端最新 commit，并明确“当前适配的精确版本边界”：源码 repo/分支/commit、模型权重 repo/文件/commit 或校验值、辅助模型版本，以及明确排除同系列其他变体。区分上游源码改动和当前适配脚本。上游已有文件的修改必须生成 patch；新增 `infer.py` 不放进 patch，直接放当前模型目录。`infer.py` 只保留一个，默认 `--device npu`，CPU 验证用 `--device cpu`，不要使用 `auto/use_gpu`，不要写死 `npu:0/cuda:0`，实际设备由环境变量控制。适配/评测脚本必须按项目级“严格失败”原则实现：必需依赖统一前置 import，缺依赖、缺官方预期字段或版本不匹配时直接暴露原始错误，不添加不必要的 `try/except`、`hasattr/getattr`、regex/basic 替代、CPU/远端 fallback 等静默兼容。必须补全环境搭建、权重下载、测试数据下载、CPU 当前环境验证、NPU 验证说明。还必须生成 `ACCEPTANCE_PLAN.md`，先写清“原始测试集是什么、原始指标是多少、NPU 如何对齐原始模型结果”。当前最低验收只保留两层：功能验证和 L2 正式对齐；L2 必须同时包含主要精度/质量与性能指标，并尽量使用原始公开 benchmark 全量数据和官方配置，否则使用公开数据或内部固定集并说明降级口径。L1/L3 和长稳扩展项可选。模型目录默认维护 `README.md`、`NPU_ADAPTATION.md`、`ACCEPTANCE_PLAN.md` 三类主文档；最后验证 `git apply --check`、`py_compile`、下载 URL/脚本可用性、测试数据可用性和可执行入口，不能只补文档不做验证。

---

## 完成定义：先判定交付状态，再决定是否可以结束任务

过去出现过“版本、patch 和三份文档都齐全，但新人无法从头执行”的交付。根因不是
缺少验收概念，而是没有把“说明应该做什么”和“已经提供可执行闭环”分开。此后所有
模型必须使用下面的状态，不得模糊表述：

### 旧流程为什么没有阻止不完整交付

复盘确认旧流程存在以下执行层缺陷，本次规则专门补齐：

1. **有要求但无完成状态**：文档同时出现“必须 CPU 验证”和“无法执行则记录原因”，
   agent 容易把“已记录未执行”误判为整个任务完成。S0-S4 现在将“记录阻塞”和
   “验证通过”分开。
2. **有验收计划但无产物门禁**：旧流程要求写 `ACCEPTANCE_PLAN.md`，却没有明确
   “写出比较公式”不等于“已有 compare/eval 入口”。现在要求可执行闭环。
3. **最终检查偏静态**：原清单重点是文档存在、grep、compile 和 patch apply，
   无法发现服务名、context、布尔 CLI、输出覆盖等运行时矛盾。现在增加 clean-room
   重放和命令级检查。
4. **“建议/尽量”过多**：数据脚本、离线目录、metadata、下载 check-only 等虽然
   分散写到流程中，但缺少“不满足则不得宣称完成”的统一硬门禁。
5. **没有区分三组 baseline**：只写 CPU/CUDA vs NPU，容易漏掉“未修改 upstream”
   与“patch 后同设备”的回归证明。
6. **缺少模型类型专用检查**：通用规则发现不了 ONNX provider 分区回退、vLLM
   served model/context 冲突、agent 外部工具、fine-tuning 方差和 dscore
   `store_true` 参数等问题。
7. **没有独立重放者视角**：实现者依赖本地 clone、cache 和已知上下文，文档中的
   隐含步骤未暴露。现在 clean-room 是提交硬门禁。

| 状态 | 必须具备的证据 | 允许的结论 |
|---|---|---|
| S0 分析完成 | upstream/权重/参考版本、设备节点、官方测试集和指标取证 | “完成适配分析” |
| S1 静态适配完成 | 最小 patch/infer、`git apply --check`、`py_compile`、静态扫描 | “静态门禁通过” |
| S2 功能验证完成 | 干净环境安装、权重/功能输入准备、CPU/CUDA 或 NPU 至少一条真实端到端输出 | “某设备功能链路实测通过” |
| S3 L2 迁移对齐完成 | L2 上同 checkpoint/manifest/脚本/参数的三组精度/质量结果、性能结果及自动比较 | “NPU L2 精度和性能对齐通过” |
| S4 扩展验收通过 | 用户明确要求的 L3、长稳或业务扩展项完成 | “扩展验收通过” |

硬规则：

1. 文档、patch、compile 只能证明 S0/S1，不能写“适配和验证完成”。
2. dummy/单条样例最多证明 S2 功能链路，不证明 S3/S4。
3. 用户要求“完成适配和验证”时，目标默认至少是 S3；如果环境缺 NPU、权重、
   数据许可或外部服务，任务仍是待验收，不能把“已写验收计划”重新定义为完成。
4. 每个 `NPU_ADAPTATION.md` 必须在末尾写当前状态及缺少的下一等级证据。
5. 项目级汇总表必须使用上述状态名，不能只写“已完成/未完成”。

## 可执行闭环：文档不能替代工具

除非官方固定 harness 已提供完全等价的入口，且当前文档固定了其 commit 和完整命令，
每个模型交付至少应形成：

```text
环境/依赖检查
  -> 权重 metadata check 或实际下载
  -> 测试数据准备和 manifest/meta
  -> 未修改 upstream CPU/CUDA 基线
  -> patch 后 CPU/CUDA 回归
  -> NPU 推理
  -> 自动比较数值/指标
  -> 验收报告
```

最低可执行产物按任务选择：

- `prepare_eval_data.py` 或固定官方数据准备命令：只准备数据，生成 manifest 和
  metadata，不加载模型；
- `infer.py` / 官方推理入口：读取固定输入并写独立结果；
- `eval_*.py` / 官方 evaluator：计算官方 metric；
- `compare_*.py`：按稳定 ID 比较 CPU/CUDA/NPU，阈值失败时返回非零退出码；
- 固定功能输入：可提交的小样本、prompt JSONL 或明确下载的一条官方样例；
- 输出 sidecar：命令、版本、权重/manifest SHA、设备/provider、耗时。

如果项目已有公共工具，应优先复用，例如 OpenAI-compatible 服务使用：

```bash
python tools/openai_service_eval.py --help
python tools/compare_openai_service_results.py --help
```

只在 `ACCEPTANCE_PLAN.md` 写“比较 logits/CSV/DER/embedding”但没有脚本、官方命令
或人工步骤的精确定义，视为未交付。

## 三组基线不能省略

存在上游 patch 时，验收至少保留三组独立结果：

1. **原始 baseline**：精确 upstream commit 未应用 patch 的 CPU/CUDA 路径；
2. **回归 baseline**：应用 patch 后的同设备 CPU/CUDA 路径，证明 patch 没有改变
   原始行为；
3. **NPU candidate**：应用 patch 后的 NPU 路径。

三组必须使用同一 checkpoint、manifest、参数和 evaluator，结果写入不同目录。禁止：

- 让 CPU/NPU 都覆盖 `results.csv`；
- 只比较 patch 后 CPU 与 NPU，却未证明 patch 保持原始 CUDA 行为；
- 使用不同 batch/decode/normalizer 后直接比较指标；
- 用 reference README 截图代替当前运行结果。

若 patch 的目的就是新增上游原本不支持的模型架构或后端，未修改 upstream 可能无法
产生数值结果。此时允许把第一组改为“原始路径预期失败证据”，但必须同时满足：

- 给出可直接执行的原始命令、非零退出码和独立日志路径；
- 记录稳定且与能力缺失直接相关的错误，不接受下载失败、OOM 或环境缺包代替；
- patch 后同设备 CPU/CUDA 必须产生数值结果，作为 NPU 的主要迁移基线；
- `ACCEPTANCE_PLAN.md` 明确这是架构新增例外，不能写成原始数值已对齐。

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

必须记录到该模型的 `NPU_ADAPTATION.md`，并在 `README.md` 和 `ACCEPTANCE_PLAN.md` 中按各自用途保留必要的版本边界；如维护根目录 `NPU_ADAPTATION_ANALYSIS.md`，只同步项目级索引信息，避免复制完整内容：

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
README.md
NPU_ADAPTATION.md
ACCEPTANCE_PLAN.md
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

#### 5.1 FlashAttention / SDPA 迁移原则

本原则适用于所有使用 Transformers、PyTorch attention、CUDA/ROCm `flash-attn`、ONNX/OM attention 图改写或 vLLM-Ascend 的模型目录。2026-06-21 已重新核对官方 `Ascend/ModelZoo-PyTorch` 的 `ACL_PyTorch/built-in` master 快照 `6fecdfba771499ecf4cbc3ed975884720e2a8635`；官方上库文档持续采用固定源码 commit、明确固件/驱动/CANN/PyTorch 配套、真实目录和性能/精度表。后续适配前仍需按当前日期重新复查上游文档和目标 CANN / `torch-npu` 版本。

官方仓中常见做法主要有四类：

| 类型 | 代表路径 / 场景 | 迁移做法 |
|---|---|---|
| 禁用 FA，改 `eager` | `embodied_ai/IsaacGR00T/diff.patch`、`audio/Index-TTS-vLLM-v2/README.md` 等 | 移除 `flash_attn` / `flash_attention_2`，NPU 上显式设置 `attn_implementation="eager"`。 |
| 走 PyTorch SDPA | 多个 HF / Transformers 适配 | 使用 `torch.nn.functional.scaled_dot_product_attention` 或 Transformers `attn_implementation="sdpa"`，由 `torch-npu` 适配。 |
| 手动改成 `torch_npu` FA 算子 | `audio/CosyVoice2/800I/modeling_qwen2.py`、`audio/whisper/whisper_torchair/modeling_whisper.py`、`audio/CosyVoice3/diff.patch`、`cv/MuseTalk/rewrite_models.py` 等 | 在模型 attention 实现中显式调用 `torch_npu.npu_prompt_flash_attention`、`torch_npu.npu_incre_flash_attention` 或 `torch_npu.npu_fusion_attention`。 |
| ONNX/OM 图改写 | `cv/SAM*`、`foundation_models/DiT`、`ControlNet`、`blip_vqa` 等 | 将 ONNX 中的 QK / Softmax / V pattern 改写为 `FlashAttentionTik` / `FlashAttentionSoftmaxFp32` 等图算子。 |

项目级默认策略：

- **默认验收路径优先使用 SDPA**：若原模型能接受 PyTorch / Transformers SDPA，NPU 适配优先显式使用 `attn_implementation="sdpa"` 或 `scaled_dot_product_attention`。这是最适合作为迁移对齐验收的默认路径；验收仍必须同 checkpoint、同测试集 / manifest、同评测脚本、同 decode / 推理参数比较 CPU/CUDA 原始路径与 NPU 结果。
- **`eager` 只能作为显式保守路径**：若目标 `torch-npu` 组合不支持当前 SDPA 形状、mask 或 dtype，可显式切到 `eager` 并在文档和验收报告中记录原因、命令、影响和复测结果；不得在代码中静默回退。
- **不要把 CUDA `flash-attn` 当成 NPU 依赖**：不要采用“安装某个昇腾 flash-attn 包，然后继续在 Transformers 中使用 `attn_implementation="flash_attention_2"`”作为通用方案。官方 `ACL_PyTorch/built-in` 目录未体现这种通用替换路径，Transformers 的 `flash_attention_2` 默认仍主要指向 CUDA/ROCm `flash-attn` 生态。NPU 环境安装依赖时应过滤 CUDA/ROCm 专用 `flash-attn`，除非该模型有明确验证过的独立 NPU 实现并已写入版本边界和验证报告。
- **显式 NPU FA 属于性能实验 / 专项优化路径**：如需追求性能，可参考 CosyVoice/Qwen 类改法，prefill 使用 `npu_prompt_flash_attention`，decode 使用 `npu_incre_flash_attention`，或在训练 / 通用注意力路径中使用 `npu_fusion_attention`。这不是一行替换，必须处理 mask 语义、KV cache、GQA/MQA、layout（如 `BSH` / `BNSD`）、causal sparse mode、dtype、连续性、序列长度和实际 `torch-npu` API 约束，并提供 SDPA/eager baseline 与精度、性能对齐报告。
- **ONNX/OM 图改写只适用于图部署链路**：若模型交付形态是 ONNX / OM，可走图 pattern 改写；该路径不等价于 PyTorch / Transformers 脚本中的 `flash_attention_2`。
- **vLLM-Ascend FA3 / `flash_attn_npu` 不是 Transformers 脚本的直接替换项**：它属于 vLLM-Ascend 后端能力，有特定版本、安装、配置和限制。只有当当前模型明确切换到 vLLM-Ascend 推理后端并按其文档启用时，才能作为单独路径验证；不能用来替代普通 Transformers 推理脚本中的 attention 参数。

参考文档：

- 昇腾 FlashAttentionScore 融合算子替换文档：<https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0027.html>。该文档说明 NPU 上 `scaled_dot_product_attention` 已适配；若原模型直接调用 `flash_attn_func` / `flash_attn_varlen_func` 等 CUDA `flash-attn` 接口，其余模式需通过 `torch_npu.npu_fusion_attention` 等接口迁移。
- `torch_npu.npu_prompt_flash_attention` API：<https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000453.html>。
- `torch_npu.npu_incre_flash_attention` API：<https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000451.html>。
- vLLM-Ascend Flash Attention 3 文档：<https://docs.vllm.ai/projects/ascend/en/main/user_guide/feature_guide/flash_attention.html>。

---

### Step 6：环境搭建必须补全

每个模型必须在 `README.md` 和 `NPU_ADAPTATION.md` 中按文档用途说明环境搭建方式。

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

#### 6.0.1 CPU 与 NPU 环境安装命令必须物理分离

下列写法属于错误交付，审计必须失败：

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-npu
```

原因是第一条命令明确安装 CPU-only PyTorch，随后单独增加 `torch-npu` 不能证明该
PyTorch wheel 与目标 CANN/torch-npu ABI 配套。正确要求：

- CPU baseline 使用独立 venv/容器，可安装官方 CPU wheel；
- NPU 环境使用 CANN 配套表指定的 `torch`、`torch-npu`、`torchaudio` wheel 和
  安装源，三者版本必须完整写出；
- 不允许在 NPU 快速上手中只写 `pip install torch torch-npu`；
- 如果 wheel 由基础镜像预装，文档必须写镜像名和 digest，并用导入/NPU tensor
  命令核验，不再重复安装 CPU wheel；
- 模型依赖使用 `--no-deps` 或约束文件避免覆盖已验证的 NPU 框架 wheel，并记录
  实际 `pip freeze`。

#### 6.0.2 ONNX Runtime CANN 环境必须给出可安装包

使用 `CANNExecutionProvider` 时，不能只写“安装配套 CANN EP”。必须固定：

- CANN 版本；
- `onnxruntime-cann` 版本；
- Python 和 CPU 架构支持范围；
- 安装命令或内部 wheel 的完整文件名/SHA256；
- `ort.get_available_providers()` 和实际 session `get_providers()` 检查。

ONNX Runtime 官方 CANN EP 文档给出的公开配套包括
`onnxruntime-cann 1.20.0/1.21.0/1.22.1` 对应 CANN 8.2.0。若项目选择其他版本，
必须提供目标环境实测依据，不得凭包名近似匹配。CPU `onnxruntime` 和
`onnxruntime-cann` 必须放在不同环境，避免相互覆盖。

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

面向 ModelZoo 上库的推理指导文档应与模型目录中的可执行入口、数据准备方式和性能结果保持一致。生成或修改 `README.md` / 上库 README 时遵守以下通用规则：

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
- 公开参考值可简短说明一句，不要把公开榜单环境、限制、对比注意事项在性能章节反复展开；详细对齐分析和结果放到 `ACCEPTANCE_PLAN.md`。
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

依赖安装完成后，必须在 `NPU_ADAPTATION.md` 的验证记录中给出最小导入测试。不要只写“安装完成”。

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

必须在 `NPU_ADAPTATION.md` 记录实际验证使用的权重路径或权重来源。

#### 7.1 权重下载 URL 必须验证

不能只把官方页面写进文档。提交前必须至少做以下一种验证，并把命令和结果写入 `NPU_ADAPTATION.md`：

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

验证命令和输出必须写入 `NPU_ADAPTATION.md`。

#### 8.2 评测数据准备必须与评测脚本解耦

Canary-1B/FLEURS 适配暴露出几个常见坑：`--task all` 会先触发不相关数据集下载；`datasets.load_dataset(..., split="test")` 也可能在构建缓存时下载同 config 的 train/validation 文件；streaming/parquet/range request 会产生大量底层 HTTP 日志；只看命令行参数而不检查实际下载文件，容易误判是否真的只用了 test split。后续模型必须按以下规则处理功能验证和 L2 数据。

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
   - `NPU_ADAPTATION.md` 必须记录实际使用的本地数据目录和一次可读性检查结果。

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

要求：`--help` 不应因为缺少 `torch_npu`、权重文件或非必要推理依赖而失败。若上游包导入不可避免，必须在 `NPU_ADAPTATION.md` 记录失败原因并说明如何安装最小依赖。

#### 11.2 准备 CPU 可运行依赖

优先使用最小依赖，不要盲目安装超大训练环境：

```bash
pip install -r <model_dir>/requirements.txt
# 或按文档安装最小 CPU 依赖
```

如果当前环境无法安装依赖，必须在 `NPU_ADAPTATION.md` 写明阻塞原因，例如：

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

当前最低范围只包含功能验证和 L2 正式对齐。L2 必须同时报告主要精度/质量和性能
指标；稳定性、L1、L3、内部业务集和人工 MOS/CMOS 可以保留为补充。dummy /
随机输入 / 1 条样例只能作为功能链路验证，不得作为 L2 精度、质量或性能结论。

### Step 12.5：完整验收方案（必须补充）

每个模型必须新增 `ACCEPTANCE_PLAN.md`。该文件不是当前环境验证日志，而是正式交付/上线验收设计，必须参考原始模型的公开功能、性能和精度。不能只写 “smoke test 通过”。

`ACCEPTANCE_PLAN.md` 至少包含：

- **验收目标与版本边界**：明确当前适配的是哪个模型/权重/变体，排除哪些同系列变体；
- **原始模型能力**：列出原始模型支持的任务、语言、输入输出、batch、解码参数或其他关键功能；
- **原始测试集与官方/公开精度评测数据**：优先记录原始模型官方使用的测试集、split、样本规模、manifest/数据准备方式，以及模型卡、论文、README 或官方 benchmark 的关键精度指标，例如 WER/CER/BLEU/mAP/Accuracy；必须记录 normalizer/后处理、decode 参数、checkpoint/版本和来源链接；如果官方未发布，必须明确写“官方未发布”，不得编造；
- **官方/公开性能评测数据**：记录 latency、throughput、RTF/RTFx、tokens/s、
  batch、硬件、框架和来源；若硬件或配置不同，标记“仅供参考”，不得直接作为 NPU
  通过线；
- **数据集选择**：按数据集大小、获取难度、授权/登录要求、验证难度和覆盖能力进行分级；
- **功能验证**：固定最小真实样例，覆盖核心入口、主要功能开关和输出结构；
- **L2 正式验收**：使用可获取公开数据或内部固定集，计算原始模型主要精度/质量指标；
- **功能矩阵**：覆盖所有核心任务、语言/模态、batch、长输入、异常输入；
- **精度/质量验收**：优先围绕原始测试集和原始指标设计；指标、normalizer/后处理、CPU/CUDA 原始路径 vs NPU 对齐阈值、对官方/公开精度指标的允许差异必须写清楚；
- **性能验收**：在 L2 同一数据或固定性能 manifest 上记录 latency、throughput、RTF/RTFx、tokens/s、峰值 HBM/RSS 等适用指标；官方有公开硬件结果时按官方配置对齐，否则只做三组相对比较；
- **最低正式验收清单**：资源受限时也必须执行的最小集合；
- **报告模板**：环境、功能、L2 精度/质量、L2 性能和结论。

`NPU_ADAPTATION.md` 中必须说明现有 smoke test 的局限，并引用 `ACCEPTANCE_PLAN.md` 作为完整验收入口。

示例判定原则：

- NPU 适配本身优先要求同 checkpoint、同数据、同脚本下相对 CPU/CUDA 不退化；
- 只有使用原始公开数据全量、官方或等价评测脚本、匹配解码/后处理配置时，才可宣称复现原始公开指标；
- 只有性能数据的硬件、精度、batch、输入输出长度和计时口径一致时，才可与官方
  性能表直接比较；否则报告本次三组相对结果；
- dummy / 随机输入只能作为功能链路验证，不得作为 L2 精度或质量结论。

---

### Step 12.6：按模型类型补齐专用门禁

通用三文档不能覆盖所有运行时风险。至少按下列类型增加证据：

#### ONNX Runtime / provider 模型

- 不能只检查 `get_available_providers()` 包含 CANN；运行结果必须记录实际 session
  `get_providers()`，必要时打开 ORT profiling 或日志确认关键节点没有因不支持而
  分区到 CPU；
- CPU 和 CANN 环境分开，避免 `onnxruntime` 覆盖 CANN 构建；
- 同一 ONNX SHA、输入和预处理逐输出比较；provider 不符时脚本应失败。

#### Transformers / embedding 模型

- 固定 model、tokenizer 和 remote-code 文件 revision/SHA；
- 比较单样本与 padding batch 中同一样本的输出；
- 记录 shape、cosine、最大/平均误差；下游 fine-tuning 指标不得由 embedding
  smoke 替代；
- 训练类官方指标至少记录 split、seed、最佳 epoch 选择和多 seed 方差。

#### vLLM / OpenAI-compatible 服务模型

- `--served-model-name`、请求 `model`、benchmark 的 `LLM_MODEL` 必须一致；
- smoke 的 `--max-model-len` 与官方 benchmark 要求分开；8K 服务不能宣称验证
  256K；
- 固定 prompt JSONL，CUDA/NPU 写独立 JSONL；tool/reasoning/JSON/streaming
  分别检查；
- token agreement 只作为数值定位，必须结合首个分叉、结构化输出和任务正确率；
- 多卡交付必须给出容器设备挂载、镜像 digest、卡数、HCCL/网络和 rank 检查。
- 结构化配置参数必须按目标 vLLM `--help` 的真实语法书写。例如
  `--speculative-config` 是单个 JSON 参数时，不得写不存在的
  `--speculative-config.method` 点号参数。
- 功能验证输入规模必须与仓内文件或数据生成命令一致；仓内只有 4 条 prompt 时，
  不能在验收表中写成其他数量。

#### Agent / 外部工具 benchmark

- 模型服务 smoke 不等于 agent benchmark；
- 固定 agent framework、data archive、agent config、judge、工具服务、运行次数、
  context 和并发；
- 记录 API 版本、网页/搜索动态性、配额、超时和污染策略；
- 服务 model name/context 必须与 agent 脚本配置一致。

#### Diarization / ASR 等带官方 evaluator 的音频模型

- 数据准备必须固定 wav.scp/manifest、reference、UEM、normalizer、collar/overlap；
- 命令行布尔开关以 `--help` 为准；`store_true` 参数不能写成
  `--flag false`；
- evaluator 自身依赖必须安装和导入；
- 输出必须按稳定 utterance/session ID 对齐，不能依赖目录枚举顺序。

### Step 12.7：阈值必须有依据

阈值来源按优先级记录：

1. 官方明确容差；
2. 原始 CPU/CUDA 多次运行的数值波动；
3. patch 前后同设备回归误差；
4. 已知 dtype/backend 的工程容差。

如果只是初始建议值，必须标记“暂定阈值，待 baseline 校准”。随机采样、训练和动态
agent benchmark 应报告多次运行、均值/方差或置信区间；不能把单次差异直接归因于
NPU。生成模型首 token 分叉会导致后续 token 级联不同，不能单独以全序列 token
agreement 判定质量。

---

### Step 13：文档组织与内容

除非用户明确要求，不修改模型原始 `README_old.md`。每个模型默认维护以下三类主文档：

```text
README.md
NPU_ADAPTATION.md
ACCEPTANCE_PLAN.md
```

#### README.md：上库推理指导

- 面向正式仓用户，包含模型概述、版本边界、输入输出、环境配套表、交付目录、权重和数据准备、推理/评测命令、性能与精度表、公网地址；
- 只保留可直接执行的正式使用说明，路径、脚本名和参数必须与交付文件一致；
- 不堆叠本地调试日志、适配过程、失败记录和大段指标对齐分析，相关内容链接到另外两份文档。

#### NPU_ADAPTATION.md：适配实现与验证事实

- 合并原 `ANALYSIS.md` 和 `NPU_VALIDATION.md` 的职责；
- 记录 upstream repo/commit/检查日期、源码和权重版本边界、非目标变体、原目录分析、设备节点、修改范围、patch、依赖和设备适配方式；
- 记录权重/数据路径及校验、`git apply --check`、导入测试、`py_compile`、CPU/NPU 实际验证命令和结果、未执行原因、风险与限制；
- 只记录已实施或已验证事实，不在此重复完整验收方案和官方指标表。
- 文末必须使用 S0-S4 状态名，并列出升级到下一状态仍缺少的证据。

#### ACCEPTANCE_PLAN.md：验收方案与结果

- 记录原始模型功能、测试集、官方/公开精度指标、normalizer/后处理、decode 参数、checkpoint 和来源；
- 记录数据准备和评测方案、固定 manifest、CPU/CUDA/NPU 对齐方式、功能验证、L2、功能矩阵、精度/质量与性能标准、最低验收清单和报告模板；
- 正式验收完成后，在同一文档补充实际 NPU 结果、与原始/CPU/CUDA 结果的差异、结论和经验，不另建专项评测或案例总结文档。
- 必须包含“最低正式验收清单”；清单项要能指向命令或产物，不能只有抽象描述。

#### 文档收敛原则

- 不再默认新增 `ANALYSIS.md`、`NPU_VALIDATION.md`、`EVAL_*.md`、`ADAPTATION_CASE_SUMMARY.md` 等分散文档；存量内容应无损合并到上述三类主文档后再清理。
- 同一环境、命令、指标或结论只在职责所属文档完整维护，其他文档使用简短摘要和链接引用。
- 根目录 `NPU_ADAPTATION_ANALYSIS.md` 如继续维护，只保留项目级盘点、优先级和版本索引，不复制模型级详细内容。

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

# 4.5 所有交付 Python 入口的语法/help
find <model_dir> -maxdepth 2 -type f -name '*.py' -print
python <model_dir>/prepare_eval_data.py --help
python <model_dir>/<eval_or_compare>.py --help

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

# 9. 三类主文档和完整验收方案检查
test -f <model_dir>/README.md
test -f <model_dir>/NPU_ADAPTATION.md
test -f <model_dir>/ACCEPTANCE_PLAN.md
grep -E "功能验证|L2|精度|质量|性能|数据集" <model_dir>/ACCEPTANCE_PLAN.md

# 10. 项目基础审计
python tools/audit_model_delivery.py <model_dir>
```

如果某一步无法执行，不能删除该步骤，必须在 `NPU_ADAPTATION.md` 中记录：

- 未执行命令；
- 未执行原因；
- 已执行的替代轻量验证，例如 URL HEAD、metadata、`--help`、测试数据可读性；
- 需要用户提供什么，例如授权 token、浏览器下载文件、NPU 机器；
- 后续如何补验。

### Step 14.1：clean-room 重放是最终硬门禁

在提交前由未参与实现的人，或至少在新的临时目录/venv/容器中，从
`README.md` 开始执行。不得读取 `.codex-reference`、历史 shell 状态、
用户 home cache 或未记录环境变量。

最小重放记录：

| 阶段 | 必须保存的证据 |
|---|---|
| clone/checkout/patch | commit、patch SHA、`git status` |
| install/import | Python、框架版本、关键 import、NPU tensor/provider |
| weights | revision、文件清单、大小、SHA 或 metadata check |
| data | 下载/复用日志、manifest/meta、样本数、时长/字段 |
| baseline | 原始和 patch 后 CPU/CUDA 命令、独立输出 |
| NPU | 命令、独立输出、实际 device/provider |
| compare | 比较脚本输出、退出码、阈值来源 |
| report | 当前 S0-S4 状态和未完成项 |

clean-room 审查必须特别寻找：

- 文档从哪个目录执行不明确；
- 相对路径只在作者机器成立；
- clone 后未 checkout 固定 commit；
- 下载命令缺 revision；
- CLI 参数和 `--help` 不一致；
- 服务注册名与请求/benchmark model 不一致；
- smoke context/batch 与正式 benchmark 参数矛盾；
- CPU/NPU 覆盖同一输出；
- 文档引用了不存在的脚本；
- 手工下载、API key、许可或外部服务步骤未说明。
- NPU 环境是否错误安装了 CPU-only PyTorch wheel；
- CANN EP 是否只写概念，没有可执行安装命令；
- 功能验证和 L2 声明样本数是否与实际文件或生成 metadata 一致；
- vLLM JSON/嵌套配置是否误写为 CLI 点号参数；
- 外部数据集、evaluator 和 agent 仓库 clone 后是否 checkout 固定 commit；
- 有 patch 的模型是否真实保留原始、patch 后同设备、NPU 三个独立工作目录或
  可重复恢复步骤，而不是先覆盖源码后再口头描述原始基线。

任一最低路径命令未被重放时，只能报告“未验证”，不得假设它可运行。

### Step 14.2：完成审计必须按需求逐项举证

结束任务前建立一张需求—证据表：

| 要求 | 权威证据 | 状态 |
|---|---|---|
| 版本固定 | repo/weight commit 和 SHA | 已证实/缺失 |
| 代码适配 | patch、diff、静态门禁 | 已证实/缺失 |
| 环境可安装 | clean-room 安装和 import 输出 | 已证实/缺失 |
| 数据可准备 | manifest/meta 和可读性输出 | 已证实/缺失 |
| 原始 baseline | 未修改 upstream 结果 | 已证实/缺失 |
| patch 回归 | 同设备前后比较 | 已证实/缺失 |
| NPU 对齐 | NPU 结果和 compare 报告 | 已证实/缺失 |
| 正式指标 | L2 evaluator 输出 | 已证实/缺失 |
| L2 性能 | 三组性能日志和资源记录 | 已证实/缺失 |

没有证据、只有计划、只有文档描述或只有静态检查时，状态必须是“缺失”。只有所有
用户明确要求的交付项均有权威证据，才可以宣称目标完成。

---

## 版本边界核对清单（提交前必填）

提交前逐项确认：

- [ ] `NPU_ADAPTATION.md` 已记录该模型源码 repo、默认分支、HEAD commit 和检查日期；
- [ ] 已记录模型权重来源、具体文件/目录、repo HEAD 或 release/tag；
- [ ] 已记录本地实际验证权重 SHA256；如未下载，已记录 metadata 检查结果和原因；
- [ ] 已记录 tokenizer / codec / vocoder / embedding / segmentation 等辅助模型版本；
- [ ] 已明确排除同系列其他变体；
- [ ] `README.md`、`NPU_ADAPTATION.md`、`ACCEPTANCE_PLAN.md` 中的版本边界一致；
- [ ] `ACCEPTANCE_PLAN.md` 已参考原始模型功能、精度和性能，列出功能验证与 L2 的数据、命令和通过标准。
