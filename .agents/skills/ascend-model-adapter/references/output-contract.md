# ModelZoo 交付契约

完成 Ascend ModelZoo-PyTorch 适配并整理交付目录时，使用本契约作为收尾检查依据。

## 目录结构

优先在最终上库路径 `ACL_PyTorch/built-in/<category>/<model>` 下使用扁平的 ModelZoo 风格目录。脚手架和 patch 从一开始就放在该根目录，不要先生成 `ascend_adapter/diff.patch`、`adapter/README.md` 等临时子目录再搬迁，避免 README 路径来回修改。只提交必要文件；能通过 patch 修改上游代码时，优先 patch 上游代码，避免复制重复脚本。部分 ModelZoo 项目接近 README + patch + requirements 的极简形态；另一些只额外保留少量辅助或修复脚本。

```text
<ModelName>/
├── README.md                         # 必需
├── requirements.txt                  # 必需；只放业务依赖
├── diff.patch 或 <model>_NPU.patch   # 修改上游代码时必需
└── 可选，仅在需要时保留
    ├── export_onnx.py / pth2onnx.py  # ONNX/OM 路线
    ├── convert_om.sh / atc.sh        # ONNX/OM 路线
    ├── infer.py / ascend_infer.py    # 仅当上游缺少推理入口时新增
    ├── validate_acc.py / eval_accuracy.py
    ├── validate_perf.py / benchmark.sh
    ├── prepare_data.py               # 需要数据准备入口时优先单一 Python 脚本
    └── 上述脚本需要的 helper/fix 文件
```

关键原则：

- 如果上游已有推理/评测入口（`inference.py`、`infer.py`、`test.py`、`demo.py`、shell 命令等），优先 patch 这些入口以支持 NPU，不要新增重复脚本。
- 可行时使用单一 `--device npu/cpu` 参数或环境变量，默认值必须是 NPU；CPU 只能作为显式 fallback/baseline。不要默认拆成 `infer_cpu.py` 与 `infer_npu.py`。
- 不要把 agent 内部文件作为上库交付物，例如 `env_check.py`、`docker_run.sh`、`collect_report.py`、`adaptation_config.yaml`。环境检查、Docker 命令和证据收集命令应写入 README。
- 数据准备脚本默认一个主入口 `prepare_data.py`；不要同时提交功能重复的 `prepare_data.sh` 和 Python 脚本。若保留 shell，必须只是必要的系统工具编排，README 只引用一个主入口。
- 对用户提供的 checkpoint/weights，必须记录实际产物路径、期望目录树，以及 config/tokenizer/label-map 等配套关系；不得静默替换成其他 checkpoint。

## README 章节

README 结构优先跟随 `scripts/modelzoo_sampler.py --count 20 --clone` 抽到的同任务近期样本；不要机械套旧 13 章模板。推荐骨架如下，可按样本合并或改名，但信息必须可复现：

1. **标题与概述**：`<ModelName>-推理指导` 或 `<ModelName>(路线)-推理指导`；说明任务、上游链接、固定 commit/revision、checkpoint/权重、许可证、适配路线、验证硬件对外型号和支持范围。模型名、权重名、变体名必须一致，命名层级不同要解释。
2. **推理环境准备 / 环境与版本声明**：固件/驱动、CANN、Python、PyTorch、torch_npu、torchvision/torchaudio、额外 SDK、vLLM/TorchAir/ais_bench/msit 版本；推荐环境必须是成套且已验证的镜像/软件栈，说明镜像内置 `torch/torch_npu/torchvision/torchaudio` 不要重装。
3. **镜像启动 / 创建容器**：使用后台容器 `docker run -itd` + `docker exec` 模式，保留 `<container-name>`、`<宿主机工程目录>` 占位符，挂载 NPU 设备、driver、dcmi、npu-smi 和工程目录；进入容器后 `source set_env.sh` 并做版本/可用性检查。
4. **快速上手 / 操作步骤**：克隆 ModelZoo、进入最终上库目录、按固定 commit 克隆上游、执行 `git apply --check`/应用 patch、安装业务依赖、准备权重/数据。安装依赖必须写清工作目录和来源；有 editable 子包时写明顺序：`pip install -r requirements.txt` → `pip install -e ./<subpkg>` → 顶层 `pip install --no-deps -e .`。
5. **准备权重和数据**：权重清单、来源或用户提供路径、目录树、离线缓存；数据集、评测工具、protocol、reference label/RTTM、最小样例的来源和生成命令。数据准备默认只提供一个主入口 `prepare_data.py`，不要同时维护功能重复的 `.sh`。
6. **模型导出/转换或服务启动**：ONNX/OM 写导出 ONNX、校验、ATC/OM 转换和样例推理；TorchAir 写图编译、缓存、首次编译说明；vLLM-Ascend 写镜像 tag、server、client、显存/并发配置。
7. **模型推理**：NPU 上的精确命令、关键参数/环境变量、默认值、必填项、路径格式、最小输入样例和输出位置/格式；脚本不传 `--device` 时也应默认 NPU。输入输出 tensor 名称、shape、dtype、layout 可放在本节或转换节；OM 固定输入输出时必须列清，但不强制单独成“输入输出数据”章。
8. **精度与性能验证**：本 skill 面向 GPU/上游实现迁移到 NPU；有对比时默认与官方/GPU 精度和性能比较。精度优先用一张表合并“官方/源仓/GPU 参考”和“NPU 结果”，写数据集、metric、命令、差异和结论；官方已有精度指标时，不要求 CPU 精度对比。与官方精度比较必须使用官方相同的完整数据集/split；若官方有多个数据集，可只评测其中一部分并列明未评测项。不要拆成互相重复的“上游官方精度”和“NPU 精度验证”两章。性能写官方/GPU benchmark 口径、NPU 命令、warmup、loop、batch/并发、精度模式、latency/FPS/QPS/RTF、对外硬件型号；默认没有本地 GPU 环境时使用官方性能作参考，没有官方性能时只报告 NPU 性能；不写 CPU 性能对比。纯模型与端到端必须区分。
9. **FAQ/已知问题**：unsupported ops、ATC 长时间编译、依赖冲突、离线下载、CPU fallback 原因、patch 只能执行一次、cache 清理等，从不上库的适配过程记录提炼用户可复现结论。
10. **公网地址说明**：只列 README/命令实际使用或实测相关的源码、权重、数据集、评测工具、protocol、测试样例、论文、issue/release note、关键预处理工具 URL；不要堆砌未验证或未引用的地址。

不推荐单独成章的内容：

- **适配修改说明**：通常信息已由 `diff.patch`、patch 应用命令和 FAQ 体现；除非样本/任务确有必要，否则不要单独列冗余“修改点”章节。
- **输入输出数据**：可合并到推理或转换节；只有 OM/固定 shape 场景需要明显表格时才单列。
- **Pipeline 组件部署**：只在 diarization/OCR/VLM/TTS/机器人等复杂 pipeline 中保留；简单模型不要为了模板完整性增加。

可选补充：

- **交付件清单 / 文件目录**：当项目包含多个 patch、子目录或辅助脚本时列出提交文件及作用；极简 README+patch+requirements 项目可省略。
- **Pipeline 组件表**：列出组件、上游 backend、选定 backend、NPU 可行性和 CPU fallback 原因；概述、FAQ、性能表必须与组件表一致。

## 指标选择

- 精度：尽量使用上游原始指标、官方完整数据集/split、预处理、后处理和阈值。源仓已有 accuracy 数据时，NPU 结果优先对齐同一 checkpoint、官方相同完整数据集/split、随机种子和评测脚本口径，不要求 CPU 精度对比。若官方有多个数据集，可选择部分数据集评测，但对已选择的数据集必须完整评测，并列明未评测项。只有在源仓没有可复现 accuracy/官方指标或原始指标无法复现时，才在相同输入上对比 NPU 与 CPU/upstream baseline，并说明替代原因。
- 性能：优先使用原始项目已有官方/GPU benchmark/performance 脚本和可比性能指标，尽量对齐输入规格、batch/并发、warmup/loop、统计区间、端到端/纯模型定义和单位。默认没有本地 GPU 环境时使用官方发布性能作为参考；没有官方/GPU 性能时只要求 NPU 可复现。不要采集、比较或展示本地 CPU 性能。否则按路线采用常用口径：OM 用 `ais_bench` latency/FPS，vLLM 服务用 QPS/tokens/s/latency，音频默认用 RTF（耗时/音频时长，越低越好）为主，可补充 RTFx=1/RTF（实时倍速，越高越好），pipeline 同时报告纯模型和端到端 latency。
- 始终说明 warmup、loop 次数、batch/并发、输入 shape、精度模式、对外硬件型号，以及是否包含首次编译或 CPU fallback。对外硬件型号使用产品/整机名（如 `Atlas 800I A2`），不要在 README/PR 性能表中写详细芯片型号、芯片步进或内部代号；`SOC_VERSION` 仅作为 ATC 转换参数出现。

## 容器命令模板

近期样本默认使用后台容器 + `docker exec` 进入，README 中保留占位符并要求用户替换：

- `<container-name>`：容器名，例如 `<model>-infer`。
- `<宿主机工程目录>`：当前 ModelZoo 项目目录在宿主机上的绝对路径。

```bash
export IMAGE=<ascend-image-tag>
docker pull ${IMAGE}

docker run -itd -u root --net=host --privileged=true \
  --name <container-name> \
  --shm-size=256g \
  --ipc=host \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v <宿主机工程目录>:<宿主机工程目录> \
  -v /root/.cache:/root/.cache \
  ${IMAGE} bash -i

docker exec -it <container-name> bash
cd <宿主机工程目录>
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
python3 -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available())"
```

根据目标机器调整 `--device=/dev/davinci*` 数量（8 卡可挂载 0-7），以及 `/usr/local/sbin/npu-smi` 或 `/usr/local/bin/npu-smi` 的实际路径；部分样本需要额外挂载 `/usr/local/sbin`、`/usr/local/Ascend/driver` 只读或设置 `PYTORCH_NPU_ALLOC_CONF`，以同任务近期样本和实测环境为准。

## 依赖安装模板

安装前先预检，再一次性补齐业务依赖，避免反复“跑一下缺一个包”。示例：

```bash
# 1. 核心栈来自镜像，不要随意重装。
python3 - <<'PY'
mods = ["torch", "torch_npu", "torchaudio"]  # 按任务追加 transformers/pyannote/... 等
missing = []
for name in mods:
    try:
        mod = __import__(name)
        print(f"{name}: {getattr(mod, '__version__', 'ok')}")
    except Exception as exc:
        missing.append((name, repr(exc)))
if missing:
    raise SystemExit("missing/failed imports: " + str(missing))
PY

# 2. 只安装业务依赖；过滤会覆盖镜像栈或 GPU/CUDA-only 的包。
pip install -r requirements.txt

# 3. 如果有 editable 子包，先安装子包；最后安装顶层包时默认 --no-deps。
pip install -e ./<subpkg>
pip install --no-deps -e .

# 4. smoke test：import/--help/单样例至少做一个。
python3 -c "import <entry_module>; print('import ok')"
python3 <infer_or_eval_entry>.py --help
```

`requirements.txt` patch 应最小化：只删除会阻塞或冲突的条目（例如不适配的 `onnxruntime-gpu`、会覆盖镜像栈的 `torch*` 版本），只新增 smoke test 证明缺失的业务依赖；不影响安装和验证的上游依赖尽量保留。

## 多卡并行与空闲 NPU 检测

多数据集/多 split 评测可在多张空闲 NPU 上并行运行，命令前显式 `export ASCEND_RT_VISIBLE_DEVICES=<id>`，脚本参数仍使用 `--device npu`，不要传 `--device npu:<id>`；日志和输出目录按数据集/可见卡 ID 区分。

优先用 `npu-smi info -t usages` 的键值行判断空闲，不要依赖 `npu-smi info` 表格固定列。可在评测脚本中复用以下 bash 函数（阈值按任务调整，检测失败时要求用户显式指定 device）：

```bash
find_free_npu() {
  local max_mem=${1:-10}      # Memory/HBM Usage Rate(%) <= max_mem
  local max_aicore=${2:-5}    # Aicore Usage Rate(%) <= max_aicore
  local ids=${NPU_IDS:-"0 1 2 3 4 5 6 7"}
  local id out mem aicore chip_arg=()
  if [ -n "${NPU_CHIP_ID:-}" ]; then
    chip_arg=(-c "${NPU_CHIP_ID}")
  fi
  for id in ${ids}; do
    out=$(npu-smi info -t usages -i "${id}" "${chip_arg[@]}" 2>/dev/null || true)
    [ -z "${out}" ] && continue
    mem=$(printf '%s\n' "${out}" | awk -F: '/(Memory|HBM) Usage Rate\(%\)/ {gsub(/[^0-9.]/,"",$2); print $2; exit}')
    aicore=$(printf '%s\n' "${out}" | awk -F: '/Aicore Usage Rate\(%\)/ {gsub(/[^0-9.]/,"",$2); print $2; exit}')
    [ -z "${mem}" ] && continue
    [ -z "${aicore}" ] && aicore=0
    if awk -v m="${mem}" -v a="${aicore}" -v mm="${max_mem}" -v ma="${max_aicore}" \
      'BEGIN { exit !((m <= mm) && (a <= ma)) }'; then
      echo "${id}"
      return 0
    fi
  done
  return 1
}
```

## PR 就绪自检

默认在认为目录可提交 PR 前，确保以下条件成立：

- PR/README 文本没有模板占位符、重复安装段落、debug code、过期注释、不清晰变量名或残留 import。
- 上游源码固定到 commit/revision；README 包含 checkout 和 patch 应用说明。
- README 包含目录结构或交付件清单，列明提交的 README、requirements、patch、导出/推理/评测脚本及其作用；若只有 README+patch+requirements，也要让用户能判断缺省脚本由上游入口承担。
- README 不引用未随仓提交、未列入交付件清单或在公网地址说明中不可追溯的相对文档；必要背景直接写入 README FAQ/说明。
- 尊重用户提供的 checkpoint/weights，并与正确的 config/tokenizer/label map 配套；任何替换都必须显式说明。
- README 中用于推理、精度或性能的测试数据、标准数据集、评测工具、protocol 和 reference label/RTTM 均有可追溯来源或生成命令；不得只写“用户自行准备”。
- 若数据准备脚本支持多个数据集，每个分支至少用最小合成 tar/zip、短音频和最小 reference label/RTTM 做 dry run，确认输出目录、scp 路径和 README 命令一致；GitHub zip/top-level 目录嵌套要实际验证。默认只有一个主入口 `prepare_data.py`，不要让 README 在 `.py` 与 `.sh` 间来回引用。
- 上库目录保持扁平：README 中的 patch、requirements、脚本路径均相对 `ACL_PyTorch/built-in/<category>/<model>` 根目录；不得残留 `ascend_adapter/diff.patch`、`run_all_eval.sh` 等开发期旧路径或未提交脚本名。
- README 引用的评测工具、子模块和脚本路径必须真实存在或提供获取命令；上游 submodule 需写明 `git submodule update --init <name>`，不得残留本地临时路径名。
- 源仓 accuracy 只在 checkpoint、模型组件、官方完整数据集/split、预处理/后处理、评测脚本和阈值一致时直接对齐；若只跑小样本/子集，或替换了嵌入模型、tokenizer、聚类策略、label map 或任一子模型，README 必须声明与源仓完整集指标不可直接比较。
- 当前 checkpoint 对应的模型变体、配置文件、评测参数、预处理命令和后处理/聚类/解码策略已核对；不得只使用默认 config 推断 benchmark 口径。
- 推理后端替换已说明是否同架构同权重；同架构同权重需提供数值等效性证据，换模型/换权重则必须重新跑任务指标。
- CPU/NPU 小样本输出对齐不能写成官方任务 metric；没有 GT/reference 时，用“输出对齐/边界差/cosine diff”等名称，不要写 DER/WER/mAP。
- README 包含对外硬件型号/主机信息（如 Atlas 800I A2），公开表格和 PR 描述不要暴露详细芯片型号；`SOC_VERSION` 仅作为 ATC 转换参数保留为可配置项。
- README 内部口径一致：概述、Pipeline 组件表、FAQ、性能表不能互相矛盾；版本相关 FAQ 要标明触发版本；支持硬件型号只写实际验证的对外型号或明确“待验证”。
- 公开文档中的硬件字段使用 `Atlas 800I A2` 这类对外型号；不得把详细芯片型号、芯片步进、内部代号写入 README 性能表或 PR 描述。
- 推荐镜像、CANN、torch、torch_npu、torchvision/torchaudio、Python 版本彼此配套；不得推荐在不匹配镜像中直接 pip 升级核心框架作为可复现环境。
- 安装命令明确工作目录、requirements 来源、依赖预检和安装顺序；不得在上游目录中无检查地 `pip install -r requirements.txt` 或 `pip install -e .` 导致覆盖镜像内 `torch/torch_npu/torchvision/torchaudio`。有 editable 子包时，顶层安装默认 `pip install --no-deps -e .`。
- 依赖审计覆盖上游仓、子模块、vendor 包的 requirements/setup/pyproject；requirements patch 保持最小化，过滤或 patch 后已做 import/`--help`/单样例 smoke test，避免遗漏业务依赖。
- README 中所有 shell/Python 命令经过静态自检或实际执行；不得存在明显语法错误、错误环境变量、未定义脚本、错误相对路径或不可复现的“假设路径”。
- 推理、评测和 benchmark 脚本的默认设备是 NPU；`--help`、README 主命令和脚本默认值一致。CPU 命令只在 baseline/FAQ/排障场景出现，并显式传 `--device cpu` 或等价参数。
- 脚本 `--help`、脚本尾部提示和 README 后续步骤必须指向真实命令；删除未提交的 `run_eval.sh`、内部工具名、旧目录名等模板残留。
- 精度不能只用截图或输出文件表示；必须包含任务指标命令和结果。
- 性能指标和单位匹配任务，并在 README、脚本、PR 描述中保持一致；性能命令、脚本默认参数和结果表的 batch/warmup/loop/并发/是否包含模型加载必须一致；主表默认只给 NPU 性能，可引用官方/GPU 参考性能，不加入本地 CPU 性能对比列；纯模型和端到端口径不能混用。
- Pipeline CPU fallback 有具体技术原因，不能只写“上游默认 backend”。
- Long-running issues and route-changing decisions are recorded in a private process log; only reusable/user-facing conclusions are reflected in README FAQ.
- 本地 lint/import/help 检查通过；`ACL_PyTorch/ModeList.md` 无冲突标记且统计数与表格行数一致；可用 `scripts/modelzoo_pr_quickcheck.py` 做提交前快检；预期 Antipoison、CodeCheck、SCA 和 PR 流水线可通过。

## PR 描述模板

PR 描述不要临时拼凑或保留占位符。默认使用以下结构，Self-test 表格里的命令、结果和 README 数字必须一致：

```markdown
## Motivation

- 适配 <ModelName> 到 Ascend NPU，覆盖 <路线/对外硬件型号/任务>。

## Modification

- 新增/更新 `ACL_PyTorch/built-in/<category>/<model>/README.md`、`requirements.txt`、`diff.patch` 等。
- 说明核心改动：设备参数化、后端替换、导出/转换、评测/性能脚本、依赖最小化处理。

## Self-test

| 项目 | 环境/命令 | 结果 | 日志/备注 |
|---|---|---|---|
| 环境 | `npu-smi info`; `python3 -c "import torch, torch_npu; ..."` | PASS/TODO | 对外硬件型号、CANN、torch_npu |
| Patch | `git apply --check diff.patch` | PASS/TODO | 固定 commit |
| 依赖 | `pip install -r requirements.txt`; `pip install --no-deps -e .`; import/`--help` | PASS/TODO | 未重装 torch 栈 |
| 转换/编译 | `<export/atc/torchair/vllm 命令>` | PASS/TODO | ONNX/OM/cache 路径 |
| 单样例推理 | `<infer 命令>` | PASS/TODO | 输出路径 |
| 精度 | `<eval 命令>` | PASS/TODO | metric、官方完整数据集/split、delta；无官方指标时再写 CPU baseline |
| 性能 | `<benchmark 命令>` | PASS/TODO | NPU 性能、batch、warmup、loop、单位；如有则记录官方/GPU 参考口径 |

## BC-breaking

- 无 / 有：说明依赖、接口、数据格式或权重路径变化。

## Checklist

- [ ] README 无占位符，公网地址只列实测相关资源。
- [ ] 精度与性能结果和脚本输出一致。
- [ ] CodeCheck/SCA/Antipoison/本地 lint 预期通过。
```

## 验证证据清单

- [ ] 上游 URL 与 commit/revision 已固定。
- [ ] 许可证与再分发限制已检查。
- [ ] 容器镜像和宿主机 driver/CANN 兼容性已说明。
- [ ] 推荐环境的软件栈成套且已实测；未验证环境标记为待验证。
- [ ] 精度证据已记录：优先源仓 accuracy/官方指标；官方指标对比使用相同完整数据集/split，多个官方数据集可只评测部分但已列明范围；无可复现源仓精度时才记录 CPU/upstream baseline 与 NPU 对齐。
- [ ] 性能证据已记录：源仓已有官方/GPU benchmark/performance 数据时，已记录原始口径、NPU 对齐结果或差异说明；没有官方性能时只报告 NPU 性能；未展示 CPU 性能对比。
- [ ] 如适用，ONNX 导出成功且记录 ONNX checker/simplifier 结果。
- [ ] 如适用，ATC 成功且记录 `.om` 产物路径。
- [ ] NPU 单样例推理成功。
- [ ] 精度指标在容差内，或 delta 已合理解释。
- [ ] 性能命令和结果表已记录。
- [ ] README 命令、相对路径、外部链接、交付件清单已自检。
- [ ] CPU-only 限制标记为 `待 NPU 验证`，不得标记为通过。
