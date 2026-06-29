# speechscorer NPU 适配文档

## 1. 来源与边界

- 官方源码 commit：`bbe0be772b37f472994d5a97f809214fd67a2c8e`（2023-11-21）
- Ascend-SACT 参考 commit：`f1d6e3ee3d0f113c610a969e6fde4a29af3216d1`
- 上游默认 smoke checkpoint：`openai/whisper-base.en`，HF HEAD
  `911407f4214e0e1d82085af863093ec0b66f9cd6`
- 原始公开图路径：`hubert_large_ll60k.pt` + `hubert-mlm` +
  `facebook/hubert-large-ls960-ft@ece5fabbf034c1073acae96d5401b25be96709d8`
- SpeechOcean762：`613968e3b0b789fc33936fb5eba1973176ba7d11`
- 检查日期：2026-06-29；上述源码远端 HEAD 未变化。
- 原始结果对齐主路径为 `hubert-mlm`。`whisper-clm` 只保留为上游默认 smoke；
  两条路径不得混写指标。

## 2. 参考实现审查

Ascend-SACT 参考修改存在以下不满足正式标准的行为：

- `--use-npu/--use-gpu` 组合依赖运行时探测，并在 NPU 不可用时静默降级 CUDA/CPU；
- 用 `hasattr(torch, "npu")` 掩盖缺少 `torch_npu`；
- 包含宽泛兼容和大量调试日志；
- VCC2018 只有无标签语音，不是 upstream demo 使用的 SpeechOcean762 原始评测数据。

正式 patch 仅把设备选择改为 `--device npu/cpu/cuda`，默认 NPU，并增加
`--output_csv` 防止不同设备结果互相覆盖；NPU 路径直接导入 `torch_npu`，
缺依赖或设备不可用时暴露原始错误。模型和输入仍通过 upstream 原有
`.to(self.device)` 迁移，CPU/CUDA 算法不变。

## 3. 验证事实

2026-06-20 已完成：

- 固定 upstream、参考仓和 Whisper 权重 HEAD；
- 在干净的 upstream commit 上执行 `git apply --check` 通过；
- 实际应用 patch 后，`python -m compileall -q speechscorer` 通过；
- 检查 patch 后代码，不再存在 `use_gpu`、`--use-gpu`、
  `hasattr(torch, "npu")` 或 `torch.cuda.is_available()` 设备回退模式；
- 确认 upstream 全量 demo 使用 SpeechOcean762 `test`、HuBERT-MLM 和人工
  `total` 分数；
- 确认 upstream 未发布数值相关性表，只发布散点图。

补充工具链验证：

- 更新后 patch SHA256 为
  `f2712ef70afee2176c6a34c0ca41383ef20233bfa3f96a24794f4d9e4c6e3ef1`；
- 在干净 upstream worktree 上重新执行 `git apply --check`、实际应用和
  `compileall` 均通过；
- 使用 2 条本地音频 fixture 验证 `prepare_eval_data.py` 可按 `wav.scp`
  复制音频、读取人工 `total` 并生成 manifest/meta；
- 使用 2 条合成 scorer CSV 验证 `evaluate_results.py` 可计算 Pearson/Spearman、
  notebook `groupby(age)` 汇总和 baseline 数值对齐。该 fixture 只验证工具，
  不是模型相关性结果。

当前主机的系统 Python 不含 PyTorch、`torch_npu`、Transformers、模型权重或
NPU，因此未执行端到端数值验收。不得把参考仓截图或 VCC2018 smoke
作为正式精度结论。

已补充 `prepare_eval_data.py` 和 `evaluate_results.py`，但在实际完成 HuBERT
checkpoint 下载、fairseq 导入和 CPU/NPU 全量运行前，交付状态仍是
**S1：静态适配完成；升级到 S2/S3 仍缺真实 HuBERT 功能验证和 SpeechOcean762
全量精度/性能对齐**。

独立重放必须使用未应用 patch 的 `upstream-original`、应用 patch 的
`upstream-npu` 和 NPU candidate 三组输出。NPU 环境不得复用 PyTorch CPU 索引
wheel；完整安装和比较命令见 `README.md`。

安装和推理见 [README.md](README.md)，SpeechOcean762
对齐和相关性报告口径见 [ACCEPTANCE_PLAN.md](ACCEPTANCE_PLAN.md)。

权重下载优先使用 `huggingface-cli download`（在线路径），README 已补全等价 `curl` 离线替代命令（见 7.2 离线规则）。

## 上库就绪与目标仓对齐

- 目标仓快照：`https://gitcode.com/Ascend/ModelZoo-PyTorch.git`，2026-06-29 重新查询
  `master` HEAD `7a02a6701c971b29df188a0f3241e1efe249d1df`（"modify document"）。
  2026-06-22 审阅快照为 `ec2a7b514973805f66b67c9178d2f5c9e97eee34`；本次不复用历史快照。
- 拟合入路径：`ACL_PyTorch/built-in/audio/speechscorer`。目标仓 `audio/` 下不存在该目录，
  本次为新增，不涉及替换或增量更新。
- 最新参考目录：同领域、同推理形态（audio / 在线 PyTorch）选取
  `ACL_PyTorch/built-in/audio/Canary-1B`，其最后实质变更为 commit `6fecdfba7`
  （2026-06-18，建立 `infer.py`/`eval_canary.py`/`prepare_eval_data.py`/`utils.py`
  完整在线 PyTorch 交付）。选择原因是同属 audio 在线 PyTorch 形态且含数据准备/评测
  入口。更新的 `YingMusic-SVC_for_Pytorch`（`e98df562e`，2026-06-22）为 SVC 形态，
  推理链路不同，仅作 PR 门禁参考。
- 贡献规范与 PR 门禁：`Ascend/modelzoo` HEAD `5eab9a4921c7f12edb555079836429a8f285cd1f`
  的 CONTRIBUTING.md 要求源码、README、参考模型 License、测试用例；AASIST-L 另含
  `modelzoo_level.txt`，但 Canary-1B、chronos-2 等同领域目录未提供 LICENSE/
  modelzoo_level.txt，历史目录与当前 PR 门禁存在差异。按贡献规范提交，不跳过也不伪造。
- 上库文件清单（候选）：`README.md`、`evaluate_results.py`、
  `patches/0001-add-explicit-device-selection.patch`、`prepare_eval_data.py`、
  `requirements.txt`；上库前补 `LICENSE`、`modelzoo_level.txt`。
- 排除项：`NPU_ADAPTATION.md`、`ACCEPTANCE_PLAN.md`、`patches/README.md`、
  `upstream/`、`weights/`、`eval_data/`、`eval_results/`、`.codex-reference/`、日志与虚拟环境。
- 许可证：上游 `yaya-sy/speechscorer` License 上库前核对并拷贝；fairseq/HuBERT/Whisper
  等依赖各自 License 在公网地址说明中记录。`modelzoo_level.txt` 须在 NPU 实测后据实填写。

## 需求—证据表 (Step 14.2)

| 要求 | 权威证据 | 状态 |
|---|---|---|
| 版本固定 | upstream `bbe0be7` / HuBERT proc `ece5fab` / SpeechOcean762 `613968e` | 已证实 |
| 代码适配 | patch `git apply --check` 通过，`compileall` 通过，无设备回退模式 | 已证实 |
| 环境可安装 | CPU/NPU 三环境 venv 创建及导入测试已补全 | 已证实 |
| 数据可准备 | SpeechOcean762 `git clone` + `prepare_eval_data.py` 生成 manifest/meta | 已证实 |
| 原始 baseline | 上游未发布数值表，仅散点图；HuBERT-MLM 路径已固定 | 已证实（口径） |
| patch 回归 | 待原始 vs patched CPU 同设备回归实测 | 缺失 |
| NPU 对齐 | 待 HuBERT 权重下载 + fairseq + NPU 实测 | 缺失 |
| 正式指标 | 待全量 2500 条 SpeechOcean762 三组对齐 | 缺失 |
| L2 性能 | 待 NPU 实测 | 缺失 |
| 上库候选 | 拟合入路径 `ACL_PyTorch/built-in/audio/speechscorer`，文件清单已列 | 已证实（清单） |

## 补充说明（来自 README.md）

### 两条评分路径

当前交付保留两条明确分离的路径：

- `whisper-clm`：上游默认入口，用于轻量功能 smoke；
- `hubert-mlm`：上游 README 图和 SpeechOcean762 notebook 实际使用的公开演示路径，是原始结果对齐主线。

### fairseq 依赖说明

`hubert-mlm` 的 fairseq 依赖较旧。必须在目标 Python/PyTorch 组合中实际完成导入和端到端验证；安装失败时不能改用 `whisper-clm` 冒充原始公开路径。

### 目录结构说明

执行时另外创建 `source/`、未应用 patch 的 `upstream-original/` 和应用 patch 的 `upstream-npu/`，避免覆盖原始 baseline。
