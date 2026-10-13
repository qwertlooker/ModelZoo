# MHCflurry NPU 适配事实

## 版本边界与来源

- 检查日期：2026-10-13；本轮检视并修复已有代码与配套文档，未提供 NPU。
- 上游 master HEAD：`20f03d45358dd104df9df0e28e2e0ccea11c9bfb`，代码版本 2.3.14。
  本轮重新执行 ls-remote，HEAD 未变化，继续固定此版本。
  实际从 GitHub clone 并审计 `setup.py`、common、affinity/processing/presentation
  predictor、network、sizing、training、下载 catalogue 和官方评测文档。
- 当前版本是 PyTorch 实现，Python >=3.10、torch >=2.0；旧 2.1.x TensorFlow
  不能作为该版本的框架描述。
- 权重：release 2.3.0，完整 `models_class1_presentation.20260928.tar.bz2`；
  160,356,140 bytes；官方 Release API SHA256：
  `768e360832b3d45dfe9d5e8cde4ffcb90abdb87773ca65f3a86ba7cc20f614db`。
  本次未下载完整权重，未报告本地权重 SHA；本地校验由 prepare_weights.py 强制执行。
- 三个任务使用同一完整官方 bundle。无单独 tokenizer、vocoder 或辅助 checkpoint。
  不做训练，不适配旧 TensorFlow 变体，也不换成未选择 ensemble 或随机网络。
- 实际下载并检查 comparison-v4 报告包（1,248,609 bytes），SHA256
  `c21b0474fd0520151cb0eb4dbcdb6e8758309f7a273ba36972927ddbe1320739`。
  README/PROVENANCE 明确是 aggregate tables，不包含完整原始逐条 cohort。

## 目标仓快照与参考目录

- 重新执行 `git ls-remote https://gitcode.com/Ascend/ModelZoo-PyTorch.git refs/heads/master`：
  `7a1ec10305c65398ae2013abb57b9bc2edc0b8c5`，2026-10-13，
  `MQ-Det模型昇腾适配`。本轮 fetch 到该 HEAD，重新读取最近 built-in 实质变更清单、
  Laya 和 MQ-Det README；同领域主参考仍为 Laya。
- 独立 filter clone 并按路径查看近期 git log；网页 sampler 未解析出条目，
  后续使用实时 Git 历史，不使用旧缓存冒充最新。
- 拟合入路径：`ACL_PyTorch/built-in/nlp/MHCflurry`。目标树 nlp 未发现此目录，
  本次新增，无替换/删除已有模型文件。nlp 是按肽序列输入及当前生物序列模型
  分类提出的候选路径，正式提交时需维护者确认。
- 最新参考目录：`ACL_PyTorch/built-in/nlp/Laya`，最后实质变更
  `e779117e8cfeba296a7105e26adac0741ab04768` / 2026-10-12 / 新增 310P 适配，
  合并双平台 README，并将原 A2 补丁改名为 diff_laya_npu_910b.patch。
  选择理由：当前 nlp 最近的在线 PyTorch 推理实例，包含设备适配 patch、验证入口、
  Atlas 800I A2 和明确环境配套。只参考硬件、框架配套和 patch 交付；不照搬其
  TorchAir、BF16、硬编码路径和容器权限。
  MHCflurry 继续选择其中 Atlas 800I A2 配套，不扩展为 310P 验收；新合入的
  310P 补丁不是本模型算子兼容证明。参考 README 的 torch 2.10.0+cpu 字样仅
  作来源记录，不作为另装 CPU-only wheel 的依据；目标环境仍按 CANN 官方配套
  核实并保留实际安装证据。
- 生物序列补充参考：`ACL_PyTorch/built-in/nlp/ProtBert_for_Pytorch`，路径最近
  commit `6fecdfba771499ecf4cbc3ed975884720e2a8635` / 2026-06-18 是跨仓文档修改，
  不算新近实质适配。当前 clone 深度 70 的历史边界覆盖不了该目录原始迁移，
  最后实质变更 commit/date **未确认**，不用它取代已核实的 Laya 主参考。
- 最新跨领域复核：`cv/MQ-Det`，最后实质变更
  `7a1ec10305c65398ae2013abb57b9bc2edc0b8c5` / 2026-10-13，是 CV/TorchAir
  路径，环境 CANN 8.3.RC1、torch 2.5.1，不替代同领域主参考或当前已选配套。
- 近期跨领域复核：cv/OVRCOAT（2026-10-09）也是 Atlas 800I A2，但 ONNX 环境
  较旧；ocr/PaddleOCR-VL-1.5（2026-10-10）是 vLLM，不适用于本模型小型网络。
- 贡献规范：`Ascend/modelzoo` HEAD
  `5eab9a4921c7f12edb555079836429a8f285cd1f`，重新查询并读取 CONTRIBUTING.md。
  源许可证、版权头、自测试、CLA、built-in `modelzoo_level.txt` 需检查；原规范
  性能 GPU 倍数不能凭 CPU 代替。当前无实测，不生成虚假 OK/POK 状态文件。
  CLA、官方 CodeCheck/SCA/Antipoison 和实际 modelzoo_level 仍是 PR 门禁待核实项。

## 设备节点、路线与最小修改

拟验证目标 Atlas 800I A2 单卡，CANN 9.1.0、Python 3.12.13、torch 2.10.0 /
torch_npu 2.10.0.post4，与 Laya 和官方 TorchNPU 26.1.0 配套表一致。
2026-10-13 再次读取 Ascend/pytorch 官方 COMPATIBILITY.en.md，26.1.0 对应
2.10.0.post4 / torch 2.10.0 / CANN 9.1.0 的条目仍存在，保持版本边界。
参考镜像 tag 为 `quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1`；镜像 digest、
真实驱动固件和 NPU 张量探针待补，不宣称镜像已可拉取或安装通过。

| 组件 | 拟设备 | 处理 |
|---|---|---|
| peptide/allele 编码 | CPU | 原始上游编码，输入转到模型设备 |
| affinity 全 ensemble | NPU | common 显式 npu 后端，固定 batch 128 |
| processing 有/无 flank 全 ensemble | NPU | 同一后端；内部异步同步补齐 |
| ensemble 聚合、IC50、percentile | CPU | 保留官方数值语义 |
| presentation sklearn logistic regression | CPU | 保留官方 predict_proba |

路线为 hybrid（神经网络 torch_npu eager，官方 CPU 预后处理）。源码审计和
Mock 探针支持该路线的接口设计，**尚无真实算子探针**；embedding、einsum、
Conv1d、局部连接、索引和 masked 操作能否全部在 NPU 执行必须补验。
不用 TensorFlow bridge、ONNX 重写或 TorchAir，避免同时迁移框架/权重/算法。

`diff.patch` 只改固定上游的 `mhcflurry/common.py` 和 `mhcflurry/pytorch_sizing.py`：
增加显式 npu 后端及不可用时报错，补内部 NPU synchronize。默认 auto 的
CPU/CUDA/MPS 选择保持不变；适配入口默认 npu；不做全局 monkey patch。
新增 runtime 固定全模型、关闭 ensemble merge 和 CUDA compile，CPU/NPU
使用相同选项，输入和所有激活网络参数/缓冲区核验设备；无自动 CPU 降级。

上游自动 batch 的 VRAM/OOM 启发主要针对 CUDA/MPS，故入口只接受正整数
batch；OOM 原错误直接传播，不用 CPU 估算 NPU batch。性能包含 CPU
编码/后处理和设备传输，不能声称纯网络算力；实际算子 fallback 日志仍待检查。

## 可执行交付与验证

新增数据准备、curl 下载/离线校验、安全解压、显式推理、逐 ID 数值比较、
等 patient AP/AUROC/PPV@N 和报告入口。source_manifest 固定补丁后的完整
Python 源码；run sidecar 记录文件哈希、设备、版本、命令、计时范围。
所有输出目录禁止覆盖。依赖文件覆盖上游 setup 的业务依赖，核心栈由 constraints
保护，全部直接依赖 release 的 PyPI metadata 可读；**未证明依赖安装闭环**。

实际控制机：Windows，Python 3.14.6；未提供远程 NPU；不修改系统包、不创建
服务、不触碰其他模型已有未提交改动。本次测试不调用真实模型、不写假 NPU 日志。

已执行检查：

- 固定上游 patch 正向 `git apply --check`、应用、反向 check/恢复通过。
- Python 语法、CLI、数据 fixture、负向错误传播和 Mock 设备回归已执行。
- `test_data/smoke.csv` 四条可读，人工 flank，仅链路验证，无标签。
- 权重官方 Release API metadata 可读并含 digest；直链 HEAD 首次网络连接超时，
  未将 HTTP 失败报告为成功。下载脚本支持 check-only，错误非零；离线 SHA 校验
  不依赖网络。实际权重读取/解压待具备归档后执行。
- comparison-v4 完整报告包实际下载、哈希及 tar 成员/PROVENANCE 检查通过。
- 20 项 unittest 离线回归通过，含三任务调用/行顺序和身份核验 Mock 集成；
  测试没有生成持久的模拟 NPU 结果。ruff 0.14.0 check 和 format check 通过，
  检查器仅安装在项目排除目录，未改系统包。
- 基础 `python tools/audit_model_delivery.py MHCflurry` PASS；README audit PASS。
  公共结构审计工具的 6 项回归也通过。
  target-readiness 因未达 S3 返回非零，符合本次 S1 状态，不能据基础 PASS 上库。
- 独立候选按精确清单组装，重新从官方 clone 固定上游并应用 patch，用 Python
  `-S`（不加载 site-packages）重放五个 CLI help、权重 dry-run、数据准备、NPU
  参数 dry-run、20 项 Mock 回归、交付 Python 和 upstream/mhcflurry 语法检查、
  patch 反向 check。证据留在 `.codex-reference/mhcflurry-validation-release/commands.json`。
  此路径为内部证据，不进入候选。没有重放安装、真实模型或 NPU。
- 第一轮将 compileall 扩展到整个 upstream 时，Windows 把上游 git symlink
  （mode 120000）检出为目标路径文本，6 个 downloads-generation 历史生成入口
  报 SyntaxError。复核确为非本次推理代码的 symlink 后，检查范围收敛为交付
  Python 与实际安装的 mhcflurry 包；未修改上游链接，也未增加检查屏蔽注释。

## 上库文件清单与排除项

候选文件精确清单：

```text
README.md
LICENSE
diff.patch
source_manifest.json
constraints.txt
requirements.txt
adapter_utils.py
prepare_weights.py
prepare_eval_data.py
infer.py
runtime.py
compare_results.py
eval_metrics.py
test_data/smoke.csv
tests/test_adapter.py
```

排除 NPU_ADAPTATION.md、ACCEPTANCE_PLAN.md、README_old.md、upstream/、
.codex-reference/、.ascend-adaptation/、权重、数据、outputs/results、日志、
venv 和 cache；本次不存在 README_old.md。许可证直接保留上游 Apache-2.0，
新增文件有 Apache license 头，不冒认贡献者是华为员工。modelzoo_level.txt 尚未
进入清单，因为没有可填写的真实运行状态；正式候选仍未就绪。

## 2026-10-11 检视修复

- presentation 按有序 allele 集合合并 sample，避免每行单独调用 affinity；
  以 peptide_num 和 genotype 名校验输出，恢复原始行 ID，保持 allele 同分顺序。
- 推理新增 --weight-archive（默认 README 权重路径）。每次从 SHA 固定的官方
  归档重新生成可信清单，逐文件核验 model-dir；不信任可编辑 sidecar。
  权重/manifest 改动、缺失、额外文件严格失败，校验不计入性能时间。
- presentation L2 数据准备与评测共用 patient 校验，拒绝空白 sample_id、
  缺标签、非二元标签或单类别 patient，防止静默改变等 patient 指标。
- 本轮 23 项离线测试通过，覆盖组批、allele 顺序、乱序还原、权重清单及
  patient/L2 失败路径；ruff check/format、README 和基础结构审计通过。
  独立候选目录的 5 个 CLI help、dry-run 和 23 项 fixture/Mock 回归通过；
  使用控制机现有 Python，并非真实依赖安装或完整 clean-room，仍无模型/NPU 证据。

## 2026-10-12 检视修复与回归

- 指标报告与数值比较共用 provenance 字段，保留并严格核对源码 revision、
  MHCflurry/torch、task、batch、optimization/compile、输入和权重哈希。
  指标入口独立校验 CPU/NPU 网络设备，拒绝模拟 baseline 和缺字段的旧报告。
- presentation 保留官方 best_allele，数值比较逐 ID 精确核验该字段；空值、
  缺失或同分附近的 allele 选择变化均失败，不因数值误差在容差内而放行。
- 重新查询目标 master 与贡献规范 HEAD，读取最新 Laya 310P/A2 参考，
  保持本模型 A2 验收范围及 S1 状态。
- 29 项离线回归全部通过，无跳过；新增覆盖指标配置/设备不一致、旧报告缺字段、
  指标阈值非零退出、best_allele 空值/缺失/不一致、CLI 非零退出及输出顺序。
  指标算术依赖在新增测试中使用 Mock，只证明证据门禁与错误传播，不作为真实指标。
- 独立候选目录完成本地固定上游 clone/checkout、patch check/apply、五个 CLI
  help、权重和推理 dry-run、smoke 数据准备及 29 项 fixture/Mock 回归。
  记录位于工作证据 .codex-reference/mhcflurry-repair-20261012/commands.json，
  不进入候选文件；未执行依赖安装或真实模型加载，不视为完整 clean-room 验收。
- 候选与固定上游 mhcflurry 包语法检查通过。首次递归编译整个 upstream 时，
  Windows checkout 将下载生成目录的上游 symlink 表示为文本，导致六处语法错误；
  随后限定到正式候选 Python 入口、tests 和运行包检查通过，不修改上游训练工具。
  本轮控制机未安装 ruff，未重跑该检查；README/基础结构审计及 diff 检查通过。

## 2026-10-13 检视修复与回归

- 修复 CSV 重复表头造成字段静默覆盖的问题。输入与预测文件共用严格 CSV
  读取，拒绝空表头、重复表头、行列数不一致、损坏引号和空白 ID；输入另拒绝
  纯空白 allele，并在 affinity/presentation 缺 allele 列时直接报错。
- 推理、数值比较和独立指标入口共用任务输出校验：affinity 必须有限且 >0，
  processing/presentation 分数在 [0,1]，presentation_percentile 在 [0,100]，
  best_allele 非空。CPU/NPU 相同的非法值不再被容差比较放行；指标入口不会只
  检查 presentation_score 而忽略其他损坏字段。
- presentation 的 peptide_num 改为只接受整数索引，拒绝将 0.5 等异常索引
  截断为 0 后误配原始行。保持官方整数输出、allele 顺序和 CPU/CUDA 设备算法。
- compare/eval 的 --report 禁止覆盖已有文件，并使用独占创建防止检查与写入
  之间出现覆盖；误指向原 manifest、predictions.csv 或旧报告时非零退出并保留原文件。
- 38 项离线 unittest 通过，无跳过；新增 9 项测试及既有 runtime 测试扩展，覆盖
  上述正/负边界、命令非零退出和证据文件保留。指标算术与设备仍用 Mock，未生成
  持久的模拟 NPU 验收结果。ruff 0.14.0 check/format check 通过；复用项目内已有
  检查器，不安装系统包。仅修改 MHCflurry，保留其他模型的工作区改动。
- 按 15 项候选清单组装新的独立目录，实际从 GitHub clone 并固定上游 checkout，
  patch 正向 check/apply、反向 check、五个 CLI help、权重 dry-run、四条 smoke
  数据准备、NPU 参数 dry-run、38 项 fixture/Mock 及交付/运行包语法检查通过。
  Python -S 禁用 site-packages；证据位于
  .codex-reference/mhcflurry-review-20261013/commands.json 和 candidate_files.json。
  此为轻量重放，未执行安装、真实权重加载和模型推理，非完整 clean-room 验收。
- README 审计、基础交付审计、git diff --check 通过；target-readiness 返回非零，
  唯一阻塞项为当前状态未达 S3/S4，符合实际 S1，不将此门禁记为通过。
- 权重固定直链本轮 HEAD 最终 HTTP 200，Content-Length 160,356,140；未下载
  完整归档。GitHub Release API 本轮重查受到未认证请求限流，保留已固定的历史
  metadata SHA；实际下载仍必须通过归档 SHA 校验，HEAD 不能证明文件内容哈希。

## 未执行与补验

- 真实 CANN 环境安装/import、NPU kernel/placement/profile：无硬件环境。
- CPU 实际官方模型输出：控制机没有 mhcflurry/pandas 全部依赖，
  未安装模型或完整权重；未升级 S2。
- L2 原始逐条官方 cohort、真实 CPU/NPU 对照、正式指标、NPU 性能与 HBM：待补。
- 完整 clean-room install/权重/模型路径：仅轻量 CLI/fixture/Mock 重放，不能视为
  安装通过或上库候选就绪。
- PR、CLA、CodeCheck/SCA/Antipoison、modelzoo_level：本次不创建目标仓 PR。

真实环境具备后按 README 的源/依赖/权重、smoke、L2、compare、metric 命令执行，
并使用 ACCEPTANCE_PLAN.md 的最低正式验收清单核验。

当前状态：**S1 静态适配完成**。升级 S2 缺真实环境安装和同权重端到端输出；
升级 S3 缺 NPU L2 精度/质量、正式性能及阈值校准。上库候选未就绪。
