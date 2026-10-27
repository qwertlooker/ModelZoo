# ThermoMPNN 验收方案与结果

## 验收目标与版本

本轮合入精确 GELU、中心 tap head 和蛋白合批，当前 S2（CPU 功能验证）；目标 NPU 配套和 S3 保留待验收。
固定官方 commit `370f76ec62bd929f7425e311d8df04a0d094990f`、`thermoMPNN_default.pt` 和 `v_48_020.pt`，FP32、单链、默认全部蛋白合批，支持同长度合批、逐蛋白和上游 head 对照；默认完整结构邻域、LightAttention/subtract_mut 开启、最后两层 ProteinMPNN embedding。权重 SHA、命令和环境见 README。ThermoMPNN-D/I、模型训练、多链和混合精度不纳入本轮验收。

## 原始测试集与官方/公开指标

[官方 README](https://github.com/Kuhlman-Lab/ThermoMPNN/blob/370f76ec62bd929f7425e311d8df04a0d094990f/README.md) 指向 Megascale、FireProt、S669 和 protddg-bench，并提供固定 split 与模型权重。原始模型主要预测突变 ΔΔG；官方 benchmarking 计算 pooled Pearson、Spearman、MSE、RMSE、R²，不是序列生成或结构 RMSD。

| 固定源码里的数据 | 实际 CSV 行数 | 准備与可比性 |
|---|---:|---|
| `data_all/testing/mega_test.csv` | 28,312 | `prepare_megascale.py` 已盘点，含 140 行 wt，实际 28,172 突变/28 蛋白；需原始 AF PDB |
| `fireprot_HF.csv` | 2,578 | HF 表；不能当作 fireprot test split |
| `fireprot_test.csv` | 348 | 独立 test，原始 sequence alignment 与 split 规则必须复用 |
| `s669_clean_dir.csv` | 669 | 需同方向突变及结构映射 |
| `ssym-5fold_clean_dir.csv` / `inv.csv` | 各 342 | dir/inv 分开，避免符号和重复样本污染 |
| `myoglobin_clean.csv` | 42 | 外部扩展集 |

公开来源：[正式论文 DOI](https://doi.org/10.1073/pnas.2314853121)；作者预印本 [PMC10402116](https://pmc.ncbi.nlm.nih.gov/articles/PMC10402116/) 报告 Megascale 训练模型的 test Pearson 约 `0.754 ± 0.004`、FireProt test 约 `0.53 ± 0.005`，并在图中列出 Megascale N=28,312、FireProt HF N=2,578。这些是训练重复/数据版本对应的公开参考，不能当成当前发布 checkpoint 的固定 golden score。正式发表版本的表格、checkpoint 对应关系和 split 差异尚待核对，不混用预印本与最终表格。

此处存在已发现的可比性问题：论文 N 与当前 CSV 删除 wt 后的突变数不同；HF 与 test 也不同。因此本轮不设“PCC 达到 0.754 即通过”的门禁。必须先与固定上游的同权重、同数据过滤、同 PDB、同符号路径做 CPU 精度基线，之后再判断 NPU 是否对齐；原始公开指标保留为参考来源。不能用 dummy、1UBQ 的三条无标签输入或自比较结果冒充官方复现。

官方 README 公开的 V100 epoch 时间是训练耗时，不是推理性能。尚未取得可直接对齐的官方推理吞吐/延迟，NPU 性能报告绝对 mutations/s、proteins/s 及延迟，不推断 CPU 加速比。

## 数据准备与 CPU 对照

正式主线先选固定官方 Megascale test CSV + 原始 AF 单体结构。`prepare_megascale.py` 将编号转 0-based、标签取 `-ddG_ML`、排除官方不支持的 wt/indel/double/unreliable 数据；ID 含固定源行号，保留重复实验测量。`prepare_eval_data.py` 保存 CSV/manifest/PDB SHA、样本数、dataset/split 和 convention，不加载模型。

结构固定为 Zenodo 7992926 的 AlphaFold_model_PDBs.zip，按 README 的 curl 和 prepare_megascale.py 命令准备并校验发布 MD5，记录 SHA256；不得替换为 RCSB 同名蛋白或重新预测的结构。README 已提供实际功能样例及固定数据盘点/准备命令；2026-10-12 已准备 28 个 PDB 和 28,172 条 manifest；缺文件时准备入口显式失败，不静默丢样本。chain A 的单链假设必须与官方 `MegaScaleDataset` 做全量核对。保持官方只覆盖总序列 seq 的行为，seq_chain_A 保留结构序列；WT 用于突变校验。

CPU 精度基线复用 NPU Python 环境、同权重和 manifest。先按 README 运行 `baseline_upstream.py`：直接使用固定官方 `MegaScaleDataset` test split、原始 CSV/PDB 和 `TransferModelPL`，在 CPU 上预测，并输出 `baseline.json` 的来源 SHA；独立输出缓存避免历史解析污染。再运行 `infer.py --device cpu` 并比较两者，隔离新增单链预处理和样本过滤差异；NPU 数值直接对照官方 CPU 输出。官方 benchmarking 硬编码 CUDA，不修改它或伪造 CUDA 对照。CUDA 组是可选增强。

## 功能验证矩阵

| 项目 | 输入与判定 | 当前 |
|---|---|---|
| 标准单点突变 | 1UBQ A 链 M1A/Q2A/I3A，输出 3 条有限 ΔΔG、正确 ID | 真实 CPU 通过，NPU 待验 |
| 同蛋白多突变 | 共享结构 embedding，输出数与 ID 一致 | 完整 28,172 条 CPU 推理通过，NPU 待验 |
| CPU/CUDA/NPU 路由 | 默认 NPU、显式 CPU/CUDA、设备缺失立即失败 | Mock 通过，设备实测待验 |
| PDB 残基编号 | 非 1 起始编号、缺失残基、insertion code，检查 parsed index | 真实模型路径待验 |
| 异常输入 | 重复/空 ID、NaN、越界路径/索引、无 ATOM、错误 WT/缺链、混用实验 WT、突变位置缺失骨架坐标 | 离线失败测试通过；真实模型语义待验 |
| 输出证据 | 新目录、SHA、run 次数、实际 device、比较非零失败 | stdlib/Mock 通过 |
| 长输入/长度桶 | 正式 28 个蛋白长度 min/median/max 为 43/56/72 | 三条 CPU 模式全量通过，NPU 待验 |
| 批量计算等价性 | 长度 1/2/17/48/53、同长度/变长、非突变位置缺失坐标、subtract_mut 关闭 | 8 项真实 PyTorch 回归通过；人工结构不是 L2 证据 |

补丁回归必须区分两组：未修改官方 CPU 基线 → 应用补丁后的逐蛋白入口，以及补丁后的逐蛋白 → 同长度/全量合批。`--upstream-head` 仍应用补丁和精确 GELU，不能替代 `baseline_upstream.py` 的未修改官方基线。缺失坐标可能使未合批输入也含 mask=0，因此不能宣称补丁对全部未合批输入逐位不变；浮点归约形状变化也不保证逐位相同或相同 SHA。

## L2 精度与质量验收

1. 完整 manifest 冻结，核对官方 split、28 个蛋白、实际突变数、PDB SHA、实验标签符号和 WT 序列。
2. CPU/NPU 输出按相同 ID 全量比较。暂定 FP32 工程容差 `atol=1e-3 kcal/mol, rtol=1e-3`；不是官方容差，待 CPU 重复运行与首轮 NPU 数值分布校准后确认。不得为通过任意放宽。
3. `eval_metrics.py` 使用 SciPy 的 Pearson/Spearman（ties 按 SciPy 标准），NumPy 计算 pooled MSE、RMSE、R²，数学定义与上游 metric 一致；无 normalizer、绝不重标度预测。执行已知 fixture 并与上游 torchmetrics 复核后，才能声明 evaluator 等价。
4. CPU/NPU 同样计算全量指标，暂定 `|ΔPearson|/|ΔSpearman|<=1e-3`、`|ΔRMSE|<=1e-3 kcal/mol`，同为待 baseline 校准的工程建议。数值比较脚本目前只门禁逐样本输出，指标差异须验收者明确复核并记录。
5. 如官方 checkpoint/split 被证实可比，再对照正式论文公开表；不把训练均值或 FireProt HF 表移用为单 checkpoint/test 的硬线。

## L2 性能验收

在 A2 单卡使用同一全量或从全量固定出的长度分层性能 manifest，FP32，默认全蛋白合批，记录蛋白数、长度分布及每个蛋白突变数。不同模式分别验收、分别汇总，不混合计算。默认 1 warmup + 3 measured runs；正式验收至少 **三次独立进程**（输出目录分别指定），取每进程内部中位数后再取进程间中位数。

主计时包含官方 tied_featurize、GNN 输入长度填充、模型前向、按蛋白输出 CPU 传输和有限值检查，不包含 checkpoint load、PDB parse、行字典合并、写文件；这是模型及特征计算吞吐，不是完整外部调用端到端性能。报告 mutations/s、proteins/s、总秒数和峰值 HBM（实机另用监控日志）；另报首次调用和全流程时间。没有公开可比推理线时报告实际值，不编造性能通过阈值或 GPU/CPU 加速比。

运行前核对 CANN/驱动/固件/Python/torch/torch_npu/pip freeze、物理可见卡与进程逻辑卡映射。检查 stderr/profiler 的 fallback，输出 tensor 位于 NPU 不足以证明所有核心算子均在 NPU 上运行。

schema 3 的性能汇总要求 Python、逻辑设备编号、ASCEND_HOME_PATH、实际 CANN 版本、NPU 型号和主机名一致；缺失或未知环境证据立即失败。旧 schema 2 记录不能补填计划版本后继续使用，必须重新推理。主机名和设备型号不代替物理卡/驱动/固件实机证据。
本轮还要求 `infer_sha256`、`patch_sha256`、`inference_mode` 和 `gelu` 相同；缺这些字段的旧 schema 3 结果也必须重跑。用户提供的 2,923 / 38,881 / 68,308 / 278,651 mutations/s 及 452× 仅作为待核验线索，原始日志、设备配套、模式、同步和计时范围未齐全前不填入正式 NPU 性能表。
报告主入口无论是否提供独立性能目录都执行相同的 schema/环境检查，并将实测耗时与保存的中位数/吞吐核对；实际预测行数和结构/链数量必须等于声明值。指标缺失、非有限或越界立即失败，CPU/NPU 指标差异复核仍为显式 pending。

## 最低正式验收清单

- README 的安装、模型/evaluator imports 和 NPU tensor 实测通过。
- 固定源码、主/辅助权重、正式 split、原始 PDB 与 manifest SHA 记录齐全。
- 真实功能输入至少三条输出，CPU 预处理与固定官方路径回归通过。
- 全量 NPU L2 的 predictions、质量指标、CPU 数值比较及公开基线可比性复核通过。
- 三次独立 NPU 性能结果，中位数和计时范围明确，无未解释核心算子回退。
- 独立候选目录 clean-room 真实安装/推理/比较重放；许可证、贡献要求、自测试和 modelzoo_level.txt 据实满足。

执行入口和完整命令集中在 README。`report_acceptance.py --performance-dirs` 汇总至少三个独立目录，校验不同 run_id、预测 SHA 和一致的配置；缺少此参数时将独立性能验收列入 pending。它还核对 NPU device、模拟标志、次数、样本数与比较 SHA；不会把收集成功自动标为 S3。

## 报告模板与当前结果

| 字段 | 当前结果 |
|---|---|
| 日期 / 状态 | 2026-10-14 / S2 CPU 功能验证，待 NPU 验收 |
| NPU 型号 / CANN / 框架 | 模拟 Atlas 800I A2 / 8.3.RC1 / 2.1.0 + 2.1.0.post17，未安装实测 |
| 功能 / Mock | 本轮回归结果见文末，仅软件契约 |
| 原始官方数据盘点 | CSV/PDB/manifest 已准备；官方 split 与真实模型对照待验 |
| CPU 功能与指标 | Python 3.12.14/PyTorch 2.5.1+cpu：真实 1UBQ 及全量 Megascale 对照通过；目标配套另验 |
| NPU 功能 / 精度 / 性能 | 未执行，当前无硬件 |
| 阈值 | 工程暂定，待 baseline 校准 |
| clean-room | 本轮独立候选重放结果见 NPU_ADAPTATION.md；NPU 路径待验 |
| 上库候选 | 未就绪，S3 和贡献门禁待完成 |

升级 S3 后在本文件追加实际设备、软件版本、全量样本数、指标、误差分布、三次性能数据、命令和日志路径；不得覆盖此次模拟事实。

2026-10-12 检视修复：预处理恢复固定上游序列字段语义；补充固定结构归档下载、校验、离线选择提取；独立进程性能汇总及负向测试已补齐。真实 NPU 结果仍未产生，状态保持 S1。

2026-10-12 后续修复：新增独立官方 CPU 基线入口与来源/身份校验，README 给出官方 CPU → 新入口 CPU → NPU 的完整比较命令。补齐性能环境一致性门禁与 schema 3 运行记录。19 项离线回归通过；真实模型 CPU/NPU 回归仍受目标依赖和硬件阻塞，不据 Mock 升级状态。

2026-10-13 检视修复：23 项离线回归和 6 项仓库门禁回归通过；独立候选重建的 28 个真实结构、28,172 条突变与三条功能输入通过新增索引/WT/骨架坐标检查，缓存前后 manifest 逐字节一致。报告补齐省略独立性能目录时的环境校验、实际预测计数和计时摘要核对；仍需人工复核 CPU/NPU 指标差异。真实 evaluator 数学 fixture 通过，仅证明计算入口可运行。目标依赖安装、官方 CPU 模型输出及全部 NPU 精度/性能均未完成，状态保持 S1，上库候选未就绪；部分 clean-room 证据详见 NPU_ADAPTATION.md。

2026-10-14：独立候选完成 28 蛋白、28,172 突变的未修改官方 CPU 基线及默认全量合批/同长度桶/逐蛋白三条 CPU 路径。三条优化路径预测逐位相同，SHA 相同；相对官方最大绝对误差 `2.110004425048828e-05`、平均 `2.069520151413724e-06 kcal/mol`，通过既定逐样本容差。官方/合批 CPU 的 Pearson 为 `0.7539695024 / 0.7539695648`、Spearman 为 `0.7224582348 / 0.7224583221`、RMSE 为 `0.7174742204 / 0.7174741323`；该 CPU 差异仅用于 patch 回归。23 项离线回归、8 项真实 PyTorch 回归、6 项仓库门禁通过；真实 1UBQ 和独立候选 CPU 重放通过。状态为 S2，NPU 数值/指标/性能、目标配套和正式贡献门禁仍待验收。
