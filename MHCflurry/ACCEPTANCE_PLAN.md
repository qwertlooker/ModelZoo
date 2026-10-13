# MHCflurry 验收方案与结果

## 验收目标与版本

本轮范围：已有适配代码与配套文档检视、修复及离线回归，保持 S1。未来真实验收按 S2/S3
继续推进。源码 2.3.14 / `20f03d45358dd104df9df0e28e2e0ccea11c9bfb`，权重 release
2.3.0 / `models_class1_presentation.20260928.tar.bz2`，SHA 在 README 中固定。
完整 ensemble，FP32 eager，固定 batch，原始编码/集成/IC50/百分位/sklearn
后处理。非训练、非旧 TF 模型，不适配 NetMHCpan 或 MixMHCpred。

## 原始测试集与官方/公开指标

来源：固定源码 `docs/release_model_evaluation.md` 和官方
[comparison-v4](https://github.com/openvax/mhcflurry/releases/download/2.3.0/model-comparison.20260928-v4.tar.gz)。

| 项目 | 官方定义 |
|---|---|
| Cohort | 181,131 retained rows，16,400 positives / 164,731 negatives，10 patients |
| 身份 | 原始 cohort SHA `9f0fbc92fd44ac58438102ae7496687b67f109685d8f1fecfc820a7262bf46cc` |
| sampling | source-protein/length-matched negatives，seed 20260928，初始每个正例 10 个负例 |
| 排除 | 相同移除 380 rows，所有 comparator 共用同 cohort，不重新抽负例 |
| 指标 | 等 patient 平均 AP、PPV@N、AUROC；N 为该 patient retained 正例数 |
| 同分 | 截断处按同分组期望 precision；不能依输入行顺序决定 |
| 置信区间 | 10,000 paired patient bootstrap，seed 42；未多重检验校正 |
| 有 flank 2.3.0 | AP 0.7986，PPV@N 0.7450，AUROC 0.9524 |
| 无 flank 2.3.0 | AP 0.7893，PPV@N 0.7373，AUROC 0.9501 |

这些是 presentation labels 的排序结果，affinity ligand-ranking AP 0.7224
不是定量 affinity 的独立实验误差；processing component AP 也不能被说成
独立 processing 端点准确率。2020/2023 权重和有/无 flank 不可混用。

公开报告下载包只有 aggregate tables、绘图代码、PROVENANCE 和图，未含原始
逐条 cohort，当前无法直接复现该公开表。上游另有历史 `data_evaluation.20200611`
分片（固定 catalogue），但不能未经审查当成 2.3.0 comparison-v4 同 cohort。
不得声称“官方未发布任何指标”；准确结论是“已有公开指标，当前原始逐条输入未取得”。

## 数据准备和 CPU 精度对照

功能验证使用仓内四条肽和人工 flank，只检查三类任务/输出字段/异常传播，
不做准确率或性能结论。正式数据通过 prepare_eval_data.py 固定成 CSV 和
manifest.meta.json，同一 manifest 供 CPU 与 NPU 使用。
presentation L2 准备和评测共用 patient 校验：sample_id 不得为空或纯空白，
必须有二元 label 且每个 patient 同时有正负样本。推理 genotype 分组不改变
评测 patient 分组；按有序 allele 集合组批，并逐行恢复输出。
推理必须保留并校验官方归档及模型目录逐文件哈希，拒绝改动后的 ensemble。

原始 cohort 可得时逐条固定 peptide、allele sets、flanks、sample_id、labels，
核实 SHA、patient 数、正负比例和训练来源排除。否则选择有授权、真实、覆盖
8–15 肽长、多 allele、flank/无 flank 的公开或业务 held-out cohort，至少 1,000
记录，记录来源 URL/版本/split/筛选规则和 SHA；**名称只能是迁移对照集**，
不能冒充原始官方 cohort，不能靠重复四条 demo 扩充到最低规模。

无同 cohort 公开基线时必须运行同权重 CPU（可复用 NPU 配套环境）。CUDA 是
可选增强，不作为门禁。README 已给完整数据、CPU/NPU、compare 和指标命令。
独立定量 affinity 集还需实验 nM 标签、测量类型/不等式及官方协议，当前不以
presentation labels 或自定义误差冒充该任务官方指标。

## 功能验证矩阵

| 项目 | 验证要求 |
|---|---|
| affinity | 单 allele，nM 有限正值，CPU/NPU 同结果字段 |
| processing | 有/无 flanks，0–1，输入缺少 allele 可执行 |
| presentation | 每条单/多 allele，batch 1/128，最小 affinity allele 正确 |
| flank | 两列成对，空值表示已提供 flank 模式，不与无列模式混淆 |
| 输入边界 | 长度 8/15、非法字符/长度/allele、空输入、空白或重复 ID、重复/空表头、行列数不符、损坏 CSV 严格失败 |
| 输出边界 | affinity 有限且 >0；processing/presentation 分数在 [0,1]；percentile 在 [0,100]；缺字段、空白 ID、重复表头/列数不符、非整数 peptide_num 严格失败 |
| 设备 | NPU 不可用/缺 torch_npu 直接失败；CPU/CUDA 原路径不引入 NPU 依赖 |
| 证据 | 请求和实际 tensor/network device 一致，输入/权重/源码 SHA 可追溯 |
| 错误 | OOM、下载错误、哈希不符、字段缺失不静默降级；输出目录及报告文件禁止覆盖，报告路径误指向 manifest/预测文件时保留原文件 |

## L2 精度/质量和性能

数值比较逐 ID 对 affinity、processing_score、presentation_score、percentile
使用 `abs(candidate-reference) <= atol + rtol*abs(reference)`；暂定 0.001/0.001。
每个字段均有限，缺项/重复/证据不一致直接失败。阈值不是官方给出的容差，必须
用固定 CPU 重复运行及批次边界差异校准，再形成冻结的工程通过线。
比较前逐列验证任务输出的有效范围，CPU 与 NPU 即使产生相同的越界值也不能通过。
eval_metrics.py 独立执行时同样检查所有 presentation 字段，而非仅检查排序分数。
presentation 同时保存并逐 ID 精确比较官方 best_allele；空值、缺失或 allele
不一致均失败，不以数值容差豁免 allele 选择变化。覆盖近似同分和 allele 顺序回归。
presentation metrics 的暂定绝对 drop 为 0.001，按等 patient AP/AUROC/PPV@N
分别比较，不平均任务 AP。数值比较器及 eval_metrics.py 自动报告/非零失败。
指标入口独立核验网络实际设备，报告与比较共用源码 revision、MHCflurry/torch
版本、manifest/权重哈希、task、batch、optimization_level、compile 字段。
缺少字段的旧报告必须从原始运行证据重新生成，禁止静默兼容或手工推断。

原始 cohort 可得时按相同 evaluator 和 flank 配置核验公开基线；否则报告固定
迁移 cohort 的 CPU/NPU 指标和公开表参考，不写“复现官方指标”。S3 需人工
核验 cohort 合格、阈值冻结、主要任务指标及 NPU 真实执行；compare 不自动定级。

性能用同一正式 manifest，batch 128，全 ensemble、FP32、两次 warmup 后至少
三次计时取中位数；每次包括 encoding、传输、网络、ensemble 和 CPU sklearn
后处理，前后 NPU synchronize；排除模型加载和 CSV 写盘，冷启动另记。
正式复核建议三次独立进程，三个独立 output-dir，核实噪声后汇总。
指标为 peptides/s 和端到端 latency，采集真实 HBM/RSS、算子 fallback/占比，
不同 batch 不混算。当前未核实可比官方硬件性能，不设虚假 GPU 加速比或 FPS。

## 最低正式验收清单

1. 真实 Atlas 800I A2 的型号、配额、物理/逻辑卡映射、驱动/固件/CANN、
   Python、torch/torch_npu 和容器 digest；真实 NPU 张量分配/计算/同步。
2. 固定 checkout、patch 正反向 check；业务依赖安装、pip check 和源码锁核验。
3. 权重归档 SHA、全文件清单及必需子模型均可读。
4. 三任务有/无 flank 的真实 smoke；批次边界/异常输入回归。
5. 真实固定 L2 manifest/meta 资格与来源审查、同权重 CPU 基线和 NPU 输出。
6. 数值 compare、正式 presentation metrics、独立任务实验指标适用性和阈值校准。
7. NPU 三次以上真实性能与中位数；网络实际设备、隐式 CPU fallback 审查。
8. 按正式候选清单完成 clean-room 安装至报告的重放，另做许可证/CLA/CodeCheck/
   SCA/Antipoison/modelzoo_level/目标路径门禁，才可说上库候选就绪。

## 报告模板与当前结果

| 项 | 当前结果 | 正式报告应附 |
|---|---|---|
| 环境 | Windows/Python 3.14.6 控制机，无 NPU | CANN/驱动/固件/版本/digest/device mapping |
| 静态/Mock | 已执行，结果见 NPU_ADAPTATION.md 的验证记录 | 测试日志、patch/source SHA |
| 安装/模型功能 | 未执行 | import、权重清单、真实三任务输出 |
| L2 精度/质量 | 未执行 | cohort/meta、CPU/NPU、numeric/metric reports |
| L2 性能 | 未执行 | 三次 NPU 时间、median、吞吐、HBM/RSS、计时边界 |
| clean-room | 轻量入口/fixture/Mock 重放；非完整模型安装 | 独立候选完整重放日志 |
| 状态 | S1 静态适配完成，待真实 S2/S3 | 依据当前证据定级，不沿用旧状态 |

2026-10-13 检视回归新增覆盖 CSV 字段覆盖、损坏行、结果范围、完整指标输入、
非整数 presentation 行索引及报告覆盖失败路径。fixture/Mock 只证明校验和错误
传播，不作为 CPU/NPU 模型输出、正式指标或性能证据。
