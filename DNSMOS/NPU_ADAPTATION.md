# DNSMOS NPU 适配文档

## 1. 版本与来源

- 官方源码：<https://github.com/microsoft/DNS-Challenge.git>
- 分支/commit：`master` / `591184a9fcb2cbdec02520fed81a32bbbf9d73ff`
- Ascend-SACT 参考：<https://gitcode.com/Ascend-SACT/DNSMOS>，commit `d1e4c2c14df9cb935d61dc5f448e655772b12379`
- 检查日期：2026-06-29；两个远端 HEAD 与上述本地 commit 一致。
- 权重：官方仓 `DNSMOS/DNSMOS/model_v8.onnx`、`DNSMOS/DNSMOS/sig_bak_ovr.onnx`、`DNSMOS/pDNSMOS/sig_bak_ovr.onnx`。
- 变体边界：包含常规和 personalized DNSMOS P.835；不包含在线 DNSMOS API。
- 目标 NPU 运行组合：Python 3.10、CANN 8.2.0、
  `onnxruntime-cann==1.22.1`；CPU baseline 使用独立环境中的
  `onnxruntime==1.22.1`。

## 2. 代码分析与适配

官方 `dnsmos_local.py` 使用 ONNX Runtime 默认 provider。当前 `infer.py` 只增加显式设备边界：

- `--device npu` 默认选择 `CANNExecutionProvider`，provider 不存在时立即失败；
- `--device cpu` 显式选择 `CPUExecutionProvider`；
- 不调用 `npu-smi`、不修改设备性能模式、不写死卡号；
- 保留官方 16 kHz 重采样、短音频重复、9.01 秒窗口、1 秒 hop、所有窗口平均、P.808 mel 特征和多项式校正；
- 支持递归目录输入，并输出 provider、耗时和 RTF。
- 支持固定 JSONL manifest；每次运行写出模型 SHA、provider、环境和命令
  sidecar metadata；
- `prepare_eval_data.py` 负责验证 WAV 并固定 manifest，`compare_results.py`
  负责 CPU/CUDA 与 NPU 的逐字段门禁。

Ascend-SACT 脚本只评估中间窗口，长音频结果与官方逐窗平均不等价，因此未直接照搬。当前修改没有改动官方已有源码，不需要上游 patch。

## 3. 验证记录

2026-06-20 已完成：

- 固定官方与参考实现 commit；
- 对照官方 `dnsmos_local.py` 审查预处理、窗口、模型输入和校正公式；
- `python3 -m py_compile DNSMOS/infer.py` 通过；
- `prepare_eval_data.py` 和 `compare_results.py` 已加入正式交付入口；
- 权重文件在官方浅克隆中存在且不是 Git LFS 指针。

CPU 实测环境和结果：

- Python 3.12.3、ONNX Runtime 1.27.0 CPU、librosa 0.11.0、
  NumPy 2.4.6、soundfile 0.13.1；
- 输入：DiariZen `example/EN2002a_30s.wav`，30 秒，SHA256
  `55aa90540de2a01e6824ee2862d08763026d1752a3d7f6f8870015c74424e900`；
- 常规和 personalized 两条路径均使用 21 个窗口；
- 分别与同 commit 官方 `dnsmos_local.py` 比较
  `len_in_sec/sr/num_hops`、全部 raw/校正后 MOS 和 `P808_MOS`，
  最大/平均绝对误差均为 `0.0`；
- 当前脚本首次常规运行 `elapsed=9.834s, RTF=0.327785`；warm-up 后两次运行
  `elapsed=1.880~2.026s, RTF=0.062662~0.067521`。单条数据只用于 CPU
  链路和算法等价性，不作为性能结论。

补充闭环工具验证：

- `prepare_eval_data.py` 对同一 30 秒样例生成 1 条 manifest，记录采样率、声道、
  音频 SHA 和 manifest SHA；
- `infer.py --manifest ... --device cpu` 成功生成 CSV 和 `*.meta.json`，
  实际 provider 为 `CPUExecutionProvider`，本次 `RTF=0.076430`；
- `compare_results.py` 对该 CSV 自比较，七个字段最大/平均误差均为 `0.0`，
  单样本 Spearman 按完全相同处理为 `1.0`。

系统没有 `npu-smi`/CANN NPU 环境，因此未执行 NPU 数值或性能验收。正式验收必须按
[ACCEPTANCE_PLAN.md](ACCEPTANCE_PLAN.md) 生成 CPU/NPU CSV 并逐字段比较。
用户环境、权重和运行命令见 [README.md](README.md)。

当前交付状态：**S2，CPU 算法等价性已验证；升级到 S3 仍缺 NPU 同 manifest
精度和性能对齐**。不能仅凭
静态检查或本文档将状态标记为“适配验收完成”。

## 上库就绪与目标仓对齐

- 目标仓快照：`https://gitcode.com/Ascend/ModelZoo-PyTorch.git`，2026-06-29 重新查询
  `master` HEAD `7a02a6701c971b29df188a0f3241e1efe249d1df`（commit message "modify document"）。
  2026-06-22 审阅快照为 `ec2a7b514973805f66b67c9178d2f5c9e97eee34`；本次为重新查询，不复用历史快照。
- 拟合入路径：`ACL_PyTorch/built-in/audio/DNSMOS`。目标仓 `ACL_PyTorch/built-in/audio/`
  下不存在 `DNSMOS` 目录，本次为新增，不涉及替换或增量更新。
- 最新参考目录：同领域、同推理形态（audio / ONNX-OM、`CANNExecutionProvider`）按最后
  实质变更时间选取 `ACL_PyTorch/built-in/audio/AASIST-L_for_Pytorch`，其最后实质变更为
  commit `6fecdfba7`（2026-06-18，建立 `pth2onnx.py`/`modify_onnx.py`/`om_val.py` +
  `LICENSE` + `modelzoo_level.txt` 的完整 ONNX/OM 交付）。选择原因是同属 audio ONNX/OM
  推理形态且 PR 门禁文件最全。更新的 `YingMusic-SVC_for_Pytorch`（`e98df562e`，
  2026-06-22）为 SVC 形态，推理链路不同，仅作 PR 门禁参考。
- 贡献规范与 PR 门禁：`Ascend/modelzoo` HEAD `5eab9a4921c7f12edb555079836429a8f285cd1f`
  的 CONTRIBUTING.md 要求源码、README、参考模型 License、测试用例；AASIST-L 另含
  `modelzoo_level.txt`，但 Canary-1B、chronos-2 等同领域目录未提供 LICENSE/
  modelzoo_level.txt，说明历史目录与当前 PR 门禁存在差异。执行时按贡献规范提交，
  不因历史模型缺文件而跳过，也不为形式伪造未执行的自测试或状态文件。
- 上库文件清单（候选）：`README.md`、`infer.py`、`prepare_eval_data.py`、
  `compare_results.py`、`requirements.txt`；上库前补 `LICENSE`、`modelzoo_level.txt`。
- 排除项：`NPU_ADAPTATION.md`、`ACCEPTANCE_PLAN.md`、`README_old.md`、
  `patches/README.md`、`*.png` 截图、`upstream/`、`weights/`、`eval_data/`、
  `eval_results/`、`.codex-reference/`、日志与虚拟环境。
- 许可证：上游 `microsoft/DNS-Challenge` 为 MIT License，上库时在模型顶层目录拷贝上游
  LICENSE 并按贡献规范为新增脚本追加华为 License 头部。`modelzoo_level.txt` 的
  Func/Perf/Precision 状态须在 NPU 实测后据实填写，未达 S3 前不得伪造 OK。

## 需求—证据表 (Step 14.2)

| 要求 | 权威证据 | 状态 |
|---|---|---|
| 版本固定 | upstream `591184a9` / 权重 ONNX SHA256 已记录 (NPU_ADAPTATION §1) | 已证实 |
| 代码适配 | 无 patch（只增加显式 provider 选择），`py_compile` 通过 | 已证实 |
| 环境可安装 | CPU 环境 `onnxruntime==1.22.1` + requirements.txt 安装及导入测试通过 | 已证实 |
| 数据可准备 | VCC2018 `wget` 下载 + `prepare_eval_data.py` 生成 manifest/meta 通过 | 已证实 |
| 原始 baseline | CPU vs 官方 `dnsmos_local.py` 全字段误差 0.0 (NPU_ADAPTATION §3) | 已证实 |
| patch 回归 | 不适用（无 patch） | 不适用 |
| NPU 对齐 | 待 CANN 环境实测 | 缺失 |
| 正式指标 | 论文 unseen test set 未公开，VCC2018 为迁移回归集 | 已证实（口径） |
| L2 性能 | 待 CANN 环境实测 | 缺失 |
| 上库候选 | 拟合入路径 `ACL_PyTorch/built-in/audio/DNSMOS`，文件清单已列，待补 LICENSE + `modelzoo_level.txt` | 已证实（清单） |

## 补充说明（来自 README.md）

### ONNX Runtime 与 CANN 版本配套

ONNX Runtime 官方 CANN EP 配套表将 1.22.1 对应到 CANN 8.2.0。CPU 和 NPU
环境必须分开创建，避免 CPU `onnxruntime` 覆盖 CANN 构建。

### 内部构建 CANN EP 的替换要求

如果目标基础镜像使用内部构建的 CANN EP，应以对应 wheel 替换上述 PyPI 包，并在
验收报告中记录 wheel 文件名和 SHA256；不得同时安装 `onnxruntime`。
