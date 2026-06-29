# Ascend-SACT 模型 NPU 适配整合分析报告

本文整合原 `参考原始仓库.md`、`NPU_ADAPTATION_ANALYSIS.md`、`DETAILED_NPU_ADAPTATION_ANALYSIS.md` 三份文档，统一记录各模型目录的参考原始仓库、版本边界、NPU 适配静态评估、后续适配/验证工作量与落地建议。

> 基准检查日期：2026-05-25（原 9 个仓库版本边界） / 2026-05-22（原静态分析） / 2026-06-20（本轮六模型专项复查）。后续适配前需按《模型 NPU 适配标准流程》重新确认上游版本、权重校验值、依赖版本和评测数据可用性。
>
> 分析范围：`/home/pei/ModelZoo` 下已克隆的 9 个 GitCode/交付仓库。
>
> 分析方式：静态代码/文档分析，未在 Ascend NPU 上实际执行。判断依据来自各仓库 `README.md`、随仓脚本、`requirements.txt`、大文件/LFS 状态、上游工程说明以及是否已经形成类似 `Canary-1B/README.md`、`Canary-1B/README.md` 的可交付推理文档。
>
> 约束：不添加未验证的 CPU fallback、远程下载 fallback、非官方指标替代官方指标；缺少依赖、缺少官方字段、上游版本不兼容或官方评估组件不可用时应快速失败并暴露原始错误。

---

## 2026-06-20 六模型专项复查

本轮按 Canary-1B 的三类主文档结构复查以下模型，并分别新增
`README.md`、`NPU_ADAPTATION.md`、`ACCEPTANCE_PLAN.md`。

| 模型 | 参考仓 commit | 当前状态 | 当前结论与主阻塞 |
|---|---|---|---|
| DNSMOS | `d1e4c2c14df9cb935d61dc5f448e655772b12379` | S2 | CPU 常规/personalized 与官方脚本全字段误差 0；已补 manifest/compare，待 CANN/NPU S3 |
| speechscorer | `f1d6e3ee3d0f113c610a969e6fde4a29af3216d1` | S1 | 已修正原始公开路径为 HuBERT-MLM并补数据/比较入口；待 3.8GB 权重、fairseq 和 NPU 实测 |
| Hy3-preview | `eb533c1dfd9a1fa7f373f9b980a9c0f973f1dad8` | S1 | patch 静态门禁通过并补服务回归工具；待 295B/16 卡 TP/EP/MTP、parser 和官方任务实测 |
| MiroThinker-1.7 | `a4199f82dcadf88e81e296eb2d0e79bdb5805184` | S1 | 已修复服务名和 8K/256K 配置冲突并补固定 prompt；待 235B/16 卡、外部工具和 agent benchmark |
| MolFormer | `b39184dcb79501f0cd81def11e7b934176194a4c` | S2 | 固定权重和 10 条 CPU embedding 实测通过，已补 manifest/compare；待 NPU 对齐及 11 项 fine-tuning |
| BUTSpeechFIT-DiariZen | `7961b5ab79b1232b9da367f14f8cd4f592694465` | S1 | 已修复 dscore 命令并补 manifest/provider/DER 工具；待权重、RTTM 对齐和正式数据 DER |

本轮完成版本取证、代码/patch 整理、三类主文档、可执行验收入口和静态验证；当前机器没有
NPU/CANN 运行环境和模型权重，因此真实 NPU 数值、性能和官方数据集验收仍以各模型
`NPU_ADAPTATION.md`、`ACCEPTANCE_PLAN.md` 的记录为准。

### 2026-06-29 复核更新

按《模型 NPU 适配标准流程》重新查询目标仓并复核上述六模型：

- 目标仓 `https://gitcode.com/Ascend/ModelZoo-PyTorch.git` `master` HEAD 由 2026-06-22
  的 `ec2a7b514973805f66b67c9178d2f5c9e97eee34` 更新为
  `7a02a6701c971b29df188a0f3241e1efe249d1df`（commit message "modify document"）；
  贡献规范 `Ascend/modelzoo` HEAD 仍 `5eab9a4921c7f12edb555079836429a8f285cd1f`。
- 各模型 `NPU_ADAPTATION.md` 新增「上库就绪与目标仓对齐」章节，记录目标仓快照、
  拟合入路径（六个均为新增）、按最后实质变更排序的同领域/同推理形态参考目录
  （均为 commit `6fecdfba7`，2026-06-18 建立）、贡献规范与 PR 门禁结论、上库文件清单
  和排除项；版本检查日期统一更新为 2026-06-29。
- README 补全 `commit_id` 声明；Hy3-preview、MiroThinker-1.7 去除 `/workspace/ModelZoo`
  硬编码路径（改用 `MZ` 变量）并清理目录树中项目级 `tools/` 引用；MiroThinker-1.7
  新增 `serve_npu.sh` 可运行服务入口；speechscorer/MolFormer/BUTSpeechFIT-DiariZen
  去除 `待验收`、`<...>` 占位符。
- `python3 tools/audit_model_delivery.py <model>` 基础门禁六模型均 PASS；
  `--target-readiness --target-path ACL_PyTorch/built-in/<领域>/<模型>` 仅剩
  「当前状态: S3」一项。因当前机器无 NPU/CANN 环境与权重，六模型仍为 S1/S2，
  按规则不得伪造 S3，故保留为“尚未验收/上库候选未就绪”。

---

## A. 参考原始仓库与适配版本边界

本文记录当前仓库中各模型目录对应的参考原始仓库、当前适配对象和版本边界，便于后续适配、排查和文档对照。

> 检查日期：2026-05-25。对于 Git / Hugging Face / ModelScope 可通过 `git ls-remote --symref <repo> HEAD` 检查的仓库，下面记录的是本次检查到的默认分支 HEAD；后续适配前需按《模型 NPU 适配标准流程》重新确认。

| 模型系列 | 模型名称 | 当前目录 | 参考原始仓库 | 当前适配对象 / 版本边界 |
| --- | --- | --- | --- | --- |
| DNSMOS | DNSMOS | `DNSMOS/` | [microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge)；当前交付仓库：[Ascend-SACT/DNSMOS](https://gitcode.com/Ascend-SACT/DNSMOS) | 官方源码 `master` HEAD `591184a9fcb2cbdec02520fed81a32bbbf9d73ff`。适配脚本加载官方 `DNSMOS/model_v8.onnx`、`DNSMOS/sig_bak_ovr.onnx`，个性化模式加载 `pDNSMOS/sig_bak_ovr.onnx`。 |
| speechscorer | speechscorer | `speechscorer/` | [yaya-sy/speechscorer](https://github.com/yaya-sy/speechscorer)；当前交付仓库：[Ascend-SACT/speechscorer](https://gitcode.com/Ascend-SACT/speechscorer) | 上游 `main` HEAD `bbe0be772b37f472994d5a97f809214fd67a2c8e`；当前第一阶段目标固定公开主路径 `hubert-mlm`、HuBERT checkpoint `hubert_base_ls960.pt` 与 SpeechOcean762，`whisper-clm` 仅保留为后续扩展，不作为本轮验收主线。 |
| Tencent Hy | Hy3-preview | `Hy3-preview/` | [Tencent-Hunyuan/Hy3-preview](https://github.com/Tencent-Hunyuan/Hy3-preview)；当前交付仓库：[Ascend-SACT/Hy3-preview](https://gitcode.com/Ascend-SACT/Hy3-preview) | 官方代码 `main` HEAD `38ac237dc0bf4329f054d09054aaf22fdaf6f553`；权重 `tencent/Hy3-preview` HEAD `549c2b3a0fd5b9a6c6059a9935bf0d59ab69d75a`；适配基线为 vLLM/vLLM-Ascend `v0.18.0rc1`，目标是 295B Instruct 模型，非 Base/量化变体。 |
| MiroMind | MiroThinker-1.7 | `MiroThinker-1.7/` | [MiroMindAI/MiroThinker](https://github.com/MiroMindAI/MiroThinker)；当前交付仓库：[Ascend-SACT/MiroThinker-1.7](https://gitcode.com/Ascend-SACT/MiroThinker-1.7) | upstream `main` HEAD `370f98361553ddf787bedc5745760e04114cb161`；权重 `miromind-ai/MiroThinker-1.7` HEAD `1a42014ce72e1025fdbf3c48d54545715ab3eea8`；适配基线为 vLLM/vLLM-Ascend `v0.17.0rc1`，目标为 235B 模型及其 MiroFlow Agent，不是 mini/v1.5/v1.0/H1。 |
| IBM | MolFormer | `MolFormer/` | [IBM/molformer](https://github.com/IBM/molformer)；当前交付仓库：[Ascend-SACT/MolFormer](https://gitcode.com/Ascend-SACT/MolFormer) | IBM 源码 HEAD `3b9ac434db387fadf2cf99b99def654cbf193841`；目标权重为 `ibm-research/MoLFormer-XL-both-10pct` HEAD `7b12d946c181a37f6012b9dc3b002275de070314`，不是论文完整 100% 预训练权重。 |
| MMAudio | MMAudio | `MMAudio/` | [hkchengrex/MMAudio](https://github.com/hkchengrex/MMAudio)；当前交付仓库：[Ascend-SACT/MMAudio](https://ai.gitcode.com/Ascend-SACT/MMAudio) | 源码默认分支 `main`，HEAD `974010a026c731054592d8f777218bd9d85a6c24`。适配文档使用官方 MMAudio 工程及其模型资源，并额外固定手动下载依赖：`apple/DFN5B-CLIP-ViT-H-14-378`、`nvidia/bigvgan_v2_44khz_128band_512x`（HF HEAD `95a9d1dcb12906c03edd938d77b9333d6ded7dfb`）。 |
| NeMo | Canary-1B | `Canary-1B/` | [NVIDIA-NeMo/NeMo](https://github.com/NVIDIA-NeMo/NeMo)；当前交付仓库：[Ascend-SACT/Canary-1B](https://ai.gitcode.com/Ascend-SACT/Canary-1B) | NeMo 源码默认分支 `main`，HEAD `44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe`。适配权重为 [nvidia/canary-1b](https://huggingface.co/nvidia/canary-1b) / `canary-1b.nemo`（HF HEAD `1698acf1700ed316ffce1cb42d79437c7e360cfa`），非 `canary-1b-flash` / `canary-1b-v2`。本地验证权重 SHA256：`b0284183a9a1e039a2fff39427e2991fa4df0b9612a3447fc33ff82b20fdfb5a`。 |
| BUTSpeechFIT | DiariZen | `BUTSpeechFIT-DiariZen/` | [BUTSpeechFIT/DiariZen](https://github.com/BUTSpeechFIT/DiariZen)；当前交付仓库：[Ascend-SACT/BUTSpeechFIT-DiariZen](https://ai.gitcode.com/Ascend-SACT/BUTSpeechFIT-DiariZen) | 源码默认分支 `main`，HEAD `a60b18151dbbe246e4199d8ef5cd2ece3872ea94`。权重为 HF `BUT-FIT/diarizen-wavlm-large-s80-md`，HEAD `a9b1b0e7974d96dcfd63af417e9da7ad8714040f`；评测辅助 `nryant/dscore` commit `e02f949ac6592279300a2c33d03daf9e0c12fd27`。 |
| MOSS | MOSS-TTSD-v0.5 | `MOSS-TTSD-v0.5/` | [OpenMOSS/MOSS-TTSD](https://github.com/OpenMOSS/MOSS-TTSD)；[OpenMOSS-Team/MOSS-TTSD-v0.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v0.5)；[fnlp/XY_Tokenizer_TTSD_V0](https://huggingface.co/fnlp/XY_Tokenizer_TTSD_V0) | 2026-06-16 复查：GitHub 默认分支 `main` HEAD `20dbb4fc44819435fee894d644a0402a0fee736a` 已面向 v1.0；当前适配边界改为原项目 tag `v0.5` / commit `0e078c62389922d3aa873ce182daf31142860b18`，模型权重 `fnlp/MOSS-TTSD-v0.5` / `OpenMOSS-Team/MOSS-TTSD-v0.5` HEAD `8527b9136b6afefe2252ae597cecea2e80e7ebeb`，XY Tokenizer 使用原项目 `XY_Tokenizer` + `fnlp/XY_Tokenizer_TTSD_V0` 的 `xy_tokenizer.ckpt`（HEAD `c83433728e698ed0698e88cb5096bc221fb8f8c5`）。非 v0.7/v1.0/SGLang/未固定一键包。 |

---

## B. 总体静态评估摘要

分析日期：2026-05-22

说明：本报告基于 `/home/pei/ModelZoo` 下已克隆仓库的 README、脚本、requirements 与仓库文件结构做静态分析；当前环境未提供 Ascend NPU/CANN，未做真实运行验证。MMAudio 中的大 tar.gz 为 Git LFS 指针，未拉取实际大文件。

| 仓库 | 适配完整度 | 后续适配工作量 | 功能验证工作量 | 性能验证工作量 | 精度验证工作量 | 数据/权重获取难度 | 主要判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| DNSMOS | 高 | 低 | 低 | 低-中 | 中 | 低-中 | 已有 CANNExecutionProvider 推理脚本、批处理、结果 CSV；权重和 VCC2018 等数据较易获取。 |
| Canary-1B | 中 | 中 | 中 | 中 | 中 | 中-高 | NeMo 生态复杂但脚本简单；多语言 ASR/翻译验证数据准备比中文 ASR 稍复杂。 |
| MMAudio | 低-中 | 高 | 中-高 | 中-高 | 高 | 高 | 多处手工改 cuda/npu 与 dtype，2 卡要求；仓库大包为 LFS 指针，验证生成音频质量成本高。 |
| BUTSpeechFIT-DiariZen | 低-中 | 高 | 中-高 | 中 | 中-高 | 高 | 主要是安装/补丁说明，无随仓 infer.py；pyannote/DiariZen/dscore 组合依赖重。 |
| MOSS-TTSD-v0.5 | 中 | 中 | 中 | 中-高 | 高 | 高 | 已按“原项目代码优先”收敛为基于 tag v0.5 的 patch 适配；仍需下载大权重、真实 NPU 实推和主观/客观质量验证。 |

## 建议优先级

1. **第一批落地**：DNSMOS、Canary-1B。原因是单模型/单脚本链路清晰，Canary-1B 已形成较完整的可交付推理文档，可作为其他模型 README/评测脚本的标准模板。
2. **第二批落地**：speechscorer、MolFormer。原因是已固定版本并完成静态适配或 CPU 实测，补 NPU 对齐即可闭环。
3. **暂缓/专项攻关**：Hy3-preview、MiroThinker-1.7、BUTSpeechFIT-DiariZen、MMAudio、MOSS-TTSD-v0.5。原因是大模型多卡、复杂依赖栈或生成质量主观评价，需专项资源；MOSS-TTSD-v0.5 已补 patch 与文档，但大权重、NPU 实推和生成质量验收仍需专项资源。

---

## 0. 统一评价口径

### 0.1 复杂度标签

| 标签 | 含义 |
|---|---|
| 低 | 已有可直接执行脚本或清晰命令；主要工作是下载权重、替换路径、小样本冒烟和补文档。通常 0.5-2 人日可完成功能闭环。 |
| 中 | 需要补 CLI、批处理、评测脚本、数据准备脚本、上游集成或依赖版本修正。通常 2-5 人日。 |
| 高 | 需要多仓联调、多处源码补丁、复杂依赖/编译、大模型或主观评价方案。通常 5-10 人日。 |
| 很高 | 存在重大阻塞，如大文件未拉取、多卡/专用容器、上游代码大量 CUDA 假设、评价体系主观且需人工听测。通常超过 10 人日或需专项资源。 |

### 0.2 每个仓库交付件要求

后续每个仓库统一按 6 个子标题描述：

1. **仓库观察与判断依据**：说明当前仓库是否有脚本、patch、README、权重/数据说明、上游依赖和适配风险。
2. **后续适配**：说明在当前适配基础上还要做什么，是否需要补 CLI、patch、数据准备、评测脚本和推理指导文档。
3. **功能验证**：说明是否已有功能验证脚本，验证数据从哪里来、如何获取，权重如何获取，验收看哪些输出。
4. **性能验证**：参考源仓库/论文/官方模型卡或同 checkpoint 官方实现，至少选一个公开数据集与原仓库 CPU/CUDA 路径做对比；说明是否已有性能脚本，数据来源和生成方式。
5. **精度验证**：参考源仓库/论文/官方模型卡指标，至少选一个公开数据集与原仓库 CPU/CUDA 路径做对比；说明是否已有精度脚本，数据来源和生成方式。
6. **数据集获取**：汇总功能、性能、精度所需数据 URL、下载工具、可离线复用要求和注意事项。

### 0.3 统一工程标准

- 推理入口建议统一命名为 `infer.py` 或 `infer_npu.py`；评测入口建议统一命名为 `eval_*.py` 或 `benchmark.py`；数据准备入口建议统一命名为 `prepare_eval_data.py`。
- 数据准备必须支持“指定目录、存在即复用、缺失才下载、`--offline` 下不联网”。
- 权重下载必须写明官方 URL/ModelScope/Hugging Face 模型名、目标目录和完整性校验方式。
- 精度/性能对比不能只写"跑通"，必须写明对比对象：同 checkpoint、同数据、同评测脚本下，优先与公开/官方指标对比；无公开指标时与 CPU 精度对照结果对比（`--device cpu`，复用同一环境）；性能以 NPU 实测为准，不使用 CPU 性能数据推断加速比。
- 不添加未验证的 CPU fallback、远程下载 fallback、非官方指标替代官方指标；缺失依赖或缺失官方字段应快速失败并暴露原始错误。

### 0.4 FlashAttention / SDPA 适配结论

具体迁移原则、官方仓参考类型和链接统一维护在《模型 NPU 适配标准流程》的
“5.1 FlashAttention / SDPA 迁移原则”。本项目级索引不再复制完整规则；正式适配前仍需按当前日期复查目标仓、目标 CANN / `torch-npu` / vLLM-Ascend 版本和同类近期模型。

---
## 1. DNSMOS

### 1.1 仓库观察与判断依据

- 随仓包含 `DNSMOS/infer.py`，README 中也粘贴了完整推理脚本，当前功能链路最完整。
- 适配后端是 `onnxruntime` 的 `CANNExecutionProvider`，不是简单 `cuda -> npu` 字符串替换。
- 脚本已有 NPU 环境检查、`npu-smi info`、性能模式切换、批量处理和 CSV 输出。
- README 已给出 VCC2018 converted speech 下载命令：`https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_submitted_systems_converted_speech.tar.gz`。
- 权重要求来自 DNSMOS 官方 ONNX 文件：`DNSMOS/model_v8.onnx`、`DNSMOS/sig_bak_ovr.onnx`，可选 `pDNSMOS/sig_bak_ovr.onnx`。

### 1.2 后续适配

- 复杂度：低。
- 在当前脚本基础上补 `--device_id`、`--disable_performance_mode`、权重完整性检查和更清晰的错误信息。
- 固化最小依赖：`onnxruntime-cann`、CANN、`numpy`、`librosa`、`soundfile`、`pandas`，不要把完整开发环境冻结成部署依赖。
- 增加 CPU ONNXRuntime baseline 入口或 `--provider CPUExecutionProvider`，用于 NPU/CANN 输出一致性对比。
- 形成 Canary-1B 风格的 `README.md`：环境表、权重下载、数据准备、单条/批量推理、性能/精度命令。

### 1.3 功能验证

- 已有验证脚本：有，`DNSMOS/infer.py` 可直接做功能验证。
- 验证数据来源：README 推荐 VCC2018 converted speech；冒烟可用任意 16 kHz/48 kHz WAV 或自录 WAV。
- 数据获取：`wget` 下载 VCC2018 压缩包后解压到 `DNSMOS/dataset/vcc2018`。
- 权重获取：从 DNSMOS 官方仓库或当前 README 指定的 DNSMOS 目录准备 ONNX 权重，放到 `--model_root` 下的 `DNSMOS/`、可选 `pDNSMOS/`。
- 验证命令：`python infer.py -t ./dataset/vcc2018 -o ./csv/vcc2018.csv --model_root . --batch_size 4`。
- 验收：CSV 包含 `filename,len_in_sec,MOS_SIG,MOS_BAK,MOS_OVRL,P808_MOS`，MOS 在 0-5 合理范围，短音频 padding、长音频分段和批量目录均不报错。

### 1.4 性能验证

- 已有性能脚本：部分已有；`infer.py` 有批处理和耗时统计基础，但还需明确端到端/纯 ONNX 推理分段计时。
- 对比对象：同一 DNSMOS ONNX 权重在源仓 CPU ONNXRuntime 与 NPU CANNExecutionProvider 的吞吐/延迟。
- 对比数据集：至少使用 VCC2018 converted speech 中固定 100/1000 条 WAV；如需要更稳定，可额外构造 9 秒、30 秒、60 秒三档音频清单。
- 数据生成：从 VCC2018 解压目录扫描 WAV，生成固定 manifest；不要每次随机抽样。
- 指标：端到端耗时、纯推理耗时、音频秒/秒、平均单文件耗时、P95、HBM/RSS、batch size 1/2/4/8。
- 参考源仓：DNSMOS 官方实现通常以 CPU/ONNX 推理为参考；NPU 通过线应是同权重同数据下输出一致且吞吐优于或不低于 CPU baseline。

### 1.5 精度验证

- 已有精度脚本：无独立精度脚本，需要新增 `eval_dnsmos.py` 或在 `infer.py` 增加 baseline 对齐模式。
- 对比对象：官方 DNSMOS ONNXRuntime CPU 输出；如果有主观 MOS 标注，再对比 DNSMOS 论文/官方报告的 MOS 相关性口径。
- 对比数据集：最低要求用 VCC2018 固定子集做 NPU vs CPU 数值一致性；正式相关性需带人工 MOS 标签的数据，不能用无标注 WAV 直接宣称 MOS 精度。
- 指标：`MOS_SIG/MOS_BAK/MOS_OVRL/P808_MOS` 的 MAE、最大绝对误差、Pearson/Spearman；建议 MAE < 1e-3，CANN 浮点差异较大时可放宽到 1e-2 并说明。
- 验收：NPU 与 CPU 排序相关 > 0.99，无 NaN、无空 CSV、同一输入重复运行结果稳定。

### 1.6 数据集获取

- VCC2018 converted speech：`https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_submitted_systems_converted_speech.tar.gz`。
- 冒烟数据：任意公开 WAV 或本地生成 1/9/30 秒正弦/语音 WAV。
- 权重：DNSMOS 官方 ONNX 文件，目标结构为 `DNSMOS/model_v8.onnx`、`DNSMOS/sig_bak_ovr.onnx`、可选 `pDNSMOS/sig_bak_ovr.onnx`。
- 建议规模：冒烟 10 条；性能 1000 条；精度一致性 500-1000 条。

---
## 2. MMAudio

### 2.1 仓库观察与判断依据

- 仓库主要是 README 和截图，`mmaudio.tar.gz` 是 Git LFS 指针，真实内容约 10GB，当前未拉取。
- README 指向上游 `https://github.com/hkchengrex/MMAudio`，可选 Gitee 镜像 `https://gitee.com/MufcLiuKai/MMAudio`。
- 适配涉及多处手工修改：`cuda` 改 `npu`、CLIP 本地模型、VAE、BigVGAN filter/resample/bigvgan dtype 等。
- 需要 apple CLIP、nvidia BigVGAN 等额外模型，README 还要求 G8600/910B2C 2 卡。
- 任务是视频/文本到音频生成，功能和精度验证都依赖主观/客观生成质量评估。

### 2.2 后续适配

- 复杂度：高。
- 首先拉取 Git LFS 大包或改为明确 patch，不应依赖截图交付。
- 将 README 修改整理成统一 patch，覆盖 `demo.py`、`features_utils.py`、`vae.py`、BigVGAN alias-free activation 等。
- 增加 `--device npu:0`、`--dtype float32/bf16`、`--clip_dir`、`--bigvgan_dir`、`--output` 等参数。
- 明确单卡是否可运行；如果必须 2 卡，需要说明模块切分和资源要求。
- 建立 op/dtype 不支持清单，避免静默 CPU fallback。

### 2.3 功能验证

- 已有验证脚本：README 指向上游 demo，但当前仓库缺少完整可执行代码包和本地 patch。
- 验证数据来源：手写文本 prompt、短视频样例；上游 MMAudio demo assets 如可用应优先使用。
- 权重获取：apple/DFN5B-CLIP-ViT-H-14-378 可按 README 从 GitCode/ModelScope 获取；MMAudio/BigVGAN 权重按上游 README 或 ModelScope/HF 下载。
- 验证内容：文生音频、视频生音频、空 prompt/长 prompt/短视频/长视频异常输入。
- 验收：输出 WAV/视频音轨存在，采样率和时长符合参数，无全零、爆音、NaN，日志无严重 NPU op 报错。

### 2.4 性能验证

- 已有性能脚本：无，需要新增 `benchmark_mmaudio.py`。
- 对比对象：源仓 MMAudio CUDA 路径或论文/官方 demo 在同模型同 seed 下的耗时；若源仓没有硬件表，则以本地 CUDA/CPU 同数据为基线。
- 对比数据集：至少选 AudioCaps 或 VGGSound/Clotho 子集中的 50 条 prompt/视频；小规模可先构造 10 个固定 5s/10s/30s prompt/视频。
- 数据生成：统一视频分辨率、音频目标时长、seed、采样步数和模型配置，生成 manifest。
- 指标：RTF、端到端延迟、CLIP/BigVGAN/VAE 分段耗时、2 卡利用率、显存峰值；记录 float32 替换 bf16 后的性能损失。

### 2.5 精度验证

- 已有精度脚本：无，需要新增 `eval_mmaudio.py`。
- 对比对象：源仓 CUDA 输出和论文推荐指标；生成式音频不能以逐点波形作为唯一标准。
- 对比数据集：至少使用 AudioCaps 或 Clotho 做文本-音频一致性；视频任务可用 VGGSound 子集。
- 指标：CLAPScore、FAD、mel L1/频谱差异、人工 MOS/A-B 偏好；视频输入还需评估音画一致性。
- 验收：NPU 相比源仓 CUDA 在 CLAPScore/FAD 和人工抽听上无明显退化；dtype 改动不引入系统性杂音。

### 2.6 数据集获取

- 上游代码：`https://github.com/hkchengrex/MMAudio`；备选镜像 `https://gitee.com/MufcLiuKai/MMAudio`。
- CLIP：`apple/DFN5B-CLIP-ViT-H-14-378`，README 给出 GitCode 和 ModelScope 下载方式。
- 精度/性能数据：AudioCaps、Clotho、VGGSound；注意版权、下载体积和 YouTube 链接失效问题。
- 建议规模：冒烟 10 条；性能 50 条；精度 200-1000 条或人工抽检 50 条。

---
## 3. Canary-1B

### 3.1 仓库观察与判断依据

- 当前已形成较完整交付件：`README_old.md`、`README.md`、`infer.py`、`eval_canary.py`、`prepare_eval_data.py`。
- README 明确上游 NeMo commit `44cb1c7ac5cbe6fc38ecc6184a174a02e7abadbe`，模型权重为 Hugging Face `nvidia/canary-1b` 的 `canary-1b.nemo`，并记录 SHA256。
- 当前适配不修改 NeMo 上游文件，没有 `.patch`；推理脚本默认 `--device npu`，CPU 验证使用 `--device cpu`。
- 已明确 `ASCEND_RT_VISIBLE_DEVICES` 控制卡号，不写死 `npu:0`。
- README 已包含官方精度表、Open ASR Leaderboard 性能参考、MLS/LibriSpeech/FLEURS 数据准备和在线/离线复用方案，是其他仓库的参考模板。

### 3.2 后续适配

- 复杂度：中，当前已基本完成工程化。
- 后续主要是把实际 NPU 环境运行结果补回 README：CANN/驱动/torch_npu 版本、batch size、beam size、RTF/RTFx、WER/BLEU。
- 补充 NPU 失败样例和常见错误排查，例如 NeMo 版本、`.nemo` 权重路径、`torchcodec`/音频解码依赖。
- 如需发布到统一 ModelZoo 目录，保持 `README.md` 的路径命令不依赖本地绝对路径。

### 3.3 功能验证

- 已有验证脚本：有，`infer.py` 支持单条/多条音频；`eval_canary.py` 支持 manifest。
- 验证数据来源：单条 smoke 使用 PyTorch torchaudio tutorial WAV；正式数据使用 LibriSpeech、MLS、FLEURS。
- 权重获取：`https://huggingface.co/nvidia/canary-1b/resolve/main/canary-1b.nemo` 或 `huggingface_hub.snapshot_download("nvidia/canary-1b")`。
- 验证命令：README.md 中已给出 ASR/AST 单条命令，参数包括 `--task --source_lang --target_lang --pnc --batch_size --beam_size`。
- 验收：ASR/AST 输出文本非空，语言/任务参数生效，manifest 批量可运行，CPU/NPU 都能加载同一 `.nemo` 权重。

### 3.4 性能验证

- 已有性能脚本：有，`eval_canary.py --performance_mode`。
- 对比对象：Hugging Face Open ASR Leaderboard 的 `nvidia/canary-1b` A100 公开 RTFx，以及同 checkpoint 本地 CPU/CUDA/NeMo 路径。
- 对比数据集：至少使用 LibriSpeech `test-clean`；该数据在 README.md 中通过 `prepare_eval_data.py --task librispeech` 准备。
- 指标：`elapsed_seconds`、`rtf`、`RTFx=audio_seconds/elapsed_seconds`、batch size、beam size、峰值 HBM/RSS。
- 注意：Open ASR Leaderboard 的 A100 RTFx 只能作为公开 GPU 量级参考，不应直接作为 NPU 通过线。

### 3.5 精度验证

- 已有精度脚本：有，`eval_canary.py`。
- 对比对象：NVIDIA model card 官方 ASR WER/AST BLEU 表、Open ASR Leaderboard WER，以及本地 CPU/CUDA 同 checkpoint 结果。
- 对比数据集：ASR 使用 MLS/LibriSpeech；AST 使用 FLEURS，必要时 CoVoST-v2。
- 数据生成：`prepare_eval_data.py` 支持在线下载、本地目录复用和 `--offline` 禁止联网。
- 指标：ASR WER/CER，AST BLEU/chrF，NPU vs CPU 文本一致率；正式精度建议 `beam_size=5`、`length_penalty=1.0` 对齐官方口径。

### 3.6 数据集获取

- 权重：Hugging Face `nvidia/canary-1b`，文件 `canary-1b.nemo`。
- Smoke WAV：`https://download.pytorch.org/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav`。
- LibriSpeech：`https://www.openslr.org/12`。
- MLS：`https://huggingface.co/datasets/facebook/multilingual_librispeech`。
- FLEURS：`https://huggingface.co/datasets/google/fleurs`。
- 建议规模：冒烟 1-10 条；性能 LibriSpeech test-clean；精度 MLS/FLEURS 按语种子集或全量。

---
## 4. BUTSpeechFIT-DiariZen

### 4.1 仓库观察与判断依据

- 仓库主要为 README，没有随仓 `infer.py`。
- 依赖上游 `https://github.com/BUTSpeechFIT/DiariZen`、Hugging Face `BUT-FIT/diarizen-wavlm-large-s80-md`、submodule/工具 `dscore`。
- README 要安装/编译多套 pyannote 相关包，并修改 torchaudio kaldi complex abs。
- 还涉及 numpy `np.NaN` 兼容和本地模型软链接路径问题。
- 依赖较重、下载来源较多、工程化程度偏低。

### 4.2 后续适配

- 复杂度：高。
- 补 `infer_npu.py` 和 `eval_der.py`，支持 wav 输入、RTTM 输出、参考 RTTM 评估。
- 固定 DiariZen、pyannote、dscore 版本，形成可复现 `requirements` 和 patch。
- 将 torchaudio/numpy 修改变成 patch 或版本约束。
- 去除软链接中个人路径假设，全部改为 `--model_dir` 和配置文件。
- DiariZen 作为专项推进；依赖 pyannote-audio、dscore 组合，需固定版本并整理 patch。

### 4.3 功能验证

- 已有验证脚本：无随仓脚本；README 参考 Hugging Face 使用说明创建 `infer.py`。
- 验证数据来源：DiariZen example，例如 `example/EN2002a_30s.wav`，以及自制 2/3/4 说话人样本。
- 权重获取：`git clone https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md.git`，必要时设置 `HF_ENDPOINT=https://hf-mirror.com`。
- 验证内容：冒烟输出 speaker turns、RTTM 格式、重叠语音和多说话人。
- 验收：RTTM 可被 dscore 读取，结果非空，说话人数合理，缺模型或路径错误快速失败。

### 4.4 性能验证

- 已有性能脚本：无。
- 对比对象：DiariZen 官方/源仓 CUDA 路径，或同 checkpoint 本地 CPU/CUDA 结果。
- 对比数据集：至少使用 AMI 或 VoxConverse 子集；性能用 30s/5min/30min 会议音频。
- 数据生成：准备 wav manifest 和输出 RTTM 目录，固定聚类/阈值参数。
- 指标：RTF、segmentation/embedding/clustering 耗时、显存、CPU 后处理占比。

### 4.5 精度验证

- 已有精度脚本：无，需要 `eval_der.py` 调 dscore 或 pyannote.metrics。
- 对比对象：官方 DiariZen 报告或源仓 CPU/CUDA DER。
- 对比数据集：AMI、DIHARD、VoxConverse、AliMeeting 至少选一个。
- 指标：DER、JER、miss/FA/confusion，明确 collar/overlap 策略。
- 验收：NPU 与源仓同 checkpoint 输出差异可解释，DER 不显著退化。

### 4.6 数据集获取

- 上游代码：`https://github.com/BUTSpeechFIT/DiariZen.git`。
- 权重：Hugging Face `BUT-FIT/diarizen-wavlm-large-s80-md`。
- dscore：README 使用 `https://githubfast.com/nryant/dscore.git`，正式文档应尽量给官方 GitHub 地址和镜像备选。
- 数据：AMI、DIHARD、VoxConverse、AliMeeting；功能可用 DiariZen example。
- 建议规模：冒烟 1-5 条；精度 10-100 场会议；性能 10-50 小时。

---
## 5. MOSS-TTSD-v0.5

### 5.1 仓库观察与判断依据

- 2026-06-16 已重新检查 `https://github.com/OpenMOSS/MOSS-TTSD`：默认分支 `main` HEAD 为 `20dbb4fc44819435fee894d644a0402a0fee736a`，但当前顶层代码已面向 v1.0。
- v0.5 原项目代码有 tag `v0.5`，commit 为 `0e078c62389922d3aa873ce182daf31142860b18`，包含 `inference.py`、`generation_utils.py`、`XY_Tokenizer/`、`examples/` 等原始推理链路。
- 根据新约束，本目录不新增独立代码文件；MOSS-TTSD-v0.5 的代码适配收敛为 `patches/0001-adapt-v0.5-inference-to-npu.patch`。
- 当前 patch 修改原项目已有推理、模型、codec 和音频 I/O 文件；其中 `modeling_asteroid.py` 注册 NPU PFA/IFA GQA attention backend，避免 Transformers SDPA/eager 的 `repeat_kv` 实体展开。
- 当前适配边界：原项目 tag `v0.5` + `fnlp/MOSS-TTSD-v0.5` / `OpenMOSS-Team/MOSS-TTSD-v0.5` 权重 + 原项目 XY Tokenizer 代码 + `fnlp/XY_Tokenizer_TTSD_V0` 的 `xy_tokenizer.ckpt`。

### 5.2 后续适配

- 复杂度：中。已有 patch 和独立文档，但仍需真实 NPU 环境、权重下载和质量评测闭环。
- 下载权重后必须补充模型权重与 `xy_tokenizer.ckpt` 的 SHA256。
- 在目标 NPU 环境验证设备内部固定的 PFA/IFA 路径，并与同 checkpoint、同 manifest 的 CPU/CUDA 原始路径做结果对齐；推理 CLI 不增加 attention 参数，也不在代码中静默降级。
- 如果后续要适配 Gradio 或 podcast 路径，应继续基于原项目已有文件新增 patch，不新增旁路脚本。

### 5.3 功能验证

- 已有验证入口：原项目 `inference.py`，应用 patch 后通过 `--device npu/cpu/cuda` 选择设备。
- 验证数据来源：原项目 `examples/examples.jsonl`、真实中文/英文 prompt 和对话 JSONL。
- 验证内容：官方 examples、中文/英文/中英混合、双说话人 prompt、`--use_normalize`、CPU/NPU 对比。
- 验收：输出 WAV 数量正确、可播放、采样率/时长合理，无设备不一致、CUDA-only、静默 CPU fallback。

### 5.4 性能验证

- 当前没有新增 benchmark 脚本；按约束先复用原项目 `inference.py`，用外部计时或日志统计 elapsed、输出总时长、RTF/RTFx。
- 对比对象：同 checkpoint、同 JSONL、同参数的 CPU/CUDA 源路径。
- 数据规模：功能验证使用官方 examples 2 条；L2 使用 TTSD-eval 中文/英文全量
  各 50 条。
- 指标：RTF、RTFx、首条输出延迟、峰值 HBM/RSS、最大可用 batch、首次加载耗时与稳定推理耗时。

### 5.5 精度/质量验证

- 正式质量指标：ASR 回识别 CER/WER、speaker embedding cosine、DNSMOS/UTMOS、人工 MOS/CMOS/A-B preference；如使用官方 TTSD-eval，需固定其 repo/commit 和指标配置。
- 通过条件：NPU 与 CPU/CUDA 源路径相比，在可懂度、音色、自然度、说话人切换准确性上无显著退化。
- 不以“生成了 WAV”替代正式音质验收。

### 5.6 数据集获取

- 上游代码：`https://github.com/OpenMOSS/MOSS-TTSD` tag `v0.5`。
- 模型：`https://huggingface.co/fnlp/MOSS-TTSD-v0.5`；同内容别名 `https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v0.5`；本次记录 HEAD `8527b9136b6afefe2252ae597cecea2e80e7ebeb`。
- Codec：原项目 `XY_Tokenizer` 代码 + `https://huggingface.co/fnlp/XY_Tokenizer_TTSD_V0` 的 `xy_tokenizer.ckpt`；本次记录 HEAD `c83433728e698ed0698e88cb5096bc221fb8f8c5`。
- 下载命令和三工作树布局以 `MOSS-TTSD-v0.5/README.md` 为准。
  固定资产命令示意：

  ```bash
  python -m pip install -U "huggingface_hub[cli]"
  mkdir -p weights/MOSS-TTSD-v0.5 XY_Tokenizer/weights

  hf download fnlp/MOSS-TTSD-v0.5 \
    --revision 8527b9136b6afefe2252ae597cecea2e80e7ebeb \
    --local-dir weights/MOSS-TTSD-v0.5

  hf download fnlp/XY_Tokenizer_TTSD_V0 xy_tokenizer.ckpt \
    --revision c83433728e698ed0698e88cb5096bc221fb8f8c5 \
    --local-dir XY_Tokenizer/weights
  ```
- 也可用固定 URL 直接下载 codec checkpoint：`https://huggingface.co/fnlp/XY_Tokenizer_TTSD_V0/resolve/c83433728e698ed0698e88cb5096bc221fb8f8c5/xy_tokenizer.ckpt`。下载后需记录模型权重与 `xy_tokenizer.ckpt` SHA256。
- 功能数据：原项目 `examples/examples.jsonl`、官方示例 prompt、自建真实双语对话 JSONL。
- 精度/性能数据：固定 `OpenMOSS/TTSD-eval` commit 的中文/英文全量各 50 条，
  同时计算 ACC/SIM/WER 和 RTF/RTFx；其他数据只作补充。

---
## 6. 综合优先级与落地建议

### 6.1 优先级排序

| 优先级 | 仓库 | 原因 |
|---|---|---|
| P0 | DNSMOS | 已有 CANNExecutionProvider 推理脚本，权重/数据/CSV 输出链路短，最快形成 Canary-1B 风格交付。 |
| P1 | Canary-1B | 已完成较完整输出件，可作为其他仓库 README/评测脚本标准模板。 |
| P3 | BUTSpeechFIT-DiariZen | 依赖链重（pyannote-audio/dscore），当前偏文档，需补 patch 与 DER 验收。 |
| P3 | MMAudio | 大包、2 卡、多模态生成、质量评价均复杂。 |
| P3 | MOSS-TTSD-v0.5 | 已按 patch 方式适配原项目 v0.5 推理链路；大权重、NPU 实测与主观/客观质量评价仍需专项资源。 |

### 6.2 建议执行节奏

1. 第一批：DNSMOS、Canary-1B。目标是统一产出 `infer.py`/`prepare_eval_data.py`/`eval_*.py`/`README.md`，并补真实 NPU 运行结果。
2. 第二批：speechscorer、MolFormer。目标是补 NPU 同 manifest 精度/性能对齐。
3. Canary-1B 作为模板维护：后续只补真实 NPU 运行结果、环境版本和常见问题。
4. 专项批：Hy3-preview、MiroThinker-1.7、BUTSpeechFIT-DiariZen、MMAudio、MOSS-TTSD-v0.5。MOSS-TTSD-v0.5 下一步优先补权重 SHA256、CPU/NPU 实推日志和质量评测；Hy3-preview/MiroThinker-1.7 需 16 卡大模型服务与 agent benchmark；BUTSpeechFIT-DiariZen/MMAudio 需复杂依赖与生成质量评估。

### 6.3 对所有仓库的共同要求

- 每个仓库的功能验证必须写明：已有脚本、验证数据 URL/生成方式、权重 URL/下载命令、运行命令、验收输出。
- 每个仓库的性能验证必须至少选一个数据集，与源仓 CPU/CUDA/官方报告做同 checkpoint、同数据、同参数对比。
- 每个仓库的精度验证必须至少选一个公开数据集或明确人工评测协议，与源仓 CPU/CUDA/官方指标做对比。
- 数据准备脚本必须支持离线复用，禁止在评测阶段隐式联网下载。
- 缺少官方指标或官方组件时应在文档中明确“缺失/待补”，不能用简化指标或第三方非官方实现冒充官方评测。
