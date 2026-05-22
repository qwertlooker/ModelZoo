# Ascend-SACT 语音/音频模型 NPU 适配仓库静态分析

分析日期：2026-05-22

说明：本报告基于 `/home/pei/ModelZoo` 下已克隆仓库的 README、脚本、requirements 与仓库文件结构做静态分析；当前环境未提供 Ascend NPU/CANN，未做真实运行验证。MMAudio/whisper-large-v3 中的大 tar.gz 为 Git LFS 指针，未拉取实际大文件。

| 仓库 | 适配完整度 | 后续适配工作量 | 功能验证工作量 | 性能验证工作量 | 精度验证工作量 | 数据/权重获取难度 | 主要判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| DNSMOS | 高 | 低 | 低 | 低-中 | 中 | 低-中 | 已有 CANNExecutionProvider 推理脚本、批处理、结果 CSV；权重和 VCC2018 等数据较易获取。 |
| BEATs | 中-高 | 低-中 | 低 | 中 | 中-高 | 中 | 有替换版 BEATs.py 和 infer_npu.py；demo 易跑，完整分类精度需 AudioSet/ESC-50 等标准集。 |
| FireRedASR-AED | 中 | 中 | 低-中 | 中 | 中 | 中 | 简单迁移脚本，中文 ASR 数据集可得；需补批量、路径参数化、WER/CER 评估。 |
| Canary-1B | 中 | 中 | 中 | 中 | 中 | 中-高 | NeMo 生态复杂但脚本简单；多语言 ASR/翻译验证数据准备比中文 ASR 稍复杂。 |
| pyannote-speaker-diarization-3.1 | 中 | 中 | 中 | 中 | 中-高 | 中-高 | README 含本地模型配置、NPU 转移与已知补丁；DER 标准验证集和依赖版本是主要成本。 |
| MossFormer2_SE_48K | 中-低 | 中 | 低-中 | 中 | 中 | 中 | 主要是部署指导/示例，依赖 ClearerVoice；功能容易，PESQ/STOI 等精度验证需成对干净/带噪数据。 |
| whisper-large-v3 | 中-低 | 中 | 低-中 | 中 | 中 | 低-中 | README 内有较完整批量 infer.py，但仓库实际只有 LFS 大包指针；ASR 数据与 WER/CER 验证较成熟。 |
| Index-TTS-2 | 中 | 高 | 中-高 | 中 | 高 | 中-高 | 有 infer_v2.py/run_web.sh 和 RTF 结果描述，但依赖/编译/多权重复杂，TTS 精度评价偏主观。 |
| MOSS-Speech | 中-低 | 高 | 高 | 中-高 | 高 | 高 | 依赖 MOSS-Speech、Codec、Space 和多个第三方包补丁，仍有 cuda/device_map 字样依赖 transfer_to_npu。 |
| MMAudio | 低-中 | 高 | 中-高 | 中-高 | 高 | 高 | 多处手工改 cuda/npu 与 dtype，2 卡要求；仓库大包为 LFS 指针，验证生成音频质量成本高。 |
| BUTSpeechFIT-DiariZen | 低-中 | 高 | 中-高 | 中 | 中-高 | 高 | 主要是安装/补丁说明，无随仓 infer.py；pyannote/DiariZen/dscore 组合依赖重。 |
| MOSS-TTSD-v0.5 | 低-中 | 高 | 中-高 | 中-高 | 高 | 高 | 需多文件 cuda->npu、vLLM-Ascend 容器和大整合包；语音生成验证主观且链路复杂。 |

## 建议优先级

1. **第一批落地**：DNSMOS、BEATs、FireRedASR-AED、whisper-large-v3。原因是单模型/单脚本链路清晰，功能与性能验证容易快速闭环。
2. **第二批落地**：Canary-1B、pyannote-speaker-diarization-3.1、MossFormer2_SE_48K。原因是有明确部署路径，但依赖栈或标准数据集验证成本更高。
3. **暂缓/专项攻关**：Index-TTS-2、MOSS-Speech、MMAudio、BUTSpeechFIT-DiariZen、MOSS-TTSD-v0.5。原因是多仓库/多权重/多文件补丁/主观质量评价，集成和回归成本较高。
