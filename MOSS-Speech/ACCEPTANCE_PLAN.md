# MOSS-Speech 分层验收计划

## 1. 验收目标

验证当前适配的 `openmoss/MOSS-Speech` + `MOSS-Speech-Codec` + HF Space `OpenMOSS-Team/MOSS-Speech` 在 NPU 上可按官方链路完成文本/语音响应生成，并与同 checkpoint 的 CPU/CUDA 参考结果保持功能和主观质量一致。

## 2. 分层方案

| 层级 | 数据规模 | 获取难度 | 验证内容 | 通过条件 |
|---|---:|---|---|---|
| L0 冒烟 | 1-2 条文本 prompt，官方 `prompt_cn.wav` / `prompt_en.wav` | 低 | 脚本参数、模型/codec 加载、文本或音频输出落盘 | 无异常；文本非空；音频文件可读、采样率正确、非全零/非 NaN。 |
| L1 功能 | 10 条中英文本 prompt，2 个 output modality | 低 | 中英问答、短/长 prompt、text/audio 两类输出 | 全部请求完成；错误输入能暴露明确异常；NPU 不出现系统性空输出。 |
| L2 质量 | 50 条固定 prompt + 2 个 decoder prompt audio | 中 | 文本相关性、语音可懂度、音色延续、截断/噪声检查 | 人工抽检无严重音色崩坏；ASR 回识别 CER/WER 相对 CPU/CUDA 参考无明显系统性退化。 |
| L3 回归/性能 | 100+ 条请求，包含长输出和批次化运行脚本 | 高 | 稳定性、端到端延迟、RTF、峰值 HBM/RSS、连续运行 | 无内存泄漏或随机崩溃；同参数 NPU 性能报告完整；质量不低于 L2 阈值。 |

## 3. 推荐数据

- 官方 prompt audio：`MOSS-Speech/upstream/assets/prompt_cn.wav`、`prompt_en.wav`。
- 自建 prompt：覆盖事实问答、开放式对话、中文诗歌、英文简答、长文本总结。
- 音频质量辅助：可将生成音频用项目内 ASR 模型或人工转写做 CER/WER；说话人相似度可使用固定 speaker embedding 模型，但不能替代人工听感结论。

## 4. 指标

### 功能指标

- 输出文本非空；
- 输出 wav 可被 `soundfile` / `torchaudio` 读取；
- 音频时长大于 0，幅值不是全零，无 NaN/Inf；
- `--device npu`、`--device cpu` 行为明确，不自动切换。

### 性能指标

- 端到端耗时；
- 首 token / 首音频耗时；
- 音频 RTF = 生成耗时 / 输出音频时长；
- 峰值 HBM、CPU RSS；
- 生成参数：temperature、top_p、top_k、max_new_tokens、min_new_tokens。

### 质量指标

- 文本人工相关性：0/1 或 1-5 分；
- 音频人工 MOS/CMOS；
- ASR 回识别 CER/WER；
- speaker embedding 相似度（如使用，需记录模型版本）。

## 5. 报告模板

```text
验收层级：L0/L1/L2/L3
日期：
模型/codec/space commit：
设备与软件版本：
命令：
输入集合：
输出目录：
成功数/总数：
文本样例：
音频样例、采样率、时长：
性能：平均耗时、P50/P90、RTF、峰值 HBM/RSS
质量：人工评分、CER/WER、异常样例
结论：通过/不通过
后续问题：
```
