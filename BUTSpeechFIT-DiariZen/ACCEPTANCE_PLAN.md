# DiariZen 验收计划

## 0. 版本边界

- DiariZen：`main@a60b18151dbbe246e4199d8ef5cd2ece3872ea94`
- 目标权重：`diarizen-wavlm-large-s80-md@a9b1b0e7974d96dcfd63af417e9da7ad8714040f`
- dscore：`master@e02f949ac6592279300a2c33d03daf9e0c12fd27`
- Ascend-SACT 参考：`7961b5ab79b1232b9da367f14f8cd4f592694465`
- 不包含 large-s80-v2、base 或 pruning 权重。

## 1. 原始测试集与官方指标

upstream 对 `diarizen-wavlm-large-s80-md` 公布无 collar、保留 overlap、无数据集单独 domain adaptation、所有数据集共用 clustering 参数的 DER（%）：

| 数据集 | 官方 DER |
|---|---:|
| AMI-SDM | 14.0 |
| AISHELL-4 | 9.8 |
| AliMeeting far | 12.5 |
| NOTSOFAR-1 single-channel | 17.9 |
| MSDWild | 15.6 |
| DIHARD3 full | 14.5 |
| RAMC | 11.0 |
| VoxConverse | 9.2 |

特殊口径：AISHELL-4 先用 `sox in.wav -c 1 out.wav` 转单声道；NOTSOFAR-1 仅使用 single-channel 录音。来源：<https://github.com/BUTSpeechFIT/DiariZen>。

后处理为固定 checkpoint `config.toml` 中的 segmentation、median filtering、
speaker count、embedding 和 clustering 参数，输出 RTTM；metric 使用 dscore DER，
`collar=0.0` 且不忽略 overlap。该路径没有文本 normalizer 或 decode beam 参数。

upstream README 未发布上述每项的精确 split revision、样本规模、
wav.scp/RTTM/UEM SHA、完整评测命令和逐文件结果，因此这些字段明确记为
“官方未发布”。正式复现前必须从作者 recipe/数据许可确认并固定，不得自行猜测
manifest。

## 2. 迁移对齐主线

固定同一：

- DiariZen/pyannote/dscore commit 和 patch；
- segmentation、embedding、PLDA、config 文件 SHA；
- WAV、RTTM、UEM；
- clustering 参数、speaker 上下限、median filtering；
- batch 和音频预处理。

先在 CUDA 原始路径生成 RTTM，再在 NPU 生成 RTTM，使用 vendored dscore：

```bash
python3 upstream/dscore/score.py \
  -r reference.rttm -s system.rttm \
  --collar 0.0 --ignore_overlaps false
```

通过条件：

- NPU DER 相对 CUDA 绝对劣化 `<= 0.2` 个百分点；
- miss/false alarm/confusion 分项都报告；
- 同输入 RTTM session、时间轴范围和 speaker 数约束一致；
- 不允许 embedding ONNX session 使用 `CPUExecutionProvider` 冒充 NPU。

## 3. 分层验收

| 层级 | 数据 | 目标 |
|---|---|---|
| L0 | upstream `EN2002a_30s.wav` | 生成 RTTM；不作为 DER 结论 |
| L1 | 至少 10 个带 RTTM/UEM 的短 session | CUDA/NPU 分项 DER 对齐 |
| L2 | AMI-SDM 或 VoxConverse 固定公开 split | 全量迁移精度、性能、稳定性 |
| L3 | upstream 表中 8 个数据集 | 复现官方多域 DER 表 |

## 4. 性能与稳定性

记录总音频时长、wall time、RTF、分割/embedding 阶段耗时、batch、峰值 HBM/RSS、首次编译时间。至少连续处理 1 小时音频或 30 轮 L1，无崩溃和持续内存增长。

## 5. 当前验收状态

- 已通过：patch 在固定 DiariZen commit 上 `git apply --check`；应用后
  DiariZen 和修改的 pyannote 文件 `compileall`；ModelZoo `infer.py`
  语法检查。
- 未执行：主/embedding 权重下载、example RTTM、CUDA/NPU RTTM 对齐、
  官方数据 DER、性能和稳定性。
- 当前结论：源码适配静态门禁通过；NPU diarization 精度未验收。

## 6. 报告模板

```text
源码、patch、模型、embedding、PLDA SHA:
数据集/split/wav.scp/RTTM/UEM SHA与规模:
环境和硬件:
clustering/median/batch参数:
CUDA DER(miss/fa/confusion/total):
NPU DER(miss/fa/confusion/total):
差异、RTF、内存、稳定性:
provider核验:
结论和未完成项:
```
