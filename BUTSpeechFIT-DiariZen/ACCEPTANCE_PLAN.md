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

必须保留三组结果：

1. 未应用 patch 的固定 upstream CUDA 原始路径；
2. 应用 patch 后的 CUDA 回归路径，证明 CPU/CUDA 行为未改变；
3. 应用 patch 后的 NPU 路径。

三组使用同一 manifest、config 和权重。先用 `prepare_eval_data.py` 固定输入，再使用
vendored dscore：

```bash
python prepare_eval_data.py \
  --wav_scp eval_data/ami/wav.scp \
  --reference_rttm eval_data/ami/reference.rttm \
  --uem eval_data/ami/all.uem \
  --output_manifest eval_data/ami/manifest.jsonl \
  --dataset AMI --split SDM-eval
python score_diarization.py \
  --dscore_dir upstream/dscore \
  --reference_rttm eval_data/ami/reference.rttm \
  --system_rttm results/npu/*.rttm \
  --uem eval_data/ami/all.uem \
  --collar 0.0 \
  --output results/npu/der.txt
```

`--ignore_overlaps` 是 `store_true` 开关。官方口径保留 overlap 时必须完全省略该
参数，不能写成无效的 `--ignore_overlaps false`。

通过条件：

- NPU DER 相对 CUDA 绝对劣化 `<= 0.2` 个百分点；
- miss/false alarm/confusion 分项都报告；
- 同输入 RTTM session、时间轴范围和 speaker 数约束一致；
- 不允许 embedding ONNX session 使用 `CPUExecutionProvider` 冒充 NPU。

`0.2` 个百分点是暂定迁移门禁，不是 upstream 官方容差。正式 L2 必须先测量
原始 CUDA 与 patch 后 CUDA 的重复运行/聚类波动，再决定是否收紧或放宽。

## 3. 功能验证与 L2

| 层级 | 数据 | 目标 |
|---|---|---|
| 功能验证 | upstream `EN2002a_30s.wav` | 三组生成 RTTM、输出结构和失败用例；不作为 DER 结论 |
| L2 | 优先按 upstream 公布数据集全量和官方口径；数据许可受限时使用 AMI-SDM/VoxConverse 可取得公开 split | 全量三组 DER、RTF 和资源对齐 |

## 4. 功能矩阵

| 维度 | 必测值 |
|---|---|
| 路径 | 原始 CUDA、patch 后 CUDA、NPU |
| 输入 | 单文件、JSONL manifest、多 session |
| 音频 | mono、stereo/downmix、长短 session |
| 聚类 | checkpoint 默认配置；如切换 AHC/VBx 必须独立报告 |
| metric | collar=0、保留 overlap；其他口径单独标注 |
| 异常 | 缺 PLDA、缺 embedding、CANN EP 缺失、RTTM/UEM ID 不匹配 |

## 5. L2 精度与性能验证

精度优先选择 upstream 表中可取得的完整公开 split，并保持 `collar=0`、保留
overlap、同 clustering 配置；许可或精确 recipe 不可取得时固定 AMI-SDM 或
VoxConverse 公开 split，明确这是迁移对齐。

`infer.py` 的 `run.meta.json` 已记录音频总时长、elapsed 和 RTF。三组正式 L2
分别用 `/usr/bin/time -v` 补峰值 RSS，NPU 侧记录峰值 HBM；至少重复 3 次并报告
RTF 中位数。官方未发布可比硬件性能值，因此报告 NPU/CUDA RTF 比值，不伪造
speedup 通过线。

## 6. 最低正式验收清单

- [ ] 源码、patch、主模型、embedding、PLDA 和 dscore SHA 已固定。
- [ ] 依赖导入、NPU tensor 和 CANN provider 检查通过。
- [ ] CPU/NPU 分属独立环境，ONNX Runtime 版本分别为 1.22.1 CPU/CANN。
- [ ] 功能 example 在原始 CUDA、patch 后 CUDA与 NPU 生成 RTTM。
- [ ] 原始 CUDA与 patch 后 CUDA RTTM/DER 回归无非预期变化。
- [ ] dscore 使用 collar=0 且保留 overlap，命令和原始输出归档。
- [ ] `run.meta.json` 证明 NPU embedding provider 为 CANN。
- [ ] L2 正式 split 完成三组 DER 对齐，数据许可已记录。
- [ ] L2 三组 RTF、峰值 RSS/HBM 和相对比值已归档。

## 7. 当前验收状态

- 已通过：patch 在固定 DiariZen commit 上 `git apply --check`；应用后
  DiariZen 和修改的 pyannote 文件 `compileall`；ModelZoo `infer.py`
  语法检查。
- 未执行：主/embedding 权重下载、功能 RTTM 和 L2 DER/性能对齐。
- 当前结论：源码适配静态门禁通过；NPU diarization 精度未验收。

## 8. 报告模板

```text
源码、patch、模型、embedding、PLDA SHA:
数据集/split/wav.scp/RTTM/UEM SHA与规模:
环境和硬件:
clustering/median/batch参数:
CUDA DER(miss/fa/confusion/total):
NPU DER(miss/fa/confusion/total):
差异:
三组elapsed、RTF、峰值RSS/HBM及比值:
provider核验:
结论和未完成项:
```

## 补充说明（来自 README.md）

### 权重与数据许可

- 模型权重使用 CC BY-NC 4.0，正式部署前必须确认非商业使用及数据许可要求。
- 数据许可和 split 必须由使用者根据官方 recipe 固定；不能自行猜测后宣称复现官方表。

### 功能验证与正式 DER 对齐

- 功能样例无 reference RTTM，只比较原始与 patch 后 RTTM 是否完全一致，并确认 NPU
  成功生成 RTTM。正式 DER 对齐必须使用带 reference RTTM/UEM 的 L2 数据。
- 需要忽略 overlap 的独立非官方模式才显式增加 `--ignore_overlaps`。

### 性能评测方法

优先在 upstream 公布数据集的可取得全量 split 上记录总音频时长、RTF、
分割/embedding 阶段耗时、batch 和峰值 HBM/RSS。`infer.py` 会写
`run.meta.json`；三组命令分别用 `/usr/bin/time -v -o` 保存独立资源日志。官方
README 只发布 DER，未发布与当前 Atlas 路径可直接比较的硬件性能数值。

`results/npu/run.meta.json` 提供 elapsed/RTF/provider。原始 CUDA、patch 后 CUDA
使用同一 manifest 和独立输出目录/日志；正式轮次至少重复 3 次。

### 当前状态

| 项目 | 当前状态 |
|---|---|
| dscore 工具 fixture | collar=0、保留 overlap，DER/JER 0.00 |
| 功能验证模型 RTTM | 待权重环境实测 |
| CUDA/NPU DER | 待正式 reference RTTM/UEM |
| Atlas RTF/HBM | 官方未发布，尚未验收 |
