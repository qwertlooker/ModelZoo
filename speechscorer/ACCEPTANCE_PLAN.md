# speechscorer 验收计划

## 0. 版本边界

- speechscorer：`main@bbe0be772b37f472994d5a97f809214fd67a2c8e`
- NPU patch：`patches/0001-add-explicit-device-selection.patch`
- 原始公开结果主路径：`hubert-mlm`
- checkpoint：官方 notebook 下载的 `hubert_large_ll60k.pt`
- processor：`facebook/hubert-large-ls960-ft@ece5fabbf034c1073acae96d5401b25be96709d8`
- SpeechOcean762：`main@613968e3b0b789fc33936fb5eba1973176ba7d11`
- `whisper-clm` 是上游默认 smoke 路径，不能替代 HuBERT-MLM 的原始公开图对齐。

## 1. 原始测试集和官方指标

upstream 官方 demo 使用 [SpeechOcean762](https://github.com/jimbozhang/speechocean762) 的 `test` split：

- 语料总计 5,000 条英语句子；train/test 各 2,500 条，主验收使用 test 2,500 条；
- 音频：16 kHz，母语为普通话的儿童和成人英语学习者朗读语音；
- 人工标签：`accuracy`、`completeness`、`fluency`、`prosodic` 和 `total`；
- upstream scorer 输出：utterance 级 entropy 和 perplexity；
- README 中的公开图及 notebook 全量 SpeechOcean 实验使用
  `hubert_large_ll60k.pt`、`hubert-mlm`、`padding=longest` 和
  `facebook/hubert-large-ls960-ft` processor；
- 该路径不做文本 normalizer；HuBERT masked-token logits 经 softmax 后计算平均
  entropy，并取 `exp(entropy)` 得到 perplexity；
- 上游 notebook 将结果和人工标签合并后按 `age` 分组绘图，但没有发布固定数值表。

来源：

- <https://github.com/yaya-sy/speechscorer>
- <https://github.com/jimbozhang/speechocean762>

speechscorer upstream 未发布该固定 checkpoint 下的 Pearson/Spearman 数值、
通过阈值或完整结果 CSV，因此必须明确写“官方数值指标未发布”。VCC2018
不是原始评测集，不能用于宣称复现官方评分能力。

## 2. 迁移对齐

固定：

- speechscorer commit 和 patch；
- HuBERT checkpoint SHA256 和 processor revision；
- SpeechOcean762 `test/wav.scp` 顺序和音频；
- batch size、padding、max length、Transformers/PyTorch 版本。

必须保存未应用 patch 的原始 CPU/CUDA、应用 patch 后的同设备回归、NPU 三组
CSV。先证明原始与 patch 后同设备无非预期变化，再按 `utterance_id` 比较
patch 后 CPU/CUDA 与 NPU：

- 样本集合完全一致；
- entropy 最大绝对误差 `<= 1e-4`、平均绝对误差 `<= 1e-5`；
- perplexity 相对误差 `<= 1e-4`；
- CPU/CUDA 与 NPU entropy 排序 Spearman `>= 0.9999`。

这些是暂定迁移门禁，不是 upstream 官方容差。首次 HuBERT 全量 baseline 应先报告
同设备重复运行波动，再确认阈值是否合理。

然后将预测与 `test/all-info.json` 的人工 `total` 合并：

- 按 notebook 原步骤报告 `groupby(age).mean()` 后的 `total/entropy/perplexity`
  表，用于复现公开散点图输入；
- 另报告 utterance-level Pearson/Spearman，作为本项目新增的迁移诊断；
- NPU 与 CPU/CUDA 的 utterance-level 相关系数差绝对值应 `<= 0.001`。

上述相关性都是当前复现实测值，不得标注为 upstream 官方发布值。

执行入口：

```bash
python prepare_eval_data.py \
  --dataset_dir eval_data/speechocean762 \
  --output_dir eval_data/speechocean762-test
python evaluate_results.py \
  --results results/npu.csv \
  --baseline results/patched_cpu.csv \
  --manifest eval_data/speechocean762-test/manifest.jsonl \
  --output results/cpu_vs_npu.json
```

## 3. 功能验证与 L2

| 层级 | 数据 | 结论范围 |
|---|---|---|
| 功能验证 | 1 条 16 kHz 英语 WAV；分别检查 `whisper-clm` 和 `hubert-mlm` 入口 | 链路、输出字段和失败用例 |
| L2 | 原始公开 SpeechOcean762 test 全量 2,500 条，`hubert-mlm` | 三组数值对齐、排序、人工分数相关性和 RTF/吞吐 |

## 4. 功能矩阵

| 维度 | 必测值 |
|---|---|
| scorer | `hubert-mlm` 原始主线；`whisper-clm` 单独 smoke |
| 输入 | 单文件、固定目录、16 kHz/非 16 kHz |
| batch | 1/4/8，资源允许时 16 |
| padding | `longest`（原始路径） |
| 异常 | 空目录、缺 checkpoint、缺 fairseq、NPU 后端缺失 |

## 5. L2 性能验证

三组全量命令都用 `/usr/bin/time -v -o <独立日志>` 包裹。记录模型加载在内的 wall
time、总音频时长、RTF、samples/s、batch 1/4/8/16 和峰值 RSS/HBM；至少对正式
batch 重复 3 次并报告中位数。所有性能运行必须仍产生完整 CSV，并通过同一精度比较。

upstream 未发布固定硬件性能值，因此不声明伪造的官方 speedup。最低要求是全量
2,500 条无失败、指标通过、三组性能日志完整，并报告 NPU 相对 patch 后 CPU/CUDA
的 RTF/吞吐比；项目另有性能目标时再执行该目标。

## 6. 最低正式验收清单

- [ ] HuBERT checkpoint 和 processor revision 已固定。
- [ ] fairseq、Transformers、模型入口和 NPU tensor 导入测试通过。
- [ ] SpeechOcean762 test manifest/metadata 生成并归档。
- [ ] CPU/CUDA、patch 后 CPU/CUDA、NPU 分别写入独立 CSV。
- [ ] 原始与 patch 后同设备比较报告通过，两个 editable 源码没有安装到同一环境。
- [ ] 全量 2,500 条数值误差、排序和人工 `total` 相关性已报告。
- [ ] `whisper-clm` 结果没有被标注为 HuBERT 原始公开路径结果。
- [ ] 三组 L2 的 wall time、RTF、samples/s、峰值 RSS/HBM 和相对比值已归档。

## 7. 当前验收状态

- 已通过：patch 在固定 upstream commit 上 `git apply --check`；应用后全
  `speechscorer` 包 `compileall`；设备回退模式静态扫描。
- 未执行：HuBERT 权重下载、功能验证、SpeechOcean762 2,500 条 CPU/CUDA 与
  NPU 数值/相关性和性能验收。
- 当前结论：源码适配静态门禁通过；迁移精度未验收。

## 8. 报告模板

```text
源码/patch/checkpoint SHA:
环境、NPU、CANN、torch/torch-npu/transformers:
数据 split、样本数、manifest SHA:
decode/padding/batch:
CPU/CUDA vs NPU entropy/perplexity误差:
CPU/CUDA vs NPU排序Spearman:
各自与人工total的Pearson/Spearman:
三组wall time、RTF、samples/s、峰值RSS/HBM:
结论/未完成项:
```
