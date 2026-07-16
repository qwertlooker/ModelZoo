# MechVL-4B-RL 验收方案与结果

## 验收对象

| 项目 | 固定值 |
|---|---|
| checkpoint | `XiaofengAlg/MechVL-4B-RL@2c6fda8a16e57d8a6fe1019412092d09a0363850` |
| target model ID | `MechVL-4B-RL` |
| upstream/evaluator | `xiaofengShi/MechVQA@8841ee083c2704f2d8ccf426a8c0bb61ad911890` |
| NPU 后端 | vLLM Ascend 0.18.0，BF16，单卡 |
| 上下文/批处理 token 上限 | 16,384 / 16,384 |
| 生成参数 | temperature 0.6，top-p 0.95，top-k 20，max tokens 4,096 |
| prompt | upstream `mech_r1` RL 后缀，要求 `&lt;think&gt;/&lt;answer&gt;` 格式 |

## 原始测试集

| 数据 | split | 数量 | 固定证据 |
|---|---|---:|---|
| MechVQA public benchmark | public_test | 1,185 QA / 562 images | upstream commit + manifest SHA256 `e9ff49a26742d24ac6c1cdff5279aa4eb75e3a17787da54efe75faff2adaeba2` |
| 论文完整数据集 | drawing-level 8:1:1 | 20,778 QA | 论文 v2；完整内部 SFT 语料未公开，不作为本次可重放输入 |

## 官方指标与原始基线

论文 v2 和模型卡报告 MechVL-4B-RL `Total=84.85`。官方 evaluator 使用
GPT-OSS-120B、DeepSeek-V3.2、Kimi-k2 三个 judge，`temperature=0.1`，以多数投票
形成 `overall_voted_score`。但 84.85 没有附带一份与当前固定 public manifest、三
judge endpoint/version 完全绑定的原始响应包，因此这里把它标为“公开参考基线”，
而不是已经在本环境复现的 CPU/CUDA 基线。

资料：<https://arxiv.org/html/2605.30794v2>、
<https://huggingface.co/XiaofengAlg/MechVL-4B-RL>、
<https://github.com/xiaofengShi/MechVQA>。

## 功能验收矩阵

| 用例 | 数量 | 通过标准 | 当前结果 |
|---|---:|---|---|
| 权重完整性 | 15 文件 | `sha256sum --check` 全部成功 | 哈希已固定；大文件未在构建机下载 |
| 数据完整性 | 全量 | commit/SHA/1,185/562/路径全部匹配 | 通过 |
| 单图中文 hard reasoning | 1 | 非空 `&lt;answer&gt;`，JSON 元数据完整，无 fallback | 待 NPU |
| public test 前 5 条冒烟 | 5 | 5/5 请求成功，无 error，输出可供 judge 读取 | 待 NPU |
| 多图记录 | public test 中含多图的固定记录 | 所有图片参与请求，非空答案 | 待 NPU |
| 16K 长上下文边界 | 合成长提示 + 合法图片 | 不超过上限成功，超限显式失败 | 待 NPU |
| 服务模型标识 | 1 | `/v1/models` 精确包含 `MechVL-4B-RL` | 待 NPU |

功能结果必须保存 `npu-smi info`、镜像名称与 digest、`pip show`/版本打印、服务完整
启动命令、客户端 JSON 和服务日志。只生成 HTTP 200 但答案为空、发生 CPU fallback
或图片未被处理都判失败。

## L2 精度/质量

使用 README 第 4 节生成的固定 evaluator config，对 public test 1,185 条全量执行。

| 指标 | 公开参考 | 工程门槛 | 附加条件 | 当前 NPU |
|---|---:|---:|---|---|
| 三裁判多数投票准确率 | 0.8485 | `>=0.8385` | 1,185 条索引完整，目标和三个 judge 均零错误 | 待验证 |

`0.01` 最大绝对下降是待校准的保守工程容差，不是论文或 Ascend ModelZoo 官方门槛。
正式 S3 前必须先在同 manifest、同 prompt、同 target generation、同三个 judge 模型
版本/endpoint 上重放参考后端，并由维护者批准阈值。禁止用单裁判、字符串匹配、抽样
或改变 prompt 后的分数替代。

需要归档：

- `mechvqa_public_test.meta.json` 与输出 manifest SHA256；
- evaluator config 的脱敏副本，记录 judge 的可审计版本标识；
- `responses_MechVL-4B-RL.jsonl`、`evaluated_MechVL-4B-RL.jsonl`、
  `stats_MechVL-4B-RL.json`；
- `accuracy_comparison.json` 与命令退出码；
- 按 capability/subcategory/difficulty/language 的 breakdown。

## 性能验收

保持与精度相同的 checkpoint、BF16、16K 服务边界和生成参数，固定 public manifest
前 10 条。每轮 1 次预热，串行测量 10 个请求，共 3 轮，轮间重启服务并记录冷启动。

| 范围 | 指标 | 汇总方式 | 当前结果 |
|---|---|---|---|
| 端到端 HTTP | mean/P50/P95 latency | 每轮原始值 + 三轮中位数 | 待 NPU |
| 端到端 HTTP | requests/s | 三轮中位数 | 待 NPU |
| 模型输出 | completion tokens/s | 三轮中位数；usage 缺失则判测试无效 | 待 NPU |
| 资源 | 峰值 HBM、功耗、卡利用率 | `npu-smi` 同步采样 | 待 NPU |
| 启动 | 首次服务 ready 时间 | 冷启动单列，不混入稳态 | 待 NPU |

`benchmark.py --dry-run` 只验证图片、问题和 payload 能被构造，报告中固定写
`performance_valid=false`。它不能作为 NPU 性能证据。若与 CUDA/其他 NPU 结果
对比，必须同时固定硬件数量、并发、输入、最大输出、量化/精度和测量范围。

## 回归与失败注入

| 检查 | 预期 |
|---|---|
| 修改任意权重字节 | `serve.sh` 在加载模型前失败 |
| 替换 upstream commit 或 manifest | `prepare_eval_data.py` 非零退出 |
| 图片缺失或 `../` 越界 | 数据准备/benchmark 非零退出 |
| 服务 ID 不一致 | 正式 benchmark 在计时前失败 |
| 少于三轮正式性能测试 | benchmark 非零退出 |
| stats 少样本、有 error 或低于门槛 | `compare_accuracy.py` 非零退出 |
| NPU 不可用 | `serve.sh` 的 tensor probe 失败，不启动服务 |

## Clean-room 与上库验收

在一个新的空目录，仅复制 `NPU_ADAPTATION.md` 所列候选文件，然后严格按 README：

1. 固定 revision 下载并校验权重；
2. 固定 commit 克隆 upstream，生成全量 manifest；
3. 使用官方镜像启动单卡 NPU 服务；
4. 完成功能、全量精度和三轮性能；
5. 对候选文件执行 Python/shell lint、敏感信息与大文件扫描；
6. 在固定 ModelZoo-PyTorch commit 的拟合入路径做 dry run，复核贡献规范；
7. 只有 S3 全部通过并由维护者确认后，才生成正确的 `modelzoo_level.txt`。

最低正式验收清单：

- [x] model/upstream/target revision 与权重哈希固定；
- [x] 数据准备生成固定 manifest/meta，且离线可复用；
- [x] NPU 路线与官方支持矩阵匹配；
- [ ] 单卡 NPU 功能矩阵通过；
- [ ] 三 judge 全量 1,185 条精度通过并校准阈值；
- [ ] 至少三轮真实 NPU 性能完成；
- [ ] clean-room 重放完成；
- [ ] 目标仓贡献规范与许可证复核完成。

当前可声明 **S1 静态交付**；S2、S3、S4 均未完成。

## 报告模板

| 项目 | 必须归档的证据 | 当前状态 |
|---|---|---|
| 原始基线 | 固定 manifest、参考响应/stats、judge 可审计版本 | 公开分数已记录，等价重放待补 |
| NPU 功能 | 环境、服务日志、输入输出 JSON、失败注入 | 待 NPU |
| NPU 精度 | 全量 evaluator 输出与 comparison report | 待 NPU 和三个 judge 服务 |
| NPU 性能 | 三轮原始报告、HBM/功耗、镜像摘要 | 待 NPU |
| clean-room | 新目录重放命令、日志、候选文件哈希 | 待 S2/S3 完成 |
