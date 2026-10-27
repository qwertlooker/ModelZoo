# 五个优先模型：适配分析与验收依据

核查日期：2026-10-27。指定ModelZoo master：`153c97646d940f0eefefe81cbc891693d7243eaa`。

已按用户顺序提高优先级；版本选择是实施建议，未擅自把系列名缩成已完成的单个变体。原119项清单保持原编号，本表新增TimesFM、TTM-R2、Toto、MOMENT四项；DINOv2对应原第84项。

**当前没有本次NPU实测结果，五项均未达到S3。** 以下GPU数据来自原始官方资料；NPU数据来自已读社区源码/报告，写明局限。正式硬件、软件配套与首个变体尚待冻结，不能据此直接安装依赖或宣称可部署。

[GPU/NPU完整分列CSV](priority-five.csv) · [权重revision与SHA表](assets.csv) · [结构化证据](analysis.json)

## 实施决策

| 顺序 | 模型 | 首个候选 | 首要工作 |
|---|---|---|---|
| 1 | TimesFM | 2.5-200M Transformers | NPU强制设备、修正异步计时、真实时序质量；3.0许可单列 |
| 2 | TTM-R2 | Granite R2 512→96 | 保留官方预处理，补ETTh1全量MSE，不以合成信号代替 |
| 3 | DINOv2 | ViT-L/14，无register | backbone+匹配head，补ImageNet全量分类 |
| 4 | Toto | 2.0-22m；4m仅前置探针 | 原始u-μP/SDPA/GQA，九分位，BOOM；1.0独立处理 |
| 5 | MOMENT | small功能、large+trained head预测 | 先解决预测头训练与资产，禁止随机头验收 |

上述顺序遵循用户清单，不是测量出的难度排名。TTM的模型小，适合率先形成真实任务验证范例；这不改变TimesFM的最高优先级。

## 1. TimesFM

**版本：** 首选候选：TimesFM 2.5 200M Transformers版；原生PyTorch版作转换对照。3.0单列，不能静默替换。

**结论：** 分析中；已有社区NPU推理，正式真实任务基准缺失。

**许可：** 2.5代码/权重Apache-2.0；3.0下载权重为非商业、非生产许可，不可直接用于本地生产部署。

| 项目 | GPU：上游实现及公开基准 | NPU：现有实现及缺口 |
|---|---|---|
| 代码 | Google原生2.5与官方Transformers端口均有实现。原生代码默认检测CUDA/CPU、torch_compile=True；Transformers入口TimesFm2_5ModelForPrediction，两套权重键和调用API不可混用。<br>[证据1](https://github.com/google-research/timesfm/blob/e51928e27119cb17bebc005be2696b75e0a9e688/src/timesfm/timesfm_2p5/timesfm_2p5_torch.py)；[证据2](https://huggingface.co/google/timesfm-2.5-200m-transformers/resolve/5a9806b9b291fad9233b5249d88263f1846304d3/README.md) | 社区已有TimesFm2_5ModelForPrediction + torch_npu源码，默认eager/BF16；自动cuda→npu→cpu且捕获导入异常，不满足NPU缺失即失败要求。支持文本单序列逐条运行，文件多行不等于张量batch。<br>[证据1](https://gitcode.com/weixin_72661020/timesfm-2.5-200m/blob/a4da0bda9990850b363ce6945cc19fad1a579f2f/inference.py)；[证据2](https://gitcode.com/weixin_72661020/timesfm-2.5-200m/blob/a4da0bda9990850b363ce6945cc19fad1a579f2f/readme.md) |
| 权重 | 2.5原生与Transformers权重均已取得固定revision及LFS SHA元数据；文件各约925MB，二者SHA不同。尚未下载验证实体哈希。<br>[证据1](https://huggingface.co/google/timesfm-2.5-200m-pytorch/resolve/1d952420fba87f3c6dee4f240de0f1a0fbc790e3/model.safetensors)；[证据2](https://huggingface.co/google/timesfm-2.5-200m-transformers/resolve/5a9806b9b291fad9233b5249d88263f1846304d3/model.safetensors) | 社区使用google/timesfm-2.5-200m-transformers，不能换原生pytorch checkpoint直接加载。正式方案固定HF revision和LFS SHA，strict核对加载缺失、异常键及config。<br>[证据1](https://gitcode.com/weixin_72661020/timesfm-2.5-200m/blob/a4da0bda9990850b363ce6945cc19fad1a579f2f/inference.py)；[证据2](https://huggingface.co/google/timesfm-2.5-200m-transformers/resolve/5a9806b9b291fad9233b5249d88263f1846304d3/model.safetensors) |
| 数据集 | BOOM仓提供TimesFM 2.5独立评测入口；另可用ETTh1/2、ETTm1/2、Electricity、Weather作固定窗口验证。2.5逐通道预测与3.0原生多变量/协变量能力不同；不得用3.0排行榜代替2.5。 BOOM原始notebook固定context2048、normalize_inputs/continuous_quantile_head/flip_invariance/infer_is_positive/fix_quantile_crossing全开，多变量展开成单变量；这才是上述榜单对照协议。<br>[证据1](https://github.com/DataDog/toto/blob/1d3e1c97ccf615016d38fe910602c905ae31fbb7/boom/README.md)；[证据2](https://github.com/zhouhaoyi/ETDataset)；[证据3](https://github.com/google-research/timesfm/blob/e51928e27119cb17bebc005be2696b75e0a9e688/README.md)；[证据4](https://github.com/DataDog/toto/blob/1d3e1c97ccf615016d38fe910602c905ae31fbb7/boom/notebooks/timesfm2_5.ipynb) | 社区展示7点输入、512点合成信号，报告正弦/趋势RMSE；没有真实测试集split、manifest或全量结果。正式先ETTh1 512→96/128，再扩真实多数据集；截断、缺失掩码、归一化参数写入协议。<br>[证据1](https://gitcode.com/weixin_72661020/timesfm-2.5-200m/blob/a4da0bda9990850b363ce6945cc19fad1a579f2f/readme.md)；[证据2](https://gitcode.com/weixin_72661020/timesfm-2.5-200m/blob/a4da0bda9990850b363ce6945cc19fad1a579f2f/inference.py) |
| 性能基准 | 本轮未取得同checkpoint、同上下文/预测长度且注明GPU型号的2.5原始GPU耗时表。README的M4 Max 11.1ms等属于3.0 MLX，不能放入2.5 GPU栏。<br>[证据1](https://github.com/google-research/timesfm/blob/e51928e27119cb17bebc005be2696b75e0a9e688/README.md) | 社区0.39/0.42s计时在输出.cpu()前结束，且无NPU同步、预热和重复，异步计算可能未计全，不作正式延迟。--horizon仅切片输出，可能不改变模型实际计算量。<br>[证据1](https://gitcode.com/weixin_72661020/timesfm-2.5-200m/blob/a4da0bda9990850b363ce6945cc19fad1a579f2f/inference.py)；[证据2](https://gitcode.com/weixin_72661020/timesfm-2.5-200m/blob/a4da0bda9990850b363ce6945cc19fad1a579f2f/readme.md) |
| 精度基准 | 已取得BOOM原始榜单：TimesFM2.5 FP32，MASE=0.632、CRPS=0.382（榜单归一化聚合值）；对应官方原生pytorch权重。另取得GIFT逐任务原始CSV，不能把单任务分数直接称全榜聚合。Transformers端口要先证明与原生相同预测flags，不能直接套用原生榜单分数。<br>[证据1](https://huggingface.co/spaces/Datadog/BOOM/resolve/2bc23d1a3eeebb7ac73a610afcb5db9470c72552/results/leaderboards/BOOM_leaderboard.csv)；[证据2](https://huggingface.co/spaces/Datadog/BOOM/resolve/2bc23d1a3eeebb7ac73a610afcb5db9470c72552/results/models_info/timesfm_2_5_200m/config.json)；[证据3](https://huggingface.co/spaces/Salesforce/GIFT-Eval/resolve/35b8db119291d791ceb7c63f362133e8ba6c905b/results/TimesFM-2.5/all_results.csv) | 社区合成信号RMSE0.03~0.07、BF16/FP32最大差约0.17属于第三方样例，非ETT/BOOM验收。需同权重CPU FP32/原生2.5转换对照，NPU FP32先过完整任务，再评估BF16的MSE/MAE与分位损失。 BOOM正式目标参考MASE0.632/CRPS0.382，但须复用context2048与全部预测flags；社区512/128默认接口不满足该协议。<br>[证据1](https://gitcode.com/weixin_72661020/timesfm-2.5-200m/blob/a4da0bda9990850b363ce6945cc19fad1a579f2f/readme.md)；[证据2](https://huggingface.co/google/timesfm-2.5-200m-transformers/resolve/5a9806b9b291fad9233b5249d88263f1846304d3/README.md)；[证据3](https://huggingface.co/spaces/Datadog/BOOM/resolve/2bc23d1a3eeebb7ac73a610afcb5db9470c72552/results/leaderboards/BOOM_leaderboard.csv) |

**适配路线：** 先torch_npu eager FP32，验证Transformers端口；通过后BF16及SDPA，再评估TorchAir。原生2.5需显式设备入口并关闭默认torch.compile，不能仅model.to(npu)而保留内部self.device=cpu。

**必须处理的问题：**

- Transformers版本下限来自社区>=5.3.0，需固定真实可用版本，与CANN配套torch/torch_npu联测，不能从社区README直接视为已配套。
- 上下文最高16k不等于短样例验证覆盖；预测头128步和可选长预测头/递归路径分别验收。
- GPU官方2.5固定协议性能值仍未取得；不得以其他设备或版本数据补空。
- 新核实BOOM基线使用原生checkpoint、context2048及五个预测flags；Transformers移植若不能逐项保持这些语义，则先用原生2.5路线完成同协议验收，不能借原生基准为移植版本背书。

**验收步骤：**

1. 固定2.5 Transformers checkpoint及原生对照checkpoint、模型代码/评测器提交和资产SHA。
2. ETTh1 train/val/test按时间切分，仅train拟合外部scaler；逐通道或联合指标明确，不能看test调参。
3. 验证短序列padding、512/1024上下文、96/128 horizon、常量/缺失输入和输出长度；不支持输入显式失败。
4. 在真实测试集计算MSE/MAE；概率路径另算WQL/CRPS和分位顺序，NPU/CPU同样本逐项比较。
5. 固定B=1/8/32、L=512/1024、H=96/128，三次独立NPU运行，同步计时，记录冷启动、纯模型、端到端及峰值HBM。
6. 补BOOM context2048及全部官方ForecastConfig flags的完整评测；与512上下文ETT先导验证分开记录。

## 2. TTM-R2

**版本：** IBM Granite TTM-R2 main的512→96：805280参数；不替换成research-r2、R1或R2.1。

**结论：** 分析中；已有较完整社区功能/合成数值对照，真实官方任务NPU验收待补。

**许可：** Granite TTM-R2 Apache-2.0；research版为不同发布边界。

| 项目 | GPU：上游实现及公开基准 | NPU：现有实现及缺口 |
|---|---|---|
| 代码 | IBM tsfm_public.models.tinytimemixer.TinyTimeMixerForPrediction；不是transformers.AutoModel内置类。当前主干代码已扩展，社区固定granite-tsfm0.3.8，实施应冻结其tag/commit后核对config语义。<br>[证据1](https://github.com/ibm-granite/granite-tsfm/blob/9d0b839f997eb4918822a43036012ce8c277687c/tsfm_public/models/tinytimemixer/modeling_tinytimemixer.py)；[证据2](https://github.com/ibm-granite/granite-tsfm/blob/9d0b839f997eb4918822a43036012ce8c277687c/notebooks/hfdemo/tinytimemixer/full_benchmarking/README.md) | 社区已有granite-tsfm0.3.8、torch2.9/torch_npu2.9.0.post1/CANN8.5.1的原生NPU入口，加载缺失/异常/新初始化检查和CPU对照；默认固定逻辑卡0，需改为显式设备参数并保持CPU行为。 PyPI 0.3.8 正式声明 torch>=2.10,<2.11，社区案例的 torch2.9 与声明冲突，不可视为配套证明。<br>[证据1](https://gitcode.com/m0_63918125/granite-timeseries-ttm-r2/blob/a5255ba55d8b3c172dd54cc734fe6f1c0a76f8a4/inference.py)；[证据2](https://gitcode.com/m0_63918125/granite-timeseries-ttm-r2/blob/a5255ba55d8b3c172dd54cc734fe6f1c0a76f8a4/README.md)；[证据3](https://pypi.org/pypi/granite-tsfm/0.3.8/json) |
| 权重 | Granite R2 main固定d6a79570...，3,240,592字节，LFS SHA=a706726a...；按context/horizon/频率使用不同分支。main没有freq prefix，research-r2带freq_token的结果不可混算。<br>[证据1](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2/resolve/d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f/model.safetensors)；[证据2](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2/resolve/d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f/README.md) | 社区记录完整SHA a706726a7eb01bbcb42994b7dcb3c06ea9557898dbae8d480eb04fe8ccb89710，与本轮HF LFS元数据一致；实体尚未下载。代码revision='main'仍漂移，正式冻结d6a79570...。<br>[证据1](https://gitcode.com/m0_63918125/granite-timeseries-ttm-r2/blob/a5255ba55d8b3c172dd54cc734fe6f1c0a76f8a4/README.md)；[证据2](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2/resolve/d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f/model.safetensors) |
| 数据集 | 官方full_benchmarking覆盖ETTh1/2、ETTm1/2等，context512/1024/1536、horizon96/192/336/720分配置；zero-shot与5%few-shot分开。官方卡要求外部逐通道标准化，不能因为内部std scaler就删掉recipe预处理。<br>[证据1](https://github.com/ibm-granite/granite-tsfm/blob/9d0b839f997eb4918822a43036012ce8c277687c/notebooks/hfdemo/tinytimemixer/full_benchmarking/ttm_full_benchmarking.py)；[证据2](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2/resolve/d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f/README.md) | 社区3组合成序列、5/3/7通道、608点前512后96留出，共15通道序列；有GT但规模小，不是官方ETT测试集，不算S3任务验收。<br>[证据1](https://gitcode.com/m0_63918125/granite-timeseries-ttm-r2/blob/a5255ba55d8b3c172dd54cc734fe6f1c0a76f8a4/inference.py)；[证据2](https://gitcode.com/m0_63918125/granite-timeseries-ttm-r2/blob/a5255ba55d8b3c172dd54cc734fe6f1c0a76f8a4/README.md) |
| 性能基准 | 官方结果说明单张NVIDIA A100取得MSE，但无该512→96配置的同步推理延迟/吞吐表。论文的训练/模型相对效率不可直接作为本次NPU加速比。<br>[证据1](https://github.com/ibm-granite/granite-tsfm/blob/9d0b839f997eb4918822a43036012ce8c277687c/notebooks/hfdemo/tinytimemixer/full_benchmarking/README.md) | 社区Ascend910_9362，FP32/HF32关、7通道，warmup5/iters30并同步：B1 p50=4.627ms，B32 p50=4.431ms；分别216.10/7222.08 windows/s。是同进程迭代分布，非三次独立运行中位数，也不是GPU对照。<br>[证据1](https://gitcode.com/m0_63918125/granite-timeseries-ttm-r2/blob/a5255ba55d8b3c172dd54cc734fe6f1c0a76f8a4/README.md)；[证据2](https://gitcode.com/m0_63918125/granite-timeseries-ttm-r2/blob/a5255ba55d8b3c172dd54cc734fe6f1c0a76f8a4/inference.py) |
| 精度基准 | 官方原始CSV：ETTh1 CL512/FL96 zero-shot MSE=0.363、5%few-shot=0.362；512/192为0.387/0.386。这是不同训练状态/权重配置，首个512→96只对照0.363，须复用完整预处理和划分。<br>[证据1](https://github.com/ibm-granite/granite-tsfm/blob/9d0b839f997eb4918822a43036012ce8c277687c/notebooks/hfdemo/tinytimemixer/full_benchmarking/combined_results-granite-r2.csv) | 合成三组MAE0.6443/1.8274/0.4427；随机游走不如naive。社区CPU FP32/NPU最大绝对差1.381e-3；这些不等于ETTh1官方MSE0.363复现。需补真实测试集和外部标准化recipe。<br>[证据1](https://gitcode.com/m0_63918125/granite-timeseries-ttm-r2/blob/a5255ba55d8b3c172dd54cc734fe6f1c0a76f8a4/README.md)；[证据2](https://github.com/ibm-granite/granite-tsfm/blob/9d0b839f997eb4918822a43036012ce8c277687c/notebooks/hfdemo/tinytimemixer/full_benchmarking/combined_results-granite-r2.csv) |

**适配路线：** torch_npu eager FP32最先；纯小型MLP任务优先扩大batch减少host调度开销，再决定是否TorchAir。不得为轻量数值预测套vLLM。

**必须处理的问题：**

- 模型小且同步时延接近B1/B32，可能受host调度影响；此为根据社区数据的推断，需profile证实。
- 输入[B,L,C]、输出[B,H,C]，与MOMENT[B,C,L]不同；布局错误可能维度兼容但语义错误。
- R2/R2.1/research与branch变化会改变频率前缀、长度和质量，模型名不够，必须固定revision。
- granite-tsfm 0.3.8 的 PyPI 元数据要求 Python 3.11–3.13 及 torch 2.10.x；社区 2.9 + --no-deps 不符合该要求，需查明可用昇腾配套栈。

**验收步骤：**

1. 以Granite R2 main 512→96为首个候选，记录55key等配置真实落地，不盲信零missing。
2. 复用full_benchmarking的ETTh1 test完整滑窗，外部scaler只fit train，zero-shot不训练；参考MSE0.363。
3. 记录原始数据文件SHA、切分边界、窗口步长和样本总数；CPU对照复用NPU环境。
4. 验证1/3/7通道、observed mask、长短上下文策略；输出还原到原始量纲与标准化指标各自注明。
5. 三次独立NPU运行B1/32，L512/H96/C7，纯模型与数据搬运分开；不以CPU耗时推算加速比。

## 3. DINOv2

**版本：** 首个候选ViT-L/14无register：facebook/dinov2-large；后续再扩S/B/g及reg4。

**结论：** 分析中；已有910B4随机特征报告，ImageNet任务精度未完成。

**许可：** 标准DINOv2 Apache-2.0；XRay-DINO另有非商业许可，不能串用。

| 项目 | GPU：上游实现及公开基准 | NPU：现有实现及缺口 |
|---|---|---|
| 代码 | Meta官方backbone、kNN/linear评估及预训练heads；HF AutoModel是另一权重封装。官方attention支持SDPA回退，MemEffAttention在xFormers可用时调用CUDA扩展；NPU需要显式选择兼容路径。<br>[证据1](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/layers/attention.py)；[证据2](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/eval/linear.py) | 社区xujiashuai/dinov2-large有AutoModel特征推理及评估，但精度异常被捕获后仍写SUCCESS，不满足失败非零门禁；需要修正比较失败传播。<br>[证据1](https://gitcode.com/xujiashuai/dinov2-large/blob/7159789e728fc6cd9edb62e5b626bc418e0338b1/evaluate.py) |
| 权重 | 已取得facebook/dinov2-large固定47b73eef...、1,217,522,888字节及LFS SHA。官方另有ViT-L/14 backbone、linear/linear4头；checkpoint和head/normalization必须同一protocol，reg4不能混用。<br>[证据1](https://huggingface.co/facebook/dinov2-large/resolve/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c/model.safetensors)；[证据2](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/README.md) | 社区from_pretrained未冻结revision；正式固定HF47b73eef...及preprocessor_config。NPU checkpoint无需另训，但上游Meta pth与HF safetensors不能跨加载器直接互换。<br>[证据1](https://huggingface.co/facebook/dinov2-large/resolve/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c/model.safetensors)；[证据2](https://gitcode.com/xujiashuai/dinov2-large/blob/7159789e728fc6cd9edb62e5b626bc418e0338b1/evaluate.py) |
| 数据集 | ImageNet-1k TRAIN用于kNN特征库/linear训练、VAL用于评测，需授权本地数据；原始142M预训练图像不是推理验收下载条件。README示例ImageNet22k预训练recipe另算。<br>[证据1](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/README.md)；[证据2](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/eval/linear.py) | 现有报告torch.randn、B1/224×224、输出[1,257,1024]；未固定随机种子。既不是ImageNet输入，也没有类别标签，仅功能样例。<br>[证据1](https://gitcode.com/xujiashuai/dinov2-large/blob/7159789e728fc6cd9edb62e5b626bc418e0338b1/report.json) |
| 性能基准 | README 32×A100-80GB约1天属于ViT-L/16训练recipe，不能当ViT-L/14推理速度；本轮未确认与社区224×224/B1严格同口径GPU推理值。<br>[证据1](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/README.md) | 社区910B4 FP16，预热3次后10次同步均值11.805ms、峰值分配0.611GiB；不同于README旧12.18ms。只覆盖随机B1/224输入，不是完整加载+预处理/分类性能，无三次独立运行统计。<br>[证据1](https://gitcode.com/xujiashuai/dinov2-large/blob/7159789e728fc6cd9edb62e5b626bc418e0338b1/report.json) |
| 精度基准 | 官方ViT-L/14无register：ImageNet kNN83.5%、linear86.3%；reg4为83.8%/86.7%。旧表81.6/82.9属于另一个L/16训练recipe，应从L/14适配基线剔除。公开分数只有复用对应图像变换、特征层和classifier/head才能直接比较。<br>[证据1](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/README.md) | 社区CPU FP32/NPU FP16 cosine0.9999527只衡量随机单例特征；(1-cosine)×100=0.0047%不是分类错误率。须补官方head或kNN的ImageNet全量指标，不宣称已复现86.3%。<br>[证据1](https://gitcode.com/xujiashuai/dinov2-large/blob/7159789e728fc6cd9edb62e5b626bc418e0338b1/report.json)；[证据2](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/README.md) |

**适配路线：** 先原生torch_npu/eager保证CLS与patch特征一致，SDPA需同输入数值验证；随后固定尺寸TorchAir，再按部署卡型考虑ONNX/OM。训练与多裁剪nested tensor不纳入首个推理交付。

**必须处理的问题：**

- 224图像token257由1CLS+16×16 patch得到；register版本token数不同。
- 官方backbone self.head=Identity，不含1000类分类器；只下载backbone不能计算分类top1。
- 官方linear.py多处.cuda及distributed CUDA假设，不能只改模型device；需最小可重复patch。

**验收步骤：**

1. 冻结L/14无register、权重与preprocessor、classifier head及官方评估protocol，不借用L/16训练分数。
2. ImageNet-1k全量VAL，记录可访问图像数/缺失/损坏与固定manifest；不足时只标子集，不报全量top1。
3. 至少比较CLS、patch特征及head logits的CPU/NPU误差，再算top1/top5；需要kNN时预先生成TRAIN特征库并冻结k/温度。
4. 固定resize/crop/interpolation/antialias/ImageNet均值方差；224功能检查与官方评分分辨率分开。
5. NPU性能B1/8/32和224/518分别测，三次独立运行；报告images/s、p50/p95、HBM，排除/包含预处理两口径。

## 4. Toto

**版本：** 当前官方推荐2.0；首个候选22m，4m仅探针；1.0-151M为需微调/外生变量的独立分支。

**结论：** 分析中；已找到1.0及2.0-4m社区NPU实现，但不能互相覆盖规模或指标。

**许可：** 官方代码与公开权重Apache-2.0。

| 项目 | GPU：上游实现及公开基准 | NPU：现有实现及缺口 |
|---|---|---|
| 代码 | Toto2Model采用时间/变量交替注意力、u-μP与quantile头；1.0用Student-T mixture+采样，API/概率输出不同。2.0官方Python>=3.12，fine-tuning/EV尚未发布。 固定源码中的 toto-2 与 dd-unit-scaling 均声明 torch>=2.14.0，不能据较低版本社区案例推断兼容。<br>[证据1](https://github.com/DataDog/toto/blob/1d3e1c97ccf615016d38fe910602c905ae31fbb7/README.md)；[证据2](https://github.com/DataDog/toto/blob/1d3e1c97ccf615016d38fe910602c905ae31fbb7/toto2/toto2/model.py)；[证据3](https://github.com/DataDog/toto/blob/1d3e1c97ccf615016d38fe910602c905ae31fbb7/toto2/pyproject.toml)；[证据4](https://github.com/DataDog/toto/blob/1d3e1c97ccf615016d38fe910602c905ae31fbb7/dd_unit_scaling/pyproject.toml) | 社区2.0-4m调用官方Toto2Model/torch_npu，无需LLM服务框架；但cuda→npu→cpu自动回退且包装器只保留单变量。1.0社区内置fake_gluonts shim并有CPU fallback，不应直接复用为正式实现。<br>[证据1](https://gitcode.com/weixin_72661020/Toto-2.0-4m/blob/658a07a206dcc3b23c1ad98ac7a68d411477c9c8/inference.py)；[证据2](https://gitcode.com/wrhhh/Datadog-Toto-Open-Base-1.0-NPU/blob/e8c6c836de217b4a19b02b0e72956ee0fe33b302/inference.py) |
| 权重 | 2.0有4m/22m/313m/1B/2.5B；22m固定685e4ae3...，87,669,368字节。1.0固定0411ceb2...，605,239,264字节；两者完整LFS SHA已记录，不能把4m验证写成22m/2.5B支持。<br>[证据1](https://huggingface.co/Datadog/Toto-2.0-22m/resolve/685e4ae3e2be8d8998025e53dd98e7fdcb296a89/model.safetensors)；[证据2](https://huggingface.co/Datadog/Toto-Open-Base-1.0/resolve/0411ceb27bdf7fc3e4892e99edc8ad08192dc3c5/model.safetensors) | 2.0社区4m和官方22m不是同checkpoint；4m冒烟不能证明22m权重完整加载。正式选22m后冻结config+权重revision和SHA，保留dd-unit-scaling原始前向，不擅自简化模型。<br>[证据1](https://huggingface.co/Datadog/Toto-2.0-22m/resolve/685e4ae3e2be8d8998025e53dd98e7fdcb296a89/model.safetensors)；[证据2](https://gitcode.com/weixin_72661020/Toto-2.0-4m/blob/658a07a206dcc3b23c1ad98ac7a68d411477c9c8/inference.py) |
| 数据集 | 官方BOOM、GIFT-Eval、TIME各自协议；2.0 leaderboard使用decode_block_size=None单次前向，quick-start=768并不相同。target[B,V,T]、mask和series_ids需要成对传递。<br>[证据1](https://github.com/DataDog/toto/blob/1d3e1c97ccf615016d38fe910602c905ae31fbb7/README.md)；[证据2](https://github.com/DataDog/toto/blob/1d3e1c97ccf615016d38fe910602c905ae31fbb7/boom/README.md)；[证据3](https://huggingface.co/datasets/Datadog/BOOM) | 4m社区输入20/512点文本序列，未提供BOOM真实GT评测；首值padding后target_mask全True，填充点当成真实观测，需按官方缺失/补齐语义复核。1.0随机3变量不是2.0真实基准。<br>[证据1](https://gitcode.com/weixin_72661020/Toto-2.0-4m/blob/658a07a206dcc3b23c1ad98ac7a68d411477c9c8/inference.py)；[证据2](https://gitcode.com/weixin_72661020/Toto-2.0-4m/blob/658a07a206dcc3b23c1ad98ac7a68d411477c9c8/readme.md) |
| 性能基准 | 官方22m卡：单A100、batch8、1024步single-pass forecast，22m约5.0ms；4m3.8ms、313m15.4ms。模型卡未在该表明确context和计时预热细节，不能与512→96社区0.29s直接算加速比。<br>[证据1](https://huggingface.co/Datadog/Toto-2.0-22m/resolve/685e4ae3e2be8d8998025e53dd98e7fdcb296a89/README.md) | 4m社区0.27~0.32s在输出.cpu()前停止计时，无同步/预热/重复。1.0社区warmup3/iters10同步p50=32.5ms、采样256=0.7079s且含_standard_gamma CPU fallback；两者不可混成2.0-22m性能。<br>[证据1](https://gitcode.com/weixin_72661020/Toto-2.0-4m/blob/658a07a206dcc3b23c1ad98ac7a68d411477c9c8/inference.py)；[证据2](https://gitcode.com/wrhhh/Datadog-Toto-Open-Base-1.0-NPU/blob/e8c6c836de217b4a19b02b0e72956ee0fe33b302/readme.md) |
| 精度基准 | 22m固定模型卡：BOOM CRPS0.363/MASE0.601；GIFT CRPS0.496/MASE0.719；TIME0.556/0.668。保留各基准聚合/归一化口径，不能与逐窗口原始CRPS或4m值混用。<br>[证据1](https://huggingface.co/Datadog/Toto-2.0-22m/resolve/685e4ae3e2be8d8998025e53dd98e7fdcb296a89/README.md) | 4m社区列BOOM CRPS0.377/MASE0.624是模型介绍引用，没有NPU重跑BOOM证据。正式22m需九分位WQL/CRPS、median MASE，并覆盖缺失、多变量、长预测；不能只验证输出形状。<br>[证据1](https://gitcode.com/weixin_72661020/Toto-2.0-4m/blob/658a07a206dcc3b23c1ad98ac7a68d411477c9c8/readme.md)；[证据2](https://huggingface.co/Datadog/Toto-2.0-22m/resolve/685e4ae3e2be8d8998025e53dd98e7fdcb296a89/README.md) |

**适配路线：** 先2.0-22m torch_npu eager FP32；检查SDPA的enable_gqa、因果/变量mask与arcsinh scaler，保留官方u-μP缩放；功能通过后BF16/TorchAir。1.0只有明确需微调或外生变量时独立适配。

**必须处理的问题：**

- Python>=3.12与BOOM README旧评测栈要求3.11存在依赖冲突，需分进程/离线评测交换结果或更新固定评测器，不绕过Requires-Python。
- 部分社区强制CPU-only torch再加torch_npu、解包wheel绕版本、fake_gluonts等不符合本仓规范，不复制安装方案。
- 源码float64统计支持检查、GQA/SDPA与mask路径需要目标NPU实测；不能以eager标签当算子全支持。
- 固定 Toto 源码两个包均要求 torch>=2.14.0；未确认该版本的 torch_npu/CANN 配套，不能绕过元数据强装。

**验收步骤：**

1. 22m固定revision与87.7MB资产；先1/3/7变量、缺失mask、NaN和不同horizon功能验证，未支持的输入直接报错。
2. BOOM任务manifest与官方evaluator固定commit；先命名子集调通，再完整评测，子集不能冒充全部。
3. decode_block_size=None与768分别记录；无缺失才允许has_missing_values=False，禁止掩盖真实缺失。
4. 与官方22m模型卡对照CRPS/MASE聚合，并用CPU同权重输出核对九分位/尺度恢复；记录跨分位交叉率。
5. NPU三次独立运行B1/8、L512/1024、H96/1024、V1/7；probabilistic输出与点预测计时分开。

## 5. MOMENT

**版本：** 功能探针small；正式预测基准large+已训练forecast head（官方教程ETTh1/192），重建/补全可复用预训练head。

**结论：** 分析中；已有small NPU功能样例；预测head未训练时不能形成有效预测适配。

**许可：** MOMENT代码/公开small、large权重元数据MIT；数据集授权分别处理。

| 项目 | GPU：上游实现及公开基准 | NPU：现有实现及缺口 |
|---|---|---|
| 代码 | 官方momentfm.MOMENTPipeline；输入[B,C,512]、patch8、T5 encoder。_get_head明确仅reconstruction预训练；forecasting/classification创建新head且需要微调。<br>[证据1](https://github.com/moment-timeseries-foundation-model/moment/blob/38f7310ad594100747ca2a8357e9c7ca7d323e0e/momentfm/models/moment.py) | 社区small提供forecast/reconstruct/classify/detect_anomalies/embed等入口，显式input_mask与x同设备；默认无NPU会走CPU，且静默屏蔽torch_npu初始化输出。不能直接作为失败可见的正式交付。<br>[证据1](https://gitcode.com/anning-2026/MOMENT-1-small/blob/bb481bd53393223e9675006c69702bad80aefcef/inference.py)；[证据2](https://gitcode.com/anning-2026/MOMENT-1-small/blob/bb481bd53393223e9675006c69702bad80aefcef/readme.md) |
| 权重 | small固定411e2882...，model.safetensors151,615,328字节；large固定ca58581b...，1,385,501,256字节。bin/safetensors是可替代序列化，不能相加称模型参数规模。<br>[证据1](https://huggingface.co/AutonLab/MOMENT-1-small/resolve/411e288267f82cce86296dbe4d6c8bc533cc162f/model.safetensors)；[证据2](https://huggingface.co/AutonLab/MOMENT-1-large/resolve/ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc/model.safetensors) | 社区加载公开MOMENT-1-small后model.init()，没有预测/分类head训练步骤和checkpoint。预测输出即使有限也可能来自新随机head；必须选reconstruction功能线或先训练并冻结head。<br>[证据1](https://gitcode.com/anning-2026/MOMENT-1-small/blob/bb481bd53393223e9675006c69702bad80aefcef/inference.py)；[证据2](https://github.com/moment-timeseries-foundation-model/moment/blob/38f7310ad594100747ca2a8357e9c7ca7d323e0e/momentfm/models/moment.py) |
| 数据集 | 官方forecasting教程：ETTh1、seq512、horizon192、7通道；InformerDataset按时间切分，StandardScaler只fit训练区间。Timeseries-PILE是预训练资源，不能充当无泄漏测试集；重建/异常各自另设mask/标签。<br>[证据1](https://github.com/moment-timeseries-foundation-model/moment/blob/38f7310ad594100747ca2a8357e9c7ca7d323e0e/tutorials/forecasting.ipynb)；[证据2](https://github.com/moment-timeseries-foundation-model/moment/blob/38f7310ad594100747ca2a8357e9c7ca7d323e0e/momentfm/data/informer_dataset.py) | 社区np.cumsum(np.random.randn)随机游走、B1/C1/L512，未固定seed且无holdout/真实GT。全True input_mask只验证无缺失路径，不是补全/异常检测验收。<br>[证据1](https://gitcode.com/anning-2026/MOMENT-1-small/blob/bb481bd53393223e9675006c69702bad80aefcef/inference.py) |
| 性能基准 | 官方README只说明教程可在单A6000-48GiB复现，没有完整同口径推理时延表；显存容量不是模型峰值实测，更不能从参数量推导NPU加速比。<br>[证据1](https://github.com/moment-timeseries-foundation-model/moment/blob/38f7310ad594100747ca2a8357e9c7ca7d323e0e/README.md) | 社区预测96步0.123s、重建0.114s、分类0.109s均单样例日志；需补NPU同步、预热和三次独立计时。随机预测head的耗时只证明算子链路，不是有效预测服务性能。<br>[证据1](https://gitcode.com/anning-2026/MOMENT-1-small/blob/bb481bd53393223e9675006c69702bad80aefcef/readme.md)；[证据2](https://gitcode.com/anning-2026/MOMENT-1-small/blob/bb481bd53393223e9675006c69702bad80aefcef/inference.py) |
| 精度基准 | 官方forecasting notebook保存large+预测head训练1epoch的ETTh1 H192结果：MSE0.422/MAE0.432，seed13、batch8；是教程运行记录，未提供本轮重跑及已训练head资产，不能用small随机head直接对齐。<br>[证据1](https://github.com/moment-timeseries-foundation-model/moment/blob/38f7310ad594100747ca2a8357e9c7ca7d323e0e/tutorials/forecasting.ipynb) | 社区无MSE/MAE、异常标签F1或CPU/NPU误差表，称FP32稳定不足验收。large官方0.422/0.432不是small零样本预测基线；正式预测先产出并冻结trained head，再跑同协议完整test。<br>[证据1](https://gitcode.com/anning-2026/MOMENT-1-small/blob/bb481bd53393223e9675006c69702bad80aefcef/readme.md)；[证据2](https://github.com/moment-timeseries-foundation-model/moment/blob/38f7310ad594100747ca2a8357e9c7ca7d323e0e/tutorials/forecasting.ipynb) |

**适配路线：** 先torch_npu FP32重建/embedding验证预训练主干，再做large ETTh1预测head训练与推理；训练AMP/GradScaler替换为配套NPU入口，保持CPU分支。不要把随机head预测当zero-shot交付。

**必须处理的问题：**

- 固定small config的num_layers=8、d_model=512、num_heads=6；单个safetensors约151.6MB（144.6MiB）。bin/safetensors为替代格式，不能相加称290MB模型参数规模。
- NPU scalar/NaN/mask/RevIN统计及T5相对位置bias需数值核对；mask全True覆盖不到缺失补全。
- 尚无已训练预测head可用性证据；如果不训练head，本阶段只能交付重建/embedding而不能宣称完成预测。

**验收步骤：**

1. 先锁定任务：重建/补全与预测分线。保留所有五模型优先范围，MOMENT预测head训练是必要工作，不以缩成embedding替代目标。
2. 重建先真实ETTh1固定mask评估缺失位置MSE/MAE；可视化只辅助，不算指标。
3. 预测按官方large/H192/head1epoch/seed13协议，保存head或完整checkpoint及SHA；训练集外scaler不能泄漏。
4. CPU参考必须加载同一个已训练head；随机head对照需明确作为失败控制，不能用于质量汇报。
5. 完成512输入、H96/192、C1/7边界和NPU三次性能；异常/分类仅在用户明确任务与标签协议后单独扩展。

## 资产准备与可比性

assets.csv记录8份候选权重的固定HF revision、完整SHA256、字节数及下载URL。SHA来自HF LFS元数据，尚未下载文件核验；不是本次实体校验成功记录。还需分别下载config、预处理配置、必要任务head及许可证并生成完整manifest，单独的model.safetensors不足以完成交付。

下载应使用curl/wget固定revision URL，再以sha256sum校验；不得直接运行来源不明的自动下载器、社区发布脚本或修改系统依赖。本批没有执行下载。

TTM与MOMENT的MSE基于各自官方标准化协议；不能把反归一化MAE与标准化MSE混算。Toto的榜单聚合分数也不能与单窗口未归一化CRPS直接比较。DINOv2特征余弦相似度不能当分类准确率。

## 交付与停止条件

- 正式路径拟为`ACL_PyTorch/built-in/nlp/<timesfm/ttm-r2/toto/moment>`及`ACL_PyTorch/built-in/cv/dinov2`，仅为建议；需按目标仓当前领域分类和已有目录比较后冻结。没有在这些路径写入假适配。
- 当前报告依据固定目标master及本仓已有完整目录清单/增量核查；社区项目单列，不计入指定ModelZoo仓内已有适配。实施前还需采同领域最新实质合入样例及贡献规范，记录对应commit/date；本报告不冒充NPU_ADAPTATION或上库候选。
- 每个实施工作区维护README.md、NPU_ADAPTATION.md、ACCEPTANCE_PLAN.md；需可执行数据准备、运行、比较及报告入口，代码修改有测试。
- NPU不可用必须非零失败，CPU仅显式baseline；禁止自动换卡/换模型、静默回退、伪造质量、修改失败后仍写SUCCESS。
- 无GPU环境不阻断NPU验收：优先官方同协议指标，必要时同环境CPU精度对照；CPU性能不推导GPU/NPU加速比。
- 所有阈值在运行前冻结。暂不编造统一1%精度阈值或NPU吞吐目标；任务精度容差、业务SLA和设备支持范围需明确，当前保留待验收。
- 每项通过完整NPU精度和三次独立性能才可S3；上库还需结构门禁、许可证、正式README自包含及clean-room重放。

## 本轮验证

只执行互联网源码/模型卡/JSON元数据读取、固定HEAD查询、CSV/JSON一致性与既有报告回归；未运行模型、安装依赖、下载权重/数据或访问NPU服务器。缺少目标执行机信息，不能确认环境配套和具体算子兼容。
