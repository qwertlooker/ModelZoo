# 修正版验证记录

日期：2026-10-25，Asia/Shanghai；首次全量基线为2026-10-23。

- 119项按原始编号逐项保留；代码、权重、数据、性能、精度分别记录证据类型和链接。
- 18项回归测试通过：覆盖顺序、字段结构、非认证边界、易混指标回归、原始资源边界、固定ModelZoo文件存在、CSV/Markdown完整性、链接清单覆盖、GPU重查、后续证据增补、发布状态、计时范围、设备/仓库归属及可重复构建。
- 当前链接清单574个：540个有成功HEAD记录、34个未确认（27个Hugging Face限流429、5个403、Kvasir-SEG证书错误1个、华为云567共1个）。本批检查23个新增URL，均成功；退出当前引用集合的6个URL从清单移除。其他URL保留原检查时间，不把限流/访问限制等同于链接不存在。
- 固定提交链接173个；动态或未版本化链接401个。数量按URL路径形式统计，含同页不同锚点，不表示全部外部来源已冻结。
- 原始文件/JSON读取和HTTP结果均不等于独立模型复测。
- 未执行：NPU/CPU模型运行、依赖安装、资产下载及SHA、完整任务精度、三次独立NPU性能、clean-room和上库门禁。
- 详细边界、修正内容和命令见REVIEW_FIXES.md；逐URL状态见link-check.json。

## 提交边界

本次只暂存reports/ascend-audit-20261022内的修订，不触碰其他16个既有修改。远程origin属于当前ModelZoo项目，计划普通推送当前main到origin/main；已执行push --dry-run，失败后没有继续实际推送、强推或更改凭据。本地提交保留。

错误原文：`remote: Permission to qwertlooker/ModelZoo.git denied to pcb1990.`；`fatal: unable to access 'https://github.com/qwertlooker/ModelZoo.git/': The requested URL returned error: 403`。

首次diff检查发现Windows生成文件CRLF被仓库识别为尾空白；已修正生成器固定LF并重新生成，复查通过。

## GPU/NPU分列验证

CSV共119行、25列，保留原始编号与顺序。GPU与NPU分别包含代码、权重、数据、性能、精度及每项独立链接列；两类结论和注意事项分开；对照原文保存在JSON。新增设备字段/列完整性与明确GPU/NPU数值归属测试。GPU尚未专项核实的字段明确保留待核实，不以NPU支持代替。

## 阅读版修订

CSV删除每格重复免责声明、内部证据类型和长段对照原文；保留GPU/NPU五项说明及独立链接列。未查到、未核实分别表达无证据和资源使用未确认。新增阅读版回归测试，保留指标数字、batch和不确定性。链接集合不变，本次未重复网络检查。

## GPU重新查询（本轮）

119项GPU侧均已重新查询；105项读取原始资料并记录具体发现，14项查到发布/论文/CPU实现或名称仍不明确，未认定GPU交付。其中87项找到明确官方GPU实现，其余类型见GPU_RECHECK.md。GPU查询完成标记只表示进行了资料检索，不表示五项证据全部闭合。生成器已接入gpu_research.py，阅读版不再用“存在上游链接”覆盖实际GPU结论。

新增回归覆盖：GPU结论正确显示、N1.6版本及硬件数值、YOLOv8和RT-DETR数值、WALL-WM未发权重、Swin-Unet最终权重未分享、平台渲染不能冒充模型支持。`git diff --check`通过。ModelZoo master重新在线查询仍为dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202。

## 2026-10-25前一批增补（历史记录）

- 保留119项、25列以及GPU/NPU五维结构。新增evidence_updates.json记录18项逐字段增补及17次GitHub正文读取的文件blob SHA；blob SHA不当作仓库commit使用。
- 补充AlphaFold3按tokens/A100/H100耗时，BIT/ChangeFormer切块与测试集指标，Swin-Unet与U-Net分开，Anomaly Transformer的point-adjust F1，InternVL2/3、LLaVA、Qwen2/2.5-VL的指定规模结果，Depth Anything度量/相对版本边界。
- 补充Cosmos-Predict2 GPU生成耗时及输出视频帧率区别、Reason1分阶段分口径评测、GigaBrain clean-only训练协议、LingBot基础/后训练/深度变体，撤回Pelican-VL未经确认的3B发布范围。
- 已记录原始资料冲突：LingBot论文与README的261 samples/s归属不一致；N1.6 LIBERO成功次数与部分百分比不一致；ChangeFormer论文写随机初始化而当前README脚本使用ADE预训练，DSIFN下载链接误指LEVIR。
- NPU字段未改动，设备检查日期分别保留；ModelZoo master重新查询仍为dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202。
- 13项回归通过，仅证明生成及已知归属规则；不证明全量分析完成。GPU仍有41项性能、14项精度保留字面“数值待核对”，还有其他具体待核说明。目标仍为全119项完整分析，不将本次18项增补当作完成。

## 2026-10-25继续复核

- 本批实际修改27项GPU证据；增补文件累计覆盖44项、43次GitHub正文读取的blob SHA。原始119项、25列及NPU五维内容与前一提交逐字段一致。
- Cosmos-Predict2官方模型页与PBench页首列顺序冲突，按PBench定义及分项均值注明；Transfer1补充B200单卡141.7 s、64卡4.2 s端到端数据。Drive-Dreams的下游车道检测F1不冒充生成模型精度。
- G0、Tiny、Plus分别核对；N1.5补LIBERO成功次数与8步去噪配置；GigaWorld-1区分Stage1发布权重与Stage2蒸馏加速；GigaWorld-0的H20耗时明确为训练step。π0、Wall-OSS、XR-1区分模型推理、动作块与机器人执行频率。
- DeepAb、ProteinMPNN、LigandMPNN补原始科学指标，CPU或未注明硬件的耗时不填成GPU基准；MHCflurry区分软件/权重版本及flanks输入；ThermoMPNN采用正式PNAS结果，标明预印本差异。ESM3开放1.4B与论文其他规模分开。
- BEVFormer补V100整体1.7 FPS及模块耗时；SAM区分A100图像编码与浏览器CPU解码；Grounding DINO的TAO商业重训版性能单独标注，不与原始OGC精度配对。UA-NWM补4090离线8.47 ms/frame、3090在线2.70 s/step及各自评测条件。
- 新增运行范围与科学指标回归，14项全部通过；`git diff --check`通过。链接HEAD不证明正文、权重或运行结果正确。
- 当前GPU性能仍有23项保留字面“数值待核对”；另外还有未匹配指定权重、硬件或测试协议的项目。字面占位减少不等于全量分析完成。DL-foils及MLOps训推一体框架仍需准确项目身份，已请求名称或链接，未用相似项目替代。
- 再次查询ModelZoo master，仍为dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202。推送dry-run仍返回上述403；保留本地提交，不更改凭据。

## 2026-10-25第三批复核

- 相对此前提交新增或修正23项GPU五维证据，增补文件累计63项、56次GitHub正文/图片/JSON读取的blob SHA。119项及25列不变，NPU五维内容与前一提交逐字段一致。
- Thinker-4B七项具身指标来自实际读取的官方JPG表；7B技术报告的63.5/58.21另列。WALL-SS官方TODO明确代码未发布，增加可覆盖设备结论的增补字段，CSV显示与原始发布状态一致。
- BGE-M3采用2024-06修正版v4区分dense/sparse/multi-vector；BGE reranker改为其本身的Qwen团队重测，列明top100候选来源。实际读取issue1088后确认5.4 TPS为transactions/s、23文档×512；保留A100切片条件，撤销此前“单位/输入未说明”的错误解释。
- Qwen3-Embedding-8B性能取Superlinked固定数据仓revision的原始summary JSON：H100、并发8、60秒stock29051 input tok/s；PR与stock分列。第三方服务负载不是Qwen官方统一速度，0.6B/4B和Reranker不继承该速度。
- RoseTTAFold通过Europe PMC全文XML核对原始论文，分搜索、GPU网络、CPU pyRosetta以及复合物耗时；RFdiffusion的11秒限100残基15步。AlphaFold-Multimer采用2022-03修订版的界面指标，并注明当前代码与论文版本边界。
- SegNet原始Caffe时延与PaddleSeg Cityscapes精度分开；ResUNet与ResUNet++分开。Depth Anything V2补RTX3080/Hypersim-vits/TensorRT原始100次样本JSON，4.3084 ms只覆盖H2D/核函数/D2H。论文DA-2K与in-domain度量微调另列。
- DreamerV3从PDF表2/9读取单A100训练GPU天数与Atari100k分数，避免HTML漏表；QMIX的heuristic胜率不能当QMIX结果；V-JEPA2-AC的16秒是800候选×10轮CEM规划。
- 15项回归通过，新增发布状态、具体模型变体、训练/推理计时与召回候选源归属检查；可重复构建与git diff --check通过。没有模型实测或新增认证。
- 原来23项字面“数值待核对”均已进一步查阅并改为具体来源结果或明确的证据边界；这不表示所有性能数据已取得。StarVLA、MinerU3.3、BLASTn、ESM2、Faster R-CNN两行、InternVL2/3、LightGBM、MAPPO及Qwen3仍有更具体的性能待核；其他模型也存在指定权重/完整资产获取未闭合项。全量完整核查目标继续保留。
- ModelZoo master本批再次核对无变化。origin/main普通推送dry-run仍因pcb1990无qwertlooker/ModelZoo写权限返回403；没有实际推送或凭据变更。

## 2026-10-25第四批复核

- 本批实际修正23项GPU记录（3、24、25、27、29、32、33、39、46、49、53、58、69、70、75、82、83、93、94、97、106、109、112）。增补文件累计84项、67份GitHub来源读取记录；119项及25列保留，NPU五维内容与前一提交逐字段一致。
- StarVLA从官方issue158补A100训练速度，保留同一报告两张表的条件/数值差异；没有当作动作推理Hz。Faster R-CNN从metafile补V100、FP32、batch1、800×1333及具体checkpoint；69/70仍是原始两行，采用同一GPU实现，不重复计算独立适配。
- MinerU严格取3.3.0提交及2605模型卡：补权重、A100并发2.12 fps，后端95.39/95.26与模型卡95.72分开。Qwen3取8B/H20具体测试，注明prompt+generation计速公式，thinking/non-thinking精度分开。
- InternVL3从Drive-KD表1/附录E.2取得A100-PCIE-40GB、32.6新生成token/s、TTFT1.58 s；这是第三方原模型基线。InternVL2读取ICLR官方论文PDF第42页，testmini整套评测0.97 h不能换算成单请求延迟。
- ESM2补NVIDIA优化15B训练吞吐，不能给650M推理继承；LightGBM实际读取官方PNG图，Higgs/63bins/500轮GTX1080为112 s、RX480为134 s。G-BLASTN论文blastn/megablast及单/多CPU核对照分开，旧论文与2019社区版本分开。
- RALLY原论文明确GPU部署，并有单4090推理14.48 s/15.39 s，撤回此前笼统“GPU交付未确认”结论；公开专项源码、LoRA、RMIX权重仍未取得。RT-2原论文是多TPU云服务，撤回按GPU理解其1–3Hz的可能。Z-1补1199条演示、24任务67.4%→80.6%及48/64次评估协议，撤回不准确的“流式VLA”描述。
- AirVLA保留Coming Soon发布状态，补真实数据范围、条件成功率，动作执行10Hz不当作GPU吞吐。Thinker-VLA与开源Thinker视觉语言底座分开；Spirit1.6与1.5、WALL-A/B与WALL-OSS分别核实；未把公司历史新闻等同当前榜单或GPU测速。
- Bracken和CBBA不需要神经权重，分别记录概率数据库与任务配置；原始CPU精度/性能有来源，不能冒充GPU实测。UNet-ESH对应SiC晶圆缺陷检测，出版方摘要可检索，全文403，具体资产及数值表仍未核实。
- 17项回归、可重复构建、NPU字段一致性与git diff --check通过。测试只验证报告结构及已知归属规则；公开结果未独立复测，权重/数据未下载校验。没有GPU/NPU实测或S3认证。
- 未闭合项仍包括MAPPO统一GPU墙钟性能、部分论文的输入长度/batch/硬件条件、未发布或未取得的专项资产；DL-foils与MLOps训推一体框架仍无法唯一确定对象，之前已请求准确项目身份。未把剩余缺口解释为互联网没有，也未将本批增补当作全量完成。
- ModelZoo master重新查询仍为dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202。

## 2026-10-25第五批复核

- 实际修改4项GPU（96、101、102、110）和6项NPU（42、43、44、45、76、81），原119项和25列不变。增补文件累计86条GPU、6条NPU，覆盖88个原始条目；79份GitHub正文记录和4份GitCode API正文记录。该计数不代表全量完整验收。
- YOLOv8从“未取得专项NPU证据”修正为其他官方仓已有OM实现。重新查询modelzoo-GPL master为aa0696bd53c3724c8438860243b55025539de64a，通过GitCode contents API按该commit实际读取README、pth2onnx.py、detect_infer.py、classify_infer.py，记录各文件blob SHA。原ModelZoo-PyTorch master仍为dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202，目录树无YOLOv8，不把GPL仓支持算成指定仓收录。
- YOLOv8：310P3/bs8/640×640，n/s纯OM吞吐753.23/443.60fps，COCO val2017 mAP@0.5为52.5/61.6；与GPU常见mAP50:95分开。检测conf0.001/iou0.7已在源码核实。分类s-cls/224×224/bs8为5204fps、ImageNet Top1 73.7%；未复测，不视为三次中位数。旧om_infer.py在当前commit返回404，当前README实际入口为detect_infer.py/classify_infer.py，未使用旧文件链接替代。
- ByteTrack社区源码83a2fb52c7be9ad40ddc8999a40b295b6161ddac明确YOLO OM走InferSession，关联和卡尔曼走NumPy/SciPy CPU。原来“只有原始资料”结论过弱，改为检测NPU/跟踪CPU组合；VideoWriter取视频fps不是处理吞吐，未取得完整MOT正式指标。
- Qwen3 Embedding/Reranker重新读取官方教程和支持矩阵：🔵代表实验性支持，非发布认证；示例用0.6B，8B只在权重表列出。Embedding用LeCaRDv2/batch2，Reranker代码随后将任务覆盖为MultiLongDocReranking；随机200token测速命令未附结果表，双句响应/relevance_score不当作质量基准。
- LSTM固定原提交同时读train_ctc/test_ctc，确认CUDA训练和测试入口；原文无GPU最终CER/WER或吞吐结果，不能用NPU权重/数字代替。MeshGraphNets原论文表1/附录A.5.1补单V100、图重算/重网格计时及带10^-3单位的rollout误差。
- MapAnything重新核对论文v3与当前模型/V1及许可证：13个训练集与Apache版6个分开；表2两视图仅图像指标与附加相机内参分开。图S.1为H200，当前README另列V1曲线，不用近似读图值充当精确时延。Qwen2.5-VL-7B补Drive-KD第三方A100-PCIE-40GB的32.0新生成token/s、TTFT0.87s，未报固定长度/batch的限制保留。
- 增补去重键改为编号+设备，允许同一条目GPU/NPU分别复核；新回归检查源码仓、CPU/NPU边界、模型规模、输入条件和单位。18项回归、可重复构建、git diff --check通过；本次未运行模型、下载权重/数据或认证S3。
- 待补仍有项目身份（DL-foils、MLOps训推一体框架）、未取得的专项资产和完整测试协议。保留完整目标，不把本批10项修正当作全部完成。
