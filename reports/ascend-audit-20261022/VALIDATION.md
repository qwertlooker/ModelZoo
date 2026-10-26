# 修正版验证记录

日期：2026-10-26，Asia/Shanghai；首次全量基线为2026-10-23。

- 119项按原始编号逐项保留；代码、权重、数据、性能、精度分别记录证据类型和链接。
- 27项回归测试通过：覆盖顺序、字段结构、非认证边界、易混指标回归、原始资源边界、固定ModelZoo文件存在、CSV/Markdown完整性、链接清单覆盖、GPU重查、后续证据增补、发布状态、计时范围、设备/仓库归属及可重复构建。
- 当前链接清单691个：656个有成功HEAD记录、35个未确认（25个Hugging Face限流429、5个403、2个G0.5授权401、1个404、Kvasir-SEG证书错误1个、华为云567共1个）。本批检查10个新增URL，全部成功；移除1个无当前字段引用的URL记录。其他URL保留原检查时间，HTTP状态不代替正文证据。
- 固定提交链接240个；动态或未版本化链接451个。数量按URL路径形式统计，含同页不同锚点，不表示全部外部来源已冻结。
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

## 第六批：机器人、科学模型与强化学习基准

- 本批补核7项GPU（OpenVLA、Spirit-v1.5、AlphaFold3、RFdiffusion2、PPO、TimesNet、PaddleOCR-VL-1.6）及2项NPU（MADDPG、QMIX）。累计增补100个设备条目，涉及95个模型；source_checks含98次源码/图片/论文正文读取记录。历史覆盖数字不表示五项资料全部闭合。
- OpenVLA补入四套LIBERO微调权重及论文表12成功率，写明每套独立微调及评测预算；Spirit-v1.5直接查看官方结果图，记录50.33%及2026-01-11快照，未臆测试验次数或测试GPU。
- RFdiffusion2补入AME的41/41位点覆盖及RMSD/碰撞门槛，区分单设计成功率、位点覆盖与湿实验酶活；AlphaFold3补核PoseBusters v1/v2规模及2019截止模型边界，当前开放参数同协议汇总分仍待确认。
- TimesNet补入论文v3表13的指定输入/预测长度MSE/MAE；PPO补入SB3的PyBullet回报，明确2M步、6种子及设备未注明，不能冒充GPU实测。
- MADDPG/QMIX按MindRL固定commit 7f55a67f90d248e8c535a2b6e61a6d68ca2c3c2d读取README、train.py及配置。QMIX当前README仅写GPU/CPU，源码接受Ascend；历史90%/-145与-140未绑定卡型，不能认作当前NPU实测。
- 用户明确表示不知道DL-foils、MLOps训推一体框架对应对象，先不核查。两项保留原始位置和已有检索记录，GPU/NPU结论均标暂缓，其余117项继续核查。
- 19项回归通过；新增覆盖位点/设计成功率、设备未注明回报、模型训练截止、文档/源码冲突及用户暂缓范围。构建重放和git diff --check通过。未运行模型，未下载权重/数据。
- 远程仍为本项目origin。再次普通push --dry-run因pcb1990无qwertlooker/ModelZoo写权限返回403，未实际推送、改凭据或改远程。

- 继续补查TimesNet原论文表11：TITAN RTX 24GB、ETTh1补全任务的0.024–0.073秒/迭代及显存，未写成单样本推理延迟；补核PaddleOCR-VL-1.6论文表2的96.33综合分及四项分数，1.6专属GPU计时仍未在原论文确认。

## 第七批：专项NPU源码与版本边界

- 重新在线查询ModelZoo master得到153c97646d940f0eefefe81cbc891693d7243eaa。GitCode compare确认相对原dd592快照仅1个提交、8个文件，truncated=false：新增DEIMv2目录并修改模型列表；119项清单不含DEIMv2，已有模型文件未变，保留原固定证据。完整变更路径记录在evidence_updates.json的master_review。
- AlphaFold3读取专项README、run_alphafold.py和参数映射源码，补明确Atlas 800T A2/FP32、MSA/模板数据库、Google参数授权；2K Evo Attention与默认<1000的文档范围分开，confidence不当作真实结构质量指标。
- PINN原模型总表引用不能证明具体实例，改为SciAI XPINNs及MindFlow Poisson2D专项源码。GPU补SciAI历史V100/MindSpore1.2数据和训练/评估规模；不将GPU结果写入NPU，不将变体指标覆盖整个PINN方法族。
- TimesNet补PaddleX预测器、npu:0入口、scaler.pkl要求、ETTH1 test.csv。Tesla T4的15.19/13.77ms写入GPU并与原PyTorch逐迭代计时分开；NPU列明确没有独立耗时表和迁移精度对照。
- 官方训练参数URL TimesNet_pretrained.pdparams经HEAD及Range GET均404；推理包链接可访问。两设备权重列分别说明，不认定训练资产齐备；没有下载模型资产。
- 累计104个设备增补条目，涉及96个原始条目；20项回归通过。保持119项顺序、GPU/NPU五维分列及两项用户暂缓范围；表格测试不构成模型验证，完整目标仍未完成。

- 本批推送前再次确认origin为https://github.com/qwertlooker/ModelZoo.git；push --dry-run仍报Permission denied to pcb1990及HTTP403。仅保留本地提交，未实际推送或更改远程/凭据。

## 第八批：Multimer、ThermoMPNN、PatchCore及RT-DETR专项复核

- ModelZoo master再次查询仍为153c97646d940f0eefefe81cbc891693d7243eaa；保留前批已核实的变更边界。本批只修订报告，未运行任何模型或外部训练/安装命令。
- AlphaFold-Multimer：固定MindScience r0.7 HEAD b233dc1ab152dfe29e0ff2cafb55c780dff34c67，读取模型卡、Multimer实现、predict_256配置及910B单卡测试。训练方法为空；测试仅断言6T36 confidence>92，注释虽提耗时，代码没有计时断言。置信度不视为真实结构精度。参数及示例特征URL HEAD均200，未下载校验。
- ThermoMPNN：固定原适配提交读取device_utils.py、训练及推理入口。发现NPU加速器缺失时可返回CPU，README的torch2.10与脚本前提2.5.1不一致；0.4x/0.7x是日志示意，不能作为Spearman实测。2OCJ 3880条一致性声明仅展示5行，缺完整误差产物与全测试集质量表。
- PatchCore：固定main HEAD ddca1579c779b889443ccaecb2010579d0d545e4，仓库仅README.md及.gitattributes。指导要求手工添加transfer_to_npu、不启用FAISS GPU；非完整源码补丁交付。保留社区AUROC/FPS来源，撤回此前E2E定性，因缺计时起止和重复次数不能确定端到端范围。
- RT-DETR：读取第一代NPU支持清单及检测预测器，区分Runner/Transformers路径；R50训练参数及推理包HEAD均200。H/L/R18/R50/X的COCO发布指标不标作NPU独立复测，不借用v2数据。
- 累计108个设备增补条目，涉及98个原始条目；source_checks保留118次正文读取记录。新增13个URL可访问，现共614个链接（580可访问、34未确认）；21项回归、重复构建通过。链接可访问不代表资产已完整下载或模型已验证。
- 原始119项和设备五维分列不变；两项名称不明条目按用户要求暂缓。仍有只取得支持记录、公开指标缺硬件条件、专项资产未发布等缺口，未将本批修订视为全量完成。

- 本批再次只读确认origin归属并执行普通push --dry-run，仍因pcb1990无qwertlooker/ModelZoo写权限返回403；未实际推送或更改远程/凭据。

## 第九批：ESM2、DINOv2、Depth Anything V2社区源码核查

- ModelZoo master再次查询仍为153c97646d940f0eefefe81cbc891693d7243eaa。固定三个社区HEAD，读取实际脚本、JSON报告和日志；未运行不可信项目、下载权重或使用社区宣称的SUCCESS作为验收。
- ESM2固定e1b33d04fc0f603e925bbfd6ef28499d5eb6fd64：补650M掩码语言模型实际入口与逐元素比较公式。报告仅单条输入[1,62,33]，没有正式计时；不借用另一个同名适配仓243ms。README称第48位mask真实残基N，UniProt P69905原序列对应D，且mask前后缀均一致，故撤回生物预测正确性说明。数值一致性结果仍按社区报告引用，不视作下游任务质量。
- DINOv2-large固定7159789e728fc6cd9edb62e5b626bc418e0338b1：当前report.json为FP16、batch1、224随机输入、10轮同步均值11.805ms、cosine0.9999527，替换旧12.18ms/0.999936。读取源码确认3次预热、CPU FP32对照、未固定随机种子，以及精度比较异常被捕获后仍可写SUCCESS。0.0047%是1-cosine，不是任务错误率。
- Depth Anything V2：原vipe-v1.0仓只有说明；沿README追溯到vipe-npu固定47694287e5a3a2ed17b0a3bc5ec76152b3dfa6de，读取DAV2封装、run.py、test_infer.sh。必须经入口启用transfer_to_npu，缺torch_npu可转CPU，DAV2含CPU/NumPy预处理；不称独立纯NPU实现。相对/Hypersim/VKITTI权重分开，不把分支存在当成全覆盖验证。
- ViPE演示59帧及SLAM/对齐/可视化日志不能作为DAV2独立速度；test-all检查视频文件且可SKIP，无AbsRel/δ1或深度误差阈值，PASS不代表模型精度通过。
- 累计111个设备增补条目，涉及99个原始条目，source_checks共132条。新增16个链接均可访问；当前630个URL中596可访问、34未确认。22项回归、可重复构建及diff检查通过；资料分析与上机验证分开，完整目标仍未完成。

- 本批push --dry-run仍因pcb1990无qwertlooker/ModelZoo写权限返回403；只保留本地提交，未实际推送、改远程或改凭据。

## 第十批：BLUP、GBLUP、ssGBLUP算法范围与数值来源

- ModelZoo master再次查询仍为153c97646d940f0eefefe81cbc891693d7243eaa；固定三个社区仓当前HEAD，读取README、求解与计时源码、测试及结果JSON，没有运行外部算法代码。
- BLUP-NPU固定5dfd7f7a8baa041d8de80a4086823c2d53b4a43b：Python包装器依赖仓外../../npu/libbglr_npu.so，该提交未交付库或底层C/C++构建源码，deploy.sh不获取/编译它。G计算失败能回退NumPy；Cholesky、谱分解与Gibbs在CPU。README的XXᵀ 11.13ms及约228倍是局部kernel结果，不能称完整预测加速；矩阵数值误差不当作遗传预测质量。
- GBLUP固定6b1724a553ed5b9596961457790c324677cfc935：确认NPU/CPU混合路径及CPU上的REML/ML优化。benchmark计划三次均值，但仓无对应计时报告。发布CPU/NPU各599维u，直接算术核对得到RMSE1.127531262e-5、最大绝对差3.953979574e-5；README的9.75e-6是对R参考，二者参考对象不同。派生结果及固定源链接记录于derived_checks，不标作独立模型运行。
- ssGBLUP-npu_4212固定1714c823fec4150fbee95ac4907cebf835011651：README明确H矩阵/SNP整合未实现，实际为系谱animal-BLUP/MME/PCG。20/22方程求解0.3431/0.3805s，CPU均0.0009s，小样例NPU更慢；源码计时排除稠密化、上传和ridge构造。秩亏例全解差0.536、动物EBV差1.85e-6，测试只约束EBV并额外加正则，不能标完整ssGBLUP验收或全解等价。
- 三项无神经网络预训练权重，但仍需模型参数与数据；本批将这些条件写清，不用“不需要权重”代替完整算法/依赖交付。
- 累计114个设备增补条目、涉及102个原始条目、146条源码/正文读取记录。新增14个链接均可访问，现644个URL（610可访问、34未确认），23项回归及可重复构建通过。119项原始顺序和GPU/NPU分列不变，两项用户暂缓；仍未认定全量分析完成。

- 推送前再次只读确认origin归属；push --dry-run仍返回pcb1990无写权限/HTTP403。只保留本地提交，未实际推送、更改远程或凭据。

## 第十一批：LoRA/PEFT与InternVL3支持边界

- ModelZoo master仍为153c97646d940f0eefefe81cbc891693d7243eaa。分别查询PEFT、ms-swift、MindSpeed-MM当前HEAD，并使用固定提交读取资料；没有安装依赖、训练模型或启动服务。
- LoRA/PEFT：PEFT固定a7e8ea4a12275b9b5846eb3e899e75ea6fdeeee1，读取infer_device及checkpoint格式；adapter参数必须配套原底座和adapter_config。ms-swift固定942be5a6120d46443f1b1940f2902c3757e312a5，官方NPU指导覆盖训练、合并及推理；明确该路线支持LoRA、不支持QLoRA，部分Transformers5 MoE fused-expert组合不稳定，不能外推全部方法/底座。
- 记录官方Qwen2-7B/blossom-math-v2、5epochs的910B3单卡8h/28GB、4卡DDP2h/22GB示例，但缺完整batch/长度及多次统计，不作为LoRA统一吞吐。Qwen3.5-4B的300步loss/grad稳定只作为该组合稳定性记录，未当作任务精度结果。
- InternVL3：ms-swift当前支持表列8B SFT组合。另读取官方MindSpeed-MM历史提交76a1db58a2f87c4ae6ec1bb6f495ec17ab805542的README、8B微调/推理脚本和配置；当前HEAD 2d9db8318ef74661c2567b53d90daa8ca51facdf对应examples/internvl3目录404，需固定历史版本，未宣称当前master仍保留该路径。
- 历史8B权重需InternVLConverter转换，PP4训练/单卡推理层切分不同，tokenizer使用原始HF目录。训练数据为InternVL-Chat-V1-2-SFT-Data中的ShareGPT4V caption100k，448图像/动态切片1–12；没有完整NPU任务基准表。脚本seq4096、BF16等配置不当作性能数值。
- 同步读取dummy_optimizer特性说明：它用于参数冻结的pipeline stage，不能因示例含该参数就误判全模型没有权重更新。
- 累计116个设备增补条目、涉及103个原始条目、155条正文读取记录。新增9个URL均可访问，现653个URL（619可访问、34未确认）；24项回归、可重复构建及diff检查通过。保留原119项与两项用户暂缓，尚未将全量目标标完成。

- 本批普通push --dry-run仍因pcb1990无qwertlooker/ModelZoo写权限返回403；保留本地提交，未实际推送或修改远程/凭据。

## 2026-10-26第十二批复核：分割模型与VGGT

- 复核#68 DeepLabv3+、#71 SegFormer、#72 Swin+FPN、#115 SAM2、#119 VGGT的GPU/NPU五项字段，共10条设备更新；累计126条设备更新、108个不同模型，171条正文读取记录。此数量不代表其余模型或全部条件已经验收闭合。#86、#100继续按用户要求暂缓。
- GPU补充4条OpenMMLab具体权重直链与相应模型/数据集配置，SAM2明确2.1 Tiny；VGGT新增原论文v1单H100/FlashAttention v3的骨干耗时、显存及任务指标。其10/50/100帧计时为0.14/1.04/3.12秒，不包含所有任务头，不能引用为完整重建速度。
- NPU复核5份ModelZoo README、DeepLab训练README及3份后处理源码；VGGT补读DrivingSDK固定提交训练README和8卡性能脚本。后者不计作ModelZoo仓内交付。
- DeepLab的165.545及Swin的5.17/T4 2.5原表均缺单位，明确保留为表值；DeepLab训练252.83 FPS不混同推理。SegFormer补齐GPU Cityscapes同checkpoint76.54，NPU75.94，未闭合协议前不认定转换损失；原表10.61/10.65未注明GPU硬件。
- Swin后处理只调用bbox评估，缺失输出分支尝试移除评估项，需要额外检查5000张完整性；不能认定mask AP已验收。SAM2源码使用COCO非crowd真实框提示，按实例平均IoU；0.7815/0.7830与GPU视频J&F不同，encoder/decoder速度分别记录。
- VGGT NPU25.04由全局batch120除以Batch Time计算，脚本标images/sec但未见乘每样本2帧；属于训练吞吐，不能视作视频帧率。原表竞品A没有GPU型号，loss0.0165/0.0164不能代替几何任务精度。
- 执行build_report.py、test_report.py，25项通过；119行25列保持完整，CSV/Markdown按GPU/NPU分别输出五项及链接。新增URL仅HEAD检查，不下载权重、不执行上游脚本、不做模型实测。

## 2026-10-26第十三批复核：Fast-WAM、G0.5、WoW和PaddleOCR

- 四项各更新GPU/NPU五条件，新增8条设备更新。累计134条设备更新、112个不同条目、192条正文/JSON读取记录；#86、#100继续按用户要求暂缓。条目更新数量不等于全部条件闭合，尚需完成其余条目及整体一致性复核。
- ModelZoo master重新执行git ls-remote，仍为153c97646d940f0eefefe81cbc891693d7243eaa；已有compare记录证明相对固定证据快照仅新增DEIMv2，未改变本批OCR目录。
- PaddleOCR补读官方文本检测/识别文档及ModelZoo v5/v6 README、v6 infer.py和paddleocr.patch。GPU补入T4 v5 server检测89.55/70.19ms、识别8.46/2.36ms；A100静态引擎v5检测/识别26.91/11.37ms、v6 medium16.02/5.38ms，注明单图纯推理而非整产线速度。不同自建质量集、示例集不能直接对齐。
- 修正v6精度对照原错误：46.73%/77.27%对应CPU，不是A10。v5对照才为A10。NPU v5明确EP=300I DUO、RC=310P RC176T；v6为800I A2。v6评测patch固定InferSession(0)及修改ParallelEnv私有字段，记录源码风险，未实际运行或修改上游。
- Fast-WAM读取社区NPU固定提交的README_NPU、VALIDATION_REPORT、3份结果JSON及评测脚本。20.27→33.30是排除缓存构建的训练吞吐，基线60步/优化100步；200配对episodes并非当前GPU Optional IDM的2000次结果，MuJoCo版本也不同。公开NPU评测权重缺失，脚本包含服务器路径，不认定clean-room可复现。社区证据不计ModelZoo仓内适配。
- G0.5核实两个具体权重路径、ActionCodec及stats需求；模型卡明确登录接受条件，HEAD401作为授权边界。论文补充RoboTwin clean/rand及LIBERO试验次数，DROID82.5%明确为DROID后训练后的未见环境转移，不能写成从未训练DROID的纯零样本。
- WoW补查原论文，表1WoW-DiT/Wan的人工/自动Overall为12.37/34.97，Cosmos2为13.34/41.07，是组合质量分数而非成功率；README仍列训练与评测实现待发布，不能将论文结果缺失与代码缺失混为一谈。显存驻留建议不等于吞吐基准，15FPS保存帧率不等于生成速度。
- G0.5与WoW重新做中英文Ascend/NPU检索，未取得专项完整模型交付证据，查询记录写入search_checks；没有证据的性能/精度字段保留空链接并明确未确认，禁止为满足结构检查填入无关上游链接。回归相应校验空证据必须具有明确未确认角色及说明。
- 执行build_report.py、test_report.py，26项通过；119行25列维持不变，新增回归覆盖CPU/A10归属、纯推理/产线计时、G0.5授权、续训与完整任务范围。未执行模型推理、训练或资产下载。

## 2026-10-26第十四批复核：系列、平台、算法及VGGT推理补漏

- 本批补核cosmos、AirSim、GPR、Gazebo+PX4 SITL、ORB-SLAM3，共10条设备更新；追加VGGT既有GPU/NPU字段，累计144条设备更新，117个未暂缓条目均有后续修订记录，另2项按用户要求暂缓。这个覆盖数只证明曾修订，不能当作五条件分析完成证明。
- 最新ModelZoo master仍为153c97646d940f0eefefe81cbc891693d7243eaa。cosmos按原编号#4–7具体成员汇总，DrivingSDK固定清单四者Released=N、Active=Y，开发自验不等于版本测试或ModelZoo仓内交付。
- AirSim与Gazebo+PX4是仿真平台，权重和统一模型准确率不适用；区分Unreal/OpenGL GPU渲染与CUDA模型计算。GPR具体核实GPyTorch CUDA RBF教程；新增OptEx源码的NPU设备/MaternKernel线索，但不能认定通用ExactGP业务回归验收。
- ORB-SLAM3补读官方CMake和ORBextractor源码，未配置CUDA且使用cv::Mat/FAST/CPU描述子循环；可视化不能证明SLAM计算有GPU后端。EuRoC坐标系及TUM-VI真值覆盖限制写入数据/精度列。搜索命中的Ascend Engineering不是华为昇腾，未纳为证据。
- 空间智能原仓归档说明未列ORB-SLAM3，却揭示此前VGGT推理路线遗漏。继续访问当前cann-recipes-embodied-ai，并查询固定master 21a5ed789ceb2a4b72277573f5ec5ec531b92ee1，读取推理README、精度/性能文档及demo源码。此为外部CANN路线，不计ModelZoo仓内交付。
- VGGT当前README支持A2/A3；任务表NPU/GPU AUC@30为91.1/90.5，ETH3D Overall0.456/0.464，DTU Overall1.0072/1.0244。DTU与论文0.382差异明显，原文承认协议差异，不认定复现论文。BF16的0.919→0.911为0.8个百分点，原文称0.5%以内存在算术矛盾；部分INT8结果另列。
- VGGT性能表1121.09ms是kitchen25图模型计时，文档说8卡机器但未明确该行启用卡数，不能写为八卡绝对延迟；代码同步计时且排除读图/预处理，单卡多次均值、多卡丢前2次，不是三次独立中位数。保留DrivingSDK训练吞吐，严格分开两条路线。
- 执行build_report.py、test_report.py，27项通过；119行25列不变。新增回归覆盖平台/算法不适用项、OptEx线索边界、cosmos发布状态、VGGT新增推理及数值问题。未执行任何模型、平台运行、权重/数据下载。
- 全表内容复审发现，NPU侧仍有58个条目包含历史概括或未闭合字段（关键词筛选仅用于安排复核，不等于这些模型无适配）。编号：2,3,4,5,6,7,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,27,28,29,30,31,32,33,34,35,37,38,39,49,53,54,57,60,62,63,64,66,67,74,75,77,79,82,83,87,89,90,97,101,102,112,117,118。后续须逐项补查，不能因117项有更新记录或27项检查通过而宣称完整分析结束。
