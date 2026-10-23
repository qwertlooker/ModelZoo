# GPU补查记录

日期：2026-10-23。原始119项全部保留，GPU/NPU五项证据分别列出。

上一版GPU结论主要由上游入口推导，缺少逐项查询，不能作为完整核查。本轮重新读取原始仓库、模型卡、官方文档和论文，更新119项GPU记录。87项已找到明确官方GPU实现；其余项分别标注社区实现、指定模块支持、平台渲染或交付未确认。该计数仅针对代码支持，不能解释成权重、数据、性能和精度全部齐备。

## 主要补充和纠错

| 条目 | 本轮确认 | 原始证据 |
|---|---|---|
| FastWAM | 官方CUDA代码；H20端到端210ms、4090为110ms；Optional-IDM权重的LIBERO平均98.55%，First-frame为97.75%，40任务×50次 | [官方README](https://github.com/yuantianyuan01/FastWAM/blob/main/README.md) |
| G0.5 | CUDA12.8；4090推理、A100/H20微调；显存要求不当作速度 | [官方README](https://github.com/OpenGalaxea/GalaxeaVLA/blob/main/README.md) |
| GR00T N1.6 | 对应历史版本torch.compile：4090端到端44ms/22.8Hz；H100为38ms/26.3Hz | [对应版本README](https://github.com/NVIDIA/Isaac-GR00T/blob/ead52833afbbf4243f8cd5e7664f48a94de03b19/README.md) |
| OpenVLA | CUDA推理、A100-80GB LoRA；7B和完整微调用prismatic权重分开 | [官方README](https://github.com/openvla/openvla/blob/main/README.md) |
| Thinker | 4B权重已经公开；推理脚本支持cuda:0；Thinking版待发布 | [推理源码](https://github.com/UBTECH-Robot/Thinker/blob/main/inference.py)、[权重](https://huggingface.co/UBTECH-Robotics/Thinker-4B) |
| Thinker-WM | 另有专项仓库和权重，不能与Thinker-VLA合并 | [官方README](https://github.com/UBTECH-Robot/Thinker-WM/blob/main/README.md)、[权重](https://huggingface.co/UBTECH-Robotics/Thinker-WM) |
| WALL-WM | CUDA训练和评估代码已公开；预训练checkpoint仍Coming Soon | [官方README](https://github.com/X-Square-Robot/wall-wm/blob/main/README.md) |
| WoW | Wan demo明确CUDA；多规模权重及基准样例公开；完整训练流程和WoWBench指标仍列路线图 | [CUDA源码](https://github.com/wow-world-model/wow-world-model/blob/main/demo/wan_infer_demo.py)、[发布范围](https://github.com/wow-world-model/wow-world-model/blob/main/README.md) |
| BGE-M3 / Reranker | 已进一步定位CUDA设备选择源码，超出仅列仓库入口 | [M3](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/inference/embedder/encoder_only/m3.py)、[reranker](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/inference/reranker/encoder_only/base.py) |
| Swin-Unet | V100训练代码公开；预训练Swin权重公开；作者明确最终分割模型不能分享 | [官方README](https://github.com/HuCaoFighting/Swin-Unet/blob/main/README.md) |
| LSTM | 固定提交中use_gpu选择CUDA并迁移模型/输入；TIMIT dev与test分开 | [固定源码](https://github.com/Diamondfan/CTC_pytorch/blob/eddd2550224dfe5ac28b6c4d20df5dde7eaf6119/timit/steps/train_ctc.py) |
| YOLOv8 | 检测640输入，A100 TensorRT：n为0.99ms、s为1.20ms；COCO AP分别37.3、44.9 | [官方模型表](https://docs.ultralytics.com/models/yolov8/) |
| RT-DETR | 第一代T4 TensorRT FP16：R18为217FPS/AP46.5；R50为108FPS/AP53.1 | [官方模型表](https://github.com/lyuwenyu/RT-DETR/blob/main/README.md) |
| SAM2.1 Tiny | A100、torch2.5.1/CUDA12.4全模型编译91.2FPS；视频J&F不与NPU图像mIoU混用 | [官方说明](https://github.com/facebookresearch/sam2/blob/main/README.md) |
| AirSim / Gazebo+PX4 | GPU渲染支持，不能认定为CUDA飞控或模型推理实现 | [AirSim FAQ](https://github.com/microsoft/AirSim/blob/main/docs/faq.md)、[Gazebo渲染](https://gazebosim.org/docs/garden/troubleshooting/) |

## 仍未确认的内容

- 14项GPU交付未确认：AirVLA、RT-2、Spirit v1.6、Thinker-VLA、WALL-A、WALL-B、Z-1、Bracken、UNet-ESH、CBBA、CoordField、DL-foils、MLOps训推一体框架、RALLY。各项已记录实际找到的入口或名称歧义；不把论文、公司披露、CPU实现写成CUDA交付。
- Z-1已定位为π0.5+GRPO的RoboCasa论文，24任务80.6%；全文没有找到GPU/H100硬件描述或实现仓库链接，未将该结果认证为GPU基准。
- DL-foils多种拼写和airfoil组合检索仍无法唯一定位；同名描述逻辑算法DL-Foil不计入。
- 性能或精度字段写“待核对/未核实”的，代表本轮没有完成该数值及条件的确认，不代表公开资料一定不存在。已找到代码不等于整行五项验收完成。
- 本轮HTTP检查403个链接，368个成功；32个Hugging Face请求返回429，另有证书、567和403各1个。不把限流/访问限制当作资产缺失；没有下载权重或数据文件。

ModelZoo master已重新在线查询，仍为`dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202`。本轮修改GPU证据和展示逻辑，NPU既有固定提交证据保留；未运行GPU/NPU模型。
