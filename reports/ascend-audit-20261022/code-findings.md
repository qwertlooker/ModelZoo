# 代码与文档交叉核查重点

以下均基于固定提交 `dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202`。这是静态观察，不是运行故障复现记录。没有实际加载权重、构建OM或执行NPU评测。

| 对象 | 精确位置与证据 | 观察 | 对核查结论的影响 |
|---|---|---|---|
| BGE-M3 | [infer.py L189–220](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/built-in/embedding/bge-m3/infer.py#L189) | 显式请求dense、sparse、ColBERT三种输出，并展示相似度；计时循环每轮调用两次encode | 三类输出有实现入口，但没有任务评测集、检索指标和NPU对照；E2E数字须按两次调用的工作量解读，不能作为单条query延迟 |
| BGE-Reranker | [infer.py L189–197](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/built-in/embedding/bge-reranker-v2-m3/infer.py#L189) | 示例调用compute_score(normalize=True)并计时 | 有打分功能入口，不等于完成nDCG/MRR任务基准；README性能表模型名误写不能直接作为独立可信验收记录 |
| Pi-0.5 LeRobot | [infer.py L41–73](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/built-in/embodied_ai/vla/pi05_lerobot/infer_with_torchair/infer.py#L41) | 明确加载NPU模型，直接构造LeRobotDataset("lerobot/libero")，取样例推理计时 | 运行包含在线数据获取依赖；离线准备、数据revision和manifest尚需补证。单frame时延与仿真成功率是两种证据 |
| Pi-0.5 OpenPI | [inference_pytorch.py](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/built-in/embodied_ai/vla/pi05_openpi/inference_pytorch.py) | 使用transfer_to_npu、预热和record次循环均值；未提供独立任务成功率对照表 | 不能直接继承LeRobot路线的精度结论；应固定转换后权重和归一化统计，独立评测 |
| SAM2 | [sam2_coco_metric.py L91–219](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/built-in/cv/SAM2/sam2_coco_metric.py#L91) | COCO图像按实例提示分割，计算IoU并取平均；encoder可batch、decoder逐实例 | 这是图像提示分割评估，不能用来认证视频分割；必须保持提示生成协议一致 |
| Qwen2.5-VL | [7B GRPO 16p脚本](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/built-in/rl/VeRL_for_PyTorch/test/train_qwen2_5_vl_7b_GRPO_full_16p.sh) | 存在具体训练配置和geo3k任务入口；主README表为Throughput与Max Training TimeSteps | 能证实专项训练交付存在；表未注明吞吐单位，也未给该任务精度对照，不能视为完整质量验收 |
| Qwen2-VL | [7B README](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/MindIE/MultiModal/Qwen2-VL/Qwen2_vl_7b_instruct/README.md) | 服务配置modelName为qwen2_vl，OpenAI示例请求model为internvl | 使用严格模型名匹配的服务时可能请求失败或无法对应目标模型；需统一名称并实际重放请求 |
| Qwen3-30B | [README训练配置章节](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/built-in/foundation/Qwen3-30B-Torchtitan/README.md) | 明确单机8卡默认配置OOM，建议n_layers=24；安装示例有numpy==1.2x和框架版本不一致 | 默认命令存在可执行性缺口；减层模型不能证明完整30B配置验收；不照搬安装示例 |
| LLaVA | [README L4、L71、结果表](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/built-in/mm/LLaVA/README.md#L4) | 声明不再维护，转向MindSpeed-MM；包含torch==2.1.0+cpu安装示例；仅列训练samples/s | 需重新验证CANN配套环境；已有目录不能证明现环境可安装，也没有航拍业务质量结论 |

本次没有执行第三方入口的 `--help`，因为这些入口会导入当前Windows控制机没有的NPU/官方评测依赖；也没有为完成审计而修改或安装系统包。回归只针对本次审计工具和输出完整性，不替代模型运行。
