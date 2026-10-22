## ACL_PyTorch/built-in/audio/LSTM

文件清单：

```text
ACL_PyTorch/built-in/audio/LSTM/LICENSE
ACL_PyTorch/built-in/audio/LSTM/LSTM_get_info.py
ACL_PyTorch/built-in/audio/LSTM/LSTM_postprocess_data.py
ACL_PyTorch/built-in/audio/LSTM/LSTM_preprocess_data.py
ACL_PyTorch/built-in/audio/LSTM/LSTM_pth2onnx.py
ACL_PyTorch/built-in/audio/LSTM/ReadMe.md
ACL_PyTorch/built-in/audio/LSTM/acl_net.py
ACL_PyTorch/built-in/audio/LSTM/lstm_atc.sh
ACL_PyTorch/built-in/audio/LSTM/lstm_onnx_infer.py
ACL_PyTorch/built-in/audio/LSTM/prepare_data.sh
ACL_PyTorch/built-in/audio/LSTM/pyacl_infer.py
ACL_PyTorch/built-in/audio/LSTM/requirements.txt
```

### [ACL_PyTorch/built-in/audio/LSTM/ReadMe.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/built-in/audio/LSTM/ReadMe.md)

文本SHA256：`3c7427cec946b9d44eb051d3fc6655a180f6441c733b3f38b9f5595b4bbee720`

- L15: - [准备数据集](#section183221994411)
- L18: - [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)
- L32: url=https://gitcode.com/ascend/ModelZoo-PyTorch.git
- L46: | input    | FLOAT32 | batchsize x 390 x 243 | ND         |
- L53: | output1  | FLOAT32  | 195 x batchsize x 41 | ND           |
- L64: | 固件与驱动                                                   | 22.0.4  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
- L65: | CANN                                                         | 6.3.RC1 | -                                                            |
- L66: | Python                                                       | 3.7.5   | -                                                            |
- L67: | PyTorch                                                      | 1.8.0   | -                                                            |
- L68: | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |
- L79: git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
- L96: git clone https://github.com/kaldi-asr/kaldi
- L150: ## 准备数据集<a name="section183221994411"></a>
- L152: 1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）
- L154: 本模型支持timit语音包的验证集。timit数据集与训练对齐，使用训练提供的语音数据包。需用户自行获取数据集，并将数据集命名为data.zip，并上传数据集data.zip至服务器模型源码包所在目录。数据集结构如下。
- L163: 2. 数据预处理，将原始数据集转换为模型输入的数据。
- L166: 1. 解压数据集。
- L180: 在当前目录下会生成tmp文件夹和在data文件夹下生成dev,test,train三个数据集文件夹。
- L195: 使用data文件夹下的dev数据集进行验证。
- L211: 使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。
- L213: 1. 获取权重文件。
- L215: ctc_best_model.pth权重文件[下载链接](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/LSTM/PTH/ctc_best_model.pth)。在当前目录下创建checkpoint/ctc_fbank_cnn/目录并将权重文件移到到该目录下。
- L250: | 0       310P3     | OK              | 15.8         42                0    / 0              |
- L251: | 0       0         | 0000:82:00.0    | 0            1074 / 21534                            |
- L253: | 1       310P3     | OK              | 15.4         43                0    / 0              |
- L254: | 0       1         | 0000:89:00.0    | 0            1070 / 21534                            |
- L280: 请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。
- L296: 3. 精度验证。
- L310: 4. 性能验证。
- L312: 可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：
- L323: # 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>
- L325: 调用ACL接口推理计算，性能参考下列数据。
- L327: | 芯片型号 | Batch Size | 数据集 | 精度                      | 性能          |
- L329: |   300I Pro        | 1          |   TIMIT         | CER:13.5491/WER:18.8003 | 5.3457 fps  |
- L330: |   300I Pro        | 4          |   TIMIT         | CER:13.5370/WER:18.7877 | 21.3256 fps |
- L331: |   300I Pro        | 8          |   TIMIT         | CER:13.5156/WER:18.7729 | 42.0914 fps |
- L332: |   300I Pro        | 16         |   TIMIT         | CER:13.5502/WER:18.8313 | 82.6230 fps |
- L333: |   300I Pro        | 32         |   TIMIT         | CER:13.5502/WER:18.8313 | 83.1510 fps |
- L334: |   300I Pro        | 64         |   TIMIT         | CER:13.5502/WER:18.8313 | 83.5096 fps |

## ACL_PyTorch/built-in/cv/Depth-Anything-3

文件清单：

```text
ACL_PyTorch/built-in/cv/Depth-Anything-3/README.md
ACL_PyTorch/built-in/cv/Depth-Anything-3/compare_onnx_om.py
ACL_PyTorch/built-in/cv/Depth-Anything-3/compare_torch_onnx.py
ACL_PyTorch/built-in/cv/Depth-Anything-3/depth_anything_v3.patch
ACL_PyTorch/built-in/cv/Depth-Anything-3/export_onnx.py
ACL_PyTorch/built-in/cv/Depth-Anything-3/fix_onnx_gather.py
ACL_PyTorch/built-in/cv/Depth-Anything-3/infer_om.py
ACL_PyTorch/built-in/cv/Depth-Anything-3/utils.py
```

### [ACL_PyTorch/built-in/cv/Depth-Anything-3/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/built-in/cv/Depth-Anything-3/README.md)

文本SHA256：`771e594399ff8525e14028d74277b356707e132b2e0fcb3684bdece021378856`

- L14: - [准备数据集](#section3-3)
- L17: - [模型推理性能&精度](#TOPIC_4)
- L30: [Depth Anything 3: Recovering the Visual Space from Any Views](https://arxiv.org/pdf/2511.10647)
- L35: url= https://github.com/ByteDance-Seed/Depth-Anything-3
- L47: | input    | RGB_FP32 | batchsize x 3 x 518 x 518 | NCHW         |
- L53: | depth | FLOAT32 | (batchsize, 1, 518, 518)  | NCHW，逐像素深度值（exp 激活，正值）。   |
- L54: | depth_conf | FLOAT32  | (batchsize, 1, 518, 518)  | NCHW，逐像素深度置信度（exp+1，取值 [1, +∞)）。   |
- L55: | extrinsics | FLOAT32  | (batchsize, 1, 3, 4)  | 行优先矩阵，相机外参（world-to-camera 的旋转平移矩阵）。   |
- L56: | intrinsics | FLOAT32  | (batchsize, 1, 3, 3)  | 行优先矩阵，相机内参矩阵（fx、fy、cx、cy，单位：像素）。   |
- L67: | NPU | Atlas 800I A2 * 1 |
- L68: | 驱动固件 | 25.5.2 |
- L69: | 容器镜像 | swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-onnx-inference:cann8.3.rc1_torch2.1.0-800I-A2-openeuler24.03-py3.11-aarch64|
- L96: git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
- L101: git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
- L115: ## 准备数据集<a name="section3-3"></a>
- L116: 1. 获取原始数据集
- L119: 精度与性能评测使用 HiRoom 室内场景数据集，数据集存放在 workspace/benchmark_dataset 文件夹中
- L121: https://www.modelscope.cn/datasets/depth-anything/DA3-BENCH
- L124: ./workspace/benchmark_dataset/hiroom/
- L145: 使用PyTorch将模型权重文件.safetensors 转换为.onnx文件，修补后使用ATC工具将.onnx文件转为离线推理模型文件.om文件。
- L147: 1. 获取权重文件
- L149: 从 HuggingFace 仓库 [depth-anything/DA3-SMALL](https://huggingface.co/depth-anything/DA3-SMALL) 获取权重文件（国内可用 hf-mirror.com 镜像下载）,放入如下目录
- L168: | weights    | pytorch权重文件 |
- L170: | height/width    | 输入高宽（须为14的整数倍）                            |
- L201: ├──compare_torch_onnx.py # pytorch vs onnx 精度对比脚本
- L202: ├──compare_onnx_om.py    # onnx vs om 精度性能对比脚本
- L206: ├──workspace/benchmark_dataset/hiroom/
- L227: | om    | om 权重文件                            |
- L234: # 模型推理性能&精度<a name="TOPIC_4"></a>
- L236: 在 **HiRoom** 数据集（518x518 输入）上的精度/性能评测,分別评测pytorch(cpu)、onnx(cpu)、om(npu)上的精度和性能。精度指标为各后端输出与 PyTorch(CPU) 基准输出之间的一致性误差（MAE / 相对MAE）。
- L239: # 批量精度/性能对比
- L246: | 指标 | Pytorch(CPU) | ONNX (CPU) | OM (Ascend NPU) |
- L248: | **depth MAE** | 基准 | 2.1e-7 | 2.4e-4 |
- L249: | **depth_conf 相对MAE** | 基准 | <1e-5 | 5.8e-3 |
- L250: | **extrinsics MAE** | 基准 | <1e-5 | 4.9e-5 |
- L251: | **intrinsics 相对MAE** | 基准 | <1e-5 | 5.3e-4 |
- L252: | **推理性能** | 39.37 s/it | **0.805 s/it**  | **0.014 s/it** |
- L255: >精度指标为与 PyTorch(CPU) 输出的一致性误差；推理性能为单张 518x518 图片的平均时延（OM 为预热后 20 次均值）。OM(Ascend Atlas 800I A2) 相对 ONNX(CPU) 加速约 58 倍。

## ACL_PyTorch/built-in/cv/GroundingDINO

文件清单：

```text
ACL_PyTorch/built-in/cv/GroundingDINO/README.md
ACL_PyTorch/built-in/cv/GroundingDINO/image_demo_npu.py
ACL_PyTorch/built-in/cv/GroundingDINO/install_requirements.sh
ACL_PyTorch/built-in/cv/GroundingDINO/mmcv_diff.patch
ACL_PyTorch/built-in/cv/GroundingDINO/mmdetection_diff.patch
ACL_PyTorch/built-in/cv/GroundingDINO/mmengine_diff.patch
ACL_PyTorch/built-in/cv/GroundingDINO/register_im2col_to_torchair.py
ACL_PyTorch/built-in/cv/GroundingDINO/register_roll_to_torchair.py
ACL_PyTorch/built-in/cv/GroundingDINO/requirements.txt
ACL_PyTorch/built-in/cv/GroundingDINO/video_demo_npu.py
```

### [ACL_PyTorch/built-in/cv/GroundingDINO/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/built-in/cv/GroundingDINO/README.md)

文本SHA256：`f43249382adcf751bee978aef4ef2d1deae619026028ba9b2aa8a1a16b5e0817`

- L9: - [2 性能](#2-性能)
- L19: url=https://github.com/open-mmlab/mmdetection
- L30: | 固件与驱动                                                       | 25.2.RC1 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
- L31: | CANN                                                            |  8.2.RC1 | 包含kernels包和toolkit包                                                                                                   |
- L32: | Python                                                          |  3.11 | -                                                                                                     |
- L33: | PyTorch                                                         | 2.1.0 | -                                                                                                     |
- L34: | Ascend Extension PyTorch                                        | 2.1.0.post13 | -                                                                                                     |
- L35: | 说明：Atlas 800I A2 推理卡和Atlas 300I DUO 推理卡请以CANN版本选择实际固件与驱动版本。目前Atlas 300I DUO仅支持推理图像。 |      \ | \                                                                                                     |
- L44: git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
- L50: git clone https://github.com/open-mmlab/mmdetection
- L54: 2. 下载相关权重和图片
- L55: - 下载[BERT权重](https://huggingface.co/google-bert/bert-base-uncased/tree/main)，并放置于mmdetection目录下
- L56: - 下载[MM-GroundingDINO权重](https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-b_pretrain_obj365_goldg_v3det/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth)，并放置于weights目录下
- L57: - 下载[animals.png](https://github.com/microsoft/X-Decoder/tree/main/inference_demo/images)，并放置于images目录下
- L58: - 下载NLTK权重（可选）。MM-GroundingDINO在进行Phrase Grounding推理时可能会进行名词短语提取，虽然会在运行时下载特定的模型，但是考虑到有些用户运行环境无法联网，因此可以提前下载。
- L59: - 方式一：下载[模型](https://www.nltk.org/nltk_data)到`~/nltk_data`路径下。
- L97: ├── bert-base-uncased //BERT权重
- L99: │   └── grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth //MM-GroundingDINO权重
- L138: 在推理开始后，首先会默认执行warm_up，目的是执行首次编译，首次编译时间较长，在warm_up结束后，会执行推理操作，并打屏计算结果和性能数据。
- L140: ### 2 性能
- L142: - 单图性能，以animals.png为例，测得以下数据：
- L145: |模型|芯片|E2E|forward|
- L147: |MM-GroundingDINO|Atlas 800I A2|877ms|274ms|
- L148: |MM-GroundingDINO|Atlas 300I DUO(单芯)|1378ms|740ms|
- L150: - forward性能在`mmdet/apis/det_inferencer.py`中的`preds = self.forward(data, **forward_kwargs)`前后打点得到。
- L152: - 视频性能：
- L156: |MM-GroundingDINO|Atlas 800I A2|1080P|1bs|140ms|
- L157: |MM-GroundingDINO|Atlas 800I A2|1080P|16bs|124ms|
- L159: - forward性能从`demo/video_demo_npu.py`执行结束打屏信息获取，对应单针耗时`per frame infer time`
- L160: - config配置文件`configs\mm_grounding_dino\grounding_dino_swin-t_pretrain_obj365.py`中的`scale=(800, 1333)`控制图片缩放，减小缩放比例可提升模型性能，已经测试coco数据集精度在(800, 1152)配置下精度mAP为0.522，(800, 1024)配置下精度mAP为0.52，具体配置选择由用户自行评估。
- L163: 1. mmcv源码安装报错参考：https://mmcv.readthedocs.io/zh-cn/2.x/faq.html

## ACL_PyTorch/built-in/cv/RT-DETRv2

文件清单：

```text
ACL_PyTorch/built-in/cv/RT-DETRv2/README.md
ACL_PyTorch/built-in/cv/RT-DETRv2/eval_om_coco.py
ACL_PyTorch/built-in/cv/RT-DETRv2/export_om.py
ACL_PyTorch/built-in/cv/RT-DETRv2/infer_om.py
ACL_PyTorch/built-in/cv/RT-DETRv2/patch/npuv1.patch
```

### [ACL_PyTorch/built-in/cv/RT-DETRv2/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/built-in/cv/RT-DETRv2/README.md)

文本SHA256：`b7a9da4cce2b8cb404c178335196187e548a55c558036162687dd5badc7b4ece`

- L10: - [准备数据集](##准备数据集)
- L13: - [模型推理性能](##模型推理性能)
- L19: RT-DETR（Real-Time Detection Transformer）是实时端到端目标检测模型，RT-DETRv2 在 RT-DETR 的基础上进一步优化了训练策略和检测性能。当前目录提供 Ascend 推理侧所需的补丁、PyTorch 转 ONNX/OM 脚本、OM 单图推理脚本以及 COCO 精度评估脚本，适用于 RT-DETRv2 模型的 Ascend 离线推理部署。
- L28: url=https://github.com/lyuwenyu/RT-DETR
- L37: url=https://gitcode.com/ascend/ModelZoo-PyTorch
- L50: | images | RGB_FP32 | 1 x 3 x 640 x 640 | NCHW |
- L51: | orig_target_sizes | INT64 | 1 x 2 | ND |
- L58: | labels | 1 x 300 | INT64 | ND |
- L59: | boxes | 1 x 300 x 4 | FLOAT32 | ND |
- L60: | scores | 1 x 300 | FLOAT32 | ND |
- L71: | CANN | 8.5.0 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
- L72: | Python | 3.12 | - |
- L73: | PyTorch | 2.9.0 | - |
- L74: | torch_npu | 2.9.0 | - |
- L87: git clone https://github.com/lyuwenyu/RT-DETR.git
- L100: │      ├── eval_om_coco.py          # OM 模型 COCO 精度评估脚本
- L118: https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench
- L126: ## 准备数据集
- L128: 1. 获取原始数据集。（解压命令可参考 `tar -xvf *.tar` 或 `unzip *.zip`）
- L130: 本模型精度验证使用 COCO val2017 数据集及其标注文件。请用户自行获取数据集，并上传到服务器目录（如：`/home/RT-DETR/rtdetrv2_pytorch/dataset/coco`）。
- L132: 数据集下载链接：
- L133: - `http://images.cocodataset.org/zips/val2017.zip`
- L134: - `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`
- L136: 数据集目录结构请参考：
- L158: 使用 PyTorch 将模型权重文件 `.pth` 转换为 `.onnx` 文件，再使用 ATC 工具将 `.onnx` 文件转换为离线推理模型文件 `.om`。
- L160: 1. 获取权重文件。
- L162: 准备与配置文件匹配的 PyTorch 权重文件，放入 `RT-DETR/rtdetrv2_pytorch` 目录或可访问路径下。
- L164: 下载路径：https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch
- L192: - `-r`：PyTorch 权重文件路径
- L223: - `-d/--data-path`：COCO 数据集根目录，脚本默认读取其中的 `val2017`
- L232: c.  性能测试。
- L238: d.  精度验证。
- L258: 当前导出的 RT-DETRv2 部署后处理默认输出 `0~79` 类别索引，评估 COCO 精度时建议加上 `--zero-based-label`。
- L261: # 模型推理性能&精度
- L263: ###性能
- L267: | RT-DETRv2 | 1 | 640x640 | 223 | Ascend910B4 |
- L268: | RT-DETRv2 | 1 | 640x640 | 217 | T4 GPU |
- L270: ###精度
- L272: | Model | 数据集 | APval | AP50val | Card |
- L274: | RT-DETRv2 | coco val2017 | 48.0 | 65.0 | Ascend910B4 |
- L275: | RT-DETRv2 | coco val2017 | 48.1 | 65.1 | T4 GPU |
- L282: | 开源代码仓 | RT-DETR 上游源码 | https://github.com/lyuwenyu/RT-DETR |
- L283: | 数据集 | COCO val2017 | http://images.cocodataset.org/zips/val2017.zip |
- L284: | 数据集 | COCO annotations | http://images.cocodataset.org/annotations/annotations_trainval2017.zip |
- L285: | 推理工具 | ais_bench | https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench |

## ACL_PyTorch/built-in/cv/SAM

文件清单：

```text
ACL_PyTorch/built-in/cv/SAM/LICENSE
ACL_PyTorch/built-in/cv/SAM/README.md
ACL_PyTorch/built-in/cv/SAM/assets/om_truck_result.JPG
ACL_PyTorch/built-in/cv/SAM/assets/pth_truck_result.JPG
ACL_PyTorch/built-in/cv/SAM/encoder_onnx_modify.py
ACL_PyTorch/built-in/cv/SAM/public_address_statement.md
ACL_PyTorch/built-in/cv/SAM/requirements.txt
ACL_PyTorch/built-in/cv/SAM/sam.aippconfig
ACL_PyTorch/built-in/cv/SAM/sam_amg_infer.py
ACL_PyTorch/built-in/cv/SAM/sam_coco_metric.py
ACL_PyTorch/built-in/cv/SAM/sam_end2end_infer.py
ACL_PyTorch/built-in/cv/SAM/sam_om_automatic_mask_generator.py
ACL_PyTorch/built-in/cv/SAM/sam_om_predictor.py
ACL_PyTorch/built-in/cv/SAM/sam_postprocessing_pytorch.py
ACL_PyTorch/built-in/cv/SAM/sam_preprocessing_pytorch.py
ACL_PyTorch/built-in/cv/SAM/sam_verify_multimask.py
ACL_PyTorch/built-in/cv/SAM/segment_anything_diff.patch
```

### [ACL_PyTorch/built-in/cv/SAM/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/built-in/cv/SAM/README.md)

文本SHA256：`17bd413e890cfe4eaab32581d809dcb76fdbb880cd6335f2eadc0d2cb2c28104`

- L9: [Segment Anything](https://arxiv.org/abs/2304.02643)
- L16: url=https://github.com/facebookresearch/segment-anything.git
- L29: | 固件与驱动 | 25.2.RC1 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
- L30: | CANN | 8.2.RC1 | - |
- L31: | Python | 3.11.10 | - |
- L32: | PyTorch | 2.1.0 | - |
- L33: | 说明：仅支持Atlas 300I Duo 推理卡，请以CANN版本选择实际固件与驱动版本。 | \ | \ |
- L40: git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
- L42: git clone https://github.com/facebookresearch/segment-anything.git
- L59: - 安装 [msit](https://gitcode.com/ascend/msit/tree/master/msit/) 的 surgeon 组件和 benchmark 组件。
- L69: | x    | FLOAT32 | 1 x 3 x 1024 x 1024 | NCHW         |
- L75: | image_embeddings  | FLOAT32  |  1 x 256 x 64 x 64 | NCHW           |
- L82: | image_embeddings    | FLOAT32 | 1 x 256 x 64 x 64 | NCHW         |
- L83: | point_coords    | FLOAT32 | 1 x -1 x 2 | ND         |
- L84: | point_labels    | FLOAT32 | 1 x -1 | ND         |
- L85: | mask_input    | FLOAT32 | 1 x 1 x 256 x 256 | NCHW         |
- L86: | has_mask_input    | FLOAT32 | 1 | ND         |
- L93: | iou_predictions  | FLOAT32  | -1 x 1  | ND           |
- L94: | low_res_masks  | FLOAT32  |  -1 x 1 x -1 x -1 | ND           |
- L97: ### 4. 准备数据集
- L103: wget -O demo.jpg https://raw.githubusercontent.com/facebookresearch/segment-anything/571794162e0887c15d12b809505b902c7bf8b4db/notebooks/images/truck.jpg
- L107: - 下载coco2017数据集进行精度测试。
- L109: 下载并解压COCO-2017数据集的[图片](https://gitee.com/link?target=http%3A%2F%2Fimages.cocodataset.org%2Fzips%2Fval2017.zip)与[标注](https://gitee.com/link?target=http%3A%2F%2Fimages.cocodataset.org%2Fannotations%2Fannotations_trainval2017.zip)，放置coco2017目录下
- L123: #### 5.1 获取权重文件
- L125: GitHub 仓库提供了三种大小的权重文件：vit_h、vit_l、vit_b。这里以 vit_b 为例。
- L130: wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
- L148: - checkpoint：模型权重文件路径。
- L193: > **说明：** 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。
- L205: | 0       310P3     | OK              | 15.8         42                0    / 0              |
- L206: | 0       0         | 0000:82:00.0    | 0            1074 / 21534                            |
- L208: | 1       310P3     | OK              | 15.4         43                0    / 0              |
- L209: | 0       1         | 0000:89:00.0    | 0            1070 / 21534                            |
- L241: - enable_small_channel：是否使能small channel的优化，使能后在channel<=4的首层卷积会有性能收益。
- L247: 更多参数说明请参考 [ATC 参数概览](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha002/devaids/auxiliarydevtool/atlasatc_16_0039.html)（如果链接失效，请从 [CANN 社区版文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition) 查找目录「应用开发 > ATC 模型转换 > 参数说明 > 参数概览」）
- L286: 6.2 性能验证。
- L288: 1. encoder 纯推理性能验证。
- L299: 2. decoder 纯推理性能验证。
- L320: 6.3 精度验证。
- L322: SAM 官方未提供精度评测手段，这里提供对应脚本，基于 COCO 验证集标注框作为输入提示，使用 SAM 预测分割掩码，并与 COCO 标注掩码逐实例进行 IoU 计算，最后对所有实例的 IoU 结果取平均，得到整体的平均交并比（mIoU）。
- L336: - dataset-path: coco数据集目录
- L351: | decoder_sim.om | 1 | 加 --return-single-mask | 单提示分割（6.1 端到端） |
- L352: | decoder_multimask_sim.om | 4 | 不加 --return-single-mask | AMG 自动分割（7.6） |
- L421: - crop-n-layers：分层裁剪层数（0=不裁剪只整图；2=4×4块放大扫，小物体多时推荐，精度无损）。
- L425: - no-npu-postprocess：禁用 NPU 后处理（默认启用，比 CPU 快约24倍、精度无损；无 torch_npu 时自动退回 CPU）。
- L437: | 8×8（64点，快速验证） | 约15 | [0.9, 1.0] | [0.95, 0.99] |
- L438: | 32×32（1024点，默认） | 49-50 | [0.88, 1.01] | [0.95, 0.99] |
- L440: ## 4. 模型推理性能 & 精度
- L441: 性能结果：
- L442: | 芯片型号 | 模型 | Batch Size | 性能 |
- L444: | 300I Pro | encoder | 1 | 4.43 fps |
- L445: | 300I Pro | decoder | 1 | 679.77 fps  |
- L446: | 300I Pro | decoder(multimask) | 1 | 679.77 fps |
- L447: | 300I Pro | AMG(32×32,1024点,端到端整图) | 1 | 45 s/图(≈0.022 fps,NPU后处理) |
- L449: > 注：decoder 的 fps 为单次推理性能(纯算子)；AMG 为端到端整图自动分割(含撒点+多次decoder推理+后处理)，二者口径不同，不可直接比较。AMG 默认启用 NPU 后处理加速(比 CPU 快约24×)；关闭 NPU 后处理(--no-npu-postprocess)约 175 s/图。
- L451: 精度结果：
- L452: | 芯片型号 | 模型 | Batch Size | 精度（mIoU） |
- L454: | 300I Pro | SAM | 1 | 0.7654 |

## ACL_PyTorch/built-in/cv/SAM2

文件清单：

```text
ACL_PyTorch/built-in/cv/SAM2/README.md
ACL_PyTorch/built-in/cv/SAM2/requirements.txt
ACL_PyTorch/built-in/cv/SAM2/sam2_coco_metric.py
ACL_PyTorch/built-in/cv/SAM2/sam2_convert_om.sh
ACL_PyTorch/built-in/cv/SAM2/sam2_flashattentiontik_onnx.py
ACL_PyTorch/built-in/cv/SAM2/sam2_image_predictor_export_onnx.py
ACL_PyTorch/built-in/cv/SAM2/sam2_postprocessing.py
ACL_PyTorch/built-in/cv/SAM2/sam2_preprocessing.py
```

### [ACL_PyTorch/built-in/cv/SAM2/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/built-in/cv/SAM2/README.md)

文本SHA256：`599b9352240a86ec787aa5c765d1726088b69461af82d240e5d089483891b8b5`

- L4: SAM2是Meta开源的多模态视觉分割模型，对比SAM1不仅优化了图像分割的精度，还新增了视频实时分割能力。
- L9: url=https://github.com/facebookresearch/sam2.git
- L22: | 固件与驱动 | 25.3.RC1 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
- L23: | CANN | 8.3.RC2 | - |
- L24: | Python | 3.11.10 | - |
- L25: | PyTorch | 2.5.1 | - |
- L26: | Ascend Extension PyTorch | 2.5.1.post1 | - |
- L28: - 本文已验证Atlas 300V PRO。推荐拉取华为昇腾社区镜像部署 [ascend hub](https://www.hiascend.com/developer/ascendhub/detail/0bc31428b0984445a39496fdcfce7c2b)
- L35: git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
- L37: git clone https://github.com/facebookresearch/sam2.git
- L65: | image    | FLOAT32 | bs x 3 x 1024 x 1024 | NCHW         |
- L71: | high_res_feats_0  | FLOAT32  |  bs x 32 x 256 x 256 | NCHW           |
- L72: | high_res_feats_1  | FLOAT32  |  bs x 64 x 128 x 128 | NCHW           |
- L73: | image_embed  | FLOAT32  |  bs x 256 x 64 x 64  | NCHW           |
- L80: | high_res_feats_0  | FLOAT32  |  1 x 32 x 256 x 256 | NCHW           |
- L81: | high_res_feats_1  | FLOAT32  |  1 x 64 x 128 x 128 | NCHW           |
- L82: | image_embed  | FLOAT32  |  1 x 256 x 64 x 64  | NCHW           |
- L83: | point_coords    | FLOAT32 | 1 x pointnums x 2 | ND         |
- L84: | point_labels    | int8 | 1 x pointnums | ND         |
- L85: | mask_input    | FLOAT32 | 1 x 1 x 256 x 256 | NCHW         |
- L86: | has_mask_input    | int8 | 1 | ND         |
- L93: | masks  | FLOAT32  | 1, 1, 256, 256  | ND           |
- L94: | iou_predictions  | FLOAT32  |  1, 1 | ND           |
- L95: | low_res_masks  | FLOAT32  | 1, 1, 256, 256  | ND           |
- L98: ### 4. 准备数据集
- L104: 下载并解压COCO-2017数据集的[图片](https://gitee.com/link?target=http%3A%2F%2Fimages.cocodataset.org%2Fzips%2Fval2017.zip)与[标注](https://gitee.com/link?target=http%3A%2F%2Fimages.cocodataset.org%2Fannotations%2Fannotations_trainval2017.zip)，放置coco2017目录下。
- L118: #### 5.1 获取权重文件
- L120: 选择下载[模型权重](https://github.com/facebookresearch/sam2?tab=readme-ov-file#download-checkpoints)，下面步骤以sam2.1_hiera_tiny.pt为例，下载权重文件[sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)，放到checkpoints目录下。
- L166: | 0       310P3     | OK              | 15.8         42                0    / 0              |
- L167: | 0       0         | 0000:82:00.0    | 0            1074 / 21534                            |
- L169: | 1       310P3     | OK              | 15.4         43                0    / 0              |
- L170: | 0       1         | 0000:89:00.0    | 0            1070 / 21534                            |
- L201: 请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。
- L203: #### 6.2 性能验证
- L205: - encoder 纯推理性能验证。
- L216: - decoder 纯推理性能验证。
- L227: #### 6.3 精度验证
- L241: - dataset_path: coco数据集目录。
- L249: ## 模型推理性能 & 精度
- L250: encoder模型性能结果：
- L251: | 芯片型号 | 模型 | bs | 性能(fps) |
- L253: | 300V PRO | sam2_hiera_tiny | 1 | 15.64 |
- L254: | 300V PRO | sam2_hiera_tiny | 4 | 15.87 |
- L255: | 300V PRO | sam2_hiera_tiny | 8 | 15.77 |
- L257: decoder模型性能结果：
- L258: | 芯片型号 | 模型 | bs | 性能(fps) |
- L260: | 300V PRO | sam2_hiera_tiny | 1 | 476.97 |
- L262: 精度结果：
- L263: | 芯片型号 | 模型 | 数据集 | bs(encoder 模型) | 精度（mIoU） | GPU精度（mIoU） |
- L265: | 300V PRO | sam2_hiera_tiny | coco2017 | 1 | 0.7815 | 0.7830 |
- L266: | 300V PRO | sam2_hiera_tiny | coco2017 | 4 | 0.7815 | 0.7830 |
- L267: | 300V PRO | sam2_hiera_tiny | coco2017 | 8 | 0.7815 | 0.7830 |

## ACL_PyTorch/built-in/embedding/bge-m3

文件清单：

```text
ACL_PyTorch/built-in/embedding/bge-m3/README.md
ACL_PyTorch/built-in/embedding/bge-m3/infer.py
ACL_PyTorch/built-in/embedding/bge-m3/requirements.txt
```

### [ACL_PyTorch/built-in/embedding/bge-m3/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/built-in/embedding/bge-m3/README.md)

文本SHA256：`ed837c758dfca0526ef1870e181764357106747e2e665344d010acb3ee9f6897`

- L9: - [性能](#性能)
- L22: | 固件与驱动                                                        | 25.0.RC1 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
- L23: | CANN                                                         |  8.1.RC1 | 包含kernels包和toolkit包                                                                                                   |
- L24: | Python                                                       |  3.10 | -                                                                                                     |
- L25: | PyTorch                                                      | 2.5.1 | -                                                                                                     |
- L26: | Ascend Extension PyTorch                                     | 2.5.1.post2 | -                                                                                                     |
- L27: | 说明：Atlas 800I A2 推理卡和Atlas 300I DUO 推理卡请以CANN版本选择实际固件与驱动版本。  |      \ | \                                                                                                     |
- L35: git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
- L39: 1. 获取开源模型源码和权重（可选）
- L40: > 如果您的设备可以方便的直接从hugging-hub下载权重和代码，则不需要执行这一步
- L43: git clone https://huggingface.co/BAAI/bge-m3
- L75: # 如果可以方便快速从huggingface-hub下载权重，则可以使用如下命令
- L77: python3 infer.py  # 可以使用 --model_path 指定权重路径
- L79: 在推理开始后，首先会默认执行warm_up，目的是执行首次编译，首次编译时间较长，在warm_up结束后，会执行推理操作，并打屏E2E性能数据。如果想测试模型推理耗时，可以在 `YOUR_ENV\FlagEmbedding\inference\embedder\encoder_only\m3.py` 文件423行 `outputs = self.model(...)` 前后添加时间打点。
- L83: ### 性能
- L84: | 模型     | 芯片             | E2E      | forward |
- L86: | bge-m3 | Atlas 300I DUO(单芯) | 137.59ms | 23.23ms |
- L87: | bge-m3 | Atlas 800I A2  | 103.88ms | 14.71ms |

## ACL_PyTorch/built-in/embedding/bge-reranker-v2-m3

文件清单：

```text
ACL_PyTorch/built-in/embedding/bge-reranker-v2-m3/README.md
ACL_PyTorch/built-in/embedding/bge-reranker-v2-m3/infer.py
ACL_PyTorch/built-in/embedding/bge-reranker-v2-m3/requirements.txt
```

### [ACL_PyTorch/built-in/embedding/bge-reranker-v2-m3/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/built-in/embedding/bge-reranker-v2-m3/README.md)

文本SHA256：`8141dc97b105f7953b577ee47503b3a1302ebeffb88e4e62436f20083caf2dc6`

- L9: - [性能](#性能)
- L22: | 固件与驱动                                                        | 25.0.RC1 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
- L23: | CANN                                                         |  8.1.RC1 | 包含kernels包和toolkit包                                                                                                   |
- L24: | Python                                                       |  3.10 | -                                                                                                     |
- L25: | PyTorch                                                      | 2.5.1 | -                                                                                                     |
- L26: | Ascend Extension PyTorch                                     | 2.5.1.post2 | -                                                                                                     |
- L27: | 说明：Atlas 800I A2 推理卡和Atlas 300I DUO 推理卡请以CANN版本选择实际固件与驱动版本。  |      \ | \                                                                                                     |
- L35: git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
- L39: 1. 获取开源模型源码和权重（可选）
- L40: > 如果您的设备可以方便的直接从hugging-hub下载权重和代码，则不需要执行这一步
- L43: git clone https://huggingface.co/BAAI/bge-reranker-v2-m3
- L70: # 如果可以方便快速从huggingface-hub下载权重，则可以使用如下命令
- L72: python3 infer.py  # 可以使用 --model_path 指定权重路径
- L74: 在推理开始后，首先会默认执行warm_up，目的是执行首次编译，首次编译时间较长，在warm_up结束后，会执行推理操作，并打屏E2E性能数据。如果想测试模型推理耗时，可以在 `YOUR_ENV\FlagEmbedding\inference\reranker\encoder_only\base.py` 文件186行 `scores = self.model(...)` 前后添加时间打点。
- L78: ### 性能
- L79: | 模型     | 芯片             | E2E     | forward |
- L81: | bge-m3 | Atlas 300I DUO(单芯) | 60.47ms | 22.57ms |
- L82: | bge-m3 | Atlas 800I A2  | 46.64ms | 15.08ms |

## ACL_PyTorch/built-in/embodied_ai/vla/GR00T_n1d6

文件清单：

```text
ACL_PyTorch/built-in/embodied_ai/vla/GR00T_n1d6/README.md
ACL_PyTorch/built-in/embodied_ai/vla/GR00T_n1d6/export_om/common.py
ACL_PyTorch/built-in/embodied_ai/vla/GR00T_n1d6/export_om/export_action_head.py
ACL_PyTorch/built-in/embodied_ai/vla/GR00T_n1d6/export_om/export_language_model.py
ACL_PyTorch/built-in/embodied_ai/vla/GR00T_n1d6/export_om/export_vision_model.py
ACL_PyTorch/built-in/embodied_ai/vla/GR00T_n1d6/export_om/npu_adaptor.py
ACL_PyTorch/built-in/embodied_ai/vla/GR00T_n1d6/gr00t_n1d6_om.py
ACL_PyTorch/built-in/embodied_ai/vla/GR00T_n1d6/gr00t_n1d6_om_policy.py
ACL_PyTorch/built-in/embodied_ai/vla/GR00T_n1d6/requirements.txt
ACL_PyTorch/built-in/embodied_ai/vla/GR00T_n1d6/scripts/install_torchcodec_aarch64.sh
ACL_PyTorch/built-in/embodied_ai/vla/GR00T_n1d6/scripts/run_gr00t_om_server.py
ACL_PyTorch/built-in/embodied_ai/vla/GR00T_n1d6/scripts/setup_om_inference_env.sh
ACL_PyTorch/built-in/embodied_ai/vla/GR00T_n1d6/scripts/setup_robocasa_gr1_client.sh
ACL_PyTorch/built-in/embodied_ai/vla/GR00T_n1d6/scripts/standalone_inference_om.py
```

### [ACL_PyTorch/built-in/embodied_ai/vla/GR00T_n1d6/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/built-in/embodied_ai/vla/GR00T_n1d6/README.md)

文本SHA256：`171c8eaeb116d09308a7db69a7d4e7c639d0a8d87d5e7574a6760e7b504bd590`

- L7: [NVIDIA Isaac GR00T N1.6](https://github.com/NVIDIA/Isaac-GR00T) 是面向通用人形机器人技能的**开源视觉-语言-动作（Vision-Language-Action, VLA）**基础模型。模型以多模态输入（自然语言指令、多视角图像、机器人本体状态）为条件，输出连续动作序列，用于完成桌面操作、双臂协作等操控任务。
- L16: 预训练 checkpoint 可参考 [nvidia/GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B)。
- L34: 1. **离线性能评测**：运行 `scripts/standalone_inference_om.py`，在 LeRobot 数据集上对比 GT action 并统计逐步推理时延
- L49: | 固件与驱动 | 25.3.RC1 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
- L50: | CANN | 8.3.RC1 | 昇腾镜像中已预装 |
- L51: | Python | 3.10 | conda 环境 |
- L52: | PyTorch | 2.7.1 | 见 `requirements.txt` |
- L53: | torch-npu | 2.7.1 | 与 torch 版本对齐 |
- L56: 此模型已在 Atlas 300V Pro 上验证。推荐使用华为昇腾社区推理镜像部署 [ascend hub](https://www.hiascend.com/developer/ascendhub/detail/0bc31428b0984445a39496fdcfce7c2b)；镜像提供 CANN 与 conda，但 **ais_bench / aclruntime 需在 conda 环境内单独安装**。
- L61: git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
- L66: * 下载权重[nvidia/GR00T-N1.6-3B](https://modelscope.cn/models/nv-community/GR00T-N1.6-3B/files)放在当前目录下
- L72: | `gr00t_n1d6_om` | `scripts/setup_om_inference_env.sh` | OM 推理、policy server |
- L73: | `robocasa_gr1_client` | `scripts/setup_robocasa_gr1_client.sh` | RoboCasa 仿真 rollout client |
- L79: OM 推理依赖 `ais_bench` 与 `aclruntime`。请从 [ais_bench 工具仓](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench) 下载 wheel，放到：
- L106: | `CONDA_ENV_NAME` | `gr00t_n1d6_om` | conda 环境名 |
- L107: | `RECREATE_ENV` | `1` | 是否重建已有环境 |
- L108: | `INSTALL_TORCHCODEC` | `1` | 是否安装 torchcodec（数据集视频解码） |
- L128: - [sketchfab.zip](https://huggingface.co/datasets/nvidia/PhysicalAI-DigitalCousin-Assets/resolve/1b018839a6da865dffecd3185fe054211bc71270/sketchfab.zip)
- L129: - [lightwheel.zip](https://huggingface.co/datasets/nvidia/PhysicalAI-DigitalCousin-Assets/resolve/1b018839a6da865dffecd3185fe054211bc71270/lightwheel.zip)
- L131: [robocasa-assets](https://huggingface.co/datasets/jianzhang96/robocasa-assets/tree/main)
- L166: │   ├── standalone_inference_om.py # 数据集离线评测
- L190: #    Language 含多层 Attention/Softmax，ATC 须保持原始精度，否则 ONNX vs OM 余弦相似度会明显下降
- L199: 三段脚本默认 `--embodiment-tag GR1`，与后续 GR1 推理 / 仿真评测一致。导出时会自动完成 torch vs ONNX、ONNX vs OM 精度对比。
- L205: | `--model-path` | GR00T checkpoint 路径 | 必填 |
- L206: | `--embodiment-tag` | 具身标签 | `GR1` |
- L207: | `--device` | PyTorch 导出设备 | `cpu` |
- L208: | `--device-id` | OM 精度对比时的 NPU 设备 ID | `0` |
- L210: | `--opset` | ONNX opset 版本 | `14` |
- L212: | `--precision-mode-v2` | ATC 精度模式 | `fp16` |
- L214: | `--skip-compare` | 跳过精度对比 | 关闭 |
- L221: | `--num-images` | 输入图像数量（相机路数） | `1` |
- L222: | `--num-channels` | 图像通道数 | `3` |
- L223: | `--image-height` | 图像高度 | `252` |
- L224: | `--image-width` | 图像宽度 | `252` |
- L225: | `--input-fp16-nodes` | ATC 输入节点设为 FP16 | 开启 |
- L227: GR1（含 `gr1.PickNPlace` 离线数据集与 RoboCasa GR1 tabletop 仿真）使用**单路** ego 视角，video key 为 `ego_view_bg_crop_pad_res256_freq20`。导出 vision OM 时保持默认 `--num-images 1` 即可。
- L236: | `--batch-size` | batch 大小 | `1` |
- L237: | `--seq-len` | 导出用序列长度（用于构造 dummy 输入） | `122(动态维度)` |
- L238: | `--precision-mode-v2` | ATC 精度模式；Language 建议 `origin`，避免 Attention/Softmax 被强制 FP16 降精度 | `fp16`（**建议改为 `origin`**） |
- L240: Language OM 仅导出 causal mask 路径，**不支持 left-padding**；推理时 `attention_mask` 须全 1，序列长度按实际输入动态变化。导出 Language OM 时请显式指定 `--precision-mode-v2 origin`，以保证 ONNX vs OM 精度对齐。
- L247: | `--batch-size` | batch 大小 | `1` |
- L248: | `--seq-len` | `vl_embeds` 导出用序列长度 | `122(动态维度)` |
- L249: | `--ff-net0-chunks` | DiT `ff/net.0/proj` 切分块数，用于降低单算子耗时 | `4`（可选 `1/2/4/8`） |
- L255: ## 性能评测（离线）
- L257: ### 数据集逐步评测（推荐）
- L259: 在 LeRobot 数据集上逐步推理，输出 MSE/MAE 与每步 OM 推理时延（batch=1，与导出一致）：
- L282: | `--model-path` | GR00T checkpoint 路径 | 必填 |
- L283: | `--dataset-path` | LeRobot 数据集路径 | `demo_data/gr1.PickNPlace/` |
- L284: | `--embodiment-tag` | 具身标签 | `GR1` |
- L285: | `--traj-ids` | 评测轨迹 ID 列表 | `[0]` |
- L286: | `--steps` | 每条轨迹最大步数 | `200` |
- L287: | `--action-horizon` | 轨迹步进 action horizon | `16` |
- L291: | `--device-id` | Ascend 设备 ID | `0` |
- L298: ## 性能数据
- L304: | GR00T-N1.6-3B（E2E `get_action`） | batch=1 合成观测 | Atlas 300V Pro| 1.3s |
- L356: | `--n_episodes` | 评测 episode 数 | `10` |
- L357: | `--policy_client_host` | server 地址 | `127.0.0.1` |
- L358: | `--policy_client_port` | server 端口 | `5555` |
- L359: | `--env_name` | 仿真任务 | `gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env` |
- L360: | `--max_episode_steps` | 每个 episode 最大步数 | `720` |
- L361: | `--n_action_steps` | 每次 policy 调用执行的 action 步数 | `8` |
- L362: | `--n_envs` | 并行仿真环境数（与 OM 导出 batch 一致） | `1` |
- L364: ## 精度数据
- L370: | `PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env` | Atlas 300V Pro| 60.0% | 51.5% |
- L391: 4. **在运行性能测试时遇到无法打开 Parquet 和视频文件**

## ACL_PyTorch/built-in/embodied_ai/vla/pi0

文件清单：

```text
ACL_PyTorch/built-in/embodied_ai/vla/pi0/README.md
ACL_PyTorch/built-in/embodied_ai/vla/pi0/convert_verify_onnx_action_expert.py
ACL_PyTorch/built-in/embodied_ai/vla/pi0/convert_verify_onnx_vlm.py
ACL_PyTorch/built-in/embodied_ai/vla/pi0/infer.py
ACL_PyTorch/built-in/embodied_ai/vla/pi0/modeling_pi0_om.py
ACL_PyTorch/built-in/embodied_ai/vla/pi0/patches/lerobot_diff.patch
ACL_PyTorch/built-in/embodied_ai/vla/pi0/patches/modeling_gemma.patch
ACL_PyTorch/built-in/embodied_ai/vla/pi0/requirements.txt
ACL_PyTorch/built-in/embodied_ai/vla/pi0/utils.py
ACL_PyTorch/built-in/embodied_ai/vla/pi0/verify_om_onnx_action_expert.py
ACL_PyTorch/built-in/embodied_ai/vla/pi0/verify_om_onnx_vlm.py
```

### [ACL_PyTorch/built-in/embodied_ai/vla/pi0/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/built-in/embodied_ai/vla/pi0/README.md)

文本SHA256：`c3beb87c887232c5b5306e83f094908e758efac46930278c5fd498de86a0e36c`

- L5: pi0模型在π0: A Vision-Language-Action Flow Model for General Robot Control 论文中提出，原仓库地址为：https://github.com/Physical-Intelligence/openpi
- L7: LeRobot 是一套基于 PyTorch 构建的面向真实世界机器人应用的工具集，核心目标是降低机器人领域的研发门槛，让开发者均可参与共享数据集、预训练模型的建设。该指导基于LeRobot框架，采用OM加速路线部署π0，https://huggingface.co/BrunoM42/pi0_aloha_transfer_cube。由于Lerobot在v0.4.0后重构了pi0和pi0.5代码结构。基于老Lerobot版本代码训练的模型在新版本代码推理上会出现诸多config类报错， 因此我们在本样例下，主要适配的基于‘577cd10974b84bea1f06b6472eb9e5e74e07f77a’ commit的Lerbot代码。
- L14: | 固件与驱动                                                       | 24.1.RC3    | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
- L15: | CANN                                                            | 8.3.RC1     | 包含kernels包和toolkit包                                                                                                   |
- L16: | Python                                                          | 3.11        | -                                                                                                     |
- L17: | PyTorch                                                         | 2.7.1       | -                                                                                                     |
- L18: | 说明：支持Atlas 300I DUO/Atlas 300I Pro，不支持Atlas 800I A2 | \           | \                                                                                                     |
- L23: git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
- L33: git clone https://github.com/huggingface/lerobot.git
- L58: * 下载模型权重：
- L64: * pi0_aloha_transfer_cube [下载链接](https://huggingface.co/BrunoM42/pi0_aloha_transfer_cube/tree/main)
- L65: * paligemma-3b-pt-224 tokenizer，下载added_tokens.json, special_tokens_map.json, tokenizer_config.json, tokenizer.json和tokenizer.model这个五个文件即可：[下载链接](https://modelscope.cn/models/AI-ModelScope/paligemma-3b-pt-224/files)
- L70: * 运行om时需要ACLLite组件，ACLLite安装指导：https://gitee.com/ascend/ACLLite/tree/master/python
- L82: |   |── 📁 pi0_aloha_transfer_cube
- L83: |   └── 📁 paligemma-3b-pt-224
- L87: |── 📄 modeling_pi0_om.py
- L105: * --pretrained-policy-path: pi0模型权重路径，默认为"./weight/pi0_aloha_transfer_cube"
- L108: * --device: 默认为"cpu"
- L114: * --no-validate: 设置该参数可跳过精度验证
- L126: 模型转换完成后会在指定的output目录下生成pi0-vlm.om和pi0-action_expert.om或者pi0-action_expert_{arch}.om，模型{arch}后缀为当前使用的CPU操作系统。
- L145: ## 性能数据
- L146: infer.py取libero数据集中的一个frame进行推理，性能如下
- L148: | 模型                     | 芯片 | 数据集 | 单次推理耗时
- L150: |π0    |  300I DUO(单芯) | 随机数      | 220ms |

## ACL_PyTorch/built-in/embodied_ai/vla/pi05_lerobot/infer_with_torchair

文件清单：

```text
ACL_PyTorch/built-in/embodied_ai/vla/pi05_lerobot/infer_with_torchair/README.md
ACL_PyTorch/built-in/embodied_ai/vla/pi05_lerobot/infer_with_torchair/infer.py
ACL_PyTorch/built-in/embodied_ai/vla/pi05_lerobot/infer_with_torchair/patches/apply_patch.py
ACL_PyTorch/built-in/embodied_ai/vla/pi05_lerobot/infer_with_torchair/patches/lerobot_diff.patch
ACL_PyTorch/built-in/embodied_ai/vla/pi05_lerobot/infer_with_torchair/requirements.txt
```

### [ACL_PyTorch/built-in/embodied_ai/vla/pi05_lerobot/infer_with_torchair/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/built-in/embodied_ai/vla/pi05_lerobot/infer_with_torchair/README.md)

文本SHA256：`a92eb61b62bd825fa5fb0e1dfede6462130144d03f06669f41a5efba58440442`

- L4: LeRobot 是一套基于 PyTorch 构建的面向真实世界机器人应用的工具集，核心目标是降低机器人领域的研发门槛，让开发者均可参与共享数据集、预训练模型的建设。该指导基于LeRobot框架，采用TorchAir加速路线部署VLA模型--π0.5。
- L12: | 固件与驱动                                                       | 24.1.RC3    | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
- L13: | CANN                                                            | 8.3.RC1     | 包含kernels包和toolkit包                                                                                                   |
- L14: | Python                                                          | 3.11        | -                                                                                                     |
- L15: | PyTorch                                                         | 2.7.1       | -                                                                                                     |
- L16: | Ascend Extension PyTorch                                        | 2.7.1 | -                                                                                                     |
- L17: | 说明：支持Atlas 300I DUO/Atlas 300I Pro，不支持Atlas 800I A2 | \           | \                                                                                                     |
- L22: git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
- L30: git clone -b v0.4.3 https://github.com/huggingface/lerobot.git
- L54: * 下载模型权重：
- L59: * pi05_libero_finetuned：[下载链接](https://modelscope.cn/models/lerobot/pi05_libero_finetuned/files)
- L60: * paligemma-3b-pt-224 tokenizer：[下载链接](https://modelscope.cn/models/AI-ModelScope/paligemma-3b-pt-224/files)
- L85: # 2. 执行推理脚本，输出单个样本推理耗时，首次执行时会自动下载libero数据集。
- L89: * --model_path: pi05模型权重路径，默认为"./weight/pi05_libero_finetuned"
- L92: * --episode_index: 选择数据集样本的index
- L94: ## 性能数据
- L95: infer.py取libero数据集中的一个frame进行推理，性能如下
- L97: | 模型                     | 芯片 | 数据集 | 单次推理耗时
- L99: |π0.5    |  300I DUO(单芯) | libero       | 375ms |
- L101: ## 精度测试
- L102: 执行以下命令来对libero数据集做仿真精度测试，首次执行时会下载
- L107: * --policy.path: pi05模型权重路径
- L116: |π0.5      |  300I DUO(单芯)|  100 | 98.2|

## ACL_PyTorch/built-in/embodied_ai/vla/pi05_openpi

文件清单：

```text
ACL_PyTorch/built-in/embodied_ai/vla/pi05_openpi/README.md
ACL_PyTorch/built-in/embodied_ai/vla/pi05_openpi/inference_pytorch.py
ACL_PyTorch/built-in/embodied_ai/vla/pi05_openpi/norm_stats.json
ACL_PyTorch/built-in/embodied_ai/vla/pi05_openpi/patches/apply_patch.py
ACL_PyTorch/built-in/embodied_ai/vla/pi05_openpi/patches/pi05_pytorch_diff.patch
ACL_PyTorch/built-in/embodied_ai/vla/pi05_openpi/requirements_add.txt
```

### [ACL_PyTorch/built-in/embodied_ai/vla/pi05_openpi/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/built-in/embodied_ai/vla/pi05_openpi/README.md)

文本SHA256：`915cf5b15f14da002cb7d0c887255f284ef12523248f36a907aca8659049e99e`

- L6: 本指导基于 [openpi](https://github.com/Physical-Intelligence/openpi) 官方 PyTorch 代码仓库，采用 TorchAir 加速路线在昇腾 D310P 端侧部署 π0.5 模型。
- L12: | 固件与驱动 | 25.2.0 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
- L13: | CANN | 8.5.2 | 包含 kernels 包和 toolkit 包 |
- L14: | Python | 3.11 | — |
- L15: | PyTorch | 2.7.1 | — |
- L16: | Ascend Extension PyTorch | 2.7.1 | torch_npu |
- L17: | 硬件 | Atlas 300I DUO / Atlas 300I Pro | 不支持 Atlas 800I A2 |
- L30: git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
- L55: ### 5. 下载模型权重
- L58: # 下载openpi 官方预训练权重: gs://openpi-assets/checkpoints/pi05_libero
- L66: wget https://huggingface.co/ndkhanh95/Paligemma/resolve/3fc6ea2396d8d649b2e7330abbcbc1d5e010e5ec/tokenizer.model
- L111: | `--checkpoint_dir` | `xxx/pi05_libero/` | pi05 模型权重路径 |
- L113: | `--warmup` | 5 | warmup 次数（首次 warmup 时编译为图） |
- L114: | `--record` | 10 | 记录推理次数 |
- L115: | `--config` | `pi05_libero` | 训练配置名称 |
- L118: # 1. 下载libero_10数据集
- L119: https://huggingface.co/datasets/IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot/tree/main
- L125: ## 性能优化
- L127: 基于 openpi PyTorch 版本在昇腾 D310P 端侧进行了一系列优化，最终推理耗时达到 **378ms**，与 lerobot版本性能基本持平。
- L133: | **FP16 推理** | 模型加载后统一转为 FP16 精度，减少计算量和显存带宽 | `pi0_config.py` 中 `dtype: "float16"`；`policy_config.py` 中 `model.to(dtype=torch.float16)` |
- L134: | **NZ 格式转换** | 将 Linear 层权重转换为昇腾 FRACTAL_NZ 分形格式，提升矩阵乘计算效率 | `policy_config.py` 中 `format_cast_to_NZ()` |
- L135: | **NPU RoPE 算子** | 用 `torch_npu.npu_rotary_mul` 替换原生的 RoPE 实现，减少 kernel 启动开销 | `pi0_pytorch.py` 中 `apply_npu_rope()` 函数及 monkey-patch |
- L136: | **图编译加速** | 使用 TorchAir 对 `embed_image`、`paligemma_with_expert.forward`、`denoise_step` 三个子图进行编译，`dynamic=False, fullgraph=True` | `pi0_pytorch.py` 中 `torch.compile(... backend=npu_backend)` |
- L137: | **图像编码合并** | 将多张图像的 embed_image 调用合并为一次 batch 推理，减少图编译和 kernel 启动次数 | `pi0_pytorch.py` 中 `embed_prefix()` 的 `torch.cat(images, dim=0)` |
- L139: ### 性能数据
- L143: | π0.5 (openpi PyTorch + TrochAir) | 300I DUO（单芯） | **378ms** |
- L144: | π0.5 (LeRobot + TorchAir) | 300I DUO（单芯） | 375ms |
- L146: ## 精度说明
- L148: 基于 openpi PyTorch 版本在昇腾 D310P 端侧进行了一系列优化，是参考[pi05 lerobot版本](https://atomgit.com/Ascend/ModelZoo-PyTorch/tree/c0bfbfd87ee010fa3a82e8d71cd4f13e1fdd0e75/ACL_PyTorch/built-in/embodied_ai/vla/pi05/infer_with_torchair)版本优化点做适配优化，精度同lerobot版本持平。

## ACL_PyTorch/built-in/ocr/MinerU

文件清单：

```text
ACL_PyTorch/built-in/ocr/MinerU/README.md
ACL_PyTorch/built-in/ocr/MinerU/doclayout_yolo.patch
ACL_PyTorch/built-in/ocr/MinerU/infer.py
ACL_PyTorch/built-in/ocr/MinerU/mineru.patch
ACL_PyTorch/built-in/ocr/MinerU/overall_metric.py
ACL_PyTorch/built-in/ocr/MinerU/requirements.txt
ACL_PyTorch/built-in/ocr/MinerU/ultralytics.patch
```

### [ACL_PyTorch/built-in/ocr/MinerU/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/built-in/ocr/MinerU/README.md)

文本SHA256：`25c0d1841aefdda8ba6439dfec41981d1880a1d2342921de19e3ae77337c50ba`

- L7: - [获取权重](#获取权重)
- L8: - [获取数据集](#获取数据集)
- L10: - [精度测试](#精度测试)
- L22: url=https://github.com/opendatalab/MinerU.git
- L34: | 固件与驱动                                                   | 25.2.RC1    | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
- L35: | CANN                                                    | 8.3.0       | -                                                                                             |
- L36: | Python                                                  | 3.11        | -                                                                                             |
- L37: | PyTorch                                                 | 2.6.0       | -                                                                                             |
- L38: | Ascend Extension PyTorch                                | 2.6.0       | -                                                                                             |
- L39: | 说明：Atlas 800I A2/Atlas 300I Pro 推理卡请以CANN版本选择实际固件与驱动版本。 | \           | \                                                                                             |
- L48: git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
- L50: git clone https://github.com/opendatalab/MinerU.git
- L62: git clone https://gitcode.com/ascend/vision.git vision_npu
- L78: source_path=$(pip show ultralytics | grep Location | awk '{print $2}')
- L88: ## 获取权重
- L90: 运行以下指令，下载权重文件[Model weights](https://www.modelscope.cn/models/OpenDataLab/PDF-Extract-Kit-1.0/summary)，默认存放为`/root/.cache/modelscope/hub/models/OpenDataLab/PDF-Extract-Kit-1___0`
- L95: 下载完成后，默认在根目录生成`mineru.json`文件，移动数据集时，需修改`/root/mineru.json`文件中"models-dir": "pipeline"为修改后权重存放路径
- L97: 权重目录大致结构为：
- L128: ## 获取数据集
- L130: 创建数据集目录`OmniDocBench_dataset`，下载多样性文档解析评测集`OmniDocBench`数据集的[pdfs和标注](https://opendatalab.com/OpenDataLab/OmniDocBench)，解压并放置在`OmniDocBench_dataset`目录下
- L152: - data_path: 数据集路径
- L153: - model_source: 模型源类型，local表示使用本地文件，modelscope/huggingface表示在线拉取权重
- L173: ## 精度测试
- L175: 使用`OmniDocBench`数据集配套评测代码测试精度。
- L189: git clone https://github.com/opendatalab/OmniDocBench.git
- L197: - 公式精度指标CDM需要额外安装环境
- L201: wget https://nodejs.org/dist/v16.13.1/node-v16.13.1-linux-arm64.tar.xz
- L211: git clone https://github.com/ImageMagick/ImageMagick.git ImageMagick-7.1.2
- L251: 4. 精度测量结果
- L260: 运行overall_metric.py可以得到精度结果：
- L266: 在`OmniDocBench`数据集上的精度和性能数据分别为：
- L267: |模型|芯片|overall|性能（s）|
- L269: |MinerU|300I DUO(单芯)|81.68| 3.37 |
- L270: |MinerU|800I A2 64G|81.51| 1.85 |

## ACL_PyTorch/built-in/ocr/PPOCRv5

文件清单：

```text
ACL_PyTorch/built-in/ocr/PPOCRv5/README.md
ACL_PyTorch/built-in/ocr/PPOCRv5/eval_ori_accuracy.py
ACL_PyTorch/built-in/ocr/PPOCRv5/eval_uvdoc_accuracy.py
ACL_PyTorch/built-in/ocr/PPOCRv5/infer.py
ACL_PyTorch/built-in/ocr/PPOCRv5/onnx_optimizer.py
ACL_PyTorch/built-in/ocr/PPOCRv5/paddleocr.patch
ACL_PyTorch/built-in/ocr/PPOCRv5/requirements.txt
```

### [ACL_PyTorch/built-in/ocr/PPOCRv5/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/built-in/ocr/PPOCRv5/README.md)

文本SHA256：`0b10ad99942a776299985b21b8915478006c7e0ec7cd00ff40cd1507b406513d`

- L7: - [获取权重](#获取权重)
- L9: - [精度测试](#精度测试)
- L22: url=https://github.com/PaddlePaddle/PaddleOCR.git
- L34: | 固件与驱动                                                   | 25.2.RC1    | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
- L35: | CANN                                                    | 8.5.0       | -                                                                                             |
- L36: | Python                                                  | 3.11        | -                                                                                             |
- L37: | 说明：Atlas 300I Pro 推理卡请以CANN版本选择实际固件与驱动版本。 | \           | \                                                                                             |
- L46: git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
- L49: git clone --depth 1 --branch v3.3.2 https://github.com/PaddlePaddle/PaddleOCR.git
- L61: wget https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/ais_bench-0.0.2-py3-none-any.whl
- L62: wget https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/aclruntime-0.0.2-cp311-cp311-linux_aarch64.whl
- L69: 执行以下命令激活CANN环境变量。注意：该命令中文件路径仅供参考，请以实际安装环境配置环境变量。详细介绍请参见[《CANN 开发辅助工具指南 (推理)》](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=netconda&OS=openEuler)。
- L74: ## 获取权重
- L76: PPOCRv5产线主要包含文本检测模块（det）以及文本识别模块（rec），运行以下指令，分别下载det模型[Det Model weights](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar)以及rec模型[Rec Model weights](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0//PP-OCRv5_server_rec_infer.tar)权重文件，并进行解压
- L78: wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar
- L79: wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0//PP-OCRv5_server_rec_infer.tar
- L84: 另外PPOCRv5产线包含三个可选模块：文档图像方向分类模块（[doc_ori](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar)），文本行方向分类模块（[textline_ori](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_textline_ori_infer.tar)），文本图像矫正模块（[uvdoc](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar)），可分别下载其权重文件，并进行解压
- L86: wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar
- L87: wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_textline_ori_infer.tar
- L88: wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar
- L127: 执行`npu-smi info`查看芯片名称，并赋值为${soc_version}，执行ATC命令，生成det和rec的om模型，结果存放于各自权重路径，文件名均为inference_{os}_{arch}.om，{os}为当前使用的操作系统，{arch}后缀为当前使用的CPU架构，比如：inference_linux_aarch64.om。
- L146: wget https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png
- L156: - custom_size: ais_bench使用动态shape时指定的buffer size，具体数值根据数据集而异，值过小会导致内存越界，可从小往大调整直至正常推理，推荐初始值为1e8
- L161: ## 精度测试
- L162: 参照paddleocr的精度测试流程[det](https://www.paddleocr.ai/main/version3.x/module_usage/text_detection.html)、[rec](https://www.paddleocr.ai/main/version3.x/module_usage/text_recognition.html)，分别对文本检测以及识别进行精度测试，首先进入PaddleOCR目录
- L166: ### 文本检测模型精度验证
- L167: 1. 准备数据集
- L169: wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_det_dataset_examples.tar
- L173: 使用如下命令得到精度指标：
- L181: 预期输出precision、recall以及hmean指标，这里选用hmean作为文本检测模型的精度指标，其表示precision与recall的调和平均，能够更好地描述检测模型的整体质量。在ocr_det_dataset_examples数据集上，PP-OCRv5_server_det模型预期输出的hmean指标为0.5287。
- L183: ### 文本识别模型精度验证
- L184: 1. 准备数据集
- L186: wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_rec_dataset_examples.tar
- L191: 使用如下命令得到精度指标：
- L199: 预期输出acc以及norm_edit_dis两个指标，分别表示准确率以及归一化编辑距离，这里选用acc作为文本识别模型的精度指标，PP-OCRv5_server_rec模型在ocr_rec_dataset_examples数据集上的预期输出为0.7227。
- L201: ### 可选模型精度验证
- L202: 1. 数据集准备及精度评测方式
- L205: | 模型 | 评测方法 | 数据集 | 关键指标 |
- L207: | doc_ori | 旋转增强 + 分类 | 250 张真实文档，程序化旋转 90/180/270° 生成 4 类标签样本，共 1000 样本 | Top-1 Acc |
- L208: | textline_ori | 旋转增强 + 分类 | 500 张文本行，旋转 0/180° 生成 2 类标签样本，共1000 样本 | Top-1 Acc |
- L209: | UVDoc | 真实畸变 → 展平 | DocUNet Benchmark(CVPR 2018, Megvii/Face++)数据集中收集 130 配对数据 | MS-SSIM / SSIM / PSNR |
- L212: 使用如下命令进行可选模型精度评测：
- L241: ### 性能测试
- L242: 对于文本检测模型，输入的数据要求长宽为32的整数倍，我们以（1,3,960,960）的输入shape为例，运行如下命令测试纯模型推理性能，
- L246: 对于文本识别模型，以（1,3,48,320）shape为例，运行如下命令测试纯模型推理性能
- L250: 对于文档方向分类模型（可选），以（1,3,224,224）shape为例，运行如下命令测试纯模型推理性能
- L254: 对于文本行方向分类模型（可选），以（1,3,80,160）shape为例，运行如下命令测试纯模型推理性能
- L258: 对于文本图像矫正模型（可选），以（1,3,512,256）shape为例，运行如下命令测试纯模型推理性能
- L264: 文本检测模型和文本识别模型在示例数据集上的的精度和性能数据展示在下表，其中文本检测模型的精度指标为hmeans，文本识别模型的精度指标为acc：
- L269: |模型|hmeans/acc|hmeans/acc（A10）|纯模型性能（ms）(EP模式)|纯模型性能（ms）(RC模式)|
- L271: |PP-OCRv5_server_det|52.87%|52.87%|35.01 |26.85 |
- L272: |PP-OCRv5_server_rec|72.27%|72.27%|2.63 |6.09 |
- L273: |PP-LCNet_x1_0_doc_ori|92.80%|92.80%|0.78 |- |
- L274: |PP-LCNet_x1_0_textline_ori|79.20%|79.20%|0.80 |- |
- L275: |UVDoc_infer|MS-SSIM=0.4417|MS-SSIM=0.4417|7.08 |- |

## ACL_PyTorch/built-in/ocr/PPOCRv6

文件清单：

```text
ACL_PyTorch/built-in/ocr/PPOCRv6/README.md
ACL_PyTorch/built-in/ocr/PPOCRv6/infer.py
ACL_PyTorch/built-in/ocr/PPOCRv6/onnx_optimizer.py
ACL_PyTorch/built-in/ocr/PPOCRv6/paddleocr.patch
ACL_PyTorch/built-in/ocr/PPOCRv6/requirements.txt
```

### [ACL_PyTorch/built-in/ocr/PPOCRv6/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/built-in/ocr/PPOCRv6/README.md)

文本SHA256：`a35c390bbbc23e07304b6a148c4d2af9bb2f4aa07a06ca661bea4e1d4ef911b6`

- L7: - [获取权重](#获取权重)
- L9: - [精度测试](#精度测试)
- L15: PP-OCRv6 是 PP-OCR 新一代文字识别解决方案，基于全新设计的 **PPLCNetV4** 统一骨干网络。在文字类型方面，单一模型统一支持中文、英文、日文及 46 种拉丁语系共 **50 种语言**。在模型架构方面，文本检测采用 PPLCNetV4 + RepLKFPN（DB 检测头），文本识别采用 PPLCNetV4 + LightSVTR + CTC/NRTR 多头解码器。提供 **medium / small / tiny** 三档模型，兼顾精度与效率。
- L22: url=https://github.com/PaddlePaddle/PaddleOCR.git
- L34: | 固件与驱动                                                   | 25.2.RC1    | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
- L35: | CANN                                                    | 8.3.RC1       | -                                                                                             |
- L36: | Python                                                  | 3.11        | -                                                                                             |
- L37: | 说明：Atlas 800I A2 推理卡请以CANN版本选择实际固件与驱动版本。 | \           | \                                                                                             |
- L46: git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
- L49: git clone --depth 1 --branch v3.7.0 https://github.com/PaddlePaddle/PaddleOCR.git
- L63: wget https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/ais_bench-0.0.2-py3-none-any.whl
- L64: wget https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/aclruntime-0.0.2-cp311-cp311-linux_aarch64.whl
- L71: 执行以下命令激活CANN环境变量。注意：该命令中文件路径仅供参考，请以实际安装环境配置环境变量。详细介绍请参见[《CANN 开发辅助工具指南 (推理)》](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=netconda&OS=openEuler)。
- L76: ## 获取权重
- L78: PP-OCRv6 产线主要包含文本检测模块（det）以及文本识别模块（rec），提供 medium / small / tiny 三档模型。推荐使用 medium 档（最高精度）。
- L82: wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_medium_det_infer.tar
- L83: wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_medium_rec_infer.tar
- L86: # wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_small_det_infer.tar
- L87: # wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_small_rec_infer.tar
- L90: # wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_tiny_det_infer.tar
- L91: # wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_tiny_rec_infer.tar
- L115: 执行`npu-smi info`查看芯片名称，并赋值为${soc_version}，执行ATC命令，生成det和rec的om模型，结果存放于各自权重路径，文件名均为inference_{arch}.om，模型{arch}后缀为当前使用的CPU操作系统。
- L126: wget https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png
- L137: - custom_size: ais_bench使用动态shape时指定的buffer size，具体数值根据数据集而异，值过小会导致内存越界，可从小往大调整直至正常推理，推荐初始值为1e7
- L142: ## 精度测试
- L143: 参照paddleocr的精度测试流程[det](https://www.paddleocr.ai/main/version3.x/module_usage/text_detection.html)、[rec](https://www.paddleocr.ai/main/version3.x/module_usage/text_recognition.html)，分别对文本检测以及识别进行精度测试，首先进入PaddleOCR目录
- L147: ### 文本检测模型精度验证
- L148: 1. 准备数据集
- L150: wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_det_dataset_examples.tar
- L155: 使用如下命令得到精度指标。
- L164: 预期输出precision、recall以及hmean指标，这里选用hmean作为文本检测模型的精度指标，其表示precision与recall的调和平均，能够更好地描述检测模型的整体质量。在ocr_det_dataset_examples数据集上，PP-OCRv6_medium_det模型预期输出的hmean指标为0.4673。
- L166: ### 文本识别模型精度验证
- L167: 1. 准备数据集
- L169: wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_rec_dataset_examples.tar
- L174: 使用如下命令得到精度指标：
- L182: 预期输出acc以及norm_edit_dis两个指标，分别表示准确率以及归一化编辑距离，这里选用acc作为文本识别模型的精度指标，PP-OCRv6_medium_rec模型在ocr_rec_dataset_examples数据集上的预期输出为0.7727
- L184: ### 性能测试
- L185: 对于文本检测模型，输入的数据要求长宽为32的整数倍，我们以（1,3,960,960）的输入shape为例，运行如下命令测试纯模型推理性能，
- L189: 对于文本识别模型，以（1,3,48,320）shape为例，运行如下命令测试纯模型推理性能
- L195: 文本检测模型和文本识别模型在示例数据集上的的精度和性能数据展示在下表，其中文本检测模型的精度指标为hmeans，文本识别模型的精度指标为acc：
- L199: |模型|hmeans/acc|hmeans/acc（CPU）|纯模型性能（ms）|
- L201: |PP-OCRv6_medium_det|46.73%|46.73%|4.28|
- L202: |PP-OCRv6_medium_rec|77.27%|77.27%|1.43|

## ACL_PyTorch/built-in/ocr/PaddleOCR-VL-1.6

文件清单：

```text
ACL_PyTorch/built-in/ocr/PaddleOCR-VL-1.6/README.md
ACL_PyTorch/built-in/ocr/PaddleOCR-VL-1.6/infer.py
ACL_PyTorch/built-in/ocr/PaddleOCR-VL-1.6/infer_preformance.py
ACL_PyTorch/built-in/ocr/PaddleOCR-VL-1.6/overall_metric.py
ACL_PyTorch/built-in/ocr/PaddleOCR-VL-1.6/paddlex.patch
ACL_PyTorch/built-in/ocr/PaddleOCR-VL-1.6/requirements.txt
ACL_PyTorch/built-in/ocr/PaddleOCR-VL-1.6/test_omnidocbench.py
```

### [ACL_PyTorch/built-in/ocr/PaddleOCR-VL-1.6/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/built-in/ocr/PaddleOCR-VL-1.6/README.md)

文本SHA256：`9550294548fdad0e8da5eed138240653ea5f8adf11d8017fc0c5d4f3106dd7ff`

- L7: - [获取权重](#获取权重)
- L10: - [端到端性能测试](#端到端性能测试)
- L11: - [精度测试](#精度测试)
- L12: - [结果展示](#精度结果)
- L27: url=https://www.modelscope.cn/models/PaddlePaddle/PaddleOCR-VL-1.6
- L41: | 固件与驱动     | 25.2.RC1  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
- L42: | CANN          | 9.1.0     | -                                                                                             |
- L43: | Python        | 3.12.13      | -                                                                                             |
- L44: | vLLM-Ascend   | 0.23.0  | -                                                                                             |
- L45: | PyTorch       | 2.9.0     | -                                                                                             |
- L51: | 固件与驱动     | 25.2.RC1  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
- L52: | CANN          | 9.2.0weekly     | [下载链接](https://www.hiascend.com/cann/download?versionId=794&ids=d806%2Ch0502%2Ch0614%2Ch0703&currentTab=1)|
- L53: | Python        | 3.12.13      | -                                                                                             |
- L54: | PyTorch       | 2.9.0     | -                                                                                             |
- L63: git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
- L74: ## 获取权重
- L76: 下载PaddleOCR-VL-1.6[模型权重](https://modelscope.cn/models/PaddlePaddle/PaddleOCR-VL-1.6)，并放置于本地目录PaddleOCR-VL-1.6
- L109: base_url="http://localhost:8000/v1",
- L128: "url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png"
- L155: wget https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/ais_bench-0.0.2-py3-none-any.whl
- L156: wget https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/aclruntime-0.0.2-cp312-cp312-linux_x86_64.whl
- L164: source_path=$(pip show paddlex | grep Location | awk '{print $2}')
- L169: ### 获取权重
- L170: 下载PP-DocLayoutV3官方onnx[模型权重](https://www.modelscope.cn/models/PaddlePaddle/PP-DocLayoutV3_onnx)（inference.onnx），并放置于本地目录PP-DocLayoutV3-weight
- L183: ### 端到端性能测试
- L184: 在客户端使用推理脚本执行端到端推理，图片为官方示例“[paddleocr_vl_demo.png](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png)”
- L186: wget https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png
- L190: 针对官方示例图片，进行端到端耗时性能测试
- L195: ## 精度测试
- L197: ### 获取数据集
- L199: 创建数据集目录`OmniDocBenchV1.6`，下载多样性文档解析评测集`OmniDocBench`数据集的[pdfs和标注](https://opendatalab.com/OpenDataLab/OmniDocBench)，解压并放置在`OmniDocBenchV1.6`目录下
- L217: - data_dir: 数据集路径
- L221: - vllm_ip: 模型对外接口，默认为"http://127.0.0.1:8000/v1"
- L223: 推理执行完成后，解析结果存放于`OmniDocBenchV1.6_out_pdf`目录。按照如下步骤汇总解析结果，使得文件名与数据集标注对应
- L233: 下文提供一种可参考的环境构建方法，更多详细环境构建方法请访问[OmniDocBench](https://github.com/opendatalab/OmniDocBench)
- L238: git clone https://github.com/opendatalab/OmniDocBench.git
- L246: - 公式精度指标CDM需要额外安装环境
- L250: wget https://nodejs.org/dist/v16.13.1/node-v16.13.1-linux-arm64.tar.xz
- L260: git clone https://github.com/ImageMagick/ImageMagick.git ImageMagick-7.1.2
- L295: 3. 精度测量结果
- L304: 运行overall_metric.py可以得到精度结果：
- L309: ## 精度结果
- L310: 在OmniDocBenchV1.6数据集上的精度为：
- L312: |模型|硬件|overall|官方精度|
- L314: |PaddleOCR-VL-1.6|Atlas 350|95.45%|96.33%|
- L316: ## 模型推理性能
- L318: 使用官方测试图片[paddleocr_vl_demo.png](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png)测试端到端耗时为：
- L322: |PaddleOCR-VL-1.6|Atlas 350|1.39|

## ACL_PyTorch/contrib/cv/detection/Faster_R-CNN_ResNet50

文件清单：

```text
ACL_PyTorch/contrib/cv/detection/Faster_R-CNN_ResNet50/LICENSE
ACL_PyTorch/contrib/cv/detection/Faster_R-CNN_ResNet50/README.md
ACL_PyTorch/contrib/cv/detection/Faster_R-CNN_ResNet50/coco_eval.py
ACL_PyTorch/contrib/cv/detection/Faster_R-CNN_ResNet50/get_info.py
ACL_PyTorch/contrib/cv/detection/Faster_R-CNN_ResNet50/mmdetection.patch
ACL_PyTorch/contrib/cv/detection/Faster_R-CNN_ResNet50/mmdetection_coco_postprocess.py
ACL_PyTorch/contrib/cv/detection/Faster_R-CNN_ResNet50/mmdetection_coco_preprocess.py
ACL_PyTorch/contrib/cv/detection/Faster_R-CNN_ResNet50/modelzoo_level.txt
ACL_PyTorch/contrib/cv/detection/Faster_R-CNN_ResNet50/modify.py
ACL_PyTorch/contrib/cv/detection/Faster_R-CNN_ResNet50/requirements.txt
ACL_PyTorch/contrib/cv/detection/Faster_R-CNN_ResNet50/txt_to_json.py
```

### [ACL_PyTorch/contrib/cv/detection/Faster_R-CNN_ResNet50/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/contrib/cv/detection/Faster_R-CNN_ResNet50/README.md)

文本SHA256：`e8433afe5f7b4e9310fdc4f5f12a5fae66c3f00299aebc363e83ecef85b587c5`

- L11: - [准备数据集](#section183221994411)
- L14: - [模型推理性能](#ZH-CN_TOPIC_0000001172201573)
- L29: url=https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn
- L44: | input    | RGB_FP32 | batchsize x 3 x 1216 x 1216 | NCHW         |
- L51: | boxes    | 100x5 | FLOAT32  | ND          |
- L52: | labels   | 100   | INT64    | ND          |
- L61: | 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
- L62: | CANN                                                         | 6.0.RC1 | -                                                            |
- L63: | Python                                                       | 3.7.5   | -                                                            |
- L64: | PyTorch                                                      | 1.8.1   | -                                                            |
- L65: | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |
- L73: git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
- L81: ├── coco_eval.py                           # 验证推理精度的脚本
- L82: ├── get_info.py                            # 用于获取图像数据集的info文件
- L85: ├── mmdetection_coco_preprocess.py         # 数据集预处理脚本
- L86: └── txt_to_json.py                         # 将推理结果txt文件转换为coco数据集评测精度的标准json格式
- L97: git clone https://github.com/open-mmlab/mmcv.git
- L111: git clone https://github.com/open-mmlab/mmdetection.git
- L127: ## 准备数据集<a name="section183221994411"></a>
- L129: 1. 获取原始数据集和验证集
- L131: 该模型使用[COCO官网](https://cocodataset.org/#download)的coco2017的5千张验证集进行测试，图片与标签分别存放在```val2017/```与```annotations/instances_val2017.json```。
- L134: wget http://images.cocodataset.org/zips/val2017.zip --no-check-certificate
- L137: wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip --no-check-certificate
- L142: 将原始数据集转换为模型输入的数据。
- L144: 将原始数据（.jpeg）转化为二进制文件（.bin）。转化方法参考mmdetection预处理方法，以获得最佳精度。以coco_2017数据集为例，通过缩放、均值方差手段归一化，输出为二进制文件。
- L153: - --image_folder_path: 图像数据集目录。
- L160: 后处理时需要输入数据集.jpg图片的info文件。使用get_info.py脚本，输入已经获得的图片文件,输出生成图片数据集的info文件。
- L169: - 第一个参数为生成的数据集文件格式。
- L171: - 第三个参数为生成的数据集信息文件保存的路径。
- L180: 使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。
- L182: 1. 获取权重文件。
- L185: wget http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --no-check-certificate
- L193: > 由于NPU RoiExtractor算子的特殊性，适当减少其输入框的数量可以在小幅度影响精度的基础上大幅度提高性能，推荐将test_cfg中rpn层的nms_post参数从1000改为500，用户可以自行决定是否应用此项改动。
- L230: - 第二个参数为模型的权重文件。
- L236: 4. 优化onnx算子，消除冗余算子(安装[auto-optimzer](https://gitcode.com/ascend/msadvisor/tree/master/auto-optimizer)工具)：
- L252: >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。
- L261: | npu-smi 22.0.0                       Version: 22.0.2                                       |
- L266: | 0       310P3     | OK              | 16.3         53                1236 / 1236           |
- L267: | 0       0         | 0000:86:00.0    | 0            4060 / 21534                            |
- L292: 请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。
- L307: c.  精度验证。
- L323: 评测结果的mAP值需要使用官方的pycocotools工具，首先将后处理输出的txt文件转化为coco数据集评测精度的标准json格式。
- L350: # 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>
- L352: 调用ACL接口推理计算，性能参考下列数据。
- L354: | 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 | 基准性能 |
- L356: | 300I PRO | 1 | coco2017 | 37.2 | 14.84 | 19.4 |

## ACL_PyTorch/contrib/cv/segmentation/DeeplabV3+

文件清单：

```text
ACL_PyTorch/contrib/cv/segmentation/DeeplabV3+/LICENSE
ACL_PyTorch/contrib/cv/segmentation/DeeplabV3+/README.md
ACL_PyTorch/contrib/cv/segmentation/DeeplabV3+/deeplabV3plus_pth2onnx.py
ACL_PyTorch/contrib/cv/segmentation/DeeplabV3+/modelzoo_level.txt
ACL_PyTorch/contrib/cv/segmentation/DeeplabV3+/post_deeplabv3plus_pytorch.py
ACL_PyTorch/contrib/cv/segmentation/DeeplabV3+/preprocess_deeplabv3plus_pytorch.py
ACL_PyTorch/contrib/cv/segmentation/DeeplabV3+/requirements.txt
```

### [ACL_PyTorch/contrib/cv/segmentation/DeeplabV3+/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/contrib/cv/segmentation/DeeplabV3%2B/README.md)

文本SHA256：`9662032ac9d0e29b90981810789bfeaf13251a5e37171d766c8527e17f41ee9a`

- L15: - [准备数据集](#section183221994411)
- L18: - [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)
- L26: - 参考论文：[Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. (2018)](https://arxiv.org/abs/1802.02611)
- L32: url=https://github.com/jfzhang95/pytorch-deeplab-xception
- L44: | input    | RGB_FP32 | batchsize x 3 x 513 x 513 | NCHW         |
- L51: | output  | FLOAT16  | batchsize x 21 x 513 x 513 | ND           |
- L63: | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
- L64: | CANN                                                         | 6.0.RC1 | -                                                            |
- L65: | Python                                                       | 3.7.5   | -                                                            |
- L66: | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |
- L77: git clone https://github.com/jfzhang95/pytorch-deeplab-xception
- L88: ├──dataset                               //数据集目录
- L106: ## 准备数据集<a name="section183221994411"></a>
- L108: 1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）
- L110: 请用户需自行获取VOCtrainval_11-May-2012 数据集，上传数据集到服务器任意目录并解压（以当前路径"./datasets"为例） VOCtrainval_11-May-2012数据集目录结构如下：
- L121: ├── JPEGImages                 //验证数据集文件夹
- L125: 2. 数据预处理，将原始数据集转换为模型输入的数据。
- L134: - 第一个参数：验证数据集路径
- L143: 使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。
- L145: 1. 获取权重文件。
- L147: [deeplab-resnet.pth.tar](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Deeplabv3%2B/PTH/deeplab-resnet.pth.tar)
- L187: | 0       310P3     | OK              | 15.8         42                0    / 0              |
- L188: | 0       0         | 0000:82:00.0    | 0            1074 / 21534                            |
- L190: | 1       310P3     | OK              | 15.4         43                0    / 0              |
- L191: | 0       1         | 0000:89:00.0    | 0            1070 / 21534                            |
- L209: - --enable\_small\_channel:使能后在channel<=4的卷积层会有性能收益。
- L217: 请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。
- L238: 3. 精度验证。
- L240: 调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。
- L252: # 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>
- L254: 调用ACL接口推理计算，性能参考下列数据。
- L256: | 芯片型号 | Batch Size   | 数据集 | 精度(mIOU) | 性能 |
- L258: |  300I PRO  |  1  |VOCtrainval_11-May-2012       |     78.43      |     165.545      |
- L259: |  300I PRO  |  4  |VOCtrainval_11-May-2012       |     78.43      |     162.44      |
- L260: |  300I PRO  |  8  |VOCtrainval_11-May-2012       |     78.43      |     163.559      |
- L261: |  300I PRO  |  16  |VOCtrainval_11-May-2012       |     78.43      |    163.779       |
- L262: |  300I PRO  |  32  |VOCtrainval_11-May-2012       |     78.43      |    83.019       |
- L263: |  300I PRO  |  64  |VOCtrainval_11-May-2012       |     78.43      |    84.96       |

## ACL_PyTorch/contrib/cv/segmentation/Segformer

文件清单：

```text
ACL_PyTorch/contrib/cv/segmentation/Segformer/README.md
ACL_PyTorch/contrib/cv/segmentation/Segformer/Segformer.patch
ACL_PyTorch/contrib/cv/segmentation/Segformer/Segformer_postprocess.py
ACL_PyTorch/contrib/cv/segmentation/Segformer/Segformer_preprocess.py
ACL_PyTorch/contrib/cv/segmentation/Segformer/modelzoo_level.txt
ACL_PyTorch/contrib/cv/segmentation/Segformer/optimize_onnx.py
ACL_PyTorch/contrib/cv/segmentation/Segformer/requirements.txt
```

### [ACL_PyTorch/contrib/cv/segmentation/Segformer/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/contrib/cv/segmentation/Segformer/README.md)

文本SHA256：`75e85f70603cc8094a09a58fe22a0dfb7a51e6a4e45991a8ec422c705586047c`

- L11: - [准备数据集](#section183221994411)
- L14: - [模型推理性能](#ZH-CN_TOPIC_0000001172201573)
- L22: Segformer是一个简单、高效但功能强大的语义分割框架，它将 Transformer 与轻量级多层感知 (MLP) 解码器相结合。SegFormer 有两个吸引人的特性：1) SegFormer 包含一个新颖的分层结构的 Transformer 编码器，它输出多尺度特征。它不需要位置编码，从而避免了在测试分辨率与训练不同时导致性能下降的位置编码插值。2) SegFormer 避免了复杂的解码器。所提出的 MLP 解码器聚合来自不同层的信息，从而结合局部注意力和全局注意力来呈现强大的表示。
- L28: url=https://github.com/open-mmlab/mmsegmentation/tree/master/configs/segformer
- L40: | input    | RGB_FP32 | batchsize x 3 x 1024 x 2048 | NCHW         |
- L47: | output  | 1 x batchsize x 1024 x 2048 | INT64  | NCHW    |
- L58: | 固件与驱动                                                   | 22.0.4  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
- L59: | CANN                                                         | 6.0.RC1 | -                                                            |
- L60: | Python                                                       | 3.7.5   | -                                                            |
- L61: | PyTorch                                                      | 1.7.0   | -                                                            |
- L62: | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |
- L70: git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
- L77: ├── Segformer_preprocess.py    # 数据集预处理脚本
- L85: git clone https://github.com/open-mmlab/mmsegmentation.git
- L101: git clone https://github.com/open-mmlab/mmcv.git
- L110: git clone https://gitcode.com/ascend/msadvisor.git
- L117: ## 准备数据集<a name="section183221994411"></a>
- L119: 1. 获取原始数据集。
- L121: 下载 [cityscpaes数据集](https://www.cityscapes-dataset.com)，解压后数据集目录结构如下：
- L136: 将原始数据集转换为模型输入的二进制数据。
- L152: 使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。
- L154: 1. 获取权重文件。
- L157: wget https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth
- L200: > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。
- L212: | 0       310P3     | OK              | 15.8         42                0    / 0              |
- L213: | 0       0         | 0000:82:00.0    | 0            1074 / 21534                            |
- L215: | 1       310P3     | OK              | 15.4         43                0    / 0              |
- L216: | 0       1         | 0000:89:00.0    | 0            1070 / 21534                            |
- L249: 请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。
- L265: -   --input：预处理后的数据集路径。
- L274: c.  精度验证。
- L276: 我们将后处理与精度验证一起放在了Segformer_postprocess.py文件中，脚本执行完毕后即可生成精度结果。
- L285: -   --dataset_path：cityscpaes数据集所在父路径；比如：若cityscpaes存放在/opt/npu/cityscpaes，则--dataset_path=/opt/npu/
- L288: d.  性能验证。
- L290: 使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，命令如下：
- L297: # 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>
- L299: 调用ACL接口推理计算，性能参考下列数据。
- L301: | 芯片型号 | Batch Size |   数据集   |      精度      |    性能    |  基准性能  |
- L303: |   300I PRO  |      1     | cityscapes |  mIoU = 75.94  |  6.01 fps | 10.61 fps |
- L304: |   300I PRO  |      4     | cityscapes |  mIoU = 75.94  |  5.94 fps | 10.65 fps |

## ACL_PyTorch/contrib/cv/segmentation/Swin-Transformer-Semantic-Segmentation

文件清单：

```text
ACL_PyTorch/contrib/cv/segmentation/Swin-Transformer-Semantic-Segmentation/LICENSE
ACL_PyTorch/contrib/cv/segmentation/Swin-Transformer-Semantic-Segmentation/README.md
ACL_PyTorch/contrib/cv/segmentation/Swin-Transformer-Semantic-Segmentation/Swin-Transformer-Semantic-Segmentation_postprocess.py
ACL_PyTorch/contrib/cv/segmentation/Swin-Transformer-Semantic-Segmentation/Swin-Transformer-Semantic-Segmentation_preprocess.py
ACL_PyTorch/contrib/cv/segmentation/Swin-Transformer-Semantic-Segmentation/Swin-Transformer-Semantic-Segmentation_pth2onnx.py
ACL_PyTorch/contrib/cv/segmentation/Swin-Transformer-Semantic-Segmentation/change.patch
ACL_PyTorch/contrib/cv/segmentation/Swin-Transformer-Semantic-Segmentation/requirements.txt
```

### [ACL_PyTorch/contrib/cv/segmentation/Swin-Transformer-Semantic-Segmentation/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/contrib/cv/segmentation/Swin-Transformer-Semantic-Segmentation/README.md)

文本SHA256：`74a436aeb7ab8eb47dd8f5fda9534fda462b5c6a9602dc7d7b22219b5741e3ad`

- L21: url = https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation
- L33: | input    | RGB_FP32 | batchsize x 3 x 512 x 512 | NCHW     |
- L39: | output   | FLOAT32  | batchsize x 150 x 512 x 512 | ND       |
- L53: | 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
- L54: | CANN                                                         | 5.1.RC2 | -                                                            |
- L55: | Python                                                       | 3.7.5   | -                                                            |
- L56: | PyTorch                                                      | 1.7.1   | -                                                            |
- L57: | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |
- L68: git clone https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation.git   # 克隆仓库的代码
- L80: ## 准备数据集
- L82: 1. 获取原始数据集。
- L84: 本推理项目使用 ADE20K 的 2000 张验证集图片来验证模型精度，请进入 [ADE20K官网](http://groups.csail.mit.edu/vision/datasets/ADE20K/) 自行下载数据集（需要先注册）。在Swin-Transformer-Semantic-Segmentation 目录中创建data文件夹，ade数据采集存放在data中。最终，验证集原始图片与标注图片分别存放在annotations/validation和images/validation目录下。目录结构如下：
- L100: 2. 数据预处理，将原始数据集转换为模型输入的数据。
- L122: 使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。
- L124: 1. 获取权重文件。
- L126: 获取预训练好的 [pth权重文件](https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_small_patch4_window7_512x512.pth)，下载完成后将权重 pth 文件存放于 Swin-Transformer-Semantic-Segmentation/checkpoint 目录下。
- L145: + --checkpoint: 预训练权重所在路径
- L167: | npu-smi 22.0.0                       Version: 22.0.2                                       |
- L172: | 0       310P3     | OK              | 16.6         56                0    / 0              |
- L173: | 0       0         | 0000:5E:00.0    | 0            935  / 21534                            |
- L199: 请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。
- L219: 3. 精度验证。
- L221: 使用mv命令将Swin-Transformer-Semantic-Segmentation_postprocess.py脚本移动至Swin-Transformer-Semantic-Segmentation目录下。执行该脚本可以获得mIoU精度数据。
- L236: ${infer_result_datalog}是推理结果存放的文件夹名称。运行成功后，程序会打印出模型的mIoU精度指标。
- L238: 4. 性能验证。
- L240: 可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：
- L253: # 模型推理性能&精度
- L255: 1. 性能对比
- L257: 在 300I PRO 设备上，当 batchsize 为 1 时模型的性能为 19.24 fps.
- L259: | 芯片型号    | Batch Size | 数据集 | 精度   | 性能     |
- L261: | 300I Pro | 1          | ADE20K | 48.06% | 19.24fps |
- L263: 注：当 batchsize 为 4 或更高时，因内存不足导致推理失败，无法获取精度和性能数据。

## ACL_PyTorch/contrib/cv/segmentation/UNet

文件清单：

```text
ACL_PyTorch/contrib/cv/segmentation/UNet/LICENCE
ACL_PyTorch/contrib/cv/segmentation/UNet/README.md
ACL_PyTorch/contrib/cv/segmentation/UNet/json_parse.py
ACL_PyTorch/contrib/cv/segmentation/UNet/modelzoo_level.txt
ACL_PyTorch/contrib/cv/segmentation/UNet/postprocess_unet_pth.py
ACL_PyTorch/contrib/cv/segmentation/UNet/preprocess_unet_pth.py
ACL_PyTorch/contrib/cv/segmentation/UNet/public_address_statement.md
ACL_PyTorch/contrib/cv/segmentation/UNet/requirements.txt
ACL_PyTorch/contrib/cv/segmentation/UNet/revise_UNet.py
ACL_PyTorch/contrib/cv/segmentation/UNet/unet_pth2onnx.py
```

### [ACL_PyTorch/contrib/cv/segmentation/UNet/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/contrib/cv/segmentation/UNet/README.md)

文本SHA256：`22cc79273828e8262c1d43a40e13ad5606ff04a98bc0c2a0267cd7dfbff8505b`

- L12: - [准备数据集](#section183221994411)
- L15: - [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)
- L26: url=https://github.com/milesial/Pytorch-UNet
- L37: | input    | FLOAT32 | batchsize x 3 x 572 x 572 | NCHW         |
- L43: | output   | FLOAT32  | batchsize x 3 x 388 x 388 | NCHW   |
- L53: | 固件与驱动                                                   | 22.0.4  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
- L54: | CANN                                                         | 6.3.RC1 | -                                                            |
- L55: | Python                                                       | 3.7.5   | -                                                            |
- L56: | PyTorch                                                      | 1.7.0   | -                                                            |
- L57: | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |
- L66: git clone https://github.com/milesial/Pytorch-UNet.git
- L79: ## 准备数据集<a name="section183221994411"></a>
- L81: 1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）
- L83: 本模型支持carvana数据集，[下载链接](https://www.kaggle.com/competitions/carvana-image-masking-challenge/data)。数据集train.zip以及train_masks.zip分别作为训练和标签文件，上传并解压到源码包路径下，目录结构如下：
- L91: 2. 数据预处理，将原始数据集转换为模型输入的数据。
- L99: - --src_path：原始数据集所在路径。
- L106: 使用Pytorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。
- L108: 1. 获取权重文件。
- L110: UNet.pth权重文件[下载链接](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Unet/PTH/UNet.pth)。
- L147: | 0       310P3     | OK              | 15.8         42                0    / 0              |
- L148: | 0       0         | 0000:82:00.0    | 0            1074 / 21534                            |
- L150: | 1       310P3     | OK              | 15.4         43                0    / 0              |
- L151: | 0       1         | 0000:89:00.0    | 0            1070 / 21534                            |
- L177: 请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。
- L192: 3. 精度验证。
- L216: 4. 性能验证。
- L218: 可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：
- L228: # 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>
- L230: 调用ACL接口推理计算，性能参考下列数据。
- L232: | Batch Size | 数据集  | 精度         | 300I Pro       | A500 A2     |
- L234: | 1          | carvana | IOU:0.986305 | 75.0603     | 12.78     |
- L235: | 4          | carvana | IOU:0.986305 | 71.2920     | 11.75     |
- L236: | 8          | carvana | IOU:0.986305 | 68.5334     | 11.56     |
- L237: | 16         | carvana | IOU:0.986305 | 67.7102     | 11.29     |
- L238: | 32         | carvana | IOU:0.986305 | 65.5027     | 11.18     |
- L239: | 64         | carvana | IOU:0.986305 | 49.0184     | NA        |
- L240: |            |         | **最优性能** | **75.0603** | **12.78** |

## MindIE/LLM/Qwen3/Qwen3-8B

文件清单：

```text
MindIE/LLM/Qwen3/Qwen3-8B/README.md
```

### [MindIE/LLM/Qwen3/Qwen3-8B/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/MindIE/LLM/Qwen3/Qwen3-8B/README.md)

文本SHA256：`3b1e7d9c64fbf0abf0eb9aa6c804c7ae20ffdff655d867971bf090105c7fb978`

- L5: - **思维模式**（用于复杂的逻辑推理、数学和编码）和**非思维模式**（用于高效、通用的对话）在单个模型内无缝切换，确保跨各种场景的最佳性能。
- L8: - **在代理能力方面的专业知识**，能够在思考模式和非思考模式下与外部工具精确集成，在基于代理的复杂任务中实现开源模型中的领先性能。
- L11: ## 权重
- L13: **权重下载**
- L15: - [Qwen3-8B](https://www.modelscope.cn/models/Qwen/Qwen3-8B/files)
- L19: 前往[昇腾社区/开发资源](https://support.huawei.com/enterprise/zh/ascend-computing/mindie-pid-261803968/software/266130647?idAbsPath=fixnode01|23710424|251366513|254884019|261408772|261803968)下载适配本模型的镜像包mindie:2.2.T10-800I-A2-py3.11-openeuler24.03-lts-aarch64.tar.gz
- L78: > 4. 设定权重挂载的路径，`-v /path-to-weights:/path-to-weights:ro`，注意，如果使用普通用户镜像，权重路径所属应为镜像内默认的1000用户，且权限可设置为750。可使用以下命令进行修改：
- L111: --model_path {权重路径} \
- L116: ### 性能测试
- L125: 具体执行batch=1, 输入长度256, 输出长度256用例的2卡并行性能测试命令为：
- L130: > 注：ModelTest为大模型的性能和精度提供测试功能。使用文档请参考`${ATB_SPEED_HOME_PATH}/tests/modeltest/README.md`
- L205: > 注: 服务化推理的更多信息请参考[MindIE Service用户指南](https://www.hiascend.com/document/detail/zh/mindie/100/mindieservice/servicedev/mindie_service0001.html)
- L209: - 本代码仓提到的数据集和模型仅作为示例,这些数据集和模型仅供您用于非商业目的,如您使用这些数据集和模型来完成示例,请您特别注意应遵守对应数据集和模型的License,如您因使用数据集或模型而产生侵权纠纷,华为不承担任何责任。

## MindIE/MultiModal/InternVL/InternVL2/Internvl2_40b

文件清单：

```text
MindIE/MultiModal/InternVL/InternVL2/Internvl2_40b/README.md
```

### [MindIE/MultiModal/InternVL/InternVL2/Internvl2_40b/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/MindIE/MultiModal/InternVL/InternVL2/Internvl2_40b/README.md)

文本SHA256：`2621b08808cc36a1481340a803c3c2ca483b06557bb1a16e554863d2f4f59539`

- L20: 前往[昇腾社区/开发资源](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)下载适配本模型的镜像包：1.0.0-800I-A2-py311-openeuler24.03-lts
- L85: bash /usr/local/Ascend/atb-models/examples/models/internvl/run_pa.sh --run --trust_remote_code ${权重路径} ${图片或视频所在文件夹路径}
- L97: - 修改`MindIE-Service`配置文件`config.json`，在Atlas 800I A2 64GB环境下，推荐使用以下配置，请自行修改`modelWeightPath`为实际权重路径：
- L143: curl http://localhost:1025/v1/chat/completions -d '{
- L162: curl http://localhost:1025/generate -d '{
- L178: - 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。

## MindIE/MultiModal/InternVL/InternVL2/Internvl2_8b

文件清单：

```text
MindIE/MultiModal/InternVL/InternVL2/Internvl2_8b/README.md
```

### [MindIE/MultiModal/InternVL/InternVL2/Internvl2_8b/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/MindIE/MultiModal/InternVL/InternVL2/Internvl2_8b/README.md)

文本SHA256：`a4f5c0695304aab763207fe247d150d0109ee0c4eaef28bcc5b7476cf9aefca8`

- L20: 前往[昇腾社区/开发资源](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)下载适配本模型的镜像包：1.0.0-800I-A2-py311-openeuler24.03-lts
- L85: bash /usr/local/Ascend/atb-models/examples/models/internvl/run_pa.sh --run --trust_remote_code ${权重路径} ${图片或视频所在文件夹路径}
- L97: - 修改`MindIE-Service`配置文件`config.json`，在Atlas 800I A2 64GB环境下，推荐使用以下配置，请自行修改`modelWeightPath`为实际权重路径：
- L142: curl http://localhost:1025/v1/chat/completions -d '{
- L161: curl http://localhost:1025/generate -d '{
- L177: - 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。

## MindIE/MultiModal/Qwen2-VL/Qwen2_vl_72b_instruct

文件清单：

```text
MindIE/MultiModal/Qwen2-VL/Qwen2_vl_72b_instruct/README.md
```

### [MindIE/MultiModal/Qwen2-VL/Qwen2_vl_72b_instruct/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/MindIE/MultiModal/Qwen2-VL/Qwen2_vl_72b_instruct/README.md)

文本SHA256：`795c7123ceee60ed6e92a9d9a0a1d76dd1f5c52c61027737ec04224a4c3a6540`

- L20: 前往[昇腾社区/开发资源](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)下载适配本模型的镜像包：1.0.0-800I-A2-py311-openeuler24.03-lts
- L73: # 模型权重路径
- L86: # dataset_path优先级比input_image高，若要推理整个数据集，base_cmd入参中添加 ```--dataset_path $dataset_path \```
- L99: - 性能测试样例（Atlas 800I A2 推理服务器 32GB）
- L107: - 更详细的性能数据，如首token时延，参考终端performance输出
- L108: - 性能测试样例（Atlas 800I A2 推理服务器 64GB）
- L116: - 更详细的性能数据，如首token时延，参考终端performance输出
- L232: - 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。

## MindIE/MultiModal/Qwen2-VL/Qwen2_vl_7b_instruct

文件清单：

```text
MindIE/MultiModal/Qwen2-VL/Qwen2_vl_7b_instruct/README.md
```

### [MindIE/MultiModal/Qwen2-VL/Qwen2_vl_7b_instruct/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/MindIE/MultiModal/Qwen2-VL/Qwen2_vl_7b_instruct/README.md)

文本SHA256：`558833263bbafb25dd8772aae44a749ac2495a667e7ee49e5bfbbe7c81168353`

- L20: 前往[昇腾社区/开发资源](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)下载适配本模型的镜像包：1.0.0-800I-A2-py311-openeuler24.03-lts
- L73: # 模型权重路径
- L86: # dataset_path优先级比input_image高，若要推理整个数据集，base_cmd入参中添加 ```--dataset_path $dataset_path \```
- L99: - 性能测试样例（Atlas 800I A2 推理服务器 32GB）
- L107: - 更详细的性能数据，如首token时延，参考终端performance输出
- L108: - 性能测试样例（Atlas 800I A2 推理服务器 64GB）
- L116: - 更详细的性能数据，如首token时延，参考终端performance输出
- L232: - 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。

## PyTorch/built-in/autonoumous_driving/BEVFormer

文件清单：

```text
PyTorch/built-in/autonoumous_driving/BEVFormer/.gitignore
PyTorch/built-in/autonoumous_driving/BEVFormer/LICENSE
PyTorch/built-in/autonoumous_driving/BEVFormer/README.md
PyTorch/built-in/autonoumous_driving/BEVFormer/README_ORI.md
PyTorch/built-in/autonoumous_driving/BEVFormer/docs/getting_started.md
PyTorch/built-in/autonoumous_driving/BEVFormer/docs/install.md
PyTorch/built-in/autonoumous_driving/BEVFormer/docs/prepare_dataset.md
PyTorch/built-in/autonoumous_driving/BEVFormer/mmcv_need/base_runner.py
PyTorch/built-in/autonoumous_driving/BEVFormer/mmcv_need/distributed.py
PyTorch/built-in/autonoumous_driving/BEVFormer/mmcv_need/epoch_based_runner.py
PyTorch/built-in/autonoumous_driving/BEVFormer/mmcv_need/modulated_deform_conv.py
PyTorch/built-in/autonoumous_driving/BEVFormer/mmcv_need/optimizer.py
PyTorch/built-in/autonoumous_driving/BEVFormer/mmcv_need/points_in_polygons_npu.cpp
PyTorch/built-in/autonoumous_driving/BEVFormer/mmcv_need/runtime.txt
PyTorch/built-in/autonoumous_driving/BEVFormer/mmdet3d_need/__init__.py
PyTorch/built-in/autonoumous_driving/BEVFormer/mmdet3d_need/nuscenes_dataset.py
PyTorch/built-in/autonoumous_driving/BEVFormer/mmdet3d_need/runtime.txt
PyTorch/built-in/autonoumous_driving/BEVFormer/mmdet_need/__init__.py
PyTorch/built-in/autonoumous_driving/BEVFormer/mmdet_need/resnet.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/__init__.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/datasets/coco_instance.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/datasets/kitti-3d-3class.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/datasets/kitti-3d-car.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/datasets/lyft-3d.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/datasets/nuim_instance.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/datasets/nus-3d.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/datasets/nus-mono3d.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/datasets/range100_lyft-3d.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/datasets/s3dis-3d-5class.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/datasets/s3dis_seg-3d-13class.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/datasets/scannet-3d-18class.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/datasets/scannet_seg-3d-20class.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/datasets/sunrgbd-3d-10class.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/datasets/waymoD5-3d-3class.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/datasets/waymoD5-3d-car.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/default_runtime.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/models/3dssd.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/models/cascade_mask_rcnn_r50_fpn.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/models/centerpoint_01voxel_second_secfpn_nus.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/models/centerpoint_02pillar_second_secfpn_nus.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/models/fcos3d.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/models/groupfree3d.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/models/h3dnet.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/models/hv_pointpillars_fpn_lyft.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/models/hv_pointpillars_fpn_nus.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/models/hv_pointpillars_fpn_range100_lyft.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/models/hv_pointpillars_secfpn_kitti.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/models/hv_pointpillars_secfpn_waymo.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/models/hv_second_secfpn_kitti.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/models/hv_second_secfpn_waymo.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/models/imvotenet_image.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/models/mask_rcnn_r50_fpn.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/models/paconv_cuda_ssg.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/models/paconv_ssg.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/models/parta2.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/models/pointnet2_msg.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/models/pointnet2_ssg.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/models/votenet.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/schedules/cosine.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/schedules/cyclic_20e.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/schedules/cyclic_40e.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/schedules/mmdet_schedule_1x.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/schedules/schedule_2x.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/schedules/schedule_3x.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/schedules/seg_cosine_150e.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/schedules/seg_cosine_200e.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/_base_/schedules/seg_cosine_50e.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/bevformer/bevformer_base.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/bevformer/bevformer_small.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/bevformer/bevformer_tiny.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/bevformer_fp16/bevformer_tiny_fp16.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/bevformerv2/bevformerv2-r50-t1-24ep.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/bevformerv2/bevformerv2-r50-t1-48ep.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/bevformerv2/bevformerv2-r50-t1-base-24ep.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/bevformerv2/bevformerv2-r50-t1-base-48ep.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/bevformerv2/bevformerv2-r50-t2-24ep.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/bevformerv2/bevformerv2-r50-t2-48ep.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/bevformerv2/bevformerv2-r50-t8-24ep.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/datasets/custom_lyft-3d.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/datasets/custom_nus-3d.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/configs/datasets/custom_waymo-3d.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/__init__.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/bevformer/__init__.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/bevformer/apis/__init__.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/bevformer/apis/mmdet_train.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/bevformer/apis/test.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/bevformer/apis/train.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/bevformer/dense_heads/__init__.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/bevformer/dense_heads/bev_head.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/bevformer/detectors/__init__.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/bevformer/detectors/bevformer.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/bevformer/detectors/bevformerV2.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/bevformer/detectors/bevformer_fp16.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/bevformer/hooks/__init__.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/bevformer/hooks/custom_hooks.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/bevformer/modules/__init__.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/bevformer/modules/custom_base_transformer_layer.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/bevformer/modules/decoder.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/bevformer/modules/encoder.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/bevformer/modules/group_attention.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/bevformer/modules/multi_scale_deformable_attn_function.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/bevformer/modules/transformer.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/bevformer/modules/transformerV2.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/bevformer/runner/__init__.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/bevformer/runner/epoch_based_runner.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/core/bbox/assigners/__init__.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/core/bbox/assigners/hungarian_assigner_3d.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/core/bbox/coders/__init__.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/core/bbox/match_costs/__init__.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/core/bbox/match_costs/match_cost.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/core/bbox/util.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/core/evaluation/__init__.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/core/evaluation/eval_hooks.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/core/evaluation/kitti2waymo.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/datasets/__init__.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/datasets/builder.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/datasets/nuscenes_dataset.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/datasets/nuscenes_dataset_v2.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/datasets/nuscenes_mono_dataset.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/datasets/nuscnes_eval.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/datasets/pipelines/__init__.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/datasets/pipelines/augmentation.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/datasets/pipelines/dd3d_mapper.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/datasets/pipelines/formating.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/datasets/pipelines/loading.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/datasets/samplers/__init__.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/datasets/samplers/distributed_sampler.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/datasets/samplers/group_sampler.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/datasets/samplers/sampler.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/dd3d/__init__.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/dd3d/datasets/__init__.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/dd3d/datasets/nuscenes.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/dd3d/datasets/transform_utils.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/dd3d/layers/iou_loss.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/dd3d/layers/normalization.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/dd3d/layers/smooth_l1_loss.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/dd3d/modeling/__init__.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/dd3d/modeling/core.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/dd3d/modeling/disentangled_box3d_loss.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/dd3d/modeling/fcos2d.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/dd3d/modeling/fcos3d.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/dd3d/modeling/nuscenes_dd3d.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/dd3d/modeling/prepare_targets.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/dd3d/structures/__init__.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/dd3d/structures/boxes3d.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/dd3d/structures/image_list.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/dd3d/structures/pose.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/dd3d/structures/transform3d.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/dd3d/utils/comm.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/dd3d/utils/geometry.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/dd3d/utils/tasks.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/dd3d/utils/tensor2d.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/dd3d/utils/visualization.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/models/backbones/__init__.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/models/backbones/vovnet.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/models/hooks/__init__.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/models/hooks/hooks.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/models/opt/__init__.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/models/opt/adamw.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/models/utils/__init__.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/models/utils/bricks.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/models/utils/grid_mask.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/models/utils/position_embedding.py
PyTorch/built-in/autonoumous_driving/BEVFormer/projects/mmdet3d_plugin/models/utils/visual.py
PyTorch/built-in/autonoumous_driving/BEVFormer/public_address_statement.md
PyTorch/built-in/autonoumous_driving/BEVFormer/requirements.txt
PyTorch/built-in/autonoumous_driving/BEVFormer/test/env_npu.sh
PyTorch/built-in/autonoumous_driving/BEVFormer/test/train_full_8p_base_fp32.sh
PyTorch/built-in/autonoumous_driving/BEVFormer/test/train_performance_8p_base_fp32.sh
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/__init__.py
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/analysis_tools/__init__.py
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/analysis_tools/analyze_logs.py
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/analysis_tools/benchmark.py
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/analysis_tools/get_params.py
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/analysis_tools/visual.py
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/create_data.py
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/data_converter/__init__.py
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/data_converter/create_gt_database.py
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/data_converter/indoor_converter.py
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/data_converter/kitti_converter.py
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/data_converter/kitti_data_utils.py
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/data_converter/lyft_converter.py
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/data_converter/lyft_data_fixer.py
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/data_converter/nuimage_converter.py
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/data_converter/nuscenes_converter.py
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/data_converter/s3dis_data_utils.py
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/data_converter/scannet_data_utils.py
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/data_converter/sunrgbd_data_utils.py
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/data_converter/waymo_converter.py
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/dist_test.sh
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/dist_train.sh
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/fp16/dist_train.sh
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/fp16/train.py
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/misc/browse_dataset.py
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/misc/fuse_conv_bn.py
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/misc/print_config.py
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/misc/visualize_results.py
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/model_converters/convert_votenet_checkpoints.py
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/model_converters/publish_model.py
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/model_converters/regnet2mmdet.py
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/test.py
PyTorch/built-in/autonoumous_driving/BEVFormer/tools/train.py
```

### [PyTorch/built-in/autonoumous_driving/BEVFormer/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/built-in/autonoumous_driving/BEVFormer/README.md)

文本SHA256：`0a4ed9474d8b00d8cb409d8ef8e602c5cef4005daa0bfa992210b48d909cd881`

- L1: # 当前仓已不维护，请跳转至 https://gitcode.com/ascend/DrivingSDK/tree/master/model_examples/BEVFormer
- L12: url=https://github.com/fundamentalvision/BEVFormer
- L20: | R101-DCN          | BEVFormer-base  |       FP32    |
- L26: - 推荐参考[配套资源文档](https://www.hiascend.com/developer/download/commercial)使用最新的配套版本。
- L39: <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0003.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit">驱动固件安装指南</a> 》</td>
- L48: <td>《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0000.html">CANN 软件安装指南</a> 》</td>
- L53: <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/configandinstg/instg/insg_0001.html">Ascend Extension for PyTorch 配置与安装</a> 》</td>
- L63: 请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
- L69: git clone -b 1.x https://github.com/open-mmlab/mmcv.git
- L84: git clone -b v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git
- L93: git clone -b v2.24.0 https://github.com/open-mmlab/mmdetection.git
- L101: python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
- L108: 参考mxDriving官方gitcode仓README安装编译构建并安装mxDriving包：[参考链接](https://gitcode.com/ascend/mxDriving)
- L111: ## 准备数据集
- L113: 1. 用户需自行下载 nuScenes V1.0 full 和 CAN bus 数据集，结构如下：
- L125: ## 下载预训练权重
- L126: 创建 ckpts 文件夹，将预训练权重 r101_dcn_fcos3d_pretrain.pth 放入其中
- L138: bash test/train_performance_8p_base_fp32.sh # 8卡性能
- L145: |  8p-Atlas 800T A2 | R101-DCN    | BEVFormer-base  |       FP32    |        4     |      46.54   |      38.36   |      2.915    |
- L146: |  8p-竞品A   | R101-DCN          | BEVFormer-base  |       FP32    |        4     |      44.29   |      35.16   |      3.320    |
- L148: 说明：上表为历史数据，仅供参考。2024年12月31日更新的性能数据如下：
- L149: | NAME | 精度类型 | FPS |
- L151: | 8p-竞品 | FP32 | 3.32 |
- L152: | 8p-Atlas 900 A2 PoDc | FP32 | 3.79 |

## PyTorch/built-in/cv/detection/GroundingDINO_for_Pytorch

文件清单：

```text
PyTorch/built-in/cv/detection/GroundingDINO_for_Pytorch/README.md
PyTorch/built-in/cv/detection/GroundingDINO_for_Pytorch/groundingdino_npu/adapter_groundingdino.py
PyTorch/built-in/cv/detection/GroundingDINO_for_Pytorch/groundingdino_npu/env_npu.sh
PyTorch/built-in/cv/detection/GroundingDINO_for_Pytorch/groundingdino_npu/finetune_refcoco.sh
PyTorch/built-in/cv/detection/GroundingDINO_for_Pytorch/groundingdino_npu/image_demo_npu.py
PyTorch/built-in/cv/detection/GroundingDINO_for_Pytorch/groundingdino_npu/inference.sh
PyTorch/built-in/cv/detection/GroundingDINO_for_Pytorch/groundingdino_npu/mm_grounding_dino_swin-b_finetune_b2_refcoco.py
PyTorch/built-in/cv/detection/GroundingDINO_for_Pytorch/groundingdino_npu/mm_grounding_dino_swin-b_inference.py
PyTorch/built-in/cv/detection/GroundingDINO_for_Pytorch/groundingdino_npu/requirements.txt
PyTorch/built-in/cv/detection/GroundingDINO_for_Pytorch/groundingdino_npu/train_npu.py
```

### [PyTorch/built-in/cv/detection/GroundingDINO_for_Pytorch/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/built-in/cv/detection/GroundingDINO_for_Pytorch/README.md)

文本SHA256：`299498fd62b2eb3537997a8f1db6c9777619946fd19112a926ffdf4e42441f9b`

- L10: -   [准备数据集](#准备数据集)
- L22: Grounding DINO是一个开放集目标检测模型，可以根据输入文本检测文本对应的目标。该模型通过将语言模型引入闭集检测器以进行开放集概念泛化，可以检测带有人类输入的任意对象。本仓库主要将Grounding DINO模型迁移到了昇腾NPU上，并进行了性能优化。
- L24: 官方仓：https://github.com/IDEA-Research/GroundingDINO
- L41: MMDetection仓：https://github.com/open-mmlab/mmdetection/tree/main/configs/grounding_dino
- L48: url=https://gitcode.com/ascend/ModelZoo-PyTorch.git
- L55: 该模型为不随版本演进模型（随版本演进模型范围可在[此处](https://gitcode.com/ascend/ModelZoo-PyTorch/blob/master/PyTorch/README.CN.md)查看），未在最新昇腾配套软件中适配验证，您可以：
- L56: 1. 根据下面提供PyTorch版本在[软件版本配套表](https://gitcode.com/ascend/pytorch#%E6%98%87%E8%85%BE%E8%BE%85%E5%8A%A9%E8%BD%AF%E4%BB%B6)中选择匹配的CANN等软件下载使用。
- L57: 2. 查看[软件版本配套表](https://gitcode.com/ascend/pytorch#%E6%98%87%E8%85%BE%E8%BE%85%E5%8A%A9%E8%BD%AF%E4%BB%B6)后确认对该模型有新版本PyTorch和CANN中的适配需求，请在[modelzoo/issues](https://gitcode.com/ascend/modelzoo/issues)中提出您的需求。**自行适配不保证精度和性能达标。**
- L67: | FrameworkPTAdapter | 6.0.RC3  |
- L68: |        CANN        | 8.0.RC3  |
- L69: |      昇腾NPU固件       | 24.1.RC3 |
- L70: |      昇腾NPU驱动       | 24.1.RC3 |
- L71: |        Mx_Driving         | 6.0.RC3  |
- L79: |  PyTorch  | 2.1.0  |
- L104: git clone https://github.com/open-mmlab/mmcv
- L112: git clone https://github.com/open-mmlab/mmdetection.git -b v3.3.0 mmdetection
- L117: - 安装mxDriving加速库，安装方法参考[原仓](https://gitcode.com/ascend/mxDriving)，安装后根据原仓**快速上手**章节将source环境变量命令添加在groundingdino_npu/env_npu.sh中。
- L127: ### 准备数据集
- L129: #### 微调数据集
- L130: 训练与评估所使用的数据集为refcoco数据集，需要用户在mmdetection目录下创建refcoco文件夹。
- L132: 1. 用户需自行获取COCO2014数据集，将训练集命名为train2014
- L138: 参考[链接](https://huggingface.co/GLIPModel/GLIP/tree/main/mdetr_annotations)
- L165: > 以上命令会在mdetr_annotations文件夹下生成finetune_refcoco_train_vg.json文件，如果没下载refcoco+、grefcoco等数据集的train文件则会报错，可以忽略。
- L168: ### 获取预训练权重
- L175: 2. NLTK权重
- L177: 参考[链接](https://github.com/nltk/nltk_data)
- L181: 3. MM-GDINO-B权重
- L183: 下载参考[链接](https://github.com/open-mmlab/mmdetection/tree/main/configs/mm_grounding_dino)
- L187: 将MM-GDINO-B权重放置在./weights/目录下
- L193: 主要提供基于refcoco数据集全量微调的8卡训练脚本。
- L197: 1. 修改groundingdino_npu/mm_grounding_dino_swin-b_finetune_b2_refcoco.py脚本中的数据集路径和权重路径配置，以下为实际使用示例：
- L222: 基于RefCOCO数据集训练2000步验证下游任务，具体结果见表3和表4：
- L224: **表 3**  精度结果展示表
- L226: |    芯片    |   AP   | Precision @ 1 | Precision @ 5 | Precision @ 10 |
- L228: |   GPU    | 0.3148 |    0.8554     |    0.9839     |     0.9938     |
- L229: | Atlas A2 | 0.3152 |    0.8588     |    0.9836     |     0.9941     |
- L232: **表 4**  性能结果展示表
- L236: |   GPU    | 8p | 13.13 |     2      |   fp32   |      2.1      |
- L237: | Atlas A2 | 8p | 8.45  |     2      |   fp32   |      2.1      |
- L251: 2. 修改 groundingdino_npu/inference.sh 的脚本权重路径，Python环境路径
- L266: # 修改 权重路径

## PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch

文件清单：

```text
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/1.5_requirements.txt
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/1.8_requirements.txt
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/Dockerfile
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/LICENSE
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/README.md
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/dataloaders/__init__.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/dataloaders/custom_transforms.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/dataloaders/datasets/__init__.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/dataloaders/datasets/cityscapes.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/dataloaders/datasets/coco.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/dataloaders/datasets/combine_dbs.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/dataloaders/datasets/pascal.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/dataloaders/datasets/sbd.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/dataloaders/utils.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/doc/deeplab_resnet.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/doc/deeplab_xception.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/docker_start.sh
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/infer/Dockerfile
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/infer/convert/aipp_rgb.config
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/infer/convert/atc.sh
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/infer/docker_start_infer.sh
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/infer/mxbase/CMakeLists.txt
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/infer/mxbase/DeeplabV3.cpp
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/infer/mxbase/DeeplabV3.h
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/infer/mxbase/build.sh
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/infer/mxbase/main.cpp
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/infer/mxbase/run.sh
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/infer/sdk/biz/__init__.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/infer/sdk/biz/dto.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/infer/sdk/biz/get_dataset_colormap.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/infer/sdk/biz/predictor.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/infer/sdk/etc/deeplabv3.cfg
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/infer/sdk/etc/deeplabv3.names
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/infer/sdk/etc/deeplabv3plus_opencv.pipeline.json
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/infer/sdk/main.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/infer/sdk/requirements.txt
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/infer/sdk/run.sh
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/modelarts/__init__.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/modelarts/modelarts_utils.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/modelarts/train_start.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/modelarts/train_start_drn.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/modeling/__init__.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/modeling/aspp.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/modeling/backbone/__init__.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/modeling/backbone/drn.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/modeling/backbone/mobilenet.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/modeling/backbone/resnet.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/modeling/backbone/xception.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/modeling/decoder.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/modeling/deeplab.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/modeling/sync_batchnorm/__init__.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/modeling/sync_batchnorm/batchnorm.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/modeling/sync_batchnorm/comm.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/modeling/sync_batchnorm/replicate.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/modeling/sync_batchnorm/unittest.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/modelzoo_level.txt
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/mypath.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/pth2onnx.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/public_address_statement.md
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/test/env_npu.sh
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/test/set_conda.sh
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/test/train_full_1p.sh
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/test/train_full_1p_drn.sh
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/test/train_full_8p.sh
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/test/train_full_8p_drn.sh
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/test/train_no_TQE_performance_8p.sh
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/test/train_performance_1p.sh
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/test/train_performance_8p.sh
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/train.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/utils/calculate_weights.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/utils/dropout.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/utils/loss.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/utils/lr_scheduler.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/utils/metrics.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/utils/saver.py
PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/utils/summaries.py
```

### [PyTorch/built-in/cv/semantic_segmentation/DeepLabv3+_ID1695_for_PyTorch/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/built-in/cv/semantic_segmentation/DeepLabv3%2B_ID1695_for_PyTorch/README.md)

文本SHA256：`d6f14676ff14584c8eb563076b59199392a631032ae0231bdf18064776023fe9`

- L10: DeepLabv3+是Google提出的DeepLab系列的第4代语义分割网络，此模型在DilatedFCN基础上引入了EcoderDecoder的思路，并配以空洞卷积和空间金字塔池化模块来提高分割精度，本仓库为Deeplabv3+的PyTorch实现。
- L15: url=https://github.com/jfzhang95/pytorch-deeplab-xception.git
- L22: url=https://gitcode.com/ascend/ModelZoo-PyTorch.git
- L28: 该模型为不随版本演进模型（随版本演进模型范围可在[此处](https://gitcode.com/ascend/ModelZoo-PyTorch/blob/master/PyTorch/README.CN.md)查看），未在最新昇腾配套软件中适配验证，您可以：
- L29: 1. 根据下面提供PyTorch版本在[软件版本配套表](https://gitcode.com/ascend/pytorch#%E6%98%87%E8%85%BE%E8%BE%85%E5%8A%A9%E8%BD%AF%E4%BB%B6)中选择匹配的CANN等软件下载使用。
- L30: 2. 查看[软件版本配套表](https://gitcode.com/ascend/pytorch#%E6%98%87%E8%85%BE%E8%BE%85%E5%8A%A9%E8%BD%AF%E4%BB%B6)后确认对该模型有新版本PyTorch和CANN中的适配需求，请在[modelzoo/issues](https://gitcode.com/ascend/modelzoo/issues)中提出您的需求。**自行适配不保证精度和性能达标。**
- L40: |  PyTorch 1.5  | torchvision==0.6.0；pillow==8.4.0 |
- L41: |  PyTorch 1.8  | torchvision==0.9.1；pillow==9.1.0 |
- L42: |  PyTorch 1.11 | numpy==1.21.6 |
- L46: 请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
- L59: ## 准备数据集
- L61: 1. 获取数据集。
- L63: 用户自行获取原始数据集，并将数据集上传到服务器任意路径下并解压。
- L65: 以VOC2012数据集为例，数据集目录结构参考如下所示。
- L77: > 该数据集的训练过程脚本只作为一种参考示例。
- L99: bash ./test/train_full_1p.sh --data_path=/data/xxx/  # 单卡精度
- L101: bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
- L109: bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
- L111: bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
- L114: --data_path参数填写数据集路径，需写到数据集的一级目录。
- L120: --dataset                           //数据集名称
- L126: --apex                              //是否使用混合精度
- L127: --apex-opt-level                    //混合精度类型
- L133: 训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。
- L141: | 1p-NPU |   -   | 42.65 | 1        | 3      | O2     | 1.8 |
- L142: | 8p-NPU | 0.738 | 252.83 | 8        | 100      | O2   | 1.8 |

## PyTorch/built-in/foundation/Qwen3-30B-Torchtitan

文件清单：

```text
PyTorch/built-in/foundation/Qwen3-30B-Torchtitan/README.md
PyTorch/built-in/foundation/Qwen3-30B-Torchtitan/patches/chunkloss_qwen3.patch
PyTorch/built-in/foundation/Qwen3-30B-Torchtitan/qwen3_moe_30A3.toml
PyTorch/built-in/foundation/Qwen3-30B-Torchtitan/torchtitan_npu_patch.py
```

### [PyTorch/built-in/foundation/Qwen3-30B-Torchtitan/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/built-in/foundation/Qwen3-30B-Torchtitan/README.md)

文本SHA256：`189dd9fcd78725c900c1542d21d66811a6a472a47d9b08254816ff33471c6f61`

- L9: 昇腾为基于华为昇腾处理器和软件的行业应用及服务提供全栈AI计算基础设施。您可以通过访问[昇腾社区](https://www.hiascend.com/zh/)，了解关于昇腾的更多信息。
- L15: - 硬件：昇腾服务器，已验证Atlas 800T A2（Kunpeng CPU）
- L23: | 8.5.0.B120 | 2.9.0 | 2.9.0.post1.dev20260108 | 0.2.0      | 3.4.0.dev2026010713 |
- L33: # | npu-smi 25.0.rc1.1               Version: 25.0.rc1.1                                           |
- L38: # | 0     DEVICE_NAME         | OK            | 181.4       47                0    / 0             |
- L39: # | 0                         | 0000:C1:00.0  | 49          0    / 0          44421/ 65536         |
- L53: 下载和安装详情参考[Ascend环境搭建指南](https://ascend.github.io/docs/sources/ascend/quick_install.html)或[昇腾社区](https://www.hiascend.com/developer/download/)。
- L59: 参考[torch_npu环境部署](https://gitcode.com/Ascend/pytorch?tab=md#%E7%8E%AF%E5%A2%83%E9%83%A8%E7%BD%B2)，了解昇腾版本配套关系，安装PyTorch、torch_npu。
- L71: 参考[TorchTitan Installation](https://github.com/pytorch/torchtitan/tree/v0.2.0?tab=readme-ov-file#installation)安装TorchTitan。
- L72: 或者直接下载[TorcTitan-v0.2.0源码](https://github.com/pytorch/torchtitan/tree/v0.2.0)，配置 `PYTHONPATH=torchtitan根目录`。
- L83: pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
- L85: pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/ https://mirrors.huaweicloud.com/ascend/repos/pypi"
- L98: pip install torch_npu==2.9.0.dev20251225 --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
- L104: git clone --branch v0.2.0 https://github.com/pytorch/torchtitan.git
- L139: 该补丁文件主要补充了兼容性、亲和性能优化等修改。
- L150: ### 预训练数据集下载
- L152: 可使用测试数据集 `c4_test`(无需下载)，模型配置文件中已默认指定，无需修改。如果使用C4数据集，请下载并在对应模型配置文件中修改
- L185: Qwen3-30B 模型在Atlas 800T A2单机8卡开箱训练配置会OOM，建议先测试减层模型，将 `./torchtitan/models/qwen3/__init__.py`中的 `30B-A3B`设置为 `n_layers=24`。
- L200: enable=false #目前版本尚不支持compile选项，相应配置应为false
- L217: enable=false #目前版本尚不支持compile选项，相应配置应为false
- L234: enable=false #目前版本尚不支持compile选项，相应配置应为false
- L246: 精度验证/对齐，需固定随机种子，设置确定性计算/通信

## PyTorch/built-in/mm/LLaVA

文件清单：

```text
PyTorch/built-in/mm/LLaVA/.dockerignore
PyTorch/built-in/mm/LLaVA/.editorconfig
PyTorch/built-in/mm/LLaVA/.gitattributes
PyTorch/built-in/mm/LLaVA/.gitignore
PyTorch/built-in/mm/LLaVA/LICENSE
PyTorch/built-in/mm/LLaVA/README.md
PyTorch/built-in/mm/LLaVA/README_en.md
PyTorch/built-in/mm/LLaVA/cog.yaml
PyTorch/built-in/mm/LLaVA/docs/AscendNPU_Support.md
PyTorch/built-in/mm/LLaVA/docs/Customize_Component.md
PyTorch/built-in/mm/LLaVA/docs/Data.md
PyTorch/built-in/mm/LLaVA/docs/Evaluation.md
PyTorch/built-in/mm/LLaVA/docs/Finetune_Custom_Data.md
PyTorch/built-in/mm/LLaVA/docs/Intel.md
PyTorch/built-in/mm/LLaVA/docs/LLaVA_Bench.md
PyTorch/built-in/mm/LLaVA/docs/LLaVA_from_LLaMA2.md
PyTorch/built-in/mm/LLaVA/docs/LoRA.md
PyTorch/built-in/mm/LLaVA/docs/MODEL_ZOO.md
PyTorch/built-in/mm/LLaVA/docs/ScienceQA.md
PyTorch/built-in/mm/LLaVA/docs/Windows.md
PyTorch/built-in/mm/LLaVA/docs/macOS.md
PyTorch/built-in/mm/LLaVA/llava/__init__.py
PyTorch/built-in/mm/LLaVA/llava/constants.py
PyTorch/built-in/mm/LLaVA/llava/conversation.py
PyTorch/built-in/mm/LLaVA/llava/eval/eval_gpt_review.py
PyTorch/built-in/mm/LLaVA/llava/eval/eval_gpt_review_bench.py
PyTorch/built-in/mm/LLaVA/llava/eval/eval_gpt_review_visual.py
PyTorch/built-in/mm/LLaVA/llava/eval/eval_pope.py
PyTorch/built-in/mm/LLaVA/llava/eval/eval_science_qa.py
PyTorch/built-in/mm/LLaVA/llava/eval/eval_science_qa_gpt4.py
PyTorch/built-in/mm/LLaVA/llava/eval/eval_science_qa_gpt4_requery.py
PyTorch/built-in/mm/LLaVA/llava/eval/eval_textvqa.py
PyTorch/built-in/mm/LLaVA/llava/eval/generate_webpage_data_from_table.py
PyTorch/built-in/mm/LLaVA/llava/eval/m4c_evaluator.py
PyTorch/built-in/mm/LLaVA/llava/eval/model_qa.py
PyTorch/built-in/mm/LLaVA/llava/eval/model_vqa.py
PyTorch/built-in/mm/LLaVA/llava/eval/model_vqa_loader.py
PyTorch/built-in/mm/LLaVA/llava/eval/model_vqa_mmbench.py
PyTorch/built-in/mm/LLaVA/llava/eval/model_vqa_science.py
PyTorch/built-in/mm/LLaVA/llava/eval/qa_baseline_gpt35.py
PyTorch/built-in/mm/LLaVA/llava/eval/run_llava.py
PyTorch/built-in/mm/LLaVA/llava/eval/summarize_gpt_review.py
PyTorch/built-in/mm/LLaVA/llava/eval/webpage/figures/chatgpt.svg
PyTorch/built-in/mm/LLaVA/llava/eval/webpage/figures/swords_FILL0_wght300_GRAD0_opsz48.svg
PyTorch/built-in/mm/LLaVA/llava/eval/webpage/figures/vicuna.jpeg
PyTorch/built-in/mm/LLaVA/llava/eval/webpage/index.html
PyTorch/built-in/mm/LLaVA/llava/eval/webpage/script.js
PyTorch/built-in/mm/LLaVA/llava/eval/webpage/styles.css
PyTorch/built-in/mm/LLaVA/llava/mm_utils.py
PyTorch/built-in/mm/LLaVA/llava/model/__init__.py
PyTorch/built-in/mm/LLaVA/llava/model/apply_delta.py
PyTorch/built-in/mm/LLaVA/llava/model/builder.py
PyTorch/built-in/mm/LLaVA/llava/model/consolidate.py
PyTorch/built-in/mm/LLaVA/llava/model/language_model/llava_llama.py
PyTorch/built-in/mm/LLaVA/llava/model/language_model/llava_mistral.py
PyTorch/built-in/mm/LLaVA/llava/model/language_model/llava_mpt.py
PyTorch/built-in/mm/LLaVA/llava/model/llava_arch.py
PyTorch/built-in/mm/LLaVA/llava/model/make_delta.py
PyTorch/built-in/mm/LLaVA/llava/model/multimodal_encoder/builder.py
PyTorch/built-in/mm/LLaVA/llava/model/multimodal_encoder/clip_encoder.py
PyTorch/built-in/mm/LLaVA/llava/model/multimodal_projector/builder.py
PyTorch/built-in/mm/LLaVA/llava/model/utils.py
PyTorch/built-in/mm/LLaVA/llava/serve/__init__.py
PyTorch/built-in/mm/LLaVA/llava/serve/cli.py
PyTorch/built-in/mm/LLaVA/llava/serve/controller.py
PyTorch/built-in/mm/LLaVA/llava/serve/gradio_web_server.py
PyTorch/built-in/mm/LLaVA/llava/serve/model_worker.py
PyTorch/built-in/mm/LLaVA/llava/serve/register_worker.py
PyTorch/built-in/mm/LLaVA/llava/serve/sglang_worker.py
PyTorch/built-in/mm/LLaVA/llava/serve/test_message.py
PyTorch/built-in/mm/LLaVA/llava/train/llama_flash_attn_monkey_patch.py
PyTorch/built-in/mm/LLaVA/llava/train/llama_npu_monkey_patch.py
PyTorch/built-in/mm/LLaVA/llava/train/llama_xformers_attn_monkey_patch.py
PyTorch/built-in/mm/LLaVA/llava/train/llava_trainer.py
PyTorch/built-in/mm/LLaVA/llava/train/train.py
PyTorch/built-in/mm/LLaVA/llava/train/train_mem.py
PyTorch/built-in/mm/LLaVA/llava/train/train_npu.py
PyTorch/built-in/mm/LLaVA/llava/train/train_xformers.py
PyTorch/built-in/mm/LLaVA/llava/utils.py
PyTorch/built-in/mm/LLaVA/playground/data/prompts/complex_reasoning/000_caps.txt
PyTorch/built-in/mm/LLaVA/playground/data/prompts/complex_reasoning/000_conv.txt
PyTorch/built-in/mm/LLaVA/playground/data/prompts/complex_reasoning/001_caps.txt
PyTorch/built-in/mm/LLaVA/playground/data/prompts/complex_reasoning/001_conv.txt
PyTorch/built-in/mm/LLaVA/playground/data/prompts/complex_reasoning/002_caps.txt
PyTorch/built-in/mm/LLaVA/playground/data/prompts/complex_reasoning/002_conv.txt
PyTorch/built-in/mm/LLaVA/playground/data/prompts/complex_reasoning/system_message.txt
PyTorch/built-in/mm/LLaVA/playground/data/prompts/conversation/000_caps.txt
PyTorch/built-in/mm/LLaVA/playground/data/prompts/conversation/000_conv.txt
PyTorch/built-in/mm/LLaVA/playground/data/prompts/conversation/001_caps.txt
PyTorch/built-in/mm/LLaVA/playground/data/prompts/conversation/001_conv.txt
PyTorch/built-in/mm/LLaVA/playground/data/prompts/conversation/system_message.txt
PyTorch/built-in/mm/LLaVA/playground/data/prompts/detail_description/000_caps.txt
PyTorch/built-in/mm/LLaVA/playground/data/prompts/detail_description/000_conv.txt
PyTorch/built-in/mm/LLaVA/playground/data/prompts/detail_description/001_caps.txt
PyTorch/built-in/mm/LLaVA/playground/data/prompts/detail_description/001_conv.txt
PyTorch/built-in/mm/LLaVA/playground/data/prompts/detail_description/002_caps.txt
PyTorch/built-in/mm/LLaVA/playground/data/prompts/detail_description/002_conv.txt
PyTorch/built-in/mm/LLaVA/playground/data/prompts/detail_description/system_message.txt
PyTorch/built-in/mm/LLaVA/predict.py
PyTorch/built-in/mm/LLaVA/public_address_statement.md
PyTorch/built-in/mm/LLaVA/pyproject.toml
PyTorch/built-in/mm/LLaVA/scripts/convert_gqa_for_eval.py
PyTorch/built-in/mm/LLaVA/scripts/convert_mmbench_for_submission.py
PyTorch/built-in/mm/LLaVA/scripts/convert_mmvet_for_eval.py
PyTorch/built-in/mm/LLaVA/scripts/convert_seed_for_submission.py
PyTorch/built-in/mm/LLaVA/scripts/convert_sqa_to_llava.py
PyTorch/built-in/mm/LLaVA/scripts/convert_sqa_to_llava_base_prompt.py
PyTorch/built-in/mm/LLaVA/scripts/convert_vizwiz_for_submission.py
PyTorch/built-in/mm/LLaVA/scripts/convert_vqav2_for_submission.py
PyTorch/built-in/mm/LLaVA/scripts/extract_mm_projector.py
PyTorch/built-in/mm/LLaVA/scripts/finetune.sh
PyTorch/built-in/mm/LLaVA/scripts/finetune_full_schedule.sh
PyTorch/built-in/mm/LLaVA/scripts/finetune_lora.sh
PyTorch/built-in/mm/LLaVA/scripts/finetune_qlora.sh
PyTorch/built-in/mm/LLaVA/scripts/finetune_sqa.sh
PyTorch/built-in/mm/LLaVA/scripts/merge_lora_weights.py
PyTorch/built-in/mm/LLaVA/scripts/pretrain.sh
PyTorch/built-in/mm/LLaVA/scripts/pretrain_xformers.sh
PyTorch/built-in/mm/LLaVA/scripts/sqa_eval_batch.sh
PyTorch/built-in/mm/LLaVA/scripts/sqa_eval_gather.sh
PyTorch/built-in/mm/LLaVA/scripts/upload_pypi.sh
PyTorch/built-in/mm/LLaVA/scripts/v1_5/eval/gqa.sh
PyTorch/built-in/mm/LLaVA/scripts/v1_5/eval/llavabench.sh
PyTorch/built-in/mm/LLaVA/scripts/v1_5/eval/mmbench.sh
PyTorch/built-in/mm/LLaVA/scripts/v1_5/eval/mmbench_cn.sh
PyTorch/built-in/mm/LLaVA/scripts/v1_5/eval/mme.sh
PyTorch/built-in/mm/LLaVA/scripts/v1_5/eval/mmvet.sh
PyTorch/built-in/mm/LLaVA/scripts/v1_5/eval/pope.sh
PyTorch/built-in/mm/LLaVA/scripts/v1_5/eval/qbench.sh
PyTorch/built-in/mm/LLaVA/scripts/v1_5/eval/qbench_zh.sh
PyTorch/built-in/mm/LLaVA/scripts/v1_5/eval/seed.sh
PyTorch/built-in/mm/LLaVA/scripts/v1_5/eval/sqa.sh
PyTorch/built-in/mm/LLaVA/scripts/v1_5/eval/textvqa.sh
PyTorch/built-in/mm/LLaVA/scripts/v1_5/eval/vizwiz.sh
PyTorch/built-in/mm/LLaVA/scripts/v1_5/eval/vqav2.sh
PyTorch/built-in/mm/LLaVA/scripts/v1_5/finetune.sh
PyTorch/built-in/mm/LLaVA/scripts/v1_5/finetune_lora.sh
PyTorch/built-in/mm/LLaVA/scripts/v1_5/finetune_npu.sh
PyTorch/built-in/mm/LLaVA/scripts/v1_5/finetune_task.sh
PyTorch/built-in/mm/LLaVA/scripts/v1_5/finetune_task_lora.sh
PyTorch/built-in/mm/LLaVA/scripts/v1_5/pretrain.sh
PyTorch/built-in/mm/LLaVA/scripts/zero2.json
PyTorch/built-in/mm/LLaVA/scripts/zero3.json
PyTorch/built-in/mm/LLaVA/scripts/zero3_offload.json
```

### [PyTorch/built-in/mm/LLaVA/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/built-in/mm/LLaVA/README.md)

文本SHA256：`f18a24ca3ce1f6b1b8f80bf91e8c4045d4ad63fd07ff9b4cc28c1c930f523af1`

- L4: **注意**： 本仓库中LLaVA模型将不再进行维护，请使用[MindSpeed-MM](https://gitcode.com/ascend/MindSpeed-MM/tree/master/examples/llava1.5)
- L12: - [准备数据集](#准备数据集)
- L13: - [准备预训练权重](#准备预训练权重)
- L36: | LLaVA 1.5 7B |  训练  | ✔ |
- L37: | LLaVA 1.5 7B |  推理  | ✔ |
- L38: | LLaVA 1.5 7B |  评估  | ✔ |
- L44: url=https://github.com/haotian-liu/LLaVA.git
- L50: url=https://gitcode.com/ascend/ModelZoo-PyTorch.git
- L60: git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
- L71: pip3 install torch==2.1.0+cpu  --index-url https://download.pytorch.org/whl/cpu  #For X86
- L81: 请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
- L93: ### 准备数据集
- L95: - 需要自行下载llava_v1_5_mix665k.json指令微调数据集，以及图片数据集，涉及到的数据集结构如下所示：
- L111: 需要将这五个数据集放置到同一个文件夹下，数据集来源请参考 https://github.com/haotian-liu/LLaVA/blob/main/README.md 中的数据准备章节。
- L112: ### 准备预训练权重
- L126: 1. 训练脚本位置位于scripts/v1_5/finetune_npu.sh，需要手动将数据集，权重的路径传入到相应参数上，路径仅供参考，请用户根据实际情况修改。
- L128: --model_name_or_path lmsys/vicuna-7b-v1.5  # vicuna权重路径
- L130: --image_folder ./playground/data # 图片数据集的路径，路径下包含五个数据集
- L138: bash scripts/v1_5/finetune_npu.sh # 8卡精度及性能 bf16
- L140: 训练完成后，权重文件保存在参数`--output_dir`路径下。
- L147: | GPU | 8p |       18.62        |     16     | bf16 | 2.1 |
- L148: | Atlas A2 | 8p |       20.13        |     16     | bf16 | 2.1 |
- L152: 评估脚本位于scripts/v1_5/eval下面，这里以textvqa任务为例，测试之前需要准备TextVQA_0.5.1_val.json和train_val_images.zip数据集，解压并传到textvqa.sh相关参数上面，执行textvqa.sh脚本即可进行评估。
- L164: --image-file "https://llava-vl.github.io/static/images/view.jpg" \

### [PyTorch/built-in/mm/LLaVA/README_en.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/built-in/mm/LLaVA/README_en.md)

文本SHA256：`3d20dd42bea77beef0d5906f20a0cc352bde495d7b7906c6e1b0d6364c6221a5`

- L9: [📢 [LLaVA-NeXT Blog](https://llava-vl.github.io/blog/2024-01-30-llava-next/)] [[Project Page](https://llava-vl.github.io/)] [[Demo](https://llava.hliu.cc/)]  [[Data](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md)] [[Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md)]
- L11: 🤝Community Contributions: [[llama.cpp](https://github.com/ggerganov/llama.cpp/pull/3436)] [[Colab](https://github.com/camenduru/LLaVA-colab)] [[🤗Space](https://huggingface.co/spaces/badayvedat/LLaVA)] [[Replicate](https://replicate.com/yorickvp/llava-13b)] [AutoGen] [[BakLLaVA](https://github.com/SkunkworksAI/BakLLaVA)]
- L13: **Improved Baselines with Visual Instruction Tuning** [[Paper](https://arxiv.org/abs/2310.03744)] [[HF](https://huggingface.co/papers/2310.03744)] <br>
- L14: [Haotian Liu](https://hliu.cc), [Chunyuan Li](https://chunyuan.li/), [Yuheng Li](https://yuheng-li.github.io/), [Yong Jae Lee](https://pages.cs.wisc.edu/~yongjaelee/)
- L16: **Visual Instruction Tuning** (NeurIPS 2023, **Oral**) [[Paper](https://arxiv.org/abs/2304.08485)] [[HF](https://huggingface.co/papers/2304.08485)] <br>
- L17: [Haotian Liu*](https://hliu.cc), [Chunyuan Li*](https://chunyuan.li/), Qingyang Wu, [Yong Jae Lee](https://pages.cs.wisc.edu/~yongjaelee/) (*Equal Contribution)
- L20: <a href="https://llava.hliu.cc/"><img src="images/llava_logo.png" width="50%"></a> <br>
- L21: Generated by <a href="https://gligen.github.io/">GLIGEN</a> via "a cute lava llama with glasses" and box prompt
- L26: - [03/10] Releasing **LMMs-Eval**, a highly efficient evaluation pipeline we used when developing LLaVA-NeXT. It supports the evaluation of LMMs on dozens of public datasets and allows new dataset onboarding, making the dev of new LMMs much faster. [Blog] [[Codebase](https://github.com/EvolvingLMMs-Lab/lmms-eval)]
- L27: - [1/30] 🔥 LLaVA-NeXT (LLaVA-1.6) is out! With additional scaling to LLaVA-1.5, LLaVA-NeXT-34B outperforms Gemini Pro on some benchmarks. It can now process 4x more pixels and perform more tasks/applications than before. Check out the [blog post](https://llava-vl.github.io/blog/2024-01-30-llava-next/), and explore the [demo](https://llava.hliu.cc/)! Models are available in [Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md). Training/eval data and scripts coming soon.
- L28: - [11/10] [LLaVA-Plus](https://llava-vl.github.io/llava-plus/) is released: Learning to Use Tools for Creating Multimodal Agents, with LLaVA-Plus (LLaVA that Plug and Learn to Use Skills). [[Project Page](https://llava-vl.github.io/llava-plus/)] [Demo] [[Code](https://github.com/LLaVA-VL/LLaVA-Plus-Codebase)] [[Paper](https://arxiv.org/abs/2311.05437)]
- L29: - [11/2] [LLaVA-Interactive](https://llava-vl.github.io/llava-interactive/) is released: Experience the future of human-AI multimodal interaction with an all-in-one demo for Image Chat, Segmentation, Generation and Editing. [[Project Page](https://llava-vl.github.io/llava-interactive/)] [Demo] [[Code](https://github.com/LLaVA-VL/LLaVA-Interactive-Demo)] [[Paper](https://arxiv.org/abs/2311.00571)]
- L30: - [10/26] 🔥 LLaVA-1.5 with LoRA achieves comparable performance as full-model finetuning, with a reduced GPU RAM requirement ([ckpts](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md#llava-v15), [script](https://github.com/haotian-liu/LLaVA#train)). We also provide a [doc](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md) on how to finetune LLaVA-1.5 on your own dataset with LoRA.
- L31: - [10/12] Check out the Korean LLaVA (Ko-LLaVA), created by ETRI, who has generously supported our research! [[🤗 Demo](https://huggingface.co/spaces/etri-vilab/Ko-LLaVA)]
- L32: - [10/5] 🔥 LLaVA-1.5 is out! Achieving SoTA on 11 benchmarks, with just simple modifications to the original LLaVA, utilizes all public data, completes training in ~1 day on a single 8-A100 node, and surpasses methods like Qwen-VL-Chat that use billion-scale data. Check out the [technical report](https://arxiv.org/abs/2310.03744), and explore the [demo](https://llava.hliu.cc/)! Models are available in [Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md). The training data and scripts of LLaVA-1.5 are released [here](https://github.com/haotian-liu/LLaVA#train), and evaluation scripts are released [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md)!
- L33: - [9/26] LLaVA is improved with reinforcement learning from human feedback (RLHF) to improve fact grounding and reduce hallucination. Check out the new SFT and RLHF checkpoints at project [[LLavA-RLHF]](https://llava-rlhf.github.io/)
- L34: - [9/22] [LLaVA](https://arxiv.org/abs/2304.08485) is accepted by NeurIPS 2023 as **oral presentation**, and [LLaVA-Med](https://arxiv.org/abs/2306.00890) is accepted by NeurIPS 2023 Datasets and Benchmarks Track as **spotlight presentation**.
- L39: - [11/6] Support **Intel** dGPU and CPU platforms. [More details here.](https://github.com/haotian-liu/LLaVA/tree/intel/docs/intel)
- L40: - [10/12] LLaVA is now supported in [llama.cpp](https://github.com/ggerganov/llama.cpp/pull/3436) with 4-bit / 5-bit quantization support!
- L41: - [10/11] The training data and scripts of LLaVA-1.5 are released [here](https://github.com/haotian-liu/LLaVA#train), and evaluation scripts are released [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md)!
- L42: - [10/10] [Roboflow Deep Dive](https://blog.roboflow.com/first-impressions-with-llava-1-5/): First Impressions with LLaVA-1.5.
- L43: - [9/20] We summarize our empirical study of training 33B and 65B LLaVA models in a [note](https://arxiv.org/abs/2309.09958). Further, if you are interested in the comprehensive review, evolution and trend of multimodal foundation models, please check out our recent survey paper [``Multimodal Foundation Models: From Specialists to General-Purpose Assistants''.](https://arxiv.org/abs/2309.10020)
- L45: <img src="https://github.com/Computer-Vision-in-the-Wild/CVinW_Readings/blob/main/images/mfm_evolution.jpeg?raw=true" width=50%/>
- L48: - [7/19] 🔥 We release a major upgrade, including support for LLaMA-2, LoRA training, 4-/8-bit inference, higher resolution (336x336), and a lot more. We release [LLaVA Bench](https://github.com/haotian-liu/LLaVA/blob/main/docs/LLaVA_Bench.md) for benchmarking open-ended visual chat with results from Bard and Bing-Chat. We also support and verify training with RTX 3090 and RTX A6000. Check out [LLaVA-from-LLaMA-2](https://github.com/haotian-liu/LLaVA/blob/main/docs/LLaVA_from_LLaMA2.md), and our [model zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md)!
- L49: - [6/26] [CVPR 2023 Tutorial](https://vlp-tutorial.github.io/) on **Large Multimodal Models: Towards Building and Surpassing Multimodal GPT-4**!  Please check out [[Slides](https://datarelease.blob.core.windows.net/tutorial/vision_foundation_models_2023/slides/Chunyuan_cvpr2023_tutorial_lmm.pdf)] [[Notes](https://arxiv.org/abs/2306.14895)] [[YouTube](https://youtu.be/mkI7EPD1vp8)] [[Bilibli](https://www.bilibili.com/video/BV1Ng4y1T7v3/)].
- L51: - [6/1] We released **LLaVA-Med: Large Language and Vision Assistant for Biomedicine**, a step towards building biomedical domain large language and vision models with GPT-4 level capabilities.  Checkout the [paper](https://arxiv.org/abs/2306.00890) and [page](https://github.com/microsoft/LLaVA-Med).
- L52: - [5/6] We are releasing [LLaVA-Lighting-MPT-7B-preview](https://huggingface.co/liuhaotian/LLaVA-Lightning-MPT-7B-preview), based on MPT-7B-Chat! See here for more details.
- L55: - [4/17] 🔥 We released **LLaVA: Large Language and Vision Assistant**. We propose visual instruction tuning, towards building large language and vision models with GPT-4 level capabilities.  Checkout the [paper](https://arxiv.org/abs/2304.08485) and [demo](https://llava.hliu.cc/).
- L59: <!-- <a href="https://llava.hliu.cc/"><img src="assets/demo.gif" width="70%"></a> -->
- L61: [![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
- L62: **Usage and License Notices**: This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses, including but not limited to the [OpenAI Terms of Use](https://openai.com/policies/terms-of-use) for the dataset and the specific licenses for base language models for checkpoints trained using the dataset (e.g. [Llama community license](https://ai.meta.com/llama/license/) for LLaMA-2 and Vicuna-v1.5). This project does not impose any additional constraints beyond those stipulated in the original licenses. Furthermore, users are reminded to ensure that their use of the dataset and checkpoints is in compliance with all applicable laws and regulations.
- L68: - [Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md)
- L69: - [Dataset](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md)
- L75: If you are not using Linux, do *NOT* proceed, see instructions for [macOS](https://github.com/haotian-liu/LLaVA/blob/main/docs/macOS.md) and [Windows](https://github.com/haotian-liu/LLaVA/blob/main/docs/Windows.md).
- L81: git clone https://github.com/haotian-liu/LLaVA.git
- L136: image_file = "https://llava-vl.github.io/static/images/view.jpg"
- L157: Please check out our [Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md) for all public LLaVA checkpoints, and the instructions of how to use the weights.
- L172: sglw13b("SGLang Backend:<br/>llava-v1.6-34b<br/>http://localhost:30000")
- L203: python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload
- L209: This is the recommended way to serve LLaVA model with high throughput, and you need to install SGLang first. Note that currently `4-bit` quantization is not supported yet on SGLang-LLaVA, and if you have limited GPU VRAM, please check out model worker with [quantization](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#launch-a-model-worker-4-bit-8-bit-inference-quantized).
- L227: You'll then launch a LLaVA-SGLang worker that will communicate between LLaVA controller and SGLang backend to route the requests. Set `--sgl-endpoint` to `http://127.0.0.1:port` where `port` is the one you just set (default: 30000).
- L230: python -m llava.serve.sglang_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --sgl-endpoint http://127.0.0.1:30000
- L238: python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path liuhaotian/llava-v1.5-13b
- L244: python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port <different from 40000, say 40001> --worker http://localhost:<change accordingly, i.e. 40001> --model-path <ckpt2>
- L254: CUDA_VISIBLE_DEVICES=0,1 python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path liuhaotian/llava-v1.5-13b
- L262: python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path liuhaotian/llava-v1.5-13b --load-4bit
- L269: To load unmerged LoRA weights, you simply need to pass an additional argument `--model-base`, which is the base LLM that is used to train the LoRA weights. You can check the base LLM of each LoRA weights in the [model zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md).
- L272: python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path liuhaotian/llava-v1-0719-336px-lora-vicuna-13b-v1.3 --model-base lmsys/vicuna-13b-v1.3
- L282: --image-file "https://llava-vl.github.io/static/images/view.jpg" \
- L290: *Below is the latest training configuration for LLaVA v1.5. For legacy models, please refer to README of [this](https://github.com/haotian-liu/LLaVA/tree/v1.0.1) version for now. We'll add them in a separate doc later.*
- L303: | LLaVA-v1.5-13B | 256 | 1e-3 | 1 | 2048 | 0 |
- L309: | LLaVA-v1.5-13B | 128 | 2e-5 | 1 | 2048 | 0 |
- L317: Please download the 558K subset of the LAION-CC-SBU dataset with BLIP captions we use in the paper [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain).
- L321: Training script with DeepSpeed ZeRO-2: [`pretrain.sh`](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/pretrain.sh).
- L329: We provide training script with DeepSpeed [here](https://github.com/haotian-liu/LLaVA/blob/main/scripts/pretrain_xformers.sh).
- L331: - If you are using V100 which is not supported by FlashAttention, you can use the [memory-efficient attention](https://arxiv.org/abs/2112.05682) implemented in [xFormers](https://github.com/facebookresearch/xformers). Install xformers and replace `llava/train/train_mem.py` above with [llava/train/train_xformers.py](llava/train/train_xformers.py).
- L338: Please download the annotation of the final mixture our instruction tuning data [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), and download the images from constituting datasets:
- L340: - COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- L341: - GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- L342: - OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), **we save all files as `.jpg`**
- L343: - TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- L344: - VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)
- L364: You may download our pretrained projectors in [Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md). It is not recommended to use legacy projectors, as they may be trained with a different version of the codebase, and if any option is off, the model will not function/train as we expected.
- L368: Training script with DeepSpeed ZeRO-3: [`finetune.sh`](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/finetune.sh).
- L372: - Use LoRA: [`finetune_lora.sh`](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/finetune_lora.sh). We are able to fit 13B training in 8-A100-40G/8-A6000, and 7B training in 8-RTX3090. Make sure `per_device_train_batch_size*gradient_accumulation_steps` is the same as the provided script for best reproducibility.
- L373: - Replace `zero3.json` with `zero3_offload.json` which offloads some parameters to CPU RAM. This slows down the training speed.
- L375: If you are interested in finetuning LLaVA model to your own task/data, please check out [`Finetune_Custom_Data.md`](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md)。
- L388: See [Evaluation.md](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).
- L432: url={https://llava-vl.github.io/blog/2024-01-30-llava-next/},
- L455: - [Vicuna](https://github.com/lm-sys/FastChat): the codebase we built upon, and our base model Vicuna-13B that has the amazing language capabilities!
- L459: - [Instruction Tuning with GPT-4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
- L460: - [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day](https://github.com/microsoft/LLaVA-Med)
- L461: - [Otter: In-Context Multi-Modal Instruction Tuning](https://github.com/Luodian/Otter)
- L464: - [SEEM: Segment Everything Everywhere All at Once](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once)
- L465: - [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) to detect, segment, and generate anything by marrying [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) and [Segment-Anything](https://github.com/facebookresearch/segment-anything).

## PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch

文件清单：

```text
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/1.5_requirements.txt
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/1.8_requirements.txt
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/LICENSE
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/1p/conf/backup.conf
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/1p/conf/ctc_config.yaml
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/1p/conf/dev_spk.list
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/1p/conf/fbank.conf
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/1p/conf/mfcc.conf
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/1p/conf/phones.60-48-39.map
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/1p/conf/test_spk.list
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/1p/conf/train_spk.list
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/1p/docker_start.sh
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/1p/local/make_spectrum.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/1p/local/normalize_phone.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/1p/local/timit_data_prep.sh
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/1p/models/model_ctc.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/1p/path.sh
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/1p/steps/get_model_units.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/1p/steps/make_feat.sh
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/1p/steps/test_ctc.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/1p/steps/train_ctc.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/1p/steps/train_lm.sh
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/1p/steps/visualize.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/1p/utils/BeamSearch.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/1p/utils/NgramLM.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/1p/utils/ctcDecoder.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/1p/utils/data_loader.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/1p/utils/tools.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/8p/conf/backup.conf
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/8p/conf/ctc_config.yaml
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/8p/conf/dev_spk.list
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/8p/conf/fbank.conf
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/8p/conf/mfcc.conf
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/8p/conf/phones.60-48-39.map
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/8p/conf/test_spk.list
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/8p/conf/train_spk.list
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/8p/docker_start.sh
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/8p/local/make_spectrum.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/8p/local/normalize_phone.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/8p/local/timit_data_prep.sh
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/8p/models/model_ctc.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/8p/path.sh
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/8p/steps/get_model_units.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/8p/steps/make_feat.sh
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/8p/steps/test_ctc.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/8p/steps/train_ctc.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/8p/steps/train_lm.sh
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/8p/steps/visualize.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/8p/utils/BeamSearch.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/8p/utils/NgramLM.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/8p/utils/ctcDecoder.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/8p/utils/data_loader.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/8p/utils/tools.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/Dockerfile
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/README.en.md
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/README.md
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/install_kaldi.sh
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/modelzoo_level.txt
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/public_address_statement.md
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/test/env_npu.sh
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/test/train_full_1p.sh
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/test/train_full_8p.sh
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/test/train_performance_16p.sh
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/test/train_performance_1p.sh
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/test/train_performance_8p.sh
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/timit/conf/backup.conf
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/timit/conf/ctc_config.yaml
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/timit/conf/dev_spk.list
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/timit/conf/fbank.conf
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/timit/conf/mfcc.conf
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/timit/conf/phones.60-48-39.map
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/timit/conf/test_spk.list
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/timit/conf/train_spk.list
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/timit/local/make_spectrum.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/timit/local/normalize_phone.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/timit/local/timit_data_prep.sh
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/timit/models/model_ctc.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/timit/path.sh
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/timit/run.sh
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/timit/steps/get_model_units.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/timit/steps/make_feat.sh
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/timit/steps/test_ctc.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/timit/steps/train_ctc.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/timit/steps/train_lm.sh
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/timit/steps/visualize.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/timit/steps/visualize.py.old
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/timit/utils/.keep
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/timit/utils/BeamSearch.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/timit/utils/NgramLM.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/timit/utils/ctcDecoder.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/timit/utils/data_loader.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/timit/utils/tools.py
PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/url.ini
```

### [PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/README.md)

文本SHA256：`6380b6a377a8bb860f43903d3efd8d9f9819bfdcc31dccc05dc3382f3924146c`

- L18: url=https://github.com/Diamondfan/CTC_pytorch/tree/master/timit
- L25: url=https://gitcode.com/ascend/ModelZoo-PyTorch.git
- L32: 该模型为不随版本演进模型（随版本演进模型范围可在[此处](https://gitcode.com/ascend/ModelZoo-PyTorch/blob/master/PyTorch/README.CN.md)查看），未在最新昇腾配套软件中适配验证，您可以：
- L33: 1. 根据下面提供PyTorch版本在[软件版本配套表](https://gitcode.com/ascend/pytorch#%E6%98%87%E8%85%BE%E8%BE%85%E5%8A%A9%E8%BD%AF%E4%BB%B6)中选择匹配的CANN等软件下载使用。
- L34: 2. 查看[软件版本配套表](https://gitcode.com/ascend/pytorch#%E6%98%87%E8%85%BE%E8%BE%85%E5%8A%A9%E8%BD%AF%E4%BB%B6)后确认对该模型有新版本PyTorch和CANN中的适配需求，请在[modelzoo/issues](https://gitcode.com/ascend/modelzoo/issues)中提出您的需求。**自行适配不保证精度和性能达标。**
- L44: | PyTorch 1.5 | torchvision==0.2.2.post3 |
- L45: | PyTorch 1.8 | torchvision==0.9.1 |
- L49: 请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
- L63: - 安装kaldi(可选，首次处理TIMIT原始数据集需安装)。
- L72: ## 准备数据集
- L74: 1. 请用户自行获取TIMIT语音数据集并放置服务器的任意目录下。
- L75: 2. 数据集目录结构如下所示。
- L84: > 该数据集的训练过程脚本只作为一种参考示例。
- L105: bash ./test/train_full_1p.sh --data_path=/data/xxx/  # 单卡精度
- L107: bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
- L115: bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
- L117: bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
- L120: --data_path参数填写数据集路径，需写到数据集的一级目录。
- L126: --amp                               //是否使用混合精度
- L133: --opt_level                         //混合精度类型
- L138: weight_decay                        //权重衰减，默认：0.0001
- L141: 训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。
- L147: | Type | Acc@1 | FPS       | Epochs   | AMP_Type | Torch_Version |
- L149: | NPU-1p | 80.5 | 15.225 | 1      | O2    |       |
- L150: | NPU-8p | - | - | 30    | O2  |     |

## PyTorch/built-in/rl/MAPPO_for_PyTorch

文件清单：

```text
PyTorch/built-in/rl/MAPPO_for_PyTorch/.gitignore
PyTorch/built-in/rl/MAPPO_for_PyTorch/LICENSE
PyTorch/built-in/rl/MAPPO_for_PyTorch/README.md
PyTorch/built-in/rl/MAPPO_for_PyTorch/environment.yaml
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/__init__.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/algorithms/__init__.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/algorithms/r_mappo/__init__.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/algorithms/r_mappo/algorithm/rMAPPOPolicy.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/algorithms/r_mappo/algorithm/r_actor_critic.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/algorithms/r_mappo/r_mappo.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/algorithms/utils/act.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/algorithms/utils/cnn.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/algorithms/utils/distributions.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/algorithms/utils/mlp.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/algorithms/utils/popart.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/algorithms/utils/rnn.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/algorithms/utils/util.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/config.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/__init__.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/env_wrappers.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/football/Football_Env.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/hanabi/CMakeLists.txt
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/hanabi/Hanabi_Env.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/hanabi/__init__.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/hanabi/clean_all.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/hanabi/hanabi_lib/CMakeLists.txt
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/hanabi/hanabi_lib/canonical_encoders.cc
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/hanabi/hanabi_lib/canonical_encoders.h
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/hanabi/hanabi_lib/hanabi_card.cc
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/hanabi/hanabi_lib/hanabi_card.h
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/hanabi/hanabi_lib/hanabi_game.cc
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/hanabi/hanabi_lib/hanabi_game.h
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/hanabi/hanabi_lib/hanabi_hand.cc
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/hanabi/hanabi_lib/hanabi_hand.h
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/hanabi/hanabi_lib/hanabi_history_item.cc
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/hanabi/hanabi_lib/hanabi_history_item.h
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/hanabi/hanabi_lib/hanabi_move.cc
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/hanabi/hanabi_lib/hanabi_move.h
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/hanabi/hanabi_lib/hanabi_observation.cc
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/hanabi/hanabi_lib/hanabi_observation.h
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/hanabi/hanabi_lib/hanabi_state.cc
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/hanabi/hanabi_lib/hanabi_state.h
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/hanabi/hanabi_lib/observation_encoder.h
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/hanabi/hanabi_lib/util.cc
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/hanabi/hanabi_lib/util.h
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/hanabi/pyhanabi.cc
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/hanabi/pyhanabi.h
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/hanabi/pyhanabi.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/mpe/MPE_env.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/mpe/__init__.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/mpe/core.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/mpe/environment.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/mpe/multi_discrete.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/mpe/rendering.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/mpe/scenario.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/mpe/scenarios/__init__.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/mpe/scenarios/simple_adversary.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/mpe/scenarios/simple_attack.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/mpe/scenarios/simple_crypto.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/mpe/scenarios/simple_crypto_display.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/mpe/scenarios/simple_push.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/mpe/scenarios/simple_reference.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/mpe/scenarios/simple_speaker_listener.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/mpe/scenarios/simple_spread.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/mpe/scenarios/simple_tag.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/mpe/scenarios/simple_world_comm.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/starcraft2/SMACv2.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/starcraft2/SMACv2_modified.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/starcraft2/StarCraft2_Env.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/starcraft2/StarCraft2v2/distributions.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/starcraft2/StarCraft2v2/starcraft2.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/starcraft2/StarCraft2v2/wrapper.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/starcraft2/multiagentenv.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/envs/starcraft2/smac_maps.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/runner/separated/base_runner.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/runner/separated/mpe_runner.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/runner/shared/base_runner.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/runner/shared/football_runner.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/runner/shared/hanabi_runner_forward.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/runner/shared/mpe_runner.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/runner/shared/smac_runner.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/__init__.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/eval/eval_hanabi.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/eval_hanabi_forward.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/render/render_football.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/render/render_mpe.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/render_football.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/render_mpe.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train/__init__.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train/train_football.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train/train_hanabi_forward.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train/train_mpe.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train/train_smac.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_football_scripts/train_football_3v1.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_football_scripts/train_football_ca_easy.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_football_scripts/train_football_ca_hard.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_football_scripts/train_football_corner.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_football_scripts/train_football_rp.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_football_scripts/train_football_rps.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_hanabi_forward.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_mpe_scripts/train_mpe_comm.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_mpe_scripts/train_mpe_comm_performance.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_mpe_scripts/train_mpe_reference.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_mpe_scripts/train_mpe_spread.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smac_scripts/train_smac_10m_vs_11m.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smac_scripts/train_smac_1c3s5z.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smac_scripts/train_smac_25m.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smac_scripts/train_smac_27m_vs_30m.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smac_scripts/train_smac_2c_vs_64zg.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smac_scripts/train_smac_2m_vs_1z.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smac_scripts/train_smac_2s_vs_1sc.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smac_scripts/train_smac_3m.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smac_scripts/train_smac_3s5z.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smac_scripts/train_smac_3s5z_vs_3s6z.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smac_scripts/train_smac_3s_vs_3z.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smac_scripts/train_smac_3s_vs_4z.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smac_scripts/train_smac_3s_vs_5z.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smac_scripts/train_smac_5m_vs_6m.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smac_scripts/train_smac_6h_vs_8z.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smac_scripts/train_smac_8m.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smac_scripts/train_smac_8m_vs_9m.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smac_scripts/train_smac_MMM.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smac_scripts/train_smac_MMM2.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smac_scripts/train_smac_bane_vs_bane.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smac_scripts/train_smac_baneling.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smac_scripts/train_smac_corridor.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smacv2_scripts/train_protoss_10v10.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smacv2_scripts/train_protoss_10v11.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smacv2_scripts/train_protoss_20v20.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smacv2_scripts/train_protoss_20v23.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smacv2_scripts/train_protoss_5v5.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smacv2_scripts/train_terran_10v10.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smacv2_scripts/train_terran_10v11.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smacv2_scripts/train_terran_20v20.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smacv2_scripts/train_terran_20v23.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smacv2_scripts/train_terran_5v5.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smacv2_scripts/train_zerg_10v10.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smacv2_scripts/train_zerg_10v11.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smacv2_scripts/train_zerg_20v20.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smacv2_scripts/train_zerg_20v23.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/scripts/train_smacv2_scripts/train_zerg_5v5.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/utils/__init__.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/utils/multi_discrete.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/utils/separated_buffer.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/utils/shared_buffer.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/utils/util.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/onpolicy/utils/valuenorm.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/public_address_statement.md
PyTorch/built-in/rl/MAPPO_for_PyTorch/requirements.txt
PyTorch/built-in/rl/MAPPO_for_PyTorch/setup.py
PyTorch/built-in/rl/MAPPO_for_PyTorch/test/env_npu.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/test/train_full_1p.sh
PyTorch/built-in/rl/MAPPO_for_PyTorch/test/train_performance_1p.sh
```

### [PyTorch/built-in/rl/MAPPO_for_PyTorch/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/built-in/rl/MAPPO_for_PyTorch/README.md)

文本SHA256：`02847cc6afeb85ce4c1eb5ffc7111379bb2aca5d495ce0c33c9222d96032b1e7`

- L15: 多智能体近端策略优化算法（Multi-Agent Proximal Policy Optimization， MAPPO）是一种新型的Policy Gradient算法。基于现有的近端策略优化算法（Proximal Policy Optimization， PPO），在不修改算法架构的基础上，通过调整超参数，在多智能体环境中达到与大多数off-policy算法相当的性能。
- L21: url=https://github.com/marlbenchmark/on-policy
- L28: url=https://gitcode.com/ascend/ModelZoo-PyTorch.git
- L34: 该模型为不随版本演进模型（随版本演进模型范围可在[此处](https://gitcode.com/ascend/ModelZoo-PyTorch/blob/master/PyTorch/README.CN.md)查看），未在最新昇腾配套软件中适配验证，您可以：
- L35: 1. 根据下面提供PyTorch版本在[软件版本配套表](https://gitcode.com/ascend/pytorch#%E6%98%87%E8%85%BE%E8%BE%85%E5%8A%A9%E8%BD%AF%E4%BB%B6)中选择匹配的CANN等软件下载使用。
- L36: 2. 查看[软件版本配套表](https://gitcode.com/ascend/pytorch#%E6%98%87%E8%85%BE%E8%BE%85%E5%8A%A9%E8%BD%AF%E4%BB%B6)后确认对该模型有新版本PyTorch和CANN中的适配需求，请在[modelzoo/issues](https://gitcode.com/ascend/modelzoo/issues)中提出您的需求。**自行适配不保证精度和性能达标。**
- L53: <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0005.html">驱动固件安装指南</a> 》</td>
- L62: <td>《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html">CANN 软件安装指南</a> 》</td>
- L67: <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/Pytorch/700/configandinstg/instg/insg_0004.html">Ascend Extension for PyTorch 配置与安装</a> 》</td>
- L81: |  PyTorch 2.1  |  absl-py==1.4.0; gym==0.17.2; protobuf==3.20.0; wandb==0.10.5  |
- L92: ## 准备数据集
- L123: - 单机单卡性能
- L126: bash test/train_performance_1p.sh  # 单卡性能
- L129: 训练完成后，权重文件保存在`onpolicy/scripts/results`路径下，并输出模型训练精度和性能信息。
- L137: | 1p-竞品V   | 1789 | 2000000                | -15.9          |
- L138: | 1p-NPU    | 885  | 2000000                | -15.9          |
- L140: 说明：上表为历史数据，仅供参考。2025年5月10日更新的性能数据如下：
- L141: | NAME | 精度类型 | FPS |
- L143: | 1p-竞品 | FP16 | 1789 |
- L144: | 1p-Atlas 900 A2 PoDc | FP16 | 1791.33 |
- L157: 1.暂不支持Atlas 350加速卡。

## PyTorch/built-in/rl/PPO_for_Pytorch

文件清单：

```text
PyTorch/built-in/rl/PPO_for_Pytorch/.gitattributes
PyTorch/built-in/rl/PPO_for_Pytorch/LICENSE
PyTorch/built-in/rl/PPO_for_Pytorch/PPO.py
PyTorch/built-in/rl/PPO_for_Pytorch/PPO_preTrained/README.md
PyTorch/built-in/rl/PPO_for_Pytorch/README.md
PyTorch/built-in/rl/PPO_for_Pytorch/make_gif.py
PyTorch/built-in/rl/PPO_for_Pytorch/plot_graph.py
PyTorch/built-in/rl/PPO_for_Pytorch/requirements.txt
PyTorch/built-in/rl/PPO_for_Pytorch/test.py
PyTorch/built-in/rl/PPO_for_Pytorch/test/env_npu.sh
PyTorch/built-in/rl/PPO_for_Pytorch/test/train_full_1p.sh
PyTorch/built-in/rl/PPO_for_Pytorch/test/train_performance_1p.sh
PyTorch/built-in/rl/PPO_for_Pytorch/train.py
```

### [PyTorch/built-in/rl/PPO_for_Pytorch/PPO_preTrained/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/built-in/rl/PPO_for_Pytorch/PPO_preTrained/README.md)

文本SHA256：`bfc4588d501b6e4e0251a919f77b8e1012b933e945e502bad1e2db2f2c2465f2`


### [PyTorch/built-in/rl/PPO_for_Pytorch/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/built-in/rl/PPO_for_Pytorch/README.md)

文本SHA256：`846d53745a923adc2039902fcbe203ea9172dbfd0f7d81eb494e9f5143b96496`

- L21: url=https://github.com/nikhilbarhate99/PPO-PyTorch
- L28: url=https://gitcode.com/ascend/ModelZoo-PyTorch.git
- L34: 该模型为不随版本演进模型（随版本演进模型范围可在[此处](https://gitcode.com/ascend/ModelZoo-PyTorch/blob/master/PyTorch/README.CN.md)查看），未在最新昇腾配套软件中适配验证，您可以：
- L35: 1. 根据下面提供PyTorch版本在[软件版本配套表](https://gitcode.com/ascend/pytorch#%E6%98%87%E8%85%BE%E8%BE%85%E5%8A%A9%E8%BD%AF%E4%BB%B6)中选择匹配的CANN等软件下载使用。
- L36: 2. 查看[软件版本配套表](https://gitcode.com/ascend/pytorch#%E6%98%87%E8%85%BE%E8%BE%85%E5%8A%A9%E8%BD%AF%E4%BB%B6)后确认对该模型有新版本PyTorch和CANN中的适配需求，请在[modelzoo/issues](https://gitcode.com/ascend/modelzoo/issues)中提出您的需求。**自行适配不保证精度和性能达标。**
- L53: <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0005.html">驱动固件安装指南</a> 》</td>
- L62: <td>《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html">CANN 软件安装指南</a> 》</td>
- L67: <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/Pytorch/700/configandinstg/instg/insg_0004.html">Ascend Extension for PyTorch 配置与安装</a> 》</td>
- L81: |  PyTorch 2.1  | Box2D==2.3.2 Box2D-kengz==2.3.3 gym==0.15.4 |
- L92: ## 准备数据集
- L123: - 单机单卡性能
- L126: bash test/train_performance_1p.sh  # 单卡性能
- L129: 训练完成后，权重文件保存在`test/output`路径下，并输出模型训练精度和性能信息。
- L137: | 1p-竞品V       | 585.37 | 3000000                 | 197.75         |
- L138: | 1p-NPU-Atlas 800T A2 | 284.02 | 3000000                | 240        |
- L140: 说明：上表为历史数据，仅供参考。2025年5月10日更新的性能数据如下：
- L141: | NAME | 精度类型 | FPS |
- L143: | 1p-竞品 | FP16 | 585.37 |
- L144: | 1p-Atlas 900 A2 PoDc | FP16 | 413.79 |
- L145: | 1p-Atlas 800T A2 | FP16 | 336.84 |

## PyTorch/built-in/rl/VeRL_for_PyTorch

文件清单：

```text
PyTorch/built-in/rl/VeRL_for_PyTorch/.github/dependabot.yml
PyTorch/built-in/rl/VeRL_for_PyTorch/.github/workflows/checkpoints.yml
PyTorch/built-in/rl/VeRL_for_PyTorch/.github/workflows/dataset.yml
PyTorch/built-in/rl/VeRL_for_PyTorch/.github/workflows/e2e_ascend.yml
PyTorch/built-in/rl/VeRL_for_PyTorch/.github/workflows/e2e_eval_aime24.yml
PyTorch/built-in/rl/VeRL_for_PyTorch/.github/workflows/e2e_grpo.yml
PyTorch/built-in/rl/VeRL_for_PyTorch/.github/workflows/e2e_gsm8k.yml
PyTorch/built-in/rl/VeRL_for_PyTorch/.github/workflows/e2e_gsm8k_dapo.yml
PyTorch/built-in/rl/VeRL_for_PyTorch/.github/workflows/e2e_gsm8k_megatron.yml
PyTorch/built-in/rl/VeRL_for_PyTorch/.github/workflows/e2e_gsm8k_prime.yml
PyTorch/built-in/rl/VeRL_for_PyTorch/.github/workflows/e2e_lora.yml
PyTorch/built-in/rl/VeRL_for_PyTorch/.github/workflows/e2e_sft.yml
PyTorch/built-in/rl/VeRL_for_PyTorch/.github/workflows/e2e_sglang_gsm8k.yml
PyTorch/built-in/rl/VeRL_for_PyTorch/.github/workflows/e2e_vlm_geo3k.yml
PyTorch/built-in/rl/VeRL_for_PyTorch/.github/workflows/model.yml
PyTorch/built-in/rl/VeRL_for_PyTorch/.github/workflows/pylint.yml
PyTorch/built-in/rl/VeRL_for_PyTorch/.github/workflows/ray_test.yml
PyTorch/built-in/rl/VeRL_for_PyTorch/.github/workflows/sandbox.yml
PyTorch/built-in/rl/VeRL_for_PyTorch/.github/workflows/sanity.yml
PyTorch/built-in/rl/VeRL_for_PyTorch/.github/workflows/scorecard.yml
PyTorch/built-in/rl/VeRL_for_PyTorch/.github/workflows/secrets_scan.yml
PyTorch/built-in/rl/VeRL_for_PyTorch/.github/workflows/vllm.yml
PyTorch/built-in/rl/VeRL_for_PyTorch/.github/workflows/yapf_format.yml
PyTorch/built-in/rl/VeRL_for_PyTorch/.gitignore
PyTorch/built-in/rl/VeRL_for_PyTorch/.readthedocs.yaml
PyTorch/built-in/rl/VeRL_for_PyTorch/.style.yapf
PyTorch/built-in/rl/VeRL_for_PyTorch/LICENSE
PyTorch/built-in/rl/VeRL_for_PyTorch/Notice.txt
PyTorch/built-in/rl/VeRL_for_PyTorch/README.md
PyTorch/built-in/rl/VeRL_for_PyTorch/README_ORI.md
PyTorch/built-in/rl/VeRL_for_PyTorch/docker/Dockerfile.megatron
PyTorch/built-in/rl/VeRL_for_PyTorch/docker/Dockerfile.ngc.vllm
PyTorch/built-in/rl/VeRL_for_PyTorch/docker/Dockerfile.ngc.vllm0.8
PyTorch/built-in/rl/VeRL_for_PyTorch/docker/Dockerfile.ngc.vllm0.8.sagemaker
PyTorch/built-in/rl/VeRL_for_PyTorch/docker/Dockerfile.rocm
PyTorch/built-in/rl/VeRL_for_PyTorch/docker/Dockerfile.sglang
PyTorch/built-in/rl/VeRL_for_PyTorch/docker/Dockerfile.vemlp.vllm.te
PyTorch/built-in/rl/VeRL_for_PyTorch/docs/Makefile
PyTorch/built-in/rl/VeRL_for_PyTorch/docs/README.md
PyTorch/built-in/rl/VeRL_for_PyTorch/docs/README_vllm0.7.md
PyTorch/built-in/rl/VeRL_for_PyTorch/docs/README_vllm0.8.md
PyTorch/built-in/rl/VeRL_for_PyTorch/docs/conf.py
PyTorch/built-in/rl/VeRL_for_PyTorch/docs/profiler.md
PyTorch/built-in/rl/VeRL_for_PyTorch/docs/requirements-docs.txt
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/checkpoint/run_deepseek_megatron_ckpt.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/checkpoint/run_qwen_megatron_ckpt.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/data_preprocess/full_hh_rlhf.py
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/data_preprocess/geo3k.py
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/data_preprocess/gsm8k.py
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/data_preprocess/hellaswag.py
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/data_preprocess/math_dataset.py
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/data_preprocess/multiturn.py
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/generation/run_deepseek7b_mutli_node.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/generation/run_deepseek_v2_lite_math.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/grpo_trainer/run_deepseek7b_llm.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/grpo_trainer/run_deepseek7b_llm_math.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/grpo_trainer/run_deepseek7b_llm_math_megatron.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/grpo_trainer/run_deepseek7b_llm_megatron.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/grpo_trainer/run_deepseek7b_llm_seq_balance.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/grpo_trainer/run_qwen2-7b.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/grpo_trainer/run_qwen2-7b_math.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/grpo_trainer/run_qwen2-7b_math_megatron.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/grpo_trainer/run_qwen2-7b_megatron.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/grpo_trainer/run_qwen2-7b_seq_balance.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/grpo_trainer/run_qwen2_5_vl-7b.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/ppo_trainer/run_deepseek7b_llm.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/ppo_trainer/run_deepseek7b_llm_modelscope.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/ppo_trainer/run_deepseek7b_llm_sp2.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/ppo_trainer/run_deepseek_full_hh_rlhf.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/ppo_trainer/run_deepseek_math_gsm8k_megatron.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/ppo_trainer/run_deepseek_megatron.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/ppo_trainer/run_gemma.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/ppo_trainer/run_qwen2-7b_math_gsm8k_megatron.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/ppo_trainer/run_qwen2-7b_megatron.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/ppo_trainer/run_qwen2-7b_rm.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/ppo_trainer/run_qwen2-7b_rm_seq_balance.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/ppo_trainer/run_qwen2-7b_seq_balance.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/ppo_trainer/run_qwen2-7b_sglang_seq_balance.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/ppo_trainer/run_qwen2.5-32b.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/reinforce_plus_plus_trainer/run_qwen2-7b_math_rf.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/reinforce_plus_plus_trainer/run_qwen2-7b_math_rf_baseline.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/remax_trainer/run_qwen2.5-3b_seq_balance.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/remax_trainer/run_qwen2.5-7b_seq_balance.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/rloo_trainer/run_qwen2-7b.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/sft/gsm8k/run_deepseek_6b7.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/sft/gsm8k/run_gemma_2b.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/sft/gsm8k/run_gemma_7b.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/sft/gsm8k/run_qwen_05_peft.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/sft/gsm8k/run_qwen_05_sp2.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/sft/gsm8k/run_qwen_05_sp2_liger.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/sft/multiturn/run_qwen_05_sp2.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/slurm/ray_on_slurm.slurm
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/split_placement/README.md
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/split_placement/config/ppo_trainer_split.yaml
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/split_placement/main_ppo_split.py
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/split_placement/run_deepseek7b_llm.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/examples/split_placement/split_monkey_patch.py
PyTorch/built-in/rl/VeRL_for_PyTorch/patches/megatron_v4.patch
PyTorch/built-in/rl/VeRL_for_PyTorch/public_address_statement.md
PyTorch/built-in/rl/VeRL_for_PyTorch/pyproject.toml
PyTorch/built-in/rl/VeRL_for_PyTorch/recipe/dapo/README.md
PyTorch/built-in/rl/VeRL_for_PyTorch/recipe/dapo/prepare_dapo_data.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/recipe/dapo/run_dapo_early_qwen2.5_32b.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/recipe/dapo/run_dapo_qwen2.5_32b.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/recipe/dapo/src/config/dapo_trainer.yaml
PyTorch/built-in/rl/VeRL_for_PyTorch/recipe/dapo/src/dapo_ray_trainer.py
PyTorch/built-in/rl/VeRL_for_PyTorch/recipe/dapo/src/main_dapo.py
PyTorch/built-in/rl/VeRL_for_PyTorch/recipe/dapo/test_dapo_7b.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/recipe/r1/README.md
PyTorch/built-in/rl/VeRL_for_PyTorch/recipe/r1/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/recipe/r1/config/evaluation.yaml
PyTorch/built-in/rl/VeRL_for_PyTorch/recipe/r1/data_process.py
PyTorch/built-in/rl/VeRL_for_PyTorch/recipe/r1/main_eval.py
PyTorch/built-in/rl/VeRL_for_PyTorch/recipe/r1/reward_score.py
PyTorch/built-in/rl/VeRL_for_PyTorch/recipe/r1/run_r1_distill_qwen.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/recipe/r1/tasks/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/recipe/r1/tasks/gpqa.py
PyTorch/built-in/rl/VeRL_for_PyTorch/recipe/r1/tasks/livecodebench.py
PyTorch/built-in/rl/VeRL_for_PyTorch/recipe/r1/tasks/math.py
PyTorch/built-in/rl/VeRL_for_PyTorch/requirements-npu.txt
PyTorch/built-in/rl/VeRL_for_PyTorch/requirements.txt
PyTorch/built-in/rl/VeRL_for_PyTorch/requirements_sglang.txt
PyTorch/built-in/rl/VeRL_for_PyTorch/scripts/format.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/scripts/model_merger.py
PyTorch/built-in/rl/VeRL_for_PyTorch/setup.py
PyTorch/built-in/rl/VeRL_for_PyTorch/test/env_npu.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/test/runtime_env.yaml
PyTorch/built-in/rl/VeRL_for_PyTorch/test/runtime_env32.yaml
PyTorch/built-in/rl/VeRL_for_PyTorch/test/train_qwen2_5_32b_instruct_DAPO_full_32p.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/test/train_qwen2_5_32b_instruct_DAPO_performance_32p.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/test/train_qwen2_5_32b_instruct_GRPO_full_32p.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/test/train_qwen2_5_32b_instruct_GRPO_performance_32p.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/test/train_qwen2_5_7b_instruct_DAPO_full_16p.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/test/train_qwen2_5_7b_instruct_DAPO_performance_16p.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/test/train_qwen2_5_7b_instruct_GRPO_full_16p.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/test/train_qwen2_5_7b_instruct_GRPO_performance_16p.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/test/train_qwen2_5_vl_32b_GRPO_full_32p.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/test/train_qwen2_5_vl_32b_GRPO_performance_32p.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/test/train_qwen2_5_vl_3b_GRPO_full_8p.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/test/train_qwen2_5_vl_3b_GRPO_performance_8p.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/test/train_qwen2_5_vl_7b_GRPO_full_16p.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/test/train_qwen2_5_vl_7b_GRPO_performance_16p.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/checkpoint/run_deepseek_megatron_ckpt.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/checkpoint/run_qwen_megatron_ckpt.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/checkpoint/test_fsdp_ckpt.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/distributed/run_all.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/distributed/test_tensor_dict.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/arithmetic_sequence/data/create_dataset.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/arithmetic_sequence/model/config.json
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/arithmetic_sequence/model/create_model_tokenizer.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/arithmetic_sequence/model/generation_config.json
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/arithmetic_sequence/model/tokenizer_config.json
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/arithmetic_sequence/rl/README.md
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/arithmetic_sequence/rl/main_trainer.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/check_custom_rwd_fn.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/check_results.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/envs/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/envs/digit_completion/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/envs/digit_completion/task.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/envs/digit_completion/tokenizer.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/run_deepseek_grpo.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/run_deepseek_grpo_megatron.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/run_deepseek_megatron.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/run_deepseek_megatron_parallelism.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/run_qwen2vl_geo3k_function_rm.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/run_qwen_grpo.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/run_qwen_grpo_megatron.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/run_qwen_gsm8k_custom_function_rm.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/run_qwen_gsm8k_dapo.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/run_qwen_gsm8k_function_rm.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/run_qwen_gsm8k_function_rm_both_kl.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/run_qwen_gsm8k_function_rm_grpo.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/run_qwen_gsm8k_function_rm_no_rmpad.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/run_qwen_gsm8k_function_rm_remax.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/run_qwen_gsm8k_model_rm.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/run_qwen_gsm8k_model_rm_liger_kernel.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/run_qwen_gsm8k_model_rm_no_rmpad.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/run_qwen_gsm8k_model_rm_seq_balance.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/run_qwen_gsm8k_model_rm_ulysses.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/run_qwen_gsm8k_prime.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/run_qwen_megatron.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/run_qwen_megatron_parallelism.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/run_r1_distill_qwen_aime24_eval.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/run_ray_trainer.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/run_ray_trainer_fire_sampling.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/run_ray_trainer_rmpad.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/generation/run_gen_qwen05.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/gpu_utility/test_memory_buffers.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/gpu_utility/test_ops.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/gpu_utility/test_torch_functional.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/kill_github_tests.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/model/test_transformer.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/model/test_transformers_ulysses.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/ray/check_worker_alive/main.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/ray/detached_worker/README.md
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/ray/detached_worker/client.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/ray/detached_worker/run.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/ray/detached_worker/server.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/ray/test_check_worker_alive.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/ray/test_colocated_workers.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/ray/test_data_transfer.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/ray/test_driverfunc_to_worker.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/ray/test_high_level_scheduling_api.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/ray/test_ray_local_envs.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/ray/test_rvdz.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/ray/test_worker_group_basics.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/ray/test_worker_group_torch.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/rollout/run_fsdp_vllm.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/rollout/test_sglang_spmd.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/rollout/test_vllm_hf_loader.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/rollout/test_vllm_spmd.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/sandbox/test_sandbox.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/sanity/check_license.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/sanity/test_import.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/sft/run_sft.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/sft/run_sft_multiturn.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/sft/run_sft_qwen05_peft.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/sft/run_sft_qwen05_sp2_liger.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/sft/run_sft_sp_loss_match.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/sft/test_sp_loss_match.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/utility/test_tensor_dict_utilities.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/verl/utils/dataset/test_multiturn_sft_dataset.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/verl/utils/dataset/test_rl_dataset.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/verl/utils/dataset/test_rm_dataset.py
PyTorch/built-in/rl/VeRL_for_PyTorch/tests/verl/utils/dataset/test_sft_dataset.py
PyTorch/built-in/rl/VeRL_for_PyTorch/transformers_need/npu_flash_attention.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/README.md
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/llama/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/llama/megatron/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/llama/megatron/checkpoint_utils/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/llama/megatron/checkpoint_utils/llama_loader.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/llama/megatron/checkpoint_utils/llama_loader_depracated.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/llama/megatron/checkpoint_utils/llama_saver.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/llama/megatron/layers/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/llama/megatron/layers/parallel_attention.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/llama/megatron/layers/parallel_decoder.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/llama/megatron/layers/parallel_linear.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/llama/megatron/layers/parallel_mlp.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/llama/megatron/layers/parallel_rmsnorm.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/llama/megatron/modeling_llama_megatron.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/mcore/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/mcore/gpt_model.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/mcore/loader.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/mcore/saver.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/qwen2/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/qwen2/megatron/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/qwen2/megatron/checkpoint_utils/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader_depracated.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/qwen2/megatron/checkpoint_utils/qwen2_saver.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/qwen2/megatron/layers/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/qwen2/megatron/layers/parallel_attention.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/qwen2/megatron/layers/parallel_decoder.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/qwen2/megatron/layers/parallel_linear.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/qwen2/megatron/layers/parallel_mlp.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/qwen2/megatron/layers/parallel_rmsnorm.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/qwen2/megatron/modeling_qwen2_megatron.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/registry.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/transformers/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/transformers/llama.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/transformers/monkey_patch.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/transformers/qwen2.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/transformers/qwen2_vl.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/weight_loader_registry.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/protocol.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/single_controller/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/single_controller/base/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/single_controller/base/decorator.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/single_controller/base/megatron/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/single_controller/base/megatron/worker.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/single_controller/base/megatron/worker_group.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/single_controller/base/register_center/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/single_controller/base/register_center/ray.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/single_controller/base/worker.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/single_controller/base/worker_group.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/single_controller/ray/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/single_controller/ray/base.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/single_controller/ray/megatron.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/sglang/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/sglang/parallel_state.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_3_1/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_3_1/arg_utils.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_3_1/config.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_3_1/llm.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_3_1/llm_engine_sp.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_3_1/model_loader.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_3_1/model_runner.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_3_1/parallel_state.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_3_1/tokenizer.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_3_1/weight_loaders.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_3_1/worker.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_4_2/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_4_2/arg_utils.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_4_2/config.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_4_2/dtensor_weight_loaders.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_4_2/hf_weight_loader.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_4_2/llm.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_4_2/llm_engine_sp.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_4_2/megatron_weight_loaders.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_4_2/model_loader.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_4_2/model_runner.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_4_2/parallel_state.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_4_2/spmd_gpu_executor.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_4_2/tokenizer.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_4_2/worker.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_5_4/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_5_4/arg_utils.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_5_4/config.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_5_4/dtensor_weight_loaders.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_5_4/hf_weight_loader.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_5_4/llm.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_5_4/llm_engine_sp.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_5_4/megatron_weight_loaders.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_5_4/model_loader.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_5_4/model_runner.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_5_4/parallel_state.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_5_4/spmd_gpu_executor.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_5_4/tokenizer.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_5_4/worker.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_6_3/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_6_3/arg_utils.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_6_3/config.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_6_3/dtensor_weight_loaders.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_6_3/hf_weight_loader.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_6_3/llm.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_6_3/llm_engine_sp.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_6_3/megatron_weight_loaders.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_6_3/model_loader.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_6_3/model_runner.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_6_3/parallel_state.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_6_3/spmd_gpu_executor.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_6_3/tokenizer.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/third_party/vllm/vllm_v_0_6_3/worker.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/trainer/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/trainer/config/evaluation.yaml
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/trainer/config/generation.yaml
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/trainer/config/ppo_megatron_trainer.yaml
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/trainer/config/ppo_trainer.yaml
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/trainer/config/profiler_config/profiler.yaml
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/trainer/config/sft_trainer.yaml
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/trainer/fsdp_sft_trainer.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/trainer/main_eval.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/trainer/main_generation.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/trainer/main_ppo.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/trainer/ppo/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/trainer/ppo/core_algos.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/trainer/ppo/metric_utils.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/trainer/ppo/ray_trainer.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/trainer/runtime_env.yaml
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/checkpoint/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/checkpoint/checkpoint_manager.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/checkpoint/fsdp_checkpoint_manager.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/checkpoint/megatron_checkpoint_manager.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/config.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/dataset/README.md
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/dataset/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/dataset/multiturn_sft_dataset.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/dataset/rl_dataset.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/dataset/rm_dataset.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/dataset/sft_dataset.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/debug/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/debug/performance.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/debug/trajectory_tracker.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/device.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/distributed.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/flops_counter.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/fs.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/fsdp_utils.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/hdfs_io.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/import_utils.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/logger/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/logger/aggregate_logger.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/logging_utils.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/megatron/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/megatron/memory.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/megatron/optimizer.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/megatron/pipeline_parallel.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/megatron/sequence_parallel.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/megatron/tensor_parallel.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/megatron_utils.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/memory_buffer.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/model.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/npu_patch.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/profiler.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/py_functional.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/ray_utils.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/rendezvous/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/rendezvous/ray_backend.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/reward_score/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/reward_score/geo3k.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/reward_score/gsm8k.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/reward_score/math.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/reward_score/math_batch.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/reward_score/math_dapo.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/reward_score/math_verify.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/reward_score/prime_code/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/reward_score/prime_code/testing_util.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/reward_score/prime_code/utils.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/reward_score/prime_math/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/reward_score/prime_math/grader.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/reward_score/prime_math/math_normalize.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/seqlen_balancing.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/tokenizer.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/torch_dtypes.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/torch_functional.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/tracking.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/ulysses.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/version/version
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/actor/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/actor/base.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/actor/dp_actor.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/actor/megatron_actor.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/critic/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/critic/base.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/critic/dp_critic.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/critic/megatron_critic.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/fsdp_workers.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/megatron_workers.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/reward_manager/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/reward_manager/batch.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/reward_manager/dapo.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/reward_manager/naive.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/reward_manager/prime.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/reward_model/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/reward_model/base.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/reward_model/megatron/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/reward_model/megatron/reward_model.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/rollout/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/rollout/base.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/rollout/hf_rollout.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/rollout/naive/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/rollout/naive/naive_rollout.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/rollout/sglang_rollout/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/rollout/sglang_rollout/sglang_rollout.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/rollout/tokenizer.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/rollout/vllm_rollout/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/rollout/vllm_rollout/fire_vllm_rollout.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/rollout/vllm_rollout/vllm_rollout.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/sharding_manager/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/sharding_manager/base.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/sharding_manager/fsdp_sglang.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/sharding_manager/fsdp_ulysses.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/sharding_manager/fsdp_vllm.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/sharding_manager/megatron_vllm.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/sharding_manager/patch/__init__.py
PyTorch/built-in/rl/VeRL_for_PyTorch/verl/workers/sharding_manager/patch/fsdp_vllm_patch.py
PyTorch/built-in/rl/VeRL_for_PyTorch/vllm_ascend_need/qwen2_5_vl.py
PyTorch/built-in/rl/VeRL_for_PyTorch/vllm_ascend_need/rotary_embedding.py
```

### [PyTorch/built-in/rl/VeRL_for_PyTorch/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/built-in/rl/VeRL_for_PyTorch/README.md)

文本SHA256：`a8ea5ed066b2b0583ce1085a47d0a63740e03a51898d14fd621ba46166c458bf`

- L1: # 目前本仓已停止维护，请转跳至
- L16: verl‌是一个集SFT（监督学习）与RL（强化学习）于一体的灵活大模型后训练框架。它特别适用于大型语言模型（LLM）的后训练阶段，旨在通过调整预训练模型的参数以适应新的任务或数据集。
- L21: url=https://github.com/volcengine/verl
- L28: url=https://gitcode.com/ascend/ModelZoo-PyTorch.git
- L36: - 推荐参考[配套资源文档](https://www.hiascend.com/developer/download/commercial)使用最新的配套版本。
- L49: <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0003.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit">驱动固件安装指南</a> 》</td>
- L58: <td>《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0000.html">CANN 软件安装指南</a> 》</td>
- L63: <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/configandinstg/instg/insg_0001.html">Ascend Extension for PyTorch 配置与安装</a> 》</td>
- L74: git clone -b v0.7.3 --depth 1 https://github.com/vllm-project/vllm.git
- L77: VLLM_TARGET_DEVICE=empty pip install -e . --extra-index https://download.pytorch.org/whl/cpu/
- L84: git clone -b v0.7.3 --depth 1 https://github.com/vllm-project/vllm-ascend.git
- L93: git clone https://github.com/vllm-project/vllm-ascend.git
- L104: git clone --depth 1 https://github.com/huggingface/transformers.git
- L115: pip install torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu
- L120: 请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》搭建torch环境。
- L131: ## 准备数据集
- L133: VL模型使用geo3k数据集，在模型根目录下执行命令，下载并处理数据集，`--local_dir`为可选参数，不设置默认下载位置为`~/data/geo3k`。
- L139: LLM模型使用gsm8k数据集，在模型根目录下执行命令，下载并处理数据集，`--local_dir`为可选参数，不设置默认下载位置为`~/data/gsm8k`。
- L161: 1. 主从节点保证模型和数据集路径完全相同。
- L190: - 单机8卡性能
- L193: bash test/train_qwen2_5_vl_3b_GRPO_performance_8p.sh --data_path=xxx --model_path=xxx   # 8卡性能
- L204: - 单机16卡性能
- L207: bash test/train_qwen2_5_vl_7b_GRPO_performance_16p.sh --data_path=xxx --model_path=xxx   # 16卡性能
- L219: - 双机32卡性能
- L223: bash test/train_qwen2_5_vl_32b_GRPO_performance_32p.sh --data_path=xxx --model_path=xxx   # 32卡性能
- L234: - 单机16卡性能
- L237: bash test/train_qwen2_5_7b_instruct_GRPO_performance_16p.sh --data_path=xxx --model_path=xxx   # 16卡性能
- L249: - 双机32卡性能
- L253: bash test/train_qwen2_5_32b_instruct_GRPO_performance_32p.sh --data_path=xxx --model_path=xxx   # 32卡性能
- L256: 训练完成后，训练日志保存在`test/output`路径下，并输出模型训练精度和性能信息。
- L268: - 单机16卡性能
- L271: bash test/train_qwen2_5_7b_instruct_DAPO_performance_16p.sh --data_path=xxx --model_path=xxx   # 16卡性能
- L283: - 双机32卡性能
- L287: bash test/train_qwen2_5_32b_instruct_DAPO_performance_32p.sh --data_path=xxx --model_path=xxx   # 32卡性能
- L296: | Qwen2.5-VL-3B-Instruct | GRPO      | 8p-竞品A                    | 739.453    | 60                     |
- L297: | Qwen2.5-VL-3B-Instruct | GRPO      | 8P Atlas 200T A2 Box16     | 523.599    | 60                     |
- L298: | Qwen2.5-VL-7B-Instruct | GRPO      | 8p-竞品A                    | 568.452    | 60                     |
- L299: | Qwen2.5-VL-7B-Instruct | GRPO      | 16P Atlas 200T A2 Box16    | 326.37    | 60                     |
- L300: | Qwen2.5-VL-32B-Instruct |   GRPO   | 16p-竞品A                  |  109.497   | 60                     |
- L301: | Qwen2.5-VL-32B-Instruct |   GRPO   | 32P Atlas 200T A2 Box16    |  70.0735   | 60                      |
- L302: | Qwen2.5-7B-Instruct    | GRPO      | 8p-竞品A                    | 323.872    | 35                     |
- L303: | Qwen2.5-7B-Instruct    | GRPO      | 16P Atlas 200T A2 Box16    | 190.617    | 35                     |
- L304: | Qwen2.5-7B-Instruct    | DAPO      | 16P Atlas 200T A2 Box16    | 198.678    | 84                     |
- L305: | Qwen2.5-32B-Instruct   | GRPO      | 16p-竞品A                   | 79.022     | 105                    |
- L306: | Qwen2.5-32B-Instruct   | GRPO      | 32P Atlas 200T A2 Box16    | 54.162     | 105                    |
- L307: | Qwen2.5-32B-Instruct   | DAPO      | 32P Atlas 200T A2 Box16    | 47.725     | 32                     |
- L342: - 如果在`docker`容器中训练时发现性能远低于基线，可能是由于`Ray`在容器内的调度性能低于物理机导致的，可以在物理机执行以下命令拉起`Ray`，然后在容器内训练：

### [PyTorch/built-in/rl/VeRL_for_PyTorch/docs/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/built-in/rl/VeRL_for_PyTorch/docs/README.md)

文本SHA256：`09f50b789624a3bc937968fdef6b23e87b1d6b789db068c3bd78df75ae585525`

- L19: Launch your browser and navigate to http://localhost:8000 to view the documentation.

### [PyTorch/built-in/rl/VeRL_for_PyTorch/examples/split_placement/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/built-in/rl/VeRL_for_PyTorch/examples/split_placement/README.md)

文本SHA256：`f97f63cc209fc4f4d29e6f45af06332645e1b3112f677afaa0bea16385f77310`


### [PyTorch/built-in/rl/VeRL_for_PyTorch/recipe/dapo/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/built-in/rl/VeRL_for_PyTorch/recipe/dapo/README.md)

文本SHA256：`114ac6ad454ebfda31cc615af741d1bacbfb4518cc29709149eb054a31eab66f`

- L3: > Open-Source Algorithm Implementation & Expriement Running: [Yuxuan Tong](https://tongyx361.github.io/), [Guangming Sheng](https://hk.linkedin.com/in/guangming-sheng-b50640211)
- L7: > - [2025/03] We published the training record of [an early version of DAPO (w/o Token-level PG Loss & Dynamic Sampling)](./run_dapo_early_qwen2.5_32b.sh), achieving 44% on AIME 2024, in [W&B](https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl).
- L9: 🏠 [Homepage](https://dapo-sia.github.io/) | 📝 [Paper](https://dapo-sia.github.io/static/pdf/dapo_paper.pdf) | 🤗 [Datasets&Models@HF](https://huggingface.co/collections/BytedTsinghua-SIA/dapo-67d7f1517ee33c8aed059da0) | 🐱 Code@GitHub | 🐱 [Repo@GitHub](https://github.com/BytedTsinghua-SIA/DAPO)
- L11: > We propose the **D**ecoupled Clip and Dynamic s**A**mpling **P**olicy **O**ptimization (DAPO) algorithm. By making our work publicly available, we provide the broader research community and society with practical access to scalable reinforcement learning, enabling all to benefit from these advancements. Our system is based on the awesome [verl](https://github.com/volcengine/verl) framework. Thanks for their great work! Applying DAPO training to Qwen2.5-32B base model proves to outperform the previous state-of-the-art DeepSeek-R1-Zero-Qwen-32B on AIME 2024, achieving **50%** accuracy with **50%** less training steps.
- L13: > ![dapo-main-result](https://dapo-sia.github.io/static/images/score.png)
- L27: export RAY_ADDRESS="http://${RAY_IP:-localhost}:8265" # The Ray cluster address to connect to
- L36: | Setup | AIME 2024 Acc. | Training Script | Training Record |
- L38: | DAPO w/o Token-level PG Loss & Dynamic Sampling | 44% | [run_dapo_early_qwen2.5_32b.sh](./run_dapo_early_qwen2.5_32b.sh) | [W&B](https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl) |
- L39: | DAPO | 50% | [run_dapo_qwen2.5_32b.sh](./run_dapo_qwen2.5_32b.sh) | W&B (Coming soon) |

### [PyTorch/built-in/rl/VeRL_for_PyTorch/recipe/r1/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/built-in/rl/VeRL_for_PyTorch/recipe/r1/README.md)

文本SHA256：`540e9896d938614ff93526c427d7a98d736a04b7c9ca3bf80fb6725dd4b55b5d`

- L3: This recipe is under development, if you are interested, checkout the TODO list and join this project! https://github.com/volcengine/verl/issues/708
- L11: GPQA Diamond | 35.3 | 33.8
- L12: LiveCodeBench | 16.9 | 16.9
- L13: AIME 2024 | 30.4 | 28.9
- L14: CNMO 2024 (en) | 45.1 | -
- L15: CNMO 2024 (zh) | 41.0 | -
- L21: Dataset | Test Results (k=1) | Test Results (k=4) | Reported
- L23: GPQA Diamond | 67.7 | 69.6 | 71.5
- L24: LiveCodeBench | 64.7 | 63.1 | 65.9
- L25: AIME 2024 | 86.7 | 79.2 | 79.8
- L26: CNMO 2024 | 75.0 | 78.5 | 78.8

### [PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/arithmetic_sequence/rl/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/built-in/rl/VeRL_for_PyTorch/tests/e2e/arithmetic_sequence/rl/README.md)

文本SHA256：`94d3f6e0c97bce4448e371bf90e54a2d7940724b00cb4570bfbe271906832ce1`


### [PyTorch/built-in/rl/VeRL_for_PyTorch/tests/ray/detached_worker/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/built-in/rl/VeRL_for_PyTorch/tests/ray/detached_worker/README.md)

文本SHA256：`21f9d64b62e4cc7c8160ae82ec294a137be532c7da27c7535e531b7c0ad2351d`


### [PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/built-in/rl/VeRL_for_PyTorch/verl/models/README.md)

文本SHA256：`08820cb3a0f8a744383dee6fe51c531d65dc6eeed6bab738e204807db18e1b1a`

- L22: - https://pytorch.org/docs/stable/distributed.tensor.parallel.html
- L23: - https://pytorch.org/tutorials/intermediate/TP_tutorial.html

### [PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/dataset/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/built-in/rl/VeRL_for_PyTorch/verl/utils/dataset/README.md)

文本SHA256：`de346fc1a883566c1b8f1f22c0ba4563efadfdb7b53d5935796401ad18d56f57`


## PyTorch/built-in/rl/VeRL_for_PyTorch/test

文件清单：

```text
PyTorch/built-in/rl/VeRL_for_PyTorch/test/env_npu.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/test/runtime_env.yaml
PyTorch/built-in/rl/VeRL_for_PyTorch/test/runtime_env32.yaml
PyTorch/built-in/rl/VeRL_for_PyTorch/test/train_qwen2_5_32b_instruct_DAPO_full_32p.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/test/train_qwen2_5_32b_instruct_DAPO_performance_32p.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/test/train_qwen2_5_32b_instruct_GRPO_full_32p.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/test/train_qwen2_5_32b_instruct_GRPO_performance_32p.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/test/train_qwen2_5_7b_instruct_DAPO_full_16p.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/test/train_qwen2_5_7b_instruct_DAPO_performance_16p.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/test/train_qwen2_5_7b_instruct_GRPO_full_16p.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/test/train_qwen2_5_7b_instruct_GRPO_performance_16p.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/test/train_qwen2_5_vl_32b_GRPO_full_32p.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/test/train_qwen2_5_vl_32b_GRPO_performance_32p.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/test/train_qwen2_5_vl_3b_GRPO_full_8p.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/test/train_qwen2_5_vl_3b_GRPO_performance_8p.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/test/train_qwen2_5_vl_7b_GRPO_full_16p.sh
PyTorch/built-in/rl/VeRL_for_PyTorch/test/train_qwen2_5_vl_7b_GRPO_performance_16p.sh
```

## PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch

文件清单：

```text
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/.gitignore
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/1.11_requirements.txt
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/2.1_requirements.txt
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/2.2_requirements.txt
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/2.3_requirements.txt
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/2.4_requirements.txt
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/Dockerfile
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/LICENSE
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/README.md
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/archs.py
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/dataset.py
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/demo.py
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/docker_start.sh
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/infer/convert/atc.sh
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/infer/data/config/unet.pipeline
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/infer/docker_start_infer.sh
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/infer/mxbase/CMakeLists.txt
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/infer/mxbase/build.sh
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/infer/mxbase/src/UNet.cpp
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/infer/mxbase/src/UNet.h
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/infer/mxbase/src/main.cpp
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/infer/sdk/main.py
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/infer/sdk/metrics.py
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/infer/utils/cal_iou.py
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/losses.py
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/lovasz_losses.py
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/metrics.py
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/modelarts/train_start.py
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/modelzoo_level.txt
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/preprocess_dsb2018.py
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/pthtar2onnx.py
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/test/env_npu.sh
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/test/train_eval_8p.sh
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/test/train_full_1p.sh
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/test/train_full_8p.sh
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/test/train_performance_1p.sh
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/test/train_performance_8p.sh
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/train.py
PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/utils.py
```

### [PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/contrib/cv/semantic_segmentation/UNet_for_PyTorch/README.md)

文本SHA256：`022946a43c13697a2daa27f773d5167a0ed1f84f7d516eaa9a17966d8a527c6b`

- L17: url=https://github.com/4uiiurz1/pytorch-nested-unet
- L22: url=https://gitcode.com/ascend/ModelZoo-PyTorch.git
- L37: | PyTorch 1.11 | torchvision==0.12.0 |
- L38: | PyTorch 2.1 | torchvision==0.16.0 |
- L39: | PyTorch 2.2 | torchvision==0.17.0 |
- L40: | PyTorch 2.3 | torchvision==0.18.1 |
- L41: | PyTorch 2.4 | torchvision==0.19.0 |
- L45: 请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
- L65: ## 准备数据集
- L67: 1. 用户自行获取 `data-science-bowl-2018` 数据集。
- L68: 2. 上传数据集在源码包根目录下新建的 `inputs` 文件夹下并解压。
- L69: 3. 数据集需要执行预处理，在源码包根目录下执行
- L73: 数据集目录结构参考如下所示。
- L78: |   ├── 00ae65...
- L88: >该数据集的训练过程脚本只作为一种参考示例。
- L110: bash ./test/train_full_1p.sh --data_path={data/path} # 单卡精度
- L111: bash ./test/train_performance_1p.sh --data_path={data/path} # 单卡性能
- L119: bash ./test/train_full_8p.sh --data_path={data/path} # 8卡精度
- L120: bash ./test/train_performance_8p.sh --data_path={data/path} # 8卡性能
- L131: --data_path参数填写数据集路径，需写到数据集的一级目录。
- L138: --data_path             //数据集路径
- L148: 训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。
- L156: | 1p-竞品V | -     |  823 | 1      |        - | 1.5 |
- L157: | 8p-竞品V | 83.91 | 2332 | 100    |        - | 1.5 |
- L158: | 8p-竞品A | -     | 792  | 1      |        - | 2.1 |
- L159: | 8p-竞品A | 81.58 | 2560 | 100    |        - | 2.1 |
- L160: | 1p-Atlas | -     |  231.15   | 1      |       O2 | 1.8 |
- L161: | 8p-Atlas | 83.31 |  2070.295 | 100    |       O2 | 1.8 |
- L162: | 1p-Atlas 800T A2 | -     |  261.950  | 1      |       O2 | 2.1 |
- L163: | 8p-Atlas 800T A2 | 85.90 |  1722.650 | 100    |       O2 | 2.1 |

## PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch

文件清单：

```text
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/LICENSE
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/README.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/Base-RCNN-C4.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/Base-RCNN-DilatedC5.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/Base-RCNN-FPN.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/Base-RetinaNet.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-Detection/faster_rcnn_R_101_C4_3x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-Detection/faster_rcnn_R_50_C4_1x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-Detection/faster_rcnn_R_50_C4_3x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-Detection/rpn_R_50_C4_1x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-Detection/rpn_R_50_FPN_1x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x_giou.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-Keypoints/Base-Keypoint-RCNN-FPN.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-PanopticSegmentation/Base-Panoptic-FPN.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/Cityscapes/mask_rcnn_R_50_FPN.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/Detectron1-Comparisons/README.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/Detectron1-Comparisons/faster_rcnn_R_50_FPN_noaug_1x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/Detectron1-Comparisons/keypoint_rcnn_R_50_FPN_1x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/Detectron1-Comparisons/mask_rcnn_R_50_FPN_noaug_1x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/LVISv1-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/LVISv1-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/Misc/cascade_mask_rcnn_R_50_FPN_1x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/Misc/mask_rcnn_R_50_FPN_1x_cls_agnostic.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/Misc/mask_rcnn_R_50_FPN_1x_dconv_c3-c5.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/Misc/mask_rcnn_R_50_FPN_3x_dconv_c3-c5.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/Misc/mask_rcnn_R_50_FPN_3x_gn.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/Misc/mask_rcnn_R_50_FPN_3x_syncbn.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/Misc/scratch_mask_rcnn_R_50_FPN_9x_gn.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/Misc/scratch_mask_rcnn_R_50_FPN_9x_syncbn.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/Misc/semantic_R_50_FPN_1x.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/PascalVOC-Detection/faster_rcnn_R_50_C4.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/quick_schedules/README.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/quick_schedules/cascade_mask_rcnn_R_50_FPN_inference_acc_test.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/quick_schedules/cascade_mask_rcnn_R_50_FPN_instant_test.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/quick_schedules/fast_rcnn_R_50_FPN_inference_acc_test.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/quick_schedules/fast_rcnn_R_50_FPN_instant_test.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/quick_schedules/keypoint_rcnn_R_50_FPN_inference_acc_test.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/quick_schedules/keypoint_rcnn_R_50_FPN_instant_test.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/quick_schedules/keypoint_rcnn_R_50_FPN_normalized_training_acc_test.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/quick_schedules/keypoint_rcnn_R_50_FPN_training_acc_test.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/quick_schedules/mask_rcnn_R_50_C4_GCV_instant_test.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/quick_schedules/mask_rcnn_R_50_C4_inference_acc_test.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/quick_schedules/mask_rcnn_R_50_C4_instant_test.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/quick_schedules/mask_rcnn_R_50_C4_training_acc_test.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/quick_schedules/mask_rcnn_R_50_DC5_inference_acc_test.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/quick_schedules/mask_rcnn_R_50_FPN_instant_test.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/quick_schedules/mask_rcnn_R_50_FPN_pred_boxes_training_acc_test.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/quick_schedules/mask_rcnn_R_50_FPN_training_acc_test.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/quick_schedules/panoptic_fpn_R_50_inference_acc_test.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/quick_schedules/panoptic_fpn_R_50_instant_test.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/quick_schedules/panoptic_fpn_R_50_training_acc_test.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/quick_schedules/retinanet_R_50_FPN_inference_acc_test.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/quick_schedules/retinanet_R_50_FPN_instant_test.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/quick_schedules/rpn_R_50_FPN_inference_acc_test.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/quick_schedules/rpn_R_50_FPN_instant_test.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/quick_schedules/semantic_R_50_FPN_inference_acc_test.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/quick_schedules/semantic_R_50_FPN_instant_test.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/quick_schedules/semantic_R_50_FPN_training_acc_test.yaml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/datasets/README.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/datasets/prepare_cocofied_lvis.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/datasets/prepare_for_tests.sh
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/datasets/prepare_panoptic_fpn.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/demo/README.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/demo/demo.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/demo/predictor.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/__init__.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/checkpoint/__init__.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/checkpoint/c2_model_loading.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/checkpoint/catalog.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/checkpoint/detection_checkpoint.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/config/__init__.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/config/compat.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/config/config.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/config/defaults.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/data/__init__.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/data/build.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/data/catalog.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/data/common.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/data/dataset_mapper.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/data/datasets/README.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/data/datasets/__init__.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/data/datasets/builtin.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/data/datasets/builtin_meta.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/data/datasets/cityscapes.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/data/datasets/coco.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/data/datasets/lvis.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/data/datasets/lvis_v0_5_categories.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/data/datasets/lvis_v1_categories.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/data/datasets/pascal_voc.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/data/datasets/register_coco.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/data/detection_utils.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/data/samplers/__init__.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/data/samplers/distributed_sampler.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/data/samplers/grouped_batch_sampler.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/data/transforms/__init__.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/data/transforms/augmentation.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/data/transforms/augmentation_impl.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/data/transforms/transform.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/engine/__init__.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/engine/defaults.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/engine/hooks.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/engine/launch.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/engine/train_loop.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/evaluation/__init__.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/evaluation/cityscapes_evaluation.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/evaluation/coco_evaluation.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/evaluation/evaluator.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/evaluation/fast_eval_api.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/evaluation/lvis_evaluation.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/evaluation/panoptic_evaluation.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/evaluation/pascal_voc_evaluation.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/evaluation/rotated_coco_evaluation.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/evaluation/sem_seg_evaluation.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/evaluation/testing.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/export/README.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/export/__init__.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/export/api.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/export/c10.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/export/caffe2_export.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/export/caffe2_inference.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/export/caffe2_modeling.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/export/patcher.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/export/shared.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/export/torchscript.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/__init__.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/aspp.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/batch_norm.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/blocks.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/csrc/README.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/csrc/ROIAlign/ROIAlign.h
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/csrc/ROIAlign/ROIAlign_cpu.cpp
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/csrc/ROIAlign/ROIAlign_cuda.cu
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/csrc/ROIAlignRotated/ROIAlignRotated.h
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/csrc/ROIAlignRotated/ROIAlignRotated_cpu.cpp
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/csrc/ROIAlignRotated/ROIAlignRotated_cuda.cu
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated.h
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated_cpu.cpp
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated_cuda.cu
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated_utils.h
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/csrc/cocoeval/cocoeval.cpp
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/csrc/cocoeval/cocoeval.h
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/csrc/cuda_version.cu
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/csrc/deformable/deform_conv.h
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/csrc/deformable/deform_conv_cuda.cu
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/csrc/deformable/deform_conv_cuda_kernel.cu
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/csrc/nms_rotated/nms_rotated.h
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/csrc/nms_rotated/nms_rotated_cpu.cpp
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/csrc/nms_rotated/nms_rotated_cuda.cu
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/csrc/vision.cpp
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/deform_conv.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/mask_ops.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/nms.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/roi_align.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/roi_align_rotated.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/rotated_boxes.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/shape_spec.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/wrappers.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/model_zoo/__init__.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/model_zoo/model_zoo.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/__init__.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/anchor_generator.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/backbone/__init__.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/backbone/backbone.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/backbone/build.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/backbone/fpn.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/backbone/resnet.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/box_regression.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/matcher.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/meta_arch/__init__.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/meta_arch/build.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/meta_arch/panoptic_fpn.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/meta_arch/rcnn.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/meta_arch/retinanet.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/meta_arch/semantic_seg.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/poolers.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/postprocessing.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/proposal_generator/__init__.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/proposal_generator/build.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/proposal_generator/proposal_utils.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/proposal_generator/rpn.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/proposal_generator/rrpn.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/roi_heads/__init__.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/roi_heads/box_head.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/roi_heads/cascade_rcnn.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/roi_heads/fast_rcnn.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/roi_heads/keypoint_head.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/roi_heads/mask_head.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/roi_heads/roi_heads.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/roi_heads/rotated_fast_rcnn.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/sampling.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/modeling/test_time_augmentation.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/solver/__init__.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/solver/build.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/solver/lr_scheduler.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/structures/__init__.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/structures/boxes.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/structures/image_list.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/structures/instances.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/structures/keypoints.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/structures/masks.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/structures/rotated_boxes.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/utils/README.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/utils/__init__.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/utils/analysis.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/utils/collect_env.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/utils/colormap.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/utils/comm.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/utils/env.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/utils/events.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/utils/logger.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/utils/memory.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/utils/registry.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/utils/serialize.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/utils/video_visualizer.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/utils/visualizer.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/dev/README.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/dev/linter.sh
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/dev/packaging/README.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/dev/packaging/build_all_wheels.sh
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/dev/packaging/build_wheel.sh
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/dev/packaging/gen_install_table.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/dev/packaging/gen_wheel_index.sh
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/dev/packaging/pkg_helpers.bash
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/dev/parse_results.sh
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/dev/run_inference_tests.sh
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/dev/run_instant_tests.sh
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docker/Dockerfile
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docker/Dockerfile-circleci
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docker/README.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docker/docker-compose.yml
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docs/.gitignore
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docs/Makefile
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docs/README.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docs/_static/css/custom.css
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docs/conf.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docs/notes/benchmarks.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docs/notes/changelog.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docs/notes/compatibility.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docs/notes/contributing.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docs/tutorials/README.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docs/tutorials/builtin_datasets.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docs/tutorials/configs.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docs/tutorials/data_loading.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docs/tutorials/datasets.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docs/tutorials/deployment.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docs/tutorials/evaluation.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docs/tutorials/extend.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docs/tutorials/getting_started.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docs/tutorials/install.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docs/tutorials/models.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docs/tutorials/training.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docs/tutorials/write-models.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/modelzoo_level.txt
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/public_address_statement.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/requirements.txt
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/setup.cfg
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/setup.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/test/.keep
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/test/train_full_1p.sh
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/test/train_full_8p.sh
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/test/train_performance_1p.sh
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/test/train_performance_8p.sh
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/README.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/__init__.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/data/__init__.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/data/test_coco.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/data/test_coco_evaluation.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/data/test_detection_utils.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/data/test_rotation_transform.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/data/test_sampler.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/data/test_transforms.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/layers/__init__.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/layers/test_mask_ops.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/layers/test_nms.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/layers/test_nms_rotated.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/layers/test_roi_align.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/layers/test_roi_align_rotated.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/modeling/__init__.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/modeling/test_anchor_generator.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/modeling/test_box2box_transform.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/modeling/test_fast_rcnn.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/modeling/test_matcher.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/modeling/test_model_e2e.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/modeling/test_roi_heads.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/modeling/test_roi_pooler.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/modeling/test_rpn.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/structures/__init__.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/structures/test_boxes.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/structures/test_imagelist.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/structures/test_instances.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/structures/test_rotated_boxes.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/test_checkpoint.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/test_config.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/test_engine.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/test_export_caffe2.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/test_model_analysis.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/test_model_zoo.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/test_visualizer.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tools/README.md
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tools/analyze_model.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tools/benchmark.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tools/convert-torchvision-to-d2.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tools/plain_train_net.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tools/train_net.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tools/visualize_data.py
PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tools/visualize_json_results.py
```

### [PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/README.md)

文本SHA256：`8932528fc97cd16e1229a616cd9a08abd35ee0c313ccacdf348c1fad5e277872`

- L21: **精度（Precision）：Mixed**
- L34: https://github.com/facebookresearch/detectron2
- L38: https://gitcode.com/ascend/modelzoo/tree/master/built-in/PyTorch/Research/cv/detection/Faster_Mask_RCNN_for_PyTorch
- L42: 硬件环境准备请参见[各硬件产品文档](https://ascend.huawei.com/#/document?tag=developer)。需要在硬件设备上安装固件与驱动。
- L68: ## 混合精度训练
- L70: 昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。
- L72: ## 开启混合精度
- L73: 在启动脚本中执行训练脚本处配置命令行参数 AMP 1 即可开启NPU上混合精度训练模式。
- L76: ## 数据集准备
- L78: 默认使用coco2017数据集，请用户自行获取。数据集路径通过启动脚本的命令行参数--data_path配置。应有如下目录结构
- L88: wget https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl
- L90: 将其置于数据集所在目录下
- L92: 另附resnet101[下载地址](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl)

### [PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/Detectron1-Comparisons/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/Detectron1-Comparisons/README.md)

文本SHA256：`d8d842df9b215ac0ebbc77bf3f5253e9e907b7517424fd614767b65728bf2890`

- L47: <td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/Detectron1-Comparisons/faster_rcnn_R_50_FPN_noaug_1x/137781054/model_final_7ab50c.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/Detectron1-Comparisons/faster_rcnn_R_50_FPN_noaug_1x/137781054/metrics.json">metrics</a></td>
- L59: <td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/Detectron1-Comparisons/keypoint_rcnn_R_50_FPN_1x/137781195/model_final_cce136.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/Detectron1-Comparisons/keypoint_rcnn_R_50_FPN_1x/137781195/metrics.json">metrics</a></td>
- L71: <td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/Detectron1-Comparisons/mask_rcnn_R_50_FPN_noaug_1x/137781281/model_final_62ca52.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/Detectron1-Comparisons/mask_rcnn_R_50_FPN_noaug_1x/137781281/metrics.json">metrics</a></td>
- L79: [bug](https://github.com/facebookresearch/Detectron/issues/459) lead to a drop in box AP, and can be
- L83: For speed comparison, see [benchmarks](https://detectron2.readthedocs.io/notes/benchmarks.html).

### [PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/quick_schedules/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/configs/quick_schedules/README.md)

文本SHA256：`b13614960848f3902cd5466b260712525eb5f27f3ee2ec39161cbc8cd864fc66`


### [PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/datasets/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/datasets/README.md)

文本SHA256：`604876e312b4f5fb37a01cdfe0022f6965ade6defd87f5b212d67e7bf021eb0e`

- L3: A dataset can be used by accessing [DatasetCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.DatasetCatalog)
- L4: for its data, or [MetadataCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.MetadataCatalog) for its metadata (class names, etc).
- L6: [Use Custom Datasets](https://detectron2.readthedocs.io/tutorials/datasets.html) gives a deeper dive on how to use `DatasetCatalog` and `MetadataCatalog`,
- L24: The [model zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)
- L55: pip install git+https://github.com/cocodataset/panopticapi.git
- L72: pip install git+https://github.com/lvis-dataset/lvis-api.git
- L96: pip install git+https://github.com/mcordts/cityscapesScripts.git

### [PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/demo/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/demo/README.md)

文本SHA256：`babf45da4d65fdf011ddcae91fb3aae448a7542a4eeb5e7abe6b3e33f7002ebd`

- L7: See our [blog post](https://ai.facebook.com/blog/-detectron2-a-pytorch-based-modular-object-detection-library-)

### [PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/data/datasets/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/data/datasets/README.md)

文本SHA256：`c96415f60571b0c72128ece3bd31c2341dfa027a38549bc10bebf6f2c7d746fb`


### [PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/export/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/export/README.md)

文本SHA256：`7d241a9231593b7fd55f7fdc649112089415a19896735e3f33bd9b8262881de8`

- L5: Please see [documentation](https://detectron2.readthedocs.io/tutorials/deployment.html) for its usage.

### [PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/csrc/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/layers/csrc/README.md)

文本SHA256：`371d9f6640807707af998d0ca42eea9926f83cf68f00fe96133391313b16747b`


### [PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/utils/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/detectron2/utils/README.md)

文本SHA256：`9f079e9d79dd855c420c1622dca935b11a3a56ded1250f72eff49722b1b888e8`


### [PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/dev/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/dev/README.md)

文本SHA256：`b965a10ee92c3fa1ece560dff9d958ee7511e3b59fd18aff45057d0e1e3c8e74`


### [PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/dev/packaging/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/dev/packaging/README.md)

文本SHA256：`b5f3fdf992771372c6f326808b95881062ea2580bced1e182cd10881f9ff6c40`

- L7: # git clone https://github.com/facebookresearch/detectron2/

### [PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docker/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docker/README.md)

文本SHA256：`b30693a922b1cfea474fc8bf788b0910e194821903b1703d11ebb64a5981ba3b`


### [PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docs/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docs/README.md)

文本SHA256：`683d0036e363f18e15affcfd2a5839eefeb114c53c93db1f9ee600c66058a121`

- L3: The latest documentation built from this directory is available at [detectron2.readthedocs.io](https://detectron2.readthedocs.io/).

### [PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docs/tutorials/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/docs/tutorials/README.md)

文本SHA256：`6e3bf8de12cb95fde1211c50518e31b814ff8c9954a5f6cf1604dd046ef516bc`

- L3: The latest documentation built from this directory is available at [detectron2.readthedocs.io](https://detectron2.readthedocs.io/).

### [PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tests/README.md)

文本SHA256：`80a35e75f6502a86f3c29b2541994e2a1749e14d4b6c1221c9492d3800eafae4`


### [PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tools/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch/tools/README.md)

文本SHA256：`3bd35959181b603ed437189374a1fa632b5d384ccc56b0d9d82f6cefbd80b31c`


## PyTorch/built-in/others/OpenFold_for_PyTorch

文件清单：

```text
PyTorch/built-in/others/OpenFold_for_PyTorch/.github/dependabot.yml
PyTorch/built-in/others/OpenFold_for_PyTorch/.github/workflows/docker-image.yml
PyTorch/built-in/others/OpenFold_for_PyTorch/.github/workflows/undefined_names.yml
PyTorch/built-in/others/OpenFold_for_PyTorch/.gitignore
PyTorch/built-in/others/OpenFold_for_PyTorch/.readthedocs.yaml
PyTorch/built-in/others/OpenFold_for_PyTorch/2.1_requirements.txt
PyTorch/built-in/others/OpenFold_for_PyTorch/CITATION.cff
PyTorch/built-in/others/OpenFold_for_PyTorch/Dockerfile
PyTorch/built-in/others/OpenFold_for_PyTorch/LICENSE
PyTorch/built-in/others/OpenFold_for_PyTorch/README-ori.md
PyTorch/built-in/others/OpenFold_for_PyTorch/README.md
PyTorch/built-in/others/OpenFold_for_PyTorch/deepspeed_config.json
PyTorch/built-in/others/OpenFold_for_PyTorch/docs/Makefile
PyTorch/built-in/others/OpenFold_for_PyTorch/docs/environment.yml
PyTorch/built-in/others/OpenFold_for_PyTorch/docs/source/Aux_seq_files.md
PyTorch/built-in/others/OpenFold_for_PyTorch/docs/source/FAQ.md
PyTorch/built-in/others/OpenFold_for_PyTorch/docs/source/Inference.md
PyTorch/built-in/others/OpenFold_for_PyTorch/docs/source/Installation.md
PyTorch/built-in/others/OpenFold_for_PyTorch/docs/source/Multimer_Inference.md
PyTorch/built-in/others/OpenFold_for_PyTorch/docs/source/OpenFold_Parameters.md
PyTorch/built-in/others/OpenFold_for_PyTorch/docs/source/OpenFold_Training_Setup.md
PyTorch/built-in/others/OpenFold_for_PyTorch/docs/source/Single_Sequence_Inference.md
PyTorch/built-in/others/OpenFold_for_PyTorch/docs/source/Training_OpenFold.md
PyTorch/built-in/others/OpenFold_for_PyTorch/docs/source/conf.py
PyTorch/built-in/others/OpenFold_for_PyTorch/docs/source/convert_of_v1_weights.md
PyTorch/built-in/others/OpenFold_for_PyTorch/docs/source/index.md
PyTorch/built-in/others/OpenFold_for_PyTorch/docs/source/original_readme.md
PyTorch/built-in/others/OpenFold_for_PyTorch/environment.yml
PyTorch/built-in/others/OpenFold_for_PyTorch/examples/monomer/alignments/6KWC_1/bfd_uniref_hits.a3m
PyTorch/built-in/others/OpenFold_for_PyTorch/examples/monomer/alignments/6KWC_1/hhsearch_output.hhr
PyTorch/built-in/others/OpenFold_for_PyTorch/examples/monomer/alignments/6KWC_1/mgnify_hits.sto
PyTorch/built-in/others/OpenFold_for_PyTorch/examples/monomer/alignments/6KWC_1/uniref90_hits.sto
PyTorch/built-in/others/OpenFold_for_PyTorch/examples/monomer/fasta_dir/6kwc.fasta
PyTorch/built-in/others/OpenFold_for_PyTorch/examples/monomer/inference.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/examples/monomer/sample_predictions/6KWC_1_model_1_ptm_relaxed.pdb
PyTorch/built-in/others/OpenFold_for_PyTorch/examples/monomer/sample_predictions/6KWC_1_model_1_ptm_unrelaxed.pdb
PyTorch/built-in/others/OpenFold_for_PyTorch/experiments/test_templates_openfold.py
PyTorch/built-in/others/OpenFold_for_PyTorch/notebooks/environment.yml
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/__init__.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/config.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/data/__init__.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/data/data_modules.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/data/data_pipeline.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/data/data_transforms.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/data/data_transforms_multimer.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/data/errors.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/data/feature_pipeline.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/data/feature_processing_multimer.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/data/input_pipeline.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/data/input_pipeline_multimer.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/data/mmcif_parsing.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/data/msa_identifiers.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/data/msa_pairing.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/data/parsers.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/data/templates.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/data/tools/__init__.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/data/tools/hhblits.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/data/tools/hhsearch.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/data/tools/hmmbuild.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/data/tools/hmmsearch.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/data/tools/jackhmmer.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/data/tools/kalign.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/data/tools/parse_msa_files.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/data/tools/utils.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/model/__init__.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/model/dropout.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/model/embedders.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/model/evoformer.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/model/heads.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/model/model.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/model/msa.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/model/outer_product_mean.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/model/pair_transition.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/model/primitives.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/model/structure_module.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/model/template.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/model/torchscript.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/model/triangular_attention.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/model/triangular_multiplicative_update.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/np/__init__.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/np/protein.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/np/relax/__init__.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/np/relax/amber_minimize.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/np/relax/cleanup.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/np/relax/relax.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/np/relax/utils.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/np/residue_constants.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/resources/__init__.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/__init__.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/all_atom_multimer.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/argparse_utils.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/callbacks.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/checkpointing.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/chunk_utils.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/exponential_moving_average.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/feats.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/geometry/__init__.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/geometry/quat_rigid.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/geometry/rigid_matrix_vector.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/geometry/rotation_matrix.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/geometry/test_utils.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/geometry/utils.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/geometry/vector.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/import_weights.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/kernel/__init__.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/kernel/attention_core.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/kernel/csrc/compat.h
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/kernel/csrc/softmax_cuda.cpp
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/kernel/csrc/softmax_cuda_kernel.cu
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/kernel/csrc/softmax_cuda_stub.cpp
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/logger.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/loss.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/lr_schedulers.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/multi_chain_permutation.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/precision_utils.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/rigid_utils.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/script_utils.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/superimposition.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/tensor_utils.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/trace_utils.py
PyTorch/built-in/others/OpenFold_for_PyTorch/openfold/utils/validation_metrics.py
PyTorch/built-in/others/OpenFold_for_PyTorch/public_address_statement.md
PyTorch/built-in/others/OpenFold_for_PyTorch/run_pretrained_openfold.py
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/__init__.py
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/activate_conda_env.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/alignment_data_to_fasta.py
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/alignment_db_scripts/add_non_unique_to_alignment_db.py
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/alignment_db_scripts/create_alignment_db.py
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/alignment_db_scripts/create_alignment_db_sharded.py
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/alignment_db_scripts/unify_alignment_db_indices.py
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/build_deepspeed_config.py
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/colabfold_search.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/convert_of_weights_to_jax.py
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/convert_v1_to_v2_weights.py
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/data_dir_to_fasta.py
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/deactivate_conda_env.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/deepspeed_inference_test.py
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/download_alphafold_dbs.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/download_alphafold_params.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/download_bfd.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/download_cameo.py
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/download_colabfold_envdb.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/download_mgnify.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/download_mmseqs_dbs.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/download_openfold_params.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/download_openfold_params_gdrive.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/download_openfold_params_huggingface.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/download_openfold_soloseq_params.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/download_pdb70.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/download_pdb_mmcif.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/download_pdb_seqres.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/download_roda_pdbs.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/download_small_bfd.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/download_soloseq_embeddings.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/download_uniclust30.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/download_uniprot.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/download_uniref30.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/download_uniref90.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/expand_alignment_duplicates.py
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/fasta_to_clusterfile.py
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/flatten_roda.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/generate_alphafold_feature_dict.py
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/generate_chain_data_cache.py
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/generate_mmcif_cache.py
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/install_hh_suite.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/install_third_party_dependencies.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/precompute_alignments.py
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/precompute_alignments_mmseqs.py
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/precompute_embeddings.py
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/prep_mmseqs_dbs.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/prep_proteinnet_msas.py
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/run_unit_tests.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/slurm_scripts/run_uniclust30_search.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/unpack_proteinnet.py
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/utils.py
PyTorch/built-in/others/OpenFold_for_PyTorch/scripts/vars.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/setup.py
PyTorch/built-in/others/OpenFold_for_PyTorch/test/env_npu.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/test/train_openfold_8p.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/test/val_openfold_8p.sh
PyTorch/built-in/others/OpenFold_for_PyTorch/tests/__init__.py
PyTorch/built-in/others/OpenFold_for_PyTorch/tests/compare_utils.py
PyTorch/built-in/others/OpenFold_for_PyTorch/tests/config.py
PyTorch/built-in/others/OpenFold_for_PyTorch/tests/data_utils.py
PyTorch/built-in/others/OpenFold_for_PyTorch/tests/test_data_pipeline.py
PyTorch/built-in/others/OpenFold_for_PyTorch/tests/test_data_transforms.py
PyTorch/built-in/others/OpenFold_for_PyTorch/tests/test_deepspeed_evo_attention.py
PyTorch/built-in/others/OpenFold_for_PyTorch/tests/test_embedders.py
PyTorch/built-in/others/OpenFold_for_PyTorch/tests/test_evoformer.py
PyTorch/built-in/others/OpenFold_for_PyTorch/tests/test_feats.py
PyTorch/built-in/others/OpenFold_for_PyTorch/tests/test_import_weights.py
PyTorch/built-in/others/OpenFold_for_PyTorch/tests/test_kernels.py
PyTorch/built-in/others/OpenFold_for_PyTorch/tests/test_loss.py
PyTorch/built-in/others/OpenFold_for_PyTorch/tests/test_model.py
PyTorch/built-in/others/OpenFold_for_PyTorch/tests/test_msa.py
PyTorch/built-in/others/OpenFold_for_PyTorch/tests/test_multimer_datamodule.py
PyTorch/built-in/others/OpenFold_for_PyTorch/tests/test_outer_product_mean.py
PyTorch/built-in/others/OpenFold_for_PyTorch/tests/test_pair_transition.py
PyTorch/built-in/others/OpenFold_for_PyTorch/tests/test_permutation.py
PyTorch/built-in/others/OpenFold_for_PyTorch/tests/test_primitives.py
PyTorch/built-in/others/OpenFold_for_PyTorch/tests/test_structure_module.py
PyTorch/built-in/others/OpenFold_for_PyTorch/tests/test_template.py
PyTorch/built-in/others/OpenFold_for_PyTorch/tests/test_triangular_attention.py
PyTorch/built-in/others/OpenFold_for_PyTorch/tests/test_triangular_multiplicative_update.py
PyTorch/built-in/others/OpenFold_for_PyTorch/tests/test_utils.py
PyTorch/built-in/others/OpenFold_for_PyTorch/thread_sequence.py
PyTorch/built-in/others/OpenFold_for_PyTorch/train_openfold.py
```

### [PyTorch/built-in/others/OpenFold_for_PyTorch/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/PyTorch/built-in/others/OpenFold_for_PyTorch/README.md)

文本SHA256：`9f83e1ee5ab2464b4203d3fac0b5e94b8b395ca0e8248a70cd5079e0b513cb7b`

- L7: - [准备数据集](#准备数据集)
- L17: 在生物学中，结构和功能密不可分。因此，理解生物系统的机制、工程设计及其影响方式，就意味着需要了解和理解它们的结构。该联盟正在开发基于人工智能的先进蛋白质建模工具，能够以原子级精度预测分子结构，首次以开源形式将这种精度水平用于研究和商业应用。世界各地的研究人员将能够使用、改进并贡献这种“预测分子显微镜”。
- L22: url=https://github.com/aqlaboratory/openfold.git
- L29: url=https://gitcode.com/ascend/ModelZoo-PyTorch
- L43: |  PyTorch 2.1  | python=3.9 |
- L47: 请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
- L60: pip install git+https://github.com/NVIDIA/dllogger.git
- L63: wget https://github.com/soedinglab/hh-suite/releases/download/v3.3.0/hhsuite-3.3.0-AVX2-Linux.tar.gz
- L67: pip install git+https://github.com/TimoLassmann/kalign.git
- L69: wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
- L81: ## 准备数据集
- L87: curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"
- L92: 2. 获取数据集。
- L158: - 8卡验证精度
- L171: --data_path                              //训练数据集路径，必填
- L177: 训练完成后，权重文件保存在output/checkpoints下，并输出模型训练时间。
- L179: 接着运行验证脚本，输出模型训练后的精度验证数据。
- L187: | 8p-竞品A  | bf16 |  1:38:03   |  78.30   |    2.1 |
- L188: | 8p-Atlas 900 A2 PODc | bf16 |  2:37:31   |  79.32   |      2.1      |

## 补充证据 [ACL_PyTorch/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/README.md)

- L9: 在开始贡献之前，请先阅读[CONTRIBUTING](https://gitcode.com/ascend/modelzoo/blob/master/CONTRIBUTING.md)。
- L26: <td rowspan="2">《 <a href="https://www.hiascend.com/document/detail/zh/canncommercial/900/softwareinst/instg/instg_0005.html">驱动固件安装指南</a> 》</td>
- L33: <td rowspan="3">《 <a href="https://www.hiascend.com/document/detail/zh/canncommercial/900/softwareinst/instg/instg_0000.html">CANN 软件安装指南</a> 》</td>
- L43: <td rowspan="3">《 <a href="https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md">Ascend Extension for PyTorch 配置与安装</a> 》</td>
- L53: torch_npu插件版本配套表地址：[链接](https://gitcode.com/ascend/pytorch#%E6%98%87%E8%85%BE%E8%BE%85%E5%8A%A9%E8%BD%AF%E4%BB%B6)。
- L60: | `torch-onnx-inference:cann8.3.rc1_torch2.1.0-300I-DUO-ubuntu22.04-py3.11-aarch64` | Ubuntu 22.04 | Atlas 300I DUO | `ModelZoo-PyTorch/ACL_PyTorch/docker/cann8.3.rc1_torch2.1.0/Dockerfile.300IDUO.ubuntu` |
- L61: | `torch-onnx-inference:cann8.3.rc1_torch2.1.0-300I-DUO-openeuler24.03-py3.11-aarch64` | openEuler 24.03 | Atlas 300I DUO | `ModelZoo-PyTorch/ACL_PyTorch/docker/cann8.3.rc1_torch2.1.0/Dockerfile.300IDUO.openeuler` |
- L62: | `torch-onnx-inference:cann8.3.rc1_torch2.1.0-800I-A2-ubuntu22.04-py3.11-aarch64` | Ubuntu 22.04 | Atlas 800I A2 | `ModelZoo-PyTorch/ACL_PyTorch/docker/cann8.3.rc1_torch2.1.0/Dockerfile.800I_A2.ubuntu` |
- L63: | `torch-onnx-inference:cann8.3.rc1_torch2.1.0-800I-A2-openeuler24.03-py3.11-aarch64` | openEuler 24.03 | Atlas 800I A2 | `ModelZoo-PyTorch/ACL_PyTorch/docker/cann8.3.rc1_torch2.1.0/Dockerfile.800I_A2.openeuler` |
- L66: 具体介绍及使用方式可以参考：[OVERVIEW](https://gitcode.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/docker/OVERVIEW.zh.md)
- L70: 如果您希望快速了解ACL_PyTorch中的模型使用方式，推荐先体验 [YOLOV26](https://gitcode.com/Ascend/modelzoo-GPL/blob/master/built-in/ACL_Pytorch/Yolov26_for_PyTorch) 。该用例以目标检测任务为例，包含环境准备、模型转换（PyTorch模型导出为ONNX，再使用ATC工具转换为昇腾离线模型）、数据集准备以及离线推理脚本示例，适合作为首次上手的参考。
- L75: 2. 进入 [YOLOV26](https://gitcode.com/Ascend/modelzoo-GPL/blob/master/built-in/ACL_Pytorch/Yolov26_for_PyTorch)目录，阅读该用例下的 `README.md`。
- L97: 💡 除了ACL_PyTorch，Ascend-SACT组织中也存在很多推理模型，可以在[Ascend-SACT](https://gitcode.com/Ascend-SACT)中查询。
- L128: - [YOLOV8](https://gitcode.com/Ascend/modelzoo-GPL/blob/master/built-in/ACL_Pytorch/Yolov8_for_PyTorch/README.md)
- L129: - [YOLOV11](https://gitcode.com/Ascend/modelzoo-GPL/blob/master/built-in/ACL_Pytorch/Yolov11_for_PyTorch/README.md)
- L130: - [YOLOV26](https://gitcode.com/Ascend/modelzoo-GPL/blob/master/built-in/ACL_Pytorch/Yolov26_for_PyTorch/README.md)
- L131: - [SAM2](https://gitcode.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/SAM2)
- L132: - [SAM3](https://gitcode.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/SAM3)
- L136: - [CosyVoice2](https://gitcode.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/audio/CosyVoice2)
- L137: - [CosyVoice3](https://gitcode.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/audio/CosyVoice3)
- L138: - [Paraformer](https://gitcode.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/audio/Paraformer)
- L139: - [Whisper](https://gitcode.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/audio/whisper)
- L140: - [Index-TTS](https://gitcode.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/audio/Index-TTS-vLLM-v1)
- L141: - [Index-TTS2](https://gitcode.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/audio/Index-TTS-vLLM-v2)
- L145: - [Qwen3-Embedding](https://docs.vllm.ai/projects/vllm-ascend-cn/zh-cn/latest/tutorials/models/Qwen3_embedding.html)
- L146: - [Qwen3-Reranker](https://docs.vllm.ai/projects/vllm-ascend-cn/zh-cn/latest/tutorials/models/Qwen3_reranker.html)
- L150: - [PaddleOCR-VL](https://gitcode.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/ocr/PaddleOCR-VL)
- L151: - [PaddleOCR-VL-1.5](https://gitcode.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/ocr/PaddleOCR-VL-1.5)
- L152: - [PPOCRv5](https://gitcode.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/ocr/PPOCRv5)
- L153: - [DeepseekOCR2](https://docs.vllm.ai/projects/vllm-ascend-cn/zh-cn/latest/tutorials/models/DeepSeekOCR2.html)
- L157: - [pi0](https://gitcode.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/embodied_ai/vla/pi0)
- L158: - [pi0.5](https://gitcode.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/embodied_ai/vla/pi05)
- L159: - [GraspNet](https://gitcode.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/embodied_ai/GraspNet)
- L160: - [GenPosePlus](https://gitcode.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/embodied_ai/GenPosePlus)
- L161: - [GR00T N1.7](https://gitcode.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/embodied_ai/IsaacGR00T)
- L182: | 用户主目录                        |   750（rwxr-x---）            |
- L183: | 程序文件(含脚本文件、库文件等)       |   550（r-xr-x---）             |
- L184: | 程序文件目录                      |   550（r-xr-x---）            |
- L185: | 配置文件                          |  640（rw-r-----）             |
- L186: | 配置文件目录                      |   750（rwxr-x---）            |
- L187: | 日志文件(记录完毕或者已经归档)        |  440（r--r-----）             |
- L188: | 日志文件(正在记录)                |    640（rw-r-----）           |
- L189: | 日志文件目录                      |   750（rwxr-x---）            |
- L190: | Debug文件                         |  640（rw-r-----）         |
- L191: | Debug文件目录                     |   750（rwxr-x---）  |
- L192: | 临时文件目录                      |   750（rwxr-x---）   |
- L193: | 维护升级文件目录                  |   770（rwxrwx---）    |
- L194: | 业务数据文件                      |   640（rw-r-----）    |
- L195: | 业务数据文件目录                  |   750（rwxr-x---）      |
- L196: | 密钥组件、私钥、证书、密文文件目录    |  700（rwx—----）      |
- L197: | 密钥组件、私钥、证书、加密密文        | 600（rw-------）      |
- L198: | 加解密接口、加解密脚本            |   500（r-x------）        |

## 补充证据 [ACL_PyTorch/contrib/cv/detection/Mask_Rcnn_SwinS/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/contrib/cv/detection/Mask_Rcnn_SwinS/README.md)

- L22: url=https://github.com/open-mmlab/mmdetectinon
- L33: | input    | 1 x 3 x 800 x 1216         | RGB_FP32 |
- L40: | dets     | 100 x 5       | FLOAT32  |
- L41: | labels   | 100           | FLOAT32  |
- L42: | mask     | 100 x 28 x 28 | FLOAT32  |
- L53: | 固件与驱动                                                      | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
- L54: | CANN                                                            | 6.0.RC1 | -                                                                                                     |
- L55: | Python                                                          | 3.7.5   | -                                                                                                     |
- L56: | PyTorch                                                         | 1.8.0   | -                                                                                                     |
- L57: | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                                                                     |
- L74: git clone https://github.com/open-mmlab/mmcv
- L78: git clone https://github.com/open-mmlab/mmdetection
- L129: 使用训练好的pth权重文件：[mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth](https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth)
- L139: mmdetection/configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
- L140: mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth \
- L161: >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。
- L173: | 0       310P3     | OK              | 15.8         42                0    / 0              |
- L174: | 0       0         | 0000:82:00.0    | 0            1074 / 21534                            |
- L176: | 1       310P3     | OK              | 15.4         43                0    / 0              |
- L177: | 0       1         | 0000:89:00.0    | 0            1070 / 21534                            |
- L213: 请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。
- L262: | 模型                         | 官网pth精度(IoU=0.50:0.95/area=all/maxDets=100) | 300I PRO推理精度(IoU=0.50:0.95/area=all/maxDets=100) | 300I PRO性能 | T4性能 |
- L264: | Mask-Rcnn-SwinS(bs1)         | bboxmap: 48.2                                   | bboxmap: 47.8                                    | 5.17     | 2.5    |

## 补充证据 [ACL_PyTorch/built-in/embodied_ai/GenPosePlus/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/built-in/embodied_ai/GenPosePlus/README.md)

- L4: GenPose++ 采用分段点云和裁剪后的 RGB 图像作为输入，利用 PointNet++ 提取物体的几何特征；同时借助预训练的 2D 基础骨干网络 DINO v2 提取通用语义特征。随后，将这些特征融合，作为扩散模型的条件，生成物体姿态候选及其对应的能量。最后，针对具有非连续对称性（如盒子）的物体，通过聚类解决姿态多模态分布带来的聚集问题，从而有效完成姿态估计。
- L6: > [300V PRO (310P3)](#300v-pro) | [310P RC (310P1)](#310p-rc)
- L18: | 固件与驱动                                                       | 24.1.RC3    | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
- L19: | CANN                                                            | 9.0.1       | 包含kernels包和toolkit包                                                                                                   |
- L20: | Python                                                          | 3.12        | -                                                                                                     |
- L21: | PyTorch                                                         | 2.10.0      | -                                                                                                     |
- L22: | Ascend Extension PyTorch (torch_npu)                            | 2.10.0.post2| -                                                                                                     |
- L23: | 说明：支持Atlas 300V PRO | \           | \                                                                                                     |
- L45: ais_bench安装请参考[ais_bench安装指导](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)
- L53: git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
- L56: git clone https://github.com/Omni6DPose/GenPose2.git
- L71: conda install -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge opencv -y --override-channels
- L76: ### 下载 DINOv2
- L79: cd $GPPATH && git clone https://github.com/facebookresearch/dinov2.git
- L81: wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth \
- L102: 请按照 [Omni6DPoseAPI](https://github.com/Omni6DPose/Omni6DPoseAPI)页面上的说明，下载并整理 Omni6DPose 数据集中的`ROPE`分类中的`000000`号数据集，并在项目目录中组织如下
- L127: - 模型可在如下连接找到 [checkpoints](https://www.dropbox.com/scl/fo/x87lhf7sygjm1gasz153g/AIHBlaGMjhfyW1bKrDe61R4?rlkey=y1f6dqdi40tzcgepccthayudp&st=1sbkxbzf&dl=0). 请下载至如下路径 `$GPPATH/results` 并组织如下:
- L146: | 推理路径 | 触发条件 | PointNet2 实现 |
- L150: | **ctypes** | `.pth` + `POINTNET2_BACKEND=ctypes` | kernel_ctypes host 库直调 |
- L151: | **torch_native** | `.pth` + `POINTNET2_BACKEND=torch_native` | 纯 PyTorch `pointnet2_ops.py`（零依赖 fallback，NPU 上部分算子可能不兼容） |
- L198: |  300V PRO(单芯)| OM | 4  |        0.3056     | 134.35 |
- L199: |  300V PRO(单芯)| torch_ops | 16  |        0.2981     | 163.63 |
- L200: |  300V PRO(单芯)| ctypes | 16  |        0.2983     | 190.96 |
- L210: | 固件与驱动 | 24.1.RC3 |
- L211: | CANN | 8.3.RC1 |
- L212: | Python | 3.10 |
- L213: | PyTorch | 2.1.0 |
- L214: | Ascend Extension PyTorch (torch_npu) | 2.1.0.post17 |
- L221: git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
- L224: git clone https://github.com/Omni6DPose/GenPose2.git
- L295: | 310P RC | OM head + CPU pointnet2 | 4 | 0.3057 | 251.84 |

## 补充证据 [ACL_PyTorch/built-in/cv/RF-DETR/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/built-in/cv/RF-DETR/README.md)

- L5: RF-DETR（Real-Time SOTA Object Detection, Instance Segmentation, and Keypoint Detection）是 Roboflow 开发的用于目标检测、实例分割和关键点检测（预览）的实时 Transformer 架构。基于 DINOv2 视觉 Transformer 主干，RF-DETR 在 [Microsoft COCO](https://cocodataset.org/#home) 和 [RF100-VL](https://github.com/roboflow/rf100-vl) 上实现了最先进的准确性和延迟权衡。
- L12: git clone https://github.com/roboflow/rf-detr
- L30: https://gitcode.com/ascend/ModelZoo-PyTorch
- L45: |  1   | infer_coco.py  |    使用模型进行推理     |
- L46: |  2   | export_onnx.py | 根据pth文件导出onnx文件 |
- L47: |  3   |  infer_om.py   |   使用om文件进行推理    |
- L48: |  4   |  eval_coco.py  |        精度验证         |
- L59: Detection: RF_DETR_SMALL:https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth
- L60: Segmentation: RF_DETR_SEG_SMALL:https://storage.googleapis.com/rfdetr/rf-detr-seg-s-ft.pth
- L66: curl -O https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth
- L68: curl -O https://storage.googleapis.com/rfdetr/rf-detr-seg-s-ft.pth
- L130: - --soc_version：处理器型号。可通过npu-smi info查看，也可以参考soc信息[https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/devaids/devtools/atc/atlasatcparam_16_0036.html](https://link.gitcode.com/?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2Fcanncommercial%2F80RC3%2Fdevaids%2Fdevtools%2Fatc%2Fatlasatcparam_16_0036.html&from=https%3A%2F%2Fgitcode.com%2FAscend%2FModelZoo-PyTorch%2Fblob%2Fmaster%2FACL_PyTorch%2Fbuilt-in%2Fcv%2FD-FINE%2FREADME.md%3Finit%3DinitTree&lang=zh&theme=white)
- L149: 请访问[ais_bench推理工具]([ais-bench_workload · Ascend/tools - 码云 - 开源中国](https://gitee.com/ascend/tools/tree/master/ais-bench_workload))代码仓，根据readme文档进行工具安装。
- L177: |     指标      |   CPU   | NPU(atlas 800I) |   差值   |  差值%  |
- L179: | **mAP@50:95** | 0.52466 |  0.52486   | +0.00020 | +0.038% |
- L180: |    mAP@50     | 0.71339 |  0.71380   | +0.00041 | +0.057% |
- L181: |    mAP@75     | 0.56593 |  0.56651   | +0.00058 | +0.102% |
- L182: |      mAR      | 0.70487 |  0.70537   | +0.00050 | +0.071% |
- L183: |      F1       | 0.69809 |  0.69884   | +0.00075 | +0.107% |
- L184: |   Precision   | 0.76810 |  0.76917   | +0.00107 | +0.140% |
- L185: |    Recall     | 0.64830 |  0.64879   | +0.00049 | +0.075% |

## 补充证据 [ACL_PyTorch/built-in/cv/EoMT/README.md](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/ACL_PyTorch/built-in/cv/EoMT/README.md)

- L16: | 固件与驱动 | 25.2.0（与 CANN 8.3.RC1 配套） | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
- L17: | CANN       | 8.3.RC1                 | -                                                            |
- L18: | Python     | 3.11.15                 | -                                                            |
- L19: | PyTorch    | 2.1.0           | -                                              |
- L20: | torch_npu  | 2.1.0.post17 | -                                       |
- L64: git clone https://github.com/tue-mps/eomt.git
- L94: 打开[网址](https://huggingface.co/tue-mps/coco_instance_eomt_large_640/resolve/main/pytorch_model.bin)，下载`coco_instance_eomt_large_640`配套的权重到 `$work_dir/weights/coco_instance_eomt_large_640` 目录：
- L103: 3. 下载 [COCO 数据集](http://images.cocodataset.org/zips/val2017.zip)
- L167: |   RTX 5060    | 76.42ms  | GPU推理 |
- L168: | Atlas 800I A2 | 40.46ms  | NPU推理 |
- L174: |  指标   | Atlas 800I A2 | RTX 5060 |
- L176: | mask-AP |     48.91     |  48.90   |
