"""Reviewed GPU findings from primary sources, 2026-10-23. Not runtime tests."""
def gh(repo,path='README.md',ref=None):
    defaults={'DebeshJha/ResUNetPlusPlus':'master','FlagOpen/FlagEmbedding':'master',
              'Open-X-Humanoid/pelican-vl':'rxc','OpenHero/gblastn':'master',
              'PaddlePaddle/PaddleSeg':'release/2.10','fundamentalvision/BEVFormer':'master',
              'justchenhao/BIT_CD':'master','oxwhirl/pymarl':'master',
              'pangfumin/CudaDwa':'master','starVLA/starVLA':'starVLA_dev'}
    ref=ref or defaults.get(repo,'main')
    return f'https://github.com/{repo}/blob/{ref}/{path}'

def hf(name,data=False):
    return 'https://huggingface.co/'+('datasets/' if data else '')+name

def apply(rows):
    fields=['代码','权重','数据集','性能基准','精度基准']
    def put(n,src,code,weight,data,perf='数值待核对',accuracy='数值待核对',links=None,status='支持GPU，有官方实现'):
        r=rows[n-1]
        r['GPU结论']=status
        r['GPU查询状态']='已查原始资料'
        r['设备维度']['GPU']={k:{'说明':text,'链接':(links or {}).get(k,[src]),'证据类型':'本轮已读原始资料；非本次实测'} for k,text in zip(fields,[code,weight,data,perf,accuracy])}
    for r in rows:
        r['GPU查询状态']='尚未完成专项查询'
        for v in r['设备维度']['GPU'].values():
            if v['说明'].startswith('未取得明确GPU'):v['说明']='待查询'
    put(1,gh('yuantianyuan01/FastWAM'),'官方CUDA训练、LIBERO/RoboTwin评估','yuanty/fastwam，含Optional IDM变体','LIBERO-fastwam、robotwin2.0-fastwam','官方更新：H20 470→210ms；RTX4090 190→110ms（端到端）','任务成功率需另核，不能借用NPU的200次结果',{'权重':[hf('yuanty/fastwam')],'数据集':[hf('yuanty/LIBERO-fastwam',True),hf('yuanty/robotwin2.0-fastwam',True)]})
    put(2,gh('chenjinyubuaa/AerialVLA'),'AeroVLA官方代码，提供单/多GPU评估','原仓库模型准备说明','原仓库训练/评估数据准备')
    for n,repo,w,d,code in [
      (4,'nvidia-cosmos/cosmos-predict2','Predict2 2B/14B各变体','Cosmos-NeMo-Assets、Bridge等按任务区分','明确Ampere及更新NVIDIA GPU、CUDA12.6'),
      (5,'nvidia-cosmos/cosmos-transfer1','Transfer1-7B及各控制组件','条件视频；AV任务另需地图/LiDAR','官方单/多GPU推理与后训练脚本'),
      (6,'nvidia-cosmos/cosmos-reason1','Reason1-7B','官方SFT/RL数据','官方GPU推理，要求至少24GB显存'),
      (7,'nv-tlabs/Cosmos-Drive-Dreams','AV、多视图、LiDAR各组件','PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams','官方GPU生成与预处理流水线')]:put(n,gh(repo),code,w,d)
    put(8,'https://github.com/nvidia-cosmos','GPU模型系列，分别见Predict2/Transfer1/Reason1','按成员下载','按成员和任务准备','无系列统一性能','无系列统一精度',status='系列名称，部分成员有GPU实现')
    put(9,gh('OpenGalaxea/GalaxeaVLA'),'G0.5官方CUDA12.8训练及推理','G05：base、LIBERO、DROID、RoboTwin等','Galaxea Open-World；任务评估配置分别提供','推理显存>8GB；全量微调>70GB，属于硬件要求，非吞吐','官方汇总：LIBERO98.9%、RoboTwin2 93.3%、DROID82.5%；各自任务协议',{'权重':[hf('OpenGalaxea/G05')],'数据集':[hf('OpenGalaxea/Galaxea-Open-World-Dataset',True)]})
    legacy=gh('OpenGalaxea/GalaxeaVLA',ref='13a16a9049aee8f1d799b56fccc0c5832a75fc2f')
    for n,label in [(10,'G0'),(11,'G0Tiny'),(12,'G0Plus')]:
        put(n,legacy,f'{label}历史发布代码，不能使用当前G0.5配置代替','G0-VLA历史checkpoint','Galaxea Open-World','Orin TensorRT最多10Hz（仅G0Tiny）' if n==11 else '数值待核对',links={'权重':[hf('OpenGalaxea/G0-VLA')]})
    for n,ref,weight,perf in [(13,'4af2b622892f7dcb5aae5a3fb70bcb02dc217b96','GR00T-N1.5-3B','H100完整模型47.88ms，4次去噪'),(14,'ead52833afbbf4243f8cd5e7664f48a94de03b19','GR00T-N1.6-3B','torch.compile：RTX4090 44ms/22.8Hz；H100 38ms/26.3Hz'),(15,'755876a9afdb41ca6eb6383b36f4a0adb085c73f','GR00T-N1-2B','L40完整模型62.53ms，4次去噪')]:
        put(n,gh('NVIDIA/Isaac-GR00T',ref=ref),'对应历史版本官方CUDA训练/推理；不借用N1.7结果',weight,'官方demo_data及PhysicalAI数据；正式任务各自准备',perf,links={'权重':[hf('nvidia/'+weight)]})
    put(16,gh('open-gigaai/giga-world-1'),'官方CUDA训练、推理、数据处理','Stage1 Nano/Pro已发；Stage2蒸馏权重待发','Toydata、CVPR WorldModel Track数据','8×H20/A100是参考配置；10FPS为生成视频帧率，非推理吞吐',links={'权重':[hf('open-gigaai/Giga-World-1')],'数据集':[hf('open-gigaai/Giga-World-1-Toydata',True)]})
    put(17,gh('open-gigaai/giga-brain-0'),'0.7有CUDA离线评估、GPU推理服务','0.7-3.5B Base、RoboTwin2-Clean、EBench','GigaBrain-0.7-SampleData',links={'权重':[hf('open-gigaai/GigaBrain-0.7-3.5B-Base')],'数据集':[hf('open-gigaai/GigaBrain-0.7-SampleData',True)]})
    put(18,gh('open-gigaai/giga-world-0'),'官方单/多GPU生成及LoRA路径','Video-Pretrain-2b、Video-GR1-2b','GR1及原始数据处理',links={'权重':[hf('open-gigaai/GigaWorld-0-Video-Pretrain-2b')],'数据集':[hf('nvidia/PhysicalAI-Robotics-GR00T-GR1',True)]})
    put(19,gh('Robbyant/lingbot-vla','deploy/lingbot_vla_policy.py'),'部署代码明确device=cuda及policy.cuda()','4B、4B-Depth；辅助Qwen/MoGe/Depth分别准备','RoboTwin/真实机器人LeRobot数据',links={'权重':[hf('robbyant/lingbot-vla-4b'),hf('robbyant/lingbot-vla-4b-depth')]})
    put(20,gh('openvla/openvla'),'官方CUDA推理、LoRA及完整微调','openvla-7b；完整微调用prismatic版本','BridgeData V2、Open-X混合','单A100-80GB是LoRA示例配置，非速度','任务指标需按模型、机器人和数据协议核对',{'权重':[hf('openvla/openvla-7b'),hf('openvla/openvla-7b-prismatic')],'数据集':['https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/']})
    put(21,gh('Open-X-Humanoid/pelican-vl'),'官方CUDA多卡训练命令','Pelican1.0-VL 3B/7B/72B','RoboVQA、RoboPoint、VSI-Bench数据准备',links={'权重':[hf('X-Humanoid/Pelican1.0-VL-7B')],'数据集':[hf('nvidia/Cosmos-Reason1-SFT-Dataset',True),hf('nyu-visionx/VSI-Bench',True)]})
    for n,model in [(22,'pi05'),(23,'pi0')]:put(n,gh('Physical-Intelligence/openpi'),'官方NVIDIA GPU推理/微调（JAX及PyTorch路线需区分）',model+' base及任务微调权重清单','LIBERO、DROID及自有机器人LeRobot数据','推理>8GB、LoRA>22.5GB、完整微调>70GB为显存要求',links={'权重':[gh('Physical-Intelligence/openpi')],'数据集':[hf('openvla/modified_libero_rlds',True)]})
    put(26,gh('Spirit-AI-Team/spirit-v1.5'),'官方A100-80GB测试；FSDP多GPU训练','Spirit-v1.5及move-objects-into-box微调版','RoboChallenge Table30对应任务','未在README定位固定条件速度','官方声明2026-01-11 Table30第一；未将排名写成成功率',{'权重':[hf('Spirit-AI-robotics/Spirit-v1.5')],'数据集':[hf('RoboChallenge/task_table30_move_objects_into_box',True)]})
    put(27,gh('starVLA/starVLA'),'官方多GPU训练；小模型示例可单A100','model_zoo含各架构任务权重','LIBERO、RoboTwin、SimplerEnv等','数值待按架构核对','README有LIBERO结果图，不能混用Qwen/Florence及其他架构',{'权重':[gh('starVLA/starVLA','docs/model_zoo.md')],'数据集':[hf('StarVLA/RoboTwin-Clean',True)]})
    put(28,gh('UBTECH-Robot/Thinker','inference.py'),'官方推理脚本支持device_map=cuda:0','Thinker-4B已公开；Thinking版仍待发布','原仓库评测入口，训练集公开范围待核','数值待核对','官方七项具身基准结果需按模型表逐项核对',{'权重':[hf('UBTECH-Robotics/Thinker-4B')]})
    put(30,gh('UBTECH-Robot/Thinker-WM'),'官方单NVIDIA GPU六种推理模式；flash-attn为必需','Thinker-WM；VAE和tokenizer/config另备','WorldArena、EWMBench、RoboTwin等评测','数值待核对','原表WorldArena EWMScore71.73；其余任务见原表',{'权重':[hf('UBTECH-Robotics/Thinker-WM')]})
    put(31,gh('X-Square-Robot/wall-x'),'0.5官方CUDA训练、服务和评估','wall-oss-0.5；旧FLOW/FAST属于其他版本','LeRobot数据；LIBERO及开环评估','数值待核对','任务结果待核对',{'权重':[hf('x-square-robot/wall-oss-0.5')]})
    put(34,gh('X-Square-Robot/wall-wm'),'官方CUDA算子、训练、RoboTwin多GPU评估','README明确预训练checkpoint Coming Soon','包含开环样例；RoboTwin另准备','数值待核对','有评测入口；独立数值待核对')
    put(35,gh('X-Square-Robot/wall-ss'),'已找到官方论文及代码仓；GPU执行文件仍待逐个定位','公开checkpoint未定位','600组sim-real评估记录；完整数据可获取性待核','数值待核对','官方报告MAE0.062、相关系数0.93；硬件未与该表绑定',status='已找到项目，GPU执行细节待核')
    put(37,gh('EmbodiedCity/Worldscape-MoE.code'),'官方CUDA训练；8卡/64卡脚本','Worldscape-MoE','公开数据子集','H20预计算latent使每步耗时约减50%；不含准备成本','数值待核对',{'权重':[hf('EmbodiedCity/Worldscape-MoE')],'数据集':[hf('EmbodiedCity/Worldscape-MoE-Dataset',True)]})
    put(38,gh('Open-X-Humanoid/XR-1'),'官方训练部署代码；README未充分定位GPU执行段','Stage1 UVMC、Stage2','XR-1-Dataset-Sample','数值待核对','数值待核对',{'权重':['https://huggingface.co/collections/X-Humanoid/xr-1'],'数据集':[hf('X-Humanoid/XR-1-Dataset-Sample',True)]},status='资源已公开，GPU执行细节待核')
    for n,model in [(40,'bge-m3'),(41,'bge-reranker-v2-m3')]:put(n,hf('BAAI/'+model),'FlagEmbedding官方模型卡及推理代码；具体GPU实现见库','BAAI/'+model,'MIRACL、MLDR、MKQA等按模型区分','数值待核对','BGE-M3的MIRACL结果2024-07已修正；不得使用旧删除query协议',{'权重':[hf('BAAI/'+model)],'代码':[gh('FlagOpen/FlagEmbedding')]},status='官方实现及模型公开；GPU配置需按库核对')
    for n,kind in [(42,'Embedding'),(43,'Reranker'),(44,'Embedding'),(45,'Reranker')]:put(n,gh('QwenLM/Qwen3-Embedding'),'官方示例明确.cuda()；Embedding及Reranker分开','Qwen3-'+kind+' 0.6B/4B/8B','MTEB Multilingual及检索/重排基准','数值待核对','Embedding-8B官方MTEB Multilingual70.58（2025-06-05）；非重排结果' if kind=='Embedding' else '原仓库有重排结果表；数值待逐任务核对',{'权重':[hf('Qwen/Qwen3-'+kind+'-8B')]})
    for n in [50,51,52]:put(n,'https://manual.mixblup.eu/','MiXBLUP的hpblup可调用MIRACULIX_GPU；仅部分矩阵/求解加速','不需要预训练权重','基因型、表型、系谱；按模型配置','无统一端到端基准','不能将GPU矩阵核支持写成所有BLUP/ssGBLUP完整验证',{'代码':['https://github.com/alexfreudenberg/miraculix','https://manual.mixblup.eu/']},status='有GPU加速路线，限指定求解器/模块')
    put(49,gh('OpenHero/gblastn'),'G-BLASTN社区CUDA10.1实现；不同于NCBI官方原版','不需要权重','核酸query及数据库','数值待按原论文核对','兼容范围与比对一致性待核',status='有GPU实现（社区，旧版）')
    put(54,gh('RosettaCommons/DeepAb','predict.py'),'--use_gpu选择CUDA；PyRosetta结构优化另算','ensemble_abresnet_v1','仓库data样例及论文评估','数值待核对','结构精度需计入PyRosetta流程',{'权重':['https://data.graylab.jhu.edu/ensemble_abresnet_v1.tar.gz']})
    put(55,gh('google-deepmind/alphafold3'),'官方模型推理明确需要GPU；数据流水线可CPU','官方参数需申请授权','MSA/模板数据库及结构测试','数值待按序列长度/硬件核对','原始AF3结构基准；不得以其他复现版代替')
    put(56,'https://github.com/google-deepmind/alphafold','官方Multimer路线；GPU部署配置','Multimer参数下载','MSA/模板数据库及复合物输入','数值待核对','对应Multimer论文/模型评测需单列')
    put(57,gh('Biohub/esm','esm/models/esm3.py'),'ESM3源码明确cuda设备选择；原仓库已迁移Biohub','ESM3开放小模型，需核对许可/访问权限','依生成任务；不是ESM2/ESMFold2基准','数值待核对','不能把ESMC/ESMFold2指标填给ESM3')
    put(58,gh('facebookresearch/esm'),'官方ESM2 GPU推理、大模型CPU offload','ESM2各规模模型下载表','UR50/D及对应下游评测','数值待按规模/长度核对','ESM-IF/ESMFold指标不属于ESM2',{'权重':[hf('facebook/esm2_t33_650M_UR50D')]})
    put(59,gh('dauparas/LigandMPNN','run.py'),'run.py明确CUDA设备及model.to(device)','get_model_params.sh及各噪声/模型变体','示例PDB及论文测试集','数值待核对','sequence recovery需按测试集/重设计区域核对')
    put(60,'https://openvax.github.io/mhcflurry/commandline_tools.html','官方命令支持gpu后端；不是NPU','官方model_downloads提供版本化模型','官方训练/评估数据说明','数值待核对','2.3.0模型评估另有官方页面',{'权重':['https://openvax.github.io/mhcflurry/model_downloads.html'],'精度基准':['https://openvax.github.io/mhcflurry/release_model_evaluation.html']})
    put(61,gh('dauparas/ProteinMPNN'),'官方CUDA11.3安装及GPU batch说明','仓库vanilla/soluble/ca_model_weights','PDB训练数据；示例sequence recovery不是正式基准','数值待核对','样例恢复率不能当全测试集指标',{'权重':[gh('dauparas/ProteinMPNN','vanilla_model_weights/v_48_020.pt')]})
    put(62,gh('RosettaCommons/RFdiffusion2'),'官方GPU设计pipeline；Chai1有GPU架构限制','官方权重/容器准备','demo及AME enzyme benchmark','RTX2060单case最多约10分钟，原文估计非统一基准','完整pipeline应含LigandMPNN及Chai1')
    put(63,gh('RosettaCommons/RFdiffusion'),'官方CUDA11.1环境及GPU容器','官方多任务checkpoint','设计输入PDB及任务基准','数值待核对','按无条件/约束/binder任务分开')
    put(64,gh('RosettaCommons/RoseTTAFold'),'官方CUDA10/11安装配置','weights.tar.gz；非商用许可','MSA/模板数据库','数值待核对','第一代模型，不能借用RF2结果',{'权重':['https://files.ipd.uw.edu/pub/RoseTTAFold/weights.tar.gz']})
    put(65,gh('Kuhlman-Lab/ThermoMPNN'),'官方CUDA训练，明确检查torch.cuda','仓库models目录','MegaScale/FireProt等','单V100：小数据<30s/epoch；大数据8–12min/epoch','任务质量待按官方测试split核对',{'权重':['https://github.com/Kuhlman-Lab/ThermoMPNN/tree/main/models']})
    for n,repo in [(66,'justchenhao/BIT_CD'),(67,'wgcban/ChangeFormer')]:put(n,gh(repo,'models/networks.py'),'明确CUDA检查、网络转GPU及DataParallel','README提供预训练checkpoint','LEVIR-CD；ChangeFormer另有DSIFN','数值待核对','原仓库结果图需按变体读取',{'权重':[gh(repo)],'数据集':[gh(repo)]})
    put(68,gh('open-mmlab/mmsegmentation','configs/deeplabv3plus/README.md'),'MMSeg DeepLabV3+对应配置','原表模型文件及训练日志','Cityscapes 512×1024，R50-D8 40k','V100：3.94 FPS','mIoU79.61；与NPU VOC实验不是同一任务')
    for n in [69,70]:put(n,gh('open-mmlab/mmdetection','configs/faster_rcnn/README.md'),'MMDetection Faster R-CNN R50-FPN配置','R50-FPN 1x PyTorch checkpoint','COCO','原表21.4 FPS，GPU型号需另核','bbox AP37.4，R50-FPN 1x；不借用C4/DC5变体')
    put(71,gh('open-mmlab/mmsegmentation','configs/segformer/README.md'),'SegFormer MIT-B0 GPU配置','原表MIT-B0 checkpoint','ADE20K 512×512，160k','GTX1080Ti：51.32 FPS','mIoU37.41；不是NPU Cityscapes75.94')
    put(72,gh('open-mmlab/mmdetection','configs/swin/README.md'),'Mask R-CNN Swin-S FPN FP16 ms-crop3x','原表checkpoint及日志','COCO','原表该行未列FPS','box AP48.2、mask AP43.2')
    put(73,gh('HuCaoFighting/Swin-Unet'),'Swin-Unet官方V100训练代码；U-Net须另列范围','Swin-T预训练公开；作者声明最终分割模型不能分享','Synapse/BTCV、ACDC处理数据','数值待核对','论文结果需按数据和配置核对；不可填普通UNet结果')
    put(74,gh('DebeshJha/ResUNetPlusPlus'),'原始ResUNet++实现；具体GPU训练入口待核','原仓库说明','Kvasir-SEG','数值待核对','论文与代码版本需匹配',status='已查原始实现，GPU执行细节待核')
    put(76,'https://docs.ultralytics.com/models/yolov8/','官方YOLOv8 GPU训练/推理','各任务权重及下载表','COCO等，检测/分割/姿态分别统计','官方表提供A100 TensorRT速度；须按型号读取','官方模型表提供COCO mAP；不填新YOLO版本')
    put(77,gh('PaddlePaddle/PaddleSeg','configs/segnet/README.md'),'PaddleSeg SegNet配置及训练记录','原表model.pdparams','Cityscapes 1024×512，80k','数值待核对','mIoU60.09%；GPU硬件在训练日志中待核',status='有原始训练记录，具体GPU配置待核')
    put(79,gh('thuml/Anomaly-Transformer','solver.py'),'solver明确cuda:0及model.cuda()','按数据从头训练','原仓库SMD/MSL/SMAP/PSM/SWaT准备','数值待核对','F1需说明point-adjust协议',{'数据集':[gh('thuml/Anomaly-Transformer')]})
    put(80,gh('fundamentalvision/BEVFormer'),'原始训练/评估代码及模型表','官方model zoo','nuScenes及CAN bus','数值待核对','原始完整周期结果不能与NPU四轮配置直接比较')
    put(81,gh('FoundationVision/ByteTrack'),'官方V100 GPU检测跟踪流水线','README模型下载表','MOT17/MOT20及训练混合','单V100 MOT17：29.6 FPS','MOT17 MOTA80.3、IDF1 77.3、HOTA63.1')
    put(84,gh('facebookresearch/dinov2'),'官方CUDA训练和下游评估','预训练模型下载表','ImageNet1k/22k；各recipe分开','ViT-L/16：32×A100-80GB约1天；非推理速度','该recipe kNN81.6%、linear82.9%，非所有DINOv2规模')
    put(85,gh('IDEA-Research/GroundingDINO'),'官方CUDA扩展和GPU demo','SwinT OGC、SwinB COGCOOR等','COCO零样本/微调；与RefCOCO训练分开','数值待核对','官方模型级COCO零样本52.5 AP、微调63.0 AP；不可填给每个checkpoint',{'权重':[hf('ShilongLiu/GroundingDINO')]})
    put(87,gh('pangfumin/CudaDwa'),'社区CUDA10.2 DWA玩具实现，不是完整ROS量产栈','不需要权重','内置规划demo','数值待核对','无真实机器人验收',status='有GPU实现（社区demo）')
    put(88,gh('DepthAnything/Depth-Anything-V2'),'官方代码明确CUDA设备选择','V2 Small/Base/Large，metric版本另列','DA-2K；NYU/KITTI等需区分相对/度量深度','数值待核对','官方相对/度量深度结果不能混用',{'权重':[hf('depth-anything/Depth-Anything-V2-Base')],'数据集':[hf('depth-anything/DA-2K',True)]})
    put(89,'https://docs.nvidia.com/cugraph/26.10/api_docs/api/cugraph/cugraph.sssp/','cuGraph GPU最短路SSSP；并非逐字复现经典Dijkstra实现','不需要权重','输入加权图；负权环不支持','数值待核对','需比较最短路距离/前驱；不是模型精度',status='有GPU最短路替代实现')
    put(90,gh('danijar/dreamerv3'),'官方JAX默认GPU运行，可切CPU/TPU','按环境训练产生','在线RL环境','数值待核对','按环境、多seed及训练预算比较')
    put(91,'https://docs.gpytorch.ai/en/stable/examples/02_Scalable_Exact_GPs/Simple_GP_Regression_CUDA.html','GPyTorch官方CUDA RBF回归教程','按任务拟合','100训练点、51测试点的sin函数样例','不是正式性能基准','示例正确性，不替代业务预测精度')
    for n,model in [(93,'InternVL3-8B'),(94,'InternVL2-8B')]:put(n,hf('OpenGVLab/'+model),'模型卡明确model.eval().cuda()及多GPU映射',model,'模型卡的评测集及图像/视频样例','数值待按规模和输入核对','模型卡有官方评测表；业务航拍精度另验',{'权重':[hf('OpenGVLab/'+model)]})
    put(95,gh('haotian-liu/LLaVA'),'官方CUDA训练，支持A100及3090/A6000等','1.5-7B等模型清单','LLaVA1.5训练数据；TextVQA等','官方1.5训练约1天/8×A100；不是推理速度','官方11基准结果与任务配置见模型表',{'权重':[gh('haotian-liu/LLaVA','docs/MODEL_ZOO.md')],'数据集':[gh('haotian-liu/LLaVA','docs/Data.md')]})
    put(97,'https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html','官方USE_GPU=1、device=gpu；该教程为OpenCL路线，另有CUDA实现','训练产生树模型','HIGGS官方教程数据','数值待按GPU及数据规模核对','备件预测业务数据未提供，不借用HIGGS结果')
    put(98,'https://github.com/huggingface/peft','官方PEFT/LoRA库；具体CUDA组合按底座核查','底座与adapter分别准备','按微调任务','无统一方法级速度','无统一方法级精度',status='有GPU使用路线，按底座和量化组合核查')
    put(101,gh('facebookresearch/map-anything'),'官方CUDA推理、训练及几何模型接口','官方MapAnything checkpoint','13类训练数据及215GB元数据','数值待核对','原仓库几何评测协议，不用DINO骨干结果代替',{'数据集':[gh('facebookresearch/map-anything','data_processing/README.md')]})
    put(106,gh('marlbenchmark/on-policy'),'MAPPO原实现给出CUDA10.1安装及训练脚本','从头训练产生','SMAC/MPE/Hanabi/Football','数值待按环境核对','原论文多环境结果；需固定SC2版本及seed')
    put(107,gh('amazon-science/patchcore-inspection'),'官方--gpu及--faiss_on_gpu','预训练backbone与memory bank','MVTec AD','多数实验显存不超过11GB是内存条件，非吞吐','WR50 baseline：图像AUROC99.2%、分割98.1%、PRO94.4%')
    put(108,gh('oxwhirl/pymarl'),'QMIX官方PyMARL：run.sh $GPU','按SMAC场景训练产生','SMAC；原结果SC2 4.6.2.69232','数值待核对','环境版本不同不能直接对比胜率')
    put(109,gh('QwenLM/Qwen3'),'官方CUDA推理，Transformers/vLLM/SGLang等','dense/MoE不同规模','官方评测数据；业务审核数据另验','数值待按模型、batch、长度核对','不能用通用榜单代替业务正确率')
    for n,name in [(110,'Qwen2.5-VL-7B-Instruct'),(111,'Qwen2-VL-7B-Instruct')]:put(n,hf('Qwen/'+name),'官方模型卡提供GPU推理/部署及显存配置',name,'官方视觉基准与样例；业务集另验','数值待核对','模型卡原始评测；不借用其他规模',{'权重':[hf('Qwen/'+name)]})
    put(113,gh('lyuwenyu/RT-DETR'),'第一代RT-DETR官方Paddle/PyTorch实现','R18/R50/R101等权重','COCO；Objects365预训练另列','T4 TensorRT FP16：R18 217FPS；R50 108FPS，640输入','COCO R18 AP46.5；R50 AP53.1（非Objects365预训练行）')
    put(114,gh('facebookresearch/segment-anything'),'官方推荐CUDA；图像分割及AMG','ViT-B/L/H公开checkpoint','SA-1B及原论文评测集','数值待核对','原始SAM协议与ModelZoo自定义COCO提示协议不同')
    put(115,gh('facebookresearch/sam2'),'官方CUDA图像及视频预测；SAM2与2.1分开','2.1 Hiera Tiny/Small/Base+/Large','SA-V、MOSE、LVOS v2','官方2.1 Tiny表91.2FPS；具体硬件条件需与速度说明核对','2.1 Tiny视频J&F：SA-V76.5、MOSE71.8、LVOSv2 77.3；不是图像mIoU')
    put(117,gh('DurYi/UA-NWM'),'官方CUDA12.1训练及导航评估','UA-NWM checkpoint，DINOv3需授权；RAEv2仅可视化','AirGoal-10k','数值待核对','导航质量需按UA/确定性两种推理策略分开',{'权重':[hf('DurYi/UA-NWM-Checkpoints')],'数据集':[hf('DurYi/AirGoal-10k',True)]})
    put(118,gh('facebookresearch/vjepa2'),'官方CUDA训练、probe评估及2-AC推理','ViT-L/H/g及2-AC权重','SSv2、Diving48、EPIC-KITCHENS100','数值待核对','ViT-L/16：SSv2 73.7%、Diving48 89.0%、EK100 32.7 R@5；非2.1结果',{'权重':['https://dl.fbaipublicfiles.com/vjepa2/vitl.pt']})
    put(119,gh('facebookresearch/vggt'),'官方CUDA/bf16推理及训练','VGGT-1B；Commercial版许可另列','CO3D及原始训练数据说明','原文通常重建<1秒，缺固定规模硬件，不能当规范性能','CO3D AUC@30：原版89.98、Commercial90.37',{'权重':[hf('facebook/VGGT-1B'),hf('facebook/VGGT-1B-Commercial')]})
    put(36,gh('wow-world-model/wow-world-model','demo/wan_infer_demo.py'),'官方Wan demo明确cuda:{gpu_id}；训练流水线和SOPHIA尚在路线图','1.3B/14B Wan、2B/7B DiT分别公开','WoW-1-Benchmark-Samples；不等于完整200万轨迹训练集','demo的15FPS是视频编码帧率，不是生成吞吐','WoWBench完整设计/指标仍列待发布',{'权重':[hf('X-Humanoid/WoW-1-Wan-1.3B-2M'),hf('WoW-world-model/WoW-1-Wan-14B-2M')],'数据集':[hf('WoW-world-model/WoW-1-Benchmark-Samples',True)]})
    put(46,gh('opendatalab/MinerU',ref='1a59613'),'3.3.0发布提交：pipeline CPU/GPU；VLM vLLM/LMDeploy；不借用4.0命令','3.3后端对应模型；需按该版本model config下载','官方文档解析样例及评测数据；不等于业务集','Volta及以后GPU；具体吞吐待核','版本级精度表待核，不能借用4.0结果')
    put(47,gh('PaddlePaddle/PaddleOCR'),'官方1.6发布与NVIDIA部署入口；需选择1.6配套后端','PaddleOCR-VL-1.6，0.9B','OmniDocBench v1.6、v1.5、Real5','未提取同硬件同后端性能','官方OmniDocBench v1.6：96.3%；该分数不是NPU实测',{'权重':[hf('PaddlePaddle/PaddleOCR-VL-1.6')]})
    put(48,gh('PaddlePaddle/PaddleOCR'),'官方工具库支持NVIDIA GPU；按PP-OCR具体版本选模型','检测/识别/方向等组件分别下载','各版本OCR评测集','无工具库统一FPS','无工具库统一精度',status='支持GPU，需指定OCR版本和组件')
    put(78,gh('microsoft/AirSim','docs/faq.md',ref='main'),'Unreal仿真渲染使用GPU；不是CUDA模型推理实现','不适用；环境资源单独准备','场景和传感器配置，不是统一训练集','官方场景要求4GB/8GB显存；开发机TitanX，非FPS基准','应验仿真/传感器正确性，不用模型准确率',status='GPU渲染平台，非模型')
    put(92,'https://docs.px4.io/v1.18/en/sim_gazebo_gz/','Gazebo OGRE图形渲染；PX4 SITL不因此成为CUDA实现','不适用','场景、机型和传感器配置','未找到固定场景GPU实时因子；headless模式另列','仿真功能/飞控验证，不是模型精度',{'代码':['https://docs.px4.io/v1.18/en/sim_gazebo_gz/','https://gazebosim.org/docs/garden/troubleshooting/']},status='GPU图形渲染，非CUDA飞控')
    put(96,'https://www.hiascend.com/software/modelzoo/models/detail/46fe0872ddfd4e19a5104189913b5fee','已定位TIMIT+CTC原实现Diamondfan/CTC_pytorch固定提交；CUDA代码仍待确认','不能把NPU ctc_best_model.pth直接认定为已验证GPU资产','TIMIT需授权；与通用时序LSTM不同','未确认该实现GPU性能','未确认同一TIMIT split的GPU结果',{'代码':[gh('Diamondfan/CTC_pytorch','timit',ref='eddd2550224dfe5ac28b6c4d20df5dde7eaf6119'),'https://www.hiascend.com/software/modelzoo/models/detail/46fe0872ddfd4e19a5104189913b5fee']},status='已定位原实现，GPU运行证据待补')
    put(99,gh('shariqiqbal2810/maddpg-pytorch','algorithms/maddpg.py',ref='master'),'社区PyTorch实现prep_training(device=gpu)明确.cuda()；非OpenAI原TF仓','按MPE场景训练产生','OpenAI multiagent-particle-envs','未确认GPU吞吐','按环境、seed及训练预算比较；不借用原TF结果',status='有GPU实现（社区PyTorch版本）')
    put(102,gh('google-deepmind/deepmind-research','meshgraphnets/requirements.txt',ref='master'),'原实现依赖tensorflow-gpu>=1.15,<2；完整训练/评估示例限cylinder_flow和flag_simple','README提供从头训练入口；未确认通用预训练包','airfoil、cylinder_flow、flag/sphere各变体等10套','未提取原论文同条件GPU速度','按物理量rollout误差评估；flag_minimal只做集成测试',{'数据集':[gh('google-deepmind/deepmind-research','meshgraphnets/README.md',ref='master')]})
    put(103,gh('UZ-SLAMLab/ORB_SLAM3',ref='master'),'官方C++ SLAM实现；已读依赖/运行说明，未确认原生CUDA后端','ORB词袋资源，不是神经网络权重','EuRoC、TUM-VI、TUM RGB-D等','不能将CPU实时性能归给GPU','按ATE/RPE等轨迹指标；GPU专项结果未确认',status='原始实现已查，GPU后端未确认')
    put(104,gh('rezaakb/pinns-torch'),'PINNs-Torch作者实现使用CUDA Graphs/JIT；PINN是方法，不是单一模型','按PDE问题训练产生','Navier-Stokes/Schrodinger等算例','作者报告最高9倍相对原TF1；非NPU加速比，具体算例条件见论文','按各PDE相对误差，不能给方法统一准确率',{'性能基准':['https://openreview.net/forum?id=nl1ZzdHpab']},status='有GPU实现，按PDE算例核查')
    put(105,'https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html','SB3官方PPO支持device=cuda；MLP通常推荐CPU，不承诺GPU更快','按环境训练；预训练策略按环境下载','Gymnasium/Atari等在线环境','无统一GPU FPS，取决于环境和策略网络','按环境回报/成功率及训练预算比较',status='有GPU实现，按环境和策略核查')
    put(116,gh('thuml/Time-Series-Library','exp/exp_basic.py'),'官方TimesNet整合库明确torch.device(cuda)及model.to(device)','按任务/数据集训练产生','ETT/ECL/Traffic等；严格固定split和预测长度','未提取同GPU训练/推理速度','预测MSE/MAE等按任务比较，不借用分类指标')
    rows[114]['设备维度']['GPU']['性能基准']['说明']='SAM2.1 Tiny 91.2FPS；A100、torch2.5.1、CUDA12.4，全模型编译'
    rows[75]['设备维度']['GPU']['性能基准']['说明']='检测640输入，A100 TensorRT：v8n 0.99ms；v8s 1.20ms；v8m 1.83ms；v8l 2.39ms；v8x 3.53ms'
    rows[75]['设备维度']['GPU']['精度基准']['说明']='COCO val mAP50-95：v8n 37.3、v8s 44.9、v8m 50.2、v8l 52.9、v8x 53.9'
    for n in [42,44]:rows[n-1]['设备维度']['GPU']['精度基准']['说明']='MTEB Multilingual Mean(Task)：0.6B 64.33、4B 69.45、8B 70.58；官方2025-06结果'
    rows[26]['设备维度']['GPU']['精度基准']['说明']='官方model zoo：Bridge+RT-1训练的Qwen2.5-FAST WidowX58.6、OFT41.8、PI62.5、GR00T63.6；不是同一模型'
    rows[26]['设备维度']['GPU']['精度基准']['链接']=[gh('starVLA/starVLA','docs/model_zoo.md')]
    rows[79]['设备维度']['GPU']['精度基准']['说明']='nuScenes val、24ep：tiny NDS35.4/mAP25.2；base NDS51.7/mAP41.6；不同于NPU四轮配置'
    rows[79]['设备维度']['GPU']['权重']['链接']=['https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth']
    for n,path in [(40,'FlagEmbedding/inference/embedder/encoder_only/m3.py'),(41,'FlagEmbedding/inference/reranker/encoder_only/base.py')]:
        r=rows[n-1];r['GPU结论']='支持GPU，有官方实现'
        r['设备维度']['GPU']['代码']['说明']='官方推理类支持cuda:0/cuda:1设备；fp16、dense/sparse/ColBERT输出按对应模型选择' if n==40 else '官方encoder-only reranker明确CUDA设备、model.to(device)及GPU显存处理'
        r['设备维度']['GPU']['代码']['链接']=[gh('FlagOpen/FlagEmbedding',path)]
    r=rows[37];r['GPU结论']='支持GPU，有官方实现'
    r['设备维度']['GPU']['代码']['说明']='官方Accelerate单/多GPU训练入口，启用CUDA TF32；部署样例仍在路线图'
    r['设备维度']['GPU']['代码']['链接']=[gh('Open-X-Humanoid/XR-1','lerobot/scripts/train_humanoid_multigpu.py')]
    r=rows[73];r['GPU结论']='有GPU实现（社区PyTorch版本）'
    r['设备维度']['GPU']['代码']['说明']='原作者链接的PyTorch复现：ResUnetPlusPlus及输入/标签明确.cuda()；与原TF实现分开'
    r['设备维度']['GPU']['代码']['链接']=[gh('rishikksh20/ResUnet','train.py',ref='master'),gh('DebeshJha/ResUNetPlusPlus')]
    r=rows[95];r['GPU结论']='支持GPU，有官方实现'
    r['设备维度']['GPU']['代码']['说明']='TIMIT+CTC原实现固定提交：use_gpu选择torch.device(cuda)，模型及输入均转该设备'
    r['设备维度']['GPU']['代码']['链接']=[gh('Diamondfan/CTC_pytorch','timit/steps/train_ctc.py',ref='eddd2550224dfe5ac28b6c4d20df5dde7eaf6119')]
    r['设备维度']['GPU']['数据集']['说明']='TIMIT需授权；原README：train3696句（排除SA）、dev400句、test192句'
    r['设备维度']['GPU']['数据集']['链接']=[gh('Diamondfan/CTC_pytorch',ref='master')]
    rows[0]['设备维度']['GPU']['精度基准']['说明']='Optional-IDM发布权重，完整LIBERO 40任务、每任务50次：IDM平均98.55%；First-frame平均97.75%；不对应NPU旧版200次评估'
    # Keep the earlier explicitly labelled CUDA comparison alongside original-model results.
    old=rows[84]['证据']['精度基准']
    rows[84]['设备维度']['GPU']['精度基准']['说明']+='；另有ModelZoo训练对照：GPU RefCOCO AP0.3148，仅对应该训练配置'
    rows[84]['设备维度']['GPU']['精度基准']['链接']+=old['链接']
    # Search performed, but a paper/announcement alone does not prove CUDA code.
    searched={
      3:('https://airvla.github.io/','项目页代码/数据仍待发布，未定位GPU执行实现'),
      24:('https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/','找到官方论文；完整代码/权重交付未确认'),
      25:('https://spirit-ai.com/news/34','找到官方发布；未定位1.6公开GPU代码/权重，不能借用1.5'),
      29:('https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0331/2026033102608_c.pdf','找到Thinker-VLA公司披露；不等同Thinker-4B或Thinker-WM开源'),
      32:('https://x2robot.com/about','有WALL-A发布记录；未定位该版本公开GPU实现'),
      33:('https://x2robot.com/about','有WALL-B发布记录；未定位该版本公开GPU实现'),
      39:('https://arxiv.org/abs/2606.31846','已定位Z-1流式VLA强化学习论文，非Z1机器人；公开代码待核'),
      53:('https://ccb.jhu.edu/software/bracken/','已查Bracken官方文档，未定位原生GPU后端'),
      75:('https://www.sciencedirect.com/science/article/pii/S1369800126002477','论文对应UNet-ESH；公开代码/权重及GPU条件未确认'),
      82:('https://github.com/zehuilu/CBBA-Python','找到CPU算法实现，未定位GPU专项实现'),
      83:('https://arxiv.org/abs/2505.00091','已查原论文入口，GPU完整交付未确认'),
      86:('', '已检索DL-foils、DLFoils、DL-foil及airfoil组合；未唯一定位，不能借用同名逻辑学习DL-Foil'),
      100:('https://www.kubeflow.org/docs/started/installing-kubeflow/','MLOps训练一体化是平台类别，未给出产品/项目；Kubeflow只是参考，不能判定原条目GPU交付'),
      112:('https://arxiv.org/abs/2507.01378','已查RALLY论文，GPU完整交付未确认')}
    for n,(url,text) in searched.items():
        r=rows[n-1];r['GPU查询状态']='已查询；实现未确认';r['GPU结论']='已查资料，GPU交付未确认'
        r['设备维度']['GPU']['代码']={'说明':text,'链接':[url] if url else [],'证据类型':'已查询原始入口，不证明GPU实现'}
        for k in fields[1:]:r['设备维度']['GPU'][k]={'说明':'尚未核实','链接':[],'证据类型':'查询后证据未闭合'}
    rows[38]['结论']='已定位Z-1论文，NPU实现未确认'
    rows[38]['NPU结论']=rows[38]['结论']
    rows[38]['分析与缺口']='Z-1是π0.5基础上的GRPO后训练框架；RoboCasa24任务80.6%为论文结果，硬件归属另核。'
    return rows
