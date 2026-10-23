"""Separate device evidence without treating an unspecified competitor as GPU."""
from copy import deepcopy

UPSTREAM = {
2:'chenjinyubuaa/AerialVLA',9:'OpenGalaxea/GalaxeaVLA',10:'OpenGalaxea/GalaxeaVLA',11:'OpenGalaxea/GalaxeaVLA',12:'OpenGalaxea/GalaxeaVLA',
13:'NVIDIA/Isaac-GR00T',14:'NVIDIA/Isaac-GR00T',15:'NVIDIA/Isaac-GR00T',16:'open-gigaai/giga-world-1',17:'open-gigaai/giga-brain-0',18:'open-gigaai/giga-world-0',19:'Robbyant/lingbot-vla',20:'openvla/openvla',21:'Open-X-Humanoid/pelican-vl',22:'huggingface/lerobot',23:'huggingface/lerobot',27:'starVLA/starVLA',31:'X-Square-Robot/wall-x',37:'EmbodiedCity/Worldscape-MoE.code',38:'Open-X-Humanoid/XR-1',
40:'FlagOpen/FlagEmbedding',41:'FlagOpen/FlagEmbedding',46:'opendatalab/MinerU',47:'PaddlePaddle/PaddleOCR',48:'PaddlePaddle/PaddleOCR',53:'jenniferlu717/Bracken',54:'RosettaCommons/DeepAb',55:'google-deepmind/alphafold3',56:'google-deepmind/alphafold',57:'evolutionaryscale/esm',58:'facebookresearch/esm',59:'dauparas/LigandMPNN',61:'dauparas/ProteinMPNN',62:'RosettaCommons/RFdiffusion2',63:'RosettaCommons/RFdiffusion',64:'RosettaCommons/RoseTTAFold',
66:'justchenhao/BIT_CD',67:'wgcban/ChangeFormer',74:'DebeshJha/ResUNetPlusPlus',78:'microsoft/AirSim',79:'thuml/Anomaly-Transformer',80:'fundamentalvision/BEVFormer',81:'ifzhang/ByteTrack',84:'facebookresearch/dinov2',85:'IDEA-Research/GroundingDINO',88:'DepthAnything/Depth-Anything-V2',90:'danijar/dreamerv3',93:'OpenGVLab/InternVL',94:'OpenGVLab/InternVL',95:'haotian-liu/LLaVA',97:'microsoft/LightGBM',98:'huggingface/peft',101:'facebookresearch/map-anything',102:'google-deepmind/deepmind-research/tree/master/meshgraphnets',103:'UZ-SLAMLab/ORB_SLAM3',107:'amazon-science/patchcore-inspection',109:'QwenLM/Qwen3',110:'QwenLM/Qwen2.5-VL',111:'QwenLM/Qwen2-VL',113:'lyuwenyu/RT-DETR',114:'facebookresearch/segment-anything',115:'facebookresearch/sam2',116:'thuml/Time-Series-Library',117:'DurYi/UA-NWM',118:'facebookresearch/vjepa2',119:'facebookresearch/vggt'}

def ev(text,links=(),role='待GPU专项核实'):
    return {'说明':text,'链接':list(links),'证据类型':role}

def attach(rows):
    for r in rows:
        n=r['编号'];npu=deepcopy(r['证据'])
        gpu={k:ev('未取得明确GPU专项证据；不以NPU结果代替') for k in npu}
        if n in UPSTREAM:
            url='https://github.com/'+UPSTREAM[n]
            gpu['代码']=ev('上游代码入口；CUDA实现、版本与运行范围尚未逐项核验',[url],'上游入口；非已验证GPU运行')
        elif '原始资源' in npu['代码']['证据类型']:
            gpu['代码']=ev('原始项目/发布入口；GPU实现未独立核验',npu['代码']['链接'],'原始入口；非GPU适配证明')
        for k in ['权重','数据集']:
            v=npu[k]
            if v['链接'] and not v['说明'].startswith('未找到'):
                gpu[k]=ev('来源资源：'+v['说明']+'；GPU使用的版本、文件兼容及测试split待核实',v['链接'],'资源线索；不代表GPU验证')
            elif '不需要预训练' in v['说明'] or '无需统一预训练' in v['说明']:
                gpu[k]=ev(v['说明'],[],'不适用/依具体实现')
        r['设备维度']={'GPU':gpu,'NPU':npu}
        r['GPU结论']='有资源入口；GPU完整证据待核实' if gpu['代码']['链接'] else 'GPU证据不足/不适用'
        r['NPU结论']=r['结论']
        # Preserve the full comparison separately; do not auto-assign 竞品 to GPU.
        r['对照原文']={k:deepcopy(npu[k]) for k in ['性能基准','精度基准'] if any(s in npu[k]['说明'] for s in ['对照','/0.','/1.','/457','/87.22'])}
    # Only explicitly GPU-labelled evidence from the reviewed fixed snapshot.
    for n,k,text in [
        (85,'精度基准','GPU RefCOCO AP0.3148；对应训练配置'),
        (115,'精度基准','GPU图像分割mIoU0.7830；SAM2.1 Tiny，非视频任务'),
        (95,'性能基准','GPU训练18.62 samples/s；原表对照，环境可比性未独立复测')]:
        r=rows[n-1]
        r['设备维度']['GPU'][k]=ev(text,r['证据'][k]['链接'],'固定快照明确GPU对照；未复测')
    for n,k,text in [
        (4,'精度基准','NPU500步训练loss0.0409；非视频任务质量'),
        (6,'精度基准','NPU SFT loss0.380、RL loss1.9219；非任务正确率'),
        (13,'性能基准','NPU A2八卡337.35 FPS'),(13,'精度基准','NPU训练loss0.0031；非任务成功率'),
        (14,'性能基准','NPU训练449 FPS；指定推理0.145s'),(14,'精度基准','NPU训练loss0.0082；指定推理MSE1.185'),
        (19,'性能基准','NPU A3八卡301.76 FPS'),(20,'性能基准','NPU A2八卡56.14 FPS'),(20,'精度基准','NPU训练loss0.1876；非任务成功率'),
        (22,'性能基准','NPU A3八卡155.1 FPS，global batch128；原对照batch64，非同条件加速比'),(22,'精度基准','NPU训练loss0.004；batch128'),
        (27,'性能基准','NPU A3八卡123.08 FPS'),(27,'精度基准','NPU部分任务88.38，非完整50任务'),
        (59,'性能基准','NPU A2单步1.332s'),(61,'性能基准','NPU A3单步0.28s；950DT0.15s'),
        (85,'精度基准','NPU RefCOCO AP0.3152；对应训练配置'),
        (115,'精度基准','NPU图像分割mIoU0.7815；非视频任务'),
        (119,'性能基准','NPU A2八卡25.04 FPS'),(119,'精度基准','NPU训练loss0.0165；仅CO3D tv场景')]:
        rows[n-1]['设备维度']['NPU'][k]['说明']=text
    # Published CPU/GPU model scores are not NPU task alignment evidence.
    r=rows[115]
    r['设备维度']['NPU']['精度基准']=ev('未取得NPU独立精度对照；发布模型MSE/MAE设备未明确',r['证据']['精度基准']['链接'],'支持清单，不证明NPU实测')
    return rows
