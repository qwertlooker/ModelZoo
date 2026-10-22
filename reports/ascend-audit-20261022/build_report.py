"""Build a source-linked inventory, preserving every original scope item."""
import csv
import hashlib
import json
import pathlib
import re
from collections import Counter
from urllib.parse import quote

ROOT = pathlib.Path(__file__).resolve().parent
SHA = 'dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202'
BASE = 'https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/' + SHA + '/'

def link(path, label='证据'):
    return f'[{label}]({BASE}{quote(path, safe="/")})'

# These are manually reviewed facts, not keyword-based completion scores.
# key, code, weights, data, performance, accuracy, boundary, source suffix
FACTS = [
('GR00T-N1.6','三段OM导出、policy/server、离线评测','nvidia/GR00T-N1.6-3B；另需仿真资产','gr1.PickNPlace；RoboCasa GR1单任务10 episodes','300V Pro：合成观测E2E 1.3s；warmup3/iters10均值','任务成功率60.0%；文档引用竞品51.5%，未给该数字直接出处','合成输入时延不是正式L2性能；10次任务成功率不代表全任务能力；权重未下载校验','embodied_ai/vla/GR00T_n1d6'),
('Pi-0','ONNX/OM导出、分段比较、infer.py','pi0_aloha_transfer_cube；PaliGemma tokenizer','示例frame；mock ONNX/OM输出比较','300I DUO单芯220ms；正文frame与表中随机数矛盾','仅描述余弦比较，未见LIBERO任务成功率数值','支持300I DUO/Pro；明确不支持800I A2；缺正式质量基准','embodied_ai/vla/pi0'),
('Pi-0.5','LeRobot TorchAir；OpenPI TorchAir两条路线','pi05_libero_finetuned；OpenPI pi05_libero及JAX转换、tokenizer','LIBERO object；OpenPI libero_10','300I DUO单芯：LeRobot375ms、OpenPI378ms','LeRobot表中成功率100，对照98.2；OpenPI仅声称持平','两路线不可共享验收结论；OpenPI缺独立精度结果；不支持800I A2','embodied_ai/vla/pi05_lerobot/infer_with_torchair'),
('BGE-M3','infer.py使用FlagEmbedding及TorchAir','BAAI/bge-m3；未固定权重revision/SHA','脚本内示例句子；未见正式检索评测集','DUO E2E137.59ms/forward23.23ms；A2 103.88/14.71ms','未提供MTEB/MIRACL等正式指标和NPU对照','代码输出分支需逐项验收；样例性能不能代表检索质量','embedding/bge-m3'),
('BGE-Reranker-v2-m3','infer.py打分及TorchAir','BAAI/bge-reranker-v2-m3；未固定revision/SHA','示例query-passage；无正式重排评测集','DUO E2E60.47ms/forward22.57ms；A2 46.64/15.08ms','未提供nDCG/MRR等正式指标','README表误写bge-m3，需核对实际模型；缺任务级质量证据','embedding/bge-reranker-v2-m3'),
('PaddleOCR-VL-1.6','vLLM-Ascend服务、文档解析与OmniDocBench评测；布局模型OM','PaddleOCR-VL-1.6及PP-DocLayoutV3 ONNX','OmniDocBenchV1.6 PDFs及标注','Atlas350：官方示例图E2E 1.39s，非全量数据性能','Atlas350 95.45%，竞品96.33%','仅支持Atlas350；模型服务与布局组件分别依赖软件配套；未重跑','ocr/PaddleOCR-VL-1.6'),
('PaddleOCR','PP-OCRv5/v6导出、OM及评测入口','det/rec；v5另有方向分类和UVDoc权重链接','OCR官方示例集；v5辅助模块另有构造集/DocUNet','v5 det35.01/26.85ms(EP/RC)、rec2.63/6.09ms；v6 det4.28/rec1.43ms','v5 hmean52.87%/acc72.27%；v6 46.73%/77.27%，各与A10相同','不同版本、硬件及子模块分开；示例集结果不等于官方全量榜单','ocr/PPOCRv5'),
('DeepLabv3+','训练源码；离线推理预处理、导出、后处理','推理deeplab-resnet.pth.tar；训练从流程生成','VOC2012','推理300I Pro bs1表值165.545；训练8p252.83 FPS','推理mIoU78.43；训练8p0.738','训练与推理checkpoint/配置不等价，不能混合对齐','@ACL_PyTorch/contrib/cv/segmentation/DeeplabV3+'),
('fasterRCNN','ResNet50 ONNX/OM及COCO后处理','README指定R50-FPN权重及配置','COCO2017 val 5000张','300I Pro bs1表值14.84；基准19.4','mAP表值37.2','rpn nms_post可从1000改500，会影响精度；需固定配置和计时单位','@ACL_PyTorch/contrib/cv/detection/Faster_R-CNN_ResNet50'),
('Faster R-CNN + FPN','Detectron2 ResNet50-FPN NPU训练与1p/8p脚本','R-50.pkl ImageNet预训练权重直链','COCO2017','主README未列NPU性能数值','主README未列NPU mAP；嵌入上游model-zoo结果不能视为NPU结果','存在指定组合实现；不可仅凭训练脚本判定完成精度性能验收','@PyTorch/dev/cv/detection/FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch'),
('SegFormer','MIT-B0离线推理与前后处理','OpenMMLab cityscapes MIT-B0指定pth','Cityscapes val；需注册获取','300I Pro bs1 6.01fps；基准10.61fps','mIoU75.94','仅对应MIT-B0、指定分辨率/旧软件栈；不覆盖全系列','@ACL_PyTorch/contrib/cv/segmentation/Segformer'),
('Swin+FPN','Mask R-CNN Swin-S+FPN离线推理，另有UPerNet分割','mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco指定pth','COCO2017','300I Pro bs1表值5.17，T4 2.5','bbox mAP NPU47.8，官网pth48.2','修正前次遗漏：有Swin+FPN检测实例；若需求指语义分割须另明确头部','@ACL_PyTorch/contrib/cv/detection/Mask_Rcnn_SwinS'),
('Swin-Unet/U-Net','U-Net训练与OM推理；未找到Swin-Unet专项','UNet.pth直链；训练另行生成','推理Carvana train/train_masks；训练DSB2018','推理300I Pro bs1表值75.0603；训练8p A2 1722.650FPS','推理IoU0.986305；训练README Acc85.90','原条目须拆开；Carvana使用train图像，不能宣称独立测试集泛化指标','@ACL_PyTorch/contrib/cv/segmentation/UNet'),
('BEVFormer','R101-DCN BEVFormer-base NPU训练','r101_dcn_fcos3d_pretrain.pth需准备，README未给直接下载URL','nuScenes V1.0 full及CAN bus','8p800T A2 FP32 4epoch 2.915FPS；另硬件表3.79FPS','NDS46.54/mAP38.36；竞品44.29/35.16','4epoch配置，非完整官方训练复现；硬件表不同不可混用','@PyTorch/built-in/autonoumous_driving/BEVFormer'),
('Grounding DINO','MM-GroundingDINO训练和TorchAir图像/视频推理','Swin-B预训练pth、BERT及NLTK语料','训练RefCOCO/COCO2014；推理animals.png与COCO缩放实验','推理A2 E2E877ms/forward274ms；8p训练8.45FPS','RefCOCO AP0.3152 vs GPU0.3148；推理缩放实验mAP0.522/0.52','并非所有原始GroundingDINO变体；DUO仅图像；单图性能与训练指标分开','cv/GroundingDINO'),
('InternVL2','8B/40B MindIE镜像内置代码与服务指导','模型路径占位；这两份README未给具体权重下载链接或SHA','自备图像/视频文件；未列固定质量测试集','未列可追溯性能结果表','未列MMBench等质量基准或NPU对照','支持指导依赖MindIE1.0.0 A2镜像；不能由对话样例认定质量验收','@MindIE/MultiModal/InternVL/InternVL2/Internvl2_8b'),
('LLaVA（航拍开放词汇感知）','LLaVA1.5-7B训练、推理、TextVQA评估入口','Vicuna7B-v1.5、视觉塔、projector需按指导准备','mix665k及五类图像；TextVQA0.5.1 val','8p A2训练20.13 samples/s；GPU18.62 samples/s','有TextVQA命令，README未给实际质量数值','该目录已声明停止维护；存在CPU-only torch安装示例风险；航拍任务未验收','@PyTorch/built-in/mm/LLaVA'),
('LSTM','TIMIT语音CTC训练与OM推理','ctc_best_model.pth直链；Kaldi数据处理','TIMIT，需自行获得许可；指导使用dev','300I Pro bs1 5.3457fps','CER13.5491/WER18.8003；训练Acc@1 80.5口径不同','此为语音LSTM，不证明任意时序预测LSTM可直接用；dev非独立test','audio/LSTM'),
('PPO','gym/Box2D NPU训练','通常从头训练，无需固定预训练checkpoint；训练产出权重','在线仿真环境；不是下载式数据集','800T A2 284.02FPS；另版本表336.84FPS','300万步Average Reward240；竞品197.75','固定环境/seed与统计区间；不同表性能不混用；非多seed统计','@PyTorch/built-in/rl/PPO_for_Pytorch'),
('MAPPO','onpolicy NPU多智能体训练','从头训练并保存checkpoint，不以缺预训练权重判失败','在线多智能体环境；参见脚本场景配置','1p NPU885FPS；另900 A2表1791.33FPS','200万步Average Reward -15.9，与竞品相同','需固定具体环境、seed、多次回报分布；不同硬件结果不可拼接','@PyTorch/built-in/rl/MAPPO_for_PyTorch'),
('Qwen3（方案生成+记录审核）','MindIE多个规模推理；另30B-A3B TorchTitan训练','8B有ModelScope来源；30B训练需tokenizer，非同模型验收','推理对话样例；训练c4_test/C4','8B提供ModelTest性能命令，未给数值；30B未给正式结果','未给方案生成/记录审核质量基准；未见完整30B对齐结果','30B单机8卡默认OOM，建议减层到24，不能据此认定完整30B训练完成；8B镜像命名存在不一致','@MindIE/LLM/Qwen3/Qwen3-8B'),
('Qwen2.5-VL','VeRL内3B/7B/32B GRPO训练脚本和结果','test脚本内model path需自备对应Instruct权重','geo3k；VL数据准备入口','3B 8p523.599；7B 16p326.37；32B 32p70.0735（Throughput，表未注明单位）','主结果表报告性能与训练步数，未给VL任务正确率对照','可确认特定RL训练路径；不能当作通用推理和任务质量完整交付','@PyTorch/built-in/rl/VeRL_for_PyTorch'),
('Qwen2-VL（主归属：载荷视觉理解）','7B/72B MindIE镜像内置推理与服务','README路径占位，缺直接下载链接和revision/SHA','自备图片/视频，无固定质量集','7B A2样例：32G 43tokens/s；64G 98.79tokens/s，batch不同','未列载荷视觉理解或公开标准质量评测结果','示例请求model写internvl而服务为qwen2_vl；硬件卡数与HBM需分别核对','@MindIE/MultiModal/Qwen2-VL/Qwen2_vl_7b_instruct'),
('SAM','ViT-B encoder/decoder OM、AMG和COCO评估','sam_vit_b_01ec64.pth直链','COCO2017 box提示逐实例mIoU，自定义评测协议','300I Pro encoder4.43fps/decoder679.77fps；AMG45s/图','COCO mIoU0.7654；非论文官方统一指标','README支持说明DUO而结果表Pro；后处理可回CPU；提示协议须固定','cv/SAM'),
('SAM2','图像encoder/decoder OM与COCO评估','sam2.1_hiera_tiny.pt直链；表中简写sam2_hiera_tiny','COCO2017图像提示分割','300V Pro encoder bs1 15.64fps；decoder476.97fps','mIoU0.7815 vs GPU0.7830，差-0.0015','实际权重是SAM2.1 tiny；不证明视频时序记忆链路已适配','cv/SAM2'),
('MinerU3.3','仅找到MinerU2.0 Pipeline相关适配','PDF-Extract-Kit1.0组件权重；非3.3完整权重证据','相关版本OmniDocBench','相关版本DUO3.37s/A2 1.85s','相关版本overall81.68/81.51','上述数据不能赋给MinerU3.3；3.3代码、权重、精度性能均待补','ocr/MinerU'),
('RT-DETR','仅找到RT-DETRv2 OM导出/推理/COCO评估','v2指导仅指向上游目录，未锁具体checkpoint文件','v2 COCO val2017','v2 910B4 bs1 223FPS；T4 217FPS','v2 AP48.0/AP50 65.0；T4 48.1/65.1','第一代待确认；v2数据不可直接填给第一代','cv/RT-DETRv2'),
('Depth Anything V2','仅找到Depth Anything3相关实现','DA3-SMALL；不是V2权重','相关DA3 HiRoom/DA3-BENCH','相关DA3 A2 OM14ms，预热后20次均值','DA3与CPU输出一致性MAE；不是V2深度任务AbsRel/δ1','V2专项未找到；DA3一致性不能替代V2任务精度或推断加速比','cv/Depth-Anything-3'),
('DINOv2','GenPose++组件；RF-DETR/EoMT骨干引用','GenPose++明确下载dinov2_vits14_pretrain.pth','组件随姿态/检测任务评估；无独立DINOv2标准集证据','只有集成模型性能，不能归因DINOv2独立吞吐','集成任务指标不能替代DINOv2 kNN/linear probe/depth基准','由上一版未找到修正为组件级线索；独立特征模型完整验收待补','embodied_ai/GenPosePlus'),
]

def main():
    scope=json.loads((ROOT/'scope.json').read_text(encoding='utf-8'))
    evidence=json.loads((ROOT/'evidence.json').read_text(encoding='utf-8'))
    readmes=json.loads((ROOT.parents[1]/'.codex-reference/model-audit-readmes.json').read_text(encoding='utf-8'))
    tree=(ROOT/'repository-files.txt').read_text(encoding='utf-8').splitlines()
    facts={r[0]:r for r in FACTS}
    method={'BLASTn','BLUP','GBLUP','ssGBLUP','Bracken','AirSim','CBBA','DWA','Dijkstra','GPR（高斯过程回归）','Gazebo+PX4 SITL','ORB-SLAM3','PINN','LoRA/PEFT（高效微调）','LightGBM（备件需求预测）'}
    ambiguous={'Thinker','Thinker-VLA','Thinker-WM','CoordField（LLM集群任务编排）','DL-foils','MLOps训推一体框架','RALLY（LLM先验+MARL）'}
    external={'Qwen3-Embedding-8B','Qwen3-Reranker-8B','Qwen3-Embedding','Qwen3-Reranker','YOLOv8'}
    aliases={'Changformer':'ChangeFormer|Changformer','RosettaFold':'RoseTTAFold|RosettaFold','Anomaly Transformer（遥测异常）':'Anomaly[ _-]?Transformer','V-JEPA 2':'V[ _-]?JEPA[ _-]?2','ResUnet/ResUnet++':'ResUNet','Swin-Unet/U-Net':'Swin[ _-]?Unet|(?:^|/)UNet','fasterRCNN':'Faster[ _-]?(?:R[ _-]?)?CNN','Grounding DINO':'Grounding[ _-]?DINO','DINOv2':'DINO[ _-]?v2','Depth Anything V2':'Depth[ _-]?Anything[ _-]?V2'}
    cache_path=ROOT/'inventory.json'
    cache={r['原始条目']:r for r in json.loads(cache_path.read_text(encoding='utf-8'))} if cache_path.exists() else {}
    rows=[]
    for idx,name in enumerate(scope['models'],1):
        pattern=aliases.get(name,re.escape(name.split('（')[0]).replace(r'\ ',r'[ _-]?'))
        regex=re.compile(r'(?<![a-zA-Z0-9])(?:'+pattern+r')(?![a-zA-Z0-9])',re.I)
        old=cache.get(name)
        if old and old['检索表达式']==pattern:
            hits=old['检索命中'];filehits=old['文件名命中']
        else:
            hits=[{'path':s['path'],'line':i,'text':l.strip()[:650]} for s in readmes if regex.search(s['text']) for i,l in enumerate(s['text'].splitlines(),1) if regex.search(l)]
            filehits=[p for p in tree if regex.search(p)]
        file_count=old['文件名命中数'] if old and old['检索表达式']==pattern else len(filehits)
        common={'编号':idx,'原始条目':name,'本次实测状态':'未做模型运行；S0-S4无法完整定级','文件名命中数':file_count,'README命中数':len(hits)}
        if name in facts:
            _,code,w,data,perf,acc,limit,suffix=facts[name]
            directory=suffix[1:] if suffix.startswith('@') else 'ACL_PyTorch/built-in/'+suffix
            path=directory+'/README.md'
            if path not in tree and directory.endswith('/LSTM'):path=directory+'/ReadMe.md'
            status='仓内专项实现/指导'
            if name in {'MinerU3.3','RT-DETR','Depth Anything V2'}:status='仅相邻版本'
            if name=='DINOv2':status='组件级证据'
            if name=='Swin-Unet/U-Net':status='组合条目部分覆盖'
            if name=='Swin+FPN':status='具体组合已找到'
            refs=[path]
            refs.extend(p for p in scope['references'] if p in tree and ((name=='Pi-0.5' and 'pi05_' in p) or (name=='PaddleOCR' and 'PPOCRv' in p) or (name=='Grounding DINO' and 'GroundingDINO' in p) or (name=='LSTM' and '/LSTM' in p) or (name.startswith('Qwen3（') and '/Qwen3' in p) or (name=='InternVL2' and '/InternVL2/' in p) or (name.startswith('Qwen2-VL（') and '/Qwen2-VL/' in p) or (name=='DeepLabv3+' and ('DeepLabv3+' in p or 'DeeplabV3+' in p)) or (name=='Swin-Unet/U-Net' and '/UNet' in p)))
            primary=next((e for e in evidence['evidence'] if e['directory']==directory),None)
            if primary:
                codefiles=[p for p in primary['files'] if p.endswith(('.py','.sh','.patch')) and p.count('/')<=directory.count('/')+2]
                preferred=[p for p in codefiles if re.search(r'infer|metric|train.*full|diff.patch|eval',p.rsplit('/',1)[-1],re.I)]
                refs += (preferred or codefiles)[:2]
            urls=[BASE+quote(p,safe='/') for p in dict.fromkeys(refs)]
        elif name in external:
            status='仓内索引外链，非仓内实现';code='ACL_PyTorch/README.md明确索引外部指导'
            data='Qwen系列外部教程提供MTEB评估方法；YOLOv8外部正文本次获取失败'
            w='Qwen教程列0.6B/4B/8B权重来源；YOLOv8未核实'
            perf='Qwen提供benchmark命令，未见8B固定条件实测表；YOLOv8未核实'
            acc='Qwen提供任务评估方法，未见8B固定结果；YOLOv8未核实'
            limit='不能写成完全无适配证据；也不能归入本仓代码交付或8B实测通过'
            ext='https://gitcode.com/Ascend/modelzoo-GPL/blob/master/built-in/ACL_Pytorch/Yolov8_for_PyTorch/README.md' if name=='YOLOv8' else 'https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/Qwen3-'+('Reranker' if 'Reranker' in name else 'Embedding')+'.html'
            urls=[BASE+'ACL_PyTorch/README.md',ext]
        else:
            status='方法/软件/系统，需指定实现' if name in method else ('对象不唯一，需原项目链接' if name in ambiguous else '未找到仓内专项适配证据')
            code='完整文件树与README检索未定位匹配专项交付；命中不等于适配'
            w='本仓未定位对应权重准备证据；未审查全网原模型权重可用性'
            data='本仓未定位该对象正式数据准备及固定split证据'
            perf='本仓未定位对应NPU性能结果'
            acc='本仓未定位对应NPU精度/质量对照'
            limit='不代表昇腾生态不支持；不能用同名、底座、相邻版本或第三方框架代替'
            if name in method:
                w='依具体实现判断；传统算法和从头训练不要求预训练权重'
                limit='应核对CPU实现/依赖库/学习组件边界，先确定是否需要NPU'
            if name in {'AlphaFold3','AlphaFold-Multimer'}:
                limit='OpenFold训练目录不能证明AlphaFold3或Multimer适配'
            urls=['https://gitcode.com/Ascend/ModelZoo-PyTorch/tree/'+SHA]
        rows.append(common|{'仓库证据分类':status,'代码':code,'权重':w,'数据集':data,'性能基准':perf,'精度基准':acc,'分析与缺口':limit,'具体证据链接':' ; '.join(urls),'检索表达式':pattern,'检索命中':hits,'文件名命中':filehits[:80]})
    assert len(rows)==len(scope['models'])==119
    assert len({r['原始条目'] for r in rows})==119
    (ROOT/'inventory.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8', newline='\n')
    fields=[k for k in rows[0] if k not in ('检索命中','文件名命中')]
    with (ROOT/'inventory.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(rows)
    counts=Counter(r['仓库证据分类'] for r in rows)
    text=['# 119项模型及系统昇腾适配证据核查','',f'检查日期：2026-10-22（Asia/Shanghai）。唯一主依据：[Ascend/ModelZoo-PyTorch固定提交](https://gitcode.com/Ascend/ModelZoo-PyTorch/tree/{SHA})。', '',f'实时执行 `git ls-remote ... refs/heads/master` 得到 `{SHA}`；浅克隆HEAD一致。完整文件树{len(tree):,}项，检索README/模型清单{len(readmes):,}份，重点提取39个目录、530份源码/补丁/依赖/说明文本。范围为原会话全部119项，并非审查ModelZoo全部模型。','',
    '## 结论与证据边界','',
    '这是一份仓库与公开文档审计，不是上机验收报告。未连接NPU、未安装上游依赖、未下载大权重或正式数据集、未执行模型推理和clean-room重放。所有性能/精度数字均为目标仓文档声明，未独立复测；有命令不等于执行成功，有链接不等于已验证下载、许可或完整性。S0-S4不能仅由目录存在赋值，本次各模型统一保留“无法完整定级”。','',
    '区分仓内实现、仓内索引外链、组件级证据、相邻版本、未找到和对象待明确。未找到仅针对固定快照下的专项交付，不能推导全昇腾生态未适配。外部文档只追踪主仓明确索引，不用于填补其他缺失模型。未开展未收录模型的全网源码/权重/论文/数据集独立可行性调查；相应字段明确为本仓证据缺口。','',
    '| 分类 | 原始条目数 |','|---|---:|']
    text += [f'| {k} | {v} |' for k,v in counts.items()]
    text += ['', '## 对上一版的重要修正','',
    '- Qwen3-Embedding、Qwen3-Reranker（含8B条目）与YOLOv8在ACL总README有明确外链，不能写成仓内完全无线索；外链不等于代码在本仓或8B已验收。旧Qwen链接跳转到教程索引，本次进一步读取新的专项教程。',
    '- Swin+FPN已有Mask R-CNN Swin-S FPN明确权重和配置，NPU bbox mAP47.8，对照48.2。上一版只查UPerNet，漏掉该检测组合；具体业务若要求语义分割仍须另核。',
    '- DINOv2作为GenPose++组件、RF-DETR/EoMT骨干已有线索；没有独立DINOv2标准任务验收证据，不能简单划入完全未适配。',
    '- SAM2指导实际下载sam2.1_hiera_tiny.pt，结果表名称简写sam2_hiera_tiny；只覆盖图像路径，视频链路未证实。',
    '- LLaVA目录已声明停止维护；Qwen3-30B指导明确默认8卡OOM并建议减层，不能把减层测试算完整30B训练验收。','',
    '## 全量核查表','',
    '性能单位缺失或含义不清时保留“表值”，不擅自转换；示例、随机输入及集成模型结果均不作为完整正式验收。权重/数据集链接及精确原文在后面的逐目录证据摘录中。','',
    '| # | 原始条目 | 分类 | 代码 | 权重 | 数据集 | 性能基准（文档） | 精度基准（文档） | 分析与缺口 | 具体证据 |','|---:|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        cells=[str(r['编号']),r['原始条目'],r['仓库证据分类'],r['代码'],r['权重'],r['数据集'],r['性能基准'],r['精度基准'],r['分析与缺口']]
        cells=[c.replace('|','/').replace('\n',' ') for c in cells]
        cells.append('、'.join(f'[证据{i+1}]({u})' for i,u in enumerate(r['具体证据链接'].split(' ; '))))
        text.append('| '+' | '.join(cells)+' |')
    text += ['', '## 验收缺口与下一步','',
    '1. 代码：固定upstream commit、适配patch及实际NPU路径；核查README命令引用的文件和参数。镜像内置实现应记录镜像digest、组件版本与可获取条件。',
    '2. 权重：固定具体变体、revision、config/tokenizer/辅助模型和SHA256；权重下载先以curl/wget检查响应类型、大小与校验。只提供仓库首页或路径占位不足以证明可复现。',
    '3. 数据：记录官方dataset、split、样本数、许可、预后处理和manifest；训练集评测、示例集、mock与正式测试集分开。PPO/MAPPO以固定在线环境和seed替代下载式数据集。',
    '4. 精度：同checkpoint、同split、同evaluator比较公开官方基准与NPU；不可比时补CPU精度基线。任务成功率需多seed/足够episodes，不能把少量成功率差异当性能优势。',
    '5. 性能：在真实目标NPU和固定输入上至少独立3次取中位数；记录batch、dtype、shape、warmup、同步和纯模型/E2E边界。不用CPU结果推断加速比。',
    '6. 门禁：本次未准备上库候选，因此未运行模型交付结构门禁或声称target_ready；未执行权重加载、官方evaluator、NPU精度/性能及clean-room。候选就绪必须另验许可证、贡献要求、自包含README和干净环境重放。','',
    '## 可复查资料','',
    '- `inventory.csv`：119行，可用表格软件筛选；`inventory.json`：每条检索表达式、README命中行及文件名命中。',
    '- `repository-files.txt`：完整95,062项文件树，可核对阴性结果边界。',
    '- `source-evidence.md`：逐目录代码文件清单、原文行号、精确权重/数据URL、性能与精度原表摘录；每份源文件附SHA256。',
    '- `code-findings.md`：重点代码与README交叉检查，包含三输出分支、计时工作量、在线数据依赖、模型名错配、减层与环境安装问题。',
    '- `collect.py`、`build_report.py`：只读采集与表格生成。原始全文快照保留本地审计工作区，不作为正式模型候选源码提交。',
    '- 复查：`python reports/ascend-audit-20261022/collect.py` 后 `python reports/ascend-audit-20261022/build_report.py`；需先有上述固定目标仓clone，以及本次获取的README索引快照。','']
    (ROOT/'REPORT.md').write_text('\n'.join(text),encoding='utf-8', newline='\n')
    sources=[]
    for item in evidence['evidence']:
        sources += [f'## {item["directory"]}', '', '文件清单：', '', '```text',*item['files'],'```','']
        for source in item['sources']:
            p=source['path'];content=source['text']
            if p.rsplit('/',1)[-1].lower() not in ('readme.md','readme_en.md'):continue
            sources += [f'### {link(p,p)}','',f'文本SHA256：`{hashlib.sha256(content.encode()).hexdigest()}`','']
            for i,line in enumerate(content.splitlines(),1):
                if re.search(r'https?://|权重|数据集|\|.*\d|精度|性能|不支持|停止维护|OOM|CPU|fallback',line,re.I):
                    sources.append(f'- L{i}: '+line.replace('\t',' ').strip())
            sources.append('')
    for p in ['ACL_PyTorch/README.md','ACL_PyTorch/contrib/cv/detection/Mask_Rcnn_SwinS/README.md','ACL_PyTorch/built-in/embodied_ai/GenPosePlus/README.md','ACL_PyTorch/built-in/cv/RF-DETR/README.md','ACL_PyTorch/built-in/cv/EoMT/README.md']:
        s=next(x for x in readmes if x['path']==p)
        sources += [f'## 补充证据 {link(p,p)}','']
        sources += [f'- L{i}: {l.strip()}' for i,l in enumerate(s['text'].splitlines(),1) if re.search(r'https?://|DINO|FPN|fpn|\|.*\d',l)]
        sources += ['']
    (ROOT/'source-evidence.md').write_text('\n'.join(sources),encoding='utf-8', newline='\n')
    print(json.dumps(dict(counts),ensure_ascii=False))

if __name__=='__main__':
    main()
