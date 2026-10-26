"""Public evidence audit. No model execution or readiness certification."""
import csv
import json
from pathlib import Path
from urllib.parse import unquote, quote
from device_dimensions import attach
from reader_view import display, status, note

ROOT=Path(__file__).resolve().parent
MZ='dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202'
DR='27375a9c561578d6aee7270edfac6ae7955942f5'
PX='ffb64904d23708863ff5b8da312a5cbd52a7f462'
LS='27065e4053aa860774b9a005e6a5307854d062a8'
FW='5a7f8443a84bad1125298662d279436fb3edfe07'
FIELDS=['代码','权重','数据集','性能基准','精度基准']

def e(text,urls=(),role='来源报告；未独立复测'):
    return {'说明':text,'链接':list(dict.fromkeys(urls)),'证据类型':role}

def hf(name,data=False):
    return 'https://huggingface.co/'+('datasets/' if data else '')+name

def make_rows():
    scope=json.loads((ROOT/'scope.json').read_text(encoding='utf-8'))['models']
    old=json.loads((ROOT/'modelzoo-snapshot.json').read_text(encoding='utf-8'))
    tree=(ROOT/'repository-files.txt').read_text(encoding='utf-8').splitlines()
    rows=[]
    for i,name in enumerate(scope,1):
        r={'编号':i,'原始条目':name,'核查日期':'2026-10-23','结论':'未取得充分专项NPU证据','分析与缺口':'未找到不等于不支持；原始资源不证明NPU适配。','本次实测状态':'未做模型运行；不认证S3/S4；未校验权重与数据SHA','证据':{k:e('未取得充分专项NPU证据') for k in FIELDS}}
        p=old[i-1]
        if p['仓库证据分类'] in ['仓内专项实现/指导','具体组合已找到','组合条目部分覆盖']:
            urls=[u for u in p['具体证据链接'].split(' ; ') if '/blob/' in u]
            r['结论']='有指定实现/指导，非完整验收认证'
            r['分析与缺口']=p['分析与缺口']
            r['证据']={k:e(p[k],urls,'固定快照文档；未复测') for k in FIELDS}
            files=[]
            for u in urls:
                if f'/blob/{MZ}/' not in u:continue
                parent=unquote(u.split(f'/blob/{MZ}/')[1]).rsplit('/',1)[0]
                files += [x for x in tree if x.startswith(parent+'/') and x.endswith(('.py','.patch','.sh')) and x.count('/')<=parent.count('/')+2]
            files=sorted(set(files),key=lambda x:(not x.endswith('.patch'),len(x),x))[:4]
            if files:r['证据']['代码']=e(p['代码'],['https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/'+MZ+'/'+quote(x,safe='/') for x in files],'交付文件存在；未运行')
        rows.append(r)

    def put(n,src,texts,resources=None,boundary='来源报告未复测；缺完整资产SHA、正式任务对齐或三次独立NPU测量时不认证完成。'):
        r=rows[n-1];r['结论']='有公开实现/指导，完整验收未认证';r['分析与缺口']=boundary
        r['证据']={k:e(t,[] if t.startswith('未找到') else [src]) for k,t in zip(FIELDS,texts)}
        for k,urls in (resources or {}).items():r['证据'][k]['链接']=urls
    def driving(n,folder,w,d,p,a,resources=None):
        base=f'https://github.com/Ascend/DrivingSDK/tree/{DR}/model_examples/{folder}'
        put(n,base.replace('/tree/','/blob/')+'/README.md',['专项patch/test交付目录；未逐文件运行',w,d,p,a],resources)
        rows[n-1]['证据']['代码']=e('固定提交专项交付目录；非逐文件验收',[base],'源码目录')
    driving(4,'Cosmos-Predict2','NVIDIA各变体及T5/guardrail','Cosmos-NeMo-Assets','未找到完整生成吞吐表','500步loss0.0409/0.0402，非视频质量',{'权重':[hf('nvidia/Cosmos-Predict2-2B-Video2World')],'数据集':[hf('nvidia/Cosmos-NeMo-Assets',True)]})
    driving(5,'Cosmos-Transfer1','7B及辅助模型，部分需授权','HD-VILA/Waymo子集','文档有迭代耗时，非统一流水线性能','训练loss表，未找到完整生成质量')
    driving(6,'Cosmos-Reason1','Reason1-7B','SFT/RL数据','A2八卡SFT16.57s、RL43.58s/迭代','loss SFT0.380/0.378、RL1.9219/1.9688，非任务正确率',{'权重':[hf('nvidia/Cosmos-Reason1-7B')],'数据集':[hf('nvidia/Cosmos-Reason1-SFT-Dataset',True),hf('nvidia/Cosmos-Reason1-RL-Dataset',True)]})
    driving(7,'Cosmos-Drive-Dreams','多个组件，分别准备','地图、视频、LiDAR资产','未找到完整性能表','未找到完整质量对照')
    driving(13,'GR00T-N1.5','N1.5-3B','SO101 table-cleanup','A2八卡337.35 FPS，对照276.38','loss0.0031/0.0032，非任务成功率',{'权重':[hf('nvidia/GR00T-N1.5-3B')],'数据集':[hf('youliangtan/so101-table-cleanup',True)]})
    driving(14,'GR00T-N1.6','N1.6-3B','LIBERO；推理任务另列','训练449/457 FPS；指定推理0.145/0.147s','loss0.0082/0.0084；MSE1.185/1.179',{'权重':[hf('nvidia/GR00T-N1.6-3B')]})
    driving(19,'Lingbot-vla','4B/Depth及辅助模型，不覆盖2.0','RoboTwin2 LeRobot，A3不能生成渲染数据','A3八卡301.76 FPS，对照114.74','未找到完整任务成功率',{'权重':[hf('robbyant/lingbot-vla-4b')]})
    driving(20,'OpenVLA','openvla-7b-prismatic','Bridge TFDS','A2八卡56.14 FPS，对照73.12','loss0.1876/0.1873，非任务成功率',{'权重':[hf('openvla/openvla-7b-prismatic')],'数据集':['https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/']})
    driving(22,'Pi-0.5','pi05_base及PaliGemma；与端侧推理路线分开','koch_test','A3八卡155.1 FPS，对照70.8；global batch128/64，不是同条件加速比','loss0.004/0.005；batch不同',{'权重':[hf('lerobot/pi05_base'),hf('google/paligemma-3b-pt-224')],'数据集':[hf('jianqiang03/koch_test',True)]})
    driving(27,'StarVLA','Qwen3-VL-4B-Instruct','RoboTwin-Clean/Randomized','A3八卡123.08 FPS，对照49.61','部分任务88.38/87.22，非完整50任务',{'权重':[hf('Qwen/Qwen3-VL-4B-Instruct')],'数据集':[hf('StarVLA/RoboTwin-Clean',True),hf('StarVLA/RoboTwin-Randomized',True)]})
    driving(119,'VGGT','按指导准备checkpoint','CO3D仅tv场景','A2八卡25.04 FPS，对照15.30','loss0.0165/0.0164，非全量几何指标')
    base=f'https://gitcode.com/wangguiqin/FastWAM-NPU/blob/{FW}/'
    put(1,base+'VALIDATION_REPORT.md',['NPU训练、评测及优化代码','上游公开权重；优化评测权重仅服务器路径','LIBERO200配对episodes；继续训练120步','4×910B3：20.2706→33.2963 samples/s；单窗口，不是三次中位数','184/200→185/200；差值95%区间[-1.5,+2.5]个百分点，不证明提升'],{'代码':[base+'src/fastwam/runtime.py',base+'scripts/eval_npu_suite.py'],'权重':[hf('yuanty/fastwam')],'数据集':[hf('yuanty/LIBERO-fastwam',True)],'性能基准':[base+'validation/baseline_workers8/performance.json',base+'validation/overlap_final/performance.json'],'精度基准':[base+'validation/accuracy_comparison.json']},'NPU原版/优化版继续训练对照，不是CUDA迁移或完整从头收敛对照；原始JSON已读取，未独立复测。')
    extras=[
      (26,'https://gitcode.com/gcw_1yOcCgy7/310p-spiritv1.5-infer',['社区310P功能实现','加载准备说明','接口样例','约2.5s早期观测，非benchmark','未找到任务质量对照']),
      (50,'https://gitcode.com/xiaotao5571/BLUP-NPU',['社区矩阵卸载','不需要预训练权重','X/y及随机矩阵','kernel结果，非端到端','相对误差约1.3e-6，非预测质量']),
      (51,'https://gitcode.com/xiaotao5571/NPU',['rrBLUP衍生Python/R桥接','不需要预训练权重','wheat599×1279','G矩阵上NPU，完整流水线加速未证明','对R cor0.9999999999、RMSE9.75e-6']),
      (52,'https://gitcode.com/xiaotao5571/ssGBLUP-npu_4212',['animal-BLUP/MME/PCG，完整ssGBLUP未证实','不需要预训练权重','小规模系谱测试','局部求解，缺完整真实规模性能','局部数值对照，不认定完整ssGBLUP']),
      (58,'https://ai.gitcode.com/gcw_93rlw6ed/esm2_t33_650M_UR50D-npu',['650M社区功能实现','ESM2 t33 650M','示例序列','未找到正式性能表','fill-mask样例，非下游任务']),
      (65,'https://gitcode.com/Ascend-SACT/ThermoMPNN/blob/34e25f601bebfa70a31e91d85af838a67a409460/README.md',['910C专项指导；源码未逐文件核验','ProteinMPNN骨干及预测头','MegaScale/FireProt','未找到完整正式性能表','2OCJ样例；缺全测试集NPU Spearman']),
      (84,'https://gitcode.com/xujiashuai/dinov2-large',['社区large推理/evaluate','DINOv2-large','特征样例','约12.18ms，样例条件','余弦0.999936，非下游任务']),
      (88,'https://ai.gitcode.com/Ascend-SACT/vipe-v1.0',['NPU集成组件线索，独立设备边界未确认','V2组件权重','未找到独立NPU测试集','未找到独立性能','未找到AbsRel/δ1对照']),
      (107,'https://ai.atomgit.com/Ascend-SACT/PatchCore',['社区PyTorch NPU/Faiss CPU','WideResNet及memory bank','MVTec AD','E2E约2.151 FPS，CPU/NPU混合','图像AUROC0.99053、像素0.98123']),
      (98,'https://github.com/huggingface/peft/blob/main/src/peft/utils/other.py',['PEFT有NPU设备选择，非所有组合通过','依底座/adapter','依具体任务','未找到统一基准','未找到统一精度']),
      (99,'https://www.mindspore.cn/docs/zh-CN/r2.2/note/official_models.html',['历史官方Ascend支持记录，非源码运行','从头训练','simple_spread','未找到当前栈完整数表','历史指定环境回报-140']),
      (108,'https://www.mindspore.cn/docs/zh-CN/r2.3.1/note/official_models.html',['官方Ascend支持记录','从头训练','SMAC/SimpleSpread','未找到统一可比性能','指定SMAC胜率90%，环境需固定']),
      (104,'https://www.mindspore.cn/docs/en/r2.5.0/model_train/models/official_models.html',['官方Ascend PINN实例，非整个方法族','依方程训练','方程/边界/采样点','按实例，无统一性能','按实例，无统一精度']),
      (93,'https://support.huaweicloud.com/bestpractice-modelarts/ModelArts最佳实践（中国站）-pdf.pdf',['官方平台支持指导，非源码逐文件核验','各规模分别准备','未找到固定NPU评测集','未找到完整数表','未找到各规模独立对齐'])]
    for n,src,texts in extras:put(n,src,texts)
    for n,folder,texts in [
      (59,'LigandMPNN',['固定上游及NPU patch指导','get_model_params.sh','示例PDB','A2单步1.332s，对照1.303s','未找到正式恢复率对照']),
      (61,'ProteinMPNN',['固定上游及优化patch指导','训练产生checkpoint','pdb_2021aug02','A3单步0.28s，950DT0.15s，对照0.54s','2034步valid_acc0.324，与对照相同']),
      (62,'RFdiffusion2',['NPU/PyG指导，非全pipeline认证','checkpoint及辅助组件','demo及AME入口','未找到完整pipeline性能','算子比较不能代替设计质量']),
      (63,'RFdiffusion_pyg',['910B/PyG专项指导','原始权重准备','设计demo','未找到完整正式性能','核心算子误差<1e-4，非设计成功率'])]:put(n,f'https://gitcode.com/AI4Science/LifeScience/blob/{LS}/PyTorch/{folder}/README.md',texts)
    put(55,'https://github.com/mindspore-ai/mindscience/blob/971c015b0111d229608ff05b9e77704dbb0793b4/MindSPONGE/README.md',['官方AF3推理发布记录','需申请参数授权','MSA/模板数据库，完整split未核实','性能优化声明，缺统一条件表','未找到完整结构质量对照'])
    put(56,'https://www.mindspore.cn/mindsponge/docs/en/stable/user/structure_prediction.html',['官方Ascend推理支持，训练不支持','模型卡参数准备','多链输入/MSA/模板','未找到长度分档性能','未找到完整多聚体质量对照'])
    for n,kind in [(42,'Embedding'),(43,'Reranker'),(44,'Embedding'),(45,'Reranker')]:
        put(n,f'https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/Qwen3-{kind}.html',['官方部署评测教程，非完整源码核验','0.6B/4B/8B','MTEB；示例规模与8B分开','有命令，未找到8B完整固定条件数表','有命令，未找到8B完整质量对照'],{'权重':[hf(f'Qwen/Qwen3-{kind}-8B')]})
    put(46,'https://opendatalab.github.io/MinerU/zh/usage/acceleration_cards/Ascend/',['官方Ascend指导，仅保证MinerU<4','3.3模型版本见发布记录，文件未逐项校验','OmniDocBench1.6，需冻结backend/effort','未找到3.3固定后端完整NPU表','一致性声明，非3.3各后端独立基准'],boundary='撤销仅相邻2.0结论；不外推4.0。动态文档未冻结。')
    px=f'https://github.com/PaddlePaddle/PaddleX/blob/{PX}/docs/support_list/model_list_npu.en.md'
    put(113,px,['第一代RT-DETR官方NPU清单，不是v2','H/L/R18/R50/X训练/推理权重','COCO对应任务','未找到第一代完整独立NPU数表','不借用v2 AP48.0'])
    put(116,px,['TimesNet官方NPU清单','训练/推理权重','ETTH1；分类另列','未找到NPU独立耗时；原表是GPU/CPU','发布模型MSE0.416、MAE0.429，非NPU独立对齐'],{'权重':['https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/TimesNet_infer.tar']})
    originals={
      2:'https://github.com/chenjinyubuaa/AerialVLA',3:'https://airvla.github.io/',9:'https://github.com/OpenGalaxea/GalaxeaVLA',11:'https://github.com/OpenGalaxea/GalaxeaVLA',12:'https://github.com/OpenGalaxea/GalaxeaVLA',15:'https://github.com/NVIDIA/Isaac-GR00T',16:'https://github.com/open-gigaai/giga-world-1',17:'https://github.com/open-gigaai/giga-brain-0',18:'https://github.com/open-gigaai/giga-world-0',21:'https://github.com/Open-X-Humanoid/pelican-vl',24:'https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/',25:'https://spirit-ai.com/news/34',28:'https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0622/2026062201645_c.pdf',29:'https://stockn.xueqiu.com/09880/20260414835102.pdf',30:'https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0622/2026062201645_c.pdf',31:'https://github.com/X-Square-Robot/wall-x',32:'https://x2robot.com/',33:'https://x2robot.com/',34:'https://shalfun.github.io/',35:'https://x2robot.com/',36:'https://huggingface.co/X-Humanoid/WoW-1-Wan-1.3B-2M',37:'https://github.com/EmbodiedCity/Worldscape-MoE.code',38:'https://github.com/Open-X-Humanoid/XR-1',49:'https://www.osc.edu/book/export/html/2793',53:'https://github.com/jenniferlu717/Bracken',54:'https://github.com/RosettaCommons/DeepAb',57:'https://github.com/evolutionaryscale/esm',60:'https://openvax.github.io/mhcflurry/commandline_tools.html',64:'https://github.com/RosettaCommons/RoseTTAFold',66:'https://github.com/justchenhao/BIT_CD',67:'https://github.com/wgcban/ChangeFormer',74:'https://github.com/DebeshJha/ResUNetPlusPlus',75:'https://www.sciencedirect.com/science/article/pii/S1369800126002477',77:'https://github.com/PaddlePaddle/PaddleSeg',78:'https://github.com/microsoft/AirSim',79:'https://github.com/thuml/Anomaly-Transformer',81:'https://github.com/ifzhang/ByteTrack',83:'https://arxiv.org/abs/2505.00091',87:'https://github.com/ros-planning/navigation/tree/noetic-devel/dwa_local_planner',90:'https://github.com/danijar/dreamerv3',92:'https://docs.px4.io/main/en/simulation/',97:'https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html',101:'https://github.com/facebookresearch/map-anything',102:'https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets',103:'https://github.com/UZ-SLAMLab/ORB_SLAM3',112:'https://arxiv.org/abs/2507.01378',117:'https://github.com/DurYi/UA-NWM',118:'https://github.com/facebookresearch/vjepa2'}
    for n,url in originals.items():
        r=rows[n-1];r['证据']['代码']=e('仅原始项目/发布材料；未取得充分专项NPU代码',[url],'原始资源，不证明NPU支持')
        for k in ['权重','数据集']:r['证据'][k]=e('原始资源入口；精确文件、许可和可用性未逐项闭合，不认定缺失',[url],'入口，非已验证文件')
    for n in [39,86,100]:rows[n-1]['结论']='对象未唯一确认'
    for n in [49,53,78,82,87,89,91,92,97,103]:rows[n-1]['证据']['权重']=e('算法/平台通常无需统一预训练权重；依具体实现',[],'不适用/待指定实现')
    rows[7]['结论']='系列名称；按具体成员核查'
    rows[9]['结论']='仅系统支持名单线索，缺完整专项交付'
    rows[9]['证据']['代码']=e('列名不等于可复现适配',['https://sjtu-ipads.github.io/Ascend-Embody-Web/'],'名单线索')
    rows[48]['证据']['代码']=e('搜索命中的是Ohio同名Ascend集群，不是华为NPU；不构成适配证据',['https://www.osc.edu/book/export/html/2793'],'排除同名误匹配')
    resource_links={
      16:(None,'open-gigaai/Giga-World-1','open-gigaai/Giga-World-1-Toydata'),
      36:(None,'X-Humanoid/WoW-1-Wan-1.3B-2M',None),
      37:(None,'EmbodiedCity/Worldscape-MoE','EmbodiedCity/Worldscape-MoE-Dataset'),
      38:(None,None,'X-Humanoid/XR-1-Dataset-Sample'),
      40:(None,'BAAI/bge-m3',None),41:(None,'BAAI/bge-reranker-v2-m3',None),
      58:(None,'facebook/esm2_t33_650M_UR50D',None),
      117:(None,'DurYi/UA-NWM-Checkpoints','DurYi/AirGoal-10k')}
    for n,(_,weight,data) in resource_links.items():
        for k,value in [('权重',weight),('数据集',data)]:
            if value:rows[n-1]['证据'][k]=e('具体原始资源地址；未下载校验SHA/授权，不独立证明NPU适配',[hf(value,k=='数据集')],'模型卡/数据仓入口')
    direct={
      (23,'权重'):'https://huggingface.co/BrunoM42/pi0_aloha_transfer_cube/tree/main',
      (47,'权重'):'https://modelscope.cn/models/PaddlePaddle/PaddleOCR-VL-1.6',
      (54,'权重'):'https://data.graylab.jhu.edu/ensemble_abresnet_v1.tar.gz',
      (61,'数据集'):'https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz',
      (62,'权重'):'https://modelers.cn/panxiaomu/rfdiffusion2-weights',
      (96,'权重'):'https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/LSTM/PTH/ctc_best_model.pth',
      (114,'权重'):'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
      (115,'权重'):'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt'}
    for (n,k),url in direct.items():rows[n-1]['证据'][k]['链接']=[url];rows[n-1]['证据'][k]['证据类型']='具体下载/资源地址；未校验文件SHA'
    return rows

def cell(v):
    links=' '.join(f'[证据{i}]({u})' for i,u in enumerate(v['链接'],1))
    return display(v).replace('|','/').replace('\n',' ')+('；'+links if links else '')

def main():
    from gpu_research import apply as apply_gpu
    rows=apply_gpu(attach(make_rows()))
    updates=json.loads((ROOT/'evidence_updates.json').read_text(encoding='utf-8'))
    for update in updates['updates']:
        row=rows[update['number']-1]
        row.setdefault('设备核查日期',{'GPU':row['核查日期'],'NPU':row['核查日期']})[update['device']]=update['date']
        row['核查日期']=update['date']
        row['设备维度'][update['device']].update(update['fields'])
        if update.get('conclusion'):
            row[update['device']+'结论']=update['conclusion']
        if update.get('note'):row['分析与缺口']=update['note']
    for number in updates.get('deferred_items', []):
        row=rows[number-1]
        row['GPU结论']=row['NPU结论']='按用户要求暂缓核查'
        row['分析与缺口']='名称尚不能唯一定位；用户表示不知道对应项目，先不核查。保留原始条目和已有检索记录。'
    (ROOT/'inventory.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    cols=['编号','原始条目','GPU结论','NPU结论']
    for device in ['GPU','NPU']:
        for field in FIELDS:cols += [device+'_'+field,device+'_'+field+'_证据链接']
    cols += ['注意事项']
    with (ROOT/'inventory.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,lineterminator='\n');w.writeheader()
        for r in rows:
            out={'编号':r['编号'],'原始条目':r['原始条目'],'GPU结论':status(r,'GPU'),'NPU结论':status(r,'NPU'),'注意事项':note(r)}
            for device in ['GPU','NPU']:
                for field in FIELDS:
                    v=r['设备维度'][device][field]
                    out[device+'_'+field]=display(v)
                    out[device+'_'+field+'_证据链接']=' ; '.join(v['链接'])
            w.writerow(out)
    lines=[f'''# 119项模型GPU/NPU支持情况

全量基线核查：2026-10-23；后续增补：{updates['checked_date']}。原始119项；CSV一项一行，GPU/NPU分列；本文每项分GPU/NPU两行。
其中DL-foils、MLOps训推一体框架两项按用户要求暂缓核查，保留原始位置，其余117项继续核查。
ModelZoo master `{MZ}`；DrivingSDK master `{DR}`；PaddleX release/3.7 `{PX}`。

数值取自公开资料，未上机复测。GPU侧已重新逐项检索；找到代码不表示五项全部齐备。“待核对/未核实”表示本轮尚未确认，不能解释成网上没有。平台、方法和具体模型分开判断。原始对照表和完整核查记录保存在inventory.json。

| # | 模型 | 设备 | 代码 | 权重 | 数据集 | 性能基准 | 精度基准 | 结论与限制 |
|---:|---|---|---|---|---|---|---|---|
''']
    for r in rows:
        for device in ['GPU','NPU']:
            lines.append('| '+' | '.join([str(r['编号']),r['原始条目'],device,*[cell(r['设备维度'][device][k]) for k in FIELDS],status(r,device)+('；'+note(r) if note(r) else '')])+' |\n')
    (ROOT/'REPORT.md').write_text(''.join(lines),encoding='utf-8',newline='\n')

if __name__=='__main__':main()
