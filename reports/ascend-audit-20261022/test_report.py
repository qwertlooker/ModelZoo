"""Regression checks on coverage, evidence roles and benchmark attribution."""
import csv
import json
import pathlib
import subprocess
import sys
import unittest
from urllib.parse import unquote
ROOT=pathlib.Path(__file__).resolve().parent
class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=json.loads((ROOT/'inventory.json').read_text(encoding='utf-8'))
        cls.scope=json.loads((ROOT/'scope.json').read_text(encoding='utf-8'))['models']
    def test_119_items_preserved_in_order(self):
        self.assertEqual(len(self.rows),119)
        self.assertEqual([r['编号'] for r in self.rows],list(range(1,120)))
        self.assertEqual([r['原始条目'] for r in self.rows],self.scope)
    def test_every_dimension_has_text_role_and_urls(self):
        for r in self.rows:
            self.assertEqual(set(r['证据']),{'代码','权重','数据集','性能基准','精度基准'})
            for v in r['证据'].values():
                self.assertTrue(v['说明']);self.assertTrue(v['证据类型'])
                self.assertIsInstance(v['链接'],list)
                self.assertTrue(all(u.startswith('https://') for u in v['链接']))
    def test_no_runtime_certification(self):
        for r in self.rows:
            self.assertIn('未做模型运行',r['本次实测状态'])
            self.assertNotIn('验收通过',r['结论'])
    def test_specific_attribution_regressions(self):
        self.assertIn('128/64',self.rows[21]['证据']['性能基准']['说明'])
        self.assertIn('GPU/CPU',self.rows[115]['证据']['性能基准']['说明'])
        self.assertIn('不是三次中位数',self.rows[0]['证据']['性能基准']['说明'])
        self.assertIn('不证明提升',self.rows[0]['证据']['精度基准']['说明'])
        self.assertIn('完整ssGBLUP未证实',self.rows[51]['证据']['代码']['说明'])
        self.assertIn('MinerU<4',self.rows[45]['证据']['代码']['说明'])
        self.assertIn('第一代',self.rows[112]['证据']['代码']['说明'])
        self.assertIn('RefCOCO',self.rows[84]['证据']['精度基准']['说明'])
    def test_original_sources_not_npu_proof(self):
        for n in [2,3,9,15,24,28,32,53,74,81,83,103,112,117,118]:
            r=self.rows[n-1]
            self.assertEqual(r['证据']['代码']['证据类型'],'原始资源，不证明NPU支持')
            self.assertNotIn('有公开实现',r['结论'])
    def test_fixed_modelzoo_files_exist(self):
        tree=set((ROOT/'repository-files.txt').read_text(encoding='utf-8').splitlines())
        marker='/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/'
        for r in self.rows:
            for v in r['证据'].values():
                for url in v['链接']:
                    if marker in url:self.assertIn(unquote(url.split(marker)[1]),tree)
    def test_csv_and_markdown_complete(self):
        with (ROOT/'inventory.csv').open(encoding='utf-8-sig',newline='') as f:
            csvrows=list(csv.DictReader(f))
        self.assertEqual([r['原始条目'] for r in csvrows],self.scope)
        self.assertTrue(all(None not in r for r in csvrows))
        md=(ROOT/'REPORT.md').read_text(encoding='utf-8')
        for n,name in enumerate(self.scope,1):self.assertIn(f'| {n} | {name} |',md)
        self.assertNotIn('唯一主依据',md)
    def test_link_check_coverage_and_boundaries(self):
        checks=json.loads((ROOT/'link-check.json').read_text(encoding='utf-8'))
        expected={u for r in self.rows for fields in [r['证据'],*r['设备维度'].values()] for v in fields.values() for u in v['链接']}
        self.assertEqual({r['url'] for r in checks},expected)
        self.assertTrue(all(r['method']=='HEAD' and r['content_verified'] is False for r in checks))
    def test_device_dimensions_and_csv_columns(self):
        fields=['代码','权重','数据集','性能基准','精度基准']
        with (ROOT/'inventory.csv').open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f);cols=reader.fieldnames
            self.assertEqual(len(list(reader)),119)
        for r in self.rows:
            self.assertEqual(set(r['设备维度']),{'GPU','NPU'})
            for device in ['GPU','NPU']:
                self.assertEqual(set(r['设备维度'][device]),set(fields))
                for k in fields:
                    self.assertIn(device+'_'+k,cols)
                    self.assertIn(device+'_'+k+'_证据链接',cols)
                    self.assertTrue(r['设备维度'][device][k]['说明'])
        self.assertIn('0.3148',self.rows[84]['设备维度']['GPU']['精度基准']['说明'])
        self.assertIn('0.3152',self.rows[84]['设备维度']['NPU']['精度基准']['说明'])
        self.assertNotIn('155.1',self.rows[21]['设备维度']['GPU']['性能基准']['说明'])
    def test_reader_view_keeps_numbers_and_removes_boilerplate(self):
        from reader_view import display
        self.assertIn('GPU RefCOCO AP0.3148',display(self.rows[84]['设备维度']['GPU']['精度基准']))
        self.assertIn('52.5',display(self.rows[84]['设备维度']['GPU']['精度基准']))
        self.assertIn('128',display(self.rows[21]['设备维度']['NPU']['性能基准']))
        with (ROOT/'inventory.csv').open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f);rows=list(reader)
        self.assertIn('注意事项',reader.fieldnames)
        self.assertNotIn('对照设备未明确的原表记录',reader.fieldnames)
        for r in rows:
            for key,value in r.items():
                if key.endswith('_证据链接'):continue
                self.assertNotIn('不以NPU结果代替',value)
                self.assertNotIn('来源报告；未独立复测',value)
                self.assertNotIn('GPU使用的版本、文件兼容及测试split待核实',value)
    def test_build_is_reproducible(self):
        names=['REPORT.md','inventory.json','inventory.csv']
        before={n:(ROOT/n).read_bytes() for n in names}
        subprocess.run([sys.executable,str(ROOT/'build_report.py')],check=True)
        self.assertEqual(before,{n:(ROOT/n).read_bytes() for n in names})
    def test_gpu_research_replaces_placeholder_conclusions(self):
        from reader_view import status
        for r in self.rows:
            self.assertNotEqual(r['GPU查询状态'],'尚未完成专项查询')
            self.assertNotIn('找到原始资料，GPU支持待核实',status(r,'GPU'))
        for n,token in [(1,'4090'),(14,'22.8'),(76,'0.99'),(81,'29.6'),(113,'217'),(115,'A100')]:
            self.assertIn(token,self.rows[n-1]['设备维度']['GPU']['性能基准']['说明'])
        self.assertIn('Coming Soon',self.rows[33]['设备维度']['GPU']['权重']['说明'])
        self.assertIn('不能分享',self.rows[72]['设备维度']['GPU']['权重']['说明'])
        self.assertIn('渲染',status(self.rows[77],'GPU'))
        self.assertIn('未唯一定位',self.rows[85]['设备维度']['GPU']['代码']['说明'])
        self.assertIn('Thinker-4B',self.rows[27]['设备维度']['GPU']['权重']['说明'])
    def test_followup_source_attribution_and_updates(self):
        updates=json.loads((ROOT/'evidence_updates.json').read_text(encoding='utf-8'))['updates']
        self.assertEqual(len({(u['number'],u['device']) for u in updates}),len(updates))
        for update in updates:
            self.assertIn(update['number'],range(1,120))
            self.assertIn(update['device'],['GPU','NPU'])
            self.assertTrue(set(update['fields']) <= {'代码','权重','数据集','性能基准','精度基准'})
            row=self.rows[update['number']-1]
            self.assertEqual(row['设备核查日期'][update['device']],update['date'])
            if update.get('conclusion'):
                self.assertEqual(row[update['device']+'结论'],update['conclusion'])
            for field,value in update['fields'].items():
                self.assertEqual(row['设备维度'][update['device']][field],value)
                self.assertTrue(value['链接'])
        self.assertIn('不含编译',self.rows[54]['设备维度']['GPU']['性能基准']['说明'])
        self.assertIn('DSIFN14400/1360/192',self.rows[66]['设备维度']['GPU']['数据集']['说明'])
        self.assertIn('非百分比',self.rows[72]['设备维度']['GPU']['精度基准']['说明'])
        self.assertIn('point-adjust',self.rows[78]['设备维度']['GPU']['精度基准']['说明'])
        self.assertIn('不是2.5结果',self.rows[110]['设备维度']['GPU']['精度基准']['说明'])
        self.assertIn('261',self.rows[18]['设备维度']['GPU']['性能基准']['说明'])
        self.assertIn('冲突',self.rows[18]['设备维度']['GPU']['性能基准']['说明'])
        self.assertIn('不一致',self.rows[13]['设备维度']['GPU']['精度基准']['说明'])
        self.assertIn('3B没有',self.rows[20]['设备维度']['GPU']['权重']['说明'])
        self.assertIn('输出视频帧率',self.rows[3]['设备维度']['GPU']['性能基准']['说明'])
    def test_runtime_scope_and_scientific_benchmarks(self):
        def gpu(number, field):
            return self.rows[number-1]['设备维度']['GPU'][field]['说明']
        for number, field, tokens in [
            (5,'性能基准',['141.7','64','4.2']),
            (13,'精度基准',['46/50','38/50']),
            (16,'性能基准',['Stage1','35.93']),
            (16,'权重',['Stage2','待发']),
            (18,'性能基准',['训练','32.84']),
            (38,'性能基准',['5','200']),
            (59,'性能基准',['单CPU','0.9']),
            (59,'精度基准',['63.3','50.5','77.5']),
            (60,'精度基准',['0.7986','0.7893','flanks']),
            (65,'精度基准',['PNAS','0.65','预印本']),
            (80,'性能基准',['V100','1.7 FPS','不是整模型']),
            (85,'性能基准',['商业重训','batch32','84.06','不是IDEA']),
            (114,'性能基准',['A100','0.15','CPU','预计算']),
            (117,'性能基准',['8.47','4090','2.70','3090']),
            (117,'数据集',['train9000','val1000','test1000']),
        ]:
            for token in tokens:
                with self.subTest(number=number,field=field,token=token):
                    self.assertIn(token,gpu(number,field))
    def test_published_variants_and_runtime_protocols(self):
        def gpu(number, field):
            return self.rows[number-1]['设备维度']['GPU'][field]['说明']
        for number, field, tokens in [
            (6,'精度基准',['60.7','65.7','5.0','5个推理seed']),
            (28,'精度基准',['4B','62.66','7B','63.5']),
            (30,'精度基准',['八项','4.66','3候选']),
            (30,'性能基准',['5 fps','不是GPU']),
            (35,'代码',['未发布','只有项目页']),
            (41,'精度基准',['57.03','72.16','Qwen3-Embedding-0.6B']),
            (41,'性能基准',['23文档','512','transactions/s','切片']),
            (42,'性能基准',['第三方','H100','并发8','29051','60 s']),
            (43,'精度基准',['69.02','77.45','top100']),
            (63,'性能基准',['A4000','15步','11 s','200步']),
            (64,'性能基准',['RTX2080','15 CPU核','1.5 h']),
            (74,'精度基准',['ResUNet++','ResUNet','0.8133','0.5144']),
            (88,'性能基准',['Hypersim-vits','4.3084','100次','排除前后处理']),
            (90,'性能基准',['训练成本','8.9','A100']),
            (108,'精度基准',['heuristic','不能误填']),
            (110,'性能基准',['纯文本Qwen2.5','不能套到VL']),
            (118,'性能基准',['2-AC','16秒','800候选','4090']),
        ]:
            for token in tokens:
                with self.subTest(number=number,field=field,token=token):
                    self.assertIn(token,gpu(number,field))
        with (ROOT/'inventory.csv').open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(rows[34]['GPU结论'],'论文已发布；训练/推理代码和权重尚未发布')
    def test_gpu_benchmark_device_and_measurement_boundaries(self):
        def gpu(number, field):
            return self.rows[number-1]['设备维度']['GPU'][field]['说明']
        for number, field, tokens in [
            (27,'性能基准',['训练','8×A100','1.131','56.6','不是动作推理']),
            (46,'权重',['2605','3.3.0']),
            (46,'性能基准',['A100','2.12','未给并发数']),
            (46,'精度基准',['95.39','95.26','95.72','口径不同']),
            (49,'性能基准',['blastn','4.32','1.56','megablast','7.15']),
            (58,'性能基准',['15B','8×H200','7119','不是650M']),
            (69,'性能基准',['V100','FP32','batch1','46.73','排除数据加载']),
            (70,'性能基准',['V100','FP32','batch1','46.73','排除数据加载']),
            (93,'性能基准',['第三方','A100-PCIE-40GB','32.6','1.58','新生成token']),
            (94,'性能基准',['MMIU','A100-80GB','0.97 h','testmini','不是单请求']),
            (97,'性能基准',['OpenCL','500轮','112 s','63','旧版训练']),
            (97,'精度基准',['AUC0.845209','业务时间切分']),
            (106,'性能基准',['RTX3090','没有可直接引用']),
            (109,'性能基准',['H20','81.73','prompt+generation','不是纯解码']),
            (109,'精度基准',['thinking','87.5','non-thinking','79.5']),
        ]:
            for token in tokens:
                with self.subTest(number=number,field=field,token=token):
                    self.assertIn(token,gpu(number,field))
        # 两个原始名称不合并，也不把同一实现计作独立适配。
        for number in [69,70]:
            self.assertIn('不把重复名称视作两项独立适配',self.rows[number-1]['分析与缺口'])
    def test_paper_results_do_not_imply_gpu_delivery(self):
        def gpu(number, field):
            return self.rows[number-1]['设备维度']['GPU'][field]['说明']
        for number, field, tokens in [
            (3,'精度基准',['条件Place50%','成功抓取']),
            (24,'性能基准',['多TPU','1–3Hz','GPU专项速度']),
            (29,'权重',['视觉语言底座','Thinker-VLA专项']),
            (39,'代码',['OpenPI JAX','RLinf','流式VLA','撤回']),
            (39,'精度基准',['67.4%','80.6%','48次','64次']),
            (53,'权重',['不需要神经网络权重','kmer_distrib']),
            (53,'精度基准',['1.75%','1.89%','没有GPU专项']),
            (75,'代码',['晶圆','不是医学','403']),
            (82,'权重',['无神经网络checkpoint']),
            (83,'代码',['DeepSeek API','不是已公开的本地GPU模型']),
            (112,'性能基准',['RTX4090','14.48 s','15.39 s','不能外推']),
        ]:
            for token in tokens:
                with self.subTest(number=number,field=field,token=token):
                    self.assertIn(token,gpu(number,field))
        self.assertIn('多TPU',self.rows[23]['GPU结论'])
        self.assertIn('GPU实验',self.rows[111]['GPU结论'])
    def test_supplemental_npu_sources_and_gpu_protocols(self):
        def evidence(number,device,field):
            return self.rows[number-1]['设备维度'][device][field]
        for number, device, field, tokens in [
            (76,'NPU','代码',['modelzoo-GPL','ModelZoo-PyTorch','未收录']),
            (76,'NPU','性能基准',['310P3','batch8','753.23','443.60','纯OM']),
            (76,'NPU','精度基准',['mAP@0.5','不是mAP@0.5:0.95']),
            (81,'NPU','代码',['InferSession','CPU路径','不是全模块']),
            (81,'NPU','性能基准',['VideoWriter','不是处理速度']),
            (42,'NPU','代码',['实验性','0.6B','8B']),
            (43,'NPU','精度基准',['relevance_score','不是准确率']),
            (44,'NPU','数据集',['LeCaRDv2','batch2']),
            (45,'NPU','数据集',['覆盖为MultiLongDocReranking']),
            (96,'GPU','精度基准',['test192','CER/WER']),
            (101,'GPU','权重',['CC-BY-NC','Apache','2025-09']),
            (101,'GPU','精度基准',['两视图','0.08','56.0','V1']),
            (102,'GPU','性能基准',['V100','21','23','43/837','8核CPU']),
            (102,'GPU','精度基准',['10^-3','2.34','40.88']),
            (110,'GPU','性能基准',['第三方','32.0','0.87s','新生成token']),
        ]:
            for token in tokens:
                with self.subTest(number=number,device=device,field=field,token=token):
                    self.assertIn(token,evidence(number,device,field)['说明'])
        self.assertTrue(all('/modelzoo-GPL/' in u for u in evidence(76,'NPU','代码')['链接']))
        self.assertIn('CPU',self.rows[80]['NPU结论'])
    def test_new_benchmark_protocols_and_deferred_items(self):
        def text(n, device, field):
            return self.rows[n-1]['设备维度'][device][field]['说明']
        for n, device, field, tokens in [
            (20,'GPU','精度基准',['76.5','分别LoRA','不是OFT']),
            (26,'GPU','精度基准',['50.33%','2026-01-11','未给试验次数']),
            (55,'GPU','数据集',['428','308','2019-09-30']),
            (62,'GPU','精度基准',['41/41','至少有一个','不是单设计100%']),
            (105,'GPU','精度基准',['2M步','6种子','不标GPU实测']),
            (116,'GPU','精度基准',['ETTh1','0.384','0.402']),
            (116,'GPU','性能基准',['TITAN RTX 24GB','0.024','秒/迭代','不是单样本']),
            (47,'GPU','精度基准',['96.33','综合分','不是NPU实测']),
            (99,'NPU','性能基准',['未注明设备','不能认作NPU']),
            (108,'NPU','代码',['Ascend','只写GPU/CPU','不一致']),
        ]:
            for token in tokens:
                with self.subTest(number=n,device=device,field=field,token=token):
                    self.assertIn(token,text(n,device,field))
        for n in [86,100]:
            for device in ['GPU','NPU']:
                self.assertEqual(self.rows[n-1][device+'结论'],'按用户要求暂缓核查')
    def test_concrete_npu_sources_and_framework_boundaries(self):
        def text(n, device, field):
            return self.rows[n-1]['设备维度'][device][field]['说明']
        for n, device, field, tokens in [
            (55,'NPU','代码',['800T A2','Evo','2K','<1000']),
            (55,'NPU','精度基准',['置信分数不等于','RMSD']),
            (104,'NPU','代码',['XPINNs','Poisson2D','不能覆盖']),
            (104,'GPU','精度基准',['MindSpore1.2','0.01323','0.2545%']),
            (104,'NPU','性能基准',['未标卡型','不能引用为NPU']),
            (116,'NPU','性能基准',['Tesla T4','不能填作NPU']),
            (116,'NPU','精度基准',['0.416','0.429','不是同一交付']),
            (116,'GPU','性能基准',['Tesla T4','15.19ms','不可直接相除']),
            (116,'GPU','权重',['Paddle格式','不可直接加载']),
            (116,'NPU','权重',['404','不能认定训练权重']),
        ]:
            for token in tokens:
                with self.subTest(number=n,device=device,field=field,token=token):
                    self.assertIn(token,text(n,device,field))
        review=json.loads((ROOT/'evidence_updates.json').read_text(encoding='utf-8'))['master_review']
        self.assertFalse(review['truncated'])
        self.assertEqual(review['commits'],1)
        self.assertTrue(all(p=='ACL_PyTorch/ModeList.md' or p.startswith('ACL_PyTorch/built-in/cv/DEIMv2/') for p in review['changed_paths']))
    def test_inference_test_is_not_structural_accuracy_and_mixed_device_limits(self):
        def text(n, field):
            return self.rows[n-1]['设备维度']['NPU'][field]['说明']
        for n, field, tokens in [
            (56,'代码',['910B','训练方法为空']),
            (56,'数据集',['6T36','固定长度256','不是完整']),
            (56,'精度基准',['confidence>92','置信度','不能视为92%']),
            (56,'性能基准',['实际代码没有耗时测量']),
            (65,'代码',['返回CPU','2.10','2.5.1']),
            (65,'性能基准',['0.4x','占位']),
            (65,'精度基准',['3880','仅展示5条','不是实验稳定性']),
            (107,'代码',['仅README','没有配套源码补丁','FAISS留在CPU']),
            (107,'性能基准',['2.1510577','撤回此前E2E','不能确定端到端']),
            (113,'精度基准',['53.1','发布指标参考','不借用v2']),
        ]:
            for token in tokens:
                with self.subTest(number=n,field=field,token=token):
                    self.assertIn(token,text(n,field))
if __name__=='__main__':unittest.main()
