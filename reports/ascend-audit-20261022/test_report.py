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
        self.assertEqual(len({u['number'] for u in updates}),len(updates))
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
if __name__=='__main__':unittest.main()
