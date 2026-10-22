"""Regression checks on scope preservation and source attribution."""
import csv
import json
import pathlib
import unittest
from urllib.parse import unquote

ROOT = pathlib.Path(__file__).resolve().parent

class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=json.loads((ROOT/'inventory.json').read_text(encoding='utf-8'))
        cls.scope=json.loads((ROOT/'scope.json').read_text(encoding='utf-8'))
        cls.tree=set((ROOT/'repository-files.txt').read_text(encoding='utf-8').splitlines())

    def test_original_scope_preserved_in_order(self):
        self.assertEqual(len(self.rows),119)
        self.assertEqual([r['原始条目'] for r in self.rows],self.scope['models'])

    def test_every_item_has_all_requested_dimensions(self):
        for row in self.rows:
            for field in ['代码','权重','数据集','性能基准','精度基准','分析与缺口','具体证据链接']:
                self.assertTrue(row[field],(row['原始条目'],field))

    def test_all_pinned_source_files_exist(self):
        marker='/blob/dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202/'
        for row in self.rows:
            for url in row['具体证据链接'].split(' ; '):
                if marker in url:self.assertIn(unquote(url.split(marker,1)[1]),self.tree)

    def test_csv_matches_json_and_has_no_truncated_rows(self):
        with (ROOT/'inventory.csv').open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        self.assertEqual([r['原始条目'] for r in rows],self.scope['models'])
        self.assertTrue(all(None not in r for r in rows))

    def test_neighbor_versions_not_certified_as_exact(self):
        for name in ['MinerU3.3','Depth Anything V2','RT-DETR']:
            row=next(r for r in self.rows if r['原始条目']==name)
            self.assertEqual(row['仓库证据分类'],'仅相邻版本')
        self.assertTrue(all('未做模型运行' in r['本次实测状态'] for r in self.rows))

    def test_index_and_component_corrections_retained(self):
        for name in ['Qwen3-Embedding','Qwen3-Reranker','YOLOv8']:
            self.assertIn('索引外链',next(r for r in self.rows if r['原始条目']==name)['仓库证据分类'])
        self.assertEqual(next(r for r in self.rows if r['原始条目']=='DINOv2')['仓库证据分类'],'组件级证据')

if __name__=='__main__':unittest.main()
