"""Read-only extraction from the pinned target repository; never runs model code."""
import json
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent
REPO = ROOT.parents[1] / '.codex-reference/model-audit-20261022'

def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args]).decode('utf-8')

def main():
    scope = json.loads((ROOT / 'scope.json').read_text(encoding='utf-8'))
    sha = git('rev-parse', 'HEAD').strip()
    if sha != 'dd5924481f8f43e2c7d51f39eca8bf1a7e3d4202':
        raise ValueError('Unexpected repository revision; review scope before refreshing')
    tree = git('ls-tree', '-r', '--name-only', sha).splitlines()
    (ROOT / 'repository-files.txt').write_text('\n'.join(tree)+'\n', encoding='utf-8', newline='\n')
    refs = [p for p in scope['references'] if p in tree]
    dirs = sorted({p.rsplit('/', 1)[0] for p in refs if p.lower().endswith(('.md','.sh'))})
    # OpenFold is a related implementation, not proof for AlphaFold3/Multimer.
    dirs += ['PyTorch/built-in/others/OpenFold_for_PyTorch']
    candidates = [p for p in tree if any(p.startswith(d+'/') and (p.rsplit('/',1)[-1].lower() in ('readme.md','readme_en.md') or (p.count('/') <= d.count('/')+2 and p.endswith(('.py','.sh','.patch','.diff','.txt','.yaml','.json')))) for d in dirs)]
    all_readmes = [p for p in tree if p.rsplit('/',1)[-1].lower() in ('readme.md','readme_en.md','modellist.md','modelzoo.md')]
    candidates = list(dict.fromkeys(candidates + all_readmes))
    object_map = {row.split('\t',1)[1]:row.split()[2] for row in git('ls-tree','-r',sha).splitlines()}
    object_ids = sorted({object_map[p] for p in candidates})
    subprocess.run(['git','-C',str(REPO),'-c','fetch.negotiationAlgorithm=noop','fetch','--no-tags','--no-write-fetch-head','origin','--stdin'],input=('\n'.join(object_ids)+'\n').encode(),check=True,capture_output=True)
    raw = subprocess.check_output(['git','-C',str(REPO),'cat-file','--batch'], input=''.join(sha+':'+p+'\n' for p in candidates).encode())
    content = {}
    offset = 0
    for p in candidates:
        end = raw.index(b'\n', offset)
        size = int(raw[offset:end].split()[-1])
        content[p] = raw[end+1:end+1+size].decode('utf-8',errors='replace')
        offset = end+size+2
    (ROOT.parents[1]/'.codex-reference/model-audit-readmes.json').write_text(json.dumps([{'path':p,'text':content[p]} for p in all_readmes],ensure_ascii=False),encoding='utf-8', newline='\n')
    evidence = []
    for directory in dirs:
        files = [p for p in tree if p.startswith(directory+'/')]
        selected = [p for p in files if (p.rsplit('/',1)[-1].lower() in ('readme.md','readme_en.md') or (p.count('/') <= directory.count('/')+2 and p.endswith(('.py','.sh','.patch','.diff','.txt','.yaml','.json'))))]
        item = {'directory':directory,'files':files,'sources':[]}
        for p in selected:
            text = content[p]
            if len(text)>200000:
                continue
            item['sources'].append({'path':p,'text':text})
        evidence.append(item)
    data = {'sha':sha,'date':'2026-10-22','tree_count':len(tree),'evidence':evidence}
    (ROOT / 'evidence.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8', newline='\n')
    excerpts=[]
    for item in evidence:
        excerpts.append('\n## '+item['directory'])
        excerpts.append('FILES: '+', '.join(p.rsplit('/',1)[-1] for p in item['files'][:45]))
        for source in item['sources']:
            if source['path'].lower().endswith('.md'):
                excerpts.append('\n### '+source['path'])
                excerpts.extend(f'{i}: {line}' for i,line in enumerate(source['text'].splitlines(),1) if re.search(r'https?://|精度|性能|accuracy|fps|latency|throughput|mAP|mIoU|success|成功率|Atlas|910|310|CANN|数据集|dataset|checkpoint|权重|误差|cosine|相似度|RTF|耗时|\|',line,re.I))
    (ROOT / 'readme-excerpts.txt').write_text('\n'.join(excerpts)+'\n',encoding='utf-8', newline='\n')
    print(json.dumps({'sha':sha,'files':len(tree),'directories':len(evidence),'source_files':sum(len(x['sources']) for x in evidence)}))

if __name__ == '__main__':
    main()
