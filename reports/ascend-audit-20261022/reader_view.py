"""Short display text. Preserve full audit notes in inventory.json."""
from urllib.parse import unquote, urlsplit

def resource_name(url):
    p=urlsplit(url)
    parts=unquote(p.path).strip('/').split('/')
    if p.netloc=='huggingface.co':
        if parts and parts[0]=='datasets':parts=parts[1:]
        return '/'.join(parts[:2])
    if parts and parts[-1].endswith(('.pth','.pt','.tar.gz','.pdparams','.tar')):
        return parts[-1]
    return ''

def display(v):
    text=v['说明'];role=v['证据类型']
    if text.startswith(('未取得明确GPU','未取得充分专项','未找到')):
        return '未查到'
    if role=='上游入口；非已验证GPU运行':return '原始仓库；GPU支持未核实'
    if role in ['原始入口；非GPU适配证明','原始资源，不证明NPU支持']:
        return '仅找到原始资料'
    if role in ['入口，非已验证文件','资源线索；不代表GPU验证','模型卡/数据仓入口']:
        name=next((resource_name(u) for u in v['链接'] if resource_name(u)),'')
        if name:return name+'（使用情况未核实）'
        if role=='入口，非已验证文件':return '仅找到资源入口，具体文件未确认'
        if text.startswith('来源资源：'):
            return text[len('来源资源：'):].split('；GPU使用的版本')[0]+'（GPU使用未核实）'
    replacements={
        '固定提交专项交付目录；非逐文件验收':'专项适配目录',
        'NPU原版/优化版继续训练对照，不是CUDA迁移或完整从头收敛对照；原始JSON已读取，未独立复测。':'NPU优化前后比较；继续训练120步，非GPU/NPU对比。',
        '官方部署评测教程，非完整源码核验':'官方部署教程',
        '官方平台支持指导，非源码逐文件核验':'官方部署指导',
        '固定上游及NPU patch指导':'NPU适配补丁',
        '固定上游及优化patch指导':'NPU优化补丁',
        '历史官方Ascend支持记录，非源码运行':'历史版本支持清单',
        '来源资源：':'',
    }
    for old,new in replacements.items():text=text.replace(old,new)
    return text

def status(r,device):
    if device=='GPU':
        if 'GPU查询状态' in r:
            return r['GPU结论'] if r['GPU查询状态']!='尚未完成专项查询' else '待完成GPU专项查询'
        return '找到原始资料，GPU支持待核实' if r['设备维度']['GPU']['代码']['链接'] else '未查到GPU支持证据'
    text=r['NPU结论']
    return {'有公开实现/指导，完整验收未认证':'有适配代码或指导',
            '有指定实现/指导，非完整验收认证':'有适配代码或指导',
            '未取得充分专项NPU证据':'未查到NPU适配证据',
            '对象未唯一确认':'模型名称待确认',
            '系列名称；按具体成员核查':'系列名称，需看具体版本',
            '仅系统支持名单线索，缺完整专项交付':'支持名单提及，未找到完整交付'}.get(text,text)

def note(r):
    text=r['分析与缺口']
    if text.startswith(('来源报告未复测','未找到不等于不支持')):return ''
    return text.replace('来源报告，未独立复测；','').replace('；权重未下载校验','')
