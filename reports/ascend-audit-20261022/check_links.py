"""Bounded, read-only HTTP HEAD checks; never download weights or run sources."""
import concurrent.futures
import datetime
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def check(url):
    r={'url':url,'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'method':'HEAD','content_verified':False,
       'version':'pinned' if re.search(r'/(?:blob|tree)/[0-9a-f]{40}/',url) else 'dynamic_or_unversioned'}
    try:
        p=urllib.parse.urlsplit(url)
        target=urllib.parse.urlunsplit((p.scheme,p.netloc,urllib.parse.quote(p.path,safe='/%:@'),urllib.parse.quote(p.query,safe='=&%:/?'),''))
        req=urllib.request.Request(target,method='HEAD',headers={'User-Agent':'ModelZoo-evidence-audit/1.0'})
        with urllib.request.urlopen(req,timeout=12) as response:
            r.update(status=response.status,final_url=response.url,result='http_reachable')
    except urllib.error.HTTPError as err:
        r.update(status=err.code,result='not_confirmed',error=str(err))
    except Exception as err:
        r.update(status=None,result='not_confirmed',error=str(err))
    return r

def main():
    rows=json.loads((ROOT/'inventory.json').read_text(encoding='utf-8'))
    urls=sorted({u for r in rows for value in r['证据'].values() for u in value['链接']})
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for i,result in enumerate(pool.map(check,urls),1):
            results.append(result)
            if i%25==0:print(f'checked {i}/{len(urls)}',flush=True)
    (ROOT/'link-check.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    ok=sum(x['result']=='http_reachable' for x in results)
    print(f'HTTP reachable {ok}/{len(results)}; other responses remain unconfirmed, not automatically broken.')

if __name__=='__main__':main()
