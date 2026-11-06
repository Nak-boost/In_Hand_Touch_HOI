import requests, concurrent.futures, json
from pathlib import Path
from bs4 import BeautifulSoup
root=Path(__file__).parent
sources={
'anyrotate':'https://arxiv.org/html/2405.07391v3',
'rotateit':'https://arxiv.org/html/2309.09979v2',
'dapg':'https://arxiv.org/abs/1709.10087',
'reverse_curriculum':'https://arxiv.org/abs/1707.05300',
'backplay':'https://arxiv.org/abs/1807.06919',
'relay':'https://arxiv.org/abs/1910.11956',
'goexplore':'https://arxiv.org/abs/2004.12919',
'deepmimic':'https://arxiv.org/abs/1804.02717',
'dactyl':'https://arxiv.org/abs/1808.00177',
'her':'https://arxiv.org/abs/1707.01495',
'ddpg_search':'https://html.duckduckgo.com/html/?q=reinforcement%20learning%20long%20horizon%20dexterous%20manipulation%202025',
}
def fetch(item):
 key,url=item
 try:
  r=requests.get(url,timeout=35)
  soup=BeautifulSoup(r.text,'html.parser')
  txt=soup.get_text(' ',strip=True)
  (root/(key+'.html')).write_text(r.text)
  (root/(key+'.txt')).write_text(txt)
  abstract=soup.select_one('.abstract')
  return {'key':key,'url':url,'status':r.status_code,'title':soup.title.get_text() if soup.title else '', 'abstract':abstract.get_text(' ',strip=True) if abstract else txt[:500]}
 except Exception as e:return {'key':key,'error':str(e)}
results=list(concurrent.futures.ThreadPoolExecutor(8).map(fetch,sources.items()))
(root/'sources.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
for x in results:print(json.dumps(x,ensure_ascii=False))
