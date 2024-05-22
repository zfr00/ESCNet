import requests
import json

url_template = 'https://dawa.news.cn/nodeart/page?nid={nid}&pgnum={pgnum}&cnt=16&attr=&tp=1&orderby=1&callback=_'
# <li data-nodeid="11158883">社会</li>
# <li data-nodeid="11158884">文化</li>
# <li data-nodeid="11158881">健康</li>
# <li data-nodeid="11158879">食品</li>
# <li data-nodeid="11158880">科学</li>
# urls = []

### 案例分类 ID：11158881 - 疫情
# 类别 ID：11158882 - 政治
nidlist = ['11158882', '11158883','11158884','11158881','11158879','11158880']
# nid = '11158882'
flag = 3
for nid in nidlist:
    urls = []
    for pgnum in range(100):
        url = url_template.format(nid=nid, pgnum=pgnum)
        # print(url)
        response = requests.get(url)  ###不能直接提取json，需要去掉前后的括号 
        # breakpoint()
        data = json.loads(response.text[2:-2])
        if 'data' not in data: break
        data = data['data']['list']
        for item in data:
            urls.append(item['LinkUrl'])
    print(len(urls))
    with open('url_list'+str(flag)+'.txt', 'w') as f:
        for j1 in urls:
            f.write("%s\n" % j1)
    flag += 1
    # print(response.text)
    # break
# print(urls)
# print(len(urls))