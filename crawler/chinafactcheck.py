import requests
import re
import requests
from newspaper import Article
import pandas as pd
template = 'https://chinafactcheck.com?paged={page}'

page = 1

parser = re.compile(r'''<a href="([^"]+)"[^>]+>\s+<h2[^>]+>([^<]+)</h2>''', re.M)
pattern = r'<a href=".*?"\s*target="_blank">\s*<img src="(.*?)"'
pattern2 = r"<p class='post-digest'>\s*([^<]*)"
items = []

all_title = []
all_text = []
all_img = []
all_explain = []
while True:
    resp = requests.get(template.format(page=page)).text
    parsed = parser.findall(resp)
    img_matches = re.findall(pattern, resp)
    exp_matches = re.findall(pattern2, resp)
    if not parsed: break
    for x in parsed:
        all_title.append(x[1])
        all_img.append(img_matches[parsed.index(x)])
        all_explain.append(exp_matches[parsed.index(x)])
        current_url = x[0]
        try:
            news1 = Article(current_url, language='zh')
            news1.download()
            news1.parse()
            all_text.append(news1.text.replace('\n',''))
        except Exception as e:
            print(f'无法获取 {current_url} 的内容: {e}')
            all_text.append('NULL')

    # items.extend(dict(url=x[0], title=x[1]) for x in parsed)
    print(f'page {page} done')
    page += 1
    # print(items)

data = pd.DataFrame({'title':all_title,'text':all_text,'image':all_img,'explain':all_explain})
data.to_csv('china_factcheck.csv', index=False)
