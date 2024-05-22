from newspaper import Article
import pandas as pd
import re
import requests
pathset1 = r'C:\Users\Administrator\Desktop\url_list10.txt'
all_urls = []
all_cate = []

with open(pathset1, 'r', encoding='gbk') as f2:
    text = f2.readlines()
    #print(reader_img)
    for i,line in enumerate(text):
        # if i==1:
        line = line.strip('\n')###没有类别标签的
        all_urls.append(line)
        # print(line)
        # line = line.strip('\n').split('\t') ###有类别标签的    
        # all_urls.append(line[0])
        # all_cate.append(line[1])

print(len(all_urls))
print(len(all_cate))



visited = set()##已爬过的

news_title = []
news_text = []
news_img = []
news_cate = [] ##几个类别
news_label = []##fake为1，real为0


###############无类别
for i,url in enumerate(all_urls):
    current_url = url
    if current_url not in visited:
        visited.add(current_url)##集合添加元素
        print(f'正在爬取: {current_url}')

        try:
            news = Article(current_url, language='zh')
            news.download()
            news.parse()
            content = requests.get(current_url).text
            pattern = r'<div class="rumor-title">\s*([^<]*)\s*</div>'
            matches = re.findall(pattern, content)[0]
            print(matches)
            news_title.append(matches)
            news_text.append(news.text.replace('\n',''))
            if news.top_image == '':
                news_img.append('NULL')
            else:
                news_img.append(news.top_image)
            # news_cate.append(cate)
        except Exception as e:
            print(f'无法获取 {current_url} 的内容: {e}')

print('爬取完成！')

data = pd.DataFrame({'title':news_title,'text':news_text,'image':news_img})
data.to_csv('kexue_data_clean.csv', index=False)