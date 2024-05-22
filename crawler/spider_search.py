from newspaper import Article
import pandas as pd
from bs4 import BeautifulSoup
import requests

pathset1 = r'F:\GEB-Plus-main\url_list22.txt'
all_urls = []
all_cate = []

with open(pathset1, 'r', encoding='gbk') as f2:
    text = f2.readlines()
    #print(reader_img)
    for i,line in enumerate(text):
        # if i==1:
        # line = line.strip('\n')###没有类别标签的
        # all_urls.append(line)
        # print(line)
        line = line.strip('\n').split('\t') ###有类别标签的    
        all_urls.append(line[0])
        # all_cate.append(line[1])

print(len(all_urls))
# print(len(all_cate))



visited = set()##已爬过的

news_title = []
news_text = []
news_img = []
# news_cate = [] ##几个类别
news_label = []##fake为1，real为0


###############有类别
for i,url in enumerate(all_urls):
    # cate = all_cate[i]
    current_url = url
    if current_url not in visited:
        visited.add(current_url)##集合添加元素
        print(f'正在爬取: {current_url}')

        try:
            news = Article(current_url, language='zh')
            news.download()
            news.parse()
            ##爬取图片
            page = requests.get(current_url).text
            bs = BeautifulSoup(page, 'html.parser')
            tag = bs.find_all('img')[0]
            src = tag.attrs['src']
            url_base = "/".join(url.split('/')[:-1])
            news_img.append(url_base+ '//' + src)

            news_title.append(news.title)
            news_text.append(news.text.replace('\n',''))
            # news_img.append(news.images)
            # news_cate.append(cate)

        except Exception as e:
            print(f'无法获取 {current_url} 的内容: {e}')

#############################有类别
# data = pd.DataFrame({'title':news_title,'text':news_text,'image':news_img,'cate':news_cate})
data = pd.DataFrame({'title':news_title,'text':news_text,'image':news_img})
data.to_csv('search_data_clean_supple.csv', index=False)