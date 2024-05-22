import requests

###这个是爬取图片的接口
url = 'https://www.piyao.org.cn/20230530/0c865263324a4000b39cff863ee635a9/c.html'
#https://www.piyao.org.cn/index/images/souas.png
page = requests.get(url).text

from bs4 import BeautifulSoup

bs = BeautifulSoup(page, 'html.parser')
tag = bs.find_all('img')[0]

# breakpoint()

src = tag.attrs['src']
print(type(src))
print(src)

url_base = "/".join(url.split('/')[:-1])
print(url_base)
print(url_base+ '//' + src)