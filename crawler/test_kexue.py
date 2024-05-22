import requests
import re
### 这个是搜索引擎的接口
template = 'https://piyao.kepuchina.cn/rumor/rumorlist?type=0&title=&keyword=0&page={pagenum}'

# pagenum = 1
url_list = []

for i in range(377):
    startUrl = template.format(pagenum=i+1)


# def extract_page_urls():
#     content = requests.get(startUrl).text
#     # import re
#     regexp = re.compile(r'case\s+(\d+):\n\s+window.location.href\s+=\s+"([^"]+)";', re.MULTILINE)
#     return regexp.findall(content)

# urls = extract_page_urls()   ## list of (id, url)
# breakpoint()
# for i in range(len(urls)):
#     _, url = urls[i]

    content = requests.get(startUrl).text
    # print(content)
# import re
    exp = re.compile(r'<div class="rumor-list_item pull-left">\s+<a\s+href="([^"]+)"', re.MULTILINE)
    items = (exp.findall(content))
    # breakpoint()
    for j in range(len(items)):
        useurl = items[j]
        url_list.append(useurl)
    print(i+1)
#将url_list写入txt
with open('url_list10.txt', 'w') as f:
    for item in url_list:
        f.write("%s\n" % item)
print(items, len(items))