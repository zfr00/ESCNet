import requests
import re
startUrl = 'https://piyao.jfdaily.com/py_ufkSl2NHwZTJp8MpHfQVoa8JpFlcVCrjqfxJR7Mr8tzCl1ItlTd/+5arR5y13eONtG1ACHYg72q91g'

url_list = []  ##存储所有爬取下来的网页，最后写进txt

def extract_page_urls():
    content = requests.get(startUrl).text
    # import re
    regexp = re.compile(r'case\s+(\d+):\n\s+window.location.href\s+=\s+"([^"]+)";', re.MULTILINE)
    return regexp.findall(content)

urls = extract_page_urls()   ## list of (id, url)
# breakpoint()
for i in range(len(urls)):
    _, url = urls[i]

    content = requests.get(url).text

# import re
    exp = re.compile(r'<div class="news-l">\s+<a href="([^"]+)"', re.MULTILINE)
    items = (exp.findall(content))
    # breakpoint(
    for j in range(len(items)):
        useurl = items[j]
        url_list.append(useurl)
#将url_list写入txt
with open('url_list.txt', 'w') as f:
    for item in url_list:
        f.write("%s\n" % item)
# print(items, len(items))