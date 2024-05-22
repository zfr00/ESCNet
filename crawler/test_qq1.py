import requests

template = "https://vp.fact.qq.com/api/article/list?page={pageNum}&locale=zh-CN&token=U2FsdGVkX19HokYvjfKTrbdI3PqHoVrYXAthCNLg9XtsYgGLMWRkKLYi1eL64QS0"
token1 = "U2FsdGVkX19HokYvjfKTrbdI3PqHoVrYXAthCNLg9XtsYgGLMWRkKLYi1eL64QS0"
token2 = 'U2FsdGVkX189JRyHbMjS4XKUmIbMx2wBLJOe756w7GSWEZaXGEVlOjYKgOX7f0Zl'
token3 = 'U2FsdGVkX19R3VfYLM24BVaQJUR1WcQrpQ52RMZWV1Lsqh9jOWnEQWpmQWt2aRdN'
all_title = []
all_text = []
all_img = []
all_cate = []

def process(url):
    # items = []
    resp = requests.get(url).json()
    # print(resp)
    # breakpoint()
    for x in resp["data"]["list"]:

        all_title.append(x["title"])
        all_text.append(x["content"])
        all_img.append(x["coversqual"])
        all_cate.append(x["markstyle"])
        # items.append(
        #     {
        #         "title": x["title"],
        #         "url": f'https://vp.fact.qq.com/article?id={x["id"]}',
        #         "label": x["markstyle"],  # 'fake' | 'doubt' | 'true'
        #     }
        # )
    return items, resp["data"]["hasMore"]


# all_items = []

pageNum = 1


###
a = requests.get(template.format(pageNum=pageNum)).json()
# breakpoint()

while True:
    # print(pageNum)
    items, has_more = process(template.format(pageNum=pageNum))
    print(pageNum)
    # breakpoint()
    # all_items.extend(items)
    if not has_more:
        break
    pageNum += 1
