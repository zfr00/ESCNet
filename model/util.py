import csv
import os
import numpy as np
import torch
import torch.nn as nn
import re
from nltk.corpus import stopwords
from sklearn.model_selection import StratifiedShuffleSplit,train_test_split
from transformers import BertModel, BertTokenizer, BertConfig
import pandas as pd

class StringProcess(object):
    def __init__(self):
        self.other_char = re.compile(r"[^A-Za-z0-9(),!?\'\`]", flags=0)  ###其他字符
        self.num = re.compile(r"[+-]?\d+\.?\d*", flags=0)    ##数字
        # self.url = re.compile(r"[a-z]*[:.]+\S+|\n|\s+", flags=0)
        self.url = re.compile(
                r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", flags=0)
        self.stop_words = None
        self.nlp = None


    def clean_str_BERT(self,string):
        r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'  # 用户也可以在此进行自定义过滤字符
        r2 = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"
        r3 = "[.!//_,$&%^*()<>+\"'?@#-|:~{}]+|[——！\\\\，。=？、：“”‘’《》【】￥……（）]+"
        r4 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
        string = string.split('http')[0]
        cleanr = re.compile('<.*?>')
        string = re.sub(cleanr, ' ', string)   ###乱七八糟的字符给它消掉
        # string = re.sub(r1, ' ', string)
        # string = re.sub(r2, ' ', string)
        # string = re.sub(r3, ' ', string)
        string = re.sub(r4, ' ', string)
        return string


def pre_training_www(pathset, config, mode):
    text_id = []  # 592595287815757825\t\t 这种格式的
    tweet = []
    image_id = []
    label = []  # fake\n 这种格式的
    mids_ctxt = []
    mids_cimg = []
    mids_dtxt = []
    mids_dimg = []
    document = []
    mymode = mode
    # with open('/gdata1/data/data/twitter/twitter_all2.txt', 'r', encoding='utf-8') as f2:
    

    # with open(pathset.path_txt_data, 'r', encoding='unicode_escape')as f:
    if mymode == 'train':
        data = pd.read_csv(pathset.path_txt_data_train)
    elif mymode == 'test':
        data = pd.read_csv(pathset.path_txt_data_test)
    for i,ti in enumerate(data['title']):
        # print(line[0])
        text_id.append(str(data['num'][i]).strip('\t').strip('\ufeff').strip('"').strip('\t'))
        tweet.append(data['title'][i].strip('\t'))
        
        image_id.append(str(data['num'][i]).strip('\t').strip('\ufeff').strip('"').strip('\t'))   ## 1.jpg
        label.append(str(data['label'][i]).strip('\t'))
        mids_ctxt.append(data['claim_txt_entity'][i].strip('\t')) #对应进入一个列表
        mids_cimg.append(data['claim_img_entity'][i].strip('\t'))
        mids_dtxt.append(data['doc_txt_entity'][i].strip('\t')) #对应进入一个列表
        mids_dimg.append(data['doc_img_entity'][i].strip('\t'))
        document.append(data['text'][i].strip('\t'))
            # line = line[1].strip('[').strip(']')
    UNCASED = pathset.path_bert
    VOCAB = pathset.VOCAB
    tokenizer = BertTokenizer.from_pretrained(os.path.join(UNCASED, VOCAB))
    ctokens_ids = []
    dtokens_ids = []
    # for tw in tweet:
    #     tw = "[CLS] " + tw + " [SEP]"
    #     tkn = tokenizer.tokenize(tw)
    #     tkn_id = tokenizer.convert_tokens_to_ids(tkn)
    #     tokens_ids.append(tkn_id)
    string_process = StringProcess()
    for i in range(len(tweet)):
        sentence = string_process.clean_str_BERT(tweet[i])
        tokenizer_encoding = tokenizer(sentence, return_tensors='pt', padding='max_length',\
                                       truncation=True, max_length=config.sen_len)  ###句子编码   128的长度  TODO
        ctokens_ids.append(tokenizer_encoding)

    for i in range(len(tweet)):
        sentence = string_process.clean_str_BERT(document[i])
        tokenizer_encoding = tokenizer(sentence, return_tensors='pt', padding='max_length',\
                                       truncation=True, max_length=config.sen_len)  ###句子编码   128的长度
        dtokens_ids.append(tokenizer_encoding)


    mids_ctxt_all = []   ###mid??
    mids_cimg_all = []
    mids_dtxt_all = []   ###mid??
    mids_dimg_all = []

    for i in range(len(text_id)):
        mid = []
        # print(i)
        # print(image_id_train[i])
        # if image_id_train[i] in img2mid:
        #     img_mid = img2mid[image_id_train[i]]
        # mid = mids_txt[i] + img_mid
        for en in mids_ctxt[i].strip('[').strip(']').split(','):
            if en.strip(' ').strip('\'') != str(None):
                mid.append(en.strip(' ').strip('\'').lstrip('\''))
            # mid.append(en)

        mid = list(set(mid))  ### 一个text 对应的entity组成一个集合
        mids_ctxt_all.append(mid)   ### 所有的实体

    for i in range(len(text_id)):
        mid = []
        # mid = []
        # print(i)
        # print(image_id_train[i])
        # if image_id_train[i] in img2mid:
        #     img_mid = img2mid[image_id_train[i]]
        # mid = mids_txt[i] + img_mid
        for en in mids_cimg[i].strip('[').strip(']').split(','):
            if en.strip(' ').strip('\'') != str(None):
                mid.append(en.strip(' ').strip('\'').lstrip('\''))
            # mid.append(en)

        mid = list(set(mid))  ### 一个text 对应的entity组成一个集合
        mids_cimg_all.append(mid)   ### 所有的实体

    for i in range(len(text_id)):
        mid = []
        # print(i)
        # print(image_id_train[i])
        # if image_id_train[i] in img2mid:
        #     img_mid = img2mid[image_id_train[i]]
        # mid = mids_txt[i] + img_mid
        for en in mids_dtxt[i].strip('[').strip(']').split(','):
            if en.strip(' ').strip('\'') != str(None):
                mid.append(en.strip(' ').strip('\'').lstrip('\''))
            # mid.append(en)

        mid = list(set(mid))  ### 一个text 对应的entity组成一个集合
        mids_dtxt_all.append(mid)   ### 所有的实体

    for i in range(len(text_id)):
        mid = []
        # mid = []
        # print(i)
        # print(image_id_train[i])
        # if image_id_train[i] in img2mid:
        #     img_mid = img2mid[image_id_train[i]]
        # mid = mids_txt[i] + img_mid
        for en in mids_dimg[i].strip('[').strip(']').split(','):
            if en.strip(' ').strip('\'') != str(None):
                mid.append(en.strip(' ').strip('\'').lstrip('\''))
            # mid.append(en)

        mid = list(set(mid))  ### 一个text 对应的entity组成一个集合
        mids_dimg_all.append(mid)   ### 所有的实体



    X_title = ctokens_ids
    X_doc = dtokens_ids
    X_img = np.array(image_id)

    X_kg_ctxt = mids_ctxt_all   ### 所有entity
    x_kg_cimg = mids_cimg_all
    X_kg_dtxt = mids_dtxt_all   ### 所有entity
    x_kg_dimg = mids_dimg_all
    # X = np.array(tokens_ids)
    y = np.array(label)

    return X_title, X_doc, X_img, X_kg_ctxt, x_kg_cimg, X_kg_dtxt, x_kg_dimg, y


