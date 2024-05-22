import os
import sys

class path_set_BERT():
    def __init__(self,dataset):
        # self.path_data_dir = '/data/www888/pheme/data/pheme'
        #text\image\label
        if dataset == 'collected':
            self.path_txt_data_train = "../data/collected_data_trian.csv"
            self.path_txt_data_test = "../data/collected_data_test.csv"
            self.path_img_data_real = "../data/collected_img/img_real/"
            self.path_img_data_fake = "../data/collected_img/img_fake/"


        else:
            print('jump to synthetic')
            self.path_txt_data_train = "../data/synthetic_data_trian.csv"
            self.path_txt_data_test = "../data/synthetic_data_test.csv"
            self.path_img_data_real = "../data/synthetic_img/img_real/"
            self.path_img_data_fake = "../data/synthetic_img/img_fake/"


        #BERT_PATH
        self.path_bert = "your_path/bert-base-chinese/"
        self.VOCAB = 'vocab.txt'
        #TransE path
        self.path_transe = './entity/Freebase/embeddings/dimension_50/transe/entity2vec.bin'
        self.path_dic = './entity/Freebase/knowledge graphs/entity2id.txt'