import os
import sys
import time

import random
# from sympy import arg
from tqdm import tqdm
import argparse
import pandas as pd
import csv
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from util import pre_training_www
from configs import www_Config
from path import path_set_BERT
from models import www_model
from data_process import data_preprocess_ATT_bert_nfold
from data_load import Dataset_all

# import sys

# sys.path.append("../process/")
import warnings

import os
import torch
import random
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if  torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


warnings.filterwarnings('ignore')





# F1 score


def generate(train_dict, val_dict, test_dict,model_mode, device, config, dataset, args,transform, pathset, p):
    # data
    model = www_model(config=config, pathset=pathset)
    # elif model_mode == 'original':
    #     pass
    model = model.to(device)
    # if torch.cuda.device_count()>1:
    model = torch.nn.DataParallel(model)

    # patience1 = patience
    # learning_rate_base1 = learning_rate_base
    # batch_size = batch_size
    # learning_rate_bert1 = learning_rate_bert


    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total number of parameters:', pytorch_total_trainable_params)
    # print(model.module)

    # =============================================================================================
    # 'input_ids': input_ids_tensor, 'attention_masks': attention_masks_tensors, 'image': self.images, \
    #         'vec_1': vec_all_1, 'vec_2': vec_all_2, 'sim_list': sim_all, 'y': y, 'y_img': y_img
    train_x_cinput_ids, train_x_cattention_masks, train_x_dinput_ids, train_x_dattention_masks, train_x_img, \
    train_x_kg_1, train_x_kg_2, train_x_kg_sim, \
    train_y = train_dict['cinput_ids'], train_dict['cattention_masks'], \
         train_dict['cinput_ids'], train_dict['cattention_masks'], train_dict['image'], \
              train_dict['vec_1'], train_dict['vec_2'], train_dict['sim_list'], \
              train_dict['y']
    

    from torchvision import transforms
    from PIL import Image
    import pickle

    www_transform = transforms.Compose([
        transforms.Resize(256),
        #图像调整为256x256像素
        transforms.CenterCrop(224),
        #将图像中心裁剪出来，大小为224x224像素
        transforms.ToTensor(),
        #图像转换为tensor数据类型
        #将图像的平均值和标准差设置为指定的值来正则化图像
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )])

    data = {}
    for i,(data_txt_cinput_ids,data_txt_cattention_masks, \
            data_txt_dinput_ids,data_txt_dattention_masks,p_image, \
                data_kg1,data_kg2,data_kg_sim,label) in enumerate(zip(train_x_cinput_ids, train_x_cattention_masks, train_x_dinput_ids, train_x_dattention_masks, train_x_img, \
    train_x_kg_1, train_x_kg_2, train_x_kg_sim, \
    train_y)):
    # for i,(post,claim,c_img,document,d_img) in enumerate(zip(all_post1,all_claim1,all_claim_image1,all_document1,all_document_image1)):
    # for n, row in tqdm(df.iterrows(), total=df.shape[0]):
    # path = '../data/train/'
    # path = '../data/val/'
        # pp_image = p_image
        # breakpoint()
        if int(label) == 0:
            print('-----------------------------')
            cimg_path = "./collected_img/img_real/" + 'claim_real/'
            dimg_path = "./collected_img/img_real/" + 'document_real/'
        else:
            cimg_path = "./collected_img/img_fake/" + 'claim_fake/'
            dimg_path = "./collected_img/img_fake/" + 'document_fake/'
        # cimg_path = self.pathset.path_img_data_claim
        cimage = Image.open(
            cimg_path + p_image + '.jpg'
        ).convert('RGB')
        cimage = www_transform(cimage)

        # dimg_path = self.pathset.path_img_data_doc
        dimage = Image.open(
            dimg_path + p_image + '.jpg'
        ).convert('RGB')
        dimage = www_transform(dimage)
        # except:
        #     breakpoint()

        # data[n] = (row['claim'], claim_image, row['document'], document_image, row['Label'])
        # data[i] = (claim, claim_image, document, document_image,label)
        data[i] = (data_txt_cinput_ids,data_txt_cattention_masks, \
            data_txt_dinput_ids,data_txt_dattention_masks,cimage, dimage, \
                data_kg1,data_kg2,data_kg_sim,label)
    with open('collected_train.pickle', 'wb') as file:
        pickle.dump(data, file)

    print('pickle train done----------------------')

    test_x_cinput_ids, test_x_cattention_masks, test_x_dinput_ids, test_x_dattention_masks, test_x_img, \
    test_x_kg_1, test_x_kg_2, test_x_kg_sim, \
    test_y = test_dict['cinput_ids'], test_dict['cattention_masks'], test_dict['dinput_ids'], test_dict['dattention_masks'],\
        test_dict['image'],  test_dict['vec_1'], test_dict['vec_2'], test_dict['sim_list'], \
            test_dict['y']

    data = {}
    for i,(data_txt_cinput_ids,data_txt_cattention_masks, \
            data_txt_dinput_ids,data_txt_dattention_masks,p_image, \
                data_kg1,data_kg2,data_kg_sim,label) in enumerate(zip(test_x_cinput_ids, test_x_cattention_masks, test_x_dinput_ids, test_x_dattention_masks, test_x_img, \
    test_x_kg_1, test_x_kg_2, test_x_kg_sim, \
    test_y)):
    # for i,(post,claim,c_img,document,d_img) in enumerate(zip(all_post1,all_claim1,all_claim_image1,all_document1,all_document_image1)):
    # for n, row in tqdm(df.iterrows(), total=df.shape[0]):
    # path = '../data/train/'
    # path = '../data/val/'
        # pp_image = p_image
        if int(label) == 0:
            cimg_path = "./collected_img/img_real/" + 'claim_real/'
            dimg_path = "./collected_img/img_real/" + 'document_real/'
        else:
            cimg_path = "./collected_img/img_fake/" + 'claim_fake/'
            dimg_path = "./collected_img/img_fake/" + 'document_fake/'
        # cimg_path = self.pathset.path_img_data_claim
        cimage = Image.open(
            cimg_path + p_image + '.jpg'
        ).convert('RGB')
        cimage = www_transform(cimage)

        # dimg_path = self.pathset.path_img_data_doc
        dimage = Image.open(
            dimg_path + p_image + '.jpg'
        ).convert('RGB')
        dimage = www_transform(dimage)
        # except:
        #     breakpoint()

        # data[n] = (row['claim'], claim_image, row['document'], document_image, row['Label'])
        # data[i] = (claim, claim_image, document, document_image,label)
        data[i] = (data_txt_cinput_ids,data_txt_cattention_masks, \
            data_txt_dinput_ids,data_txt_dattention_masks,cimage, dimage, \
                data_kg1,data_kg2,data_kg_sim,label)
    with open('collected_test.pickle', 'wb') as file:
        pickle.dump(data, file)

    print('pickle test done----------------------')



def main(args):
    os.environ['HOME'] = '/tmp'
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    set_seed(24)

    device = torch.device("cuda:0")

    dataset = args.dataset
    model_mode = args.model  #
    config = www_Config()
    pathset = path_set_BERT(dataset)

    print("loading data...")

    X_title, X_doc, X_img, X_kg_ctxt, x_kg_cimg, X_kg_dtxt, x_kg_dimg, y = pre_training_www(pathset, config,mode='train')


    preprocess = data_preprocess_ATT_bert_nfold(X_title, X_doc, X_img, X_kg_ctxt, x_kg_cimg, X_kg_dtxt, x_kg_dimg, y\
                                                , config, pathset)
    train_dict_0, val_dict_0 = preprocess.load_data()
    transform = preprocess.img_trans()

    ##处理 test 的地方
    print("loading test data...")
    X_title, X_doc, X_img, X_kg_ctxt, x_kg_cimg, X_kg_dtxt, x_kg_dimg, y = pre_training_www(pathset, config,mode='test')

    preprocess = data_preprocess_ATT_bert_nfold(X_title, X_doc, X_img, X_kg_ctxt, x_kg_cimg, X_kg_dtxt, x_kg_dimg, y,\
                                                 config, pathset)
    _, _, test_dict_0 = preprocess.onlytest_load_data()

        # f1 = train(train_dict_0, val_dict_0, test_dict_0, model_mode, device, config, dataset, args, transform, pathset, p=0, patience,learning_rate_base,batch_size,learning_rate_bert)
    generate(train_dict_0, val_dict_0, test_dict_0,model_mode, device, config, dataset, args, transform, pathset, p=7)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='www_matters')
    parser.add_argument('--dataset', type=str, default='collected')  #
    parser.add_argument('--model', type=str, default='www')
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--lambda_orth', type=float, default=1.5)
    # parser.add_argument('--model',type=str,default='initial')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--p',type=int,default=0)
    # parser.add_argument('--lambda_orth',type=float,default=1.5)
    args = parser.parse_args()
    main(args)