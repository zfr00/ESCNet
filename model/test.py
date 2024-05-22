import os
import sys
import time

import random
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
# from loss import Orth_Loss
#from ../process.earlystopping import *
import sys
sys.path.append("../process/")
from earlystopping import *
import warnings
warnings.filterwarnings('ignore')
#大于0.5的输出为1

def Log(log):
    print(log)
    f = open('./log/'+args.dataset+'_test'+".log", "a")
    f.write(log+"\n")
    f.close()


def evaluation(outputs,labels):
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    correct = torch.sum(torch.eq(outputs,labels)).item()
    return correct
#ACC模块
def accuracy(pred, targ):
    # pred = torch.max(pred, 1)[1]
    acc = ((pred == targ).float()).sum().item() / targ.size()[0]

    return acc
#F1 score
def macro_f1(pred, targ, num_classes=None):
    # pred = torch.max(pred, 1)[1]
    tp_out = []
    fp_out = []
    fn_out = []
    if num_classes is None:
        num_classes = sorted(set(targ.cpu().numpy().tolist()))
    else:
        num_classes = range(num_classes)
    for i in num_classes:
        tp = ((pred == i) & (targ == i)).sum().item()  # 预测为i，且标签的确为i的
        fp = ((pred == i) & (targ != i)).sum().item()  # 预测为i，但标签不是为i的
        fn = ((pred != i) & (targ == i)).sum().item()  # 预测不是i，但标签是i的
        tp_out.append(tp)
        fp_out.append(fp)
        fn_out.append(fn)

    eval_tp = np.array(tp_out)
    eval_fp = np.array(fp_out)
    eval_fn = np.array(fn_out)

    precision = eval_tp / (eval_tp + eval_fp)
    precision[np.isnan(precision)] = 0
    precision_real = precision[0]
    precision_fake = precision[1]
    precision = np.mean(precision)

    recall = eval_tp / (eval_tp + eval_fn)
    recall[np.isnan(recall)] = 0
    recall_real = recall[0]
    recall_fake = recall[1]
    recall = np.mean(recall)

    f1 = 2 * (precision * recall) / (precision + recall)
    f1_real = 2 * (precision_real * recall_real) / (precision_real + recall_real)
    f1_fake = 2 * (precision_fake * recall_fake) / (precision_fake + recall_fake)
    return f1, precision, recall, f1_real, precision_real, recall_real, f1_fake, precision_fake, recall_fake


def test(test_dict,model,device,config,dataset,model_mode,args,transform,pathset,p):
    test_x_input_ids, test_x_attention_masks, test_x_img, \
    test_x_kg_1, test_x_kg_2, test_x_kg_sim, \
    test_y = test_dict['input_ids'], test_dict['attention_masks'], test_dict['image'], \
             test_dict['vec_1'], test_dict['vec_2'], test_dict['sim_list'], \
             test_dict['y']

    test_dataset = Dataset_all(x_txt_input_ids=test_x_input_ids, \
                               x_txt_attention_masks=test_x_attention_masks, \
                               x_img=test_x_img, x_kg1=test_x_kg_1, x_kg2=test_x_kg_2, \
                               x_kg_sim=test_x_kg_sim, y=test_y, transform=transform,
                               pathset=pathset, )
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch,
                                              shuffle=False,
                                              # sampler=train_sampler,
                                              num_workers=2, drop_last=True)
    model.eval()
    with torch.no_grad():
        for i, (input_ids, attention_masks, img, kg1, kg2, kg_sim, labels) in enumerate(test_loader):
            input_ids = input_ids.to(device, dtype=torch.long)
            attention_masks = attention_masks.to(device, dtype=torch.long)
            img = img.to(device)
            # kg1 = kg1.to(device, dtype=torch.float)
            # kg2 = kg2.to(device, dtype=torch.float)
            # kg_sim = kg_sim.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            # breakpoint()
            outputs_class = model(input_ids, attention_masks, img, kg1, kg2, kg_sim)
            outputs_class = outputs_class.squeeze()

            outputs_class[outputs_class >= 0.5] = 1
            outputs_class[outputs_class < 0.5] = 0
            if i == 0:
                outputs_class_all = outputs_class
                labels_all = labels
            else:
                outputs_class_all = torch.cat([outputs_class_all, outputs_class], dim=0)
                labels_all = torch.cat([labels_all, labels], dim=0)
    acc = accuracy(outputs_class_all, labels_all)
    f1, precision, recall, f1_real, precision_real, recall_real, f1_fake, precision_fake, recall_fake = macro_f1(
        outputs_class_all, labels_all, num_classes=2)
    Log('----------------------------------------')
    Log('acc:' + str(acc) + '  prec:' + str(precision) + '  rec:' + str(recall) + '  f1:' + str(f1))
    Log('prec-fake:' + str(precision_fake) + '  rec-fake:' + str(recall_fake) + '  f1-fake:' + str(f1_fake))
    Log('prec-real:' + str(precision_real) + '  rec-real:' + str(recall_real) + '  f1-real:' + str(f1_real))

    return f1



def main(args):
    os.environ['HOME'] = '/tmp'
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    device = torch.device("cuda:0")
    print('=========================================')

    dataset = args.dataset
    model_mode = args.model  ##www?
    config = www_Config()
    pathset = path_set_BERT(dataset)

     ##处理 test 的地方
    print("loading test data...")
    X_title, X_doc, X_img, X_kg_ctxt, x_kg_cimg, X_kg_dtxt, x_kg_dimg, y = pre_training_www(pathset, config,mode='test')

    preprocess = data_preprocess_ATT_bert_nfold(X_title, X_doc, X_img, X_kg_ctxt, x_kg_cimg, X_kg_dtxt, x_kg_dimg, y,\
                                                 config, pathset)
    _, _, test_dict = preprocess.onlytest_load_data()
    transform = preprocess.img_trans()


    p = 68


    model = www_model(config=config,pathset=pathset)
    # elif model_mode == 'original':
    #     pass
    model = model.to(device)
    # if torch.cuda.device_count()>1:
    #model = torch.nn.DataParallel(model)
    
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total number of parameters:',pytorch_total_trainable_params)
    # print(model.module)
    earlystopping = EarlyStopping(dataset,p,10)   #早停

    model = earlystopping.load_model()   ###
    model = model.module    
    test_result_dict = test(test_dict,model,device,config,dataset,model_mode,args,transform,pathset,p)

    print('finish')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='www_matters')
    parser.add_argument('--dataset',type=str,default='twitter')#twitter/weibo
    parser.add_argument('--model',type=str,default='www')
    parser.add_argument('--cuda',type=str,default='0')
    parser.add_argument('--epoch',type=int,default=50)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--lambda_orth',type=float,default=1.5)
    # parser.add_argument('--model',type=str,default='initial')
    parser.add_argument('--batch',type=int,default=16)
    # parser.add_argument('--lambda_orth',type=float,default=1.5)
    args = parser.parse_args()
    main(args)
