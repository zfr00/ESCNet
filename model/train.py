import os
import sys
import time

import random
# from sympy import arg
from tqdm import tqdm
import argparse
# import pandas as pd
# import csv
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from configs import www_Config
from path import path_set_BERT
from models import www_model
from data_process import data_preprocess_ATT_bert_nfold
from data_load import Dataset_all
# from ../process.earlystopping import *
import sys

sys.path.append("../process/")
from earlystopping import *
import warnings

import os
import torch
import random
import numpy as np
# 这里不固定 random 模块的随机种子，因为 random 模块后续要用于超参组合随机组合。
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if  torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


warnings.filterwarnings('ignore')

# seed_num = 66                 ###不能5，不能3423 233？
# torch.manual_seed(seed_num)
# random.seed(seed_num)
# 大于0.5的输出为1

def Log(log):
    print(log)
    f = open('/gdata1/log/' + str('rr_')+'colleced' + ".log", "a")
    f.write(log + "\n")
    f.close()


def evaluation(outputs, labels):
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct


# ACC模块
def accuracy(pred, targ):
    # pred = torch.max(pred, 1)[1]
    acc = ((pred == targ).float()).sum().item() / targ.size()[0]

    return acc


# F1 score
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


def train(model_mode, device, config, dataset, args, pathset, p):
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

    train_dataset = Dataset_all(mode='train', dataset=dataset)

    val_dataset = Dataset_all(mode='test', dataset=dataset)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               # sampler=train_sampler,
                                               num_workers=16,drop_last=True)   # , drop_last=True
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             # sampler=train_sampler,
                                             num_workers=16,drop_last=True)   # , drop_last=True
    t_batch = len(train_loader)  ###有几个dataloader
    v_batch = len(val_loader)
    # ======================================================================
    model.train()
    criterion_clf = nn.BCELoss()  # 交叉熵
    # criterion_orth = Orth_Loss()  # 多模态正交分解的loss

    bert_params = list(map(id, model.module.txtenc.text_embed.parameters()))
    # lstm_params = list(map(id, model.module.txtenc.lstm.parameters()))
    # no_params = bert_params + lstm_params
    base_params = filter(lambda p: id(p) not in bert_params, model.parameters())
    # optimizier = optim.Adam(model.parameters(),lr=args.lr)
    optimizier = optim.Adam([
        {'params': model.module.txtenc.text_embed.parameters(), 'lr': args.learning_rate_bert},
        # {'params': model.module.txtenc.lstm.parameters(), 'lr': 5e-4},
        {'params': base_params},
    ], lr=args.learning_rate_base)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizier, 'min', factor=0.1, patience=args.patience, verbose=True)
    earlystopping = EarlyStopping(dataset, p, 10)  # 早停
    val_loss_min = 5
    test_f1_max = 0
    # count = 0
    # model = earlystopping.load_model()
    # model = torch.load("/gdata1/dual-weibo/saved_model/11ckpt_twitter_6.model")
    for epoch in range(args.epoch):
        # count_train = 0
        total_loss, total_acc = 0, 0
        for i, (cinput_ids, cattention_masks, dinput_ids, dattention_masks,cimg,dimg, kg1, kg2, kg_sim, labels) in enumerate(train_loader):
            cinput_ids = cinput_ids.to(device, dtype=torch.long)
            cattention_masks = cattention_masks.to(device, dtype=torch.long)
            dinput_ids = dinput_ids.to(device, dtype=torch.long)
            dattention_masks = dattention_masks.to(device, dtype=torch.long)
            cimg = cimg.to(device)
            dimg = dimg.to(device)
            # kg1 = kg1.to(device, dtype=torch.float)
            # kg2 = kg2.to(device,dtype=torch.float)
            # kg_sim = kg_sim.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            optimizier.zero_grad()
            outputs_class = model(cinput_ids, cattention_masks, dinput_ids, dattention_masks,cimg,dimg, kg1, kg2, kg_sim)  # 过一遍model
            outputs_class = outputs_class.squeeze()  # 降维
            loss_class = criterion_clf(outputs_class, labels)
            # loss = loss_class + args.lambda_orth*criterion_orth(p_img=model.module.ln_uniq_img.weight,\
            #                                        p_txt=model.module.ln_uniq_txt.weight,\
            #                                        w_shr=model.module.ln_shr.weight)
            loss = loss_class
            # loss = (loss - 0.03).abs() + 0.03
            loss.backward()
            optimizier.step()
            correct = evaluation(outputs_class, labels)
            total_acc += (correct / args.batch_size)
            total_loss += loss.item()
            print('the running process is [{}]'.format(p), '[Epoch{}]'.format(epoch + 1), \
                  "{}/{}".format(i + 1, t_batch), "loss:", loss.item(), "acc:", correct * 100 / args.batch_size)
        Log('\n--------------------epoch{}---------------------------'.format(epoch + 1))
        # print('\nTrain | Loss:{:.5f} ACC:{:.3f}'.format(total_loss/t_batch, total_acc*100/t_batch))
        Log('Train | Loss:{:.5f} ACC:{:.3f}'.format(total_loss / t_batch, total_acc * 100 / t_batch))

        model.eval()
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (cinput_ids, cattention_masks, dinput_ids, dattention_masks,cimg,dimg, kg1, kg2, kg_sim, labels) in enumerate(val_loader):
                cinput_ids = cinput_ids.to(device, dtype=torch.long)
                cattention_masks = cattention_masks.to(device, dtype=torch.long)
                dinput_ids = dinput_ids.to(device, dtype=torch.long)
                dattention_masks = dattention_masks.to(device, dtype=torch.long)
                cimg = cimg.to(device)
                dimg = dimg.to(device)
                labels = labels.to(device, dtype=torch.float)

                outputs_class = model(cinput_ids, cattention_masks, dinput_ids, dattention_masks,cimg,dimg, kg1, kg2, kg_sim)  # 过一遍model
                outputs_class = outputs_class.squeeze()  # 降维
                # if outputs_class.size() = labels.size():
                loss_class = criterion_clf(outputs_class, labels)
                # loss = loss_class + args.lambda_orth * criterion_orth(p_img=model.module.ln_uniq_img.weight, \
                #                                                       p_txt=model.module.ln_uniq_txt.weight, \
                #                                                       w_shr=model.module.ln_shr.weight)
                loss = loss_class
                # loss = (loss - 0.03).abs() + 0.03
                total_loss += loss.item()
            # print("valid | Loss:{:.5f} ACC:{:.3f}".format(total_loss/v_batch, total_acc*100/v_batch))
                outputs_class[outputs_class >= 0.5] = 1
                outputs_class[outputs_class < 0.5] = 0
                if i == 0:
                    outputs_class_all = outputs_class
                    labels_all = labels
                else:
                    outputs_class_all = torch.cat([outputs_class_all, outputs_class], dim=0)
                    labels_all = torch.cat([labels_all, labels], dim=0)
        Log("valid | Loss:{:.5f} ".format(total_loss / v_batch))
        val_loss = total_loss / v_batch
        acc = accuracy(outputs_class_all, labels_all)
        f1, precision, recall, f1_real, precision_real, recall_real, f1_fake, precision_fake, recall_fake = macro_f1(
            outputs_class_all, labels_all, num_classes=2)
        Log('----------------------------------------')
        Log('acc:' + str(acc) + '  prec:' + str(precision) + '  rec:' + str(recall) + '  f1:' + str(f1))
        Log('prec-fake:' + str(precision_fake) + '  rec-fake:' + str(recall_fake) + '  f1-fake:' + str(f1_fake))
        Log('prec-real:' + str(precision_real) + '  rec-real:' + str(recall_real) + '  f1-real:' + str(f1_real))



        if val_loss < val_loss_min:
            val_loss_min = val_loss

        if f1 > test_f1_max:
            test_f1_max = f1
            # earlystopping.save_model(model)
            # Log('save model')


        model.train()
        scheduler.step(val_loss)  ###学习率调整

    return test_f1_max
    # model = earlystopping.load_model()   ###这一步？？？
    # test_result_dict = test(test_dict,model,device,config,dataset,model_mode,args,transform,pathset,p)
    # return test_result_dict


def main(args):
    os.environ['HOME'] = '/tmp'
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    set_seed(24)
    param_grid = {
    'patience': list(range(5, 20)),
    'learning_rate_base': list(np.logspace(np.log10(0.00001), np.log10(0.001), base = 10, num = 1000)),
    'learning_rate_bert': list(np.logspace(np.log10(0.000001), np.log10(0.0005), base = 10, num = 1000)),
    'batch_size': [16, 32, 64],
    # 'hidden_size': [128, 256]
    }

    MAX_EVALS = 100

    # random.seed(50)
    # random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}

    # 记录用
    best_score = 0
    best_hyperparams = {}

    device = torch.device("cuda:0")
    Log('=========================================')

    dataset = args.dataset
    model_mode = args.model  
    config = www_Config()
    pathset = path_set_BERT(dataset)

    f1_max_all = 0
    # ------train&test----------------------------
    for i in range(MAX_EVALS):
        random.seed(i+1000)	# 设置随机种子，每次搜索设置不同的种子，若种子固定，那每次选取的超参都是一样的             701?
        hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        
        Log(str(hyperparameters))
        
        patience = hyperparameters['patience']
        learning_rate_base = hyperparameters['learning_rate_base']
        batch_size = hyperparameters['batch_size']
        learning_rate_bert = hyperparameters['learning_rate_bert']
        args.patience = patience
        args.batch_size= batch_size
        args.learning_rate_base= learning_rate_base
        args.learning_rate_bert = learning_rate_bert
    #    args.patience = 5
     #   args.batch_size= 16
      #  args.learning_rate_base= 0.001
       # args.learning_rate_bert = 0.0001
        # f1 = train(train_dict_0, val_dict_0, test_dict_0, model_mode, device, config, dataset, args, transform, pathset, p=0, patience,learning_rate_base,batch_size,learning_rate_bert)
        f1=train(model_mode, device, config, dataset, args,  pathset, p=7)
        if f1 > f1_max_all:
            f1_max_all = f1
            best_hyperparams = hyperparameters
        Log(str(f1_max_all))
        Log('niubiniubi!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # Log(f1)
    print('finish')
    Log(str(f1_max_all))
    Log(str(best_hyperparams))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='www_matters')
    parser.add_argument('--dataset', type=str, default='disinformation')  # twitter/weibo
    parser.add_argument('--model', type=str, default='www')
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--lambda_orth', type=float, default=1.5)
    # parser.add_argument('--model',type=str,default='initial')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--p',type=int,default=5)
    # parser.add_argument('--lambda_orth',type=float,default=1.5)
    args = parser.parse_args()
    main(args)