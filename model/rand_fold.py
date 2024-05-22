import random
from random import shuffle
import os

def split_data(size,y, val=0, train=0.8, shuffle=True): #train6:val2:test2
    idx = list(range(size))   ##多少组数据
    # label_dict = {}
    idx_pos = []
    idx_neg = []
    for idx_temp in idx:
        # label_dict[idx_temp] = y[idx_temp].item()
        if y[idx_temp].item() == 1:
            idx_neg.append(idx_temp)
        elif y[idx_temp].item() == 0:
            idx_pos.append(idx_temp)
    print('数据正负例分布：','pos:',len(idx_pos),'neg',len(idx_neg))
    if shuffle:
        random.shuffle(idx_neg)
        random.shuffle(idx_pos)
    fold0_x_train = []
    fold0_x_val = []
    leng1 = int(len(idx_pos) * val)
    leng2 = int(len(idx_neg) * val)

    fold0_x_val.extend(idx_pos[0:leng1])
    fold0_x_val.extend(idx_neg[0:leng2])
    # temp_pos = idx_pos-idx_pos[0:leng1]
    fold0_x_train.extend(idx_pos[leng1:])
    fold0_x_train.extend(idx_neg[leng2:])


    fold0_x_train = list(fold0_x_train)
    random.shuffle(fold0_x_train)
    fold0_val = list(fold0_x_val)
    random.shuffle(fold0_val)


    # print('val idx:',val_idx)
    # print('train数据正负例分布：', 'pos:', len(train_idx_pos), 'neg', len(train_idx_neg))
    # print('val数据正负例分布：', 'pos:', len(val_idx_pos), 'neg', len(val_idx_neg))
    # print('test数据正负例分布：', 'pos:', len(test_idx_pos), 'neg', len(test_idx_neg))
    return list(fold0_x_train),list(fold0_val)




####  数据交叉验证
def split_data_5fold(size,y, train, val, test, shuffle=True): #train6:val2:test2
    idx = list(range(size))   ##duosh多少组数据
    # label_dict = {}
    idx_pos = []
    idx_neg = []
    for idx_temp in idx:
        # label_dict[idx_temp] = y[idx_temp].item()
        if y[idx_temp].item() == 1:   ###取出单元素张量的元素值并返回该值，保持原元素类型不变
            idx_neg.append(idx_temp)
        elif y[idx_temp].item() == 0:
            idx_pos.append(idx_temp)
    print('数据正负例分布：','pos:',len(idx_pos),'neg',len(idx_neg))
    if shuffle:
        random.shuffle(idx_neg)
        random.shuffle(idx_pos)
    fold0_x_test, fold1_x_test, fold2_x_test, fold3_x_test, fold4_x_test = [], [], [], [], []
    fold0_x_train, fold1_x_train, fold2_x_train, fold3_x_train, fold4_x_train = [], [], [], [], []
    fold0_x_val, fold1_x_val, fold2_x_val, fold3_x_val, fold4_x_val = [], [], [], [], []
    leng1 = int(len(idx_pos) * test)
    leng2 = int(len(idx_neg) * test)



    fold0_x_test.extend(idx_pos[0:leng1])
    fold0_x_test.extend(idx_neg[0:leng2])
    # temp_pos = idx_pos-idx_pos[0:leng1]
    temp_pos = [idx for idx in idx_pos if idx not in idx_pos[0:leng1]]  ###剩下的idx数据 pos
    # temp_neg = idx_neg-idx_neg[0:leng2]
    temp_neg = [idx for idx in idx_neg if idx not in idx_neg[0:leng2]]
    leng3 = int(len(temp_pos) * val)
    leng4 = int(len(temp_neg) * val)
    fold0_x_val.extend(temp_pos[0:leng3])
    fold0_x_val.extend(temp_neg[0:leng4])
    fold0_x_train.extend(temp_pos[leng3:])
    fold0_x_train.extend(temp_neg[leng4:])

    fold1_x_test.extend(idx_pos[leng1:leng1*2])
    fold1_x_test.extend(idx_neg[leng2:leng2*2])
    # temp_pos = idx_pos - idx_pos[leng1:leng1*2]
    # temp_neg = idx_neg - idx_neg[leng2:leng2*2]
    temp_pos = [idx for idx in idx_pos if idx not in idx_pos[leng1:leng1*2]]
    temp_neg = [idx for idx in idx_neg if idx not in idx_neg[leng2:leng2*2]]
    leng3 = int(len(temp_pos) * val)
    leng4 = int(len(temp_neg) * val)
    fold1_x_val.extend(temp_pos[0:leng3])
    fold1_x_val.extend(temp_neg[0:leng4])
    fold1_x_train.extend(temp_pos[leng3:])
    fold1_x_train.extend(temp_neg[leng4:])

    fold2_x_test.extend(idx_pos[leng1*2:leng1*3])
    fold2_x_test.extend(idx_neg[leng2*2:leng2*3])
    # temp_pos = idx_pos - idx_pos[leng1*2:leng1*3]
    # temp_neg = idx_neg - idx_neg[leng2*2:leng2*3]
    temp_pos = [idx for idx in idx_pos if idx not in idx_pos[leng1*2:leng1*3]]
    temp_neg = [idx for idx in idx_neg if idx not in idx_neg[leng2*2:leng2*3]]
    leng3 = int(len(temp_pos) * val)
    leng4 = int(len(temp_neg) * val)
    fold2_x_val.extend(temp_pos[0:leng3])
    fold2_x_val.extend(temp_neg[0:leng4])
    fold2_x_train.extend(temp_pos[leng3:])
    fold2_x_train.extend(temp_neg[leng4:])

    fold3_x_test.extend(idx_pos[leng1*3:leng1*4])
    fold3_x_test.extend(idx_neg[leng2*3:leng2*4])
    # temp_pos = idx_pos - idx_pos[leng1*3:leng1*4]
    # temp_neg = idx_neg - idx_neg[leng2*3:leng2*4]
    temp_pos = [idx for idx in idx_pos if idx not in idx_pos[leng1*3:leng1*4]]
    temp_neg = [idx for idx in idx_neg if idx not in idx_neg[leng2*3:leng2*4]]
    leng3 = int(len(temp_pos) * val)
    leng4 = int(len(temp_neg) * val)
    fold3_x_val.extend(temp_pos[0:leng3])
    fold3_x_val.extend(temp_neg[0:leng4])
    fold3_x_train.extend(temp_pos[leng3:])
    fold3_x_train.extend(temp_neg[leng4:])

    fold4_x_test.extend(idx_pos[leng1*4:])
    fold4_x_test.extend(idx_neg[leng2*4:])
    # temp_pos = idx_pos - idx_pos[leng1*4:]
    # temp_neg = idx_neg - idx_neg[leng2*4:]
    temp_pos = [idx for idx in idx_pos if idx not in idx_pos[leng1*4:]]
    temp_neg = [idx for idx in idx_neg if idx not in idx_neg[leng2*4:]]
    leng3 = int(len(temp_pos) * val)
    leng4 = int(len(temp_neg) * val)
    fold4_x_val.extend(temp_pos[0:leng3])
    fold4_x_val.extend(temp_neg[0:leng4])
    fold4_x_train.extend(temp_pos[leng3:])
    fold4_x_train.extend(temp_neg[leng4:])

    fold0_test = list(fold0_x_test)
    random.shuffle(fold0_test)
    fold0_val = list(fold0_x_val)
    random.shuffle(fold0_val)
    fold0_train = list(fold0_x_train)
    random.shuffle(fold0_train)

    fold1_test = list(fold1_x_test)
    random.shuffle(fold1_test)
    fold1_val = list(fold1_x_val)
    random.shuffle(fold1_val)
    fold1_train = list(fold1_x_train)
    random.shuffle(fold1_train)

    fold2_test = list(fold2_x_test)
    random.shuffle(fold2_test)
    fold2_val = list(fold2_x_val)
    random.shuffle(fold2_val)
    fold2_train = list(fold2_x_train)
    random.shuffle(fold2_train)

    fold3_test = list(fold3_x_test)
    random.shuffle(fold3_test)
    fold3_val = list(fold3_x_val)
    random.shuffle(fold3_val)
    fold3_train = list(fold3_x_train)
    random.shuffle(fold3_train)

    fold4_test = list(fold4_x_test)
    random.shuffle(fold4_test)
    fold4_val = list(fold4_x_val)
    random.shuffle(fold4_val)
    fold4_train = list(fold4_x_train)
    random.shuffle(fold4_train)

    # print('val idx:',val_idx)
    # print('train数据正负例分布：', 'pos:', len(train_idx_pos), 'neg', len(train_idx_neg))
    # print('val数据正负例分布：', 'pos:', len(val_idx_pos), 'neg', len(val_idx_neg))
    # print('test数据正负例分布：', 'pos:', len(test_idx_pos), 'neg', len(test_idx_neg))
    return list(fold0_test),list(fold0_val), list(fold0_train), \
           list(fold1_test),list(fold1_val), list(fold1_train), \
           list(fold2_test),list(fold2_val), list(fold2_train), \
           list(fold3_test),list(fold3_val), list(fold3_train), \
           list(fold4_test),list(fold4_val), list(fold4_train)