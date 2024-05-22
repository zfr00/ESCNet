from torch.utils import data
from PIL import Image
import os
import numpy as np
import torch
import pickle
class Dataset_all(data.Dataset):
    def __init__(self,mode,dataset):
        
        # if dataset == 'disinformation':
        print('read data: /gdata1/model/{}_{}.pickle'.format(dataset,mode))
        with open('/gdata1/model/{}_{}.pickle'.format(dataset,mode), 'rb') as f:  ### your_path
            self.data = pickle.load(f)


    def __getitem__(self, index):

        self.data_txt_cinput_ids,self.data_txt_cattention_masks, \
            self.data_txt_dinput_ids,self.data_txt_dattention_masks,cimage, dimage, \
                self.data_kg1,self.data_kg2,self.data_kg_sim,self.label = self.data[index]        


        # else:
        #     image = Image.fromarray(128*np.ones((256,256,3),dtype = np.uint8))
        return self.data_txt_cinput_ids,self.data_txt_cattention_masks, \
            self.data_txt_dinput_ids,self.data_txt_dattention_masks,cimage, dimage, \
                self.data_kg1,self.data_kg2,self.data_kg_sim,self.label
    def __len__(self):
        return len(self.data)