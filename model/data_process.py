import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#import torchtext.vocab as vocab
from rand_fold import *
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append("../entity/")
import openke_query.query as query

# mid2fid = {}
# term2fid = {}
# term2mid = {}

class data_preprocess_ATT_bert_nfold():
    ## X_title, X_doc, X_img, X_kg_ctxt, x_kg_cimg, X_kg_dtxt, x_kg_dimg, y
    def __init__(self,title,document,images,mids_ctxt,mids_cimg,mids_dtxt,mids_dimg,y,config,pathset): 

        self.claim = title
        self.document = document
        self.images = images
        self.mids_ctxt = mids_ctxt
        self.mids_cimg = mids_cimg
        self.mids_dtxt = mids_dtxt
        self.mids_dimg = mids_dimg
        self.y = y

        
        query.build_index()
        query.build_dict() # m.xxxx -> 42

        self.mid2index = query.mid2fid

        self.embedding = query.get_embedding

        



        self.sen_len = config.sen_len                ## 128
        self.entity_len = config.entity_len      ### self.entity_len = 4

        # self.transE_path = pathset.path_transe  #/Freebase/embeddings/dimension_50/transe/entity2vec.bin
        self.pathset = pathset
        self.config = config
        ###?????
        # self.index2mid = []   ##用不上
        # self.mid2index = {}   ## entity的index
        # self.index2word = []   ## 用不上
        # self.word2index = {}   ### 用不上 
        # self.embedding_matrix = []
        # self.embedding_glove = []
    

    def top_dis_3entity_manhattan(self,tt_emb):
        top_dis = []
        # top_vec1 = []
        # top_vec2 = []
        dis_vec = {}

        tt_emb = torch.tensor(tt_emb)
        length = tt_emb.size(0)

        # print('length:{}'.format(length))

        for i in range(length):
            vec_list = []

            vec = tt_emb[i]

            vec1 = vec[0]
            vec2 = vec[1]
            vec_sub = vec1 - vec2
            manhattan_sim = vec_sub.norm(1)

            top_dis.append(manhattan_sim)
            vec_list.append(vec1)
            vec_list.append(vec2)
            dis_vec[manhattan_sim] = vec_list  


        count_sim = 0
        mid_post_list_top = []
        sim_post_list_top = []
        for sim in sorted(dis_vec,reverse=True):   #降序
            if count_sim >= 3:
                break
            else:
                count_sim += 1
                # print('top sim:',sim,'vec_list:',dis_vec[sim])
                mid_post_list_top.append(dis_vec[sim])   ###返回距离最大的向量对
                sim_post_list_top.append(sim)            ###返回距离由大到小


        count_sim = 0
        mid_post_list_botm = []
        sim_post_list_botm = []
        for sim in sorted(dis_vec,reverse=False):   #升序
            if count_sim >= 3:
                break
            else:
                count_sim += 1
                # print('top sim:',sim,'vec_list:',dis_vec[sim])
                mid_post_list_botm.append(dis_vec[sim])   ###返回距离最小的向量对
                sim_post_list_botm.append(sim)            ###返回距离由小到到


        return mid_post_list_top, sim_post_list_top,mid_post_list_botm,sim_post_list_botm
        


    def img_trans(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]
            )])
        return transform



    def load_data(self):
        #对bert处理
        # segments = []
        cattention_masks = []
        cinput_ids = []
        ctokens = self.claim    ### X_txt ，
        dattention_masks = []
        dinput_ids = []
        dtokens = self.document    ### X_txt ，
        # print(tkn['input_ids'],type(tkn['input_ids']))
        # print(type(tokens['input_ids']),tokens['input_ids'])
        for tkn in ctokens:
            # print(tkn['input_ids'],type(tkn['input_ids']))
            cinput_ids.append(tkn['input_ids'].squeeze(0))   ###
            cattention_masks.append(tkn['attention_mask'].squeeze(0))
        for tkn in dtokens:
            # print(tkn['input_ids'],type(tkn['input_ids']))
            dinput_ids.append(tkn['input_ids'].squeeze(0))   ###
            dattention_masks.append(tkn['attention_mask'].squeeze(0))

        # input_ids_tensor = torch.tensor(input_ids)
        # segments_tensors = torch.tensor(segments)
        # attention_masks_tensors = torch.tensor(attention_masks)

        #对label处理
        y = [int(float(label)) for label in self.y]
        y = torch.LongTensor(y)


        #对实体mids进行处理
        mid2index = self.mid2index      ## 
        # embedding = self.embedding  # torch.Size([86054151, 50]) 
        # print('transe size:',embedding.size())
        # count = 0

        # vec_post_1_top , vec_post_2_top , sim_post_top = 0,0,0
        # vec_post_1_botm, vec_post_2_botm, sim_post_botm = 0,0,0

        vec_all_1, vec_all_2, sim_all_list = [],[],[]

        for i in range(len(self.mids_ctxt)):   ### 2226

            # vec_1, vec_2, sim_list = {},{},{}
            vec_1 = {}
            vec_2 = {}
            sim_all = {}

            mid_ctxt_index = []## 提取编号
            mid_cimg_index = []
            mid_dtxt_index = []## 提取编号
            mid_dimg_index = []

            for mid in self.mids_ctxt[i]:       ### 一个txt里对应的mids
                if mid in mid2index:
                    mid_ctxt_index.append(mid2index[mid])  ###把一个post对应的entity编号提取出来  5或者6个

            for mid in self.mids_cimg[i]:
                if mid in mid2index:
                    mid_cimg_index.append(mid2index[mid])  ###把一个post对应的entity编号提取出来   5或者6个

            ## 下面讲document的编号也加进来

            for mid in self.mids_dimg[i]:
                if mid in mid2index:
                    mid_dimg_index.append(mid2index[mid])  ###把一个post对应的entity编号提取出来   5或者6个

            for mid in self.mids_dtxt[i]:
                if mid in mid2index:
                    mid_dtxt_index.append(mid2index[mid])  ###把一个post对应的entity编号提取出来   5或者6个
            
      
            # tt_emb, tv_emb, vv_emb = query.cross_modal_embedding(mid_ctxt_index,mid_cimg_index,4,20)
            tv_emb = query.cross_modal_embedding(mid_ctxt_index,mid_cimg_index,3,20)
            tt_emb = query.cross_modal_embedding(mid_ctxt_index,mid_dtxt_index,3,20)
            vv_emb = query.cross_modal_embedding(mid_cimg_index,mid_dimg_index,3,20)

        # # tt_emb: C(nt, 2) x 2 x 50
        # # tv_emb: (nt * nv) x 2 x 50
        # # vv_emb: C(nv, 2) x 2 x 50  
        #     list_emb.append(tt_emb)  
        #     list_emb.append(tv_emb)  
        #     list_emb.append(vv_emb)      

            i_emb = 0
            for what_emb in [tt_emb,tv_emb,vv_emb]:
                
                # what_emb = list_emb[j]

                ###每个entity集合对应的四个关键列表
                mid_post_list_top, sim_post_list_top,mid_post_list_botm,sim_post_list_botm = self.top_dis_3entity_manhattan(what_emb)


            # print(embedding.shape)
            # print(mid_post_index)
            # print(mid_post_index[1])
                sim_post_top = torch.tensor(sim_post_list_top)
                sim_post_top = sim_post_top.unsqueeze(0)  ###Expanding on the first dimension
                vec_post_1_1_top = mid_post_list_top[0][0].unsqueeze(0)
                vec_post_1_2_top = mid_post_list_top[0][1].unsqueeze(0)
                vec_post_2_1_top = mid_post_list_top[1][0].unsqueeze(0)
                vec_post_2_2_top = mid_post_list_top[1][1].unsqueeze(0)
                vec_post_3_1_top = mid_post_list_top[2][0].unsqueeze(0)
                vec_post_3_2_top = mid_post_list_top[2][1].unsqueeze(0)

                vec_post_1_top = torch.cat((vec_post_1_1_top, vec_post_2_1_top, vec_post_3_1_top),
                                    axis=0)   
                vec_post_2_top = torch.cat((vec_post_1_2_top, vec_post_2_2_top, vec_post_3_2_top),
                                    axis=0)



                sim_post_botm = torch.tensor(sim_post_list_botm)
                sim_post_botm = sim_post_botm.unsqueeze(0) 
                vec_post_1_1_botm = mid_post_list_botm[0][0].unsqueeze(0)
                vec_post_1_2_botm = mid_post_list_botm[0][1].unsqueeze(0)
                vec_post_2_1_botm = mid_post_list_botm[1][0].unsqueeze(0)
                vec_post_2_2_botm = mid_post_list_botm[1][1].unsqueeze(0)
                vec_post_3_1_botm = mid_post_list_botm[2][0].unsqueeze(0)
                vec_post_3_2_botm = mid_post_list_botm[2][1].unsqueeze(0)

                vec_post_1_botm = torch.cat((vec_post_1_1_botm, vec_post_2_1_botm, vec_post_3_1_botm),
                                    axis=0)   ###维度是怎样的 5* x * 50???
                vec_post_2_botm = torch.cat((vec_post_1_2_botm, vec_post_2_2_botm, vec_post_3_2_botm),
                                    axis=0)
                # sim_post = sim_post_top   ###五个最大距离


                if i_emb == 0:

                    vec_all_1_tt_top, vec_all_2_tt_top, sim_all_tt_top = vec_post_1_top,vec_post_2_top,sim_post_top
                    vec_all_1_tt_botm, vec_all_2_tt_botm, sim_all_tt_botm = vec_post_1_botm, vec_post_2_botm , sim_post_botm
                    
                    # nm_1 ='_tt'

                elif i_emb == 1:
                    vec_all_1_tv_top, vec_all_2_tv_top, sim_all_tv_top = vec_post_1_top,vec_post_2_top,sim_post_top
                    vec_all_1_tv_botm, vec_all_2_tv_botm, sim_all_tv_botm = vec_post_1_botm, vec_post_2_botm , sim_post_botm

                    # nm_1 = '_tv'

                elif i_emb == 2:
                    vec_all_1_vv_top, vec_all_2_vv_top, sim_all_vv_top = vec_post_1_top,vec_post_2_top,sim_post_top
                    vec_all_1_vv_botm, vec_all_2_vv_botm, sim_all_vv_botm = vec_post_1_botm, vec_post_2_botm , sim_post_botm
                    
                    # nm_1 = '_vv'

                else:
                    print('eroooooooooooooooooooooooooorrrrr')


                i_emb += 1
                
            # print(vec_all_1_tt_top) 
            ### 用字典封存一下
            vec_1['tt_top']= vec_all_1_tt_top
            vec_1['tt_botm']= vec_all_1_tt_botm
            vec_1['tv_top']= vec_all_1_tv_top
            vec_1['tv_botm']= vec_all_1_tv_botm
            vec_1['vv_top']= vec_all_1_vv_top
            vec_1['vv_botm']= vec_all_1_vv_botm

            vec_2['tt_top']= vec_all_2_tt_top
            vec_2['tt_botm']= vec_all_2_tt_botm
            vec_2['tv_top']= vec_all_2_tv_top
            vec_2['tv_botm']= vec_all_2_tv_botm
            vec_2['vv_top']= vec_all_2_vv_top
            vec_2['vv_botm']= vec_all_2_vv_botm

            sim_all['tt_top']= sim_all_tt_top
            sim_all['tt_botm']= sim_all_tt_botm
            sim_all['tv_top']= sim_all_tv_top
            sim_all['tv_botm']= sim_all_tv_botm
            sim_all['vv_top']= sim_all_vv_top
            sim_all['vv_botm']= sim_all_vv_botm        

            ###字典放到一个大列表里
            vec_all_1.append(vec_1)
            vec_all_2.append(vec_2)
            sim_all_list.append(sim_all)

        print('data preprocess OK!!!')               
        #train\val\test
        fold0_train,fold0_val= split_data(len(y), y)

        train_dict_0, val_dict_0, test_dict_0 = {}, {}, {}

       ###前两个参数是bert后的句子，image是图片编号，vec_1，vec_2是entity的向量
        names_dict = {'cinput_ids':cinput_ids,'cattention_masks':cattention_masks,'dinput_ids':dinput_ids,'dattention_masks':dattention_masks,\
                      'image':self.images,\
                      'vec_1':vec_all_1, 'vec_2':vec_all_2, 'sim_list':sim_all_list,'y':y}
        for name in names_dict:
            train_dict_0[name] = [names_dict[name][i] for i in fold0_train]
            # val_dict_0[name] = [names_dict[name][i] for i in fold0_val]
            # test_dict_0[name] = [names_dict[name][i] for i in fold0_test]



        return train_dict_0, val_dict_0

    def load_data_no_split(self):
        #对bert处理
        # segments = []
        attention_masks = []
        input_ids = []
        tokens = self.claim    ### X_txt ，用bert编码后的句子向量表示
        # print(tkn['input_ids'],type(tkn['input_ids']))
        # print(type(tokens['input_ids']),tokens['input_ids'])
        for tkn in tokens:
            # print(tkn['input_ids'],type(tkn['input_ids']))
            input_ids.append(tkn['input_ids'].squeeze(0))   ###去掉第一个维度
            attention_masks.append(tkn['attention_mask'].squeeze(0))

        # input_ids_tensor = torch.tensor(input_ids)
        # segments_tensors = torch.tensor(segments)
        # attention_masks_tensors = torch.tensor(attention_masks)

        #对label处理
        y = [int(label) for label in self.y]
        y = torch.LongTensor(y)


        #对实体mids进行处理
        mid2index = self.mid2index      ## 实体对应的编号
        # embedding = self.embedding  # torch.Size([86054151, 50]) 所有实体的代表向量
        # print('transe size:',embedding.size())
        # count = 0

        # vec_post_1_top , vec_post_2_top , sim_post_top = 0,0,0
        # vec_post_1_botm, vec_post_2_botm, sim_post_botm = 0,0,0

        vec_all_1, vec_all_2, sim_all_list = [],[],[]

        for i in range(len(self.mids_txt)):   ### 2226

            # vec_1, vec_2, sim_list = {},{},{}
            vec_1 = {}
            vec_2 = {}
            sim_all = {}

            mid_txt_index = []## 提取编号
            mid_img_index = []
            for mid in self.mids_txt[i]:       ### 一个txt里对应的mids
                if mid in mid2index:
                    mid_txt_index.append(mid2index[mid])  ###把一个post对应的entity编号提取出来  5或者6个

            for mid in self.mids_img[i]:
                if mid in mid2index:
                    mid_img_index.append(mid2index[mid])  ###把一个post对应的entity编号提取出来   5或者6个


            tt_emb, tv_emb, vv_emb = query.cross_modal_embedding(mid_txt_index,mid_img_index,4,20)
      
      
        #     list_emb = []   ###三个集合
        # # tt_emb: C(nt, 2) x 2 x 50
        # # tv_emb: (nt * nv) x 2 x 50
        # # vv_emb: C(nv, 2) x 2 x 50  
        #     list_emb.append(tt_emb)  
        #     list_emb.append(tv_emb)  
        #     list_emb.append(vv_emb)      

            i_emb = 0
            for what_emb in [tt_emb,tv_emb,vv_emb]:
                
                # what_emb = list_emb[j]

                ###每个entity集合对应的四个关键列表
                mid_post_list_top, sim_post_list_top,mid_post_list_botm,sim_post_list_botm = self.top_dis_3entity_manhattan(what_emb)


            # print(embedding.shape)
            # print(mid_post_index)
            # print(mid_post_index[1])
                sim_post_top = torch.tensor(sim_post_list_top)
                sim_post_top = sim_post_top.unsqueeze(0)  ###第一个维度扩展一下
                vec_post_1_1_top = mid_post_list_top[0][0].unsqueeze(0)
                vec_post_1_2_top = mid_post_list_top[0][1].unsqueeze(0)
                vec_post_2_1_top = mid_post_list_top[1][0].unsqueeze(0)
                vec_post_2_2_top = mid_post_list_top[1][1].unsqueeze(0)
                vec_post_3_1_top = mid_post_list_top[2][0].unsqueeze(0)
                vec_post_3_2_top = mid_post_list_top[2][1].unsqueeze(0)

                vec_post_1_top = torch.cat((vec_post_1_1_top, vec_post_2_1_top, vec_post_3_1_top),
                                    axis=0)   ###维度是怎样的 5* x * 50???
                vec_post_2_top = torch.cat((vec_post_1_2_top, vec_post_2_2_top, vec_post_3_2_top),
                                    axis=0)



                sim_post_botm = torch.tensor(sim_post_list_botm)
                sim_post_botm = sim_post_botm.unsqueeze(0)  ###第一个维度扩展一下
                vec_post_1_1_botm = mid_post_list_botm[0][0].unsqueeze(0)
                vec_post_1_2_botm = mid_post_list_botm[0][1].unsqueeze(0)
                vec_post_2_1_botm = mid_post_list_botm[1][0].unsqueeze(0)
                vec_post_2_2_botm = mid_post_list_botm[1][1].unsqueeze(0)
                vec_post_3_1_botm = mid_post_list_botm[2][0].unsqueeze(0)
                vec_post_3_2_botm = mid_post_list_botm[2][1].unsqueeze(0)

                vec_post_1_botm = torch.cat((vec_post_1_1_botm, vec_post_2_1_botm, vec_post_3_1_botm),
                                    axis=0)   ###维度是怎样的 5* x * 50???
                vec_post_2_botm = torch.cat((vec_post_1_2_botm, vec_post_2_2_botm, vec_post_3_2_botm),
                                    axis=0)
                # sim_post = sim_post_top   ###五个最大距离


                if i_emb == 0:

                    vec_all_1_tt_top, vec_all_2_tt_top, sim_all_tt_top = vec_post_1_top,vec_post_2_top,sim_post_top
                    vec_all_1_tt_botm, vec_all_2_tt_botm, sim_all_tt_botm = vec_post_1_botm, vec_post_2_botm , sim_post_botm
                    
                    # nm_1 ='_tt'

                elif i_emb == 1:
                    vec_all_1_tv_top, vec_all_2_tv_top, sim_all_tv_top = vec_post_1_top,vec_post_2_top,sim_post_top
                    vec_all_1_tv_botm, vec_all_2_tv_botm, sim_all_tv_botm = vec_post_1_botm, vec_post_2_botm , sim_post_botm

                    # nm_1 = '_tv'

                elif i_emb == 2:
                    vec_all_1_vv_top, vec_all_2_vv_top, sim_all_vv_top = vec_post_1_top,vec_post_2_top,sim_post_top
                    vec_all_1_vv_botm, vec_all_2_vv_botm, sim_all_vv_botm = vec_post_1_botm, vec_post_2_botm , sim_post_botm
                    
                    # nm_1 = '_vv'

                else:
                    print('eroooooooooooooooooooooooooorrrrr')


                i_emb += 1
                
            # print(vec_all_1_tt_top) 
            ### 用字典封存一下
            vec_1['tt_top']= vec_all_1_tt_top
            vec_1['tt_botm']= vec_all_1_tt_botm
            vec_1['tv_top']= vec_all_1_tv_top
            vec_1['tv_botm']= vec_all_1_tv_botm
            vec_1['vv_top']= vec_all_1_vv_top
            vec_1['vv_botm']= vec_all_1_vv_botm

            vec_2['tt_top']= vec_all_2_tt_top
            vec_2['tt_botm']= vec_all_2_tt_botm
            vec_2['tv_top']= vec_all_2_tv_top
            vec_2['tv_botm']= vec_all_2_tv_botm
            vec_2['vv_top']= vec_all_2_vv_top
            vec_2['vv_botm']= vec_all_2_vv_botm

            sim_all['tt_top']= sim_all_tt_top
            sim_all['tt_botm']= sim_all_tt_botm
            sim_all['tv_top']= sim_all_tv_top
            sim_all['tv_botm']= sim_all_tv_botm
            sim_all['vv_top']= sim_all_vv_top
            sim_all['vv_botm']= sim_all_vv_botm        

            ###字典放到一个大列表里
            vec_all_1.append(vec_1)
            vec_all_2.append(vec_2)
            sim_all_list.append(sim_all)

        print('data preprocess OK!!!')               
        #train\val\test
        fold0_train,fold0_val= split_data(len(y), y)

        train_dict_0, val_dict_0, test_dict_0 = {}, {}, {}

       ###前两个参数是bert后的句子，image是图片编号，vec_1，vec_2是entity的向量
        names_dict = {'input_ids':input_ids,'attention_masks':attention_masks,'image':self.images,\
                      'vec_1':vec_all_1, 'vec_2':vec_all_2, 'sim_list':sim_all_list,'y':y}
        for name in names_dict:
            train_dict_0[name] = [names_dict[name][i] for i in fold0_train]
            val_dict_0[name] = [names_dict[name][i] for i in fold0_val]
            # test_dict_0[name] = [names_dict[name][i] for i in fold0_test]


        return train_dict_0, val_dict_0

    def onlytest_load_data(self):
        #对bert处理
        # segments = []
        cattention_masks = []
        cinput_ids = []
        ctokens = self.claim    ### X_txt ，用bert编码后的句子向量表示
        dattention_masks = []
        dinput_ids = []
        dtokens = self.document    ### X_txt ，用bert编码后的句子向量表示
        # print(tkn['input_ids'],type(tkn['input_ids']))
        # print(type(tokens['input_ids']),tokens['input_ids'])
        for tkn in ctokens:
            # print(tkn['input_ids'],type(tkn['input_ids']))
            cinput_ids.append(tkn['input_ids'].squeeze(0))   ###去掉第一个维度
            cattention_masks.append(tkn['attention_mask'].squeeze(0))
        for tkn in dtokens:
            # print(tkn['input_ids'],type(tkn['input_ids']))
            dinput_ids.append(tkn['input_ids'].squeeze(0))   ###去掉第一个维度
            dattention_masks.append(tkn['attention_mask'].squeeze(0))

        # input_ids_tensor = torch.tensor(input_ids)
        # segments_tensors = torch.tensor(segments)
        # attention_masks_tensors = torch.tensor(attention_masks)

        #对label处理
        y = [int(float(label)) for label in self.y]
        y = torch.LongTensor(y)


        #对实体mids进行处理
        mid2index = self.mid2index      ## 实体对应的编号
        # embedding = self.embedding  # torch.Size([86054151, 50]) 所有实体的代表向量
        # print('transe size:',embedding.size())
        # count = 0

        # vec_post_1_top , vec_post_2_top , sim_post_top = 0,0,0
        # vec_post_1_botm, vec_post_2_botm, sim_post_botm = 0,0,0

        vec_all_1, vec_all_2, sim_all_list = [],[],[]

        for i in range(len(self.mids_ctxt)):   ### 2226

            # vec_1, vec_2, sim_list = {},{},{}
            vec_1 = {}
            vec_2 = {}
            sim_all = {}

            mid_ctxt_index = []## 提取编号
            mid_cimg_index = []
            mid_dtxt_index = []## 提取编号
            mid_dimg_index = []

            for mid in self.mids_ctxt[i]:       ### 一个txt里对应的mids
                if mid in mid2index:
                    mid_ctxt_index.append(mid2index[mid])  ###把一个post对应的entity编号提取出来  5或者6个

            for mid in self.mids_cimg[i]:
                if mid in mid2index:
                    mid_cimg_index.append(mid2index[mid])  ###把一个post对应的entity编号提取出来   5或者6个

            ## 下面讲document的编号也加进来

            for mid in self.mids_dimg[i]:
                if mid in mid2index:
                    mid_dimg_index.append(mid2index[mid])  ###把一个post对应的entity编号提取出来   5或者6个

            for mid in self.mids_dtxt[i]:
                if mid in mid2index:
                    mid_dtxt_index.append(mid2index[mid])  ###把一个post对应的entity编号提取出来   5或者6个
            
      
            ## 这次改了，只要tv了
            # tt_emb, tv_emb, vv_emb = query.cross_modal_embedding(mid_ctxt_index,mid_cimg_index,4,20)
            tv_emb = query.cross_modal_embedding(mid_ctxt_index,mid_cimg_index,4,20)
            tt_emb = query.cross_modal_embedding(mid_ctxt_index,mid_dtxt_index,4,20)
            vv_emb = query.cross_modal_embedding(mid_cimg_index,mid_dimg_index,4,20)
      
        #     list_emb = []   ###三个集合
        # # tt_emb: C(nt, 2) x 2 x 50
        # # tv_emb: (nt * nv) x 2 x 50
        # # vv_emb: C(nv, 2) x 2 x 50  
        #     list_emb.append(tt_emb)  
        #     list_emb.append(tv_emb)  
        #     list_emb.append(vv_emb)      

            i_emb = 0
            for what_emb in [tt_emb,tv_emb,vv_emb]:
                
                # what_emb = list_emb[j]

                ###每个entity集合对应的四个关键列表
                mid_post_list_top, sim_post_list_top,mid_post_list_botm,sim_post_list_botm = self.top_dis_3entity_manhattan(what_emb)


            # print(embedding.shape)
            # print(mid_post_index)
            # print(mid_post_index[1])
                sim_post_top = torch.tensor(sim_post_list_top)
                sim_post_top = sim_post_top.unsqueeze(0)  ###第一个维度扩展一下
                vec_post_1_1_top = mid_post_list_top[0][0].unsqueeze(0)
                vec_post_1_2_top = mid_post_list_top[0][1].unsqueeze(0)
                vec_post_2_1_top = mid_post_list_top[1][0].unsqueeze(0)
                vec_post_2_2_top = mid_post_list_top[1][1].unsqueeze(0)
                vec_post_3_1_top = mid_post_list_top[2][0].unsqueeze(0)
                vec_post_3_2_top = mid_post_list_top[2][1].unsqueeze(0)

                vec_post_1_top = torch.cat((vec_post_1_1_top, vec_post_2_1_top, vec_post_3_1_top),
                                    axis=0)   ###维度是怎样的 5* x * 50???
                vec_post_2_top = torch.cat((vec_post_1_2_top, vec_post_2_2_top, vec_post_3_2_top),
                                    axis=0)



                sim_post_botm = torch.tensor(sim_post_list_botm)
                sim_post_botm = sim_post_botm.unsqueeze(0)  ###第一个维度扩展一下
                vec_post_1_1_botm = mid_post_list_botm[0][0].unsqueeze(0)
                vec_post_1_2_botm = mid_post_list_botm[0][1].unsqueeze(0)
                vec_post_2_1_botm = mid_post_list_botm[1][0].unsqueeze(0)
                vec_post_2_2_botm = mid_post_list_botm[1][1].unsqueeze(0)
                vec_post_3_1_botm = mid_post_list_botm[2][0].unsqueeze(0)
                vec_post_3_2_botm = mid_post_list_botm[2][1].unsqueeze(0)

                vec_post_1_botm = torch.cat((vec_post_1_1_botm, vec_post_2_1_botm, vec_post_3_1_botm),
                                    axis=0)   ###维度是怎样的 5* x * 50???
                vec_post_2_botm = torch.cat((vec_post_1_2_botm, vec_post_2_2_botm, vec_post_3_2_botm),
                                    axis=0)
                # sim_post = sim_post_top   ###五个最大距离


                if i_emb == 0:

                    vec_all_1_tt_top, vec_all_2_tt_top, sim_all_tt_top = vec_post_1_top,vec_post_2_top,sim_post_top
                    vec_all_1_tt_botm, vec_all_2_tt_botm, sim_all_tt_botm = vec_post_1_botm, vec_post_2_botm , sim_post_botm
                    
                    # nm_1 ='_tt'

                elif i_emb == 1:
                    vec_all_1_tv_top, vec_all_2_tv_top, sim_all_tv_top = vec_post_1_top,vec_post_2_top,sim_post_top
                    vec_all_1_tv_botm, vec_all_2_tv_botm, sim_all_tv_botm = vec_post_1_botm, vec_post_2_botm , sim_post_botm

                    # nm_1 = '_tv'

                elif i_emb == 2:
                    vec_all_1_vv_top, vec_all_2_vv_top, sim_all_vv_top = vec_post_1_top,vec_post_2_top,sim_post_top
                    vec_all_1_vv_botm, vec_all_2_vv_botm, sim_all_vv_botm = vec_post_1_botm, vec_post_2_botm , sim_post_botm
                    
                    # nm_1 = '_vv'

                else:
                    print('eroooooooooooooooooooooooooorrrrr')


                i_emb += 1
                
            # print(vec_all_1_tt_top) 
            ### 用字典封存一下
            vec_1['tt_top']= vec_all_1_tt_top
            vec_1['tt_botm']= vec_all_1_tt_botm
            vec_1['tv_top']= vec_all_1_tv_top
            vec_1['tv_botm']= vec_all_1_tv_botm
            vec_1['vv_top']= vec_all_1_vv_top
            vec_1['vv_botm']= vec_all_1_vv_botm

            vec_2['tt_top']= vec_all_2_tt_top
            vec_2['tt_botm']= vec_all_2_tt_botm
            vec_2['tv_top']= vec_all_2_tv_top
            vec_2['tv_botm']= vec_all_2_tv_botm
            vec_2['vv_top']= vec_all_2_vv_top
            vec_2['vv_botm']= vec_all_2_vv_botm

            sim_all['tt_top']= sim_all_tt_top
            sim_all['tt_botm']= sim_all_tt_botm
            sim_all['tv_top']= sim_all_tv_top
            sim_all['tv_botm']= sim_all_tv_botm
            sim_all['vv_top']= sim_all_vv_top
            sim_all['vv_botm']= sim_all_vv_botm        

            ###字典放到一个大列表里
            vec_all_1.append(vec_1)
            vec_all_2.append(vec_2)
            sim_all_list.append(sim_all)

        print('test data preprocess OK!!!')


        idx = list(range(len(y)))   ##多少组数据
        # label_dict = {}
        idx_pos = []
        idx_neg = []
        for idx_temp in idx:
            # label_dict[idx_temp] = y[idx_temp].item()
            if y[idx_temp].item() == 1:
                idx_neg.append(idx_temp)
            elif y[idx_temp].item() == 0:
                idx_pos.append(idx_temp)
        print('测试数据正负例分布：','pos:',len(idx_pos),'neg',len(idx_neg))
        
        train_dict_0, val_dict_0= {}, {}    
        ###前两个参数是bert后的句子，image是图片编号，vec_1，vec_2是entity的向量
        names_dict = {'cinput_ids':cinput_ids,'cattention_masks':cattention_masks,'dinput_ids':dinput_ids,'dattention_masks':dattention_masks,\
                      'image':self.images,\
                      'vec_1':vec_all_1, 'vec_2':vec_all_2, 'sim_list':sim_all_list,'y':y}
        # for name in names_dict:

        #     train_dict_0[name] = [names_dict[name][i] for i in range(0,int(len(y)*0.8))]
        #     val_dict_0[name] = [names_dict[name][i] for i in range(int(len(y)*0.8),int(len(y)*0.9))]
        #     test_dict_0[name] = [names_dict[name][i] for i in range(int(len(y)*0.9),len(y))]

        

        return train_dict_0,val_dict_0,names_dict



