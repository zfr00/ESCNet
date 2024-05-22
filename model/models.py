# from numpy import imag
import math
from torch.nn import init
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable,Function
from layers import TextExtract_Bert_lstm,ImgPNet,mydecoder
from layers import *
from transformers import BertModel, BertConfig
# from einops import rearrange

class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        
        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)     ###  384  *  384
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)     ###   384  * 384

        self.do = nn.Dropout(dropout)             

        self.scale = math.sqrt(hid_dim / n_heads)
        # self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))


    def forward(self, query, key, value, mask=None):


        bsz = query.shape[0]    ## 16

        Q = self.w_q(query)   
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)     ###  16 * 1 *  6  *  64
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //  ### 16 * 400 *    6  * 64
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)


        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale


        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = self.do(torch.softmax(energy, dim=-1))

        x = torch.matmul(attention, V)

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        x = self.fc(x)

        return x

class Scaled_Dot_Product_Attention_pos(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention_pos, self).__init__()

    def forward(self, Q, K, V, scale,kg_sim):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]   ## 16 * 1 * 80
            K: [batch_size, len_K, dim_K]   ## 16 * 1 * 80
            V: [batch_size, len_V, dim_V]
        '''  ## k # 128,5,80
        attention = torch.matmul(Q, K.permute(0, 2, 1))     ###
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)   
        beta = torch.mul(attention, kg_sim)
        beta = F.softmax(beta,dim = -1)
        # print('beta size:',beta.size()) #128,1,5
        # print('v size:',V.size())#128,5,80
        context = torch.matmul(beta, V)   ## 16 * 1 * 80
        # print('v after attention:',context.size()) #128,1,80
        return context

class Scaled_Dot_Product_Attention_neg(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention_neg, self).__init__()

    def forward(self, Q, K, V, scale, kg_sim):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
        '''
        attention = -1*torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = -1*F.softmax(attention, dim=-1)
        beta = torch.mul(attention, kg_sim)
        beta = F.softmax(beta, dim=-1)
        # print('beta size:', beta.size())  # 128,1,5
        # print('v size:',V.size())#128,5,80
        context = torch.matmul(beta, V)
        # print('v after attention:', context.size())  # 128,1,80
        # context = torch.matmul(attention, V)
        return context

class www_model(nn.Module):
    def __init__(self,config,pathset):
        #img_hidden_size, bert_path, hidden_dim, num_layers, dropout=0.5
        super(www_model, self).__init__()
        self.txt_hidden_dim = config.hidden_dim   # 150
        self.img_hidden_size = config.img_hidden_size  #2048
        self.bert_path = pathset.path_bert  
        self.dropout = config.dropout
        self.num_layers = config.num_layers   # 1
        # self.clf = nn.Linear(self.img_hidden_size+self.txt_hidden_dim*2,2)
        # self.clf = nn.Sequential(nn.Linear(self.img_hidden_size+self.txt_hidden_dim*2+100, 1),
        #                          nn.Sigmoid())
        # self.dropout = dropout
        self.ln_c = nn.Linear(768, 384)    # 150 * 2？
        self.ln_d = nn.Linear(768, 384)
        self.ln_shr = nn.Linear(384, 40, bias=False)             ## share
        # self.ln_uniq_txt = nn.Linear(200, 40, bias=False)        ### unique
        # self.ln_uniq_img = nn.Linear(200, 40, bias=False)        ## unique
        self.ln_kg1 = nn.Linear(50, 40)
        # self.ln_kg2 = nn.Linear(160, 120)      
        self.ln_kg2 = nn.Linear(160, 40)      

        # self.ln_kg3= nn.Linear(120,40)
        self.txtenc = TextExtract_Bert_lstm()
        self.imgenc = ImgPNet()

        # ###  MSD
        self.dict_feature = nn.Parameter(torch.randn(1, 40, 384))
        self.MSD = mydecoder(dim=384, depth=1, heads=6,
                                        mlp_dim=768, pool='cls', dim_head=64,
                                        dropout=0.3, emb_dropout=0.)
        # self.MSD = SelfAttention(384,6,0.)
        # 损失函数增加unique layer和shared layer
        # ---------------------------------
        # 多头attention

        self.num_head = 1
        self.dim_model = 80
        assert self.dim_model % self.num_head == 0
        self.dim_head = self.dim_model // self.num_head   ## 80？
        self.fc_Q_pos = nn.Linear(self.dim_model, self.num_head * self.dim_head)   # 80*80
        self.fc_K_pos = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_V_pos = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_Q_neg = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_K_neg = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_V_neg = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.attention_pos = Scaled_Dot_Product_Attention_pos()
        self.attention_neg = Scaled_Dot_Product_Attention_neg()
        self.fc1 = nn.Linear(self.num_head * self.dim_head, self.dim_model)   # 80 * 80
        self.fc2 = nn.Linear(self.num_head * self.dim_head, self.dim_model)
        self.dropout = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.dim_model)  ## 80  



        self.rumor_classifier = nn.Sequential()
        self.rumor_classifier.add_module('r_fc1',nn.Linear(360,180))
        self.rumor_classifier.add_module('r_relu1',nn.LeakyReLU(True))
        self.rumor_classifier.add_module('r_fc2', nn.Linear(180, 1))
        self.rumor_classifier.add_module('r_softmax', nn.Sigmoid())

        # self.device = torch.device("cuda:0")
        self.atten = SelfAttention(40,1,0.)

    def stance_transformer(self, c_feature, d_feature):
        c_feature = self.ln_c(c_feature)
        d_feature = self.ln_d(d_feature)
        B = c_feature.size(0)
        dict_feature = self.dict_feature.repeat(B,1,1)   ## 16 * 40 * 384
        # print(dict_feature.shape)
        c_share_raw = self.MSD(dict_feature,c_feature)    ### B * 40 * 384
        d_share_raw = self.MSD(dict_feature,d_feature)    ### B * 40 * 384
        # print(txt_share_raw.shape)
        c_share = torch.mean(c_share_raw,dim=1)
        d_share = torch.mean(d_share_raw,dim=1)
        c_share = self.ln_shr(c_share)
        d_share = self.ln_shr(d_share)
        modal_shr = torch.cat([c_share, d_share], -1)  # 16 * 80

        return modal_shr
    
    ## cinput_ids, cattention_masks, dinput_ids, dattention_masks,cimg,dimg, kg1, kg2, kg_sim
    def forward(self, cinput_ids, cattention_masks, dinput_ids, dattention_masks,cimg,dimg,\
                 vec_all_1, vec_all_2, kg_Sim):
        device = cinput_ids.device
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ctxt = self.txtenc(cinput_ids, cattention_masks,device)   ### B * L * 40
        cimg = self.imgenc(cimg)                    ### B *  40 * 7 * 7
        dtxt = self.txtenc(dinput_ids, dattention_masks,device)   ### B * L * 40
        dimg = self.imgenc(dimg)                    ### B *  40 * 7 * 7

        ttmodal_shr = self.stance_transformer(ctxt, dtxt)
        vvmodal_shr = self.stance_transformer(cimg, dimg)
        tvmodal_shr = self.stance_transformer(ctxt, dimg)

        # modal_shr = torch.cat([txt_share, img_share], -1)  # 16 * 80

        ## vec_all_1 = {} tt tv vv _top _botm    vec_all_2
        for i_emb in ['tt','tv','vv']:

            if i_emb == 'tt':
                modal_shr = ttmodal_shr    ### 16 * 120   or  16 * 40 ?
            elif i_emb == 'tv':               
                modal_shr = tvmodal_shr    ### 16 * 120
            elif i_emb == 'vv':
                modal_shr = vvmodal_shr    ### 16 * 120
            
            kg1 = vec_all_1[i_emb+ '_top'].to(device, dtype=torch.float)
            kg2 = vec_all_2[i_emb+ '_top'].to(device, dtype=torch.float)
            kg_sim =  kg_Sim[i_emb + '_top'].to(device, dtype=torch.float)

            kg_neg = self.decoder_neg(kg1,kg2,kg_sim,modal_shr)
                        
            kg1 = vec_all_1[i_emb+ '_botm'].to(device, dtype=torch.float)
            kg2 = vec_all_2[i_emb+ '_botm'].to(device, dtype=torch.float)
            kg_sim = kg_Sim[i_emb + '_botm'].to(device, dtype=torch.float)

            kg_pos = self.decoder_pos(kg1,kg2,kg_sim,modal_shr)  



            cat_context = torch.cat([kg_pos, kg_neg], -1)
            kg_context = F.leaky_relu(self.ln_kg2(cat_context))  # 16 * 120   or  16 * 40?

            if i_emb == 'tt':
                kg_context_tt = kg_context    ### 16 * 120   or  16 * 40 ?
            elif i_emb == 'tv':               
                kg_context_tv = kg_context    ### 16 * 120
            elif i_emb == 'vv':
                kg_context_vv = kg_context    ### 16 * 120

        # breakpoint()
        kg_context_all = torch.cat([kg_context_tt,kg_context_tv,kg_context_vv],-1)


        
        # print('model:',post_share_context.size(),post_uniq_context.size(),kg_context.size())
        cat = torch.cat([ttmodal_shr, vvmodal_shr, tvmodal_shr, kg_context_all], -1)
        # cat = torch.cat([post_share_context, post_uniq_context], -1)
        # print('cat size:', cat.size())
        output_class = self.rumor_classifier(cat)

        return output_class


    def decoder_pos(self,kg1,kg2,kg_sim,modal_shr):
        
        kg1 = F.leaky_relu(self.ln_kg1(kg1))
        kg2 = F.leaky_relu(self.ln_kg1(kg2))
        # kg2 = torch.dropout(kg2, self.dropout, train=self.training)
        # kg1 = self.ln_kg1(kg1)
        cat_kg = torch.cat([kg1, kg2], -1)  # 80
        # print('modal_shr size:',modal_shr.size())
        # print('cat_kg size:',cat_kg.size())
        # Q: modal_shr K:cat_kg V:cat_kg
        # ----------------------------------------------
        batch_size = cat_kg.size(0)
        Q_pos = self.fc_Q_pos(modal_shr)
        K_pos = self.fc_K_pos(cat_kg)    ### 16 * 80
        V_pos = self.fc_V_pos(cat_kg)
        Q_pos = Q_pos.view(batch_size * self.num_head, -1, self.dim_head) ### 16 * 1 * 80
        K_pos = K_pos.view(batch_size * self.num_head, -1, self.dim_head) ### 16 * 1 * 80
        V_pos = V_pos.view(batch_size * self.num_head, -1, self.dim_head) ### 16 * 1 * 80

        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K_pos.size(-1) ** -0.5  # 缩放因子
        kg_context_pos = self.attention_pos(Q_pos, K_pos, V_pos, scale, kg_sim)
        kg_context_pos = kg_context_pos.view(batch_size, -1, self.dim_head * self.num_head)
        kg_context_pos = self.fc1(kg_context_pos)
        kg_context_pos = self.dropout(kg_context_pos)
        # kg_context_pos = torch.dropout(kg_context_pos, self.dropout, train=self.training)
        # out = out + x  # 残差连接
        kg_context_pos = self.layer_norm(kg_context_pos)
        kg_context_pos = kg_context_pos.squeeze(1)

        return kg_context_pos

    def decoder_neg(self,kg1,kg2,kg_sim,modal_shr):


        kg1 = F.leaky_relu(self.ln_kg1(kg1))
        kg2 = F.leaky_relu(self.ln_kg1(kg2))
        # kg2 = torch.dropout(kg2, self.dropout, train=self.training)
        # kg1 = self.ln_kg1(kg1)
        cat_kg = torch.cat([kg1, kg2], -1)  # 80
        # print('modal_shr size:',modal_shr.size())
        # print('cat_kg size:',cat_kg.size())
        # Q: modal_shr K:cat_kg V:cat_kg
        # ----------------------------------------------
        batch_size = cat_kg.size(0)

        Q_neg = self.fc_Q_neg(modal_shr)
        K_neg = self.fc_K_neg(cat_kg)
        V_neg = self.fc_V_neg(cat_kg)
        Q_neg = Q_neg.view(batch_size * self.num_head, -1, self.dim_head)
        K_neg = K_neg.view(batch_size * self.num_head, -1, self.dim_head)
        V_neg = V_neg.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K_neg.size(-1) ** -0.5  # 缩放因子  

        kg_context_neg = self.attention_neg(Q_neg, K_neg, V_neg, scale, kg_sim)
        kg_context_neg = kg_context_neg.view(batch_size, -1, self.dim_head * self.num_head)
        kg_context_neg = self.fc2(kg_context_neg)
        kg_context_neg = self.dropout(kg_context_neg)
        # kg_cintext_neg = torch.dropout(kg_context_neg, self.dropout, train=self.training)
        # out = out + x  # 残差连接
        kg_context_neg = self.layer_norm(kg_context_neg)
        kg_context_neg = kg_context_neg.squeeze(1)     ### 16 * 1 * 80
        # ------------------------------------------------------------------
        # print('kg_context size:',kg_context.size())
        return kg_context_neg
