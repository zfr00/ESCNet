import torch
import torch.nn as nn
import torchvision
from transformers import BertModel, BertTokenizer, BertConfig, BertForSequenceClassification
from torchvision import transforms
import os
import numpy as np
import transformers
# from zmq import device
from einops import rearrange, repeat
from torch import nn, einsum
from torch.nn import init
from torchvision import models


class TextExtract_Bert_lstm(nn.Module):
    def __init__(self):
        super(TextExtract_Bert_lstm, self).__init__()
        
        # self.model_txt = Vit_text(768) ### different backbone
        # self.last_lstm = args.last_lstm
        bert_path = '/your_path/bert-base-chinese' ### 
        model_class, tokenizer_class, pretrained_weights = (transformers.BertModel, transformers.BertTokenizer, bert_path)
        self.text_embed = model_class.from_pretrained(pretrained_weights)
        # self.text_embed.eval()
        # self.text_embed.eval()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for p in self.text_embed.parameters():
            p.requires_grad = False
        # self.dropout = nn.Dropout(0.3)
        self.dropout = nn.Dropout(0.0)
        # self.lstm = nn.LSTM(768, 20, num_layers=1, bias=True, batch_first=True,
        #                     bidirectional=True)
        # self.atten = SelfAttention(40,2,0,self.device)

    def forward(self, txt, mask,device):
        # length = mask.sum(1)      ### 16*序列长度
        # # print(length.size())    16
        # length = length.cpu()    #每个batch按照最长的序列进行padding处理等长的形式
        # with torch.no_grad():
        txt = self.text_embed(txt, attention_mask=mask)#
        txt = txt[0]   ##16 * 128 * 768
            # print(txt.shape)
            # print(txt.size())
            # txt = txt.unsqueeze(1)
            # txt = txt.permute(0, 3, 1, 2) ##64 * 768 * 1 * 64
        # txt = self.model_txt(txt , trans_mask)  # txt4: batch x 2048 x 1 x 64

        # txt,_ = self.lstm(txt)
        # print(txt.shape)
        # b = txt[:,-1,:40]
        # # print(b.shape)
        # c = txt[:,1,40:]
        # # print(c.shape)
        # txt= self.atten(b,c,c)
        # txt = bi_hidden.reshape([bi_hidden.shape[0], -1])
        # print(txt.size())
        return txt  # 16 * 128 * 768




def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        # print('1111111111')
        init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        # init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.constant(m.weight.data, 1)
        init.constant(m.bias.data, 0)


class conv(nn.Module):

    def __init__(self, input_dim, output_dim, relu=False, BN=True):
        super(conv, self).__init__()

        block = []
        block += [nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)]

        if BN:
            block += [nn.BatchNorm2d(output_dim)]
        if relu:
            block += [nn.ReLU(inplace=True)]

        self.block = nn.Sequential(*block)
        self.block.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.block(x)
        # print(x.size())
        x = x.squeeze(3).squeeze(2)
        return x


class ImgPNet(nn.Module):

    def __init__(self):
        super(ImgPNet, self).__init__()

        # self.opt = opt
        resnet50 = models.resnet50(pretrained=True)
        # vit =    #different backbone

        self.ImageExtract = nn.Sequential(*(list(resnet50.children())[:-2]))
        # self.TextExtract = TextExtract(opt)

        self.avg_global = nn.AdaptiveMaxPool2d((1, 1))
        # self.avg_global = nn.AdaptiveAvgPool2d((1, 1))

        self.conv_1X1 = conv(2048, 768)

    def forward(self, image):

        image_feature = self.img_embedding(image)

        # print(text_feature.shape)
        image_feature = rearrange(image_feature,'b n h d -> b (h d) n')
        return image_feature   # 16 * 49 * 768

    def img_embedding(self, image):
        image_feature = self.ImageExtract(image)
        # print(image_feature.size())

        # image_feature = self.avg_global(image_feature)
        # print(image_feature.size())
        image_feature = self.conv_1X1(image_feature)
        # print(image_feature.size())

        return image_feature



class mydecoder(nn.Module):  #384 ,  1 ,    6 ,   768                          64
    def __init__(self,        dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        # self.to_DIM_txt= nn.Linear(patch_dim, dim)
        # self.to_DIM_img = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)  # 0
                                                #384   1       6     64        768
        self.transformer = Transformer_mydecoder(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool

    def forward(self, txt,img, kv_mask = None,q_mask = None): # img [64,48,2048] txt [64,Length,2048]
        # img = self.to_DIM_img(img)
        # txt = self.to_DIM_txt(txt)
        # print(txt.shape)
        # print(type(txt))
        b_img, n_img, _ = img.shape
        b_txt, n_txt, _ = txt.shape
        # x += self.pos_embedding[:, :(n + 1)]
        # img = self.dropout(img)
        x = self.transformer(txt, img, kv_mask,q_mask)
        # x = self.to_latent(x)
        # return self.mlp_head(x)
        return x

class Transformer_mydecoder(nn.Module):
                    #  384   1      6       64       768
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual_3(PreNorm_3(dim, Attention_mydecoder(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                # PreNorm_3(dim, Attention_DECODER(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, y,kv_mask = None,q_mask = None):
        # for attn, attn_decode, ff in self.layers:
        #     # x = attn(x, mask = q_mask)
        #     x = attn_decode(x, y , kv_mask=kv_mask,q_mask = q_mask)
        #     x = ff(x)
        for attn_decode, ff in self.layers:
            # x = attn(x, mask = q_mask)
            x = attn_decode(x, y , kv_mask=kv_mask,q_mask = q_mask)
            x = ff(x)
        return x

class PreNorm_3(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, y, **kwargs):
        return self.fn(self.norm(x), self.norm(y),**kwargs)

class Residual_3(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, y,**kwargs):
        return self.fn(x, y, **kwargs) + x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
                    #  40   768
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            # nn.GELU(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention_mydecoder(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim*2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self,txt, image ,kv_mask = None,q_mask = None):
        b1, n1, _, h = *image.shape, self.heads
        b2, n2, _, h = *txt.shape, self.heads

        q = self.to_q(txt)
        q = rearrange(q,'b n (h d) -> b h n d', h=h)
        kv = self.to_kv(image).chunk(2, dim = -1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), kv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        # mask_value = -torch.finfo(dots.dtype).max
        # if kv_mask is not None:
        #     assert kv_mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
        #     mask = rearrange(q_mask, 'b i -> b () i ()') * rearrange(kv_mask, 'b j -> b () () j')
        #     dots.masked_fill_(~mask, mask_value)
        #     del mask
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out




