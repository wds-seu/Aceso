#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
import torch.nn as nn
import torch
import torch.nn.functional as F
from .BasicModule import BasicModule
from config import DefaultConfig
import logging
opt = DefaultConfig()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                    filename=opt.log_location,
                    filemode='a+')


class CNNNet(BasicModule):
    def __init__(self, pretrained_embeddings=None, vocab_size=None, embedding_dim=None):
        super(CNNNet, self).__init__()
        self.model_name = 'CNNNet'
        V = vocab_size
        D = opt.embedding_dim
        C = opt.class_num
        Ci = 1
        Co = opt.kernel_num
        Ks = opt.kernel_sizes
        self.embed = nn.Embedding(V, D)
        self.embed.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.embed.weight.requires_grad = opt.mode=="nonstatic"
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(opt.dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.avg_pool1d(x, x.size(2)).squeeze(2)
        # x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):  # x: (batch, sentence_len)
        x = self.embed(x)  # (N, W, D)
        # x = self.dropout(x)
        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)
        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit
