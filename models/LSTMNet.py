#!/usr/bin/env python  
# -*- coding:utf-8 _*-
import torch
import torch.nn as nn
from torch.autograd import Variable
from .BasicModule import BasicModule
from config import DefaultConfig
opt = DefaultConfig()


class LSTMNet(BasicModule):
    def __init__(self, vocab_size=None, pretrained_embeddings=None):
        super(LSTMNet, self).__init__()
        self.model_name = 'LSTMNet'
        self.embedding_dim = opt.embedding_dim
        self.hidden_dim = opt.hidden_dim
        # 4550x108
        self.embed = nn.Embedding(vocab_size, self.embedding_dim)
        self.embed.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.embed.weight.requires_grad = opt.mode == "nonstatic"
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=2)
        self.linear1 = nn.Linear(self.hidden_dim, opt.class_num)
        self.drop_out = nn.Dropout(opt.dropout)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden=None):
        # input:[batch_size, seq_len]
        seq_len, batch_size = input.size()
        if hidden is None:
            # [2, batch_size, hidden_dim]
            h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            # 正交初始化
            h_0 = nn.init.orthogonal(h_0)
            c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = nn.init.orthogonal(c_0)
            h_0, c_0 = Variable(h_0), Variable(c_0)
        else:
            h_0, c_0 = hidden

        # size : (seq_len, batch_size, embedding_dim) 【95, 128, 108】
        embeds = self.embed(input)
        x = embeds.view(seq_len, batch_size, -1)
        # output size : (seq_len, batch_size, hidden_dim)
        output, hidden = self.lstm(x, (h_0, c_0))
        output = self.drop_out(output)
        # size : (seq_len*batch_size, vocab_size)
        output = self.linear1(output[-1])
        # size : (batch_size, class_num)
        # output = self.log_softmax(output)
        return output, hidden