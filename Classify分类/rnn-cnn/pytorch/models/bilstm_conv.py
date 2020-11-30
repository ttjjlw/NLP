# -*- coding: utf-8 -*-

"""
@Author  : captain
@time    : 18-7-10 下午8:44
@ide     : PyCharm  
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule


class bilstm_conv(BasicModule):
    def __init__(self, args, vectors=None):
        self.args = args
        super(bilstm_conv, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.batch_size = args.batch_size
        # self.use_gpu = args.use_gpu
        self.lstm_layers = args.lstm_layers

        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)
        if vectors is not None:
            vectors=torch.Tensor(vectors)
            self.embedding.weight.data.copy_(vectors)

        self.bidirectional = True
        if self.lstm_layers > 1:
            self.dropout = args.lstm_dropout
            self.bilstm = nn.LSTM(args.embedding_dim, self.hidden_dim // 2, num_layers=self.lstm_layers,
                                  dropout=self.dropout, bidirectional=True)
        else:
            self.bilstm = nn.LSTM(args.embedding_dim, self.hidden_dim // 2, num_layers=self.lstm_layers,
                                  bidirectional=True)
        self.conv1 = nn.Conv1d(self.hidden_dim, 64, kernel_size=3) #filter_nums=64
        self.fc = nn.Linear(64 * 2, args.label_size)

    def forward(self, sentence):
        embed = self.embedding(sentence)  # #(batch,seq,hid_dim)

        lstm_out, _ = self.bilstm(embed)  # #(batch,seq,hid_dim)
        x = F.relu(self.conv1(lstm_out.permute(0, 2, 1)))
        avg_pool = torch.mean(x, dim=2) #(batch,filter_nums)
        max_pool, _ = torch.max(x, dim=2)
        feat = torch.cat((avg_pool, max_pool), dim=1)
        y = self.fc(feat)
        return y
