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

class bigru_attention(BasicModule):
    def __init__(self, args, vectors=None):
        self.args = args
        super(bigru_attention, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.gru_layers = args.lstm_layers

        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)
        if vectors is not None:
            vectors = torch.Tensor(vectors)
            self.embedding.weight.data.copy_(vectors)
        
        self.bigru = nn.GRU(args.embedding_dim, self.hidden_dim // 2, num_layers=self.gru_layers, bidirectional=True)
        self.weight_W = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.weight_proj = nn.Parameter(torch.Tensor(self.hidden_dim, 1))
        self.fc = nn.Linear(self.hidden_dim, args.label_size)

        nn.init.uniform_(self.weight_W, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)

    def forward(self, sentence):
        embeds = self.embedding(sentence) # [bs, seq_len, emb_dim]
        gru_out, _ = self.bigru(embeds) # [bs, seq_len, hid_dim]
        # x = gru_out.permute(1, 2, 0) #[seq_len,hid_dim,bs]
        u = torch.tanh(torch.matmul(gru_out, self.weight_W)) #matmul 变成了mm
        att = torch.matmul(u, self.weight_proj)
        att_score = F.softmax(att, dim=1)
        scored_x = gru_out * att_score
        feat = torch.sum(scored_x, dim=1)
        y = self.fc(feat)
        return y
