# D:\localE\python
# -*-coding:utf-8-*-
# Author ycx
# Date: 2020/11/2
import pandas as pd
import numpy as np

tq=pd.read_csv('./data/train/train.query.tsv',sep='\t',engine='python',encoding='utf-8',header=None)
tr=pd.read_csv('./data/train/train.reply.tsv',sep='\t',engine='python',encoding='utf-8',header=None)
tr=tr.dropna()
tq.columns=['idx','query']
tr.columns=['idx','seq_id','reply','label']

tq_valid=tq[tq.idx<1000]
tr_valid=tr[tr.idx<1000]

tq_train=tq[tq.idx>=1000]
tr_train=tr[tr.idx>=1000]

train=pd.merge(tq_train,tr_train,on=['idx'],how='inner') #内联，有query无回复，或有回复无query的都过滤掉
valid=pd.merge(tq_valid,tr_valid,on=['idx'],how='inner') #内联，有query无回复，或有回复无query的都过滤掉

train_all=pd.merge(tq,tr,on=['idx'],how='inner')

train_p=train[train.label==1]
train_n=train[train.label==0]
print('train正样本数：%d'%int(train_p['idx'].count()))
print('train负样本数：%d'%int(train_n['idx'].count()))

valid_p=valid[valid.label==1]
valid_n=valid[valid.label==0]
print('valid正样本数：%d'%int(valid_p['idx'].count()))
print('valid负样本数：%d'%int(valid_n['idx'].count()))

print('训练集空行：', train[train.isnull().T.any()])
print('验证集空行：', valid[valid.isnull().T.any()])
print('全训练集空行：', train_all[train_all.isnull().T.any()])


train.to_csv('./data/train/train.csv', sep='\t', header=True, index=False)
train_all.to_csv('./data/train/train_all.csv', sep='\t', header=True, index=False)
valid.to_csv('./data/train/valid.csv', sep='\t', header=True, index=False)
