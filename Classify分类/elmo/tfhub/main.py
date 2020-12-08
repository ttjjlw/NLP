# F:\itsoftware\Anaconda
# -*- coding:utf-8 -*-
# Author = TJL
# date:2020/12/3
from dataset import batch_yield
from config import Config
from model import Model
args=Config()


with open(args.train_data_path,'r',encoding='utf-8') as f:
    train=f.readlines()
with open(args.eval_data_path,'r',encoding='utf-8') as f:
    valid=f.readlines()
with open(args.test_data_path,'r',encoding='utf-8') as f:
    test=f.readlines()

model=Model(args)
model.build_train_graph()

model.train(args,train,valid,test)