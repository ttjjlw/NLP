# F:\itsoftware\Anaconda
# -*- coding:utf-8 -*-
# Author = TJL
# date:2020/11/23
import os,pickle,json
import numpy as np
import argparse,random
from model import Word2vec
import pandas as pd
from dataset import get_dictionary
from tools import drawing_and_save_picture
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 只显示 Error
parser = argparse.ArgumentParser(description='skip model generate word2vec')
parser.add_argument('--mode', type=str, default='train',help='the mode of model,input train or predict')
parser.add_argument('--vocab', type=dict, default={},help='the vocab of corpus')
parser.add_argument('--embed_dim', type=int, default=128,help='the dim of word2vec')
parser.add_argument('--init_rate', type=float, default=0.001,help='the init learn rate')
parser.add_argument('--neg_samples', type=int, default=5,help='#Mikolov等人在论文中说：对于小数据集，负采样的个数在5-20个；对于大数据集，负采样的个数在2-5个。')
parser.add_argument('--epochs', type=int, default=10,help='the train epochs')
parser.add_argument('--log_per_steps', type=int, default=100,help='每隔多少步打印一次信息')
parser.add_argument('--decay_steps', type=int, default=1000,help='learn rate decay steps')
parser.add_argument('--is_load', type=int, default=0,help='是否加载模型再训练')
parser.add_argument('--save_path', type=str, default='export/model/',help='the save path of word2vec model')
parser.add_argument('--embeddings_save_path', type=str, default='export/embed/',help='the save path of word2vec')
parser.add_argument('--picture_path', type=str, default='export/pic.png',help='the save path of vocab')
parser.add_argument('--is_save_vector', type=bool, default=False,help='预测时是否同时保存词向量')
parser.add_argument('--batch_size', type=int, default=64,help='batch_size of every train')
parser.add_argument('--window_size', type=int, default=1,help='window_size of center word')
parser.add_argument('--valid_size', type=int, default=20,help='训练时取多少个词作验证')
parser.add_argument('--valid_window', type=int, default=80,help='验证单词只从频率最高的valid_window个单词中选出')
parser.add_argument('--top_k', type=int, default=5,help='输出和验证词最相似的前top_k个词')
args = parser.parse_args()

if not os.path.exists(args.save_path):os.makedirs(args.save_path)
if not os.path.exists(args.embeddings_save_path):os.makedirs(args.embeddings_save_path)
if not os.path.exists(args.picture_path):os.makedirs(args.picture_path)

train = pd.read_csv('./data/train.csv', sep='\t', encoding='utf-8', header=0)
#valid_window 必须小于vocab的长度
#从1~valid_window范围中随机选valid_size个词用作验证
args.valid_example=np.random.choice(range(0,args.valid_window+1),args.valid_size,replace=False)#repalce=False表示不重复，不重复从0-119选出20个 type array
print(args)
args.vocab = get_dictionary(train.text)
with open(args.embeddings_save_path+'vocab.json','w',encoding='utf-8') as f:
    json.dump(args.vocab,f,ensure_ascii=False)
args.raw_data=train.text
print('vocab_size:%d'%len(args.vocab))


args.mode='train'
args.is_save_vector=True
model=Word2vec(args)
model.build_graph()
if args.mode=='train':
    model.add_loss()
    model.train()
    drawing_and_save_picture(args)
if args.mode=='predict':model.predict()