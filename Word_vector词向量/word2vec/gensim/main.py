# F:\itsoftware\Anaconda
# -*- coding:utf-8 -*-
# Author = TJL
# date:2020/11/24
import os,pickle
import numpy as np
import argparse,random
from dataset import get_train_corpus
from Word2Vec import train,get_vocab_and_embed
from tools import drawing_and_save_picture


parser = argparse.ArgumentParser(description='generate word2vec by gensim')
parser.add_argument('--raw_data_path', type=str, default='./data/raw_data/',help='the dir of raw data file,in this dir can contain more than one files')
parser.add_argument('--train_data_path', type=str, default='data/train_corpus/corpus.txt',help='the path of train data file')
parser.add_argument('--stop_word_path', type=str, default='data/stop_words.txt',help='the path of stop words file')
parser.add_argument('--embed_path_txt', type=str, default="export/Vector.txt",help='the save path of word2vec with type txt')
parser.add_argument('--embed_path_pkl', type=str, default="export/Vector.pkl",help='the save path of word2vec with type pkl,which is array after pickle.load ')
parser.add_argument('--vocab_path', type=str, default='export/vocab.json',help='the save path of vocab')
parser.add_argument('--picture_path', type=str, default='export/pic.png',help='the save path of vocab')
parser.add_argument('--w_num', type=str, default=200,help='how many words will plot')
parser.add_argument('--embed_dim', type=int, default=128,help='the dim of word2vec')
parser.add_argument('--window_size', type=int, default=3,help='window size')
parser.add_argument('--negative_size', type=int, default=5,help='负采样个数')
parser.add_argument('--iter', type=int, default=5,help='训练次数')
parser.add_argument('--sg', type=int, default=0,help='0是CBOW,1是skipGram')
parser.add_argument('--hs', type=int, default=0,help='即我们的word2vec两个解法的选择了，如果是0， 则是Negative Sampling，是1的话并且负采样个数negative大于0， 则是Hierarchical Softmax。默认是0即Negative Sampling')
args = parser.parse_args()
print(args)

if not os.path.exists(os.path.dirname(args.embed_path_txt)):os.makedirs(os.path.dirname(args.embed_path_txt))
if not os.path.exists(os.path.dirname(args.embed_path_pkl)):os.makedirs(os.path.dirname(args.embed_path_pkl))
if not os.path.exists(os.path.dirname(args.vocab_path)):os.makedirs(os.path.dirname(args.vocab_path))
if not os.path.exists(os.path.dirname(args.picture_path)):os.makedirs(os.path.dirname(args.picture_path))

get_train_corpus(args.raw_data_path, args.train_data_path,args.stop_word_path)
train(args)
get_vocab_and_embed(args)
drawing_and_save_picture(args)