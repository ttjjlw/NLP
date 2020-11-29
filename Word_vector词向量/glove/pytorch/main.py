# F:\itsoftware\Anaconda
# -*- coding:utf-8 -*-
# Author = TJL
# date:2020/11/24
import os,pickle
import numpy as np
import argparse,random
from dataset import get_train_corpus
from glove import train,get_vocab_and_embed
from tools import  VectorEvaluation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate word vector by glove')
    parser.add_argument('--mode', type=str, default='train',help='train or test')
    parser.add_argument('--raw_data_path', type=str, default='./data/raw_data/',help='the dir of raw data file,in this dir can contain more than one files')
    parser.add_argument('--train_data_path', type=str, default='data/train_corpus/corpus.txt',help='the path of train data file')
    parser.add_argument('--stop_word_path', type=str, default='data/stop_words.txt',help='the path of stop words file')
    parser.add_argument('--embed_path_txt', type=str, default="export/Vector.txt",help='the save path of word vector with type txt')
    parser.add_argument('--embed_path_pkl', type=str, default="export/Vector.pkl",help='the save path of word vector with type pkl,which is array after pickle.load ')
    parser.add_argument('--pic_path', type=str, default="export/glove.png",help='the save path of word vector with type pkl,which is array after pickle.load ')
    parser.add_argument('--vocab_path', type=str, default='export/vocab.json',help='the save path of vocab')
    parser.add_argument('--embed_dim', type=int, default=128,help='the dim of word vector')
    parser.add_argument('--x_max', type=int, default=100,help='两个词共现出现的次数大于x_max后，衡量两词相似性的权重不再增加，论文推荐100')
    parser.add_argument('--alpha', type=float, default=0.75,help='两个词共现出现的次数x小于x_max时，衡量两词相似性的权重为(x/x_max)^alpha 论文推荐0.75')
    parser.add_argument('--epoches', type=int, default=3,help='训练回合')
    parser.add_argument('--min_count', type=int, default=0,help='过滤掉出现小于min_count的词')
    parser.add_argument('--batch_size', type=int, default=64,help='训练批次')
    parser.add_argument('--windows_size', type=int, default=5,help='窗口大小')
    parser.add_argument('--learning_rate', type=int, default=0.001,help='学习率')
    
    args = parser.parse_args()
    print(args)
    
    if not os.path.exists(os.path.dirname(args.train_data_path)):os.makedirs(os.path.dirname(args.train_data_path))
    if not os.path.exists(os.path.dirname(args.embed_path_txt)):os.makedirs(os.path.dirname(args.embed_path_txt))
    if not os.path.exists(os.path.dirname(args.embed_path_pkl)):os.makedirs(os.path.dirname(args.embed_path_pkl))
    if not os.path.exists(os.path.dirname(args.vocab_path)):os.makedirs(os.path.dirname(args.vocab_path))
    if args.mode=='train':
        get_train_corpus(args.raw_data_path, args.train_data_path,args.stop_word_path)
        train(args)
        get_vocab_and_embed(args)
        # vec_eval.drawing_and_save_picture(args.pic_path) #可视化
    else:
        vec_eval = VectorEvaluation(args.embed_path_txt)
        vec_eval.get_similar_words("加拿大")