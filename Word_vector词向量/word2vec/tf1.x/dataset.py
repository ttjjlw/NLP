# F:\itsoftware\Anaconda
# -*- coding:utf-8 -*-
# Author = TJL
# date:2020/11/16
import numpy as np
import pandas as pd
import os, pickle, re, jieba, collections

def delete_tag(s):
    '''
    对一段文本进行过滤
    :param s:string 文本
    :return: string 文本
    '''
    s = re.sub('\{IMG:.?.?.?\}', '', s)  # 图片
    s = re.sub(re.compile(r'[a-zA-Z]+://[^\u4e00-\u9fa5]+'), '', s)  # 网址
    s = re.sub(re.compile('<.*?>'), '', s)  # 网页标签
    s = re.sub(re.compile('&[a-zA-Z]+;?'), ' ', s)  # 网页标签
    # s = re.sub(re.compile('[a-zA-Z0-9]*[./]+[a-zA-Z0-9./]+[a-zA-Z0-9./]*'), ' ', s)
    # s = re.sub("\?{2,}", "", s)
    # s = re.sub("\r", "", s)
    # s = re.sub("\n", ",", s)
    s = re.sub("\t", "", s)
    s = re.sub("（", "", s)
    s = re.sub("）", "", s)
    s = re.sub("\u3000", "", s)  # 全角空格(中文符号)
    s = re.sub(" ", "", s)
    r4 = re.compile('\d{4}[-/]\d{2}[-/]\d{2}')  # 日期
    s = re.sub(r4, '某时', s)
    s = re.sub('“', '"', s)
    s = re.sub('”', '"', s)
    return s


def get_dictionary(col):
    '''
    获取包含语料中所有词的字典
    :param corpus: sring 语料
    :return:dict 字典
    '''
    corpus = []
    for line in col:
        words = jieba.lcut(line)
        corpus.extend(words)
    counter = dict(collections.Counter(corpus).most_common(100000))
    word2id = {}
    for i, w in enumerate(counter):
        word2id[w] = i + 1
    word2id['<pad>'] = 0
    word2id['<unk>'] = len(word2id)
    # print(word2id['<unk>'])
    # print(max(list(word2id.values())))
    return word2id


def text2id(text, vocab):
    '''
    把一段文本转化成number
    :param text: string 一段文本
    :param vocab: dict 语料字典
    :return: list number list
    '''
    word = jieba.lcut(text)
    id = [vocab.get(w, len(vocab) - 1) for w in word]  # 注意是len(vocab)-1
    return id

def get_data(data, vocab):
    '''
    把一行行的文本全部进行to id
    :param data:一行行的文本
    :param batch_size:
    :param vocab:
    :param shuffle:
    :param max_len:
    :return:
    '''
    input_ids = []
    for text in data:
        input_id = text2id(text, vocab=vocab)
        input_ids.extend(input_id)
    return input_ids

data_index=0
def build_batch(raw_data,vocab,batch_size,window_size=1):
    '''
    假设raw_data经id化后变为[1,2,3],最后经过该函数后train_batch=[2,2],train_label=[1,3]
    :param raw_data: string 原始一行行的文本
    :param vocab: dict 语料字典
    :param batch_size: 是skip_window*2的倍数
    :param window_size: 窗口大小
    :return:
    '''
    global data_index           #data_index=0，不能放在里面，因为每次生成batch，data_index都要继承前面的
    num_skip=window_size*2
    assert batch_size % num_skip == 0
    data=get_data(raw_data,vocab)
    train_batch=np.ndarray(shape=(batch_size),dtype=np.int32)
    train_label=np.ndarray(shape=(batch_size,1),dtype=np.int32)
    span=2*window_size+1 #入队长度
    deque=collections.deque(maxlen=span) #创建双向队列，如deque=[1,2,3],deque.append(4),则deque=[2,3,4]
    #初始化deque,把data前三个元素，放入deque中
    for _ in range(span):
        deque.append(data[data_index])#修改于2019-12-28
        data_index+=1
    for i in range(batch_size//num_skip):
        for j in range(span):
            if j>window_size:
                train_batch[num_skip*i+j-1]=deque[window_size]  ##为什么是num_skip*i，num_skip表示每次i循环，train_batch,添加了几个元素，所以需要向前偏移num_skip
                train_label[num_skip*i+j-1,0]=deque[j] #中心词右侧
            elif j==window_size:
                continue
            else:
                train_batch[num_skip*i+j]=deque[window_size] #train_batch=中心词
                train_label[num_skip*i+j,0]=deque[j] #中心词左侧
        deque.append(data[data_index])
        data_index+=1
        data_index%=len(data) #防止最后一个batch时，data_index溢出
    return train_batch, train_label


if __name__ == '__main__':
    train = pd.read_csv('./data/train.csv', sep='\t', encoding='utf-8', header=0)
    print(train.count())
    vocab = get_dictionary(train.text)
    input_id, label=build_batch(raw_data=['采荷一小是分校吧','房本都是五年外的'],vocab=vocab,batch_size=4)
    print(input_id)
    print(label)