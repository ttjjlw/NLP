# F:\itsoftware\Anaconda
# -*- coding:utf-8 -*-
# Author = TJL
# date:2020/11/16
import numpy as np
import pandas as pd
import os,pickle,re,jieba,collections
import torch
import torch.utils.data as Data
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
    corpus=[]
    for line in col:
        words=jieba.lcut(line)
        corpus.extend(words)
    counter=dict(collections.Counter(corpus).most_common(100000))
    word2id={}
    for i,w in enumerate(counter):
        word2id[w]=i+1
    word2id['<pad>']=0
    word2id['<unk>']=len(word2id)
    # print(word2id['<unk>'])
    # print(max(list(word2id.values())))
    return word2id

def text2id(text,vocab):
    '''
    把一段文本转化成number
    :param text: string 一段文本
    :param vocab: dict 语料字典
    :return: list number list
    '''
    word=jieba.lcut(text)
    id=[vocab.get(w,len(vocab)-1) for w in word] #注意是len(vocab)-1
    return id
def padding(lis,max_len):
    '''
    每条样本padding到指定长度
    :param lis: list number list
    :param max_len: int
    :return: list
    '''
    if len(lis)<max_len:
        lis.extend([0] * (max_len - len(lis)))
        
    else:
        lis=lis[:max_len]
    assert len(lis) == max_len
    return lis
def text2id_padding(line,max_len,vocab):
    '''
    encode2id and padding
    :param line: string 一行样本
    :param max_len: int
    :return: list
    '''
    line=delete_tag(line)
    line_id=text2id(line,vocab)
    line_id_pading=padding(line_id,max_len)
    return line_id_pading

def batch_yield(data, batch_size, vocab, shuffle=True, max_len=32):
    '''
    生成一批训练数据
    :param data: list
    :param batch_size:
    :param vocab: 语料库字典
    :param shuffle:
    :param max_len:
    :return:
    '''
    if shuffle:
        np.random.shuffle(data)
    input_ids,input_labels=[],[]
    for label, text in data:
        assert type(text)==str
        input_id = text2id_padding(text,max_len=max_len,vocab=vocab)
        input_ids.append(input_id)
        input_labels.append(int(label))
        if len(input_ids) == batch_size:
            assert (len(input_ids) == len(input_labels))
            yield input_ids,input_labels
            input_ids,input_labels=[],[]
    if len(input_ids) != 0:
        yield input_ids,input_labels
def get_data(data,  vocab, shuffle=True, max_len=32):
    '''
    把所有数据全部进行to id 然后 pandding
    :param data:
    :param batch_size:
    :param vocab:
    :param shuffle:
    :param max_len:
    :return:
    '''
    if shuffle:
        np.random.shuffle(data)
    input_ids,input_labels=[],[]
    for label, text in data:
        assert int(label) in [0,1]
        input_id = text2id_padding(text, max_len=max_len, vocab=vocab)
        input_ids.append(input_id)
        input_labels.append(label)
    assert len(input_labels)==len(input_ids)
    assert len(input_ids[0])==max_len
    assert  len(input_ids)==len(data)
    return input_ids,input_labels
def get_iter(data,vocab,batch_size,shuffle,max_len=32):
    input_ids,input_labels=get_data(data, vocab, shuffle=shuffle, max_len=max_len)
    input_ids = torch.from_numpy(np.array(input_ids)).long()
    train_label = torch.from_numpy(np.array(input_labels)).long()
    train_torch_data = Data.TensorDataset(input_ids, train_label)
    train_loader = Data.DataLoader(dataset=train_torch_data, batch_size=batch_size)
    return train_loader
    
if __name__ == '__main__':
    train = pd.read_csv('../data/train/torchClassify/train.csv', sep='\t', encoding='utf-8', header=0)
    test = pd.read_csv('../data/test/torchClassify/test.csv', sep='\t', encoding='utf-8', header=0)
    corpus_all=pd.concat([train,test],axis=0)
    print(corpus_all.count())
    print(train.count())
    print(test.count())
    vocab=get_dictionary(corpus_all.text)
    train = list(zip(train.label, train.text))
    test = list(zip(test.label, test.text))
    
    for input_id, label in batch_yield(train, batch_size=64,vocab=vocab,shuffle=False):
        input_id = np.array(input_id)
        label = np.array(label)
        print(input_id.shape)
        print(label.shape)
        print('=' * 10)
        break
    for input_id, label in batch_yield(test, batch_size=64, vocab=vocab, shuffle=False):
        input_id = np.array(input_id)
        label = np.array(label)
        print(input_id.shape)
        print(label.shape)
        print('=' * 10)
        break
    train, label=get_data(train,  vocab, shuffle=False, max_len=32)
    train=np.array(train)
    label=np.array(label)
    print(train.shape)
    print(label.shape)
    print('=' * 10)
    test, vlabel=get_data(test,  vocab, shuffle=False, max_len=32)
    test = np.array(test)
    vlabel = np.array(vlabel)
    print(train.shape)
    print(vlabel.shape)