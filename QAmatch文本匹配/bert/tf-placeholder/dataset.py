# D:\localE\python
# -*-coding:utf-8-*-
# Author ycx
# Date: 2020/11/2
# D:\localE\python
# -*-coding:utf-8-*-
# Author ycx
# Date: 2020/10/31
import pandas as pd
import numpy as np
import re,random,math
from TJl_function import get_logger


def delete_tag(s):
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

def random_drop(word_lis):
    if random.randint(1,10)>5:
        return word_lis
    if len(word_lis)<=3:
        return word_lis
    if len(word_lis)<=10:
        index=random.randint(0,len(word_lis)-1)
        word_lis.pop(index)
        return word_lis
    else:
        length=len(word_lis)
        num=math.ceil(length/10)
        for i in range(num):
            index=random.randint(0,len(word_lis)-1)
            word_lis.pop(index)
        return word_lis
def word2id_pad(tokenize, text1, text2,max_len,shuffle):
    word_lis1 = tokenize.tokenize(text1)
    if shuffle:word_lis1=random_drop(word_lis1)
    word_lis2 = tokenize.tokenize(text2)
    if shuffle:word_lis2 = random_drop(word_lis2)
    if len(word_lis1) > max_len - 2:word_lis1 = word_lis1[:max_len - 2]
    if len(word_lis2) > max_len - 2:word_lis2 = word_lis2[:max_len - 2]
    word_lis = ['[CLS]'] + word_lis1 + ['[SEP]'] + ['[CLS]'] + word_lis2 + ['[SEP]']
    word_id = tokenize.convert_tokens_to_ids(word_lis)
    mask_id = [1] * len(word_id) #pad处为0
    type_id = [0] * (2+len(word_lis1)) +[1] * (2+len(word_lis2))#区分句子
    assert len(word_id)==len(type_id)
    if len(word_id) < 2*max_len:
        word_id.extend([0] * (2*max_len - len(word_id)))
        mask_id.extend([0] * (2*max_len - len(mask_id)))#这里不能用word_id了，因为已经pad了
        type_id.extend([0] * (2*max_len - len(type_id)))
    assert len(word_id) == 2*max_len
    return word_id, mask_id, type_id


def text2id(label, querys, reply, tokenize, max_len,shuffle):
    querys = delete_tag(querys)
    reply = delete_tag(reply)
    input_id, mask_id, type_id = word2id_pad(tokenize=tokenize, text1=querys,text2=reply, max_len=max_len,shuffle=shuffle)
    assert len(input_id)==len(mask_id)
    assert len(mask_id)==len(type_id)
    return input_id, mask_id, type_id, int(label)


def batch_yield(data, batch_size, tokenize, shuffle=True, max_len=32):
    """
    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        np.random.shuffle(data)
    input_ids, mask_ids, type_ids, labels,seq_ids = [], [], [],[],[]
    for label, querys, reply,seq_id in data:
        input_id, mask_id, type_id, label = text2id(label, querys,reply, tokenize, max_len,shuffle)
        input_ids.append(input_id)
        mask_ids.append(mask_id)
        type_ids.append(type_id)
        labels.append(int(label))
        seq_ids.append(str(seq_id))

        if len(input_ids) == batch_size:
            assert (len(input_ids) == len(labels))
            assert len(input_ids)==len(mask_ids)
            yield input_ids, mask_ids, type_ids, labels,seq_ids
            input_ids, mask_ids, type_ids, labels,seq_ids = [], [], [],[],[]
    if len(input_ids) != 0:
        yield input_ids, mask_ids, type_ids, labels,seq_ids

if  __name__ == '__main__':
    import os
    from bert_model import tokenization
    abs_path=os.path.abspath('.')
    # print(abs_path)
    file_path = os.path.dirname(abs_path)
    os.sys.path.append(file_path)
    train=pd.read_csv('../data/train/classify/train.csv',sep='\t',encoding='utf-8',header=0)
    valid=pd.read_csv('../data/train/classify/valid.csv',sep='\t',encoding='utf-8',header=0)
    test=pd.read_csv('../data/test/test.csv',sep='\t',encoding='utf-8',header=0)
    # q=train['query']
    # r=train.reply
    # e=
    train=list(zip(train.label,train['query'],train.reply,train.seq_id))
    valid=list(zip(valid.label,valid['query'],valid.reply,valid.seq_id))
    test=list(zip(test.label,test['query'],test.reply,test.seq_id))

    tokenize = tokenization.FullTokenizer('./bert_model/chinese_L-12_H-768_A-12/vocab.txt', do_lower_case=True)
    for input_ids, mask_ids, type_ids, labels,seq_ids in batch_yield(train,batch_size=64,tokenize=tokenize):
        input_ids=np.array(input_ids)
        labels=np.array(labels)
        seq_ids=np.array(seq_ids)
        print(input_ids.shape)
        print(labels.shape)
        print(seq_ids.shape)
        print('=' * 10)
        break
    # print(train.head())
    for input_ids, mask_ids, type_ids, labels,seq_ids in batch_yield(valid, batch_size=64, tokenize=tokenize):
        input_ids = np.array(input_ids)
        labels = np.array(labels)
        seq_ids = np.array(seq_ids)
        print(input_ids.shape)
        print(labels.shape)
        print(seq_ids.shape)
        print('=' * 10)
        break
    for input_ids, mask_ids, type_ids, labels,seq_ids in batch_yield(test, batch_size=64, tokenize=tokenize):
        input_ids = np.array(input_ids)
        labels = np.array(labels)
        seq_ids = np.array(seq_ids)
        print(input_ids.shape)
        print(labels.shape)
        print(seq_ids.shape)
        print('=' * 10)
        break