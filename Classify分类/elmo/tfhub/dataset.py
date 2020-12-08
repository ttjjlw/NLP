# F:\itsoftware\Anaconda
# -*- coding:utf-8 -*-
# Author = TJL
# date:2020/12/3
import os,re
import numpy as np

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
    s = re.sub(r4, 'time', s)
    s = re.sub('“', '"', s)
    s = re.sub('”', '"', s)
    return s

def batch_yield(data, batch_size, shuffle=True):
    '''
    生成一批训练数据
    :param data: list
    :param batch_size:
    :param shuffle:
    :return:
    '''
    if shuffle:
        np.random.shuffle(data)
    inputs,input_labels=[],[]
    for line in data:
        assert type(line)==str
        assert len(line.strip().split('<SEP>'))==2 , '输入的文本格式不对，没用<SEP>隔开'
        input_text,label=line.strip().split('<SEP>')
        input_text=delete_tag(input_text.strip())
        inputs.append(input_text)
        input_labels.append(int(label.strip()))
        if len(inputs) == batch_size:
            assert (len(inputs) == len(input_labels))
            yield inputs,input_labels
            inputs,input_labels=[],[]
    if len(inputs) != 0:
        assert (len(inputs) == len(input_labels))
        yield inputs,input_labels