# F:\itsoftware\Anaconda
# -*- coding:utf-8 -*-
# Author = TJL
# date:2020/3/13

import jieba.analyse
import jieba
import os

# 添加专有名词，增加分词力度
def add_specific_word(word_lis):
    for word in word_lis:
        jieba.suggest_freq(word, True)

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r').readlines()]
    return stopwords


def get_train_corpus(raw_data_path, train_data_path,stop_word_path=None,specific_word=None):
    if specific_word==None: add_specific_word(word_lis=[])
    if stop_word_path==None:
        stopwords=[]
    else:
        stopwords = stopwordslist(stop_word_path)
    data_file_list = os.listdir(raw_data_path)
    corpus = []
    temp = 0
    for file in data_file_list:
        with open(raw_data_path + file, 'rb') as f:
            print('处理第{}个原始语料文件'.format(temp + 1))
            temp += 1
            lines = f.readlines()
            for line in lines:
                line=line.decode('utf-8').strip().replace(' ','')
                document_cut = jieba.cut(line, cut_all=False)
                document_cut=[word for word in document_cut if word not in stopwords and len(word.strip())>0]
            # print('/'.join(document_cut))
                result = ' '.join(document_cut)
                corpus.append(result)
        #  print(result)
    with open(train_data_path, 'w', encoding='utf-8') as f:
        for line in corpus:
            f.write(line+'\n')  # 读取的方式和写入的方式要一致




if __name__ == "__main__":
    raw_data_path = 'data/raw_data/'
    train_data_path = 'data/train_corpus/'
    stop_word_path = 'data/stop_words.txt'  # 这里只是样例，可以替换自己的
    get_train_corpus(raw_data_path, train_data_path,stop_word_path)
