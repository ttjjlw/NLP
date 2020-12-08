# F:\itsoftware\Anaconda
# -*- coding:utf-8 -*-
# Author = TJL
# date:2020/11/27

#
# print(model.predict("Which baking dish is best to bake a banana bread ?", k=3)) #输出最可能前k个label
# print(model.words)  #所有的词
# print(model.labels) #所有的标签
# print(model.get_word_vector)

# input             # training file path (required)
# lr                # learning rate [0.1]
# dim               # size of word vectors [100]
# ws                # size of the context window [5]
# epoch             # number of epochs [5]
# minCount          # minimal number of word occurences [1]
# minCountLabel     # minimal number of label occurences [1]
# minn              # min length of char ngram [0]
# maxn              # max length of char ngram [0]
# neg               # number of negatives sampled [5]
# wordNgrams        # max length of word ngram [1]
# loss              # loss function {ns, hs, softmax, ova} [softmax]
# bucket            # number of buckets [2000000]
# thread            # number of threads [number of cpus]
# lrUpdateRate      # change the rate of updates for the learning rate [100]
# t                 # sampling threshold [0.0001]
# label             # label prefix ['__label__']
# verbose           # verbose [2]
# pretrainedVectors # pretrained word vectors (.vec file) for supervised learning []
import fasttext,os,json,pickle
import numpy as np
train_path='./data/train.txt'
valid_path='./data/valid.txt'
embed_path_txt='export/Vector.txt'
embed_path_pkl='export/Vector.pkl'
vocab_path='export/vocab.json'
embeds_dim=100
if not os.path.exists(os.path.dirname(embed_path_txt)):os.makedirs(os.path.dirname(embed_path_txt))
if not os.path.exists(os.path.dirname(embed_path_pkl)):os.makedirs(os.path.dirname(embed_path_pkl))
model = fasttext.train_supervised(train_path,epoch=25,lr=1.0,wordNgrams=2,dim=embeds_dim)

print(model.test(valid_path,k=1)) #返回值(测试样本数， precision at k,recall at k)

txt=open(embed_path_txt,'w')
vocab={'<pad>':0}
embeds=[[0]*embeds_dim]
for idx,w in enumerate(model.words):
    vocab[w]=idx+1
    v=model.get_word_vector(w)
    embeds.append(v)
    line=w+' '+' '.join([str(v) for v in model.get_word_vector(w)])
    txt.write(line+'\n')
txt.close()
vector=np.array(embeds)
print('embed_pkl的shape为：{}'.format(vector.shape))
print('embed_txt的shape为：{}*{}'.format(idx+1,len(v)))
print('embed_pkl的index为0的位置加了<pad>的向量，所以比embed_txt多1')
print('vocab的长度: %d'%len(vocab))
with open(embed_path_pkl, 'wb') as p:
    pickle.dump(vector, p)
with open(vocab_path, 'w', encoding='utf-8') as p1:
    json.dump(vocab, p1,ensure_ascii=False)
print('完成！')