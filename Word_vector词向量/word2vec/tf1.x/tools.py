# F:\itsoftware\Anaconda
# -*- coding:utf-8 -*-
# Author = TJL
# date:2020/12/7
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import pickle,json
import matplotlib.pyplot as plt
def drawing_and_save_picture(args):
    with open(args.embed_path_txt,'r',encoding='utf-8') as f:
        lines=f.readlines()[:1000]
    reduce_dim = TSNE(n_components=2)
    vectors,words=[],[]
    for idx,line in enumerate(lines):
        if idx%500==0:print('正在处理%d/%d'%(idx+1,len(lines)))
        if len(line.strip().split())<3:continue
        word=line.strip().split()[0]
        vector=list(map(lambda x:float(x),line.strip().split()[1:]))
        vectors.append(vector)
        words.append(word)
    vectors=np.array(vectors)
    vector = reduce_dim.fit_transform(np.array(vectors))
    idex = np.random.choice(np.arange(len(lines)), size=args.w_num, replace=False)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    for idx,i in enumerate(idex):
        if idx % 10 == 0: print('正在处理%d/%d' % (idx + 1, len(idex)))
        plt.scatter(vector[i][0], vector[i][1])
        plt.annotate(words[i], xy=(vector[i][0], vector[i][1]), xytext=(5, 2), textcoords='offset points')
    plt.title("tsne - " + args.picture_path.split("/")[-1].split(".")[0])
    plt.show()
    plt.savefig(args.picture_path)
    print(f"save picture to {args.picture_path}")