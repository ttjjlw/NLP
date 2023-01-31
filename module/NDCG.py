#!/usr/bin/env python
import numpy as np

def rerank(rank_scores,true_relevance):
    union=list(zip(rank_scores,true_relevance))
    rerank=sorted(union,key=lambda x:x[0],reverse=True)
    return [x[1] for x in rerank]

def getDCG(scores):
    '''
    :param scores: 类型为list/numpy.narray ，相关性得分
    :return:
    '''
    scores=np.array(scores)
    return np.sum(
        np.divide(np.power(2, scores) - 1, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)),
        dtype=np.float32)

def getNDCG(rank_scores,true_relevance):
    '''
    :param rank_scores: 类型为numpy.narray   预测的相关性得分
    :param true_relevance: 类型为numpy.narray  真实相关性得分
    :return:
    '''
    # 根据预测相关性得分对真实相关性得分排序，得到score
    score = rerank(rank_scores, true_relevance)
    idcg = getDCG(np.array(true_relevance))
    dcg = getDCG(np.array(score))

    if dcg == 0.0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg

if __name__ == '__main__':
    a=[i*1 for i in[2,5,4,3,1,1.4,1.1,1.3]]
    b=[5,4,3,2,0,0,0,0]
    print(rerank(a,b))
    print(getDCG(b))
    print(getNDCG(a,b))