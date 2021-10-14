#!/usr/bin/env python
#  ------------------------------------------------------------------------------------------------
#  Descripe: 
#  Auther: jialiangtu 
#  CopyRight: Tencent Company
# ------------------------------------------------------------------------------------------------
import numpy as np


def getDCG(scores):
    '''
    :param scores: 类型为numpy.narray ，相关性得分
    :return:
    '''
    return np.sum(
        np.divide(np.power(2, scores) - 1, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)),
        dtype=np.float32)

def getNDCG(rank_scores,true_relevance):
    '''
    :param rank_scores: 类型为numpy.narray   预测的相关性得分
    :param true_relevance: 类型为numpy.narray  真实相关性得分
    :return:
    '''
    idcg = getDCG(true_relevance)

    dcg = getDCG(rank_scores[:len(true_relevance)])

    if dcg == 0.0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg

if __name__ == '__main__':

    print(getDCG(np.array([-1,-1,-1.0])))
    print(getNDCG(np.array([-1.0,-1,-1]),np.array([1,2,3])))