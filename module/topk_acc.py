#!/usr/bin/env python
#  ------------------------------------------------------------------------------------------------
#  Descripe: 
#  Auther: jialiangtu 
#  CopyRight: Tencent Company
# ------------------------------------------------------------------------------------------------
def topk_accuracy(label,pred,k):

    max_k_preds = pred.argsort(axis=1)[:, -k:][:, ::-1]  # 得到top-k label
    match_array = np.logical_or.reduce(max_k_preds == label, axis=1)  # 得到匹配结果
    topk_acc_score = match_array.sum() / match_array.shape[0]
    return topk_acc_score

if __name__ == '__main__':
    import numpy as np
    pred = np.array([[0.02, 0.23, 0.22],
                     [0.05, 0.21, 0.39],
                     [0.24, 0.23, 0.22],
                     [0.05, 0.21, 0.39]]
                    )
    label=np.array([[1],[2],[1],[0]])
    # label=[[1],[2]]
    res1=topk_accuracy(label,pred,2)
    print(res1)


    #方法2
    from sklearn.metrics import top_k_accuracy_score

    y_true=np.reshape(label,(-1,))#当y_score的axis=1维度上，只有三个元素，则y_true中不能出现大于2的label
    # y_true = np.array([0, 1, 2]) #这里不能只有0和1，否则会认定为二分类，则y_score则必须是一维数组
    y_score = np.array([[0.02, 0.23, 0.22],
                     [0.05, 0.21, 0.39],
                     [0.24, 0.23, 0.22],
                     [0.05, 0.21, 0.39]])

    print(y_true.shape)
    print(top_k_accuracy_score(y_true, y_score, k=2))
    # print('res2:', res2)