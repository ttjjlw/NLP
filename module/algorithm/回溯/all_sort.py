#!/usr/bin/env python
def all_sort(A):
    res=[]
    path=[]
    traceback(res,A,path)
    return res
#本质还是递归，递归只看第一层是否符合预期即可
def traceback(res,A,path):
    '''
    函数的含义：在当前确定的路径下，从候选集选择所有的全排，并放入res中 （只需要递归第一层是满足这个功能即可，顺下递归完成即可）
    :param res: 最终的结果
    :param A: 候选集
    :param path: 已选择的集合
    :return: 无返回值
    '''
    #递归退出条件
    if len(path)==len(A):
        if path not in res:
            res.append(path[:])
            return
    #分治思想+递归
    for i in range(0,len(A)):
        if A[i]  in path: continue
        path.append(A[i])
        traceback(res,A,path)
        path.pop()
if __name__ == '__main__':
    A=[1,2,3]
    print(all_sort(A))