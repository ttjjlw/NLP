#!/usr/bin/env python
#  ------------------------------------------------------------------------------------------------
#  Descripe: 
#  Auther: jialiangtu 
#  CopyRight: Tencent Company
# ------------------------------------------------------------------------------------------------
#归并排序
def merge_sort(lis):
    '''
    分治+递归
    :param lis:
    :return:
    '''
    #终止条件
    if len(lis)<=1:return lis
    #分治思想：首先一分为二，分别调用自身排序（递归），然后合并排序即可
    #一分为二
    mid=len(lis)//2
    #递归调用
    left=merge_sort(lis[:mid])
    right=merge_sort(lis[mid:])
    #合并排序
    def combine_sort(left, right):
        l, r = 0, 0
        res = []
        while l <= len(left) - 1 and r <= len(right) - 1:
            while r <= len(right) - 1 and right[r] <= left[l]:
                res.append(right[r])
                r += 1
            while l <= len(left) - 1 and r <= len(right) - 1 and left[l] < right[r]:
                res.append(left[l])
                l += 1
        res.extend(left[l:])
        res.extend(right[r:])
        return res
    res=combine_sort(left, right)
    return res

def combine_sort(left,right):
    l,r=0,0
    res=[]
    while l<=len(left)-1 and r<=len(right)-1:
        while r<=len(right)-1 and right[r]<=left[l]:
            res.append(right[r])
            r+=1
        while l<=len(left)-1 and r<=len(right)-1 and left[l]<right[r]:
            res.append(left[l])
            l+=1
    res.extend(left[l:])
    res.extend(right[r:])
    return res
if __name__ == '__main__':
    left=[2,6,8,10]
    right=[1,3,5,7,9]
    print(combine_sort(left,right))

    lis=[1,9,7,8,4,12,0,4,0,12]
    print(merge_sort(lis))
