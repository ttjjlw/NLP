#!/usr/bin/env python

def get_topk(lis,k):
    '''
    借鉴快排，复杂度为O（N），因为左边》=lis[mid]》右边，所以如果 mid=k-1 则lis[mid]刚好是第k大
    if mid>k-1,搜索范围变成 [start,mid-1]
    if mid<k-1 搜索范围变成 [mid+1,end]
    :param lis:
    :return:
    '''
    start=0
    end=len(lis)-1
    if k>end+1:return "不存在"
    while 1:
        low=start
        high=end
        while low<high:
            while low<high and lis[high]<=lis[low]: #小于lis[mid]往右移
                high-=1
            lis[high],lis[low]=lis[low],lis[high]
            while low<high and lis[low]>lis[high]:
                low+=1
            lis[high], lis[low] = lis[low], lis[high]
        assert low==high
        mid=low
        if mid==k-1:return lis[mid]
        elif mid>k-1:
            end=mid-1
        else:
            start=mid+1

if __name__ == '__main__':
    lis=[11,10,8,2,5,7,6,8,9]
    k=3
    print(get_topk(lis,k))