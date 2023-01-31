#!/usr/bin/env python
def quick_sort(lis):
    '''
    思想：分治+递归
    :param lis:
    :return:
    '''
    #终止条件
    if type(lis)==int:
        print(lis)
    if len(lis)<2:return lis
    #把第一个元素移动动某个位置，使得其左侧小于该值，右侧大于等于该值
    high=len(lis)-1
    low=0
    while low<high:
        while lis[high]>=lis[low] and high>low and high>=0 and low<len(lis)-1:
            high-=1
        lis[high],lis[low]=lis[low],lis[high]
        while lis[low]<lis[high]and high>low and high>=0 and low<len(lis)-1:
            low+=1
        lis[high], lis[low] = lis[low], lis[high]
    # return lis,low
    mid=low
    #递归
    left=quick_sort(lis[:mid])
    right=quick_sort(lis[mid+1:])
    left.append(lis[mid])
    left.extend(right)
    return left

def quick_sort1(lis,start,end):
    if end-start<1:return
    low=start
    high=end
    while high>low:
        while high>low and lis[high]>=lis[low]:
            high-=1
        lis[high], lis[low] = lis[low], lis[high]
        while low<end and lis[low]<lis[high]:
            low+=1
        lis[high], lis[low] = lis[low], lis[high]
    assert low==high
    mid=low
    quick_sort1(lis,start=start,end=mid-1)
    quick_sort1(lis,start=mid+1,end=end)
    return lis



if __name__ == '__main__':
    lis=[7,0, 3, 7, 9, 21, 12]
    # lis=[21, 12]
    print(quick_sort(lis))
    print(quick_sort1(lis,0,len(lis)-1))