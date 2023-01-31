#!/usr/bin/env python

def test(lis):
    '''

    :param lis:
    :return:
    '''


def adjust(lis,start,end):
    root=start
    left=2*root+1
    right=2*root+2
    while right<=end:
        if lis[root]<lis[left] or lis[root]<lis[right]:
            if lis[left]>=lis[right]:
                lis[root],lis[left]=lis[left],lis[root]
                root=left
            else:
                lis[root],lis[right]=lis[right],lis[root]
                root=right
            left = 2 * root+1
            right = 2 * root + 2
        else:
            break
    if left<=end:
        if lis[root] < lis[left]:
            lis[root], lis[left] = lis[left], lis[root]
def build_seap(lis):
    for i in range(len(lis)-1,-1,-1):
        if i==2:
            print(i)
        adjust(lis,i,len(lis)-1)

def sort(lis):
    start=0
    end=len(lis)-1
    while end>start:
        lis[start],lis[end]=lis[end],lis[start]
        end -= 1
        if end>=0:adjust(lis,start,end)


if __name__ == '__main__':
    lis=[-1,5,4,9,2,6,2,3]
    # adjust(lis,0,len(lis)-1)
    # print(lis)
    build_seap(lis)
    print(lis)
    sort(lis)
    print(lis)