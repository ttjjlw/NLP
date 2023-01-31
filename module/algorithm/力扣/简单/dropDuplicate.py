#!/usr/bin/env python
#  ------------------------------------------------------------------------------------------------
#  Descripe: 
#  Auther: jialiangtu 
#  CopyRight: Tencent Company
# ------------------------------------------------------------------------------------------------
#升序数组，原地删除重复数组，不能改变原来的顺序，o(1)空间


























#[0,0,1,1,2,2,2,2,3,4]
def dropDuplicate(lis):
    '''
    双指针p1,p2 起初p1=0 p2=1,if lis[p2]==lis[p1] 则 p2+=1,if lis[p2]!=lis[p1],把p1+1指向的值与p2指向的值交换位置，然后p1前移一个位置，p2前移一个位置。
    重复以上步骤
    :param lis:
    :return:
    '''
    if len(lis)<2:return lis
    p1,p2=0,1
    while p2<=len(lis)-1:
        if lis[p1]==lis[p2]:
            p2+=1
        else:
            lis[p1+1],lis[p2]=lis[p2],lis[p1+1]
            p1+=1
            p2+=1
    return lis[:p1+1]

if __name__ == '__main__':
    lis=[0,0,1,1,2,2,2,2,3,4]
    print(dropDuplicate(lis))