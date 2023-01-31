#!/usr/bin/env python
#  ------------------------------------------------------------------------------------------------
#  Descripe: 
#  Auther: jialiangtu 
#  CopyRight: Tencent Company
# ------------------------------------------------------------------------------------------------

#旋转升序数组,值互不相同

def findInseqArr(lis,target):
    '''
    二分查找法，start=0,end=len(lis-1),mid=(start+end)//2
    if target>lis[0]:target在旋转左侧
        if lis[mid]>lis[0]:mid 在旋转左侧 ，否则右侧
            if lis[mid]>target:
                新范围 [start,mid]
            if lis[mid]<target:
                新范围 [mid,end]
        else:mid 在旋转右侧，恒小于target
            新范围 【start,mid】
    else:target在旋转右侧
        if lis[mid]>lis[0]:mid 在旋转左侧
            mid恒大于target
            新范围 【mid,end】
        else: mid 在旋转右侧
            if lis[mid]>target:
                新范围 [start,mid]
            if lis[mid]<target:
                新范围 [mid,end]

    :param lis:
    :return:
    '''

    def isleft(x,lis):
        if x>=lis[0]:
            return 1
        else:
            return 0
    start=0
    end=len(lis)-1
    while end-start>1:
        mid=(start+end)//2
        if isleft(target,lis):
            if isleft(lis[mid],lis):
                if lis[mid] > target:
                    # 新范围[start, mid]
                    end=mid
                elif lis[mid] < target:
                    # 新范围[mid, end]
                    start=end
                else:
                    return mid
            else:
                # 新范围 【start, mid】
                end=mid
        else:
            if isleft(lis[mid],lis):
                # 新范围 【mid, end】
                start=mid
            else:
                if lis[mid] > target:
                    # 新范围[start, mid]
                    end=mid
                elif lis[mid] < target:
                    # 新范围[mid, end]
                    start=mid
                else:
                    return mid
    if lis[start]==target:
        return start
    elif lis[end]==target:
        return end
    else:
        return -1


if __name__ == '__main__':
    lis=[4,5,6,1,2,3]
    target=4
    print(findInseqArr(lis,target))