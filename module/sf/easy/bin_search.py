def bin_search(arr,target,low,high):
    '''
    :param arr:从小到大的有序数组
    :param target: 需要查找的目标，能找到返回索引，找不到返回-1
    :return:
    '''
    mid = (low + high) // 2
    while high-low>1:
        if arr[mid]==target:
            return mid

        if arr[mid]>target:
            high=mid
            mid=(low+high)//2
        else:
            low = mid
            mid = (low + high) // 2
    if arr[low] == target:
        return low
    if arr[high] == target:
        return high
    return -1

if __name__ == '__main__':
    arr=[1,2,3,4,5]
    res=bin_search(arr,1,0,len(arr)-1)
    print(res)










































