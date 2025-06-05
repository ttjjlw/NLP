def find_position(arr,val):
    '''
    :param arr:从小到大有序数组
    :param val: 存在就返回arr，不存在就插入合适的位置
    :return:
    '''
    length = len(arr)
    for i in range(length):
        if arr[i]<val:continue
        if arr[i]==val:
            print( arr)
            return
        val,arr[i] = arr[i],val
    arr.append(val)
    print(arr)
    return

if __name__ == '__main__':
    arr=[1,2,4,6,6,8]
    find_position(arr,5)

