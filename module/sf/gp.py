def gp(arr,start,end):
    length = end-start
    if length<=0:
        return arr
    # mid=length//2
    mid = (start+end)//2
    gp(arr,start,mid)
    gp(arr,mid+1,end)
    merge_result = merge(arr[start:mid+1],arr[mid+1:end+1])
    #错误原因：在gp函数中，merge_result合并后直接赋值给arr，但arr是局部变量，不会影响外部的原始数组。因此，合并后的结果无法正确传递到上层递归调用。
    arr = merge_result
    return arr
def gp1(arr):
    length = len(arr)
    if length<=1:
        return arr
    # mid=length//2
    mid = length//2
    left = gp1(arr[0:mid])
    right = gp1(arr[mid:length])
    merge_result = merge(left,right)
    return merge_result

def merge(arr1,arr2):
    i,j=0,0
    merge_result=[]
    while i<len(arr1) and j <len(arr2):
        while i<len(arr1) and j <len(arr2) and arr1[i]<=arr2[j]:
            merge_result.append(arr1[i])
            i+=1
        while i<len(arr1) and j <len(arr2) and arr1[i]>arr2[j]:
            merge_result.append(arr2[j])
            j+=1

    merge_result.extend(arr1[i:])
    merge_result.extend(arr2[j:])
    return merge_result

if __name__ == '__main__':
    arr = [4, 10, 3, 5, 5, 1, 7, 7, 8, 2]
    print(gp1(arr))