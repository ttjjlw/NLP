def findthTK(arr,k):
    def quickTk(arr,k,left,right):
        if right-left+1<k:return 'null'
        seq = partition(arr,left,right)

        pos = right-seq+1 # seq及后面元素个数
        if pos>k:
            return quickTk(arr, k, seq+1, right) # 一定要seq+1而不是seq，要缩小搜索范围避免陷入死循环
        elif pos<k:
            return quickTk(arr, k-pos, left, seq-1)
        else:
            return arr[seq]
    return quickTk(arr,k,left=0,right=len(arr)-1)


def partition(arr,left,right):
    while right>left:
        while right>left and arr[right]>=arr[left]:
            right-=1
        arr[left], arr[right] = arr[right], arr[left]
        while right>left and arr[left]<=arr[right]:
            left+=1
        arr[left], arr[right] = arr[right], arr[left]
    sep = left
    return sep


def findTK(arr,k):
    def quickTk(arr,k,left,right):
        if right-left+1<k:return 'null'
        seq = partition(arr,left,right)

        pos = right-seq+1 # seq及后面元素个数
        if pos>k:
            return quickTk(arr, k, seq+1, right) # 一定要seq+1而不是seq，要缩小搜索范围避免陷入死循环
        elif pos<k:
            return quickTk(arr, k-pos, left, seq-1)+arr[seq:right+1]
        else:
            return arr[seq:right+1]
    return quickTk(arr,k,left=0,right=len(arr)-1)

if __name__ == '__main__':
    result = findthTK(arr=[8,4,6,1,2,-9,0,-1],k=7)
    print(result)
    result1 = findTK(arr=[8, 4, 6, 1, 2, -9, 0, -1], k=2)
    print(result1)