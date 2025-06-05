def kp(arr,left,right):
    if right<=0:
        return arr
    while True:
        if left>=right:break
        while left<right and arr[left]<=arr[right]:
            left+=1
        arr[left],arr[right]=arr[right],arr[left]
        while left<right and arr[left]<=arr[right] :
            right-=1
        arr[left], arr[right] = arr[right], arr[left]
    arr_left = kp(arr[:left],left=0,right=len(arr[:left])-1)
    arr_right = kp(arr[left+1:],left=0,right=len(arr[left+1:])-1)
    result = arr_left+ [arr[left]] + arr_right
    return result

def kp1(arr,left,right): #优化
    start =left
    end = right
    if end-start<=0:
        return arr
    while True:
        if left>=right:break
        while left<right and arr[left]<=arr[right] :
            left+=1
        arr[left],arr[right]=arr[right],arr[left]
        while left<right and arr[left]<=arr[right]:
            right-=1
        arr[left], arr[right] = arr[right], arr[left]
    kp1(arr,left=start,right=left-1)
    kp1(arr,left=left+1,right=end)

    return arr





if __name__ == '__main__':
    arr = [4, 10, 3, 5, 5, 1, 7, 7, 8, 2]
    print(kp(arr,0,len(arr)-1))