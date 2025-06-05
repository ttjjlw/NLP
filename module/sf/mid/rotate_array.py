def rotate_array(arr,target):
    low=0
    high=len(arr)-1

    mid=(low+high)//2
    while high-low>1:
        if arr[-1]>=target: #在右半部分
            if arr[high]>=arr[mid]: # mid 在右半部分
                if target>arr[mid]:
                    low=mid
                    mid=(low+high)//2
                elif target < arr[mid]:
                    high=mid
                    mid = (low + high) // 2
                else:
                    return mid
            if arr[mid]>=arr[0]: # mid 在左半部分
                low=mid
                mid = (low + high) // 2

        if target>=arr[0]:
            if arr[mid]>=arr[0]:
                if target>arr[mid]:
                    low=mid
                    mid=(low+high)//2
                elif target < arr[mid]:
                    high=mid
                    mid = (low + high) // 2
                else:
                    return mid
            if arr[mid]<=arr[-1]:
                high=mid
                mid = (low + high) // 2
    if arr[high]==target:return high
    if arr[low]==target:return low

    return -1

if __name__ == '__main__':
    arr=[6,7,8,9,1,2,3,4,5]
    print(rotate_array(arr,5))



