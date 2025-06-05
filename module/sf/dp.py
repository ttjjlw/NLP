def dp(arr):
    # build stack
    length=len(arr)
    for i in range(length//2-1,-1,-1):
        adjust(arr=arr,root=i,end=length-1)

    # exchange & adjust
    for i in range(length):
        arr[0],arr[length-1-i] = arr[length-1-i],arr[0]
        adjust(arr=arr, root=0, end=length - 1-i-1)

    print(arr)



def adjust(arr,root,end):
    i=root
    j=2*root+1
    while j<=end:
        if j+1<=end and arr[j+1]>arr[j]:
            j=j+1
        if arr[i]<arr[j]:
            arr[i],arr[j] = arr[j],arr[i]
            i=j
            j=2*i+1
        else:
            break

if __name__ == '__main__':
    arr = [4, 10, 3, 5, 5, 1, 7, 7, 8, 2]
    dp(arr)