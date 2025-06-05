def xp(arr):
    length = len(arr)
    for i in range(length):#依次选择开头位置
        j=i
        while j+1<length: #依次比较，开头位置始终放最小的值
            if arr[i]>arr[j+1]:
                arr[i],arr[j+1] = arr[j+1],arr[i]
                j+=1
            else:
                j+=1
    print(arr)

if __name__ == '__main__':
    arr = [4, 10, 3, 5, 5, 1, 7, 7, 8, 2]
    xp(arr)