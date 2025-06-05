def mp(arr):
    length=len(arr)
    for i in range(length): # 每次从第一个元素开始
        start = 0
        while start+1< length-i:
            if arr[start]>arr[start+1]: #依次与右边+1位比较
                arr[start],arr[start+1] = arr[start+1],arr[start]
                start+=1
            else:
                start+=1
    print(arr)

if __name__ == '__main__':
    arr=[4, 10, 3, 5,5, 1, 7, 7,8, 2]
    mp(arr)



