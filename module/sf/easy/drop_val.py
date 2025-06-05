def drop_val(arr,val):
    length=len(arr)
    for i in range(length):
        if arr[i]==val:
            if i==length-1:
                arr[i]='_'
                break
            for j in range(i,length-1):
                if arr[j]=='_':
                    break
                arr[j]=arr[j+1]
            arr[j+1]='_'
    print(arr)

if __name__ == '__main__':
    arr=[1,2,5,4,5]
    drop_val(arr,5)


