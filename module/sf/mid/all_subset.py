def all_subset(arr):
    if len(arr)==0:return []
    #包含top i 元素的子集合
    dp=[[[],[arr[0]]]]
    for i in range(1,len(arr)):
        pre = dp[i-1]
        tmp = [s+[arr[i]] for s in pre]
        dp.append(pre+tmp)
    return dp[-1]

if __name__ == '__main__':
    print(all_subset([1,2,3]))