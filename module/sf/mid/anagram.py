def solution(arr):
    '''
    给你一个字符串数组，请你将字母异位词组合在一起。可以按任意顺序返回结果列表。
    :param arr:
    :return:
    '''
    dp=[]
    dp.append([[arr[0]]])
    for i in range(1,len(arr)):
        pre = dp[i-1]
        dp.append(adjust(pre,arr[i]))
    return dp[-1]

def adjust(pre,ele):
    result=[]
    for i in range(len(pre)):
        if isanagram(pre[i][0],ele):
            result.append(pre[i]+[ele])
            result.extend(pre[i+1:])
            return result
        else:
            result.append(pre[i])
    result.append([ele])
    return result




def isanagram(ele1,ele2):
    dic1={}
    for e1 in ele1:
        if e1 in dic1:
            dic1[e1]+=1
        else:
            dic1[e1]=1

    dic2 = {}
    for e2 in ele2:
        if e2 in dic2:
            dic2[e2] += 1
        else:
            dic2[e2] = 1
    return dic1==dic2


from collections import defaultdict

def groupAnagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        key = ''.join(sorted(s))  # 排序法
        groups[key].append(s)
    return list(groups.values())
if __name__ == '__main__':
    print(solution(["eat", "tea", "tan", "ate", "nat", "bat",'tab']))