#!/usr/bin/env python
#  ------------------------------------------------------------------------------------------------
#  Descripe: 
#  Auther: jialiangtu 
#  CopyRight: Tencent Company
# ------------------------------------------------------------------------------------------------
from typing import List, Callable, Tuple, Any, Optional

#题意：仅有一组满足情况
def twoSum(nums: List[int], target: int) -> List[int]:
    dic={}
    for i in range(len(nums)):
        if nums[i] not in dic:
            dic[nums[i]]=[i]
        else:
            dic[nums[i]].append(i)
    for i in range(len(nums)):
        v=nums[i]
        if target-v in dic:
            if len(dic[target-v])>=2:
                return dic[target-v][:2] #因为仅有一组满足情况，所以这里成立
            else:
                if i!=dic[target-v][0]:
                    return [i,dic[target-v][0]]
    return [-1]
if __name__ == '__main__':
    nums=[2,3,1,3]
    target=6
    print(twoSum(nums,target))