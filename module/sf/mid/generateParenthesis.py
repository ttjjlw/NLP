def generateParenthesis(n):
    if n==0:return ['']
    res=[]
    backtrack(S=[], res=res, n=n, right=0, left=0)
    return res

def backtrack(S,res,n,right,left):
    if len(S)==2*n:
        res.append(''.join(S))
        return
    for s in ['(',')']: # 选择
        if s=='(':
            if left<n:
                S.append(s)
                backtrack(S, res, n, right, left+1)
            else:
                continue # 剪枝

        else:
            if right<left:
                S.append(s)
                backtrack(S, res, n, right+1, left)
            else:
                continue # 剪枝


        S.pop() #回溯
# n=3为例
# S=['(']
# backtrack(S, res=[], n, 0, 1)    1
#     S=['(','(']
#     backtrack(S, res=[], n, 0, 2)  2
#         S=['(','(','(']
#         backtrack(S, res=[], n, 0, 3)  3
#             # 选择'（',被剪枝
#             # 选择'）'
#             S=['(','(','(',')']
#             backtrack(S, res=[], n, 1, 3)  4
#                 # 选择'（',被剪枝
#                 # 选择'）'
#                 S=['(','(','(',')',')']
#                 backtrack(S, res=[], n, 2, 3)   5
#                     # 选择'（',被剪枝
#                     # 选择'）'
#                     S=['(','(','(',')',')',')']
#                     backtrack(S, res=[], n, 3, 3)  6
#                     res.append(S)
#                 S.pop()
# 当第6次backtrack执行完，在这次递归中走到了return,相当于第5次的backtrack(S, res=[], n, 2, 3)执行完了，下一步就是执行S.pop()
# 同理第5次执行S.pop()，就相当于第4次的backtrack(S, res=[], n, 1, 3)执行完了，下一步执行S.pop() S=['(','(','(']
# 同理第3次执行完了 S.pop() -> S=['(','('] ,这一步是'（'分支执行完了，进入'）'分支
#
# S=['(','(',')']
# backtrack(S, res=['((()))'], n, 1, 2)  3
#     S=['(','(',')','(']
if __name__ == '__main__':
    print(generateParenthesis(2))
