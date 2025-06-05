def permutation(s):
    '''
    全排列
    :param s: 不能有重复的
    :return:
    '''
    if len(s)<=1:
        return [s]
    res=[]
    path=[]
    backtrace(path,res,s)
    return res

def backtrace(path,res,s):
    if len(path)==len(s):
        res.append(''.join(path))
    for i in s:
        if i in path:
            continue
        path.append(i)
        backtrace(path,res,s)
        path.pop()
if __name__ == '__main__':
    print(permutation('ABC'))
    print(permutation('A'))
