def shuzi_reverse(x):
    if x<0:return '负数翻转不了'
    if x%10==0:return '最后一位为0，翻转不了'
    shuzi = []
    while x>0:
        last = x%10
        shuzi.append(last)
        drop_last=x//10
        x = drop_last
    result=0
    length = len(shuzi)
    for i in range(length):
        result+=shuzi[i]*10**(length-1-i)
    print(result)

if __name__ == '__main__':
    x=12345
    shuzi_reverse(x)