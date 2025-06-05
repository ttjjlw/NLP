
def coinChange(coins, amount: int) -> int:
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1


def coinChange1(coins,amount):
    '''
    :param coins:
    :param amount:
    :return:
    dp[i] 拼凑金额为i，需要最小coin数
    dp[n]=dp[n-c]+1 假如coin 有 2，3，5，拼凑为9元coin数，会等于 dp[9-5]+1,也有可能是dp[9-3]+1 或 dp[9-3]+1;三者取最小
    即 dp[n]=min(dp[n-c1],dp[n-c2],dp[n-c3])+1
    '''
    dp= [float('inf')] * (amount + 1)
    dp[0]=0
    for i in coins:
        if i <=len(dp)-1:
            dp[i]=1
    for i in range(1,amount+1):
        tmp=[]
        for c in coins:
            if i<c:
                tmp.append(float('inf'))
            else:
                tmp.append(dp[i-c]+1)
        dp[i]=min(tmp)
    return dp[amount] if dp[amount] != float('inf') else -1

if __name__ == '__main__':
    print(coinChange1([2,3,5],1))