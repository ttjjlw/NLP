def pension(rate):
    # 定义每月社保缴费基数（假设为5000元）
    sheping = 5000
    
    # 计算每月社保缴费金额：企业缴纳8%，个人缴纳12%，总比例为20%
    jiaofei = (0.08 + 0.16) * sheping * rate
    print('每个月需缴费：', jiaofei, '元')
    
    # 计算30年后每月领取的养老金
    # 统筹账户每月领取金额：退休时基数的0.5倍乘以个人账户积累比例（30%）
    lingqu = (1 + rate) * 0.5 * sheping * 0.3
    # 个人账户每月领取金额：个人缴费部分（8%）累计30年，按139个月折算
    lingqu1 = 0.08 * sheping * rate * 30 * 12 / 139
    print('每个月统筹账户可领取：%s元，个人账户可领取：%s元' % (str(lingqu), str(lingqu1)))
    print('每个月总共可领取：%s元' % str(lingqu + lingqu1))
    
    # 假设能领取15年（180个月）
    x = 15
    print('总共可领取：%s元' % (139 * lingqu1 + 15 * 12 * lingqu))
    print('总共缴费：%s元' % str(jiaofei * 30 * 12))
    return

if __name__ == '__main__':
    # 调用函数并传入缴费比例（例如0.6）
    print(pension(0.6))
