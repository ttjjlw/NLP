#!/usr/bin/env python
import datetime
def date_diff(start,end=datetime.datetime.now(),style='%Y%m%d%H'):
    '''
    :param start: str 类型 与 style相符 ，如 style='%Y%m%d%H'，则 start可为2021101801
    :param end: datetime.datetime or str 类型
    :param style: 与start相符
    :return: 相差小时数/24,即不足24小时，则返回0
    '''
    if isinstance(end,str):
        d1=datetime.datetime.strptime(end,style)
    else:
        d1=end
    d2=datetime.datetime.strptime(start,style)
    delta=d1-d2
    return delta.days

import datetime
def get_last_xhours(x,date_str,style='%Y%m%d%H'):

    d1 = datetime.datetime.strptime(date_str, style)
    xhours=[]
    for i in range(x):
        h=(d1+datetime.timedelta(hours=-i)).strftime(style)
        xhours.append(h)
    return xhours

if __name__ == '__main__':
    print(date_diff('2021101801'))
    print(get_last_xhours(24,date_str='2021102812'))