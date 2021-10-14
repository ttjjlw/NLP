#!/usr/bin/env python
#  ------------------------------------------------------------------------------------------------
#  Descripe: 
#  Auther: jialiangtu 
#  CopyRight: Tencent Company
# ------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import math

def func(x):
    y=math.log(x)*-0.35+2.3
    return y

def plot(func):
    x=[i for i in range(10,100,2) if i!=0]
    y=[func(i) for i in x]
    plt.plot(x,y)
    plt.show()
if __name__ == '__main__':
    print(func(40))
    plot(func)