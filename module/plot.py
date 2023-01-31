#!/usr/bin/env python
import matplotlib.pyplot as plt
import math

def func(x):
    y=math.log(x)*-0.35+2.3
    return y
def func_long(x):
    y =1.5842 * math.pow(x, 0.8387)
    return y
def func_cjs(x):
    if x>510:x=510
    y=math.log(x)*(-0.235)+1.7
    return y
def plot(func,b):
    x=[i for i in range(10000,1000000,100) if i!=0]
    y=[func(i,b) for i in x]
    plt.plot(x,y)
    plt.show()

def func_hot_supress(x):
    a=10 ** 3
    y=(math.sqrt(float(x) / a) + 5) *(a / float(x))
    return y
if __name__ == '__main__':
    print(func(50))
    print(func_long(3600))
    print(func_cjs(3600)*3600)
    for w in [0.1,1,10]:
        plot(func_hot_supress,w)
    plt.show()
