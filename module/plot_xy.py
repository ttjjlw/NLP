#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_line():
    df=pd.read_csv('label0_dictribute.csv',sep=',',header=0)
    df=df.sort_values('duration')
    print(df.head())
    x=list(df.duration.astype(int))
    y=list(df.c)
    print(x)
    print(y)
    plt.plot(x,y,color='r',linewidth=1.0,label='neg')
    plt.xlabel("duration")
    plt.ylabel("count")
    plt.xlim(0,400)              #设置x,y的区间
    # plt.ylim(0,400)
    plt.show()

def plot_bar():
    # 这两行代码解决 plt 中文显示的问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    waters = ('time', 'inter', 'exposure')
    buy_number = [13050200, 4500000, 12800890]

    a=plt.bar(waters, buy_number,width = 0.4,alpha=0.7)

    plt.title('One-hour sample distribution')

    plt.show()

if __name__ == '__main__':
    plot_bar()