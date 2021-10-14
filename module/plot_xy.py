#!/usr/bin/env python
#  ------------------------------------------------------------------------------------------------
#  Descripe: 
#  Auther: jialiangtu 
#  CopyRight: Tencent Company
# ------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
