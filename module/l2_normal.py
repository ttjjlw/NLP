#!/usr/bin/env python
#  ------------------------------------------------------------------------------------------------
#  Descripe: 
#  Auther: jialiangtu 
#  CopyRight: Tencent Company
# ------------------------------------------------------------------------------------------------
import numpy as np
from sklearn import preprocessing
x=np.array([[1,2,3],[3,2,1]])
x_norm= preprocessing.normalize(x, norm='l2',axis=1)
print(x_norm)
