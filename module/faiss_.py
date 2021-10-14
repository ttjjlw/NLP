#!/usr/bin/env python
#  ------------------------------------------------------------------------------------------------
#  Descripe: 
#  Auther: jialiangtu 
#  CopyRight: Tencent Company
# ------------------------------------------------------------------------------------------------
import faiss    # make faiss available
import numpy as np


d = 64                           # dimension
nb = 100                      # database size
nq = 10                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 100.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 10.
index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)

k = 4                          # we want to see 4 nearest neighbors
D, I = index.search(xb[:5], k) # sanity check
print(I) # shape 5,k 每一行最接近的k个 index
print(D)# shape 5，k 每一行最接近的k个的实际距离
D, I = index.search(xq, k)     # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])