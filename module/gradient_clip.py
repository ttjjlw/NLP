#!/usr/bin/env python
#  ------------------------------------------------------------------------------------------------
#  Descripe: 
#  Auther: jialiangtu 
#  CopyRight: Tencent Company
# ------------------------------------------------------------------------------------------------
import tensorflow as tf
loss = 1.0
optimizer = tf.train.GradientDescentOptimizer(0.1)
grads_and_vars = optimizer.compute_gradients(loss)
# grads = tf.gradients(loss,[w,x])
# 修正梯度
for i,(gradient,var) in enumerate(grads_and_vars):
    if gradient is not None:
        grads_and_vars[i] = (tf.clip_by_norm(gradient,5),var)
train_op = optimizer.apply_gradients(grads_and_vars)
