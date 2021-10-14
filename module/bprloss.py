#!/usr/bin/env python
#  ------------------------------------------------------------------------------------------------
#  Descripe: 
#  Auther: jialiangtu 
#  CopyRight: Tencent Company
# ------------------------------------------------------------------------------------------------
import tensorflow as tf

def dot(a, b):
    return tf.reduce_sum(tf.multiply(a, b), -1, keep_dims=False)


def cosin(a, b):
    a_b = dot(a, b)
    a_norm = tf.sqrt(tf.reduce_sum(tf.square(a), axis=-1))
    b_norm = tf.sqrt(tf.reduce_sum(tf.square(b), axis=-1))
    return tf.div(a_b, tf.multiply(a_norm, b_norm))
def bpr_loss(a,b,weight,l2_normal=False, cos=False):
    if not cos:
        if l2_normal:
            a = tf.nn.l2_normalize(a, axis=-1)
            b = tf.nn.l2_normalize(b, axis=-1)
        logits = dot(a, b)
    else:
        logits = cosin(a, b)
    pos=tf.expand_dims(logits[0],axis=0)
    loss=-tf.reduce_mean(tf.log(tf.nn.sigmoid(pos-logits[1:])),axis=0) #40,bt->bt
    loss=tf.reduce_mean(loss*weight)
    return loss