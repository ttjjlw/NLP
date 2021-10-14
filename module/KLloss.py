#!/usr/bin/env python
#  ------------------------------------------------------------------------------------------------
#  Descripe: 
#  Auther: jialiangtu 
#  CopyRight: Tencent Company
# ------------------------------------------------------------------------------------------------
import tensorflow as tf


def kl_loss(label_prob,pred_prob):
    """
    :param label_prob: 真实概率分布
    :param pred_prob: 预测概率分布
    :return:
    """
    p_q = label_prob/tf.clip_by_value(pred_prob,1e-8,1)
    kl_loss=tf.reduce_sum(label_prob * tf.log(p_q), axis=-1)
    with tf.Session() as sess:
        kl_loss=sess.run(kl_loss)
    return kl_loss




if __name__ == '__main__':
    pred_prob = tf.nn.softmax([[2, 1.0, 0, -2, -3], [100, 1, -1, -2, -3]])
    label_prob = tf.nn.softmax([[5, 4, 3, 2, 1.0], [5, 4, 3, 2, 1.0]])
    print(kl_loss(label_prob,pred_prob))
    print(kl_loss(label_prob=tf.constant([0.5,0.5]),pred_prob=tf.constant([0.25,0.75])))