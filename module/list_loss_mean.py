#!/usr/bin/env python
#  ------------------------------------------------------------------------------------------------
#  Descripe: 
#  Auther: jialiangtu 
#  CopyRight: Tencent Company
# ------------------------------------------------------------------------------------------------
import tensorflow as tf
def mean_loss(true_scores,rank_sores,threshold=2.4):
    '''
    :param true_scores: 真实得分序列 true_scores=tf.constant([[3.5,3.0,2.5,1.0,2.5,1.0,1.0],[3.5,2.5,2.5,1.0,3.0,1.0,1.0]])
    :param rank_sores: 预测得分序列 rank_sores=tf.constant([[5,4,3,1.0,1.0,2.0,1.0],[4,4,4,1.0,3.0,1.0,1.0]])
    :param threshold: 大于阈值的就代表正样本
    :return: loss
    含义：正样本对应的rank_score分值越大，负样本对应的rank_score越小，则loss越小，同时正样本的分值越接近，则loss越小，反之则越大
    '''
    mask=tf.cast(tf.cast(true_scores>threshold,tf.bool),tf.float32)
    log_prob = tf.nn.log_softmax(rank_sores, axis=-1)
    n = tf.reduce_sum(mask, axis=-1, keep_dims=True)
    loss = -1 * n * tf.log(n + 1e-9) - tf.reduce_sum(log_prob * mask, axis=-1, keep_dims=True)
    loss = tf.reduce_mean(loss)
    return loss

if __name__ == '__main__':
    true_scores=tf.constant([[3.5,3.0,2.5,1.0,1.0,1.0,1.0],[3.5,2.5,2.5,1.0,1.0,1.0,1.0]])
    rank_sores=tf.constant([[5,4,3,1.0,1.0,1.0,1.0],[4,4,4,1.0,1.0,1.0,1.0]])
    loss=mean_loss(true_scores,rank_sores)
    with tf.Session() as sess:
        # print(sess.run(loss1))
        print(sess.run(loss))