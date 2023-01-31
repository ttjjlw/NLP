#!/usr/bin/env python
# 参考：https://zhuanlan.zhihu.com/p/66514492
import tensorflow as tf


def list_wise(true_scores, rank_scores, list_size, pos_list_size):
    '''
    :param true_scores: 真实得分（分值不重要，能确定先后顺序就行）,shape (bt,list_size)
    :param rank_scores: 预测得分, shape (bt,list_size)
    :param list_size 真实list_size
    :param pos_list_size 只考虑top pos_list_size的排序，后面的排序不影响loss
    :return:
    '''

    index = tf.argsort(true_scores, direction='DESCENDING')
    # print(index)[1,2,0] [1,0,2]
    # 按真实得分对rank_scores排序
    S_predict = tf.batch_gather(rank_scores, index)
    # S_predict=tf.gather(rank_scores, index,axis=-1,batch_dims=0)
    # 分子
    initm_up = tf.constant([[1.0]])  # shape 为 1,1
    for i in range(pos_list_size):  # 3为 pos_list_size
        # index=tf.constant([[i]]*batch_size,dtype=tf.int32) #shape 为 bt,1
        # a=tf.batch_gather(S_predict,index)
        a = tf.slice(S_predict, [0, i], [-1, 1])  # shape 为 bt,1
        # a+=1
        initm_up = initm_up * a  # bt,1

    # 分母
    initm_down = tf.constant([[1.0]])  # shape 为 1,1
    for i in range(pos_list_size):
        b = tf.reduce_sum(tf.slice(S_predict, [0, i], [-1, list_size - i]), axis=-1, keep_dims=True)  # bt，1
        # b+=1
        initm_down *= b  # shape 为 bt,1
    loss = tf.divide(initm_up, initm_down)
    mleloss = -tf.reduce_mean(tf.log(loss))
    # return mleloss

    with tf.Session() as sess:
        print(sess.run(S_predict))
        print('up:',sess.run(initm_up))
        print('down:',sess.run(initm_down))
        print(sess.run(loss))
        print(sess.run(mleloss))


def mle1(true_scores, rank_scores, list_size, pos_list_size):
    '''
    :param true_scores: 真实得分（分值不重要，能确定先后顺序就行）,shape (bt,list_size)
    :param rank_scores: 预测得分, shape (bt,list_size)
    :param list_size 真实list_size
    :param pos_list_size 只考虑top pos_list_size的排序，后面的排序不影响loss
    :return:
    '''

    index = tf.argsort(true_scores, direction='DESCENDING')
    # print(index)[1,2,0] [1,0,2]
    # 按真实得分对rank_scores排序
    S_predict = tf.batch_gather(rank_scores, index)
    loss=tf.constant([[0.0]])
    for i in range(pos_list_size):  # 3为 pos_list_size
        # 分子
        up = tf.slice(S_predict, [0, i], [-1, 1])  # shape 为 bt,1
        # 分母
        down = tf.reduce_sum(tf.slice(S_predict, [0, i], [-1, list_size - i]), axis=-1, keep_dims=True)  # bt，1
        loss+=tf.log(tf.divide(up,down))
    mleloss = -tf.reduce_mean(loss)
    # return mleloss

    with tf.Session() as sess:
        print(sess.run(S_predict))
        print('up:',sess.run(up))
        print('down:',sess.run(down))
        print(sess.run(loss))
        print(sess.run(mleloss))
if __name__ == '__main__':
    true_scores = tf.constant([[3.5, 3, 2.5], [5, 4, 3.0]])
    rank_scores = tf.constant([[6, 5,4], [3, 4, 5.0]])

    # true_scores = tf.nn.softmax(true_scores)
    rank_scores = tf.nn.softmax(rank_scores)
    mle1(true_scores, rank_scores,list_size=3,pos_list_size=3)
    print('====')
    mle1(true_scores, rank_scores,list_size=3,pos_list_size=3)
    # print(rank_scores)
    # sess=tf.Session()
    # print(sess.run(mle_loss(true_scores,rank_scores)))

    # [[-1.6272742]
    #  [-1.5100404]]
