#!/usr/bin/env python
import tensorflow as tf
def get_multi_hot(idx_tensor,total_num):
    '''
    :param idx_tensor: tensor shape=[bt,num] 如：[[0,3],[1,2]]  第二维这里不能有相同的元素，如 ：[[1,1],[2,3],否则get_drop会报错
    :param total_num: int 总数，决定最后multi_hot最后一维的维度 如：4
    :return: tensor 如：[[1,0,0,1],[0,1,1,0]]
    '''
    one_hot = tf.one_hot(idx_tensor, depth=total_num) #bt,x,total_num
    multi_hot = tf.reduce_sum(one_hot, axis=1)
    return multi_hot

def get_drop(raw,mask,drop_num):
    '''
    :param raw: Tensor  shape 为 [bt,seq,dim]
    :param mask: tensor[bool] [bt,seq] 注意True的个数必须等于 drop_num
    :param drop_num: seq中drop drop_num个元素
    :return: Tensor shape为[bt,seq-drop_num,dim]
    '''
    dim=raw.get_shape().as_list()[-1]
    seq=raw.get_shape().as_list()[-2]
    mask = tf.reshape(mask, [-1])
    raw = tf.reshape(raw, [-1, dim])
    after_drop = tf.boolean_mask(raw, mask, axis=0)  # 2,4,3
    after_drop = tf.reshape(after_drop, [-1, seq-drop_num, dim])
    return after_drop

def get_mask_zero(raw,mask):
    mask=tf.tile(tf.expand_dims(mask,axis=-1),[1,1,tf.shape(raw)[-1]])
    mask=tf.cast(mask,tf.float32)
    return raw*mask




if __name__ == '__main__':
    raw = tf.reshape(tf.constant(list(range(24)), dtype=tf.float32), shape=[2, 4, 3])  # bt,seq,dim
    drop_idx = tf.random.uniform(shape=(2, 2), minval=0, maxval=4, dtype=tf.int32)  # bt,num
    take_raw = tf.compat.v1.batch_gather(raw, indices=drop_idx)

    multi_hot = get_multi_hot(idx_tensor=drop_idx, total_num=4)
    mask = tf.cast(tf.equal(multi_hot, 0), tf.bool)  # 2,4
    # after_drop=get_drop(raw,mask,drop_num=2)
    after_mask = get_mask_zero(raw, mask)
    with tf.Session() as sess:
        #因为存在random，为保证结果一致，所以只能执行一次sess.run
        raw,drop_idx,multi_hot,take_raw,after_mask=sess.run([raw,drop_idx,multi_hot,take_raw,after_mask])
        print(raw)
        print("==========")
        print(drop_idx)
        print(multi_hot)
        print("====take raw =====")
        print(take_raw)
        print("==========")
        print(after_mask)

