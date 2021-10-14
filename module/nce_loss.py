#!/usr/bin/env python
#  ------------------------------------------------------------------------------------------------
#  Descripe: 
#  Auther: jialiangtu 
#  CopyRight: Tencent Company
# ------------------------------------------------------------------------------------------------
import tensorflow as tf

#weights 为 utube 网络中的item vector
#inputs 为用户的vector

# [batch_size, dim] = [3, 2]
inputs = tf.constant([[0.2, 0.1], [0.4, 0.1], [0.22, 0.12]])
# [num_classes, dim] = [10, 2]
weights = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]],dtype=tf.float32
                      )
biases = tf.constant([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])# [num_classes] = [10]
# [batch_size, num_true] = [3, 1]
labels = tf.constant([[2], [3], [5]])
num_classes = 10


nce_loss = tf.nn.sampled_softmax_loss(
    weights=tf.constant([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]),
    # [num_classes, dim] = [10, 2]
    biases=tf.constant([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    # [num_classes] = [10]
    labels=tf.constant([[2], [3], [5]]),
    # [batch_size, num_true] = [3, 1]
    inputs=tf.constant([[0.2, 0.1], [0.4, 0.1], [0.22, 0.12]]),
    # [batch_size, dim] = [3, 2]

    num_sampled=3,
    num_classes=10,
    num_true=1,
    seed=2020,
    name="sampled_softmax_loss"
    )

logits = tf.matmul(inputs, tf.transpose(weights))
logits = tf.nn.bias_add(logits, biases)
labels_one_hot = tf.one_hot(labels, num_classes)
entroy_loss = tf.nn.softmax_cross_entropy_with_logits(
    labels=labels_one_hot,
    logits=logits)
if __name__ == '__main__':
    with tf.compat.v1.Session() as sess:
        print(sess.run(logits))
        print(sess.run(nce_loss))
        print(sess.run(entroy_loss))

    # [0.37478584 0.2859666  0.0703702 ]