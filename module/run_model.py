#!/usr/bin/env python
#  ------------------------------------------------------------------------------------------------
#  Descripe: 
#  Auther: jialiangtu 
#  CopyRight: Tencent Company
# ------------------------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST.data",one_hot=True)

print('输入数据：',mnist.train.images)
print('输入数据的大小：',mnist.train.images.shape)
import pylab
im=mnist.train.images[1]
im=im.reshape(-1,28)
pylab.imshow(im)
pylab.show()

tf.reset_default_graph()

inputs=tf.placeholder(tf.float32,[None,784])
Y=tf.placeholder(tf.float32,[None,10])
labels=tf.reshape(tf.argmax(Y,1),[-1,1])

W=tf.Variable(tf.random_normal([784,10]))
Weight=tf.transpose(W)
bias=tf.Variable(tf.zeros([10]))
num_sampled = 3
num_true = 1
num_classes = 10

cost=tf.reduce_mean(tf.nn.sampled_softmax_loss(
                     weights=Weight,
                     biases=bias,
                     labels=labels,
                     inputs=inputs,
                     num_sampled=num_sampled,
                     num_true=num_true,
                     num_classes=num_classes))


learning_rate=0.001
optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
training_epochs=100
batch_size=100
display_step=1
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost=0.
        total_batch=int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs,bathc_ys=mnist.train.next_batch(batch_size)
            _,c=sess.run([optimizer,cost],feed_dict={inputs:batch_xs,Y:bathc_ys})
            avg_cost+=c
        avg_cost/=i+1
        if (epoch+1) % display_step == 0:
            print("Epoch:",'%04d'%(epoch+1),"cost=","{:.9f}".format(avg_cost))
    print("Finished")
