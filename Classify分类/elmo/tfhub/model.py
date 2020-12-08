# F:\itsoftware\Anaconda
# -*- coding:utf-8 -*-
# Author = TJL
# date:2020/12/3
import tensorflow as tf
import os
import tensorflow_hub as hub
from dataset import batch_yield
import numpy as np
from sklearn.metrics import f1_score


os.environ["TFHUB_CACHE_DIR"]='preprocessing/tfhub' #设置模型缓存路径
elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)

class Model(object):
    def __init__(self,args):
        self.lr=args.lr
        self.export_path=args.export_path
        self.result_path=args.result_path
        self.label_num=args.label_num
        self.epochs=args.epochs
        self.batch_size=args.batch_size
    def build_train_graph(self):
        self.add_placehoder()
        self.add_elmo()
        self.add_dense()
        self.add_loss()
        self.add_optimizer()
    def add_placehoder(self):
        self.input=tf.placeholder(shape=[None,],dtype=tf.string)
        self.label=tf.placeholder(shape=[None],dtype=tf.int32)
        self.keep_rate=tf.placeholder(shape=[],dtype=tf.float32)
    def add_elmo(self):
        self.elmo_embeddings=elmo(self.input)
    def add_dense(self):
        self.output=tf.layers.dense(self.elmo_embeddings, units=self.label_num,activation=None)
        self.logits=tf.nn.softmax(self.output)
    def add_loss(self):
        self.loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output,labels=self.label)
        self.loss = tf.reduce_mean(self.loss)
    def add_optimizer(self):
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optim = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optim.minimize(self.loss, global_step=self.global_step)
    def train(self,args,train,valid,test):
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            for epoch in range(self.epochs):
                generator_batch=batch_yield(train, batch_size=self.batch_size, shuffle=True)
                loss_epoch=0
                pred_labels=[]
                label_lis=[]
                for steps,(input,label) in enumerate(generator_batch):
                    # if steps==0:args.learn_rate=1e-7
                    # if steps==1:args.learn_rate=3e-5
                    feed_dict={
                               self.input:input,
                               self.label:label,
                               self.keep_rate: args.keep_rate,
                        # self.is_train: True,
                               }
                    _, loss_train, step_num_,logits= sess.run([self.train_op, self.loss, self.global_step,self.logits],
                                                         feed_dict=feed_dict)
                    pred_label=np.argmax(logits,axis=-1)
                    pred_labels.extend(pred_label)
                    label_lis.extend(label)
                    loss_epoch+=loss_train
                    if steps%100==0:
                        print('epoch: {} loss_train:{:.4}'.format(epoch+1,loss_epoch/(steps+1)))
                t_score=f1_score(label_lis,pred_labels)
                pred_labels, input_labels, score, loss_avg=self.evaluate(sess,valid)
                print('train: epoch: {} tf1score: {} train_loss: {}'.format(epoch + 1, t_score, loss_epoch/(steps+1)))
                print('valid: epoch: {} eval_f1_score: {} eval_loss: {}'.format(epoch+1,score,loss_avg))
                # self.loginfo.info('#'*100)
                # if not os.path.exists(self.export_model_path):os.makedirs(self.export_model_path)
                # saver.save(sess,save_path=self.export_model_path+'ckpt',global_step=epoch )

         
    def evaluate(self,sess,data):
        generator_batch = batch_yield(data, batch_size=self.batch_size, shuffle=False)
        # Initialize accumulators
        input_labels = []
        pred_labels = []
        cos_lis = []
        losses = []
        for steps, (input, label) in enumerate(generator_batch):
            feed_dict = {
                self.input: input,
                self.label: label,
                self.keep_rate: 1,
            }
            loss_eval, logits = sess.run([self.loss, self.logits],  # self.cos,self.score0],
                                         feed_dict=feed_dict)
            pred_label = np.argmax(logits, axis=-1)
            pred_labels.extend(list(pred_label))
            # add batch to accumulators
            assert (len(logits) <= self.batch_size)
            losses.append(loss_eval)
            input_labels.extend(list(label))
            # self.loginfo.info('cos shape:{}'.format(cos.shape))
            # self.loginfo.info('cos:{}'.format(cos))
            # self.loginfo.info('score0:{}'.format(score0))
            # cos_lis.extend(cos)
        # self.loginfo.info('vpred_logits:{}'.format(pred_logits))
        # self.loginfo.info('vinput_labels:{}'.format(input_labels))
        # self.loginfo.info('vcos:{}'.format(cos_lis))
        assert (len(data) == len(pred_labels))
        score=f1_score(input_labels,pred_labels)
        loss_avg = sum(losses) / len(losses)
        return pred_labels, input_labels, score, loss_avg