# F:\itsoftware\Anaconda
# -*- coding:utf-8 -*-
# Author = TJL
# date:2020/11/23
import tensorflow as tf
import os,pickle
import numpy as np
from dataset import build_batch
class Word2vec(object):
    def __init__(self,args):
        self.mode=args.mode
        self.vocab2id=args.vocab
        self.id2vocab=dict(zip(args.vocab.values(),args.vocab.keys()))
        self.embed_dim=args.embed_dim
        self.init_rate=args.init_rate
        self.neg_samples=args.neg_samples
        self.decay_steps=args.decay_steps
        self.raw_data=args.raw_data
        self.batch_size=args.batch_size
        self.window_size=args.window_size
        self.valid_example=args.valid_example
        self.valid_size=args.valid_size
        self.top_k = args.top_k
        self.epochs=args.epochs
        self.log_per_steps=args.log_per_steps
        self.save_path=args.save_path
        self.is_save_vector=args.is_save_vector
        self.embeddings_save_path=args.embeddings_save_path
        self.is_load=args.is_load
    def build_graph(self):
        self.train_x=tf.placeholder(tf.int32,[None],name='train_x')
        self.train_y=tf.placeholder(tf.int32,[None,1],name='train_y')
        self.embeddings=tf.Variable(tf.random_uniform([len(self.vocab2id),self.embed_dim],-1,1),name='embeddings')
        if self.mode=='train':
            self.valid_data = tf.constant(self.valid_example, tf.int32, name='valid_data')
        elif self.mode=='predict':
            self.valid_data = tf.placeholder(tf.int32, shape=None)
        else:
            ValueError('mode is train or predict')
        # embeddinga=tf.Variable(tf.random_normal([vocabulary_size,embedding_size]))
        # 编码的时候要注意，频率高的词用小数字编码，文本中词的编码是否是从零开始，
        #如果不从零编码，vocabulary_size应该等于最大的那个数字编码+1，而不应是字典的长度
        self.embed=tf.nn.embedding_lookup(self.embeddings,self.train_x)
        self.nce_weight=tf.Variable(tf.truncated_normal([len(self.vocab2id),self.embed_dim],
                                                   stddev=1.0/np.sqrt(self.embed_dim)),name='nce_weight')
        self.nce_bias=tf.Variable(tf.zeros([len(self.vocab2id)]),name='nce_bias')
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), axis=1, keep_dims=True))
        self.normalized_embeddings = self.embeddings / norm  # 除以其L2范数后得到标准化后的normalized_embeddings
        self.valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings,
                                              self.valid_data)  # 如果输入的是64，那么对应的embedding是normalized_embeddings第64行的vector
        self.similarity = tf.matmul(self.valid_embeddings, self.normalized_embeddings, transpose_b=True) #shape（20,2000） # 计算验证单词的嵌入向量与词汇表中所有单词的相似性
        print('graph build successfully!')
    def add_loss(self):
        self.nce_loss = tf.reduce_mean(tf.nn.nce_loss(inputs=self.embed, weights=self.nce_weight, biases=self.nce_bias, num_sampled=self.neg_samples, labels=self.train_y,
                           num_classes=len(self.vocab2id)))
        self.global_step=tf.Variable(0,trainable=False)
        assert self.decay_steps>0
        self.learning_rate = tf.train.exponential_decay(self.init_rate, self.global_step,self.decay_steps, 0.96)
        # self.train_ = tf.train.AdamOptimizer(self.learning_rate).minimize(self.nce_loss, global_step=self.global_step)
        self.train_ = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.nce_loss,global_step=self.global_step)
        self.init=tf.global_variables_initializer()
    def save(self,sess):
        saver=tf.train.Saver()                      # 1/save model
        self.saved_path=saver.save(sess,self.save_path,global_step=self.global_step)
        print('{}-save model finished!'.format(self.saved_path))
    def restore(self,sess):
        restored=tf.train.Saver()
        restored.restore(sess,tf.train.latest_checkpoint(self.save_path))
        print('{}-restored Finished!'.format(tf.train.latest_checkpoint(self.save_path)))
    def train(self):
        with tf.Session() as sess:
            sess.run(self.init)
            if self.is_load:
                self.restore(sess)
            for epoch in range(self.epochs):
                for steps in range((len(self.raw_data)+self.batch_size-1)//self.batch_size):
                    train_batch,train_label=build_batch(self.raw_data,self.vocab2id,self.batch_size,self.window_size)
                    feed_dict={self.train_x:train_batch,self.train_y:train_label}
                    _,loss,learn_rate=sess.run([self.train_,self.nce_loss,self.learning_rate],feed_dict=feed_dict)
                    #每隔10000次打印一次
                    if steps%self.log_per_steps==0:
                        print('learning_rate',self.learning_rate)
                        print('loss:',loss)
                    #计算相似性
                    # 每10000次，验证单词与全部单词的相似度，并将与每个验证单词最相似的5个找出来。
                    if steps % self.log_per_steps== 0:
                        sim = self.similarity.eval()
                        for i in range(self.valid_size):
                            valid_word = self.id2vocab[self.valid_example[i]]  # 得到验证单词
                            nearest = (-sim[i, :]).argsort()[0:self.top_k+1]  # 每一个valid_example相似度最高的top-k个单词,除了自己
                            log_str = "Nearest to %s:" % valid_word
                            for index in nearest:
                                close_word_similarity = sim[i, index]
                                close_word = self.id2vocab[index]
                                log_str = "%s %s(%s)," % (log_str, close_word, close_word_similarity)
                            print(log_str)
                #每epoch保存一次模型
                self.save(sess)
    def predict(self):
        with tf.Session() as sess:
            self.restore(sess)
            # 保存词向量
            if self.is_save_vector:
                embed =self.normalized_embeddings.eval()
                with open(self.embeddings_save_path+'embed.pkl','wb') as f:
                    pickle.dump(embed,f)
                print('成功保存词向量！')
            while 1:
                word = input('请输入：')
                print(word)
                if word in ['退出', 'q']:
                    break
                if word not in self.vocab2id:
                    print('该词不在语料库中')                     #用return不会打印啊
                    continue
                value_int = self.vocab2id[word]
                value_int = np.array([value_int])
    
                sim, word_emberdding = sess.run([self.similarity, self.valid_embeddings], feed_dict={self.valid_data: value_int})
                sim_sort = (-sim[0, :]).argsort()  # index从大到小排序，index对应dictionary_reverse字典
                nearest = sim_sort[1:self.top_k + 1]  # 前top_k个,不包括自己
                log_str = "Nearest to %s:" % (word)
                for index in nearest:
                    close_word_similarity = sim[0, index]
                    close_word = self.id2vocab[index]
                    log_str = "%s: %s(%s)," % (log_str, close_word, close_word_similarity)
                print(log_str)
