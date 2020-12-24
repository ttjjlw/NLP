# F:\itsoftware\Anaconda
# -*- coding:utf-8 -*-
# Author = TJL
# date:2020/5/22
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_eager_execution()#禁用Tensorflow2 默认的即时执行模式。
import argparse,random,os
import logging,pickle
import numpy as np
from bert_model import modeling,albert_modeling,robert_modeling,tokenization,albert_tokenization,robert_tokenization,optimization,albert_optimization
#from TJl_function import get_logger
from dataset import batch_yield
import matplotlib.pyplot as plt

def plot_graph(x,y):
    plt.cla()
    for i,j ,nm in zip (x,y,['tloss','tf1','tp','tr','vloss','vf1','vp','vr']):
        plt.cla()
        plt.plot(i,j,label=nm)
        plt.legend()
        if not os.path.exists('image/'): os.makedirs('image/')
        plt.savefig('image/%s.png' % nm)  # 保存图片
        plt.show()

def get_logger(filename):
    '''
    :param filename: 日志保存路径
    :return:
    '''
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger
def load_weights(init_checkpoint,model):
    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
        (assignment_map, initialized_variable_names
         ) = model.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    print("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        print("  name = %s, shape = %s%s", var.name, var.shape,
              init_string)
def get_label(threshold,logits):
    pred_label=[]
    for i in logits:
        if i>=threshold:
            pred_label.append(1)
        else:
            pred_label.append(0)
    return pred_label
def get_score(pred_label,label):
    assert len(pred_label)==len(label)
    correct=0
    for p,l in zip(pred_label,label):
        if p==1 and p==l:
            correct+=1
    if sum(pred_label)==0 or correct==0:
        return 0,0,0
    precision=correct/sum(pred_label)
    recall=correct/sum(label)
    score=2*precision*recall/(precision+recall)
    return score,precision,recall
def which_threshold(threshold,logits,label):
    optimal_td=0
    optimal_score=0
    score_lis=[]
    for td in threshold:
        td=td/100
        pred_label=get_label(td, logits)
        score=get_score(pred_label,label)
        score_lis.append(score)
        if optimal_score<score:
            optimal_td=td
            optimal_score=score
    return optimal_td,optimal_score

class Classify_BertModel(object):
    def __init__(self,args):
        self.max_len = args.max_len
        self.bert_config_path = args.bert_config_path
        self.bert_path = args.bert_path
        self.loginfo = get_logger(args.log_path)
        self.export_model_path = args.export_model_path
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.is_train = args.is_train
        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.4  # 占用40%显存
        self.tokenize = tokenization.FullTokenizer(args.vocab_path, do_lower_case=True)
        self.num_train_steps=args.num_train_steps
        self.num_warmup_steps=args.num_warmup_steps
        self.init_lr = args.init_lr
        self.restore_on_train = args.restore_on_train
        self.isload = args.isload
        self.bert_config_path=args.bert_config_path
        self.bert_path=args.bert_path
        self.epochs=args.epochs
        self.rate=args.keep_rate
        self.version=args.version
    def build_graph(self,mode):
        self.add_placeholder()
        features=self.add_bert_model(self.input_id,self.mask_id,self.type_id)
        features=self.add_textcnn()
        output=self.add_dense(features)
        if mode=='train':
            self.add_loss(output)
            self.add_bert_optimizer()
        self.add_init()
    def add_placeholder(self):
        self.input_id=tf.placeholder(dtype=tf.int32,shape=[None,2*self.max_len])
        self.type_id=tf.placeholder(dtype=tf.int32,shape=[None,2*self.max_len])
        self.mask_id=tf.placeholder(dtype=tf.int32,shape=[None,2*self.max_len])
        self.input_label=tf.placeholder(dtype=tf.int32,shape=[None,])
        self.weight=tf.placeholder(dtype=tf.float32,shape=[None,])

        self.keep_rate=tf.placeholder(dtype=tf.float32, shape=[], name="keep_rate")

    def add_bert_model(self,input_id,mask_id,type_id):
        if self.version.startswith('albert'):
            bert_model=albert_modeling
            self.bert_config = bert_model.AlbertConfig.from_json_file(self.bert_config_path)
            model = bert_model.AlbertModel(
                config=self.bert_config,
                is_training=False,
                input_ids=input_id,
                input_mask=mask_id,
                token_type_ids=type_id,
                use_one_hot_embeddings=False)
        elif self.version.startswith('robert'):
            bert_model=robert_modeling
            self.bert_config = bert_model.BertConfig.from_json_file(self.bert_config_path)
            model = bert_model.BertModel(
                config=self.bert_config,
                is_training=False,
                input_ids=input_id,
                input_mask=mask_id,
                token_type_ids=type_id,
                use_one_hot_embeddings=False)
        elif self.version.startswith('base'):
            bert_model = modeling
            self.bert_config = bert_model.BertConfig.from_json_file(self.bert_config_path)
            model = bert_model.BertModel(
                config=self.bert_config,
                is_training=False,
                input_ids=input_id,
                input_mask=mask_id,
                token_type_ids=type_id,
                use_one_hot_embeddings=False)
        else:
            raise ValueError('error! version is incorrectly')
        # load_weights(init_checkpoint)
        self.bert_embeddings = model.get_sequence_output() #[batch_size, seq_length, embedding_size]
        # self.bert_embeddings = model.get_pooled_output()
        load_weights(self.bert_path,bert_model)
        return self.bert_embeddings

    def add_textcnn(self):
        filter_size_list = [1, 3, 5, 7, 9]
        num_filters = 100
        shape = self.bert_embeddings.get_shape().as_list()
        self.bert_embeddings_expanded = tf.expand_dims(self.bert_embeddings, -1, name='embeddings_chars_expanded')
        pooled_outputs = []
        for i, filter_size in enumerate(filter_size_list):
            filter_shape = [filter_size, shape[-1], 1, num_filters]  # 1,120,1,100
            # W=tf.get_variable(name='W_1',shape=filter_shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
            # b=tf.get_variable(name='b_1',shape=[num_filters],initializer=tf.zeros_initializer())
            self.W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")  # 卷积核权重
            self.b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(self.bert_embeddings_expanded, self.W, strides=[1, 1, 1, 1],
                                padding='VALID')  # batch_size,sentence_length-filter_size+1,1,num_filters
            # 非线性变换
            h = tf.nn.softplus(tf.add(conv, self.b), name='h')  # tf.nn.softplus(features, name = None)

            # pooled = tf.nn.quantized_max_pool(h, ksize=[1, 2*self.max_len - filter_size + 1, 1, 1],
            #                                   strides=[1, 1, 1, 1], padding='VALID')
            pooled = tf.nn.avg_pool(h, ksize=[1, 2*self.max_len - filter_size + 1, 1, 1],  # batch_size,1,1,filter_nums
                                    strides=[1, 1, 1, 1], padding='VALID')
            pooled_outputs.append(pooled)
        # 把所有的pooled features加起来
        self.feature_length = num_filters * len(filter_size_list)
        h_pool = tf.concat(pooled_outputs, axis=3)
        self.h_pool_flat = tf.reshape(h_pool, [-1, self.feature_length], name='h_pool_flat')

        # add dropout before softmax layer
        self.features = tf.nn.dropout(self.h_pool_flat, keep_prob=self.keep_rate, name='features')
        return self.features
    def add_dense(self,bert_embeddings,w_name='qw',b_name='qb'):
        shape=bert_embeddings.get_shape().as_list()
        W = tf.get_variable(name=w_name,
                            shape=[shape[1], 2],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            dtype=tf.float32)

        b = tf.get_variable(name=b_name,
                            shape=[2],
                            initializer=tf.zeros_initializer(),
                            dtype=tf.float32)
        bert_embeddings=tf.nn.dropout(bert_embeddings, self.keep_rate)
        self.output=tf.matmul(bert_embeddings,W)+b
        self.logits = tf.nn.softmax(self.output)
        # out_embeddings = tf.nn.tanh(out_embeddings)
        return self.output
    def add_loss(self,output):
        self.loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,labels=self.input_label)
        self.loss=tf.multiply(self.loss,self.weight)
        self.loss = tf.reduce_mean(self.loss)
    def add_bert_optimizer(self):
        self.train_op,self.learn_rate = optimization.create_optimizer(
            self.loss, self.init_lr, self.num_train_steps, self.num_warmup_steps, False)
    def add_init(self):
        self.init_op = tf.global_variables_initializer()

    def train(self, train, valid, fold_num):
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        with tf.Session(config=self.config) as sess:

            sess.run(self.init_op)
            if self.isload:
                saver.restore(sess,  tf.train.latest_checkpoint(self.export_model_path)+str(fold_num))
            # variables = tf.contrib.framework.get_variables_to_restore()
            # saver.restore(sess, args.bert_path)
            # num_batches = (len(data) + self.batch_size - 1) // self.batch_size
            train_loss=[]
            epoch_lis=[]
            train_f1_score=[]
            train_p=[]
            train_r=[]

            eval_loss=[]
            eval_f1_score=[]
            eval_p,eval_r=[],[]
            best_f1score=-1
            for epoch in range(self.epochs):
                generator_batch = batch_yield(train, batch_size=self.batch_size, shuffle=True,
                                              tokenize=self.tokenize,
                                              max_len=self.max_len)
                loss_epoch = 0
                pred_labels = []
                input_label = []
                for steps,(input_ids, mask_ids, type_ids, labels,seq_ids) in enumerate(generator_batch):
                    # if steps==0:args.learn_rate=1e-7
                    # if steps==1:args.learn_rate=3e-5
                    feed_dict = {
                        self.input_id: input_ids,
                        self.mask_id: mask_ids,
                        self.type_id: type_ids,
                        self.keep_rate: self.rate,
                        self.input_label:labels,
                        self.weight:[i+0.5 if i==1 else 1 for i in labels]
                        # self.is_train: 1,
                    }
                    _, loss_train, logits,learn_rate = sess.run([self.train_op, self.loss, self.logits,self.learn_rate],feed_dict=feed_dict)
                    pred_label=np.argmax(logits,1)
                    pred_labels.extend(list(pred_label))
                    input_label.extend(labels)
                    loss_epoch += loss_train
                    if steps % 100 == 0:
                        self.loginfo.info('steps/epoch: {}/{} loss_train:{:.4}'.format(steps,epoch + 1, loss_epoch / (steps + 1)))
                        print('learn_rate:',learn_rate)
                epoch_lis.append(epoch+1)
                tf1score,precisions,recalls = get_score(pred_labels, input_label)
                train_loss.append(loss_epoch / (steps + 1))
                train_f1_score.append(tf1score)
                train_p.append(precisions)
                train_r.append(recalls)
                # self.loginfo.info('input_label:{}'.format(input_label[:10]))
                self.loginfo.info('train: epoch: {} tf1_score: {} train_loss: {}'.format(epoch + 1, tf1score,
                                                                                          loss_epoch / (steps + 1)))
                logits_lis,pred_labels,  vf1score, vprecision,vrecall,loss_avg = self.evaluate(sess=sess, data=valid)
                eval_loss.append(loss_avg)#用于后面画图
                eval_f1_score.append(vf1score)
                eval_p.append(vprecision)
                eval_r.append(vrecall)
                self.loginfo.info(
                    'valid: epoch: {} vf1_score: {} eval_loss: {}'.format(epoch + 1, vf1score, loss_avg))
                self.loginfo.info('#' * 100)
                if not os.path.exists(self.export_model_path+str(fold_num)): os.makedirs(self.export_model_path+str(fold_num))
                if self.restore_on_train:
                    if best_f1score<vf1score:
                        saver.save(sess, save_path=self.export_model_path+str(fold_num)+'/' + 'ckpt', global_step=epoch+1)
                        best_f1score=vf1score
                    else:
                        saver.restore(sess, tf.train.latest_checkpoint(self.export_model_path+str(fold_num)))
                else:
                    saver.save(sess, save_path=self.export_model_path + str(fold_num) + '/' + 'ckpt',
                               global_step=epoch + 1)
            #画图
            y=[train_loss,train_f1_score,train_p,train_r,eval_loss,eval_f1_score,eval_p,eval_r]
            x=[epoch_lis]*len(y)
            plot_graph(x,y)
            return best_f1score if self.restore_on_train else vf1score

    def evaluate(self, sess, data):
        generator_batch = batch_yield(data, batch_size=self.batch_size, shuffle=False, tokenize=self.tokenize,
                                       max_len=self.max_len)
        # Initialize accumulators
        input_labels = []
        pred_labels = []
        logits_lis=[]
        losses = []
        for steps,(input_ids, mask_ids, type_ids, labels,seq_ids) in enumerate(generator_batch):
            feed_dict = {
                self.input_id: input_ids,
                self.mask_id: mask_ids,
                self.type_id: type_ids,
                self.keep_rate: 1,
                self.input_label: labels,
                self.weight:[i+0.5 if i==1 else 1 for i in labels]
            }
            loss_eval, logits = sess.run([self.loss, self.logits],
                                         feed_dict=feed_dict)
            logits_lis.extend(list(logits))
            pred_label = np.argmax(logits, 1)
            pred_labels.extend(list(pred_label))
            # add batch to accumulators
            assert (len(pred_label) <= self.batch_size)
            losses.append(loss_eval)
            input_labels.extend(list(labels))
        assert (len(data) == len(pred_labels))
        try:
            score,precision,recall = get_score(pred_labels, input_labels)
        except:
            score, precision, recall = get_score(pred_labels, input_labels)
            print(score)
        loss_avg = sum(losses) / len(losses)
        return  logits_lis,pred_labels, score, precision,recall,loss_avg

    def predict(self, test,fold_num):
        tf.reset_default_graph()
        self.is_train=False
        self.build_graph(mode='test')
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            saver.restore(sess, tf.train.latest_checkpoint(self.export_model_path+str(fold_num)))
            generator_batch = batch_yield(test, batch_size=self.batch_size, shuffle=False, tokenize=self.tokenize,
                                           max_len=self.max_len)
            # Initialize accumulators
            logits_lis=[]
            for steps,(input_ids, mask_ids, type_ids, _,seq_ids) in enumerate(generator_batch):
                feed_dict = {
                    self.input_id: input_ids,
                    self.mask_id: mask_ids,
                    self.type_id: type_ids,
                    self.keep_rate: 1
                }
                logits = sess.run(self.logits,feed_dict=feed_dict)
                logits_lis.extend(list(logits))
                # add batch to accumulators
                assert (len(logits) <= self.batch_size)
            assert (len(test) == len(logits_lis))
            return  logits_lis


