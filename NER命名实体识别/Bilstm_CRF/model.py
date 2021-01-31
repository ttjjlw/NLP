import os
import sys
import time
import copy

import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.rnn import LSTMCell

from data import pad_sequences, batch_yield
from eval import conlleval
from utils import get_logger


class BiLSTM_CRF(object):
    def __init__(self, args, embeddings, tag2label, vocab, paths, config):
        self.mode=args.mode
        self.negative_label=args.negative_label
        self.iob2iobes=args.iob2iobes
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.hidden_dim = args.hidden_dim
        self.embeddings = embeddings
        self.CRF = args.CRF
        self.update_embedding = args.update_embedding
        self.dropout_keep_prob = args.dropout
        self.optimizer = args.optimizer
        self.lr = args.lr
        self.clip_grad = args.clip
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab
        self.shuffle = args.shuffle
        self.model_path = paths['model_path']
        self.check_path=paths['check_path']
        self.summary_path = paths['summary_path']
        self.logger = get_logger(paths['log_path'])
        self.result_path = paths['result_path']
        if args.mode=='train':self.train_path=paths['train_path']
        if args.mode!='demo':self.test_path=paths['test_path']
        self.config = config
        self.f1_list_v=[]
        self.f1_list_t=[]

    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer_op()
        self.biLSTM_layer_op()
        self.softmax_pred_op()
        self.loss_op()
        if self.mode=='train':
            self.trainstep_op()
        self.init_op()

    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def lookup_layer_op(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings,
                                           dtype=tf.float32,
                                           trainable=self.update_embedding,
                                           name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.word_ids,
                                                     name="word_embeddings")
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)

    def biLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)

            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)

        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2 * self.hidden_dim])
            pred = tf.matmul(output, W) + b

            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])

    def loss_op(self):
        if self.CRF:
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                        tag_indices=self.labels,
                                                                        sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood)

        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        tf.summary.scalar("loss", self.loss)

    def softmax_pred_op(self):
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)

    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def init_op(self):
        self.init_op = tf.global_variables_initializer()

    def add_summary(self, sess):
        """

        :param sess:
        :return:
        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def train(self, train, dev,model_path=None):
        """

        :param train:
        :param dev:
        :return:
        """
        train_no_shuffle=copy.deepcopy(train)
        vl = [v for v in tf.global_variables() if "Adam" not in v.name]
        saver = tf.train.Saver(vl)

        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            self.add_summary(sess)

            for epoch in range(self.epoch_num):
                model_path=self.run_one_epoch(sess, train_no_shuffle,train, dev, self.tag2label, epoch, saver,model_path=model_path)

    def test(self, test):
        vl = [v for v in tf.global_variables() if "Adam" not in v.name]
        saver = tf.train.Saver(vl)
        with tf.Session(config=self.config) as sess:
            self.logger.info('=========== testing ===========')
            saver.restore(sess, self.model_path)
            label_list, seq_len_list = self.dev_one_epoch(sess, test)
            self.evaluate(label_list, seq_len_list, test,mode=self.mode)

    def demo_one(self, sess, sent):
        """

        :param sess:
        :param sent: 
        :return:
        """
        from utils import iobes_iob
        label_list = []
        for seqs, labels in batch_yield(sent, self.batch_size, self.vocab, self.tag2label, shuffle=False,iob2iobes=self.iob2iobes):
            label_list_, _ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag
        tag = [label2tag[label] for label in label_list[0]]
        if self.iob2iobes:tag=iobes_iob(tag)
        return tag

    def run_one_epoch(self, sess, train_no_shuffle,train, dev, tag2label, epoch, saver,model_path):
        """
        :param sess:
        :param train:
        :param dev:
        :param tag2label:
        :param epoch:
        :param saver:
        :return:
        """
        if model_path:
            saver.restore(sess,model_path)
            print("载入模型:{}".format(model_path))
            if epoch==0:
                label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)  # dev 验证集
                f1score_valid = self.evaluate(label_list_dev, seq_len_list_dev, dev,self.test_path, epoch,'valid')
                label_list_train, seq_len_list_train = self.dev_one_epoch(sess, train)
                f1score_train = self.evaluate(label_list_train, seq_len_list_train, train, self.train_path, epoch,'train')
                self.f1_list_v.append(f1score_valid)
                self.f1_list_t.append(f1score_train)
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size

        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        batches = batch_yield(train, self.batch_size, self.vocab, self.tag2label, shuffle=self.shuffle,iob2iobes=self.iob2iobes)
        for step, (seqs, labels) in enumerate(batches):

            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, labels, self.lr, self.dropout_keep_prob)
            _, loss_train, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                         feed_dict=feed_dict)
            print_steps=self.batch_size
            if self.batch_size<=10:print_steps=self.batch_size*10#每隔print_steps步打印一次
            if step + 1 == 1 or (step + 1) % print_steps == 0 or step + 1 == num_batches:
                print('logger info')
                print(
                    '{} epoch {}, step {}/{}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1,num_batches,
                                                                                loss_train, step_num))

            self.file_writer.add_summary(summary, step_num)

            # if step + 1 == num_batches:
            #     saver.save(sess, self.model_path, global_step=step_num)

        self.logger.info('===========validation / test===========')
        # dev [([train],[label])]
        label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev) #dev 验证集
        f1score_valid = self.evaluate(label_list_dev, seq_len_list_dev, dev, self.test_path, epoch,'valid')
        label_list_train,seq_len_list_train=self.dev_one_epoch(sess, train_no_shuffle)
        f1score_train=self.evaluate(label_list_train, seq_len_list_train, train_no_shuffle,self.train_path, epoch,'train')

        if len(self.f1_list_v)>=1:
            if len(self.f1_list_v) - self.f1_list_v.index(max(self.f1_list_v)) - 1 > 15 or self.lr<0.0001:
                self.logger.info('=========训练:结束==========')
                self.logger.info('最终保存最新的模型为{}'.format(self.model_path))
                exit()
            if f1score_valid<max(self.f1_list_v) and f1score_train>max(self.f1_list_t)-0.01:
                if self.lr>0.00002:self.lr=self.lr*0.95
                if self.batch_size<=15:self.batch_size*=2
                self.model_path = tf.train.latest_checkpoint(self.check_path)
                self.logger.info("epoch: {} lr: {} self.batch_size: {}".format(epoch,self.lr,self.batch_size))
                print('最新加载的模型路径{}'.format(self.model_path))
                model_path=self.model_path
                return model_path
        save_path=saver.save(sess, self.check_path+'model.ckpt', global_step=epoch+1)
        self.f1_list_v.append(f1score_valid)
        self.f1_list_t.append(f1score_train)
        print('保存的模型路径{}'.format(save_path))

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        """

        :param seqs:
        :param labels:在生成batch的过程中，seqs,和label发生了2id的步骤
        :param lr:
        :param dropout:
        :return: feed_dict
        """
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)

        feed_dict = {self.word_ids: word_ids,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_list

    def dev_one_epoch(self, sess, dev):
        """

        :param sess:
        :param dev:
        :return:
        """
        label_list, seq_len_list = [], []
        for seqs, labels in batch_yield(dev, self.batch_size, self.vocab, self.tag2label, shuffle=False,iob2iobes=self.iob2iobes):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def predict_one_batch(self, sess, seqs):
        """

        :param sess:
        :param seqs:
        :return: label_list
                 seq_len_list
        """
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)

        if self.CRF:
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list, seq_len_list

        else:
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            return label_list, seq_len_list

    def evaluate(self, label_list, seq_len_list, data,raw_data=None,epoch=None,mode='train'):
        """

        :param label_list: pading后的句子的长度预测出来的标签，标签的长度=句子不含pading部分的长度
        :param seq_len_list:
        :param data: 这里的data 没有pading
        :param epoch:
        :return:
        """
        label2tag = {}
        for tag, label in self.tag2label.items():
            # label2tag[label] = tag if label != 0 else label  为什么要把"O" 换成"0"
            label2tag[label]=tag

        model_predict = []
        for label_, (sent, tag) in zip(label_list, data):
            tag_ = [label2tag[label__] for label__ in label_]
            sent_res = []
            if len(label_) != len(sent):
                print(sent)
                print(len(label_))
                print(tag)
            for i in range(len(sent)):
                sent_res.append([sent[i], tag[i], tag_[i]])
            model_predict.append(sent_res)
        result_file=os.path.join(self.result_path, 'result_file')
        f1score,precision,recall = conlleval(mode,model_predict, result_file, self.negative_label,self.iob2iobes)
        if mode=='train':
            self.logger.info('epochs: {} 训练集: recall:{}, precision:{}, f1score: {}'.format(epoch + 1, recall,precision,f1score))
        else:
            self.logger.info('epochs: {} 验证集: recall:{}, precision:{}, f1score: {}'.format(epoch + 1, recall,precision,f1score))
        return f1score
