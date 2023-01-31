#!/usr/bin/python
# -*- coding: utf-8 -*-

from features.feature import Feature
import numerous
import random
import tensorflow as tf
if tf.__version__ >= '2.0.0':
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
from features.feature_common import reshape_varlen_feature

class Feature(object):
    def __init__(self, op_name, ft_params):
        self.op_name = op_name
        self.ft_params = ft_params


    def transform_fn(self):
        pass


class SuperBertEmbedding(Feature):
    def __init__(self, op_name, ft_params):
        super(SuperBertEmbedding, self).__init__(op_name=op_name, ft_params=ft_params)
        self.mask_num = self.ft_params.extras['mask_num']
        self.slide_window = self.ft_params.extras['slide_window']
        self.slide_step = self.ft_params.extras['slide_step']
        self.embedding_size = self.ft_params.extras['embedding_size']
        self.sequence_length = self.ft_params.extras['sequence_length']
        self.num_blocks = self.ft_params.extras['num_blocks']
        self.num_heads = self.ft_params.extras['num_heads']
        self.filters = self.ft_params.extras['filters']
        self.pad_num=64
        self.is_train = tf.placeholder(tf.bool, name="is_train_placeholder")
        numerous.reader.ControlPlaceholder(self.is_train, training_phase=True, inference_phase=False)
        self.mask_embedings=tf.compat.v1.get_variable(name='mask_embedding',dtype=tf.float32,shape=[self.embedding_size],initializer=tf.random_normal_initializer())
    def get_multi_hot(self, idx_tensor, total_num):
        one_hot = tf.one_hot(idx_tensor, depth=total_num)
        multi_hot = tf.reduce_sum(one_hot, axis=1)
        return multi_hot

    def get_drop(self,raw, mask, drop_num):
        dim = raw.get_shape().as_list()[-1]
        seq = raw.get_shape().as_list()[-2]
        mask = tf.reshape(mask, [-1])
        raw = tf.reshape(raw, [-1, dim])
        after_drop = tf.boolean_mask(raw, mask, axis=0)
        after_drop = tf.reshape(after_drop, [-1, seq - drop_num, dim])
        return after_drop

    def get_mask_zero(self,raw, mask, mask_embeds):
        mask = tf.tile(tf.expand_dims(mask, axis=-1), [1, 1, tf.shape(raw)[-1]])
        mask = tf.cast(mask, tf.float32)
        raw = raw * mask
        mask = tf.cast(tf.equal(mask,0),tf.float32)
        mask_embeds=mask_embeds*mask
        return raw+mask_embeds
    def get_session_mask_idx(self,sess_embed,sess_mask,pos_num):
        sequence_length=sess_embed.get_shape().as_list()[1]
        pos_idx = tf.random.uniform(shape=(tf.shape(sess_embed)[0],pos_num), minval=0,
                                    maxval=sequence_length, dtype=tf.int32)
        pos_idx = tf.cast(pos_idx, tf.float32)
        bt_length = tf.reduce_sum(tf.cast(sess_mask,tf.float32),axis=1)
        bt_length = tf.tile(tf.expand_dims(bt_length, axis=1), [1, pos_num])
        pos_idx = tf.where(pos_idx < bt_length, pos_idx, tf.zeros_like(pos_idx))
        pos_idx = tf.cast(pos_idx, tf.int32)  # bt,num
        return pos_idx
    def get_session_mask(self,idx,sess_embed,sess_mask):
        sess_mask=tf.cast(sess_mask,tf.float32)
        sequence_length = sess_embed.get_shape().as_list()[1]
        pos_item_feature = tf.compat.v1.batch_gather(sess_embed, indices=idx)  # bt,num,dim
        multi_hot = self.get_multi_hot(idx_tensor=idx, total_num=sequence_length)
        mask_ = tf.cast(tf.equal(multi_hot, 0), tf.bool)
        trs_true_inputs_after_drop = self.get_mask_zero(sess_embed, mask_,mask_embeds=tf.tile(tf.reshape(self.mask_embedings,[1,1,-1]),[tf.shape(sess_embed)[0],sequence_length,1]))
        trs_true_mask_drop = self.get_mask_zero(tf.expand_dims(sess_mask, axis=-1), mask_,mask_embeds=tf.ones_like(tf.expand_dims(sess_mask, axis=-1)))
        trs_true_mask_drop = tf.squeeze(trs_true_mask_drop, axis=-1)
        return pos_item_feature,trs_true_inputs_after_drop,trs_true_mask_drop

    def get_forward_mask(self, idx, sess_embed):
        sequence_length = sess_embed.get_shape().as_list()[1]
        dim = sess_embed.get_shape().as_list()[-1]
        pos_item_feature = tf.compat.v1.batch_gather(sess_embed, indices=idx)  # bt,num,dim
        seq_mask = tf.cast(tf.sequence_mask(tf.squeeze(idx + 1, axis=1), sequence_length), tf.float32)  # bt,seq
        seq_mask = tf.cast(tf.equal(seq_mask, 0), tf.float32)
        sess_embed_after_forwardmask = tf.compat.v1.multiply(sess_embed,
                                                             tf.tile(tf.expand_dims(seq_mask, -1), [1, 1, dim]))
        return pos_item_feature, sess_embed_after_forwardmask

    def get_windows(self,slide_step,sess_embeds, sess_mask, windows_size=64):
        start = slide_step
        end= slide_step+windows_size
        sess_embeds_window = sess_embeds[:, start:end, :]
        sess_mask_window = sess_mask[:, start:end]
        return sess_embeds_window, sess_mask_window

    def random_sample(self,sess_embeds,neg_num=5):
        batch_size = tf.shape(sess_embeds)[0]
        length=sess_embeds.get_shape().as_list()[1]
        dim=sess_embeds.get_shape().as_list()[2]
        logits = tf.ones(shape=[batch_size, batch_size]) - tf.eye(batch_size) * (2 << 32 - 1)
        sampled_indices = tf.cond(
            tf.greater(batch_size, 0), #这里恒成立
            true_fn=lambda: tf.random.categorical(logits, neg_num),  # 按概率采样，有可能会采到相同的索引 bt,neg_num
            false_fn=lambda: tf.constant([], dtype=tf.int64)
        )
        sampled_item_embs = tf.gather(
            sess_embeds, indices=tf.reshape(sampled_indices, shape=[-1]), axis=0)
        return tf.reshape(sampled_item_embs,[-1,neg_num*length,dim])

    def get_mask_ml(self,sess_embeds,mask_embeds,bert_embeds):
        '''
        :param sess_embeds: 原始的session embedings bt,seq,dim
        :param mask_embeds:
        :param bert_embeds:
        :return:
        '''
        sample_items=self.random_sample(sess_embeds)
        mask_embeds=tf.concat([mask_embeds,sample_items],axis=1)
        logits=tf.reduce_sum(tf.multiply(bert_embeds, mask_embeds), -1, keep_dims=False)
        prob = tf.nn.softmax(logits, axis=0)
        ml = -tf.reduce_mean(tf.math.log(prob[0])*self.bt_mask, name='ml')
        return ml
    def bert_transform(self,session,mask):
        embedded_representation = session
        self.transform_embedings = []
        self.transform_masks = []
        self.pos_embedings = []
        self.window_embeddings = []
        self.mask_idx=[]
        self.bt_mask = []
        for i in range((self.sequence_length -self.pad_num)// self.slide_step):
            slide_step = self.slide_step*i
            sess_embeds_window, sess_mask_window = self.get_windows(slide_step, embedded_representation, mask,
                                                                    self.slide_window)
            sess_embed_after_forwardmask = []
            sess_mask_after_forwardmask = []
            tmp_mask_embedins = []
            tmp_window_embeddings = []
            tmp_mask_idx=[]
            tmp_bt_mask=[]
            for j in range(self.mask_num):
                mask_idx = self.get_session_mask_idx(sess_embeds_window, sess_mask_window, 1)
                is_pading = tf.squeeze(tf.compat.v1.batch_gather(sess_mask_window, mask_idx), axis=1)
                mask_feature, sess_win_forwardmask = self.get_forward_mask(mask_idx, sess_embeds_window)
                _, mask_win_forwardmask = self.get_forward_mask(mask_idx, tf.expand_dims(sess_mask_window,axis=-1))
                _, sess_embed, sess_mask = self.get_session_mask(mask_idx, sess_win_forwardmask, tf.squeeze(mask_win_forwardmask,axis=-1))
                sess_embed_after_forwardmask.append(sess_embed)
                sess_mask_after_forwardmask.append(sess_mask)
                tmp_mask_embedins.append(mask_feature)
                tmp_window_embeddings.append(sess_embeds_window)
                tmp_mask_idx.append(mask_idx)
                tmp_bt_mask.append(is_pading)
            self.transform_embedings.extend(sess_embed_after_forwardmask)
            self.transform_masks.extend(sess_mask_after_forwardmask)
            self.pos_embedings.extend(tmp_mask_embedins)
            self.window_embeddings.extend(tmp_window_embeddings)
            self.mask_idx.extend(tmp_mask_idx)
            self.bt_mask.extend(tmp_bt_mask)
        embed_representation = tf.concat(self.transform_embedings, axis=0)
        self.mask = tf.concat(self.transform_masks, axis=0)
        self.pos_embedings = tf.concat(self.pos_embedings, axis=0)
        self.window_embed_representation = tf.concat(self.window_embeddings,axis=0)
        self.mask_idx = tf.concat(self.mask_idx,axis=0)
        self.bt_mask = tf.concat(self.bt_mask,axis=0)
        return embed_representation

    def get_bert_ouput(self,embedded_representation):
        repeat_num=((self.sequence_length -self.pad_num)// self.slide_step)*self.mask_num
        outputs = tf.reshape(embedded_representation,
                             [repeat_num+1,-1, self.slide_window, self.embedding_size])
        bert_ouputs=tf.reshape(outputs[:-1],[-1,self.slide_window,self.embedding_size])
        raw_output=outputs[-1]
        raw_outputs = tf.squeeze(tf.gather(raw_output, [0], axis=1), axis=1)
        bert_ouputs=tf.squeeze(tf.compat.v1.batch_gather(bert_ouputs,indices=self.mask_idx),axis=1)
        ml = self.get_mask_ml(self.window_embed_representation[:, :2, :], self.pos_embedings, bert_ouputs)
        return [raw_outputs,ml]


    def transform_fn(self):
        feat_dict = self.ft_params.feat_dict

        session = feat_dict[str(self.ft_params.inputs)]

        session = reshape_varlen_feature(session, self.embedding_size, self.sequence_length)

        self.mask = tf.cast(tf.reduce_any(tf.not_equal(session, 0.0), axis=-1), tf.float32)
        #添加mask 字符 embeding
        asession = tf.concat([tf.tile(tf.reshape(self.mask_embedings,[1,1,self.embedding_size]),[tf.shape(session)[0],1,1]),session],axis=1)
        embedded_representation=tf.cond(self.is_train,lambda: tf.concat([self.bert_transform(session,self.mask),asession[:,:self.slide_window,:]],axis=0),lambda :asession)
        with tf.name_scope("transformer"):
            for i in range(self.num_blocks):
                with tf.name_scope("transformer-{}".format(i + 1)):
                    with tf.name_scope("multi_head_atten"):
                        multihead_atten = self._multihead_attention(inputs=self.mask,
                                                                    queries=embedded_representation,
                                                                    keys=embedded_representation)
                    with tf.name_scope("feed_forward"):
                        embedded_representation = self._feed_forward(multihead_atten,
                                                                     [self.filters,
                                                                      self.embedding_size])

        outputs,ml=tf.cond(self.is_train,lambda :self.get_bert_ouput(embedded_representation),lambda :[tf.squeeze(tf.gather(embedded_representation, [0], axis=1), axis=1),tf.constant(0.0)])

        return outputs,ml

    def _layer_normalization(self, inputs):
        """
        对最后维度的结果做归一化，也就是说对每个样本每个时间步输出的向量做归一化
        :param inputs:
        :return:
        """
        epsilon = 1e-8

        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)

        beta = tf.Variable(tf.zeros(params_shape), dtype=tf.float32)

        gamma = tf.Variable(tf.ones(params_shape), dtype=tf.float32)
        normalized = (inputs - mean) / ((variance + float(epsilon)) ** .5)

        outputs = gamma * normalized + beta

        return outputs

    def _multihead_attention(self, inputs, queries, keys, num_units=None):
        num_heads = self.num_heads  # multi head 的头数

        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]

        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)

        Q_ = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=-1), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=-1), axis=0)

        similarity = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

        similarity = similarity / (K_.get_shape().as_list()[-1] ** 0.5)

        mask = tf.tile(inputs, [num_heads, 1])

        key_masks = tf.tile(tf.expand_dims(mask, 1), [1, tf.shape(queries)[1], 1])

        paddings = tf.ones_like(similarity) * (-2 ** 32 + 1)

        masked_similarity = tf.where(tf.equal(key_masks, 0), paddings,
                                     similarity)
        weights = tf.nn.softmax(masked_similarity)

        query_masks = tf.tile(tf.expand_dims(mask, -1), [1, 1, tf.shape(keys)[1]])
        mask_weights = tf.where(tf.equal(query_masks, 0), paddings,
                                weights)

        outputs = tf.matmul(mask_weights, V_)

        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

        outputs = tf.nn.dropout(outputs, self.ft_params.params.dropout, seed=1024)

        outputs += queries
        outputs = self._layer_normalization(outputs)
        return outputs

    def _feed_forward(self, inputs, filters):
        params = {"inputs": inputs, "filters": filters[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        params = {"inputs": outputs, "filters": filters[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}

        outputs = tf.layers.conv1d(**params)

        outputs += inputs

        outputs = self._layer_normalization(outputs)

        return outputs

if __name__ == '__main__':
    args={}
    bert=SuperBertEmbedding(args)
    #session 传入transform_fn，1、窗口滑动 2、获取mask_idx,3、forward_mask(mask_idx前面（含自己）全部置0) 3、在mask_idx位置set mask 字符 的embeding 4、过transfromer
    # 5、拿到对应mask_idx的mask embeding 注意：输入到transformer的session embeding 在batch_size的维度上，是加上了原始的session_embeding
    #session embeding 在 0的位置添加 mask 字符 embeding,过transformer后获取0位置的embeding作为user_embeding
    outputs,ml=bert.transform_fn() # ml 就是训练bert 得到的loss

    pred = tf.constant([[1.0, 2, 3], [-1, 0, 2]])