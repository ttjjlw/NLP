#!/usr/bin/python
# -*-- coding: utf-8 -*--

import numerous
import tensorflow as tf
if tf.__version__ >= '2.0.0':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
from utils import data_util
from utils import model_util
from utils import config
from numerous.optimizers.optimizer_context import OptimizerContext
from utils import recall_model_config as recall_config
from numerous.optimizers.adam import Adam
from numerous.optimizers.ftrl import FTRL


class RankingModel(object):
    
    def __init__(self):
        
        self.conf = conf = self.get_config()
        
        # batch normalization placeholder
        is_train = data_util.get_batch_normalization_placeholder()

        # label
        self.label_dict = data_util.get_sample_label(self.conf)
        
        # sample weights
        self.label_dict['label_readtime'] = tf.multiply(self.label_dict['label_click'],
                                                        self.label_dict['label_readtime'])
        self.label_dict['label_hudong'] = tf.cast((self.label_dict['label_hudong'] + self.label_dict['label_share']) > 0.0, tf.float32)
        self.label_dict['label_hudong'] = tf.multiply(self.label_dict['label_click'],
                                                        self.label_dict['label_hudong'])

        # shared feature
        # features
        categorical_embedding_list = []
        continuous_embedding_list = []
        user_embedding_list = []
        item_embedding_list = []
        y_first_orders = []

        for feature in conf.feature_conf:
            if feature['in_use']:
                if feature['categorical']:
                    if 'embedding_size' in feature:
                        embedding_w, slots = numerous.framework.SparseEmbedding(
                            embedding_dim=feature['embedding_size'],
                            optimizer=self.dense_optimizer,
                            distribution=numerous.distributions.Uniform(left=-0.001, right=0.001),
                            slot_ids=[str(feature['slotid'])],
                            use_sparse_placeholder=True
                        )
                        embedding = tf.sparse_tensor_dense_matmul(slots, embedding_w)
                        if feature['weighted']:
                            embedding = embedding * feature['weighted degree']
                        categorical_embedding_list.append(embedding)
                        if feature['type'] == 'user':
                            user_embedding_list.append(embedding)
                        elif feature['type'] == 'item':
                            item_embedding_list.append(embedding)
                    embedding_w, slots = numerous.framework.SparseEmbedding(
                        embedding_dim=1,
                        optimizer=FTRL(lambda1=0.1, lambda2=0.1),
                        distribution=numerous.distributions.Uniform(left=-0.001, right=0.001),
                        slot_ids=[str(feature['slotid'])],
                        use_sparse_placeholder=True
                    )
                    y_first_orders.append(tf.sparse_tensor_dense_matmul(slots, embedding_w))

                else:
                    lower = feature['lower']
                    upper = feature['upper']
                    x_data = tf.placeholder(tf.float32, name="dense_placeholder_" +
                                                             str(feature['slotid']), shape=[None, (upper - lower + 1)])
                    numerous.reader.DensePlaceholder(x_data, slot_id=str(feature['slotid']), lower_bound=lower,
                                                     upper_bound=upper)
                    continuous_embedding_list.append(x_data)
                    if feature['type'] == 'user':
                        user_embedding_list.append(x_data)
                    elif feature['type'] == 'item':
                        item_embedding_list.append(x_data)

        fm_first_order_score = tf.reduce_sum(tf.concat(values=y_first_orders, axis=1), axis=1, keepdims=True)
        wkfm_1 = self._wkfm(categorical_embedding_list, reduce_sum=False, name="wkfm_1")
        wkfm_2 = self._wkfm(wkfm_1, reduce_sum=False, name="wkfm_2")
        wkfm_concat = tf.reduce_sum(tf.concat(wkfm_2, axis=1), axis=1, keepdims=True)
        fm_second_order_score = self.get_wkfm_logits(wkfm_concat, 1)

        # click
        categorical_embedding = tf.concat(values=categorical_embedding_list, axis=1)
        continuous_embedding = tf.concat(values=continuous_embedding_list, axis=1)
        dnn_embedding = tf.concat(values=[categorical_embedding, continuous_embedding], axis=1)
        user_embedding = tf.concat(user_embedding_list, axis=1)
        item_embedding = tf.concat(item_embedding_list, axis=1)
        
        # feature=======================================================================
        self.dnn_output = model_util.mlp_new(dnn_embedding,
                                             self.conf.hidden_sizes['click'],
                                             is_train,
                                             'dnn_last_hidden_layer_click')

        self.merge_layer_click = tf.concat([fm_first_order_score,
                                            fm_second_order_score,
                                            self.dnn_output], axis=1)
    
        self.score = {'click': model_util.mlp_new(self.merge_layer_click, [1], is_train, 'score_click')}
        self.pred_ctr = tf.sigmoid(self.score['click'], name='pred_ctr')
        self.task_loss = {'click': model_util.cross_entropy(self.label_dict['label_click'],
                                                            self.score['click'])}
        
        # readtime task
        dnn_embedding_stop = tf.stop_gradient(dnn_embedding)
        pred_ctr_stop = tf.stop_gradient(self.pred_ctr)
        fm_first_order_score_stop = tf.stop_gradient(fm_first_order_score)
        stack_wkfm_score_stop = tf.stop_gradient(fm_second_order_score)
        merge_embedding_size = dnn_embedding.shape[1].value + 2
        
        self.merge_embedding_stop = tf.reshape(tf.concat([fm_first_order_score_stop,
                                                          stack_wkfm_score_stop,
                                                          dnn_embedding_stop], axis=1),
                                               [-1, merge_embedding_size])
        
        # share botttom input vector
        input_vector = model_util.mlp_new(self.merge_embedding_stop,
                                          self.conf.hidden_sizes['share_bottom'],
                                          is_train,
                                          'share_bottom',
                                          last_act=True,
                                          last_bn=True)
        
        # MMOE expert dnn
        self.experts_output = model_util.get_experts_output(input_vector,
                                                            self.conf.hidden_sizes['expert_dnn'],
                                                            self.conf.num_experts,
                                                            is_train)
        
        # cvr and loss for each non-click task
        self.gate = {}
        self.pred_cvr = {}
        self.pred_ctcvr = {}
        
        for task in self.conf.task_list[1:]:
            moe_output, self.gate[task] = model_util.get_moe_output_attention(
                                                         input_vector,
                                                         self.experts_output,
                                                         task,
                                                         is_train)
            
            self.score[task] = model_util.mlp_new(moe_output,
                                                  self.conf.hidden_sizes['task_dnn'],
                                                  is_train,
                                                  'score_' + task)
        
            [self.pred_cvr[task],
             self.pred_ctcvr[task],
             self.task_loss[task]] = model_util.get_ctcvr_loss(self.label_dict['label_' + task],
                                                               self.score[task],
                                                               pred_ctr_stop)
                
        # total loss
        self.loss_teacher = self.conf.task_weight['click'] * self.task_loss['click']
        self.loss_teacher += self.conf.task_weight['readtime'] * self.task_loss['readtime']
        self.loss_teacher += self.conf.task_weight['hudong'] * self.task_loss['hudong']


        #===================================================DSSM student model==========================================
        # user vector
        self.click_user_vector = model_util.mlp_new(user_embedding,
                                              conf.user_hidden_size,
                                              is_train,
                                              'click_user_vector_output')
        self.click_user_vector_new = tf.identity(self.click_user_vector, name='click_user_vector')

        # item vector
        self.item_vector = model_util.mlp_new(item_embedding,
                                              conf.item_hidden_size,
                                              is_train,
                                              'click_item_vector_unnormalized')
        self.item_vector = tf.nn.l2_normalize(self.item_vector, axis=1, name='item_vector')

        # score and pctr
        self.click_dssm_score = tf.reduce_sum(tf.multiply(self.click_user_vector, self.item_vector),
                                        axis=1,
                                        keep_dims=True)
        self.pred_ctr_student = tf.sigmoid(self.click_dssm_score, name='pred_ctr_student')
        self.loss_student_click = model_util.pairwise_loss(self.label_dict['label_click'],self.click_dssm_score)

        self.read_user_vector = model_util.mlp_new(user_embedding,
                                              conf.user_hidden_size,
                                              is_train,
                                              'read_user_vector_output')
        self.read_user_vector_new = tf.identity(self.read_user_vector, name='read_user_vector')

        # score and pctr
        self.read_dssm_score = tf.reduce_sum(tf.multiply(self.read_user_vector, self.item_vector),
                                        axis=1,
                                        keep_dims=True)
        self.pred_ctcvr_student = tf.sigmoid(self.read_dssm_score, name='pred_ctcvr_student')

        # student loss for readtime
        self.pred_ctcvr_teacher = self.pred_ctcvr['readtime']
        self.pred_ctcvr_student = tf.clip_by_value(self.pred_ctcvr_student, 0.001, 0.999)
        self.pred_ctcvr_teacher = tf.clip_by_value(self.pred_ctcvr_teacher, 0.001, 0.999)
        self.pred_ctcvr_teacher_stop = tf.stop_gradient(self.pred_ctcvr_teacher)
        self.loss_student = self.pred_ctcvr_teacher_stop * tf.log(
            tf.math.divide(self.pred_ctcvr_teacher_stop, self.pred_ctcvr_student))
        self.loss_student += (1 - self.pred_ctcvr_teacher_stop) * tf.log(
            tf.math.divide(1 - self.pred_ctcvr_teacher_stop, 1 - self.pred_ctcvr_student))
        self.loss_student_read = tf.reduce_mean(self.loss_student)

        self.hudong_user_vector = model_util.mlp_new(user_embedding,
                                                   conf.user_hidden_size,
                                                   is_train,
                                                   'hudong_user_vector_output')
        self.hudong_user_vector_new = tf.identity(self.hudong_user_vector, name='hudong_user_vector')

        # score and pctr
        self.hudong_dssm_score = tf.reduce_sum(tf.multiply(self.hudong_user_vector, self.item_vector),
                                             axis=1,
                                             keep_dims=True)
        self.pred_ctcvr_student1 = tf.sigmoid(self.hudong_dssm_score, name='pred_ctcvr_student1')

        # student loss for hudong
        self.pred_ctcvr_teacher1 = self.pred_ctcvr['hudong']
        self.pred_ctcvr_student1 = tf.clip_by_value(self.pred_ctcvr_student1, 0.001, 0.999)
        self.pred_ctcvr_teacher1 = tf.clip_by_value(self.pred_ctcvr_teacher1, 0.001, 0.999)
        self.pred_ctcvr_teacher_stop1 = tf.stop_gradient(self.pred_ctcvr_teacher1)
        self.loss_student1 = self.pred_ctcvr_teacher_stop1 * tf.log(
            tf.math.divide(self.pred_ctcvr_teacher_stop1, self.pred_ctcvr_student1))
        self.loss_student1 += (1 - self.pred_ctcvr_teacher_stop1) * tf.log(
            tf.math.divide(1 - self.pred_ctcvr_teacher_stop1, 1 - self.pred_ctcvr_student1))
        self.loss_student_hudong = tf.reduce_mean(self.loss_student1)

        self.task_scores = tf.stop_gradient(tf.concat([self.click_dssm_score , self.read_dssm_score , self.hudong_dssm_score],axis=-1))

        # user_type_learning
        self.user_type_vector = model_util.mlp_new(tf.stop_gradient(user_embedding),
                                              [512, 256, 128,3],
                                              is_train,
                                              'user_type_output')
        self.fusion_score_stu = tf.multiply(self.task_scores, self.user_type_vector)
        self.fusion_score_1_stu = tf.reduce_sum(self.fusion_score_stu, 1, keepdims=True)
        self.final_score_2_stu = tf.sigmoid(self.fusion_score_1_stu, name='final_score_2_stu')

        sample_weights = self.label_dict['label_click'] + self.label_dict['label_readtime'] + 3 * self.label_dict['label_hudong']
        sample_weights1 = sample_weights + 1.0 - tf.to_float(sample_weights > 0)

        # fusion loss
        self.loss_fusion_1 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.fusion_score_1_stu, labels=self.label_dict['label_click']
            ) * sample_weights1)

        # user_type_learning1
        self.user_type_vector1 = model_util.mlp_new(tf.stop_gradient(user_embedding),
                                              [512, 256, 128,3],
                                              is_train,
                                              'user_type_output')
        self.fusion_score_stu1 = tf.multiply(self.task_scores, self.user_type_vector1)
        self.fusion_score_1_stu1 = tf.reduce_sum(self.fusion_score_stu1, 1, keepdims=True)
        self.final_score_2_stu1 = tf.sigmoid(self.fusion_score_1_stu1, name='final_score_2_stu1')
        self.pairwise_click_loss = model_util.pairwise_loss_v2(self.fusion_score_1_stu1,
                                                               sample_weights)

        # total loss
        self.loss = self.loss_teacher + self.loss_student_click + self.loss_student_read \
                    + self.loss_student_hudong + self.loss_fusion_1 + self.pairwise_click_loss

        # final op for ctr plus ctcvr
        self.final_user_op1 = tf.concat([self.click_user_vector, self.read_user_vector , self.hudong_user_vector], axis=-1, name='final_user_op1')
        self.final_item_op = tf.concat([self.item_vector, self.item_vector , self.item_vector], axis=-1,
                                        name='final_item_op')
        self.final_score1 = tf.reduce_sum(tf.multiply(self.final_user_op1, self.final_item_op),
                                        axis=1,
                                        keep_dims=True)

        # final op for ctcvr
        self.final_user_op2 = tf.concat([tf.zeros_like(self.read_user_vector), self.read_user_vector,tf.zeros_like(self.read_user_vector)],
                                        axis=-1, name='final_user_op2')
        self.final_score2 = tf.reduce_sum(tf.multiply(self.final_user_op2, self.final_item_op),
                                        axis=1,
                                        keep_dims=True)

        # final op for fusion loss
        self.final_user_op3 = tf.concat([self.click_user_vector * tf.slice(self.user_type_vector, [0, 0], [-1, 1]),
                                         self.read_user_vector * tf.slice(self.user_type_vector, [0, 1], [-1, 1]),
                                         self.hudong_user_vector * tf.slice(self.user_type_vector, [0, 2], [-1, 1])], axis=-1,
                                   name='final_user_op3')
        self.final_score3 = tf.reduce_sum(tf.multiply(self.final_user_op3, self.final_item_op),
                                        axis=1,
                                        keep_dims=True)

        # final op for pairwise loss
        self.final_user_op4 = tf.concat([self.click_user_vector * tf.slice(self.user_type_vector1, [0, 0], [-1, 1]),
                                         self.read_user_vector * tf.slice(self.user_type_vector1, [0, 1], [-1, 1]),
                                         self.hudong_user_vector * tf.slice(self.user_type_vector1, [0, 2], [-1, 1])], axis=-1,
                                   name='final_user_op4')
        self.final_score4 = tf.reduce_sum(tf.multiply(self.final_user_op4, self.final_item_op),
                                        axis=1,
                                        keep_dims=True)

        # final op for rule
        stat_hudong_month = tf.placeholder(tf.float32)
        numerous.reader.DensePlaceholder(stat_hudong_month, "5603", 24064701759488, 24064701759537)
        self.thumb_up_doc_num = tf.to_int64(tf.round(tf.slice(stat_hudong_month, [0, 9], [-1, 1]) * 100))
        self.thumb_up_bool = tf.to_float(self.thumb_up_doc_num > 0)
        self.thumb_up_more_bool = tf.to_float(self.thumb_up_doc_num > 12)
        clk_doc_num_month = tf.to_float(tf.round(tf.slice(stat_hudong_month, [0, 0], [-1, 1]) * 100))
        self.clk_doc_num_month = clk_doc_num_month
        user_readtime_month = tf.to_float(tf.round(tf.slice(stat_hudong_month, [0, 4], [-1, 1]) * 100))
        self.user_readtime_month = user_readtime_month

        age_one_hot = tf.placeholder(tf.float32)
        numerous.reader.DensePlaceholder(age_one_hot, "5610", 24094766530560, 24094766530566)
        age_12_15_flag = tf.to_float(tf.slice(age_one_hot, [0, 1], [-1, 1]) > 0)
        self.age_12_15_flag = age_12_15_flag
        self.user_avg_readtime_2 = user_readtime_month / (self.clk_doc_num_month + 1e-7)
        self.readtime_group_1 = tf.to_float(self.user_avg_readtime_2 > 80)
        self.hudong_group_1 = tf.to_float((self.thumb_up_bool * self.age_12_15_flag + self.thumb_up_more_bool) > 0)

        self.final_user_op5 = tf.concat([self.click_user_vector,
                                         tf.divide(self.user_avg_readtime_2, 100) * self.read_user_vector,
                                         self.hudong_group_1 * self.hudong_user_vector], axis=-1, name='final_user_op5')
        self.final_score5 = tf.reduce_sum(tf.multiply(self.final_user_op5, self.final_item_op),
                                        axis=1,
                                        keep_dims=True)

        # loss and auc for print
        self.metric_list = self.get_metric_list()
        self.print_hook = self.get_print_hook()
        
        

    # ------------ set config for the model
    def get_config(self):
        
        conf = config.Config()
        
        # training parameters config
        conf.task_list = ['click', 'readtime','hudong']
        conf.hidden_sizes['click'] = [1024, 512, 128]
        conf.num_experts = 4
        
        # optimizer config
        self.dense_optimizer = Adam(rho1=0.9, rho2=0.999, eps=conf.step_size)
        self.sparse_optimizer = FTRL(lambda1=0.1, lambda2=0.1)
        self.zero_initializer = tf.zeros_initializer()
        OptimizerContext().set_optimizer(optimizer=self.dense_optimizer)
        
        # features config
        # data_util.remove_interation_features(conf)
        return conf

        
    # get metric list for print
    def get_metric_list(self):
        
        metric_list = [numerous.metrics.DistributeMean(self.loss, name="loss")]
        metric_list.append(numerous.metrics.DistributeAuc(predict=self.pred_ctr,
                                                          label=self.label_dict['label_click'],
                                                          name="auc_ctr_teacher"))
        metric_list.append(numerous.metrics.DistributeAuc(predict=self.pred_ctcvr['readtime'],
                                                          label=self.label_dict['label_readtime'],
                                                          name="auc_ctcvr_teacher"))
        metric_list.append(numerous.metrics.DistributeAuc(predict=self.pred_ctcvr['hudong'],
                                                          label=self.label_dict['label_hudong'],
                                                          name="auc_hudong_teacher"))

        metric_list.append(numerous.metrics.DistributeAuc(predict=self.pred_ctr_student,
                                                          label=self.label_dict['label_click'],
                                                          name="auc_ctr_student"))
        metric_list.append(numerous.metrics.DistributeAuc(predict=self.pred_ctcvr_student,
                                                          label=self.label_dict['label_readtime'],
                                                          name="auc_ctcvr_student"))
        metric_list.append(numerous.metrics.DistributeAuc(predict=self.pred_ctcvr_student1,
                                                          label=self.label_dict['label_hudong'],
                                                          name="auc_hudong_student"))

        metric_list.append(numerous.metrics.DistributeAuc(predict=tf.sigmoid(self.final_score3),
                                                          label=self.label_dict['label_click'],
                                                          name="auc_ctr_student_fusion"))
        metric_list.append(numerous.metrics.DistributeAuc(predict=tf.sigmoid(self.final_score3),
                                                          label=self.label_dict['label_readtime'],
                                                          name="auc_ctcvr_student_fusion"))
        metric_list.append(numerous.metrics.DistributeAuc(predict=tf.sigmoid(self.final_score3),
                                                          label=self.label_dict['label_hudong'],
                                                          name="auc_hudong_student_fusion"))

        metric_list.append(numerous.metrics.DistributeAuc(predict=tf.sigmoid(self.final_score4),
                                                          label=self.label_dict['label_click'],
                                                          name="auc_ctr_student_pairwise"))
        metric_list.append(numerous.metrics.DistributeAuc(predict=tf.sigmoid(self.final_score4),
                                                          label=self.label_dict['label_readtime'],
                                                          name="auc_ctcvr_student_pairwise"))
        metric_list.append(numerous.metrics.DistributeAuc(predict=tf.sigmoid(self.final_score4),
                                                          label=self.label_dict['label_hudong'],
                                                          name="auc_hudong_student_pairwise"))


        return metric_list
    
    
    # get metric hook for print
    def get_print_hook(self):

        print_hook = {
            'label_click': self.label_dict['label_click'][:1],
            'label_readtime': self.label_dict['label_readtime'][:1],
            'click_teacher_score':self.score['click'][:1],
            'click_teacher_score1': self.score['click'][127:],
            'hudong_teacher_score': self.score['hudong'][:1],
            'hudong_teacher_score1': self.score['hudong'][127:],
            'readtime_teacher_score':self.score['readtime'][:1],
            'readtime_teacher_score1':self.score['readtime'][127:],
            'click_dssm_score': self.click_dssm_score[:1],
            'click_dssm_score1':self.click_dssm_score[127:],
            'read_dssm_score': self.read_dssm_score[:1],
            'read_dssm_score1':self.read_dssm_score[127:],
            'hudong_dssm_score': self.hudong_dssm_score[:1],
            'hudong_dssm_score1': self.hudong_dssm_score[127:],
            'weighted_user_type_vector':self.user_type_vector[:1],
            'weighted_user_type_vector1':self.user_type_vector[127:],
            'pairloss_user_type_vector': self.user_type_vector1[:1],
            'pairloss_user_type_vector1': self.user_type_vector1[127:]
        }
        return print_hook

    def get_wkfm_logits(self, x, units):
        with tf.variable_scope('wkfm'):
            # x = self._wkfm(feature_vectors, embedding_size_list, reduce_sum=True)  # [B, 1]
            with tf.variable_scope('logits') as logits_scope:
                # fc just for adding a bias
                w = tf.get_variable("logits_w", shape=[1, units], dtype=tf.float32)
                b = tf.get_variable("logits_b", shape=[units], dtype=tf.float32, initializer=self.zero_initializer)
                logits = tf.add(tf.matmul(x, w), b)
            return logits

    def _wkfm(self, feature_vectors, reduce_sum=True, name="wkfm"):
        """Kernel FM
          feature_vectors: List of shape [B, ?] tensors, size N

        Half-Optimized implementation

        Return:
          Tensor of shape [B, T] if reduce_sum is True, or shape [B, 1], T is the sum
          dimentions of all features.
        """

        with tf.variable_scope(name):
            outputs = []
            x = tf.concat(feature_vectors, axis=1)   # [B, T]
            T = x.shape[1].value
            N = len(feature_vectors)
            indices = []
            for j in range(N):
                vj = feature_vectors[j]
                dj = vj.shape[1].value
                indices.extend([j] * dj)
            for i in range(N):
                vi = feature_vectors[i]
                name = 'wkfm_{}'.format(i)
                di = vi.shape[1].value
                U = tf.get_variable(name, [T, di])  # [T, di]
                name = 'wkfm_weightes_{}'.format(i)
                wkfm_weights = tf.get_variable(name, [N], initializer=tf.ones_initializer)
                weights = tf.gather(wkfm_weights, indices)
                y = tf.matmul(weights * x, U)   # [B, di]
                y = vi * y
                y = tf.reshape(y, [-1, di])
                outputs.append(y)

            # y = tf.concat(outputs, axis=1)   # [B, T]
            # y = x * y  # [B, T]
            y = outputs
            if reduce_sum:
                y = tf.concat(y, axis=1)
                y = tf.reduce_sum(y, axis=1, keepdims=True)  # [B, 1]
            return y

