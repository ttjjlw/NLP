#!/usr/bin/env python
#  ------------------------------------------------------------------------------------------------
#  Descripe: 
#  Auther: jialiangtu 
#  CopyRight: Tencent Company
# ------------------------------------------------------------------------------------------------
import tensorflow as tf

def dot(a, b):
    return tf.reduce_sum(tf.multiply(a, b), -1, keep_dims=False)


def cosin(a, b):
    a_b = dot(a, b)
    a_norm = tf.sqrt(tf.reduce_sum(tf.square(a), axis=-1))
    b_norm = tf.sqrt(tf.reduce_sum(tf.square(b), axis=-1))
    return tf.div(a_b, tf.multiply(a_norm, b_norm))
def bpr_loss(a,b,weight,l2_normal=False, cos=False):
    if not cos:
        if l2_normal:
            a = tf.nn.l2_normalize(a, axis=-1)
            b = tf.nn.l2_normalize(b, axis=-1)
        logits = dot(a, b)
    else:
        logits = cosin(a, b)
    pos=tf.expand_dims(logits[0],axis=0)
    loss=-tf.reduce_mean(tf.log(tf.nn.sigmoid(pos-logits[1:])),axis=0) #40,bt->bt
    loss=tf.reduce_mean(loss*weight)
    return loss


def pair_wise_loss_mask(a, b, pos_weight, neg_weight, mask=tf.constant(1.0), l2_normal=False, cos=False):
    if not cos:
        if l2_normal:
            a = tf.nn.l2_normalize(a, axis=-1)
            b = tf.nn.l2_normalize(b, axis=-1)
        scores = tf.reduce_sum(tf.multiply(a, b), -1, keep_dims=False)
    else:
        scores = dot(a, b)
    # 41 x B
    #scores = tf.reduce_sum(tf.multiply(a, b), axis=-1, keep_dims=False)
    # pos - neg: 40 x B
    pos_sub_neg_scores = tf.expand_dims(scores[0], axis=0) - scores[1:]
    # pos - neg: B x 40
    pos_sub_neg_scores = tf.transpose(pos_sub_neg_scores, [1, 0])

    # 融合正样本和负样本权重 B x 40: pos_weight (B,) neg_weights (B, 40)
    weights = tf.expand_dims(pos_weight, 1) + neg_weight

    prob = tf.sigmoid(pos_sub_neg_scores)
    prob_cliped = tf.clip_by_value(prob, 1e-30, 1.0 - 1e-7)
    loglikelyhood = tf.log(prob_cliped)

    loss = -tf.reduce_sum(tf.expand_dims(mask, 1) * loglikelyhood * weights)
    loss = loss / (tf.reduce_sum(mask) + 1)
    return loss


def pairwise_loss(labels, scores):
    score_diff = scores - tf.transpose(scores)
    prob = tf.sigmoid(score_diff)
    prob_cliped = tf.clip_by_value(prob, 1e-30, 1.0-1e-7)
    loglikelyhood = tf.log(prob_cliped)
    mask = tf.to_float((labels - tf.transpose(labels)) > 0)
    nonzero_num = tf.reduce_sum(labels) * tf.reduce_sum(1-labels) + 1.0
    loss = (-1.0) * tf.reduce_sum(tf.multiply(loglikelyhood, mask)) / nonzero_num
    return loss

def pairwise_inbatch_loss(score,
                     label,
                     gain_fn=lambda label: tf.pow(2.0, label) - 1):
  score_diff = score - tf.transpose(score)  # b,1-1,b -> BxB
  prob = tf.sigmoid(score_diff)  # BxB
  prob_cliped = tf.clip_by_value(prob, 1e-20, 1.0 - 1e-7)
  loglikelyhood = tf.log(prob_cliped)

  gain = gain_fn(label)  # Bx1
  pair_gain = tf.abs(gain - tf.transpose(gain))

  pairwise_label = tf.cast((label - tf.transpose(label)) > 0, tf.float32)
  nonzero_num = tf.reduce_sum(pairwise_label) + 1.0 #标量

  loss = -1.0 * tf.reduce_sum(
      loglikelyhood * pair_gain * pairwise_label) / nonzero_num
  return loss

def shape_loss(a, b,key_index,pos_index, l2_normal=False, cos=False):
    '''
    :param a: key feature shape 为 bt,dim
    :param b: 包含key feature shape 为 neg_num,bt,dim
    :param key_index: 标识 key feature 的位置 shape 为 (bt,)
    :param pos_index: 标识 正向 feature 的位置 shape 为 (bt,)
    :param l2_normal:
    :param cos: 是否用余弦
    :return:
    '''
    #8,bt,dim
    #key_index bt
    if not cos:
        if l2_normal:
            a = tf.nn.l2_normalize(a, axis=-1)
            b = tf.nn.l2_normalize(b, axis=-1)
        logits = 5 * dot(a, b) #8,bt
    else:
        logits = 5 * cosin(a, b)

    logits=tf.transpose(logits,[1,0]) #bt ,8
    key_index=tf.expand_dims(key_index,axis=1)
    mask=key_index-tf.constant(list(range(8))) #bt,8
    mask=tf.not_equal(mask,0)
    paddings = tf.ones_like(logits) * (-2 ** 32 + 1)
    new_logits=tf.where(mask,logits,paddings)

    prob_expose = tf.nn.softmax(new_logits, axis=1)
    # weight = tf.math.log(weight)+1
    pos_logits=tf.gather(prob_expose,pos_index)
    loss = -tf.reduce_mean(tf.math.log(pos_logits), name='loss')

    # hit_prob_expose = prob_expose[0]
    # loss = -tf.reduce_mean(tf.math.log(hit_prob_expose), name='loss')
    return loss, logits

def user_pairwise_loss(labels,
                       logits,
                       uin,
                       weights=None,
                       weight_sqrt=True):
    ''' 实时流 基于用户维度的pairwise loss
    :param labels: bt,1
    :param logits: bt,1
    :param uin: bt,1
    :param weights: bt,1
    :param weight_sqrt: bool
    :return:
    '''
    gain_fn = lambda label: tf.math.sqrt(tf.pow(2.0, label) - 1)

    valid_pair = tf.cast(tf.equal(uin, tf.transpose(uin, perm=[1, 0])), tf.float32)  # BxB
    score_diff = logits - tf.transpose(logits, perm=[1, 0])  # BxB
    prob = tf.sigmoid(score_diff)
    prob_cliped = tf.clip_by_value(prob, 1e-20, 1.0 - 1e-7)
    loglikelyhood = tf.math.log(prob_cliped)
    pairwise_label = tf.cast((labels - tf.transpose(labels, perm=[1, 0])) > 0, tf.float32)  # BxB

    # label加权
    gain = gain_fn(labels)
    pair_gain = tf.abs(gain - tf.transpose(gain, perm=[1, 0]))  # BxB
    pair_gain *= valid_pair

    if weights is not None:
        pair_weight = weights * tf.transpose(weights, perm=[1, 0])  # BxB
        # sqrt
        if weight_sqrt:
            pair_weight = tf.math.sqrt(pair_weight)
        nonzero_num = tf.reduce_sum(valid_pair * pairwise_label * pair_weight) + 1.0
        pair_loss = -tf.reduce_sum(
            loglikelyhood * pairwise_label * pair_gain * pair_weight) / nonzero_num
    else:
        nonzero_num = tf.reduce_sum(valid_pair * pairwise_label) + 1.0
        pair_loss = -tf.reduce_sum(
            loglikelyhood * pairwise_label * pair_gain) / nonzero_num
    loss = pair_loss
    return loss