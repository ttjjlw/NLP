import tensorflow as tf
def mle_loss(a,b,true_scores,weight,l2_normal=False, cos=False):
    '''
    :param a: 用户vector
    :param b: item vector shape 为 （list_size,bt,dim）
    :param true_scores:
    :param weight: shape(bt,list_size)
    :param l2_normal:
    :param cos:
    :return:
    '''
    if not cos:
        if l2_normal:
            a = tf.nn.l2_normalize(a, axis=-1)
            b = tf.nn.l2_normalize(b, axis=-1)
        logits = 5 * dot(a, b)
    else:
        logits = 5 * cosin(a, b) #41,bt
    logits = tf.transpose(logits, [1, 0])

    def mle(true_scores, rank_scores,list_size,pos_list_size):
        index = tf.argsort(true_scores, direction='DESCENDING')
        S_predict = tf.batch_gather(rank_scores, index)
        # 分子
        initm_up = tf.constant([[1.0]])  # shape 为 1,1
        for i in range(pos_list_size):  # 3为 pos_list_size
            a = tf.slice(S_predict, [0, i], [-1, 1])
            initm_up = initm_up * a  # bt,1

        # 分母
        initm_down = tf.constant([[1.0]])  # shape 为 1,1
        for i in range(list_size):
            b = tf.reduce_sum(tf.slice(S_predict, [0, i], [-1, list_size - i]), axis=-1, keep_dims=True)  # bt，1
            initm_down *= b
        initm_up = tf.clip_by_value(initm_up, 1e-30, 1e30)
        initm_down = tf.clip_by_value(initm_down, 1e-30, 1e30)
        loss = tf.log(tf.clip_by_value(tf.divide(initm_up, initm_down), 1e-30, 1e30))
        mleloss = -tf.reduce_mean(tf.log(loss))
        return mleloss

    return mle(true_scores,logits,list_size=41,pos_list_size=10)


def mle_loss(a,b,true_scores,weight,l2_normal=False, cos=False):
    if not cos:
        if l2_normal:
            a = tf.nn.l2_normalize(a, axis=-1)
            b = tf.nn.l2_normalize(b, axis=-1)
        logits = dot(a, b)
    else:
        logits = cosin(a, b)
    logits = tf.transpose(logits,[1,0])
    logits = tf.nn.softmax(logits,axis=-1)
    def mle(true_scores,rank_scores,list_size):
        index = tf.argsort(true_scores, direction='DESCENDING')
        S_predict=tf.batch_gather(rank_scores, index)
        # S_predict=tf.gather(rank_scores, index,axis=-1,batch_dims=0)
        #分子
        initm_up=tf.constant([[1.0]]) #shape 为 1,1
        for i in range(list_size):#3为 list_size
            a=tf.slice(S_predict,[0,i],[-1,1]) #shape 为 bt,1
            initm_up=initm_up*a #bt,1

        #分母
        initm_down=tf.constant([[1.0]]) #shape 为 1,1
        for i in range(list_size):
            b=tf.reduce_sum(tf.slice(S_predict, [0, i], [-1, list_size - i]), axis=-1,keep_dims=True)  # bt
            initm_down*=b
        initm_up=tf.clip_by_value(initm_up,1e-8,1e8)
        initm_down=tf.clip_by_value(initm_down,1e-8,1e8)
        loss=tf.log(tf.clip_by_value(tf.divide(initm_up,initm_down),1e-8,1e8))
        mleloss=-tf.reduce_mean(loss)
        return mleloss

    return mle(true_scores,logits,list_size=5)