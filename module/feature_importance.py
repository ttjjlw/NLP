import tensorflow as tf
def compute_importance(y_pred, y_true, embeddings, feature_names):
    '''
    梯度*输入 重要度评估方法实现
    :param y_pred: Tensor: (batch_size, 1). 模型的预测值（sigmoid之后的结果） 
    :param y_true: Tensor: (batch_size, 1). 真实值（标签）
    :param embeddings: List of Tensors: (feature_num, embedding_dim). 待评估特征对应的embedding
    :param feature_names: List: (feature_num). 待评估特征的名称，与embedding一一对应
    :return: Dict. 各特征的重要度
    '''
    importances = {}
    probs = tf.reduce_sum(y_pred * y_true + (1 - y_pred) * (1 - y_true))
    gradients = tf.gradients(ys=probs, xs=embeddings)
    for i, gradient in enumerate(gradients):
        importances[feature_names[i]] = tf.reduce_mean(tf.reduce_sum(tf.abs(gradient * embeddings[i]), axis=0))
    return importances