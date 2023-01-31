import tensorflow as tf # tf2
def batch_negative_sampling(
    labels: tf.Tensor,
    weights: tf.Tensor,
    user_embs: tf.Tensor,
    item_embs: tf.Tensor,
    neg_num: int,
    is_training: tf.Tensor,) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Sample negative items from the same batch.
    输入就是正常的 labels，weights,user_embeds, item_embeds,输出 shape 为 [batch_size+(pos_num*neg_num) ,dim]
    的 new_labels，new_weights，new_user_embeds，new_item_embeds
    问题是随机负采样过程会采到相同的item
    Args:
        labels (tf.Tensor): Labels of each user-item pair, with shape of (batch_size, 1).
        weights (tf.Tensor): Loss weights of each sample, with shape of (batch_size, 1).
        user_embs (tf.Tensor): Embeddings of users, with shape of (batch_size, user_dim).
        item_embs (tf.Tensor): Embeddings of items, with shape of (batch_size, item_dim).
        neg_num (int): The number of negative samples sampled for each positive sample.
        is_training (tf.Tensor): Numerous ControlPlaceholder.

    Returns:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Sampled labels, user and item embeddings.
    """
    batch_size = tf.shape(user_embs)[0]
    logits = tf.ones(shape=[batch_size, batch_size]) - tf.eye(batch_size) * (2 << 32 - 1)
    sampled_indices = tf.cond(
        tf.greater(batch_size, 0),
        true_fn=lambda: tf.random.categorical(logits, neg_num), #按概率采样，有可能会采到相同的索引
        false_fn=lambda: tf.constant([], dtype=tf.int64)
    )

    mask = tf.reshape(tf.greater(labels, 0.0), shape=[-1])
    pos_user_embs = tf.boolean_mask(user_embs, mask, axis=0)
    sampled_user_embs = tf.repeat(
        pos_user_embs,
        repeats=tf.ones(shape=[tf.shape(pos_user_embs)[0]], dtype=tf.int32) * neg_num,
        axis=0)
    sampled_indices = tf.boolean_mask(sampled_indices, mask, axis=0)
    sampled_item_embs = tf.gather(
        item_embs, indices=tf.reshape(sampled_indices, shape=[-1]), axis=0)
    sampled_labels = tf.zeros(shape=[tf.shape(sampled_user_embs)[0], 1])

    new_labels = tf.cond(
        pred=is_training,
        true_fn=lambda: tf.concat([labels, sampled_labels], axis=0),
        false_fn=lambda: labels
    )
    new_weights = tf.cond(
        pred=is_training,
        true_fn=lambda: tf.concat([weights, tf.ones_like(sampled_labels)], axis=0),
        false_fn=lambda: weights,
    )
    new_user_embs = tf.cond(
        pred=is_training,
        true_fn=lambda: tf.concat([user_embs, sampled_user_embs], axis=0),
        false_fn=lambda: user_embs,
    )
    new_item_embs = tf.cond(
        pred=is_training,
        true_fn=lambda: tf.concat([item_embs, sampled_item_embs], axis=0),
        false_fn=lambda: item_embs,
    )
    return new_labels, new_weights, new_user_embs, new_item_embs