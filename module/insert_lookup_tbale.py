#!/usr/bin/env python
import tensorflow as tf
if tf.__version__<'2.0':
    table = tf.contrib.lookup.MutableHashTable(key_dtype=tf.string,
                                                value_dtype=tf.float32,
                                                default_value=-1)
    key = tf.constant('hi', tf.string)
    val = tf.constant(1.1,tf.float32)

    with tf.Session() as sess:
        sess.run(table.insert(key, val))
        print(table.lookup(key).eval())

if tf.__version__>='2.9':
    table = tf.lookup.experimental.MutableHashTable(key_dtype=tf.string,
                                                    value_dtype=tf.int64,
                                                    default_value=-1)
    keys_tensor = tf.constant(['a', 'b', 'c'])
    vals_tensor = tf.constant([7, 8, 9], dtype=tf.int64)
    input_tensor = tf.constant(['a', 'f'])
    table.insert(keys_tensor, vals_tensor)
    table.insert('a',tf.constant(1.0))
    table.lookup(input_tensor).numpy()

    table.remove(tf.constant(['c']))
    table.lookup(keys_tensor).numpy()

    sorted(table.export()[0].numpy())

    sorted(table.export()[1].numpy())
