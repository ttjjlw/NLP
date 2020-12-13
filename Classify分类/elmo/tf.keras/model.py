# F:\itsoftware\Anaconda
# -*- coding:utf-8 -*-
# Author = TJL
# date:2020/12/11
import tensorflow as tf
import os
from tensorflow import keras
import tensorflow_hub as hub
import numpy as np
# os.environ["TFHUB_CACHE_DIR"]='preprocessing/tfhub' #设置模型缓存路径
# elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
#
train_data=np.array(['it is cool','it is bad','it is wonderful']) #3,1
# train_data=np.array([[1,2,3],[1,2,4],[1,2,5]])
train_label=np.array([1,0,1])
valid_data,valid_label=train_data,train_label
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding,
						   input_shape=[],
                           dtype=tf.string,
                           trainable=True)
model = keras.Sequential()
model.add(hub_layer)
# model.add(keras.layers.Input(shape=(3,)))
# model.add(keras.layers.Embedding(input_dim=6,output_dim=20,input_length=3,
#                          trainable=True))
# model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(2, activation='softmax'))

model.summary()

# categorical_crossentropy binary_crossentropy
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])

model.fit(train_data,train_label,epochs=3,batch_size=1,validation_data=(valid_data,valid_label))

