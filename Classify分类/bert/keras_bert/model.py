import pandas as pd
import codecs, gc
import numpy as np
from sklearn.model_selection import KFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.metrics import top_k_categorical_accuracy
from keras.layers import *
from keras.callbacks import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical
 
#读取训练集和测试集
train_df=pd.read_csv('data/data_train.csv', sep=',', names=['id', 'type', 'contents', 'labels']).astype(str)
test_df=pd.read_csv('data/data_test.csv', sep=',', names=['id', 'type', 'contents']).astype(str)
 
maxlen = 100  #设置序列长度为120，要保证序列长度不超过512
 
#预训练好的模型
config_path = 'base_bert/bert_config.json'
checkpoint_path = 'base_bert/bert_model.ckpt'
dict_path = 'base_bert/vocab.txt'
 
#将词表中的词编号转换为字典
token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
 
#重写tokenizer        
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # 用[unused1]来表示空格类字符
            else:
                R.append('[UNK]')  # 不在列表的字符用[UNK]表示
        return R
tokenizer = OurTokenizer(token_dict)
 
#让每条文本的长度相同，用0填充
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])
 
#data_generator只是一种为了节约内存的数据方式
class data_generator:
    def __init__(self, data, batch_size=3, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
 
    def __len__(self):
        return self.steps
 
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
 
            if self.shuffle:
                np.random.shuffle(idxs)
 
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y[:, 0, :]
                    [X1, X2, Y] = [], [], []
 
#计算top-k正确率,当预测值的前k个值中存在目标类别即认为预测正确                 
def acc_top2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)
 
#bert模型设置
def build_bert(nclass):
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)  #加载预训练模型
 
    for l in bert_model.layers:
        l.trainable = True
 
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
 
    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x) # 取出[CLS]对应的向量用来做分类
    p = Dense(nclass, activation='softmax')(x)
 
    model = Model([x1_in, x2_in], p)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(1e-5),    #用足够小的学习率
                  metrics=['accuracy', acc_top2])
    print(model.summary())
    return model
 
#训练数据、测试数据和标签转化为模型输入格式
DATA_LIST = []
for data_row in train_df.iloc[:].itertuples():
    DATA_LIST.append((data_row.contents, to_categorical(data_row.labels, 3)))
DATA_LIST = np.array(DATA_LIST)
 
DATA_LIST_TEST = []
for data_row in test_df.iloc[:].itertuples():
    DATA_LIST_TEST.append((data_row.contents, to_categorical(0, 3)))
DATA_LIST_TEST = np.array(DATA_LIST_TEST)
 
#交叉验证训练和测试模型
def run_cv(nfold, data, data_labels, data_test):
    kf = KFold(n_splits=nfold, shuffle=True, random_state=520).split(data)
    train_model_pred = np.zeros((len(data), 3))
    test_model_pred = np.zeros((len(data_test), 3))
 
    for i, (train_fold, test_fold) in enumerate(kf):
        X_train, X_valid, = data[train_fold, :], data[test_fold, :]
 
        model = build_bert(3)
        early_stopping = EarlyStopping(monitor='val_acc', patience=3)   #早停法，防止过拟合
        plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5, patience=2) #当评价指标不在提升时，减少学习率
        checkpoint = ModelCheckpoint('./bert_dump/' + str(i) + '.hdf5', monitor='val_acc',verbose=2, save_best_only=True, mode='max', save_weights_only=True) #保存最好的模型
 
        train_D = data_generator(X_train, shuffle=True)
        valid_D = data_generator(X_valid, shuffle=True)
        test_D = data_generator(data_test, shuffle=False)
        #模型训练
        model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=5,
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D),
            callbacks=[early_stopping, plateau, checkpoint],
        )
 
        # model.load_weights('./bert_dump/' + str(i) + '.hdf5')
 
        # return model
        train_model_pred[test_fold, :] = model.predict_generator(valid_D.__iter__(), steps=len(valid_D), verbose=1)
        test_model_pred += model.predict_generator(test_D.__iter__(), steps=len(test_D), verbose=1)
 
        del model
        gc.collect()   #清理内存
        K.clear_session()   #clear_session就是清除一个session
        # break
 
    return train_model_pred, test_model_pred
 
#n折交叉验证
train_model_pred, test_model_pred = run_cv(2, DATA_LIST, None, DATA_LIST_TEST)
 
test_pred = [np.argmax(x) for x in test_model_pred]
 
#将测试集预测结果写入文件
output=pd.DataFrame({'id':test_df.id,'sentiment':test_pred})
output.to_csv('data/results.csv', index=None)
 
