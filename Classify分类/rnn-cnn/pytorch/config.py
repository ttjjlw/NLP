# D:\localE\python
# -*-coding:utf-8-*-
# Author ycx
class DefaultConfig():
    '''
    列出所有的参数，只根据模型的需要获取参数

    '''
    env = 'default'  # visdom环境
    seed = 777
    best_score = 0
    model = 'TextCNN'  # 使用的模型，名字必须与models/__init__.py中的名字一致
    model_path = None  # 如果有就加载
    result_path = ''
    save_dir = 'snapshot/'  # where to save the snapshot
    id = 'default'  # 保存模型的名字后缀
    device = 0
    boost = False  # 是否使用adboost
    bo_layers = 5  # boost的层数
    finetune = False  # 是否对训练完的模型进行finetune
    aug = False  ##是否进行数据增强
    text_type='binary'
    result_path='result/' #test集预测结果保存目录

    # 数据参数

    train_path = './data/train.csv'
    test_path = './data/test.csv'
    pretrain_embeds_path = None #类型为pkl类型，如'embeddings300_3000001.pkl，若为None，则代表不用预训练词向量
    embedding_dim = 128
    data_line_shuffle = True
    vocab_size = 10000  # 占位设置，会根据字典大小自动重新赋值
    split_rate = 0.1  # split_rate*total_train_data的验证集

    batch_size = 16
    label_size = 2
    max_text_len = 32
    # 训练参数

    lr1 = 0.001  # leraning rate
    lr2 = 0  # embedding 层的学习率
    min_lr = 1e-5  # 当学习率低于这个nvi值时，就退出训练
    lr_decay = 0.8  # 当一个epoch的损失开始上升时，lr ＝ lr*lr_decay
    decay_every = 10000  # 每多少个batch  查看val acc，并修改学习率
    weight_decay = 0  # 权重衰减
    max_epochs = 20
    cuda = True

    # 模型通用
    linear_hidden_size = 100  # 原来为2000(500)，之后还需要修改，感觉数值有点大
    epoches = 10

    # TextCNN
    kernel_num = 200  # number of each kind of kernel
    kernel_sizes = [1, 2, 3, 4, 5]  # kernel size to use for convolution
    dropout_rate = 0.5  # the probability for dropout
    # LSTM
    hidden_dim = 256
    lstm_dropout = 0.5  # 只有当lstm_layers > 1时，设置lstm_dropout才有意义
    lstm_layers = 1
    kmax_pooling = 2

    # rcnn
    rcnn_kernel = 200

    def parse(self, kwargs):
        '''
        根据字典kwargs 更新 config参数
        '''

        # 更新配置参数
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception("Warning: config has not attribute <%s>" % k)
            setattr(self, k, v)

    def print_config(self):
        # 打印配置信息
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and k != 'parse' and k != 'print_config':
                print('    {} : {}'.format(k, getattr(self, k)))
