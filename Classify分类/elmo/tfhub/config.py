# F:\itsoftware\Anaconda
# -*- coding:utf-8 -*-
# Author = TJL
# date:2020/12/3
class Config(object):
    def __init__(self):
        self.train_data_path = './data/train_data.txt'
        self.eval_data_path = './data/eval_data.txt'
        self.test_data_path = './data/test_data.txt'
        self.lr = 0.001
        self.export_path = 'export/'
        self.result_path = 'result'
        self.label_num = 2
        self.epochs = 3
        self.batch_size = 32
        self.keep_rate=0.5