# F:\itsoftware\Anaconda
# -*- coding:utf-8 -*-
# Author = TJL
# date:2020/12/17
import tensorflow as tf
import numpy as np
import pandas as pd
import os,pickle,argparse
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from bert_model_for_bin_classify import Classify_BertModel
if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示 Error
    version='albert_base'
    parser = argparse.ArgumentParser(description='bert for classify')
    parser.add_argument('--version', type=str, default=version,
                        help='bert 模型的版本，要与bert 模型放置的目录名对应,如：base_bert/robert_base/albert_large')
    parser.add_argument('--max_len', type=int, default=32, help='max_len of query')
    parser.add_argument('--mode', type=int, default=32, help='train or predict')
    parser.add_argument('--vocab_path', type=str, default='bert_model/%s/vocab.txt'%version,
                        help='the path of vocab.txt')
    parser.add_argument('--bert_config_path', type=str,
                        default='bert_model/%s/bert_config.json'%version,
                        help='the path of bert_config.json')
    parser.add_argument('--bert_path', type=str, default='bert_model/%s/bert_model.ckpt'%version,
                        help='the path of bert_model.ckpt')
    parser.add_argument('--log_path', type=str, default='log/log.txt', help='the path of log.txt')
    parser.add_argument('--export_model_path', type=str, default='export/model/', help='the path of log.txt')
    parser.add_argument('--batch_size', type=int, default=16, help='the path of log.txt')
    parser.add_argument('--init_lr', type=float, default=2e-5, help='初始学习率，热身阶段过后，学习率会线性衰减')
    parser.add_argument('--is_train', type=bool, default=True, help='bert中的weights是否参与训练')
    parser.add_argument('--keep_rate', type=float, default=0.5, help='the path of log.txt')
    parser.add_argument('--epochs', type=int, default=3, help='the path of log.txt')
    parser.add_argument('--num_folds', type=int, default=1, help='the path of log.txt')
    parser.add_argument('--isload', type=int, default=0, help='是从头开始训练，还是载入模型接着训练')
    parser.add_argument('--restore_on_train', type=int, default=0, help='是否在训练过程中，每个epoch加载测试集指标最好的模型，再训练')
    parser.add_argument('--num_train_steps', type=int, default=0, help='占位,后面会重新赋值')
    parser.add_argument('--num_warmup_steps', type=int, default=0, help='占位，后面会重新赋值，热身阶段，在这个step内，学习率会从0慢慢增加到init_lr')
    parser.add_argument('--warmup_proportion', type=float, default=0.1, help='the path of log.txt')
    # data path
    parser.add_argument('--train_path', type=str, default='./data/train/train.csv', help='the path of train.csv')
    parser.add_argument('--test_path', type=str, default='./data/test/test.csv',
                        help='the path of test.csv')
    # result path
    parser.add_argument('--result_path', type=str, default='./result/submission', help='the path of result')
    parser.add_argument('--logits_path', type=str, default='./logits/', help='the path of logits,用于概率融合')
    args = parser.parse_args()
    if not os.path.exists(os.path.dirname(args.result_path)): os.makedirs(os.path.dirname(args.result_path))
    if not os.path.exists(os.path.dirname(args.logits_path)): os.makedirs(os.path.dirname(args.logits_path))
    if not os.path.exists(os.path.dirname(args.log_path)): os.makedirs(os.path.dirname(args.log_path))
    # 读取数据
    train = pd.read_csv(args.train_path, sep='\t', encoding='utf-8', header=0)[:10000]
    test_df = pd.read_csv(args.test_path, sep='\t', encoding='utf-8', header=0)[:1000]
    test_df['label']=test_df['query'].apply(lambda x:1) #给test增加label字段方便数据处理
    train = list(zip(train.label, train['query'], train.reply,train.seq_id))
    test = list(zip(test_df.label, test_df['query'], test_df.reply,test_df.seq_id))

    # 构建模型
    args.mode='train'
    args.num_folds=5
    if args.mode=='train':
        #如果num_folds>1,则进行num_flods折融合
        if args.num_folds>1:
            kfold = model_selection.KFold(
                n_splits=args.num_folds, shuffle=True, random_state=42)
            train = np.array(train)
            test = np.array(test)
            args.num_train_steps = int(
                len(train) * 1.0 / args.batch_size * args.epochs)
            args.num_warmup_steps = int(args.num_train_steps * args.warmup_proportion)
            model = Classify_BertModel(args=args) #一定要在args.args.num_warmup_steps重新赋值之后
            
            logits = np.zeros(shape=(len(test),2))
            for fold_num, (train_idx, valid_idx) in enumerate(kfold.split(train)):
                tf.reset_default_graph()
                model.build_graph(mode=args.mode)
                args.isload=True#N折融合，强制设置从头训练
                if fold_num+1>3: #方便测试，实际使用可注释掉
                    break
                args.lr=2e-5
                print("\nfold %02d" % (fold_num + 1))
                train_data = train[train_idx]
                valid_data = train[valid_idx]
                best_score = model.train(
                    train=train_data, valid=valid_data, fold_num=fold_num)#train,valid,test既可以是里列表也可以是array
                logit=model.predict(test,fold_num=fold_num) #predict 中 会自动重新构建图
                logits+=np.array(logit)
            pred_lb = np.argmax(logits, 1)
            logits=logits/(fold_num+1)
        else:
            train_data,valid_data=train_test_split(train,test_size=0.1)
            args.num_train_steps = int(
                len(train) * 1.0 / args.batch_size * args.epochs)
            args.num_warmup_steps = int(args.num_train_steps * args.warmup_proportion)
            model = Classify_BertModel(args=args)
            model.build_graph()
            best_score= model.train(
                train=train_data, valid=valid_data, fold_num=1)
            logits=model.predict(test,fold_num=1)
            pred_lb = np.argmax(logits, 1)
        test_df['label'] = np.array(pred_lb)
        test_df = test_df[['idx', 'seq_id', 'label']]
        test_df.to_csv(args.result_path+args.version+str(best_score)+'.csv', header=None, index=False, sep='\t')
        with open(args.logits_path+args.version+str(best_score)+'.pkl','wb') as f:
            pickle.dump(logits,f)
    else:
        model = Classify_BertModel(args=args)
        logits = model.predict(test, fold_num=1)
        pred_lb = np.argmax(logits, 1)
        test_df['label'] = np.array(pred_lb)
        test_df = test_df[['idx', 'seq_id', 'label']]
        test_df.to_csv(args.result_path + args.version + '.csv', header=None, index=False, sep='\t')
        with open(args.logits_path + args.version + '.pkl', 'wb') as f:
            pickle.dump(logits, f)