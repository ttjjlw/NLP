# -*- coding: utf-8 -*-

"""
@Author  : captain
@time    : 18-7-10 下午8:42
@ide     : PyCharm
"""

import torch
import time
import torch.nn.functional as F
import models
import data
from config import DefaultConfig
from dataset import get_iter,get_dictionary
import pandas as pd
import os,pickle
# import fire
from sklearn import metrics
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score
if torch.cuda.is_available():
    torch.cuda.current_device() #这句话放这个位置，否则有可能会报RuntimeError: CUDA error: unknown error
best_score = 0.0
t1 = time.time()

def main(**kwargs):
    args = DefaultConfig()
    args.parse(kwargs)
    if not torch.cuda.is_available():
        args.cuda = False
        args.device = None
        torch.manual_seed(args.seed)  # set random seed for cpu
    train = pd.read_csv(args.train_path, sep='\t', encoding='utf-8', header=0)
    test_df = pd.read_csv(args.test_path, sep='\t', encoding='utf-8', header=0)
    corpus_all = pd.concat([train, test_df], axis=0)
    vocab = get_dictionary(corpus_all.text)
    args.vocab_size = len(vocab)
    
    train = list(zip(train.label, train.text))
    test = list(zip(test_df.label, test_df.text))
    train_data, val_data = train_test_split(train,  test_size=0.1,random_state=1)
    train_iter=get_iter(train_data, vocab, args.batch_size, True, max_len=32)
    val_iter=get_iter(val_data, vocab, args.batch_size, True, max_len=32)
    test_iter=get_iter(test, vocab, args.batch_size, True, max_len=32)
    
    if args.pretrain_embeds_path is None:
        vectors=None
    else:
        vectors=pickle.load(args.pretrain_embeds_path)
        assert len(vectors)==args.vocab_size,'预训练的词向量shape[0]为%d,而字典大小为%d'%(len(vectors),args.vocab_size)
        assert vectors.shape[1]==args.embedding_dim,'预训练词向量的shape[1]为%d，而设置的embedding_dim为%d'%(vectors.shape[1],args.embedding_dim)
    args.print_config()

    global best_score

    # init model
    model = getattr(models, args.model)(args, vectors)
    print(model)

    # 模型保存位置
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    save_path = os.path.join(args.save_dir, '{}_{}.pth'.format(args.model, args.id))

    if args.cuda:
        torch.cuda.current_device()
        torch.cuda.set_device(args.device)
        torch.cuda.manual_seed(args.seed)  # set random seed for gpu
        model.cuda()

    # 目标函数和优化器
    criterion = F.cross_entropy
    lr1, lr2 = args.lr1, args.lr2
    optimizer = model.get_optimizer(lr1, lr2, args.weight_decay)

    for i in range(args.max_epochs):
        total_loss = 0.0
        pred_labels = []
        labels = []

        model.train()

        for idx, (b_x,b_y) in enumerate(train_iter):
            # 训练模型参数
            # 使用BatchNorm层时，batch size不能为1
            if len(b_x) == 1:
                continue
            if args.cuda:
                b_x, b_y = b_x.cuda(), b_y.cuda()

            optimizer.zero_grad()
            pred = model(b_x)
            loss = criterion(pred, b_y)
            loss.backward()
            optimizer.step()

            # 更新统计指标
            total_loss += loss.item()
            predicted = pred.max(1)[1]
            pred_labels.extend(predicted.numpy().tolist())
            label=b_y.numpy().tolist()
            labels.extend(label)
            

            if idx % 100 == 0:
                print('[{}, {}] loss: {:.3f}'.format(i + 1, idx + 1, total_loss / (idx + 1)))
                
                # total_loss = 0.0
        tf1score=metrics.f1_score(labels,pred_labels)
        print('[{}, {}] tf1_score:{}'.format(i + 1, idx + 1, total_loss / (idx + 1),tf1score))
        # 计算再验证集上的分数，并相应调整学习率
        f1score = val(model, val_iter, args)
        if f1score > best_score:
            best_score = f1score
            checkpoint = {
                'state_dict': model.state_dict(),
                'config': args
            }
            torch.save(checkpoint, save_path)
            print('Best tmp model f1score: {}'.format(best_score))
        if f1score < best_score:
            model.load_state_dict(torch.load(save_path)['state_dict'])
            lr1 *= args.lr_decay
            lr2 = 2e-4 if lr2 == 0 else lr2 *0.8
            optimizer = model.get_optimizer(lr1, lr2, 0)
            print('* load previous best model: {}'.format(best_score))
            print('* model lr:{}  emb lr:{}'.format(lr1, lr2))
            if lr1 < args.min_lr:
                print('* training over, best f1 score: {}'.format(best_score))
                break

    # 保存训练最终的模型
    args.best_score = best_score
    final_model = {
        'state_dict': model.state_dict(),
        'config': args
    }
    best_model_path = os.path.join(args.save_dir, '{}_{}_{}.pth'.format(args.model, args.text_type, best_score))
    torch.save(final_model, best_model_path)
    print('Best Final Model saved in {}'.format(best_model_path))


    # 在测试集上运行模型并生成概率结果和提交结果
    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)
    probs, pre_labels = predict(model, test_iter, args)
    result_path = args.result_path + '{}_{}_{}'.format(args.model, args.id, args.best_score)
    np.save('{}.npy'.format(result_path), probs)
    print('Prob result {}.npy saved!'.format(result_path))
    test_df['label']=np.array(pre_labels)
    test_df[['idx','seq_id','label']].to_csv('{}.csv'.format(result_path), index=None)
    print('Result {}.csv saved!'.format(result_path))

    t2 = time.time()
    print('time use: {}'.format(t2 - t1))


def predict(model, test_data, args):
    # 生成测试提交数据csv
    # 将模型设为验证模式
    model.eval()

    result = np.zeros((0,))
    probs_list = []
    with torch.no_grad():
        for b_x,y_x in test_data:
            if args.cuda:
                b_x = b_x.cuda()
            outputs = model(b_x)
            probs = F.softmax(outputs, dim=1)
            probs_list.append(probs.cpu().numpy())
            pred = outputs.max(1)[1]
            result = np.hstack((result, pred.cpu().numpy()))

    # 生成概率文件npy
    prob_cat = np.concatenate(probs_list, axis=0)

    # test = pd.read_csv('../data/test/torchClassify/test.csv')
    # test_id = test['id'].copy()
    # test_pred = pd.DataFrame({'id': test_id, 'class': result})
    # test_pred['class'] = (test_pred['class'] + 1).astype(int)

    return prob_cat, result


def val(model, test_loader, args):
    # 计算模型在验证集上的分数

    # 将模型设为验证模式
    model.eval()

    acc_n = 0
    val_n = 0
    predict = np.zeros((0,), dtype=np.int32)
    gt = np.zeros((0,), dtype=np.int32)
    with torch.no_grad():
        for _, (t_x, t_y) in enumerate(test_loader):
            if args.cuda:
                t_x = t_x.cuda()
                t_y = t_y.cuda()
            pred = model(t_x)
            pred = pred.max(1)[1]
            acc_n += (pred == t_y).sum().item()
            val_n += t_y.size(0)
            predict = np.hstack((predict, pred.cpu().numpy()))
            gt = np.hstack((gt, t_y.cpu().numpy()))
        acc = 100. * acc_n / val_n
        f1score = metrics.f1_score(predict, gt)
        print('* Test Acc: {:.3f}%({}/{}), F1 Score: {:.3f}%({}/{})'.format(acc, acc_n, val_n, 100.*f1score,acc_n, val_n))
    return f1score


if __name__ == '__main__':
    for model in ['TextCNN','RCNN','RCNN1','LSTM','AttLSTM','GRU','bilstm_conv','InCNN','FastText']:
        main(model=model)
