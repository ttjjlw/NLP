import tensorflow as tf
import numpy as np
import pickle
import os, argparse, time, random
from model import BiLSTM_CRF
from utils import str2bool, get_logger
from data import read_corpus, read_dictionary, tag2label_mapping, random_embedding, vocab_build, \
    build_character_embeddings

# Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #设置使用哪块GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0 设置运行使系统打印哪些信息

tf.logging.set_verbosity(tf.logging.INFO)
logger = tf.get_logger()
logger.propagate = False

config = tf.ConfigProto()
config.gpu_options.allow_growth = True   #动态申请GPU内存
config.gpu_options.per_process_gpu_memory_fraction = 0.8  # need ~700MB GPU memory 限制GPU使用率

# hyper parameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--dataset_name', type=str, default='ResumeNER',
                    help='choose a dataset(MSRA, ResumeNER, WeiboNER,人民日报)')
parser.add_argument('--negative_label', type=list, default=['O'], help='不参与计算f1score的tags')
parser.add_argument('--iob2iobes', type=bool, default=False, help='把iob格式的tags转变成iobes')
parser.add_argument('--batch_size', type=int, default=5, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=60, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--use_pre_emb', type=str2bool, default=False,
                    help='use pre_trained char embedding or init it randomly')
parser.add_argument('--pretrained_emb_path', type=str, default='data_path/DaGuang/embeddings785571500.pkl', help='pretrained embedding path')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='test', help='train/test/demo')
parser.add_argument('--isload2train',type=str2bool,default=False,help='is load model to train')
parser.add_argument('--demo_model', type=str, default='1566889554', help='model for test and demo')
args = parser.parse_args()
train_file='train_data.txt'
test_file='test_data.txt'
args.mode='demo'
args.demo_model='1611481059' #预测的时候一定要指定，否则找不到模型路径
args.CRF=True
args.hidden_dim=512
args.isload2train=False
# build char embeddings
if not args.use_pre_emb:
    # vocabulary build
    if not os.path.exists(os.path.join('data_path', args.dataset_name, 'word2id.pkl')):
        #原始数据集：txt文件  字1\ttag1\n
        #                  字2\ttag2\n
        # line1                    ...
        #                  字n\ttagn\n             需要注意的是每个line 用两个换行符隔开
        #                       \n
        # line2                 同上
        vocab_build(args,os.path.join('data_path', args.dataset_name, 'word2id.pkl'),
                    os.path.join('data_path', args.dataset_name, train_file))

    # get word dictionary
    word2id = read_dictionary(os.path.join('data_path', args.dataset_name, 'word2id.pkl'))
    embeddings = random_embedding(word2id, args.embedding_dim)
    log_pre = 'not_use_pretrained_embeddings'
else:
    with open('data_path//DaGuang//dr_d_td_all.pkl','rb') as f:
        id2word=pickle.load(f)
        word2id=pickle.load(f)
        print('word2id的length:',len(word2id))
        _ = pickle.load(f)
    embeddings_path = os.path.join('data_path', args.dataset_name, 'pretrain_embedding.npy')
    if not os.path.exists(embeddings_path):
        build_character_embeddings(args.pretrained_emb_path, embeddings_path)
    embeddings = np.array(np.load(embeddings_path), dtype='float32')
    log_pre = 'use_pretrained_embeddings'

# choose tag2label
tag2label = tag2label_mapping[args.dataset_name]
assert list(range(len(tag2label)))==list(tag2label.values()),'%s,for dictionary values，it should be set startwith 0 and continuous'



# paths setting
paths = {}
timestamp = str(int(time.time())) if args.mode == 'train' and args.isload2train ==False else args.demo_model
output_path = os.path.join('model_path', args.dataset_name, timestamp)
if not os.path.exists(output_path): os.makedirs(output_path)

summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)

model_path = os.path.join(output_path, "checkpoints/")
paths['check_path'] = model_path

if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix

result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)

log_path = os.path.join(result_path, args.dataset_name + log_pre + "_log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))

# read corpus and get training data
if args.mode != 'demo':
    train_path = os.path.join('data_path', args.dataset_name, train_file)
    paths['train_path']=train_path
    test_path = os.path.join('data_path', args.dataset_name, test_file)
    paths['test_path']=test_path
    train_data = read_corpus(train_path)[:100]
    test_data = read_corpus(test_path)
    test_size = len(test_data)
    print("train data: {}".format(len(train_data)))
    print("test data: {}".format(test_size))

# training model
if args.mode == 'train':
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()

    # hyperparameters-tuning, split train/dev
    # dev_data = train_data[:5000]; dev_size = len(dev_data)
    # train_data = train_data[5000:]; train_size = len(train_data)
    # print("train data: {0}\ndev data: {1}".format(train_size, dev_size))
    # model.train(train=train_data, dev=dev_data)

    # train model on the whole training data

    ckpt_file = tf.train.latest_checkpoint(model_path) #采用最新的模型训练
    # ckpt_file=r'model_path\DaGuang\1566526104\checkpoints/model.ckpt-8' #指定载入模型训练
    print(ckpt_file)
    # paths['model_path'] = ckpt_file
    model.train(train=train_data, dev=test_data,model_path=ckpt_file)  # use test_data.txt as the dev_data to see overfitting phenomena

# testing model
elif args.mode == 'test':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    # ckpt_file = r'model_path\DaGuang\1566611347\checkpoints/model.ckpt-9'
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    model.test(test_data)

# demo
elif args.mode == 'demo':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)
        while (1):
            print('Please input your sentence:')
            demo_sent = input()
            if demo_sent == '' or demo_sent.isspace():
                continue
            elif demo_sent=='exit' or demo_sent=='q':
                print('See you next time!')
                break
            else:
                demo_sent = list(demo_sent.strip())
                demo_data = [(demo_sent, ['O'] * len(demo_sent))]
                tag = model.demo_one(sess, demo_data)
                print(list(zip(demo_sent,tag)))
