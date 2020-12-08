# F:\itsoftware\Anaconda
# -*- coding:utf-8 -*-
# Author = TJL
# date:2020/3/13
# -*- coding: utf-8 -*-


from gensim.models import word2vec,KeyedVectors
import logging,collections,pickle,os,argparse,json
import numpy as np

##训练word2vec模型

# 获取日志信息
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

# 加载分词后的文本，使用的是Text8Corpus类
# with open ('data/train_data_path/corpus.txt','rb') as f:
#     lines=f.readlines()
#     corpus=[]
#     for line in lines:
#         corpus.extend(line.decode('utf-8').strip().split(' '))
# dic = collections.Counter(corpus).most_common()
def train(args):
    sentences = word2vec.LineSentence(args.train_data_path)

    # 训练模型，部分参数如下
    # max_vocab_size: 设置词向量构建期间的RAM限制，设置成None则没有限制。
    #trim_rule： 用于设置词汇表的整理规则，指定那些单词要留下，哪些要被删除。可以设置为None（min_count会被使用）。
    model = word2vec.Word2Vec(sentences, #对于大语料建议使用BrownCorpus,Text8Corpus或LineSentence构建
                              size=args.embed_dim, #size: 词向量的维度，默认值是100
                              alpha=0.025,#alpha： 是初始的学习速率默认值0.025，在训练过程中会线性地递减到min_alpha。
                              hs=args.hs,#hs: 即我们的word2vec两个解法的选择了，如果是0， 则是Negative Sampling，是1的话并且负采样个数negative大于0， 则是Hierarchical Softmax。默认是0即Negative Sampling。
                              min_count=1,#min_count:：可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5。
                              window=args.window_size, #window：即词向量上下文最大距离，skip-gram和cbow算法是基于滑动窗口来做预测。默认值为5。在实际使用中，可以根据实际的需求来动态调整这个window的大小。对于一般的语料这个值推荐在[5,10]之间。
                              sample=1e-3,#sample: 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)
                              seed=1,#seed：用于随机数发生器默认为1。与初始化词向量有关。
                              workers=3,#workers：用于控制训练的并行数默认为3。
                              min_alpha=0.0001,#min_alpha: 由于算法支持在迭代的过程中逐渐减小步长，min_alpha给出了最小的迭代步长值。随机梯度下降中每    轮的迭代步长可以由iter，alpha， min_alpha一起得出。对于大语料，需要对alpha, min_alpha,iter一起调参，来选择合适的三个值。
                              sg=args.sg,#0是CBOW,1是skipGram，默认为0
                              negative=args.negative_size,#negative:如果大于零，则会采用negativesampling，用于设置多少个noise words（一般是5-20）。
                              iter=args.iter,#随机梯度下降法中迭代的最大次数，默认是5。对于大语料，可以增大这个值
                              sorted_vocab=1,#如果为1（默认），则在分配word index 的时候会先对单词基于频率降序排序。
                              batch_words=10000 #每一批的传递给线程的单词的数量，默认为10000。

                              )


    # 保留模型，方便重用
    model.save(u'word2vec.model')#该方法保存，可用来继续训练
    #按词频由大到小写入word及其embedding
    model.wv.save_word2vec_format(args.embed_path_txt, binary=False)

def load_pretrain_model(model_dir):
    '''
    加载word2vec预训练word embedding文件
    Args:
        model_dir: word embedding文件保存路径
    '''
    model = KeyedVectors.load_word2vec_format(model_dir)
    print('similarity(不错，优秀) = {}'.format(model.similarity("不错", "优秀")))
    print('similarity(不错，糟糕) = {}'.format(model.similarity("不错", "糟糕")))
    most_sim = model.most_similar("不错", topn=10)
    print('The top10 of 不错: {}'.format(most_sim))
    words = model.vocab
    # print(1)
def get_vocab_and_embed(args):
    word2id={'<pad>':0}
    index=0
    embeddings=[]
    embeddings.append([0]*args.embed_dim)
    with open (args.embed_path_txt,'r', encoding='utf-8') as f:
        lines=f.readlines()
        for line in lines:
            if len(line.strip().split())<3:
                continue
            word=line.strip().split()[0]
            vector=line.strip().split()[1:]
            word2id[word]=index+1
            index+=1
            try:
                assert len(vector)==args.embed_dim
            except:
                ValueError('输出的vector的维度不是设置的%s维'%str(args.embed_dim))
            embeddings.append(vector)
    embeddings=np.array(embeddings)
    print('embed_txt的shape为：({}*{})'.format(index,args.embed_dim))
    print('embed_pkl的shape为：{}'.format(embeddings.shape))
    print('embed_pkl的index为0的位置加了<pad>的向量，所以比embed_txt多1')
    print('vocab的长度: %d'%len(word2id))
    with open(args.embed_path_pkl,'wb') as p:
        pickle.dump(embeddings,p)
    with open(args.vocab_path,'w',encoding='utf-8') as p1:
        json.dump(word2id,p1,ensure_ascii=False)
    print('完成！')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate word2vec by gensim')
    parser.add_argument('--raw_data_path', type=str, default='./data/raw_data/',
                        help='the dir of raw data file,in this dir can contain more than one files')
    parser.add_argument('--train_data_path', type=str, default='data/train_corpus/corpus.txt',
                        help='the path of train data file')
    parser.add_argument('--embed_path_txt', type=str, default="export/Vector.txt",
                        help='the save path of word2vec with type txt')
    parser.add_argument('--embed_path_pkl', type=str, default="export/Vector.pkl",
                        help='the save path of word2vec with type pkl,which is array after pickle.load ')
    parser.add_argument('--vocab_path', type=str, default='export/vocab.json', help='the save path of vocab')
    parser.add_argument('--embed_dim', type=int, default=128, help='the dim of word2vec')
    parser.add_argument('--window_size', type=int, default=3, help='window size')
    parser.add_argument('--negative_size', type=int, default=5, help='负采样个数')
    parser.add_argument('--iter', type=int, default=5, help='训练次数')
    parser.add_argument('--sg', type=int, default=0, help='0是CBOW,1是skipGram')
    parser.add_argument('--hs', type=int, default=0,
                        help='即我们的word2vec两个解法的选择了，如果是0， 则是Negative Sampling，是1的话并且负采样个数negative大于0， 则是Hierarchical Softmax。默认是0即Negative Sampling')
    args = parser.parse_args()
    if not os.path.exists(os.path.dirname(args.embed_path_txt)): os.makedirs(os.path.dirname(args.embed_path_txt))
    if not os.path.exists(os.path.dirname(args.embed_path_pkl)): os.makedirs(os.path.dirname(args.embed_path_pkl))
    if not os.path.exists(os.path.dirname(args.vocab_path)): os.makedirs(os.path.dirname(args.vocab_path))
    train(args)
    get_vocab_and_embed(args)
    # load_pretrain_model(model_dir)
