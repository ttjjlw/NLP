import torch, pickle, os,argparse,json
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import numpy as np
from tqdm import tqdm

from tools import CorpusPreprocess, VectorEvaluation

# get gpu
use_gpu = torch.cuda.is_available()


# calculation weight
def fw(X_c_s,x_max,alpha):
    return (X_c_s / x_max) ** alpha if X_c_s < x_max else 1


class Glove(nn.Module):
    def __init__(self, vocab, args):
        super(Glove, self).__init__()
        # center words weight and biase
        self.c_weight = nn.Embedding(len(vocab), args.embed_dim,
                                     _weight=torch.randn(len(vocab),
                                                         args.embed_dim,
                                                         dtype=torch.float,
                                                         requires_grad=True) / 100)

        self.c_biase = nn.Embedding(len(vocab), 1, _weight=torch.randn(len(vocab),
                                                                            1, dtype=torch.float,
                                                                            requires_grad=True) / 100)

        # surround words weight and biase
        self.s_weight = nn.Embedding(len(vocab), args.embed_dim,
                                     _weight=torch.randn(len(vocab),
                                                         args.embed_dim, dtype=torch.float,
                                                         requires_grad=True) / 100)

        self.s_biase = nn.Embedding(len(vocab), 1,
                                    _weight=torch.randn(len(vocab),
                                                        1, dtype=torch.float,
                                                        requires_grad=True) / 100)

    def forward(self, c, s):
        c_w = self.c_weight(c)
        c_b = self.c_biase(c)
        s_w = self.s_weight(s)
        s_b = self.s_biase(s)
        return torch.sum(c_w.mul(s_w), 1, keepdim=True) + c_b + s_b


# read data
class TrainData(Dataset):
    def __init__(self, coo_matrix,args):
        self.coo_matrix = [((i, j), coo_matrix.data[i][pos]) for i, row in enumerate(coo_matrix.rows) for pos, j in
                           enumerate(row)]
        self.x_max=args.x_max
        self.alpha=args.alpha

    def __len__(self):
        return len(self.coo_matrix)

    def __getitem__(self, idex):
        sample_data = self.coo_matrix[idex]
        sample = {"c": sample_data[0][0],
                  "s": sample_data[0][1],
                  "X_c_s": sample_data[1],
                  "W_c_s": fw(sample_data[1],self.x_max,self.alpha)}
        return sample


def loss_func(X_c_s_hat, X_c_s, W_c_s):
    X_c_s = X_c_s.view(-1, 1)
    W_c_s = X_c_s.view(-1, 1)
    loss = torch.sum(W_c_s.mul((X_c_s_hat - torch.log(X_c_s)) ** 2))
    return loss


# save vector
def save_word_vector(file_name, corpus_preprocessor, glove):
    with open(file_name, "w", encoding="utf-8") as f:
        if use_gpu:
            c_vector = glove.c_weight.weight.data.cpu().numpy()
            s_vector = glove.s_weight.weight.data.cpu().numpy()
            vector = c_vector + s_vector
        else:
            c_vector = glove.c_weight.weight.data.numpy()
            s_vector = glove.s_weight.weight.data.numpy()
            vector = c_vector + s_vector
        # try:
        #     with open('output/vector.pkl', 'wb') as p:
        #         pickle.dump(vector, p)
        #     print('vector的shape', vector.shape)
        # except:
        #     print('打印vector的shape有误')
        for i in tqdm(range(len(vector))):
            word = corpus_preprocessor.idex2word[i]
            s_vec = vector[i]
            s_vec = [str(s) for s in s_vec.tolist()]
            write_line = word + " " + " ".join(s_vec) + "\n"
            f.write(write_line)
        print("Glove vector save complete!")


def train(args):
    corpus_preprocessor = CorpusPreprocess(args.train_data_path, args.min_count)
    coo_matrix = corpus_preprocessor.get_cooccurrence_matrix(args.windows_size)
    vocab = corpus_preprocessor.get_vocab()
    glove = Glove(vocab, args)

    print(glove)
    if os.path.isfile(args.embed_path_pkl):
        glove.load_state_dict(torch.load(args.embed_path_pkl))
        print('载入模型{}'.format(args.embed_path_pkl))
    if use_gpu:
        glove.cuda()
    optimizer = torch.optim.Adam(glove.parameters(), lr=args.learning_rate)

    train_data = TrainData(coo_matrix,args)
    data_loader = DataLoader(train_data,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=2,
                             pin_memory=True)

    steps = 0
    for epoch in range(args.epoches):
        print(f"currently epoch is {epoch + 1}, all epoch is {args.epoches}")
        avg_epoch_loss = 0
        for i, batch_data in enumerate(data_loader):
            c = batch_data['c']
            s = batch_data['s']
            X_c_s = batch_data['X_c_s']
            W_c_s = batch_data["W_c_s"]

            if use_gpu:
                c = c.cuda()
                s = s.cuda()
                X_c_s = X_c_s.cuda()
                W_c_s = W_c_s.cuda()

            W_c_s_hat = glove(c, s)
            loss = loss_func(W_c_s_hat, X_c_s, W_c_s)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_epoch_loss += loss / len(train_data)
            if steps % 1000 == 0:
                print(f"Steps {steps}, loss is {loss.item()}")
            steps += 1
        print(f"Epoches {epoch + 1}, complete!, avg loss {avg_epoch_loss}.\n")
    save_word_vector(args.embed_path_txt, corpus_preprocessor, glove)
    torch.save(glove.state_dict(), args.embed_path_pkl)

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
            assert len(vector)==args.embed_dim,'输出的vector的维度不是设置的%s维'%str(args.embed_dim)
            embeddings.append(vector)
    embeddings=np.array(embeddings)
    print('embed_txt的shape为：({}*{})'.format(index,args.embed_dim))
    print('embed_pkl的shape为：{}'.format(embeddings.shape))
    print('embed_pkl的index为0的位置加了<pad>的向量，所以比embed_txt多1')
    print('vocab的长度: %d' % len(word2id))
    with open(args.embed_path_pkl,'wb') as p:
        pickle.dump(embeddings,p)
    with open(args.vocab_path,'w',encoding='utf-8') as p1:
        json.dump(word2id,p1,ensure_ascii=False)
    print('完成！')

if __name__ == "__main__":
    # file_path
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
    parser.add_argument('--x_max', type=int, default=100, help='两个词共现出现的次数大于x_max后，衡量两词相似性的权重不再增加，论文推荐100')
    parser.add_argument('--alpha', type=float, default=0.75,
                        help='两个词共现出现的次数x小于x_max时，衡量两词相似性的权重为(x/x_max)^alpha 论文推荐0.75')
    parser.add_argument('--epoches', type=int, default=3, help='训练回合')
    parser.add_argument('--min_count', type=int, default=0, help='过滤掉出现小于min_count的词')
    parser.add_argument('--batch_size', type=int, default=64, help='训练批次')
    parser.add_argument('--windows_size', type=int, default=5, help='窗口大小')
    parser.add_argument('--learning_rate', type=int, default=0.001, help='学习率')
    args = parser.parse_args()
    train(args)
    # vec_eval.drawing_and_save_picture(save_picture_file_name)
    get_vocab_and_embed(args)