import os
import logging
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random
from scipy.sparse import lil_matrix
from huffman import HuffmanTree
from collections import Counter

encoding_type = "utf-8"
split_sep = " "


class CorpusPreprocess(object):
    logger = logging.getLogger("CorpusPreprocess")

    def __init__(self, file_path, min_freq):
        self.file_path = file_path
        self.min_freq = min_freq
        self.huffman = None
        self.huffman_left = None
        self.huffman_right = None
        self.vocab = Counter()
        self.cooccurrence_matrix = None
        self.idex2word = None
        self.word2idex = None
        self.nag_sampling_vocab = None
        self._build_vocab()
    
    def _read_data(self):
        if not os.path.exists(self.file_path):
            raise FileExistsError(f"file path {self.file_path} is not exist !")
        with open(self.file_path, "r", encoding = encoding_type) as f:
            for line in f:
                if line.strip():
                    yield line.strip().split(split_sep)
    
    def _build_vocab(self):
        for line in self._read_data():
            self.vocab.update(line)
        self.vocab = dict((w.strip(), f) for w,f in self.vocab.items() if (f >= self.min_freq and w.strip()))
        self.vocab = dict(sorted(self.vocab.items(), key=lambda x: x[1], reverse=True))
        self.vocab = {w:(i, f) for i, (w, f) in enumerate(self.vocab.items())}
        self.idex2word = {i:w for w, (i,f) in self.vocab.items()}
        self.logger.info("build vocab complete!")

    def _build_cooccurrence_matrix(self, windows_size=5):
        if not self.vocab:
            self._build_vocab()
        self.cooccurrence_matrix = lil_matrix((len(self.vocab), len(self.vocab)),dtype=np.float32)
        for line in self._read_data():
            sentence_length = len(line)
            for i in range(sentence_length):
                center_w = line[i]
                if center_w not in self.vocab:
                    continue
                left_ws = line[max(i-windows_size,0):i]
                # right_ws = line[i+1:min(len(line),i+1+windows_size)]
                
                # left cooccur
                for i, w in enumerate(left_ws[::-1]):
                    if w not in self.vocab:
                        continue
                    self.cooccurrence_matrix[self.vocab[center_w][0],
                                             self.vocab[w][0]] += 1.0 / (i+1.0)
                    # cooccurrence_matrix is Symmetric Matrices
                    self.cooccurrence_matrix[self.vocab[w][0],
                                             self.vocab[center_w][0]] += 1.0 / (i+1.0)
                # for i, w in enumerate(right_ws):
                    
                #     self.cooccurrence_matrix[self.vocab[center_w][0],
                #                              self.vocab[w][0]] += 1.0 /(i+1.0)

                        
        self.logger.info("build cooccurrece matrix complete!")
    def build_vocab_for_nag_sampling(self):
        # Build nag_sampling_vocab
        vocab = self.get_vocab()
        sampling_vocab = {info[0]:info[1] for w,info in vocab.items()}
        all_count = sum([f ** (3/4) for i,f in sampling_vocab.items()])
        neg_sampling_vocab = []
        neg_sampling_prob = []
        for i,f in sampling_vocab.items():
            neg_sampling_vocab.append(i)
            neg_sampling_prob.append(f**(3/4)/all_count)
        self.nag_sampling_vocab = (neg_sampling_vocab,neg_sampling_prob)

    def get_nag_sampling_vocab(self):
        if not self.nag_sampling_vocab:
            self.build_vocab_for_nag_sampling()
        return self.nag_sampling_vocab


    def get_cooccurrence_matrix(self, windows_size):
        if self.cooccurrence_matrix == None:
            self._build_cooccurrence_matrix(windows_size)
        return self.cooccurrence_matrix
    
    def get_vocab(self):
        if not isinstance(self.vocab, dict):
            self._build_vocab()
        return self.vocab
    
    def build_huffman_tree(self):
        vocab = self.get_vocab()
        vocab = {info[0]: info[1] for w,info in vocab.items()}
        self.huffman = HuffmanTree(vocab)
        self.huffman_left, self.huffman_right = self.huffman.generate_node_left_and_right_path()
        print("build_tree_complete")
    def get_bath_huffman_tree_sample(self, batch_data):
        batch_data_from_huffman = []
        if not self.huffman_left:
            self.build_huffman_tree()
        for example in batch_data:
            pos = self.huffman_left[example[1]]
            neg = self.huffman_right[example[1]]
            batch_data_from_huffman.append(([example[0]],
            pos, neg))
        return batch_data_from_huffman
    
    def get_bath_nagative_train_data(self, batch_data, count):
        if not self.nag_sampling_vocab:
            self.build_vocab_for_nag_sampling()
        batch_nagtive_simples = []
        for example in batch_data:
            neg = np.random.choice(self.nag_sampling_vocab[0], size=count, p=self.nag_sampling_vocab[1]).tolist()
            batch_nagtive_simples.append(([example[0]],
            [example[1]], neg))
        return batch_nagtive_simples
        

    def build_cbow_tain_data(self, windows_size):
        if not self.vocab:
            self._build_vocab()
        for line in self._read_data():
            if isinstance(line, list):
                sentence_len = len(line)
                for idx, w in enumerate(line):
                    if w not in self.vocab:
                        continue
                    left = [self.vocab[w][0] for w in line[max(0, idx-windows_size) : idx] if w in self.vocab]
                    right = [self.vocab[w][0] for w in line[idx+1: min(idx + windows_size + 1, sentence_len)] if w in self.vocab]
                    yield (left+right, self.vocab[w][0])

    def get_bach_data(self, data, bach_size):
        bach_data = []
        for i in data:
            bach_data.append(i)
            if len(bach_data) == bach_size:
                yield bach_data
                bach_data = []
        yield bach_data

    def build_skip_gram_tain_data(self, windows_size):
        if not self.vocab:
            self._build_vocab()
        for line in self._read_data():
            if isinstance(line, list):
                sentence_len = len(line)
                for idx, w in enumerate(line):
                    if w not in self.vocab:
                        continue
                    left = [self.vocab[w][0] for w in line[max(0, idx-windows_size) : idx] if w in self.vocab]
                    right = [self.vocab[w][0] for w in line[idx+1: min(idx + windows_size + 1, sentence_len)] if w in self.vocab]
                    for c_w in left + right:
                        yield (self.vocab[w][0], c_w)

class VectorEvaluation(object):
    def __init__(self, vector_file_path):
        if os.path.exists(vector_file_path):
            self.vector_file_path = vector_file_path
        else:
            raise FileExistsError("file is not exists!")
        self.read_data()
    
    def _read_line(self, word, *vector):
        return word, np.asarray(vector, dtype=np.float32)

    def read_data(self):
        words = []
        vector = []
        with open(self.vector_file_path, "r", encoding=encoding_type) as f:
            for line in f:
                word, vec = self._read_line(*line.split(split_sep))
                words.append(word)
                vector.append(vec)
        assert len(vector) == len(words)
        self.vector = np.vstack(tuple(vector))
        self.vocab = {w:i for i,w in enumerate(words)}
        self.idex2word = {i:w for w,i in self.vocab.items()} 

    def drawing_and_save_picture(self, picture_path, w_num=10, mode="tsne"):
        w_num = min(len(self.vocab), w_num)
        reduce_dim = PCA(n_components=2)
        if mode == "tsne":
            reduce_dim = TSNE(n_components=2)
        vector = reduce_dim.fit_transform(self.vector)
        idex = np.random.choice(np.arange(len(self.vocab)), size=w_num, replace=False)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        for i in idex:
            plt.scatter(vector[i][0], vector[i][1])
            plt.annotate(self.idex2word[i], xy=(vector[i][0], vector[i][1]),xytext=(5, 2),textcoords='offset points')
        plt.title(f"{mode} - " + picture_path.split("/")[-1].split(".")[0])
        plt.savefig(picture_path)
        print(f"save picture to {picture_path}")

    def get_similar_words(self, word, w_num=10):
        w_num = min(len(self.vocab), w_num)
        idx = self.vocab.get(word,None)
        if not idx:
            idx = random.choice(range(self.vector.shape[0]))
        result = cosine_similarity(self.vector[idx].reshape(1,-1), self.vector)
        result = np.array(result).reshape(len(self.vocab),)
        idxs = np.argsort(result)[::-1][:w_num]
        print("<<<"*7)
        print(self.idex2word[idx])
        for i in idxs:
            print("%s : %.3f\n" % (self.idex2word[i], result[i]))
            
        print(">>>" * 7)


if __name__ == "__main__":
    dataset = CorpusPreprocess("./data/text.txt", min_freq=0)
    data = dataset.build_skip_gram_tain_data(3)
    dataset.build_huffman_tree()
    print(dataset.huffman_left)
    print(dataset.huffman_right)
    for i in dataset.get_bach_data(data, 3):
        print(i)
        print(dataset.get_bath_huffman_tree_sample(i))
