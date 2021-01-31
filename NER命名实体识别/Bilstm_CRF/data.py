import os
import pickle
import random
import codecs
import numpy as np
from utils import iob_iobes
es2label = {}

#如果iob2iobes设置为True,需要在tag2label中添加E-xx和S-xx，如tag2label_dg
#而对于非iob格式的tags如tag2label_resume_ner，iob2iobes只能设置为Fasle
#DaGuang数据集
tag2label_dg={"O":0,
              "B-a":1,"I-a":2,
              "B-b":3,"I-b":4,
              "B-c":5,"I-c":6,
              "E-a":7,'E-b':8,"E-c":9,
              "S-a":10,"S-b":11,"S-c":12
              }

# 默认数据集 MSRA tags, BIO
tag2label_msra = {"O": 0,
                  "B-PER": 1, "I-PER": 2,
                  "B-LOC": 3, "I-LOC": 4,
                  "B-ORG": 5, "I-ORG": 6,
                  "E-PER":7,"E-LOC":8,"E-ORG":9,
                  "S-PER":10,"S-LOC":11,"S-ORG":12
                  }

# 人民日报数据集
tag2label_chinadaily = {"O": 0,
                        "B-PERSON": 1, "I-PERSON": 2,
                        "B-LOC": 3, "I-LOC": 4,
                        "B-ORG": 5, "I-ORG": 6,
                        "B-GPE": 7, "I-GPE": 8,
                        "B-MISC": 9, "I-MISC": 10
                        }
# WeiboNER
tag2label_weibo_ner = {"O": 0,
                       "B-PER.NAM": 1, "I-PER.NAM": 2,
                       "B-LOC.NAM": 3, "I-LOC.NAM": 4,
                       "B-ORG.NAM": 5, "I-ORG.NAM": 6,
                       "B-GPE.NAM": 7, "I-GPE.NAM": 8,
                       "B-PER.NOM": 9, "I-PER.NOM": 10,
                       "B-LOC.NOM": 11, "I-LOC.NOM": 12,
                       "B-ORG.NOM": 13, "I-ORG.NOM": 14
                       }

# Resume_NER
tag2label_resume_ner = {"O": 0,
                        "B-NAME": 1, "M-NAME": 2, "E-NAME": 3, "S-NAME": 4,
                        "B-RACE": 5, "M-RACE": 6, "E-RACE": 7, "S-RACE": 8,
                        "B-CONT": 9, "M-CONT": 10, "E-CONT": 11, "S-CONT": 12,
                        "B-LOC": 13, "M-LOC": 14, "E-LOC": 15, "S-LOC": 16,
                        "B-PRO": 17, "M-PRO": 18, "E-PRO": 19, "S-PRO": 20,
                        "B-EDU": 21, "M-EDU": 22, "E-EDU": 23, "S-EDU": 24,
                        "B-TITLE": 25, "M-TITLE": 26, "E-TITLE": 27, "S-TITLE": 28,
                        "B-ORG": 29, "M-ORG": 30, "E-ORG": 31, "S-ORG": 32,
                        }

tag2label_mapping = {
    'DaGuang':tag2label_dg,
    'MSRA': tag2label_msra,
    '人民日报': tag2label_chinadaily,
    'WeiboNER': tag2label_weibo_ner,
    'ResumeNER': tag2label_resume_ner

}


def build_character_embeddings(pretrained_emb_path, embeddings_path, word2id=None, embedding_dim=None):
    print('loading pretrained embeddings from {}'.format(pretrained_emb_path))
    if pretrained_emb_path[-3:]=='pkl':
        with codecs.open(pretrained_emb_path,'rb') as f:
            embeddings=pickle.load(f)
        np.save(embeddings_path, embeddings)
    else:
        pre_emb = {}
        for line in codecs.open(pretrained_emb_path, 'r', 'utf-8'):
            line = line.strip().split()
            if len(line) == embedding_dim + 1:
                pre_emb[line[0]] = [float(x) for x in line[1:]]
        word_ids = sorted(word2id.items(), key=lambda x: x[1])
        characters = [c[0] for c in word_ids]
        embeddings = list()
        for i, ch in enumerate(characters):
            if ch in pre_emb:
                embeddings.append(pre_emb[ch])
            else:
                embeddings.append(np.random.uniform(-0.25, 0.25, embedding_dim).tolist())
        embeddings = np.asarray(embeddings, dtype=np.float32)
        np.save(embeddings_path, embeddings)

def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path: 字\ttag \n 字\ttag 最后一行需保留空行，否则少一行数据
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            line=line.strip().split()
            if len(line)<2:
                sent_.append(line[0])
                tag_.append('O')
            else:
                sent_.append(line[0])
                tag_.append(line[1])
        else:
            if sent_ and tag_:   #为空不添加
                data.append((sent_, tag_))
            sent_, tag_ = [], []

    return data


def vocab_build(args,vocab_path, corpus_path, min_count=1):
    """

    :param vocab_path:  字典保存位置
    :param corpus_path:原始输入数据，格式见main函数中
    :param min_count:过滤出现少于min_count次数的字
    :return:pkl格式的word2id字典
    """
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_, tag_ in data:
        for tag in tag_:
            if args.iob2iobes:assert tag.split('-')[0] not in ['B','I','O'],'tag is not iob，iob2iobes must be set False'
        for word in sent_:
            # if word.isdigit():
            #     word = '<NUM>'
            # elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            #     word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id) + 1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['[<UNK>]'] = new_id
    word2id['[<PAD>]'] = 0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)


def sentence2id(sent, word2id):
    """

    :param sent: <class 'list'>: ['14123', '20451', '18288', '23', '2128', '17808', '5163', '1027', '13964', '11541', '20733', '14478', '12474', '12617', '21224', '4216', '19907', '9779', '14796', '11255', '3277', '14123', '3452', '12851', '5163', '18826', '4246', '17166', '14126', '17359', '1866', '6006', '18826', '16201', '14796', '7747', '4808', '6523', '21224', '6196', '13046', '11255', '3277', '5163', '18826', '3999', '861', '159', '3452', '9954', '18736', '4921', '15034', '19365', '21224', '3445', '20027', '19215', '4846', '10399', '567', '10841', '11255', '1146', '3647', '18736', '4921', '13670', '2250', '6991', '18538', '10925', '2764', '12721', '8197', '17808', '4859', '7384', '12062', '15274']
    :param word2id:字典
    :return:
    """
    sentence_id = []
    for word in sent:
        # if word.isdigit():
        #     word = '<NUM>'
        # elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
        #     word = '<ENG>'
        if word not in word2id:
            word = '[<UNK>]'
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    """

    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x: len(x), sequences))#max_len是每个batch中句子的最大长度。
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False,iob2iobes=True):
    """

    :param data:list [<class 'tuple'>: (['19421', '21215', '14459', '12052', '7731', '3028', '17622', '11664', '13751', '10841', '11255', '159', '8467', '15671', '2699', '13751', '11806', '14459', '15274'], ['B-b', 'I-b', 'I-b', 'O', 'O', 'B-b', 'I-b', 'O', 'O', 'O', 'O', 'O', 'B-b', 'B-b', 'O', 'O', 'O', 'O', 'O']),...]
    :param batch_size:
    :param vocab: word2id 字典
    :param tag2label: 见data处
    :param shuffle:
    :return: train_data <class 'list'>: [[4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437, 4437],...]
            label <class 'list'>: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],...]
    """
    if shuffle:
        random.shuffle(data)
    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        if 'E-PER.NOM'in tag_:
            print(tag_)
        if iob2iobes:
            tag_=iob_iobes(tag_)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels

# if __name__ == '__main__':
#     word2id = read_dictionary(os.path.join('data_path', 'MSRA', 'word2id.pkl'))
#     build_character_embeddings('./sgns.wiki.char', './vectors.npy', word2id, 300)
