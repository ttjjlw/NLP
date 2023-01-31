#! /usr/bin/env python
# -*- coding=utf8 -*-

import argparse, subprocess
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pytoolkit import TDWSQLProvider
from pytoolkit import TableDesc
from pytoolkit import TDWUtil
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--date', type=str, help='date of train data.')
parser.add_argument('--date1', type=str, help='date of item info data.')
parser.add_argument('--version', type=str, default='', help='date of item info data.')
parser.add_argument('--model_version', type=str, help='model name and version.')
args, _ = parser.parse_known_args()
conf = SparkConf().setAppName('miniv_pair_wise'). \
    set("spark.hadoop.validateOutputSpecs", "false")
spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()

sc = spark.sparkContext
tdwu = TDWUtil(user="tdw_xxx", passwd="tdw_xxx", dbName="sng_kd_video_data_frame")
tdw = TDWSQLProvider(spark, user="tdw_xxx", passwd="tdw_xxx", db="sng_kd_video_data_frame")

model_version = args.model_version

cmd = """
hdfs='hadoop fs -Dhadoop.job.ugi=tdw_xxx:tdw_xxx';
date={date}
if [  -e ./user_vector.txt ];then
    rm ./user_vector.txt
else
    echo '不存在'
fi
if [  -e ./item_vector.txt ];then
    rm ./item_vector.txt
else
    echo '不存在'
fi

if [  -e ./log.txt ];then
    rm ./log.txt
else
    echo '不存在'
fi

$hdfs -get hdfs://xxx/data/SPARK/SNG/xxx/xxx/export/{model_version}/item_vector/$date/part-00000 ./item_vector.txt
$hdfs -get hdfs://xxx/data/SPARK/SNG/xxx/xxx/export/{model_version}/user_vector{version}/$date/part-00000 ./user_vector.txt
$hdfs -get hdfs://xxx/data/SPARK/SNG/xxx/xxx/log/{model_version}/log.txt ./log.txt
""".format(date=args.date,model_version=model_version,version=args.version)
ret = subprocess.call(cmd, shell=True)
print('ret:', ret)

import faiss
import numpy as np
import random
import sys
import mkl
from sklearn import preprocessing
from sklearn.metrics import top_k_accuracy_score

mkl.get_max_threads()

item2id = {}
item_vector_lis = []
for idx, line in enumerate(open("item_vector.txt")):
    item_nm, item_vector = line.strip().split("||")
    if len(item_nm) < 2:
        print('item_nm:', item_nm, '\n', 'idx:', idx)
        exit()
    item2id[item_nm] = idx

    item_vector = item_vector.split(",")
    item_vector_lis.append(item_vector)

xb = np.array(item_vector_lis).astype('float32')
# xb = preprocessing.normalize(xb, norm='l2',axis=1)
print('xb.shape:', xb.shape)

nlist = 1  ##聚类中心的个数
k = 100  # 召回向量个数
d = len(item_vector_lis[0])  # 向量维度

quantizer = faiss.IndexFlatL2(d)  # the other index
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
# index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
print('is_trained:', index.is_trained)

index.train(xb)
index.add(xb)  # add may be a bit slower as well

print('is_trained:', index.is_trained)
print("Index done")

labels = []
user_vector_lis = []
user_id_lis = []
for line in open("user_vector.txt"):
    item_nm, user_id, user_vector = line.strip().split("||")
    if item_nm not in item2id: continue
    labels.append([item2id[item_nm]])
    user_id_lis.append(user_id)

    user_vector = user_vector.split(",")
    user_vector_lis.append(user_vector)
assert (len(labels) == len(user_vector_lis))
xq = np.array(user_vector_lis).astype('float32')
# xq = preprocessing.normalize(xq, norm='l2',axis=1)

print('xq.shape:', xq.shape)

D, I = index.search(xq, k)  # actual search
print(type(I))


# print(I)                  # neighbors of the 5 last queri


def topk_accuracy(label, pred):
    match_array = np.logical_or.reduce(pred == label, axis=1)  # 得到匹配结果
    topk_acc_score = match_array.sum() / match_array.shape[0]
    return topk_acc_score


print(labels[:5])
print('I:', I[:5])
print('D:', D[:5])
res = topk_accuracy(labels, I)
print(res)


def topk_accuracy1(label, pred):
    pred = list(pred)
    assert len(label) == len(pred)
    total = len(label)
    acc = 0
    for l, line in zip(label, pred):
        if l[0] in line:
            acc += 1
    return acc / total

def create_partition_tb(df, tdwu, tdw, table_nm, field, value, col_nm):
    setCols = []
    for col in col_nm:
        setCols.append([col, 'string', 'col'])

    table_desc = TableDesc().setTblName(table_nm). \
        setCols(setCols). \
        setComment("this is tdw-ddl-test"). \
        setCompress(True). \
        setFileFormat("rcfile"). \
        setPartType("list"). \
        setPartField(field)

    if not tdwu.tableExist(table_nm): tdwu.createTable(table_desc)
    if tdwu.partitionExist(tblName=table_nm, partName='p_%s' % value):
        tdwu.dropPartition(table_nm, "p_%s" % value)
    tdwu.createListPartition(table_nm, "p_%s" % value, value, level=0)
    tdw.saveToTable(df, table_nm, priPart="p_%s" % value)
# print(topk_accuracy1(labels,I))


# get g
id2item = dict(zip(item2id.values(), item2id.keys()))
user_ct = len(I)
assert (user_ct == len(user_id_lis)), ValueError("user_id条数不一致")

item_nm_lis = []
for line in I:
    item_nm = [id2item[id] for id in line]
    item_nm_lis.append(item_nm)
print('user_id_lis:', user_id_lis[0])
print(item_nm_lis[0])
rdd = sc.parallelize(list(zip(user_id_lis, item_nm_lis)))
df_recall = rdd.map(lambda line:[line[0],','.join(line[1]),model_version+args.version]).toDF(['user_id1', 'item_id', 'model_version'])
create_partition_tb(df_recall,tdwu,tdw,'user_recall_tb','model_version',model_version+args.version,['user_id1', 'item_id', 'model_version'])

df_true_recall = tdw.table('user_true_recall_d', priParts=["p_%s" % args.date])
cond = [df_recall.user_id1 == df_true_recall.user_id]
df = df_recall.join(df_true_recall, on=cond, how='inner').select('user_id', 'item_id', 'item_score')


def getDCG(scores):
    return np.sum(
        np.divide(np.power(2, scores) - 1, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)),
        dtype=np.float32)

def getNDCG(rank_scores, true_relevance):
    idcg = getDCG(true_relevance)

    dcg = getDCG(rank_scores[:len(true_relevance)])

    if dcg == 0.0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg

def to_recall_score(x, y):
    item2score = {i.split('|')[0]: i.split('|')[1] for i in y.split(',')}
    true_score = np.array([float(i.split('|')[1]) for i in y.split(',')])
    recall_score = np.array([float(item2score[i]) if i in item2score else -1.0 for i in x])
    ndcg = getNDCG(recall_score, true_score)
    return ndcg


rdd = df.rdd.map(lambda line: to_recall_score(line[1], line[2]))
print(rdd.collect()[:3])

ndcg = rdd.mean()
print('ndcg:', ndcg)

with open('log.txt', 'a') as f:
    f.write(args.date+' '+str(res)+'\t'+str(ndcg)+'\t'+args.version+'\t'+str(xq.shape[0])+'\t'+str(xb.shape[0])+'\n')

cmd = """
hdfs='hadoop fs -Dhadoop.job.ugi=tdw_xxx:tdw_xxx';
if $hdfs -test -e hdfs://xxx/data/SPARK/SNG/xxx/xxx/log/{model_version}/log.txt ; then
    echo '存在'
    $hdfs -rmr hdfs://xxx/data/SPARK/SNG/xxx/xxx/log/{model_version}/log.txt
else
    echo 'bucunzai'
fi

$hdfs -cat log.txt | $hdfs -copyFromLocal - hdfs://xxx/data/SPARK/SNG/xxx/xxx/log/{model_version}/log.txt;
""".format(model_version=model_version)
subprocess.call(cmd, shell=True)