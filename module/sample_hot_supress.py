#!/usr/bin/env python
#  ------------------------------------------------------------------------------------------------
#  Descripe: 
#  Auther: jialiangtu 
#  CopyRight: Tencent Company
# ------------------------------------------------------------------------------------------------
from pyspark import SparkConf
from pyspark.sql import SparkSession
import random,math
conf = SparkConf().setAppName('get_xw_pw_train_data'). \
    set("spark.hadoop.validateOutputSpecs", "false")
spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()

def hot_suppress(train_rdd):
    """
    :param train_rdd: rdd 训练样本
    :return: 按vv热度打压后的训练样本
    """
    # total_users_num=train_rdd.count(1)
    # group by item_id count(user_id)
    train_df = train_rdd.toDF(["user_id", "item_id", "sample"])
    print('未打压前样本数：', train_rdd.count(1))
    print("train_df.show(1):")
    train_df.show(1)
    train_df.registerTempTable('train_df_tmp')
    df1 = spark.sql("""select item_id,count(1) vv from train_df_tmp group by item_id order by vv desc """)
    df1.show(100)
    mean_vv = df1.select('vv').rdd.mean()
    print('mean_vv:', mean_vv)

    def func_hot_supress(x):
        a = 10 ** 3 #建议a与mean_vv接近或50分位的vv接近，这样mean_vv以下的样本不会打压
        y = (math.sqrt(float(x) / a) + 5) * (a / float(x)) # 5 可以适度修改，越大就越往右偏移
        return y

    df1 = df1.withColumn('keep_prob', F.udf(func_hot_supress)(df1.vv))
    df1.registerTempTable('prob_tmp')
    df_final = spark.sql("""
    select keep_prob,rands,sample from 
        (select a.user_id,
                a.item_id,
                a.sample,
                b.keep_prob,
                rand() rands
            from train_df_tmp a left join prob_tmp b
            on a.item_id=b.item_id) 
    where keep_prob>rands""")
    print("df_final.show():")
    df_final.show()
    print("打压之后样本数：", df_final.count(1))
    return df_final.rdd