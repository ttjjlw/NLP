#! /usr/bin/env python
# -*- coding=utf8 -*-

import argparse, subprocess
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
import numpy as np
import re

# parser = argparse.ArgumentParser()
# parser.add_argument('--hdfs_path', type=str, help='path of data.')
# parser.add_argument('--model_version', type=str,help='model name and version.')
# args, _ = parser.parse_known_args()
conf = SparkConf().setAppName('check_data'). \
    set("spark.hadoop.validateOutputSpecs", "false")
spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()

sc = spark.sparkContext


#包含的功能：取目录最后一个文件
def check_data(hdfs_path):
    cmd = """
    set -e
    hdfs='hadoop fs -Dhadoop.job.ugi=tdw_xxx:tdw_xxx'
    # hdfs='sh /data/home/xxx/scripts/hadoop.sh dfs'
    # hdfs_path='hdfs://xxx/data/SPARK/SNG/xxx/xxx/data/ptrain/'
    hdfs_path={hdfs_path}
    latest_file=`$hdfs -ls $hdfs_path | tail -n 1`
    latest_file=${{latest_file:0-10:10}}
    if $hdfs -test -e ${{hdfs_path}}${{latest_file}}/_SUCCESS;then
        echo "最近的数据目录存在_SUCCESS"
    else
        echo "最近的数据目录不存在_SUCCESS"
        latest_file=`$hdfs -ls ${{hdfs_path}} | tail -n 2 | head -n 1`
        latest_file=${{latest_file:0-10:10}}
    fi
    echo "latest_file: $latest_file"
    create_time=`$hdfs -stat %y ${{hdfs_path}}${{latest_file}}`
    echo "create_time: $create_time"

    #create_time='2022-10-24 00:00:00'

    ctimetap=`date -d "$create_time" +%s`
    echo $ctimetap
    ((ctimetap=$ctimetap + 3600*8))
    echo $ctimetap
    now_time=`date +%s`
    echo $now_time
    ((diff=$now_time-$ctimetap))
    echo $diff
    #detail=$($hdfs -ls ${{hdfs_path}}${{latest_file}}/_SUCCESS)
    #echo $detail

    format_latest_file_h=${{latest_file:0-2:2}}
    format_latest_file_d=${{latest_file:0:8}}
    echo $format_latest_file_d
    echo $format_latest_file_h
    dir_timestap=`date -d "$format_latest_file_d $format_latest_file_h" +%s`
    echo "dir_timestap: $dir_timestap"
    ((new_dir_timestap=$dir_timestap + 3600))
    new_dir=`date -d @$new_dir_timestap "+%Y%m%d%H"`
    echo "将要生成的数据目录：$new_dir"

    ((copy_dir_timestap=$dir_timestap + 3600 - 24*3600))
    copy_dir=`date -d @$copy_dir_timestap "+%Y%m%d%H"`
    echo "copy dir: $copy_dir"
    ((threshold_time=120*60))

    if [ $diff -gt $threshold_time ];
    then
            echo "大于"
            $hdfs -mkdir ${{hdfs_path}}$new_dir
            if $hdfs -test -e ${{hdfs_path}}$copy_dir/_SUCCESS;
            then
                    echo "存在可移动的目录"
            else
                    echo "不存在可移动的目录"
                    copy_dir=$latest_file
            fi
            echo "开始移动 $copy_dir数据到$new_dir"
            $hdfs -mv ${{hdfs_path}}$copy_dir/part-* ${{hdfs_path}}$new_dir
            $hdfs -mv ${{hdfs_path}}$copy_dir/_SUCCESS ${{hdfs_path}}$new_dir
            echo "数据移动成功"
            echo $hdfs_path >> log.txt
            echo $copy_dir $new_dir >> log.txt

    else
            echo "小于等于\n"
            # echo "小于等于" >> log.txt
            # echo $new_dir >> log.txt
    fi
    """.format(hdfs_path=hdfs_path)
    ret = subprocess.call(cmd, shell=True)
    print('ret:', ret)


cmd1 = """set -e
    hdfs='hadoop fs -Dhadoop.job.ugi=tdw_xxx:tdw_xxx';
    $hdfs -get hdfs://xxx/data/SPARK/SNG/xxx/xxx/log/check_data/log.txt ./log.txt"""
ret = subprocess.call(cmd1, shell=True)

for path in [
    "hdfs://xxx/data/SPARK/SNG/xxx/xxx/data/ptrain/",
    "hdfs://xxx/data/SPARK/SNG/xxx/xxx/data/rtrain/",
    "hdfs://xxx/data/SPARK/SNG/xxx/xxx/mxw_gzh_cid/train/",
    "hdfs://xxx/data/SPARK/SNG/xxx/xxx/mxw_gzh_cid/train_neg/",
    "hdfs://xxx/data/SPARK/SNG/xxx/xxx/data/hptrain/",
    "hdfs://xxx/data/SPARK/SNG/xxx/xxx/data/xtrain/"]:
    print('path: ',path)
    check_data(path)

cmd2 = """
    set -e
    hdfs='hadoop fs -Dhadoop.job.ugi=tdw_xxx:tdw_xxx'
    if $hdfs -test -e hdfs://xxx/data/SPARK/SNG/xxx/xxx/log/check_data/log.txt ; then
        echo '存在'
        $hdfs -rmr hdfs://xxx/data/SPARK/SNG/xxx/xxx/log/check_data/log.txt
    else
        echo 'bucunzai'
    fi
    $hdfs -cat log.txt | $hdfs -copyFromLocal - hdfs://xxx/data/SPARK/SNG/xxx/xxx/log/check_data/log.txt"""
ret = subprocess.call(cmd2, shell=True)
print("finish!")

