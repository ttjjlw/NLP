# 前言
该版本是利用gensim训练词向量，模型的输入是文本，输出是词向量和对应的字典，其中pkl类型的词向量，index=0的位置是'<pad>'的词向量,其值全为0。利用gensim训练词向量比自己实现的要快很多，所以使用的时候，推荐利用gensim训练word2vec
# 运行环境
    详见requirements.txt
# 代码执行
    注：可单独只复制gensim该目录下的文件，就可以执行
    1、打开main.py，所有参数在该文件设置(一些不常用的参数，可进入Word2vec.py文件中设置)
    2、运行 main.py 即可


