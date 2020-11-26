# 前言
    该版本是利用pytorch 训练glove词向量，模型的输入是文本，输出是词向量和对应的字典，其中pkl类型的词向量，index=0的位置是'<pad>'的词向量,其值全为0。
# 运行环境    
    详见requirements.txt
# 代码执行
    注：可单独只下载pytorch目录下的文件，就可以执行
    1、打开main.py，所有参数在该文件设置
    2、运行 main.py 即可
# 参考
* [Glove](https://nlp.stanford.edu/projects/glove/)
* [GloVe详解](http://www.fanyeong.com/2018/02/19/glove-in-detail/)
* [pytorch_word2vec](https://github.com/bamtercelboo/pytorch_word2vec)
* [哈夫曼树的实现](https://blog.csdn.net/IT_iverson/article/details/79018505)
