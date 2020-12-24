# 前言
>>该版本是利用tf1.x，输入为tf.placeholder,实现的bert,albert以及robert的qa文本匹配。
数据格式为：query reply label 也就是两句话+label,用来预测reply是否对应query（seq_id该列可以为任何值，该特征没有参与训练）
    
# 运行环境    
    详见requirements.txt
# 代码执行
    注：可单独只下载tf-placeholder目录下的文件，就可以执行
    1、进入bert,运行main.py
# 需要注意的：
>> 1、设置version可切换bert的版本，目前支持bert_model目录下几种bert版本，若要添加其他版本，可参考添加。由于github的大小限制，所有版本的bert模型我留有百度云地址，可从百度云下载，并放置对应位置  
>> 2、若mode！='train',则仅进行预测。若fold_num>1则进行fold_num折融合  
>> 3、每100个steps打印一次学习率和训练loss,每个epoch打印一次训练和验证f1score,如果restore_on_train=True
仅保存验证集得分最高的模型，并且每次加载最优模型继续训练，如果restore_on_train=False,则正常训练，每个epoch保存一次模型。
    
