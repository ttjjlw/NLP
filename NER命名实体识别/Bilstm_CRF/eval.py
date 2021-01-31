import os
from utils import iobes_iob
#计算BIO.txt 的F1socre
def BIO_F1score(predict,target=None,negative_label=["O"]):
    import os
    if type(predict)==list and type(target)==list:
        p_tag_num = 0
        t_tag_num = 0
        correct_tag_num = 0
        for p_tag,t_tag in zip(predict,target):
            if p_tag not in negative_label:
                p_tag_num += 1
            if t_tag not in negative_label:
                t_tag_num += 1
            if p_tag == t_tag and p_tag not in negative_label:
                correct_tag_num += 1
        precision=round(correct_tag_num / p_tag_num, 4) if p_tag_num else 0
        print('Precision: {}'.format(precision))
        recall = round(correct_tag_num / t_tag_num, 4) if t_tag_num else 0
        print('recall: {}'.format(recall))
        F1 = round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0
    elif os.path.isfile(predict) and target:
        p_file=open(predict,'r',encoding='utf-8')
        predict=p_file.readlines()
        t_file=open(target,'r',encoding='utf-8')
        target=t_file.readlines()
        p_tag_num=0
        t_tag_num=0
        correct_tag_num=0
        for p_line,t_line in zip(predict,target):
            if p_line == '\n' and t_line=='\n':
                continue
            elif p_line == '\n' and t_line!='\n':
                raise AttributeError('预测换行符和目标换行符不匹配')
            elif p_line != '\n' and t_line=='\n':
                raise AttributeError('预测换行符和目标换行符不匹配')
            p_tag=p_line.strip().split()[-1]
            t_tag=t_line.strip().split()[-1]
            if p_tag not in negative_label:
                p_tag_num+=1
            if t_tag not in negative_label:
                t_tag_num+=1
            if p_tag == t_tag and p_tag not in negative_label:
                correct_tag_num+=1
        precision = round(correct_tag_num / p_tag_num, 4) if p_tag_num else 0
        print('Precision: {}'.format(precision))
        recall = round(correct_tag_num / t_tag_num, 4) if t_tag_num else 0
        print('recall: {}'.format(recall))
        F1 = round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0
        p_file.close()
        t_file.close()
    elif os.path.isfile(predict) and target==None:
        with open(predict,'r',encoding='utf-8') as p:
            predict=p.readlines()
        p_tag_num = 0
        t_tag_num = 0
        correct_tag_num = 0
        if len(predict[0].strip().split())!=3:
            raise AttributeError('输入的predict每行不是三个字符')
        for line in predict:
            if line=='\n':
                continue
            p_tag=line.strip().split()[-1]
            t_tag=line.strip().split()[1]
            if p_tag not in negative_label:
                p_tag_num += 1
            if t_tag not in negative_label:
                t_tag_num += 1
            if p_tag == t_tag and p_tag not in negative_label:
                correct_tag_num += 1
        precision = round(correct_tag_num / p_tag_num, 4) if p_tag_num else 0
        print('Precision: {}'.format(precision))
        recall = round(correct_tag_num / t_tag_num, 4) if t_tag_num else 0
        print('recall: {}'.format(recall))
        F1 = round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0
    else:
        raise AttributeError('输入的predict格式不对')
    return F1,precision,recall
def conlleval(mode,label_predict, result_file, negative_label,iob2iobes):
    """
    :param label_predict:
    :param label_path:
    :param negative_label:不参与计算f1score的tag
    :return:
    """
    
    line = []
    line_pre=[]
    line_real=[]
    for sent_result in label_predict:
        for char, tag, tag_ in sent_result: #字 真实标签 预测标签
            # tag = '0' if tag == 'O' else tag  为什么把tag转换成‘0’？
            # char = char.encode("utf-8")
            if iob2iobes:tag_=iobes_iob([tag_])[0]
            line.append("{} {}\n".format(char, tag_)) #字 预测标签
            line_pre.append(tag_)
            line_real.append(tag)
        line.append("\n")
    while line[-1]=='\n':
        line.pop()
    if mode!='test':
        F1,precision,recall = BIO_F1score(predict=line_pre, target=line_real, negative_label=negative_label)
        return F1,precision,recall
    with open(result_file, "w",encoding='utf-8') as fw:
        print('结果保存在%s'%result_file)
        fw.writelines(line)
        exit()
    
    